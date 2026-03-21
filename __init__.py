"""
agent — A minimal agentic loop you import and use.

    from agent import Agent

    agent = Agent(api_key="sk-...", model="gpt-4.1-mini")
    print(agent.chat("what files do I have?"))

Pass verbose=True to print tool calls and results to stderr as they happen:

    agent = Agent(api_key="sk-...", verbose=True)
    agent.chat("search for LTX-2.3 model benchmarks")

Or provide your own callbacks for full control:

    agent.chat("...", on_tool_call=lambda cmd: print(f"→ {cmd}"),
                      on_tool_result=lambda out: print(f"  {out[:200]}"))
"""

import sys
import threading
from typing import Callable, Iterator

import openai

import agent.db as db_mod
import agent.memory as mem
from agent.context import build_context
from agent.loop import run_loop
from agent.tools import Registry, register_builtins


class Agent:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        system_prompt: str = "You are a helpful assistant.",
        name: str = "agent",
        data_dir: str = "./data",
        work_dir: str = ".",
        embedding_model: str | None = "text-embedding-3-small",
        base_url: str | None = None,
        verbose: bool = False,
    ):
        """
        Create an Agent.

        Args:
            api_key:         OpenAI API key.
            model:           Model to use for chat completions.
            system_prompt:   Custom instructions for the agent.
            name:            The agent's name (used in the system prompt).
            data_dir:        Where to store the SQLite DB and topic files.
            work_dir:        Working directory for ls/cat/write (default: current dir).
            embedding_model: Embedding model for semantic memory search.
                             Set to None to disable semantic search.
            base_url:        Custom API base URL (for OpenAI-compatible providers).
            verbose:         Print tool calls and results to stderr as they happen.
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**client_kwargs)
        self._config = {
            "model":           model,
            "name":            name,
            "system_prompt":   system_prompt,
            "data_dir":        data_dir,
            "work_dir":        work_dir,
            "embedding_model": embedding_model,
        }
        self._db      = db_mod.open_db(data_dir)
        self._verbose = verbose
        self._custom_commands: list[tuple[str, str, Callable]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        topic: str = "default",
        on_token: Callable | None = None,
        on_tool_call: Callable | None = None,
        on_tool_result: Callable | None = None,
    ) -> str:
        """
        Send a message and return the agent's final text response.

        Callbacks (override verbose defaults if provided):
            on_token(text)        — each streamed token from the LLM
            on_tool_call(command) — the run() command string before execution
            on_tool_result(text)  — the command's output after execution
        """
        tc_cb = on_tool_call   or self._default_tool_call_cb()
        tr_cb = on_tool_result or self._default_tool_result_cb()
        return self._run(message, topic, on_token=on_token,
                         on_tool_call=tc_cb, on_tool_result=tr_cb)

    def stream(
        self,
        message: str,
        topic: str = "default",
        on_tool_call: Callable | None = None,
        on_tool_result: Callable | None = None,
    ) -> Iterator[str]:
        """
        Send a message and yield text tokens as they arrive.
        Tool calls still execute synchronously between LLM iterations.
        """
        import queue
        q:     queue.Queue = queue.Queue()
        error: list        = []
        tc_cb = on_tool_call   or self._default_tool_call_cb()
        tr_cb = on_tool_result or self._default_tool_result_cb()

        def worker():
            try:
                self._run(message, topic,
                          on_token=lambda t: q.put(t),
                          on_tool_call=tc_cb, on_tool_result=tr_cb)
            except Exception as e:
                error.append(e)
            finally:
                q.put(None)  # sentinel

        threading.Thread(target=worker, daemon=True).start()

        while True:
            token = q.get()
            if token is None:
                break
            yield token

        if error:
            raise error[0]

    def tool(self, name: str, description: str):
        """
        Decorator to register a custom command available to the LLM.

            @agent.tool("weather", "Get weather for a city. Usage: weather <city>")
            def get_weather(args, stdin=""):
                return f"Sunny, 22°C in {args[0]}"
        """
        def decorator(fn: Callable):
            self._custom_commands.append((name, description, fn))
            return fn
        return decorator

    def remember(self, content: str, category: str = "general"):
        """Store a permanent fact injected into every system prompt."""
        mem.store_fact(self._db, content, category)

    def facts(self) -> list[dict]:
        """List all stored facts."""
        return mem.list_facts(self._db)

    def forget(self, fact_id: int):
        """Delete a fact by ID."""
        mem.delete_fact(self._db, fact_id)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search past conversation summaries.
        Uses semantic search if embedding_model is configured, keyword search otherwise.
        """
        embedding_model = self._config.get("embedding_model")
        if embedding_model:
            emb = mem.get_embedding(self._client, embedding_model, query)
            if emb:
                return mem.search_semantic(self._db, emb, limit=limit)
        return mem.search_keyword(self._db, query, limit=limit)

    # ── Verbose helpers ───────────────────────────────────────────────────────

    def _default_tool_call_cb(self) -> Callable | None:
        if not self._verbose:
            return None
        def cb(command: str):
            print(f"\n[tool] run({command!r})", file=sys.stderr)
        return cb

    def _default_tool_result_cb(self) -> Callable | None:
        if not self._verbose:
            return None
        def cb(result: str):
            # Truncate long results so the terminal stays readable
            preview = result[:500].replace("\n", "\n       ")
            suffix  = f"\n       ... ({len(result)} chars total)" if len(result) > 500 else ""
            print(f"  →  {preview}{suffix}", file=sys.stderr)
        return cb

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(
        self,
        message: str,
        topic: str,
        on_token: Callable | None = None,
        on_tool_call: Callable | None = None,
        on_tool_result: Callable | None = None,
    ) -> str:
        db     = self._db
        client = self._client
        config = self._config
        model  = config["model"]

        # Resolve topic + create run record
        topic_row = db_mod.get_or_create_topic(db, topic)
        topic_id  = topic_row["id"]
        run_id    = db_mod.create_run(db, topic_id)

        # Build tool registry — set context fields FIRST so skill list etc. work
        registry = Registry()
        registry.topic_id = topic_id
        registry.data_dir = config["data_dir"]
        registry.work_dir = config.get("work_dir", ".")
        register_builtins(registry, db, client, config.get("embedding_model"))

        # Register any custom commands
        for cmd_name, cmd_desc, cmd_fn in self._custom_commands:
            registry.register(cmd_name, cmd_desc, cmd_fn)

        # Assemble context (system prompt + history + new message)
        system_prompt, messages = build_context(db, client, config, topic_id, message)

        # Run the agentic loop
        new_messages, final_response = run_loop(
            client, model, system_prompt, messages, registry,
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )

        # Persist this run's messages.
        # The new user message is always the last item in `messages` (from build_context).
        # It must be saved first so loaded history has proper user→assistant→tool sequences.
        user_msg = messages[-1]
        db_mod.save_messages(db, topic_id, run_id, [user_msg] + new_messages)
        db_mod.finish_run(db, run_id)

        # Generate and store summary in the background
        embedding_model = config.get("embedding_model")
        threading.Thread(
            target=mem.process_memory,
            args=(db, client, model, embedding_model, topic_id, run_id, messages + new_messages),
            daemon=True,
        ).start()

        return final_response
