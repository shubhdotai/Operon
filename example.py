"""
example.py — Demonstrates the key features of the Operon agent framework.

Operon is a minimal agentic framework built around a single unified tool
interface (Unix-style commands), persistent memory, topic-based conversations,
browser automation, and a reusable skills system.

Run any section by uncommenting it.
"""

import os
from Operon import Agent

# ── Agent setup ───────────────────────────────────────────────────────────────
# api_key   — OpenAI-compatible API key
# model     — any chat model (gpt-4.1-mini, gpt-4o, etc.)
# data_dir  — SQLite DB, topic files, and screenshots are stored here
# work_dir  — file commands (ls/cat/write/rm…) are scoped to this directory
# verbose   — print every tool call and result to stderr

agent = Agent(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1-mini",
    system_prompt="You are a helpful assistant.",
    name="demo-agent",
    data_dir="./data",
    work_dir=".",
    verbose=True,
)


# ── 1. Basic chat ─────────────────────────────────────────────────────────────
# The agent has access to a Unix-style command set: ls, cat, write, grep, etc.
# Commands can be chained with |, &&, ;, and ||.

# response = agent.chat("List the Python files in the current directory.")
# print(response)

# response = agent.chat(
#     "Read example.py, count how many lines it has, then write a one-line "
#     "summary to summary.txt."
# )
# print(response)


# ── 2. Topics — separate memory threads ───────────────────────────────────────
# Each topic maintains its own conversation history and summary chain.
# Use topics to keep unrelated tasks isolated.

# agent.chat("Remember: this project targets Python 3.12.", topic="project-info")
# agent.chat("The main entry point is example.py.", topic="project-info")
# print(agent.chat("What do you know about this project?", topic="project-info"))

# print(agent.chat("Write a haiku about the ocean.", topic="creative"))


# ── 3. Persistent memory (facts) ─────────────────────────────────────────────
# Facts are stored permanently and injected into every system prompt.
# Use them to give the agent standing context across all conversations.

# agent.remember("Always prefer concise responses.", category="style")
# agent.remember("The project uses pytest for testing.", category="testing")

# # List all facts
# for fact in agent.facts():
#     print(f"[{fact['id']}] [{fact['category']}] {fact['content']}")

# # Remove a fact by ID
# agent.forget(1)


# ── 4. Search past conversations ──────────────────────────────────────────────
# Operon summarizes every run and indexes it for semantic or keyword search.
# Semantic search requires an embedding_model (default: text-embedding-3-small).

# results = agent.search("Python version")
# for r in results:
#     similarity = f"{r.get('similarity', 0):.0%}" if "similarity" in r else "n/a"
#     print(f"({similarity}) {r['summary']}")


# ── 5. Streaming output ───────────────────────────────────────────────────────
# stream() yields tokens in real time. Tool calls still execute synchronously
# between LLM iterations; only the text output is streamed.

# print("Streaming response:")
# for token in agent.stream("Write a two-line haiku about code."):
#     print(token, end="", flush=True)
# print()


# ── 6. Callbacks for observability ────────────────────────────────────────────
# on_token, on_tool_call, and on_tool_result let you hook into every step
# without enabling full verbose mode.

# def on_call(cmd):
#     print(f"\n[tool] >>> {cmd}")

# def on_result(out):
#     print(f"[tool] <<< {out[:120]}{'…' if len(out) > 120 else ''}")

# response = agent.chat(
#     "List files, then write 'hello' to hello.txt.",
#     on_tool_call=on_call,
#     on_tool_result=on_result,
# )
# print(response)


# ── 7. Custom tools ───────────────────────────────────────────────────────────
# Register any Python function as a command the LLM can call.
# Handler signature: (args: list[str], stdin: str) -> str

# @agent.tool("weather", "Get the current weather for a city. Usage: weather <city>")
# def get_weather(args, stdin=""):
#     city = " ".join(args) if args else stdin.strip()
#     # Replace with a real weather API call as needed.
#     return f"Sunny, 22°C in {city}"

# print(agent.chat("What's the weather like in Tokyo and Paris?"))


# ── 8. Browser & web ─────────────────────────────────────────────────────────
# The browser command drives a real Chrome instance with persistent login state
# (via bb-browser). fetch and search work through the same authenticated context.

# response = agent.chat(
#     "Search the web for 'OpenAI GPT-4.1' and summarize the top 3 results."
# )
# print(response)

# response = agent.chat(
#     "Go to https://example.com, take a screenshot, and describe what you see."
# )
# print(response)


# ── 9. Skills ─────────────────────────────────────────────────────────────────
# Skills are reusable instruction sets stored as markdown files in data/skills/.
# The agent can list, load, create, and search skills at runtime.

# # Create a skill
# agent.chat(
#     "Create a skill named 'code-review' with description 'Guidelines for "
#     "reviewing Python pull requests' and content: Always check for type hints, "
#     "docstrings on public functions, and test coverage above 80%."
# )

# # Use it in a later conversation
# agent.chat("Load the code-review skill, then review example.py.")


# ── 10. Command chaining (Unix-style) ─────────────────────────────────────────
# All built-in commands support | && ; || chaining, just like a shell.

# response = agent.chat(
#     "Run: ls | grep .py | wc -l  — then tell me how many Python files exist."
# )
# print(response)
