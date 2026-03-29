# Operon

A minimal Python library for building agentic loops on top of OpenAI-compatible models. Import it, create an `Agent`, and start chatting — the agent autonomously uses tools, browses the web, manages files, and remembers past conversations.

## How it works

The agent runs a loop: send a message → LLM decides if it needs a tool → execute the tool → feed result back → repeat until done. Every capability is exposed as a Unix-style command through a single `run(command)` tool, so the LLM can chain commands with `|`, `&&`, and `;`.

**Core modules:**

| File | Purpose |
|------|---------|
| `__init__.py` | `Agent` class — the public API |
| `loop.py` | Agentic loop: LLM → tool call → execute → repeat |
| `tools.py` | Command registry + all built-in commands |
| `context.py` | Builds the full LLM context (system prompt, history, recall) |
| `memory.py` | Facts, run summaries, semantic + keyword search |
| `db.py` | SQLite persistence (topics, runs, messages, summaries, facts) |
| `skills.py` | Reusable markdown instruction files the agent can load on demand |
| `browser.py` | Real-browser automation via `bb-browser` (fetch, search, interact) |

## Setup

```bash
git clone <repo>
cd Operon
pip install -r requirements.txt

# For browser commands (optional)
npm install -g bb-browser
playwright install chromium
```

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

## Quickstart

```python
from Operon import Agent

agent = Agent(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1-mini",
    system_prompt="You are a helpful assistant.",
    data_dir="./data",    # SQLite DB and topic files stored here
    work_dir=".",         # ls/cat/write operate in this directory
    verbose=True,         # print tool calls to stderr
)

print(agent.chat("List the files in the current directory."))
```

See `example.py` for runnable demos of every feature.

## What you can do

**Chat and stream**
```python
# Single response
response = agent.chat("Summarize the README.")

# Stream tokens as they arrive
for token in agent.stream("Write a haiku about Python."):
    print(token, end="", flush=True)
```

**Separate conversation threads (topics)**
```python
agent.chat("My project uses Python 3.12.", topic="work")
agent.chat("What Python version are we using?", topic="work")  # remembers
```

**Persistent memory (facts)**
```python
agent.remember("User prefers concise responses.")

for fact in agent.facts():
    print(f"[{fact['id']}] {fact['content']}")

agent.forget(fact_id=1)
```

**Search past conversations**
```python
# Semantic search (requires embedding_model, enabled by default)
results = agent.search("Python version")
for r in results:
    print(f"({r.get('similarity', 0):.0%}) {r['summary']}")
```

**Custom tools**
```python
@agent.tool("weather", "Get weather for a city. Usage: weather <city>")
def get_weather(args, stdin=""):
    return f"Sunny, 22°C in {args[0]}"

print(agent.chat("What's the weather in Tokyo?"))
```

**File operations** (via built-in commands the agent calls autonomously)

The agent can `ls`, `cat`, `write`, `cp`, `mv`, `rm`, `mkdir`, and `stat` files inside `work_dir`.

**Web browsing** (requires `bb-browser`)

The agent can `search <query>`, `fetch <url>`, or use `browser <action>` for full Chrome automation with your real login state — including Twitter, Google, and any authenticated site.

**Skills**

Skills are markdown instruction files the agent can discover and load at runtime:

```python
# The agent calls these internally:
# skill list
# skill load <name>
# skill create <name> --desc "..." (with content via stdin)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | required | OpenAI API key |
| `model` | `gpt-4.1-mini` | Chat model |
| `system_prompt` | `"You are a helpful assistant."` | Custom instructions |
| `name` | `"agent"` | Agent identity in the prompt |
| `data_dir` | `"./data"` | Where to store the DB and topic files |
| `work_dir` | `"."` | Working directory for file commands |
| `embedding_model` | `"text-embedding-3-small"` | Set to `None` to disable semantic search |
| `base_url` | `None` | Custom API base URL for OpenAI-compatible providers |
| `verbose` | `False` | Print tool calls and results to stderr |

## Data directory

`data/` is created automatically and contains:
- `agent.db` — SQLite database (conversations, summaries, facts)
- `topics/` — per-topic files (screenshots, etc.)
- `skills/` — skill markdown files

This directory is in `.gitignore` and stays local.
