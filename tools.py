"""
tools.py — Command registry, chaining parser, and all built-in commands.

The LLM has exactly one tool: run(command, stdin?).
Every capability — file I/O, memory, topics — is a Unix-style command
registered here and dispatched through the Registry.

Command chaining is supported:
  cmd1 && cmd2   run cmd2 only if cmd1 succeeded
  cmd1 || cmd2   run cmd2 only if cmd1 failed
  cmd1 ; cmd2    always run cmd2
  cmd1 | cmd2    pipe cmd1's output into cmd2's stdin
"""

import base64
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ── Command chaining ──────────────────────────────────────────────────────────

@dataclass
class Segment:
    raw: str
    op: str = "none"   # "none" | "and" | "or" | "seq" | "pipe"


def parse_chain(command: str) -> list[Segment]:
    segments: list[Segment] = []
    current: list[str] = []
    chars = list(command)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in ('"', "'"):                          # quoted string — pass through
            quote = ch
            current.append(ch)
            i += 1
            while i < len(chars) and chars[i] != quote:
                current.append(chars[i])
                i += 1
            if i < len(chars):
                current.append(chars[i])
        elif ch == "&" and i + 1 < len(chars) and chars[i + 1] == "&":
            segments.append(Segment("".join(current).strip(), "and"))
            current = []
            i += 1                                    # skip second &
        elif ch == "|" and i + 1 < len(chars) and chars[i + 1] == "|":
            segments.append(Segment("".join(current).strip(), "or"))
            current = []
            i += 1
        elif ch == "|":
            segments.append(Segment("".join(current).strip(), "pipe"))
            current = []
        elif ch == ";":
            segments.append(Segment("".join(current).strip(), "seq"))
            current = []
        else:
            current.append(ch)
        i += 1

    last = "".join(current).strip()
    if last:
        segments.append(Segment(last, "none"))
    return segments


def tokenize(command: str) -> list[str]:
    """Split a command string into tokens, respecting quoted strings."""
    tokens: list[str] = []
    current: list[str] = []
    in_quote = False
    quote_char = ""
    for ch in command:
        if in_quote:
            if ch == quote_char:
                in_quote = False
            else:
                current.append(ch)
        elif ch in ('"', "'"):
            in_quote = True
            quote_char = ch
        elif ch in (" ", "\t"):
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


# ── Registry ──────────────────────────────────────────────────────────────────

Handler = Callable[[list[str], str], str]   # (args, stdin) -> output string


class Registry:
    """
    Holds all registered commands.
    The LLM calls run(command="ls notes") and Registry dispatches it.
    """

    def __init__(self):
        self._handlers: dict[str, Handler] = {}
        self._help: dict[str, str] = {}
        # Set per-run by the Agent before registering commands
        self.data_dir: str = "./data"   # base for SQLite DB + topic files
        self.topic_id: str = ""         # scopes agent-internal files (screenshots, etc.)
        self.work_dir: str = "."        # user's working directory — where ls/cat/write operate

    def register(self, name: str, description: str, handler: Handler):
        self._handlers[name] = handler
        self._help[name] = description

    def command(self, name: str, description: str):
        """Decorator shorthand: @registry.command('ls', 'List files')"""
        def decorator(fn: Handler):
            self.register(name, description, fn)
            return fn
        return decorator

    def help(self) -> dict[str, str]:
        return dict(self._help)

    def exec(self, command: str, stdin: str = "") -> tuple[str, bool]:
        """
        Execute a command string (with optional chaining).
        Returns (output, is_error).
        """
        segments = parse_chain(command)
        if not segments:
            return "[error] empty command", True

        collected: list[str] = []
        last_output = ""
        last_err = False

        for i, seg in enumerate(segments):
            if i > 0:
                prev_op = segments[i - 1].op
                if prev_op == "and" and last_err:
                    continue
                if prev_op == "or" and not last_err:
                    continue

            seg_stdin = (
                stdin        if i == 0 else
                last_output  if segments[i - 1].op == "pipe" else
                ""
            )
            last_output, last_err = self._exec_single(seg.raw, seg_stdin)

            # Pipe: output flows to next command's stdin — don't collect yet
            if i < len(segments) - 1 and seg.op == "pipe":
                continue
            if last_output:
                collected.append(last_output)

        return "\n".join(collected), last_err

    def _exec_single(self, command: str, stdin: str) -> tuple[str, bool]:
        parts = tokenize(command)
        if not parts:
            return "[error] empty command", True

        name, args = parts[0], parts[1:]
        handler = self._handlers.get(name)
        if not handler:
            available = ", ".join(sorted(self._handlers.keys()))
            return f"[error] unknown command: {name}\nAvailable: {available}", True

        try:
            return handler(args, stdin), False
        except Exception as e:
            return f"[error] {name}: {e}", True


# ── File path helpers ─────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def human_size(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f}MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f}KB"
    return f"{n}B"


def resolve_path(name: str, work_dir: str) -> Path:
    """
    Resolve a file path relative to work_dir (the user's working directory).
    Prevents path traversal outside work_dir.

    Examples:
        resolve_path("notes.md", ".")         → /cwd/notes.md
        resolve_path("src/app.py", "/my/project") → /my/project/src/app.py
    """
    root = Path(work_dir).resolve()
    if name in ("", "."):
        return root
    resolved = (root / name).resolve()
    if not str(resolved).startswith(str(root) + "/") and resolved != root:
        raise ValueError(f"Path escapes working directory: {name}")
    return resolved


# ── Register all built-in commands ───────────────────────────────────────────

def register_builtins(registry: Registry, db: sqlite3.Connection, client, embedding_model: str):
    """
    Register every built-in command on the registry.
    db / client / embedding_model are captured as closures.
    """
    # Shorthand: resolve a path within the user's working directory (read at call time)
    def rp(name: str) -> Path:
        return resolve_path(name, registry.work_dir)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @registry.command("echo", "Echo args or stdin")
    def cmd_echo(args, stdin=""):
        return " ".join(args) if args else stdin

    @registry.command("time", "Return current date and time")
    def cmd_time(args, stdin=""):
        return time.strftime("%Y-%m-%d %H:%M:%S %Z")

    @registry.command("help", "List all available commands")
    def cmd_help(args, stdin=""):
        return "\n".join(f"  {k} — {v.splitlines()[0]}" for k, v in sorted(registry.help().items()))

    @registry.command("grep", "Filter stdin lines. Flags: -i (case-insensitive) -v (invert) -c (count only)")
    def cmd_grep(args, stdin=""):
        flags   = {a for a in args if a.startswith("-")}
        pattern = next((a for a in args if not a.startswith("-")), None)
        if not pattern:
            raise ValueError("usage: grep [-i] [-v] [-c] <pattern>")
        lines = stdin.split("\n")
        def matches(line):
            h = line.lower() if "-i" in flags else line
            p = pattern.lower() if "-i" in flags else pattern
            return p in h
        matched = [l for l in lines if matches(l) != ("-v" in flags)]
        return str(len(matched)) if "-c" in flags else "\n".join(matched)

    @registry.command("head", "First N lines of stdin (default 10). Usage: head [N]")
    def cmd_head(args, stdin=""):
        n = int(args[0]) if args else 10
        return "\n".join(stdin.split("\n")[:n])

    @registry.command("tail", "Last N lines of stdin (default 10). Usage: tail [N]")
    def cmd_tail(args, stdin=""):
        n = int(args[0]) if args else 10
        return "\n".join(stdin.split("\n")[-n:])

    @registry.command("wc", "Count lines/words/chars. Flags: -l -w -c")
    def cmd_wc(args, stdin=""):
        if "-l" in args: return str(len(stdin.split("\n")))
        if "-w" in args: return str(len(stdin.split()))
        if "-c" in args: return str(len(stdin))
        return f"{len(stdin.split(chr(10)))} lines, {len(stdin.split())} words, {len(stdin)} chars"

    # ── File I/O ──────────────────────────────────────────────────────────────

    @registry.command("ls", "List files. Usage: ls [dir]  (Unix flags like -l, -la are accepted and ignored)")
    def cmd_ls(args, stdin=""):
        # Strip Unix flags — the LLM often adds -l, -la, -lh; we always show a long listing
        path_args = [a for a in args if not a.startswith("-")]
        path = rp(path_args[0] if path_args else ".")
        if not path.exists():
            return "(empty directory)"
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return "(empty directory)"
        lines = []
        for e in entries:
            size = human_size(e.stat().st_size) if e.is_file() else "-"
            lines.append(f"{'f' if e.is_file() else 'd'}  {size:<8}  {e.name}{'/' if e.is_dir() else ''}")
        return "\n".join(lines)

    @registry.command("cat", "Read a file. Usage: cat <path> [-b for base64 output]")
    def cmd_cat(args, stdin=""):
        b64 = "-b" in args
        path_args = [a for a in args if a != "-b"]
        if not path_args:
            raise ValueError("usage: cat <path>")
        p = rp(path_args[0])
        data = p.read_bytes()
        if b64:
            result = base64.b64encode(data).decode()
            if is_image(p.name):
                result += f"\nView: ![image](file://{p})"
            return result
        return data.decode("utf-8", errors="replace")

    @registry.command("write", "Write a file. Usage: write <path> [content] or pipe stdin. Use -b for base64 input")
    def cmd_write(args, stdin=""):
        b64  = "-b" in args
        rest = [a for a in args if a != "-b"]
        if not rest:
            raise ValueError("usage: write <path> [content]")
        p = rp(rest[0])
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = " ".join(rest[1:]) if len(rest) > 1 else stdin
        data = base64.b64decode(raw.strip()) if b64 else raw.encode()
        p.write_bytes(data)
        result = f"Written {human_size(len(data))} → {rest[0]}"
        if is_image(p.name):
            result += f"\nView: ![image](file://{p})"
        return result

    @registry.command("see", "View an image (auto-attaches to vision). Usage: see <path>")
    def cmd_see(args, stdin=""):
        if not args:
            raise ValueError("usage: see <image-path>")
        p = rp(args[0])
        if not is_image(p.name):
            raise ValueError(f"not an image file: {args[0]} (use cat for text files)")
        st = p.stat()
        return f"Image: {args[0]} ({human_size(st.st_size)})\nView: ![image](file://{p})"

    @registry.command("stat", "File info (size, type, modified). Usage: stat <path>")
    def cmd_stat(args, stdin=""):
        if not args:
            raise ValueError("usage: stat <path>")
        p  = rp(args[0])
        st = p.stat()
        return (
            f"File: {args[0]}\n"
            f"Size: {human_size(st.st_size)} ({st.st_size} bytes)\n"
            f"Modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}"
        )

    @registry.command("rm", "Delete a file or directory. Usage: rm <path>")
    def cmd_rm(args, stdin=""):
        if not args:
            raise ValueError("usage: rm <path>")
        p = rp(args[0])
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        return f"Removed {args[0]}"

    @registry.command("cp", "Copy a file. Usage: cp <src> <dst>")
    def cmd_cp(args, stdin=""):
        if len(args) < 2:
            raise ValueError("usage: cp <src> <dst>")
        src, dst = rp(args[0]), rp(args[1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return f"Copied {args[0]} → {args[1]}"

    @registry.command("mv", "Move or rename a file. Usage: mv <src> <dst>")
    def cmd_mv(args, stdin=""):
        if len(args) < 2:
            raise ValueError("usage: mv <src> <dst>")
        src, dst = rp(args[0]), rp(args[1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return f"Moved {args[0]} → {args[1]}"

    @registry.command("mkdir", "Create a directory. Usage: mkdir <dir>")
    def cmd_mkdir(args, stdin=""):
        if not args:
            raise ValueError("usage: mkdir <dir>")
        rp(args[0]).mkdir(parents=True, exist_ok=True)
        return f"Created {args[0]}"

    # ── Memory ────────────────────────────────────────────────────────────────

    import Operon.memory as mem

    @registry.command(
        "memory",
        "Manage persistent memory.\n"
        "  memory store <note>    — store a permanent fact\n"
        "  memory facts           — list all stored facts\n"
        "  memory forget <id>     — delete a fact by ID\n"
        "  memory recent [n]      — show last N run summaries\n"
        "  memory search <query>  — search past conversations",
    )
    def cmd_memory(args, stdin=""):
        if not args:
            raise ValueError("usage: memory store|facts|forget|recent|search")
        sub = args[0]

        if sub == "store":
            note = " ".join(args[1:]) or stdin
            if not note:
                raise ValueError("usage: memory store <note>")
            mem.store_fact(db, note)
            return "Fact stored."

        if sub == "facts":
            facts = mem.list_facts(db)
            if not facts:
                return "No facts stored."
            return "\n".join(f"  #{f['id']} [{f['category']}] {f['content']}" for f in facts)

        if sub == "forget":
            if len(args) < 2:
                raise ValueError("usage: memory forget <id>")
            mem.delete_fact(db, int(args[1]))
            return f"Fact #{args[1]} deleted."

        if sub == "recent":
            n = int(args[1]) if len(args) > 1 else 5
            summaries = mem.get_recent_summaries(db, n)
            return "\n".join(f"  {s}" for s in summaries) if summaries else "No summaries yet."

        if sub == "search":
            query = " ".join(args[1:])
            if not query:
                raise ValueError("usage: memory search <query>")
            emb = mem.get_embedding(client, embedding_model, query) if embedding_model else None
            results = mem.search_semantic(db, emb, limit=5) if emb else mem.search_keyword(db, query)
            if not results:
                return "No matching memories found."
            lines = []
            for r in results:
                sim = f" ({r['similarity']:.0%})" if "similarity" in r else ""
                lines.append(f"  {r['summary']}{sim}")
            return "\n".join(lines)

        raise ValueError(f"unknown subcommand: memory {sub}")

    # ── Topics ────────────────────────────────────────────────────────────────

    import Operon.db as db_mod

    @registry.command(
        "topic",
        "View conversation topics.\n"
        "  topic list         — list all topics\n"
        "  topic info <id>    — show topic details and recent runs",
    )
    def cmd_topic(args, stdin=""):
        if not args or args[0] == "list":
            topics = db_mod.list_topics(db)
            if not topics:
                return "No topics yet."
            lines = [f"  {t['id']}  {t['name']}  ({t['message_count']} msgs)" for t in topics]
            return "Topics:\n" + "\n".join(lines)

        if args[0] == "info":
            if len(args) < 2:
                raise ValueError("usage: topic info <id>")
            tid = args[1]
            runs = db.execute(
                "SELECT r.id, r.status, r.started_at, "
                "COALESCE(s.summary, '') AS summary "
                "FROM runs r LEFT JOIN summaries s ON s.run_id = r.id "
                "WHERE r.topic_id = ? ORDER BY r.started_at DESC LIMIT 10",
                (tid,),
            ).fetchall()
            if not runs:
                return f"No runs for topic {tid}."
            lines = [f"Runs for topic {tid}:"]
            for run in runs:
                ts = time.strftime("%H:%M:%S", time.localtime(run["started_at"]))
                lines.append(f"  {run['id']} [{ts}] {run['status']}")
                if run["summary"]:
                    lines.append(f"    {run['summary']}")
            return "\n".join(lines)

        raise ValueError(f"unknown subcommand: topic {args[0]}")

    # ── Skills + Browser ──────────────────────────────────────────────────────

    from Operon.skills import register_skill_commands
    from Operon.browser import register_browser_commands

    register_skill_commands(registry)
    register_browser_commands(registry)
