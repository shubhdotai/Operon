"""
db.py — SQLite persistence: topics, runs, messages, summaries, facts.
One connection per Agent instance, WAL mode for safe concurrent reads.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    topic_id    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    started_at  INTEGER NOT NULL,
    finished_at INTEGER
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id     TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT,
    tool_calls   TEXT,
    tool_call_id TEXT,
    reasoning    TEXT,
    created_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id        TEXT NOT NULL,
    run_id          TEXT,
    summary         TEXT NOT NULL,
    user_message    TEXT,
    embedding       TEXT,
    embedding_model TEXT,
    created_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    category   TEXT NOT NULL DEFAULT 'general',
    created_at INTEGER NOT NULL
);
"""

# Migrations applied to existing databases (safe to re-run — errors on duplicate columns are swallowed)
_MIGRATIONS = [
    "ALTER TABLE messages ADD COLUMN reasoning TEXT",
    "ALTER TABLE summaries ADD COLUMN embedding_model TEXT",
]

# FTS5 virtual table + trigger for keyword search over summaries
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
    summary, user_message,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS summaries_fts_ai
AFTER INSERT ON summaries BEGIN
    INSERT INTO summaries_fts(rowid, summary, user_message)
    VALUES (new.id, new.summary, COALESCE(new.user_message, ''));
END;
"""

_has_fts: bool = False


def has_fts() -> bool:
    """Return True if the summaries_fts FTS5 table is available."""
    return _has_fts


def open_db(data_dir: str) -> sqlite3.Connection:
    global _has_fts
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(
        str(Path(data_dir) / "agent.db"),
        check_same_thread=False,
    )
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    db.commit()

    # Apply column-level migrations for existing databases
    for statement in _MIGRATIONS:
        try:
            db.execute(statement)
            db.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise

    # Set up FTS5 keyword search (requires SQLite built with FTS5 support)
    try:
        db.executescript(_FTS_SCHEMA)
        db.commit()
        # Populate FTS for any summaries inserted before FTS was set up
        fts_count = db.execute("SELECT COUNT(*) FROM summaries_fts").fetchone()[0]
        summary_count = db.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        if fts_count != summary_count:
            db.execute("DELETE FROM summaries_fts")
            db.execute(
                "INSERT INTO summaries_fts(rowid, summary, user_message) "
                "SELECT id, summary, COALESCE(user_message, '') FROM summaries"
            )
            db.commit()
        _has_fts = True
    except sqlite3.OperationalError:
        _has_fts = False  # FTS5 not available — keyword search falls back to LIKE

    return db


def get_or_create_topic(db: sqlite3.Connection, name: str) -> dict:
    row = db.execute("SELECT * FROM topics WHERE name = ?", (name,)).fetchone()
    if row:
        return dict(row)
    topic_id = uuid.uuid4().hex[:8]
    db.execute(
        "INSERT INTO topics (id, name, created_at) VALUES (?, ?, ?)",
        (topic_id, name, int(time.time())),
    )
    db.commit()
    return {"id": topic_id, "name": name, "created_at": int(time.time())}


def create_run(db: sqlite3.Connection, topic_id: str) -> str:
    run_id = uuid.uuid4().hex[:8]
    db.execute(
        "INSERT INTO runs (id, topic_id, status, started_at) VALUES (?, ?, 'running', ?)",
        (run_id, topic_id, int(time.time())),
    )
    db.commit()
    return run_id


def finish_run(db: sqlite3.Connection, run_id: str, status: str = "done"):
    db.execute(
        "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
        (status, int(time.time()), run_id),
    )
    db.commit()


def save_messages(db: sqlite3.Connection, topic_id: str, run_id: str, messages: list):
    now = int(time.time())
    for msg in messages:
        # Extract text content — multimodal content stores only text parts
        content = None
        if isinstance(msg.get("content"), str):
            content = msg["content"]
        elif isinstance(msg.get("content"), list):
            texts = [p["text"] for p in msg["content"] if p.get("type") == "text"]
            content = "\n".join(texts) if texts else None

        tool_calls = json.dumps(msg["tool_calls"]) if msg.get("tool_calls") else None
        tool_call_id = msg.get("tool_call_id")
        reasoning = msg.get("reasoning")

        db.execute(
            "INSERT INTO messages "
            "(topic_id, run_id, role, content, tool_calls, tool_call_id, reasoning, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (topic_id, run_id, msg["role"], content, tool_calls, tool_call_id, reasoning, now),
        )
    db.commit()


def load_messages_by_run(db: sqlite3.Connection, run_id: str) -> list:
    rows = db.execute(
        "SELECT role, content, tool_calls, tool_call_id, reasoning "
        "FROM messages WHERE run_id = ? ORDER BY id ASC",
        (run_id,),
    ).fetchall()
    return [_row_to_message(r) for r in rows]


def get_completed_runs(db: sqlite3.Connection, topic_id: str) -> list:
    rows = db.execute(
        "SELECT id, topic_id, started_at FROM runs "
        "WHERE topic_id = ? AND status = 'done' ORDER BY started_at ASC",
        (topic_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_topics(db: sqlite3.Connection) -> list:
    rows = db.execute(
        """
        SELECT t.id, t.name, t.created_at,
               COUNT(m.id)                              AS message_count,
               COALESCE(MAX(m.created_at), t.created_at) AS last_active
        FROM topics t
        LEFT JOIN messages m ON m.topic_id = t.id
        GROUP BY t.id
        ORDER BY last_active DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_summary_for_run(db: sqlite3.Connection, run_id: str) -> str | None:
    row = db.execute(
        "SELECT summary FROM summaries WHERE run_id = ? LIMIT 1", (run_id,)
    ).fetchone()
    return row["summary"] if row else None


def _row_to_message(row: sqlite3.Row) -> dict:
    msg = {"role": row["role"]}
    content = row["content"]
    tool_calls_json = row["tool_calls"]

    if tool_calls_json:
        msg["tool_calls"] = json.loads(tool_calls_json)
        # Assistant with tool_calls: content is optional (omit if empty)
        if content:
            msg["content"] = content
    else:
        # user / tool / assistant-without-tool-calls: content must be a string
        msg["content"] = content or ""

    if row["tool_call_id"]:
        msg["tool_call_id"] = row["tool_call_id"]

    # reasoning is only present in load_messages_by_run rows
    try:
        if row["reasoning"]:
            msg["reasoning"] = row["reasoning"]
    except IndexError:
        pass

    return msg
