"""
memory.py — Three-layer memory: facts, summaries, semantic search.

Facts:     persistent notes injected into every system prompt.
Summaries: LLM-generated 1-3 sentence recap stored after each run.
Search:    FTS5 keyword search + cosine similarity over embedding vectors.
           Falls back to LIKE-based keyword search if FTS5 is unavailable.
"""

import json
import math
import sqlite3
import time
from typing import Optional

from . import db as _db


# ── Facts ────────────────────────────────────────────────────────────────────

def store_fact(db: sqlite3.Connection, content: str, category: str = "general"):
    db.execute(
        "INSERT INTO facts (content, category, created_at) VALUES (?, ?, ?)",
        (content, category, int(time.time())),
    )
    db.commit()


def list_facts(db: sqlite3.Connection) -> list:
    rows = db.execute(
        "SELECT id, content, category, created_at FROM facts ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def delete_fact(db: sqlite3.Connection, fact_id: int):
    db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    db.commit()


# ── Summaries ────────────────────────────────────────────────────────────────

def store_summary(
    conn: sqlite3.Connection,
    topic_id: str,
    run_id: str,
    summary: str,
    user_message: str = "",
    embedding: list | None = None,
    embedding_model: str | None = None,
):
    conn.execute(
        "INSERT INTO summaries "
        "(topic_id, run_id, summary, user_message, embedding, embedding_model, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            topic_id,
            run_id,
            summary,
            user_message,
            json.dumps(embedding) if embedding else None,
            embedding_model,
            int(time.time()),
        ),
    )
    conn.commit()
    # FTS insert is handled by the trigger created in open_db


def get_recent_summaries(conn: sqlite3.Connection, limit: int = 5) -> list[str]:
    rows = conn.execute(
        "SELECT summary FROM summaries ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    # Return in chronological order (oldest first)
    return [r["summary"] for r in reversed(rows)]


# ── Semantic search ───────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def search_semantic(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 3,
    topic_id: Optional[str] = None,
) -> list[dict]:
    if topic_id:
        rows = conn.execute(
            "SELECT id, topic_id, run_id, summary, user_message, embedding, created_at "
            "FROM summaries WHERE embedding IS NOT NULL AND topic_id = ?",
            (topic_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, topic_id, run_id, summary, user_message, embedding, created_at "
            "FROM summaries WHERE embedding IS NOT NULL"
        ).fetchall()

    scored = []
    for row in rows:
        emb = json.loads(row["embedding"])
        sim = cosine_similarity(query_embedding, emb)
        if sim >= 0.5:
            scored.append({**dict(row), "similarity": sim})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]


def search_keyword(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 5,
    topic_id: Optional[str] = None,
) -> list[dict]:
    if _db.has_fts():
        # FTS5 — escape double-quotes and wrap as a phrase for safety
        fts_query = '"' + query.replace('"', '""') + '"'
        if topic_id:
            rows = conn.execute(
                "SELECT s.id, s.topic_id, s.run_id, s.summary, s.user_message, s.created_at "
                "FROM summaries_fts fts "
                "JOIN summaries s ON s.id = fts.rowid "
                "WHERE summaries_fts MATCH ? AND s.topic_id = ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, topic_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT s.id, s.topic_id, s.run_id, s.summary, s.user_message, s.created_at "
                "FROM summaries_fts fts "
                "JOIN summaries s ON s.id = fts.rowid "
                "WHERE summaries_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # Fallback: LIKE-based search (no FTS5)
    pattern = f"%{query}%"
    if topic_id:
        rows = conn.execute(
            "SELECT id, topic_id, run_id, summary, user_message, created_at "
            "FROM summaries WHERE (summary LIKE ? OR user_message LIKE ?) AND topic_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (pattern, pattern, topic_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, topic_id, run_id, summary, user_message, created_at "
            "FROM summaries WHERE summary LIKE ? OR user_message LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def search_memory(
    conn: sqlite3.Connection,
    client,
    embedding_model: Optional[str],
    query: str,
    limit: int = 5,
    topic_id: Optional[str] = None,
) -> list[dict]:
    """
    Combined semantic + keyword search, deduplicated and ranked by similarity.
    Mirrors the TypeScript searchMemory() function.
    """
    results: list[dict] = []

    # 1. Semantic search (if embedding model available)
    if embedding_model:
        query_embedding = get_embedding(client, embedding_model, query)
        if query_embedding:
            results.extend(search_semantic(conn, query_embedding, limit=limit * 2, topic_id=topic_id))

    # 2. Keyword search to fill gaps
    if len(results) < limit:
        seen_ids = {r["id"] for r in results}
        keyword_results = search_keyword(conn, query, limit=limit * 2, topic_id=topic_id)
        for r in keyword_results:
            if r["id"] not in seen_ids:
                results.append(r)

    # 3. Enrich with topic names
    topic_cache: dict[str, str] = {}
    for r in results:
        tid = r.get("topic_id")
        if tid and tid not in topic_cache:
            row = conn.execute("SELECT name FROM topics WHERE id = ?", (tid,)).fetchone()
            if row:
                topic_cache[tid] = row["name"]
        if tid and tid in topic_cache:
            r["topic_name"] = topic_cache[tid]

    return results[:limit]


# ── Embedding API ─────────────────────────────────────────────────────────────

def get_embedding(client, model: str, text: str) -> list[float] | None:
    """Call the OpenAI embeddings endpoint. Returns None on failure."""
    if not model or not text.strip():
        return None
    try:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding
    except Exception:
        return None


# ── Summary generation ────────────────────────────────────────────────────────

def _render_trajectory(messages: list) -> str:
    """Convert a message list into a readable transcript for summarization."""
    lines = []
    for msg in messages:
        role = msg["role"]
        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = next((p["text"] for p in content if p.get("type") == "text"), "")
            lines.append(f"[user] {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    lines.append(f"[tool_call] run({tc['function']['arguments']})")
            if msg.get("content"):
                lines.append(f"[assistant] {msg['content']}")
        elif role == "tool":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = next((p["text"] for p in content if p.get("type") == "text"), "")
            lines.append(f"[tool_result] {str(content)[:400]}")
    return "\n".join(lines)


def generate_summary(client, model: str, messages: list, recent_summaries: list[str]) -> str:
    trajectory = _render_trajectory(messages)
    if len(trajectory) > 6000:
        trajectory = trajectory[:6000] + "\n... (truncated)"

    context = ""
    if recent_summaries:
        context = "Recent conversation summaries (for context):\n"
        context += "\n".join(f"- {s}" for s in recent_summaries) + "\n\n"

    prompt = (
        f"{context}"
        "Summarize the following conversation in 1-3 sentences.\n"
        "Include: what the user asked for, what actions were taken, and the final result.\n\n"
        f"Conversation:\n{trajectory}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conversation summarizer. "
                        "Output only the summary, nothing else. Be concise."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback: use the first user message text
        for msg in messages:
            if msg["role"] == "user" and msg.get("content"):
                return str(msg["content"])[:120]
        return ""


def process_memory(
    conn: sqlite3.Connection,
    client,
    model: str,
    embedding_model: str | None,
    topic_id: str,
    run_id: str,
    messages: list,
):
    """
    Generate and store a summary + embedding for a completed run.
    Called in a background thread after each successful run.
    """
    recent = get_recent_summaries(conn, limit=5)
    summary = generate_summary(client, model, messages, recent)
    if not summary:
        return

    # The first user message is stored alongside the summary for keyword search
    user_message = ""
    for msg in messages:
        if msg["role"] == "user" and msg.get("content"):
            user_message = str(msg["content"])[:200]
            break

    embedding = get_embedding(client, embedding_model, summary) if embedding_model else None
    store_summary(conn, topic_id, run_id, summary, user_message, embedding, embedding_model)
