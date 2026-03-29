"""
skills.py — Reusable instruction files stored as markdown with YAML frontmatter.

Skills live in data/skills/ and let the LLM load named instruction sets on demand.
Each skill is a .md file:

    ---
    description: "How to write a conventional git commit"
    ---

    1. Summarize the change in one line (max 72 chars)
    2. Use present tense: "Add feature" not "Added feature"
    ...

The LLM discovers skills via `skill list`, loads them via `skill load <name>`,
and can even create new ones via `skill create`.
"""

import re
import shutil
from pathlib import Path


# ── File helpers ──────────────────────────────────────────────────────────────

def _skills_dir(data_dir: str) -> Path:
    return Path(data_dir) / "skills"


def _seed_skills_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "seed" / "skills"


def _skill_path(data_dir: str, name: str) -> Path:
    return _skills_dir(data_dir) / f"{name}.md"


def _ensure_seed_skills(data_dir: str):
    """
    Mirror bundled seed skills into the runtime data directory without
    overwriting user-edited skill files.
    """
    dst_dir = _skills_dir(data_dir)
    seed_dir = _seed_skills_dir()
    if not seed_dir.exists():
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(seed_dir.glob("*.md")):
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def _parse_skill(raw: str) -> tuple[str, str]:
    """Split YAML frontmatter from body. Returns (description, body)."""
    if not raw.startswith("---\n"):
        return "", raw
    end = raw.find("\n---", 4)
    if end < 0:
        return "", raw
    frontmatter = raw[4:end]
    body = raw[end + 4:].strip()
    m = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', frontmatter, re.MULTILINE)
    return (m.group(1) if m else ""), body


def _write_skill(path: Path, description: str, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'---\ndescription: "{description}"\n---\n\n{content}\n')


# ── Public API ────────────────────────────────────────────────────────────────

def list_skills(data_dir: str) -> list[dict]:
    _ensure_seed_skills(data_dir)
    d = _skills_dir(data_dir)
    if not d.exists():
        return []
    return [
        {"name": p.stem, "description": _parse_skill(p.read_text())[0]}
        for p in sorted(d.glob("*.md"))
    ]


def load_skill(data_dir: str, name: str) -> tuple[str, str]:
    """Returns (description, body). Raises FileNotFoundError if not found."""
    _ensure_seed_skills(data_dir)
    p = _skill_path(data_dir, name)
    if not p.exists():
        raise FileNotFoundError(f"skill '{name}' not found")
    return _parse_skill(p.read_text())


def create_skill(data_dir: str, name: str, description: str, content: str):
    _ensure_seed_skills(data_dir)
    p = _skill_path(data_dir, name)
    if p.exists():
        raise FileExistsError(f"skill '{name}' already exists — use `skill update` to modify")
    _write_skill(p, description, content)


def update_skill(data_dir: str, name: str, description: str | None, content: str | None):
    _ensure_seed_skills(data_dir)
    p = _skill_path(data_dir, name)
    if not p.exists():
        raise FileNotFoundError(f"skill '{name}' not found")
    old_desc, old_body = _parse_skill(p.read_text())
    _write_skill(p, description or old_desc, content or old_body)


def delete_skill(data_dir: str, name: str):
    _ensure_seed_skills(data_dir)
    p = _skill_path(data_dir, name)
    if not p.exists():
        raise FileNotFoundError(f"skill '{name}' not found")
    p.unlink()


# ── Command registration ──────────────────────────────────────────────────────

def register_skill_commands(registry):
    """Register the `skill` command. Uses registry.data_dir at call time."""

    # Build description with available skills (read at registration time)
    skills = list_skills(registry.data_dir)
    desc_parts = [
        "Reusable instruction files. Match task → load → follow instructions.",
        "  skill list                          — list available skills",
        "  skill load <name>                   — load full skill content into context",
        "  skill search <query>                — search skills by keyword",
        "  skill create <name> --desc <text>   — create skill (content via stdin)",
        "  skill update <name> [--desc <text>] — update skill (content via stdin)",
        "  skill delete <name>                 — delete a skill",
    ]
    if skills:
        desc_parts.append("\nAvailable:")
        for s in skills:
            desc_parts.append(f"  {s['name']:<22} {s['description']}")

    @registry.command("skill", "\n".join(desc_parts))
    def cmd_skill(args, stdin=""):
        data_dir = registry.data_dir  # read at call time

        if not args:
            raise ValueError("usage: skill list|load|search|create|update|delete")
        sub = args[0]

        if sub == "list":
            all_skills = list_skills(data_dir)
            if not all_skills:
                return "No skills. Use `skill create <name> --desc <text>` to add one."
            lines = [f"Skills ({len(all_skills)}):"]
            lines += [f"  {s['name']:<22} {s['description']}" for s in all_skills]
            return "\n".join(lines)

        if sub == "load":
            if len(args) < 2:
                raise ValueError("usage: skill load <name>")
            desc, body = load_skill(data_dir, args[1])
            header = f'<skill name="{args[1]}">\n'
            if desc:
                header += f"> {desc}\n\n"
            return header + body + "\n</skill>"

        if sub == "search":
            query = " ".join(args[1:]).lower()
            if not query:
                raise ValueError("usage: skill search <query>")
            matches = [
                s for s in list_skills(data_dir)
                if query in s["name"].lower() or query in s["description"].lower()
            ]
            if not matches:
                return f"No skills matching '{query}'."
            return "\n".join(f"  {s['name']} — {s['description']}" for s in matches)

        if sub == "create":
            if len(args) < 2:
                raise ValueError("usage: skill create <name> --desc <description>")
            name = args[1]
            desc = next(
                (args[i + 1] for i, a in enumerate(args[2:], 2) if a in ("--desc", "-d") and i + 1 < len(args)),
                ""
            )
            if not desc:
                raise ValueError("--desc is required")
            if not stdin:
                raise ValueError("skill content required via stdin")
            create_skill(data_dir, name, desc, stdin)
            return f"Skill '{name}' created. Use `skill load {name}` to verify."

        if sub == "update":
            if len(args) < 2:
                raise ValueError("usage: skill update <name> [--desc <text>]")
            name = args[1]
            new_desc = next(
                (args[i + 1] for i, a in enumerate(args[2:], 2) if a in ("--desc", "-d") and i + 1 < len(args)),
                None
            )
            update_skill(data_dir, name, new_desc, stdin or None)
            return f"Skill '{name}' updated."

        if sub == "delete":
            if len(args) < 2:
                raise ValueError("usage: skill delete <name>")
            delete_skill(data_dir, args[1])
            return f"Skill '{args[1]}' deleted."

        raise ValueError(f"unknown subcommand: skill {sub}")
