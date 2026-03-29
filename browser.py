"""
browser.py — bb-browser-backed web/search/browser commands.

Three commands registered:

  fetch <url>           Authenticated fetch via bb-browser in the real browser context.

  search <query>        Web search via bb-browser site adapter `google/search`.

  browser <action>      Browser automation via bb-browser CLI (real Chrome with your login state).
                        This is the default path for all browser and web-required work.
                        Requires: npm install -g bb-browser

bb-browser actions:
  open, snapshot, screenshot, click, hover, fill, type, check,
  uncheck, select, wait, press, scroll, eval, get, tab, site,
  fetch, network, console, errors, trace, frame, dialog,
  history, status, back, forward, refresh, close
"""

import re
import subprocess
import time
from pathlib import Path

URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
BB_BROWSER_DIR = Path.home() / ".bb-browser"
BB_SITES_GIT_DIR = BB_BROWSER_DIR / "bb-sites" / ".git"

# ── bb-browser CLI wrapper ────────────────────────────────────────────────────

def _bb(*args: str, timeout: int = 60) -> str:
    """
    Run a bb-browser CLI command and return stdout.
    Raises RuntimeError on failure or if bb-browser is not installed.
    """
    cmd = ["bb-browser"] + [str(a) for a in args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "bb-browser is not installed.\n"
            "Install it with:\n"
            "  npm install -g bb-browser"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"bb-browser command timed out after {timeout}s: {' '.join(cmd)}")

    output = result.stdout.strip()
    if result.returncode != 0:
        err = result.stderr.strip() or output
        raise RuntimeError(err)
    return output


def _bb_timeout(action: str, rest: list[str]) -> int:
    """Longer timeout for bb-browser commands that may open tabs or hit network."""
    if action == "site" and rest[:1] == ["update"]:
        return 180
    if action in {"site", "fetch", "network", "history", "trace"}:
        return 120
    return 60


def _normalize_url(url: str) -> str:
    """Add https:// when the caller passes a bare hostname."""
    if not url:
        return url
    if URL_SCHEME_RE.match(url) or url.startswith(("//", "#", "/")):
        return url
    return "https://" + url


def _ensure_site_adapters():
    """
    Install/update the community site adapter repo on first use so commands like
    `site google/search` are available without manual setup.
    """
    if BB_SITES_GIT_DIR.exists():
        return
    BB_BROWSER_DIR.mkdir(parents=True, exist_ok=True)
    _bb("site", "update", timeout=180)
    if not BB_SITES_GIT_DIR.exists():
        raise RuntimeError(
            "bb-browser site adapters are unavailable.\n"
            "Tried: bb-browser site update"
        )


def _bb_site(*args: str, timeout: int = 120) -> str:
    """Run a bb-browser site adapter, ensuring community adapters exist first."""
    _ensure_site_adapters()
    return _bb("site", *args, timeout=timeout)


def _save_screenshot(data_dir: str, topic_id: str) -> str:
    """
    Take a screenshot via bb-browser and save it to the agent's topic directory.
    Returns a file:// URL that loop.py detects and auto-attaches as vision content.
    """
    filename = f"screenshot-{int(time.time() * 1000)}.png"
    path = Path(data_dir) / "topics" / topic_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    _bb("screenshot", str(path))
    return f"Screenshot saved: {filename}\nView: ![screenshot](file://{path.resolve()})"


# ── Command registration ──────────────────────────────────────────────────────

def register_browser_commands(registry):
    """Register `fetch`, `search`, and `browser` commands."""

    @registry.command(
        "fetch",
        "Fetch through bb-browser in the real browser context. Usage: fetch <url>\n"
        "Uses the browser's cookies/login state. For site-specific extraction, prefer `browser site ...`.",
    )
    def cmd_fetch(args, stdin=""):
        url = (" ".join(args) or stdin).strip()
        if not url:
            raise ValueError("usage: fetch <url>")
        return _bb("fetch", _normalize_url(url), timeout=120)

    @registry.command(
        "search",
        "Search the web through bb-browser Google adapter. Usage: search <query>\n"
        "Equivalent to: bb-browser site google/search <query>",
    )
    def cmd_search(args, stdin=""):
        query = (" ".join(args) or stdin).strip()
        if not query:
            raise ValueError("usage: search <query>")
        return _bb_site("google/search", query, timeout=120)

    @registry.command(
        "browser",
        "Control a real Chrome browser (with your login state) via bb-browser.\n"
        "Use this for all web tasks. Prefer site adapters for search and structured extraction;\n"
        "use manual page interaction only when no adapter fits.\n"
        "  browser site <adapter> [args...]     — preferred: run a bb-browser site adapter\n"
        "  browser search <query>               — alias for browser site google/search <query>\n"
        "  browser open <url> [--tab ...]       — open a page when manual interaction is needed\n"
        "  browser snapshot [-i]                — accessibility tree snapshot with @refs\n"
        "  browser click|hover <ref>            — interact with an element\n"
        "  browser fill|type <ref> <text>       — input text\n"
        "  browser check|uncheck <ref>          — toggle a checkbox\n"
        "  browser select <ref> <value>         — choose a select option\n"
        "  browser wait <ms|@ref>               — wait for time or an element\n"
        "  browser fetch <url> [options]        — fetch with browser cookies/login state\n"
        "  browser network ...                  — inspect or mock network traffic\n"
        "  browser console | errors | trace ... — browser debugging tools\n"
        "  browser screenshot [path]            — take screenshot (auto-attaches if no path)\n"
        "  browser get url|title|text [ref]     — get page info or element text\n"
        "  browser eval <script>                — evaluate JavaScript\n"
        "  browser tab ... | tabs | tab-new     — manage tabs\n"
        "  browser frame ... | dialog ...       — advanced page handling\n"
        "  browser back|forward|refresh|close   — navigate / close current tab\n"
        "Snapshot refs: use @N (e.g. @3) from snapshot output to click/fill/get elements.\n"
        "Requires: npm install -g bb-browser",
    )
    def cmd_browser(args, stdin=""):
        if not args:
            raise ValueError("usage: browser <action> [args...]")

        action = args[0]
        rest = args[1:]
        timeout = _bb_timeout(action, rest)

        if action == "screenshot":
            if rest:
                return _bb("screenshot", *rest, timeout=timeout)
            return _save_screenshot(registry.data_dir, registry.topic_id)

        if action == "open":
            if not rest and not stdin.strip():
                raise ValueError("usage: browser open <url> [--tab ...]")
            url = _normalize_url(rest[0] if rest else stdin.strip())
            return _bb("open", url, *rest[1:], timeout=timeout)

        if action == "search":
            query = (" ".join(rest) or stdin).strip()
            if not query:
                raise ValueError("usage: browser search <query>")
            return _bb_site("google/search", query, timeout=timeout)

        if action == "site":
            if not rest:
                raise ValueError("usage: browser site <adapter|list|search|update|info> [...]")
            if rest[:1] == ["update"]:
                return _bb("site", *rest, timeout=timeout)
            return _bb_site(*rest, timeout=timeout)

        if action == "snapshot":
            return _bb("snapshot", *rest, timeout=timeout)

        if action == "get":
            if not rest and not stdin.strip():
                raise ValueError("usage: browser get url|title|text [ref]")
            if not rest:
                return _bb("get", stdin.strip(), timeout=timeout)
            return _bb("get", *rest, timeout=timeout)

        if action == "fill" and len(rest) == 1 and stdin.strip():
            return _bb("fill", rest[0], stdin, timeout=timeout)

        if action == "type" and len(rest) == 1 and stdin.strip():
            return _bb("type", rest[0], stdin, timeout=timeout)

        if action == "select" and len(rest) == 1 and stdin.strip():
            return _bb("select", rest[0], stdin, timeout=timeout)

        if action == "eval" and not rest:
            script = stdin.strip()
            if not script:
                raise ValueError("usage: browser eval <javascript>")
            return _bb("eval", script, timeout=timeout)

        if action == "fetch" and not rest:
            url = stdin.strip()
            if not url:
                raise ValueError("usage: browser fetch <url> [options]")
            return _bb("fetch", url, timeout=timeout)

        if action == "tabs":
            return _bb("tab", *rest, timeout=timeout)

        if action == "tab-new":
            if rest:
                return _bb("tab", "new", _normalize_url(rest[0]), *rest[1:], timeout=timeout)
            return _bb("tab", "new", timeout=timeout)

        if action == "tab-close":
            return _bb("tab", "close", *rest, timeout=timeout)

        return _bb(action, *rest, timeout=timeout)
