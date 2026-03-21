"""
browser.py — Web fetch, search, and browser automation commands.

Three commands registered:

  fetch <url>           Lightweight HTTP fetch — no JS, instant, no deps beyond stdlib.
                        Falls back to urllib if httpx/bs4 not installed.

  search <query>        Web search via DuckDuckGo — no API key needed.
                        Falls back to fetching DuckDuckGo HTML if duckduckgo_search missing.

  browser <action>      Full browser automation via Playwright (headless Chromium).
                        Required: pip install playwright && playwright install chromium

Optional dependencies:
  httpx              — better HTTP client (fallback: urllib.request)
  beautifulsoup4     — HTML to text parsing (fallback: regex stripping)
  duckduckgo-search  — structured search results (fallback: DuckDuckGo HTML)
  playwright         — required for `browser` command only
"""

import re
import time
from pathlib import Path


# ── Lightweight HTTP fetch ────────────────────────────────────────────────────

def _fetch_url(url: str) -> str:
    """Fetch a URL and return readable text. Falls back to urllib if httpx unavailable."""
    try:
        import httpx
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; agent/1.0)"})
        resp.raise_for_status()
        raw = resp.text
    except ImportError:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8", errors="replace")

    return _html_to_text(raw)


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", html)

    # Collapse excessive blank lines
    lines = [l.rstrip() for l in text.splitlines()]
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return result.strip()


# ── Web search ────────────────────────────────────────────────────────────────

def _search_web(query: str, limit: int = 8) -> str:
    """
    Search via DuckDuckGo.
    Primary: duckduckgo_search library (structured JSON results).
    Fallback: DuckDuckGo Lite (plain HTML, no JS required, no bot protection).
    """
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=limit)
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r['title']}**")
            lines.append(r["href"])
            if r.get("body"):
                lines.append(r["body"][:300])
            lines.append("")
        return "\n".join(lines).strip()
    except ImportError:
        pass

    # Fallback: DuckDuckGo Lite — minimal HTML, no bot protection, no JS needed
    import urllib.parse
    encoded = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    try:
        content = _fetch_url(url)
        # Lite page has clean text; trim to keep context manageable
        return content[:4000]
    except Exception as e:
        return f"Search unavailable: {e}\nInstall duckduckgo-search: pip install duckduckgo-search"


# ── Playwright browser (stateful singleton) ───────────────────────────────────

class _BrowserManager:
    """
    Lazily initializes Playwright (headless Chromium) and keeps one page alive.
    State persists across tool calls within a session — the LLM can open a page
    in one tool call and screenshot it in the next.
    Call browser.close() or `browser close` to tear down.
    """

    def __init__(self):
        self._pw = None
        self._browser = None
        self._page = None

    def _ensure(self):
        if self._page and not self._page.is_closed():
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError(
                "playwright is not installed.\n"
                "Install it with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
        if self._pw is None:
            self._pw = sync_playwright().start()
        if self._browser is None or not self._browser.is_connected():
            self._browser = self._pw.chromium.launch(headless=True)
        self._page = self._browser.new_page(
            viewport={"width": 1280, "height": 800}
        )

    @property
    def page(self):
        self._ensure()
        return self._page

    def close(self):
        if self._page and not self._page.is_closed():
            self._page.close()
            self._page = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._pw:
            self._pw.stop()
            self._pw = None


# Module-level singleton — persists browser across runs in the same Python process
_browser = _BrowserManager()


def _save_screenshot(page, data_dir: str, topic_id: str) -> str:
    """
    Take a full-page screenshot and save it to the agent's topic directory
    (not work_dir — screenshots are agent-generated, not user project files).
    Returns a file:// URL that loop.py detects and auto-attaches as vision content.
    """
    filename = f"screenshot-{int(time.time() * 1000)}.png"
    path = Path(data_dir) / "topics" / topic_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), full_page=True)
    return f"Screenshot saved: {filename}\nView: ![screenshot](file://{path.resolve()})"


def _normalize_selector(ref: str) -> str:
    """Accept CSS selectors or plain text (wrapped in :has-text() for Playwright)."""
    # If it looks like a CSS selector already, use it as-is
    if re.search(r'[#.\[\]>~+:@]', ref):
        return ref
    # Otherwise treat as visible text
    return f"text={ref}"


# ── Command registration ──────────────────────────────────────────────────────

def register_browser_commands(registry):
    """Register `fetch`, `search`, and `browser` commands. Uses registry at call time."""

    @registry.command(
        "fetch",
        "Fetch a web page as text (no JS execution). Usage: fetch <url>\n"
        "Faster than browser — use for static pages, docs, APIs.",
    )
    def cmd_fetch(args, stdin=""):
        url = (" ".join(args) or stdin).strip()
        if not url:
            raise ValueError("usage: fetch <url>")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        content = _fetch_url(url)
        return content[:8000]  # cap at 8k chars to keep context manageable

    @registry.command(
        "search",
        "Search the web (DuckDuckGo, no API key). Usage: search <query>\n"
        "Returns titles, URLs, and snippets for top results.",
    )
    def cmd_search(args, stdin=""):
        query = (" ".join(args) or stdin).strip()
        if not query:
            raise ValueError("usage: search <query>")
        return _search_web(query)

    @registry.command(
        "browser",
        "Control a real browser (headless Chromium via Playwright).\n"
        "  browser open <url>               — navigate to URL\n"
        "  browser snapshot                 — get page text content\n"
        "  browser screenshot               — take screenshot (auto-attaches as vision)\n"
        "  browser click <selector|text>    — click element\n"
        "  browser fill <selector> <text>   — clear and fill an input\n"
        "  browser type <selector> <text>   — type into an element\n"
        "  browser press <key>              — press key (Enter, Tab, Escape, ArrowDown...)\n"
        "  browser scroll up|down [pixels]  — scroll page (default 300px)\n"
        "  browser eval <script>            — evaluate JavaScript, returns result\n"
        "  browser get url|title|text       — get page info or element text\n"
        "  browser back|forward|refresh     — navigate history\n"
        "  browser close                    — close browser session\n"
        "Requires: pip install playwright && playwright install chromium",
    )
    def cmd_browser(args, stdin=""):
        if not args:
            raise ValueError("usage: browser <action> [args...]")

        action = args[0]
        rest   = args[1:]

        # ── Navigation ────────────────────────────────────────────────────────

        if action == "open":
            url = (" ".join(rest) or stdin).strip()
            if not url:
                raise ValueError("usage: browser open <url>")
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            page = _browser.page
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            return f"Opened: {page.title()}\nURL: {page.url}"

        if action == "back":
            _browser.page.go_back(wait_until="domcontentloaded")
            return f"Back → {_browser.page.url}"

        if action == "forward":
            _browser.page.go_forward(wait_until="domcontentloaded")
            return f"Forward → {_browser.page.url}"

        if action == "refresh":
            _browser.page.reload(wait_until="domcontentloaded")
            return f"Refreshed: {_browser.page.url}"

        # ── Content ───────────────────────────────────────────────────────────

        if action == "snapshot":
            page  = _browser.page
            text  = page.evaluate("document.body.innerText")
            return f"URL: {page.url}\nTitle: {page.title()}\n\n{text[:6000]}"

        if action == "screenshot":
            return _save_screenshot(_browser.page, registry.data_dir, registry.topic_id)

        if action == "get":
            attr = rest[0] if rest else "text"
            page = _browser.page
            if attr == "url":
                return page.url
            if attr == "title":
                return page.title()
            if attr == "text":
                if len(rest) > 1:
                    sel = _normalize_selector(" ".join(rest[1:]))
                    return page.locator(sel).first.inner_text(timeout=5_000)
                return page.evaluate("document.body.innerText")[:5000]
            raise ValueError(f"unknown attribute '{attr}' — use: url|title|text")

        # ── Interaction ───────────────────────────────────────────────────────

        if action == "click":
            if not rest:
                raise ValueError("usage: browser click <selector>")
            sel  = _normalize_selector(" ".join(rest))
            page = _browser.page
            try:
                page.locator(sel).first.click(timeout=5_000)
            except Exception:
                # Try clicking by visible text as fallback
                page.get_by_text(" ".join(rest), exact=False).first.click(timeout=5_000)
            return f"Clicked: {' '.join(rest)}"

        if action == "fill":
            if len(rest) < 2:
                raise ValueError("usage: browser fill <selector> <text>")
            sel  = _normalize_selector(rest[0])
            text = " ".join(rest[1:]) or stdin
            _browser.page.fill(sel, text, timeout=5_000)
            return f"Filled '{rest[0]}'"

        if action == "type":
            if len(rest) < 2:
                raise ValueError("usage: browser type <selector> <text>")
            sel  = _normalize_selector(rest[0])
            text = " ".join(rest[1:]) or stdin
            _browser.page.locator(sel).type(text, delay=30)
            return f"Typed into '{rest[0]}'"

        if action == "press":
            if not rest:
                raise ValueError("usage: browser press <key>")
            _browser.page.keyboard.press(rest[0])
            return f"Pressed: {rest[0]}"

        if action == "scroll":
            direction = rest[0] if rest else "down"
            pixels    = int(rest[1]) if len(rest) > 1 else 300
            page      = _browser.page
            if direction == "down":
                page.evaluate(f"window.scrollBy(0, {pixels})")
            elif direction == "up":
                page.evaluate(f"window.scrollBy(0, -{pixels})")
            elif direction == "right":
                page.evaluate(f"window.scrollBy({pixels}, 0)")
            elif direction == "left":
                page.evaluate(f"window.scrollBy(-{pixels}, 0)")
            else:
                raise ValueError(f"unknown direction '{direction}' — use: up|down|left|right")
            return f"Scrolled {direction} {pixels}px"

        if action == "eval":
            script = " ".join(rest) or stdin
            if not script:
                raise ValueError("usage: browser eval <javascript>")
            result = _browser.page.evaluate(script)
            return str(result) if result is not None else "OK"

        if action == "close":
            _browser.close()
            return "Browser closed."

        raise ValueError(
            f"unknown browser action: '{action}'\n"
            "Use: open|snapshot|screenshot|click|fill|type|press|scroll|eval|get|back|forward|refresh|close"
        )
