"""
tools/browser_agent.py  —  Shifu Browser Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full browser-use replacement powered by Playwright + an agentic
LLM loop (gemma4:31b-cloud via Ollama).

Capabilities
────────────
• JS-rendered page scraping
• Login flows & session-aware browsing
• Form filling & submission
• Multi-step browser workflows
• File / CSV downloads
• Screenshot capture
• Media playback  (YouTube, Spotify, etc.)  — browser stays open!

Media / playback tasks are auto-detected.  The agent runs non-headless
and can issue  keep_open  instead of  done  to leave the browser window
alive indefinitely after the task completes.  The tool returns
immediately while the browser keeps playing in the background.

All screenshots  →  Playground/browser/screenshots/
All downloads    →  Playground/browser/downloads/

Drop this file into your  tools/  package — it is auto-loaded by
Shifu's load_all_tools_from_package() scanner.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
_BROWSER_ROOT   = Path("Playground/browser")
SCREENSHOTS_DIR = _BROWSER_ROOT / "screenshots"
DOWNLOADS_DIR   = _BROWSER_ROOT / "downloads"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Browser-LLM config ────────────────────────────────────────────────────────
_BROWSER_MODEL   = "gemma4:31b-cloud"
_BROWSER_API_KEY = os.getenv("OLLAMA_API_KEY_BROWSER", "")
_BASE_URL        = "https://ollama.com/v1"
_MAX_STEPS       = 30          # hard cap on agent loop iterations
_VIEWPORT        = {"width": 1280, "height": 800}

# ── Media-task auto-detection ─────────────────────────────────────────────────
# Tasks that involve playing/watching/listening should keep the browser alive.
_MEDIA_KEYWORDS = re.compile(
    r"\b(play|watch|listen|music|song|video|stream|youtube|spotify|netflix"
    r"|soundcloud|podcast|audio|movie|show|episode|playlist|album|track)\b",
    re.IGNORECASE,
)

def _is_media_task(task: str) -> bool:
    return bool(_MEDIA_KEYWORDS.search(task))

# ── Lazy imports (only pay the cost when the tool is actually called) ──────────
def _get_playwright_sync():
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
        return sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "Playwright is not installed. Run:  pip install playwright && playwright install chromium"
        ) from e


def _get_openai():
    try:
        from openai import OpenAI  # noqa: PLC0415
        return OpenAI
    except ImportError as e:
        raise RuntimeError("openai package is required: pip install openai") from e


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT for the browser-agent LLM
# ─────────────────────────────────────────────────────────────────────────────
_AGENT_SYSTEM = """You are a precise browser-automation agent.
You control a real Chromium browser through a JSON action protocol.

## CURRENT STATE (updated each step)
You will receive:
  - url:          current page URL
  - title:        page <title>
  - screenshot:   base-64 PNG of the viewport (use it to read content / find elements)
  - dom_snapshot: simplified text DOM (tag | id | name/placeholder/text excerpt)
  - step:         current iteration number

## RESPONSE FORMAT  (strict JSON, no prose, no markdown fences)
{
  "thought": "one-sentence reasoning about what to do next",
  "action":  "<action_name>",
  "params":  { ... }
}

## ACTIONS
navigate        params: { url }
click           params: { selector }          — CSS selector
type            params: { selector, text }    — clears field, then types
press           params: { key }               — keyboard key, e.g. "Enter"
scroll          params: { direction, amount } — direction: "up"|"down", amount: pixels
wait            params: { ms }               — wait N milliseconds (max 5000)
select          params: { selector, value }  — <select> element
check           params: { selector }         — checkbox / radio
upload          params: { selector, path }   — file path to upload
screenshot      params: { filename }         — saves to Playground/browser/screenshots/
extract         params: { description }      — extract & return text content from page
download_check  params: {}                   — check & list files in downloads dir
keep_open       params: { result }           — task done BUT keep browser alive (use for media playback, streaming, anything the user should keep watching/listening to)
done            params: { result }           — finish and CLOSE the browser; result is the final answer string

## RULES
1. Prefer CSS selectors: #id, [name="x"], button:has-text("Login"), input[type="email"]
2. After navigation or click always wait at least one step before further interaction.
3. If a CAPTCHA is detected, report it in done.result.
4. Never guess URLs — navigate to known URLs or follow links you can see.
5. For credentials supplied in the task, use them exactly as given.
6. If you cannot complete the task after several attempts, explain why in done.result.
7. Keep "thought" concise — one sentence maximum.
8. Your ONLY output must be the JSON object — nothing else.
9. IMPORTANT: For any media playback task (play a song, watch a video, stream music, etc.)
   always use  keep_open  instead of  done  so the browser window stays alive.
   Never close the browser while media is playing."""


# ─────────────────────────────────────────────────────────────────────────────
#  DOM snapshot helper
# ─────────────────────────────────────────────────────────────────────────────
_DOM_JS = """
() => {
  const interesting = ['a','button','input','textarea','select','form',
                        'h1','h2','h3','p','li','td','th','label','span'];
  const rows = [];
  const seen = new Set();
  document.querySelectorAll(interesting.join(',')).forEach(el => {
    const tag  = el.tagName.toLowerCase();
    const id   = el.id   ? '#' + el.id   : '';
    const name = el.getAttribute('name') || el.getAttribute('placeholder') || '';
    const txt  = (el.innerText || el.value || '').trim().slice(0, 80);
    const key  = tag + id + name + txt;
    if (seen.has(key) || !txt && !name && !id) return;
    seen.add(key);
    rows.push(`${tag}${id} | ${name} | ${txt}`);
  });
  return rows.slice(0, 120).join('\\n');
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Core agent loop (synchronous, wraps Playwright's sync API)
# ─────────────────────────────────────────────────────────────────────────────
class _BrowserAgent:
    def __init__(self, task: str, headless: bool = True, keep_open: bool = False):
        self.task      = task
        self.headless  = headless
        self.keep_open = keep_open   # if True, don't close browser on done/keep_open
        self._OpenAI   = _get_openai()
        self._llm      = self._OpenAI(
            api_key  = _BROWSER_API_KEY,
            base_url = _BASE_URL,
        )
        self._history: list[dict] = []   # message history for the LLM

    # ── screenshot ─────────────────────────────────────────────────────────
    def _take_screenshot(self, page, filename: str | None = None) -> str:
        ts   = int(time.time())
        name = filename or f"step_{ts}.png"
        if not name.endswith(".png"):
            name += ".png"
        path = SCREENSHOTS_DIR / name
        page.screenshot(path=str(path), full_page=False)
        return str(path)

    # ── encode screenshot for vision ───────────────────────────────────────
    def _screenshot_b64(self, page) -> str:
        raw = page.screenshot(full_page=False)
        return base64.b64encode(raw).decode()

    # ── dom snapshot ───────────────────────────────────────────────────────
    def _dom_snapshot(self, page) -> str:
        try:
            return page.evaluate(_DOM_JS)
        except Exception:
            return "(dom snapshot unavailable)"

    # ── call LLM ───────────────────────────────────────────────────────────
    def _call_llm(self, state_msg: str, screenshot_b64: str) -> dict:
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url":    f"data:image/png;base64,{screenshot_b64}",
                    "detail": "high",
                },
            },
            {"type": "text", "text": state_msg},
        ]
        self._history.append({"role": "user", "content": user_content})

        messages = [{"role": "system", "content": _AGENT_SYSTEM}] + self._history

        resp = self._llm.chat.completions.create(
            model      = _BROWSER_MODEL,
            messages   = messages,
            max_tokens = 512,
            temperature= 0.1,
        )
        raw = resp.choices[0].message.content.strip()

        # strip markdown fences if model wraps in them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract first {...} block
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else {
                "thought": "parse error",
                "action":  "done",
                "params":  {"result": f"LLM returned non-JSON: {raw[:300]}"},
            }

        self._history.append({"role": "assistant", "content": json.dumps(parsed)})
        return parsed

    # ── execute one action ─────────────────────────────────────────────────
    def _execute(self, page, action: dict) -> str | None:
        """Returns a string only for 'done' or 'extract'. None otherwise."""
        name   = action.get("action", "done")
        params = action.get("params", {})

        try:
            if name == "navigate":
                page.goto(params["url"], wait_until="domcontentloaded", timeout=20_000)

            elif name == "click":
                page.click(params["selector"], timeout=8_000)

            elif name == "type":
                page.fill(params["selector"], "")                # clear first
                page.type(params["selector"], params["text"], delay=40)

            elif name == "press":
                page.keyboard.press(params["key"])

            elif name == "scroll":
                direction = params.get("direction", "down")
                amount    = int(params.get("amount", 400))
                delta     = amount if direction == "down" else -amount
                page.evaluate(f"window.scrollBy(0, {delta})")

            elif name == "wait":
                ms = min(int(params.get("ms", 1000)), 5000)
                page.wait_for_timeout(ms)

            elif name == "select":
                page.select_option(params["selector"], params["value"])

            elif name == "check":
                page.check(params["selector"])

            elif name == "upload":
                page.set_input_files(params["selector"], params["path"])

            elif name == "screenshot":
                saved = self._take_screenshot(page, params.get("filename"))
                return f"[screenshot saved → {saved}]"

            elif name == "extract":
                content = page.evaluate("() => document.body.innerText")
                snippet = content.strip()[:4000]
                return f"[extracted]\n{snippet}"

            elif name == "download_check":
                files = list(DOWNLOADS_DIR.iterdir())
                return (
                    f"[downloads dir: {DOWNLOADS_DIR}]\n"
                    + "\n".join(str(f) for f in files)
                    if files else f"[downloads dir empty: {DOWNLOADS_DIR}]"
                )

            elif name == "keep_open":
                # Signal: task is done, but do NOT close the browser.
                return f"[keep_open] {params.get('result', 'Task complete. Browser staying open.')}"

            elif name == "done":
                return params.get("result", "Task complete.")

        except Exception as exc:
            return f"[action error: {name} — {exc}]"

        return None

    # ── main run ───────────────────────────────────────────────────────────
    def run(self) -> str:
        import threading

        sync_playwright = _get_playwright_sync()

        # We need the playwright context alive even after this method returns
        # (for keep_open / media tasks).  We run the agent loop in a closure
        # and, if keep_open fires, we leave the browser open and block a
        # background daemon thread — the tool call returns immediately.

        _result_box: list[str]  = []
        _done_event              = threading.Event()
        _keep_open_event         = threading.Event()

        def _agent_thread():
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    headless       = self.headless,
                    downloads_path = str(DOWNLOADS_DIR),
                    args           = ["--no-sandbox", "--disable-dev-shm-usage"],
                )
                context = browser.new_context(
                    viewport         = _VIEWPORT,
                    accept_downloads = True,
                    user_agent       = (
                        "Mozilla/5.0 (X11; Linux x86_64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                )
                page = context.new_page()

                page.on("download", lambda dl: dl.save_as(
                    str(DOWNLOADS_DIR / dl.suggested_filename)
                ))

                # seed conversation
                self._history.append({
                    "role":    "user",
                    "content": f"TASK: {self.task}",
                })
                self._history.append({
                    "role":    "assistant",
                    "content": json.dumps({
                        "thought": "Starting task. Will navigate first.",
                        "action":  "wait",
                        "params":  {"ms": 100},
                    }),
                })

                result = "(no result)"
                browser_should_stay_open = False

                for step in range(_MAX_STEPS):
                    try:
                        current_url   = page.url
                        current_title = page.title()
                    except Exception:
                        current_url   = "about:blank"
                        current_title = ""

                    screenshot_b64 = self._screenshot_b64(page)
                    dom            = self._dom_snapshot(page)

                    state_msg = textwrap.dedent(f"""
                        step: {step + 1} / {_MAX_STEPS}
                        url:  {current_url}
                        title:{current_title}
                        dom_snapshot:
                        {dom}
                    """).strip()

                    action  = self._call_llm(state_msg, screenshot_b64)
                    a_name  = action.get("action", "done")
                    a_params = action.get("params", {})

                    print(
                        f"  [browser-agent step {step+1:02d}] "
                        f"{action.get('thought','')[:80]}  "
                        f"→ {a_name}"
                    )

                    outcome = self._execute(page, action)

                    # ── terminal actions ───────────────────────────────────
                    if a_name == "keep_open":
                        result = (
                            a_params.get("result", "Task complete.")
                            + "\n\n🎵 Browser is staying open — enjoy!"
                        )
                        browser_should_stay_open = True
                        break

                    if a_name == "done":
                        result = outcome or a_params.get("result", "Done.")
                        break

                    # ── non-terminal: feed observation back ────────────────
                    if outcome and outcome.startswith("["):
                        self._history.append({
                            "role":    "user",
                            "content": f"[observation]: {outcome}",
                        })

                else:
                    result = (
                        f"Browser agent hit the {_MAX_STEPS}-step limit. "
                        "Task may be incomplete. Last page: " + page.url
                    )

                # ── signal the calling thread that we have a result ────────
                _result_box.append(result)
                _done_event.set()

                if browser_should_stay_open:
                    # Block here (in the background thread) until the user
                    # closes the browser window naturally.
                    print(
                        "  [browser-agent] Browser kept open for media playback. "
                        "Close the window to free resources."
                    )
                    try:
                        # Wait until the page/browser is closed by the user
                        page.wait_for_event("close", timeout=0)  # 0 = no timeout
                    except Exception:
                        pass
                    try:
                        context.close()
                        browser.close()
                    except Exception:
                        pass
                else:
                    context.close()
                    browser.close()

        t = threading.Thread(target=_agent_thread, daemon=True)
        t.start()

        # Wait until the agent has a result (keep_open or done),
        # then return — the thread keeps the browser alive if needed.
        _done_event.wait(timeout=300)   # 5-min hard ceiling
        return _result_box[0] if _result_box else "Browser agent timed out."


# ─────────────────────────────────────────────────────────────────────────────
#  The Shifu-compatible LangChain tool
# ─────────────────────────────────────────────────────────────────────────────
@tool
def browser_agent(task: str) -> str:
    """
    AI-powered browser agent (Playwright + LLM loop).

    Use this tool for ANY task that requires real browser interaction:
      - Scraping JS-rendered pages that web_search cannot access
      - Logging into websites and extracting authenticated data
      - Filling and submitting web forms
      - Multi-step browser workflows ("go to X, search for Y, download the CSV")
      - Clicking through paginated results or dynamic UIs
      - Capturing screenshots of specific pages
      - Playing music / videos / streams (YouTube, Spotify, etc.)
        → the browser window stays open after the tool call returns!

    Args:
        task: Plain-English description of what to do in the browser.
              Include credentials if login is required, e.g.:
              "Go to https://example.com, log in with user@test.com / pass123,
               then download the sales report CSV from the Reports tab."
              For media: "Open YouTube and play Baby by Justin Bieber"

    Returns:
        A string summary of what was accomplished, extracted content,
        or file paths of saved screenshots/downloads.
        For media tasks the browser stays open in the background.

    Screenshots → Playground/browser/screenshots/
    Downloads   → Playground/browser/downloads/
    """
    media = _is_media_task(task)
    # Media tasks: visible browser so the user can actually see/hear it.
    # Non-media tasks: headless is fine (faster, no GUI needed).
    agent = _BrowserAgent(task=task, headless=not media, keep_open=media)
    return agent.run()