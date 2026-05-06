"""
tools/browser_tool.py — Browser-use / Playwright automation for Shifu
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gives Shifu a real browser: click, type, screenshot, scrape, fill forms,
navigate SPAs — anything a human can do in Chrome.

Uses the `browser-use` library which wraps Playwright with a
LangChain-compatible interface and built-in DOM extraction.

Install:
    pip install browser-use playwright
    playwright install chromium

Two tools are exposed:
  • browser_task  — high-level: give it a goal in plain English, it drives
                    the browser autonomously to completion (uses an internal
                    LLM loop from browser-use).
  • browser_screenshot — low-level: navigate to a URL and return a
                         base64 screenshot for vision tasks.

All downloads land in Playground/browser_downloads/.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import subprocess
import asyncio
import base64
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
PLAYGROUND_DIR     = Path("Playground")
DOWNLOADS_DIR      = PLAYGROUND_DIR / "browser_downloads"
SCREENSHOTS_DIR    = PLAYGROUND_DIR / "browser_screenshots"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── browser-use imports (lazy so missing install gives a clean error) ─────────
def _require_browser_use():
    try:
        from browser_use import Agent as BrowserAgent
        from browser_use.browser.browser import Browser, BrowserConfig
        return BrowserAgent, Browser, BrowserConfig
    except ImportError:
        raise ImportError(
            "browser-use is not installed.\n"
            "Run:  pip install browser-use playwright && playwright install chromium"
        )

def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is not installed.\n"
            "Run:  pip install playwright && playwright install chromium"
        )


# ── Internal LLM for browser-use agent loop ───────────────────────────────────
# browser-use drives its own internal ReAct loop; it needs its own LLM handle.
# We reuse the same model/base_url as shifu.py.

def _make_browser_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("BROWSER_MODEL", "gpt-oss:120b-cloud"),
        base_url=os.getenv("BASE_URL", "https://ollama.com/v1"),
        api_key=os.getenv("OLLAMA_API_KEY_EXECUTOR"),
        temperature=0.1,
        max_tokens=2048,
    )


# ── Async helper — runs the browser-use agent loop ───────────────────────────

async def _run_browser_task(task: str, max_steps: int = 20) -> str:
    BrowserAgent, Browser, BrowserConfig = _require_browser_use()

    browser = Browser(config=BrowserConfig(
        headless=True,
        downloads_path=str(DOWNLOADS_DIR.resolve()),
    ))

    agent = BrowserAgent(
        task=task,
        llm=_make_browser_llm(),
        browser=browser,
        max_steps=max_steps,
        save_conversation_path=str(PLAYGROUND_DIR / "browser_logs"),
    )

    result = await agent.run()
    await browser.close()

    # browser-use returns an AgentHistoryList; extract final text
    if hasattr(result, "final_result"):
        return result.final_result() or "Task completed (no textual result)."
    return str(result)


# ── Tool 1: browser_task ──────────────────────────────────────────────────────

@tool
def browser_task(task: str, max_steps: int = 20) -> str:
    """
    Autonomously complete a browser-based task described in plain English.

    The agent drives a real Chromium browser, navigating pages, clicking
    buttons, filling forms, and extracting information until the task is done.

    Use this for:
    - Scraping pages that require JavaScript to render.
    - Logging into a site and extracting data (provide credentials in task).
    - Filling and submitting web forms.
    - Multi-step browser workflows ("go to X, search for Y, download the CSV").
    - Any task where web_search is not enough and you need to operate the page.

    Args:
        task:      Plain-English description of what to do in the browser.
                   Include the URL if known. Be specific about what to return.
        max_steps: Maximum number of browser actions before giving up (default 20).

    Returns:
        A text summary of the result, extracted content, or confirmation of action.

    Examples:
        browser_task("Go to https://finance.yahoo.com and get the current NVDA stock price")
        browser_task("Go to https://example.com/login, log in with user='foo' password='bar', then download the invoice PDF from the account page")
        browser_task("Search for 'LangGraph tutorial' on YouTube and return the titles and URLs of the top 5 results")
    """
    try:
        return asyncio.run(_run_browser_task(task, max_steps=max_steps))
    except Exception as e:
        return f"browser_task failed: {type(e).__name__}: {e}"


# ── Tool 2: browser_screenshot ────────────────────────────────────────────────

@tool
def browser_screenshot(url: str, filename: Optional[str] = None) -> str:
    """
    Navigate to a URL and capture a full-page screenshot.

    Saves the screenshot as a PNG to Playground/browser_screenshots/ and
    returns the file path. Use this when you need to visually inspect a page,
    capture evidence, or feed the screenshot to a vision model.

    Args:
        url:      The full URL to navigate to (must include https://).
        filename: Optional filename for the PNG (without extension).
                  Defaults to a sanitised version of the URL.

    Returns:
        Absolute path to the saved PNG file.

    Examples:
        browser_screenshot("https://anthropic.com")
        browser_screenshot("https://example.com/dashboard", filename="dashboard_capture")
    """
    sync_playwright = _require_playwright()

    if not filename:
        safe = url.replace("https://", "").replace("http://", "")
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe)
        filename = safe[:80]

    out_path = SCREENSHOTS_DIR / f"{filename}.png"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(url, wait_until="networkidle", timeout=30_000)
            page.screenshot(path=str(out_path), full_page=True)
            browser.close()
        return str(out_path.resolve())
    except Exception as e:
        return f"browser_screenshot failed: {type(e).__name__}: {e}"


# ── Tool 3: browser_extract_text ─────────────────────────────────────────────

@tool
def browser_extract_text(url: str) -> str:
    """
    Navigate to a URL and extract all visible text from the rendered page.

    Unlike web_search (which uses snippets) or a raw HTTP fetch (which includes
    raw HTML), this renders the full page in a real browser — including
    JavaScript-rendered content — and returns only the visible text.

    Use this for:
    - Reading articles behind JS rendering.
    - Extracting clean text from SPAs or React/Vue pages.
    - Getting the actual page content when web_search gives you a truncated snippet.

    Args:
        url: The full URL to visit (must include https://).

    Returns:
        Visible page text, stripped of HTML tags and scripts. Truncated to
        12,000 characters if the page is very long.
    """
    sync_playwright = _require_playwright()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30_000)
            # Remove script/style nodes, then grab innerText
            text = page.evaluate("""() => {
                document.querySelectorAll('script, style, noscript, svg').forEach(el => el.remove());
                return document.body ? document.body.innerText : '';
            }""")
            browser.close()
        text = text.strip()
        if len(text) > 12_000:
            text = text[:12_000] + "\n\n[... truncated — use browser_task to extract specific sections ...]"
        return text or "(page returned no visible text)"
    except Exception as e:
        return f"browser_extract_text failed: {type(e).__name__}: {e}"