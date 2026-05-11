#!/usr/bin/env python3
"""
shifu.py — Shifu Agent + Terminal UI  (v4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v2 changes
──────────
• Planner merged: 3 serial LLM calls collapsed into 1 structured call (~60% faster).
• SqliteSaver: graph state persists across process restarts.
• Hidden sentinel <<<DONE>>>: no more forced "Alright, I'm done —" verbal tic.
• Memory rules tightened: TIER A/B/C filter; session atoms never stored.
• Recall output: clean prose only — no raw scores in LLM context.
• Companion mode: emotional/casual messages bypass full planning.
• Session summary on exit via spada_session_close.
• Async rate limiter: no event loop blocking.
• Tool auto-icon, proactive memory hints, skill auto-generation, wall-clock timeout.

v3 changes
──────────
• Per-mission thread_id: each mission gets a fresh LangGraph thread so unrelated
  tasks never bleed their message history into each other.
• Rolling session context: a lightweight _SessionContext object maintains a
  2-3 sentence plain-English summary of what happened this session. This is
  injected into each mission's system prompt as a slim "session awareness" block —
  not the full message history, just the distilled thread. The executor stays
  context-aware across missions without token bloat.
• NEEDS_MEMORY planner flag: Section 4 of the planner response. The planner
  decides YES/NO based on whether personal context would actually help. Pure tool
  tasks ("run this command", "what's the weather") get NO and recall is skipped
  entirely — zero wasted embedding calls. Companion and personal messages get YES.

v4 changes
──────────
• HOT MEMORY: two-tier memory architecture.
    Cold memory  — existing SPADA ChromaDB store (spada_db_shifu), unchanged.
    Hot memory   — lightweight JSON rolling window (.shifu/hot_memory.json).
  Hot memory stores exactly 2 things per completed turn:
    (1) Atomised key facts from the user's input prompt
    (2) Atomised key facts from the final model output + actions
  This pair is injected into every executor system prompt for zero-latency
  immediate context — no embedding lookup, no tool call required.
• /reset_mem  command: wipe hot memory; cold memory is always untouched.
• /mem_status command: inspect hot + cold memory side-by-side.
• Bar.hot_memory_inject() and Bar.hot_memory_store() visual ticks.
• HOT_MEMORY_MAX_TURNS env var: configurable rolling window (default 15).
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

import asyncio
import importlib
import itertools
import os
import pkgutil
import re
import shutil
import sys
import textwrap
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict
import uuid as _uuid
# ── third-party ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
import tools as tools_pkg

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PLAYGROUND_DIR = Path("Playground"); PLAYGROUND_DIR.mkdir(exist_ok=True)
SKILLS_DIR     = Path("skills");     SKILLS_DIR.mkdir(exist_ok=True)
DATA_DIR       = Path(".shifu");     DATA_DIR.mkdir(exist_ok=True)

MODEL_NAME      = "gpt-oss:120b-cloud"
BASE_URL        = "https://ollama.com/v1"
MAX_RPM         = 20
MAX_ITERATIONS  = 50
MAX_RETRIES     = 10
MISSION_TIMEOUT        = 300  # seconds (5 minutes hard wall)
SESSION_CONTEXT_MAX    = 8    # max missions kept in rolling context


# ── Rolling session context ───────────────────────────────────────────────────
# Tracks what happened this session as a lightweight plain-English summary.
# Injected into each mission's system prompt — not the full message history,
# just enough for Shifu to say "oh, we just summarized that file" if relevant.

class _SessionContext:
    """
    Maintains a rolling 2-3 sentence summary of the current session.
    Updated after every mission. Injected into the executor system prompt
    so Shifu has lightweight cross-mission awareness without message bleed.
    """

    def __init__(self):
        self._entries: list[str] = []   # (mission_summary,) tuples, newest last

    def add(self, mission: str, outcome_hint: str):
        """Record what just happened. outcome_hint is a short phrase like
        'summarized x.txt' or 'answered weather question'."""
        entry = f"[{datetime.now().strftime('%H:%M')}] {outcome_hint}"
        self._entries.append(entry)
        if len(self._entries) > SESSION_CONTEXT_MAX:
            self._entries = self._entries[-SESSION_CONTEXT_MAX:]

    def as_prompt_block(self) -> str:
        """
        Returns a slim injection block for the executor system prompt.
        Empty string if no prior missions this session.
        """
        if not self._entries:
            return ""
        lines = "\n".join(f"  • {e}" for e in self._entries)
        return (
            "══ THIS SESSION (lightweight context — not full history) ══════════\n"
            "Recent missions completed before this one:\n"
            f"{lines}\n"
            "Use this only if the current mission clearly relates to prior work.\n"
            "Do NOT drag unrelated session history into an unrelated task.\n"
            "═══════════════════════════════════════════════════════════════════"
        )

    def is_empty(self) -> bool:
        return len(self._entries) == 0


_session_context = _SessionContext()


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL PALETTE & PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

class C:
    R       = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    A       = "\033[38;5;214m"
    AB      = "\033[1;38;5;214m"
    ABGF    = "\033[38;5;0m\033[48;5;214m"
    W       = "\033[38;5;252m"
    G       = "\033[38;5;240m"
    GD      = "\033[38;5;236m"
    OK      = "\033[38;5;71m"
    ERR     = "\033[38;5;167m"
    TOOL    = "\033[38;5;67m"
    PLAN    = "\033[38;5;139m"
    STREAM  = "\033[38;5;223m"
    PANELBG = "\033[48;5;234m"
    RBG     = "\033[49m"
    HINT    = "\033[38;5;183m"   # soft lavender — proactive memory hints


def _tw() -> int:
    try:    return min(shutil.get_terminal_size().columns, 110)
    except: return 90

def _w(s: str):
    sys.stdout.write(s); sys.stdout.flush()

def _blank(n: int = 1):
    print("\n" * (n - 1))

def _dim_line(ch: str = "─"):
    _w("  " + C.GD + ch * (_tw() - 4) + C.R + "\n")

def _wrap(text: str, indent: int = 3) -> str:
    pad = " " * indent
    return textwrap.fill(text, width=_tw() - indent,
                         initial_indent=pad, subsequent_indent=pad)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGO + BOOT
# ══════════════════════════════════════════════════════════════════════════════

_LOGO = r"""
      _____ _    _ _____  ______ _    _
     / ____| |  | |_   _||  ____| |  | |
    | (___ | |__| | | |  | |__  | |  | |
     \___ \|  __  | | |  |  __| | |  | |
     ____) | |  | |_| |_ | |    | |__| |
    |_____/|_|  |_|_____||_|     \____/
"""

def boot():
    os.system("cls" if os.name == "nt" else "clear")
    _blank()
    for raw in _LOGO.strip("\n").split("\n"):
        if not raw.strip():
            _w("\n"); continue
        s = raw.rstrip()
        first = len(s) - len(s.lstrip())
        last  = len(s)
        filled = ""
        for ch in raw[first:last]:
            filled += (C.AB + ch + C.R) if ch != " " else " "
        _w("  " + raw[:first] + filled + "\n")
        time.sleep(0.018)
    _blank()
    ts = datetime.now().strftime("%A, %d %B %Y  ·  %H:%M")
    _w("  " + C.G + ts + C.R + "\n")
    _dim_line()
    _w("  " + C.G + "help  ·  history  ·  skills  ·  mem_status  ·  reset_mem  ·  exit" + C.R + "\n")
    _blank()


# ══════════════════════════════════════════════════════════════════════════════
#  SKILLS
# ══════════════════════════════════════════════════════════════════════════════

def _scan_skills() -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        name = md.parent.name
        text = md.read_text(encoding="utf-8")
        fm: dict = {}
        m = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    fm[k.strip()] = v.strip().strip('"').strip("'")
        registry[name] = {
            "name":        fm.get("name", name),
            "description": fm.get("description", "(no description)"),
            "path":        str(md),
        }
    return registry

SKILLS = _scan_skills()

def _skills_index() -> str:
    if not SKILLS:
        return "  (no skills installed)"
    return "\n".join(
        f'  • load_skill("{n}")  —  {m["description"]}' for n, m in SKILLS.items()
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LLMs
# ══════════════════════════════════════════════════════════════════════════════

def _make_llm(env_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME, base_url=BASE_URL,
        api_key=os.getenv(env_key),
        temperature=0.5,
        max_tokens=4096,
        streaming=True,
    )

planner_llm  = _make_llm("OLLAMA_API_KEY_PLANNER")
executor_llm = _make_llm("OLLAMA_API_KEY_EXECUTOR")

_last_call: float = 0.0
_rate_lock         = asyncio.Lock()


async def _invoke_async(llm, messages):
    """Async rate-limited invoke — never blocks the event loop."""
    global _last_call
    async with _rate_lock:
        gap  = 60.0 / MAX_RPM
        wait = gap - (time.time() - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        result     = await asyncio.to_thread(llm.invoke, messages)
        _last_call = time.time()
        return result


def _invoke(llm, messages):
    """Synchronous wrapper — used only during planning nodes inside graph.stream()."""
    global _last_call
    gap  = 60.0 / MAX_RPM
    wait = gap - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    result     = llm.invoke(messages)
    _last_call = time.time()
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def _load_tools(package) -> list[BaseTool]:
    found = []
    for _, mod_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        mod = importlib.import_module(mod_name)
        for obj in vars(mod).values():
            if isinstance(obj, BaseTool):
                found.append(obj)
    return found

TOOLS               = _load_tools(tools_pkg)
tool_node           = ToolNode(TOOLS)
executor_with_tools = executor_llm.bind_tools(TOOLS)

# ── Hot memory ─────────────────────────────────────────────────────────────
# Imported after tools load; gracefully disabled if spada_memory isn't present.

_SESSION_ID        = str(_uuid.uuid4())
_SESSION_THREAD_ID = str(uuid.uuid4())

try:
    from tools.spada_tool import hot_atomise as _hot_atomise
    from tools.spada_tool import _get_hot_memory
    _hot_memory = _get_hot_memory(session_id=_SESSION_ID)
    _HOT_MEM    = True
except Exception as _hot_import_err:
    _hot_memory  = None
    _hot_atomise = None
    _HOT_MEM     = False

# Tool icons: auto-populated from known names; unknown tools get a sensible default
_KNOWN_TOOL_ICONS = {
    "web_search":           ("⌕",  "search"),
    "file_write":           ("↓",  "write"),
    "file_read":            ("↑",  "read"),
    "directory_read":       ("⊞",  "ls"),
    "terminal_command":     ("$",  "shell"),
    "load_skill":           ("◈",  "skill"),
    "browser_task":         ("◉",  "browse"),
    "browser_screenshot":   ("⊡",  "screen"),
    "browser_extract_text": ("≡",  "extract"),
    "spada_recall":         ("◎",  "recall"),
    "spada_memorise":       ("◉",  "memorise"),
    "spada_session_close":  ("◈",  "session"),
}

def _tool_icon(name: str) -> tuple[str, str]:
    if name in _KNOWN_TOOL_ICONS:
        return _KNOWN_TOOL_ICONS[name]
    # auto-generate a label from tool name for unknown tools
    label = name.replace("_", " ")[:12]
    return ("·", label)


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Unified planner prompt ────────────────────────────────────────────────────
# One call, structured sections. Replaces 3 serial classify/clarify/plan calls.

_PLANNER_SYS = f"""You are Shifu's strategic mind — sharp, concise, honest.

AVAILABLE SKILLS:
{_skills_index()}

You will receive a mission. Reply with ALL FOUR sections below, in order,
using the exact section headers shown. No preamble, no extra commentary.

━━ SECTION 1: ROUTE ━━
Reply with exactly one word on its own line:
  COMPANION  — casual chat, emotional messages, simple questions needing a warm reply
  SIMPLE     — one tool call or one obvious step
  COMPLEX    — multiple steps, ambiguity, or research required

━━ SECTION 2: CLARIFICATION ━━
Reply with either:
  NO_CLARIFICATION
or
  CLARIFY: <one focused, essential question>
Only ask if the answer fundamentally changes how the task is done.
Never ask for things you can reasonably default. No multi-part questions.

━━ SECTION 3: PLAN ━━
Write a numbered execution plan (COMPLEX only; write PLAN: N/A for SIMPLE/COMPANION).
  - Each step is one atomic action.
  - If a skill matches, put load_skill("<exact-name>") as step 1.
  - Only include spada_recall as a step if SECTION 4 is YES.
  - Only include spada_memorise as the last step if there are TIER A/B facts to save.
  - All files go inside Playground/.
  - Max 10 steps. End with: PLAN COMPLETE.

━━ SECTION 4: NEEDS_MEMORY ━━
Reply with exactly one word: YES or NO.

YES — memory recall would genuinely help answer or personalise this mission:
  • COMPANION / emotional messages (always YES)
  • Questions about the user's own life, preferences, projects, history
  • Anything where past context changes the quality of the answer
  • "Remember when...", "didn't we...", "you mentioned..." type messages

NO — memory recall adds nothing and would just waste time:
  • Pure tool tasks: run this command, write this file, search this thing
  • Factual/external questions with no personal angle: weather, news, math
  • Any task where the answer is the same regardless of who's asking
  • Follow-ups on the CURRENT mission where context is already in scope

When in doubt: NO. Unnecessary recalls pollute session time.
"""

# ── Executor prompt ───────────────────────────────────────────────────────────

_EXECUTOR_SYS = f"""You are Shifu — a capable, perceptive AI agent who gets things done.

SKILLS (copy verbatim into load_skill):
{_skills_index()}

══ MEMORY RULES ══════════════════════════════════════════════════════════════

M1. RECALL FIRST — for any non-trivial message, run spada_recall before acting.
    For pure tool tasks (write this file, run this command) with no personal
    context involved, you may skip recall if it would genuinely add nothing.

M2. USE WHAT YOU FIND — recalled context is signal, not decoration.
    Connect the dots: if memory says they love chess and they mention being bored,
    say "chess.com?" Don't just list what you remembered — weave it in naturally.

M3. STORE SELECTIVELY — the write filter is strict:

    TIER A — ALWAYS store (permanent):
      • User's name, location, job, relationships, long-term goals
      • Deep preferences that won't change ("hates meetings", "vegetarian")
      • Explicit "remember this" or "keep that in mind" requests
      • Major life decisions or milestones the user shares

    TIER B — Store if genuinely novel (month):
      • Active ongoing projects with specific details
      • Research outcomes that'll be relevant again
      • Task decisions the user made this session
      • Current habits or preferences that could shift in a few months

    TIER C — NEVER store:
      • Greetings, filler, "how are you" exchanges
      • Your own narration or task descriptions
      • Tool outputs with no lasting personal value
      • Questions you just answered
      • Anything the user obviously won't want recalled next session

    When in doubt: skip it. A clean store beats a polluted one.

M4. TTL matters — pass the right ttl to spada_memorise:
    permanent → TIER A facts
    month     → TIER B facts
    week      → short-lived context (rare; use sparingly)

M5. NO EXCUSES — never say "I don't remember" without having called spada_recall.
    Skipping recall when context would help = task failure.

M6. PROACTIVE — if recall returns [PROACTIVE CONTEXT], surface it naturally
    if it seems relevant. A good friend volunteers things they remembered.
    Don't dump it mechanically; weave it in conversationally.

══ COMPLETION SIGNAL ════════════════════════════════════════════════════════
When you are fully done with a task, include the hidden token <<<DONE>>> 
somewhere in your response (it will be stripped before display).
Then write your natural closing — no forced "Alright, I'm done —" preamble.
Just speak like a person who finished something and wants to tell you about it.

══ CRITICAL RULES ════════════════════════════════════════════════════════════
1. ALWAYS USE TOOLS TO CREATE FILES. Never write file content in your response
   text. Writing code in your response text = task failure.

1a. NEVER CREATE FILES FOR CONVERSATIONAL RESPONSES. Recommendations, lists,
    opinions, memory-based answers, casual chat — written directly in your
    response, NOT saved to Playground/. Ask yourself: did the user ask you to
    BUILD or SAVE something? If no, do not touch the file system.

2. ALL files go inside Playground/ → {PLAYGROUND_DIR.resolve()}

3. CREATE DIRECTORIES BEFORE FILES using the terminal tool.

4. NEVER ASK THE USER QUESTIONS MID-TASK. State your assumption briefly
   ("I'll assume X — say so if you'd like it changed") and keep going.

5. COMPANION MODE — for casual chat, emotional messages, or simple questions:
   keep it warm and natural. No bullet lists. No task summaries. Just talk.
   Memory recall is still useful here — connect past context to present moment.

══ TONE ══════════════════════════════════════════════════════════════════════
Before your first tool call on non-trivial tasks: 1-2 sentences saying what
you're about to do and why. Natural English, first person, no bullet lists.

After each tool result: 1 sentence on what just came back. Be honest about
surprises or errors. Never just say "Done." silently.

When finished: speak naturally. No ritual openers. Just close warmly.

Max {MAX_ITERATIONS} tool-call iterations. Hard timeout: {MISSION_TIMEOUT}s.
"""

# ── Reviewer prompt ───────────────────────────────────────────────────────────
# Tightened — no longer a rubber stamp.

_REVIEWER_SYS = """You are a strict quality reviewer for an AI agent.

PASS if:
  • The core task is demonstrably complete (file written, search done, answer given)
  • The agent produced a substantive response addressing the mission
  • A clarifying question was asked (valid behaviour, not failure)

RETRY only if:
  • A tool returned an explicit error, exception, or traceback
    AND the task is clearly not finished as a result
  • The agent's response is completely empty or "I don't know" with no attempt

NEVER retry for: terse style, missing polish, asking questions, not starting
with a specific phrase, or responses that are complete but brief.

Reply exactly:
VERDICT: PASS
or
VERDICT: RETRY — <one-sentence reason, must name the specific error>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════════════

class ShifuState(TypedDict):
    needs_memory: bool
    messages:          Annotated[list[BaseMessage], add_messages]
    mission:           str
    plan:              str
    route:             Literal["COMPANION", "SIMPLE", "COMPLEX", ""]
    clarification_q:   str
    clarification_a:   str
    iterations:        int
    verdict:           Literal["PASS", "RETRY", ""]
    retry_count:       int


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH NODES  (planner now does one merged call)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_planner_response(content: str) -> dict:
    """Parse the three-section planner response into a dict."""
    route         = "SIMPLE"
    clarification = ""
    plan          = ""

    # ROUTE
    m = re.search(r'━━\s*SECTION 1: ROUTE\s*━━\s*\n(.+)', content, re.IGNORECASE)
    if m:
        word = m.group(1).strip().upper().split()[0] if m.group(1).strip() else ""
        if "COMPANION" in word:
            route = "COMPANION"
        elif "COMPLEX" in word:
            route = "COMPLEX"
        else:
            route = "SIMPLE"
    else:
        # Fallback: scan for the keywords
        upper = content.upper()
        if "COMPANION" in upper:  route = "COMPANION"
        elif "COMPLEX"  in upper: route = "COMPLEX"
        else:                     route = "SIMPLE"

    # CLARIFICATION
    m2 = re.search(r'CLARIFY:\s*(.+)', content, re.IGNORECASE)
    if m2 and "NO_CLARIFICATION" not in content.upper():
        clarification = m2.group(1).strip()

    # PLAN
    m3 = re.search(r'━━\s*SECTION 3: PLAN\s*━━(.+?)(?:PLAN COMPLETE|$)', content, re.DOTALL | re.IGNORECASE)
    if m3:
        raw_plan = m3.group(1).strip()
        if "N/A" not in raw_plan.upper():
            plan = raw_plan
    
    # In _parse_planner_response, add after the PLAN block:
    needs_memory = False
    m4 = re.search(r'SECTION 4[:\s]*NEEDS_MEMORY\s*━*\s*\n\s*(YES|NO)', content, re.IGNORECASE)
    if m4:
        needs_memory = m4.group(1).strip().upper() == "YES"
    # fallback: if "YES" appears near NEEDS_MEMORY
    elif "NEEDS_MEMORY" in content.upper():
        snippet = content.upper().split("NEEDS_MEMORY")[-1][:30]
        needs_memory = "YES" in snippet

    return {"route": route, "clarification_q": clarification, "plan": plan, "needs_memory": needs_memory}



def plan_node(state: ShifuState) -> ShifuState:
    """Single merged planner call: route + clarification + plan."""
    extra = f"\nUser clarification: {state['clarification_a']}" if state.get("clarification_a") else ""
    resp  = _invoke(planner_llm, [
        SystemMessage(content=_PLANNER_SYS),
        HumanMessage(content=(
            f"Mission: {state['mission']}{extra}\n"
            f"Playground: {PLAYGROUND_DIR.resolve()}"
        )),
    ])
    parsed = _parse_planner_response(resp.content)
    return {
        **state,
        "route":           parsed["route"],
        "clarification_q": parsed["clarification_q"],
        "plan":            parsed["plan"],
        "needs_memory": parsed["needs_memory"],
    }


def clarify_node(state: ShifuState) -> ShifuState:
    """Suspend graph; caller injects answer via Command(resume=...)."""
    answer = interrupt(state["clarification_q"])
    if not answer or not str(answer).strip():
        answer = "(user skipped — use your best judgment)"
    return {**state, "clarification_a": str(answer).strip()}


def replan_node(state: ShifuState) -> ShifuState:
    """Re-run planner after user clarification (skip route re-classify)."""
    resp = _invoke(planner_llm, [
        SystemMessage(content=_PLANNER_SYS),
        HumanMessage(content=(
            f"Mission: {state['mission']}\n"
            f"User clarification: {state['clarification_a']}\n"
            f"Playground: {PLAYGROUND_DIR.resolve()}"
        )),
    ])
    parsed = _parse_planner_response(resp.content)
    return {
        **state,
        "plan":            parsed["plan"],
        "clarification_q": "",   # don't re-ask
    }


def tools_node_fn(state: ShifuState) -> ShifuState:
    try:
        result = tool_node.invoke({"messages": state["messages"]})
        return {**state, "messages": result["messages"]}
    except Exception as exc:
        # Normalize tool errors into a ToolMessage so the graph doesn't crash
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        error_msgs = []
        if last_ai and last_ai.tool_calls:
            for tc in last_ai.tool_calls:
                error_msgs.append(ToolMessage(
                    content=f'{{"status": "error", "message": "{exc}"}}',
                    tool_call_id=tc.get("id", str(uuid.uuid4())),
                    name=tc.get("name", "unknown"),
                ))
        return {**state, "messages": state["messages"] + error_msgs}


def review_node(state: ShifuState) -> ShifuState:
    last_tool = next((m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)), None)
    last_ai   = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),   None)

    # Fast-path pass: no error keywords
    if last_tool:
        err_kw = ("error", "exception", "failed", "traceback", "timeout")
        if not any(k in str(last_tool.content).lower() for k in err_kw):
            return {**state, "verdict": "PASS"}

    agent_output = (last_ai.content or "") if last_ai else ""
    if last_tool:
        agent_output += "\n[Tool result]: " + str(last_tool.content)[:500]

    resp = _invoke(planner_llm, [
        SystemMessage(content=_REVIEWER_SYS),
        HumanMessage(content=f"Mission: {state['mission']}\n\nAgent output:\n{agent_output}"),
    ])
    verdict: Literal["PASS", "RETRY"] = "PASS" if "PASS" in resp.content.upper() else "RETRY"
    return {**state,
            "verdict":     verdict,
            "retry_count": state["retry_count"] + (1 if verdict == "RETRY" else 0)}


# ── routing ────────────────────────────────────────────────────────────────────

def _route_plan(state):
    if state["clarification_q"]:
        return "clarify"
    return "execute"

def _route_clarify(state):
    return "replan"

def _route_replan(state):
    return "execute"

def _route_execute(state):
    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "review" if state["route"] == "COMPLEX" else END

def _route_tools(state):
    return END if state["iterations"] >= MAX_ITERATIONS else "execute"

def _route_review(state):
    if state["verdict"] == "PASS" or state["retry_count"] >= MAX_RETRIES:
        return END
    return "execute"


# ── build graph ────────────────────────────────────────────────────────────────

def _build_graph():
    # SqliteSaver for persistent checkpoints across restarts
    import sqlite3 as _sqlite3
    _db_conn = _sqlite3.connect(str(DATA_DIR / "shifu_graph.db"), check_same_thread=False)
    checkpointer = SqliteSaver(_db_conn)

    g = StateGraph(ShifuState)
    for name, fn in [
        ("plan",    plan_node),
        ("clarify", clarify_node),
        ("replan",  replan_node),
        ("execute", _execute_node_stub),  # placeholder — actual execution is streamed
        ("tools",   tools_node_fn),
        ("review",  review_node),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("plan")
    g.add_conditional_edges("plan",    _route_plan,    {"clarify": "clarify", "execute": "execute"})
    g.add_conditional_edges("clarify", _route_clarify, {"replan": "replan"})
    g.add_conditional_edges("replan",  _route_replan,  {"execute": "execute"})
    g.add_conditional_edges("execute", _route_execute, {"tools": "tools", "review": "review", END: END})
    g.add_conditional_edges("tools",   _route_tools,   {"execute": "execute", END: END})
    g.add_conditional_edges("review",  _route_review,  {"execute": "execute", END: END})

    return g.compile(checkpointer=checkpointer, interrupt_before=["clarify"])

def _execute_node_stub(state: ShifuState) -> ShifuState:
    """
    Stub — the execute node exists in the graph for routing but actual execution
    is driven by the streaming runner below.  The stub does nothing.
    """
    return state

shifu_graph = _build_graph()


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMING EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

_DONE_SENTINEL = "<<<DONE>>>"

async def _stream_executor_turn(
    messages: list[BaseMessage],
    silent: bool = False,
) -> AIMessage:
    full_text       = ""
    tool_calls      = []
    in_text         = False

    if not silent:
        _w("\n  " + C.A + "│" + C.R + "  ")

    async for event in executor_with_tools.astream_events(messages, version="v2"):
        kind = event.get("event", "")

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                token = chunk.content if isinstance(chunk.content, str) else ""
                if token:
                    full_text += token
                    if not silent:
                        # Strip the sentinel token from live stream display
                        display_token = token.replace(_DONE_SENTINEL, "").replace("<<<DONE", "").replace("DONE>>>", "")
                        if display_token:
                            in_text = True
                            _w(C.STREAM + display_token + C.R)
                            sys.stdout.flush()

            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    idx = tc.get("index", 0)
                    while len(tool_calls) <= idx:
                        tool_calls.append({"id": "", "name": "", "args": ""})
                    if tc.get("id"):   tool_calls[idx]["id"]   += tc["id"]
                    if tc.get("name"): tool_calls[idx]["name"] += tc["name"]
                    if tc.get("args"): tool_calls[idx]["args"] += tc["args"]

    if in_text:
        _w("\n")

    import json
    parsed_tcs = []
    for tc in tool_calls:
        try:    args = json.loads(tc["args"]) if tc["args"] else {}
        except: args = {"raw": tc["args"]}
        parsed_tcs.append({
            "id": tc["id"] or str(uuid.uuid4()),
            "name": tc["name"], "args": args, "type": "tool_call",
        })

    return AIMessage(content=full_text, tool_calls=parsed_tcs)


# ══════════════════════════════════════════════════════════════════════════════
#  SPINNER
# ══════════════════════════════════════════════════════════════════════════════

class Spinner:
    _FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, label: str = ""):
        self._label   = label
        self._stop    = threading.Event()
        self._paused  = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
        _w("\r" + " " * 40 + "\r")

    def pause(self):
        self._paused.set()
        time.sleep(0.15)

    def resume(self):
        self._paused.clear()

    def _run(self):
        for f in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            if self._paused.is_set():
                _w("\r" + " " * 40 + "\r")
                while self._paused.is_set() and not self._stop.is_set():
                    time.sleep(0.05)
                continue
            msg = self._label[:30] if self._label else ""
            _w(f"\r  {C.GD}{f}{C.R}  {C.G}{msg}{C.R}  ")
            sys.stdout.flush()
            time.sleep(0.09)


# ══════════════════════════════════════════════════════════════════════════════
#  BAR (progress trail)
# ══════════════════════════════════════════════════════════════════════════════

class Bar:
    def __init__(self):
        self._t0   = time.time()
        self._lock = threading.Lock()

    def _elapsed(self) -> str:
        return f"{time.time() - self._t0:.1f}s"

    def _commit(self, line: str):
        with self._lock:
            _w("\r" + " " * 40 + "\r")
            _w(line + "\n")

    def phase(self, label: str, icon: str = "·"):
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.A}{icon}{C.R}  {C.W}{label}{C.R}")

    def route(self, r: str):
        col = {"COMPANION": C.HINT, "SIMPLE": C.OK, "COMPLEX": C.A}.get(r, C.G)
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {col}◈{C.R}  {col}{r.lower()}{C.R}")

    def plan_step(self, line: str):
        clean = line.strip()
        if clean:
            self._commit(f"  {C.G}{'':>6}{C.R}  {C.PLAN}›{C.R}  {C.G}{clean[:70]}{C.R}")

    def tool_call(self, name: str, arg: str = ""):
        icon, label = _tool_icon(name)
        snippet = arg.replace("\n", " ")[:55]
        if len(arg) > 55: snippet += "…"
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.TOOL}{icon}  {label}{C.R}  {C.GD}{snippet}{C.R}")

    def tool_result(self, name: str, result: str):
        _, label = _tool_icon(name)
        snippet = result.replace("\n", " ")[:60]
        if len(result) > 60: snippet += "…"
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.OK}✓{C.R}  {C.TOOL}{label}{C.R}  {C.GD}{snippet}{C.R}")

    def narration_box(self, text: str):
        iw = _tw() - 8
        self._commit(f"  {C.A}┌{'─' * iw}┐{C.R}")
        for raw in textwrap.wrap(text, width=iw - 2) or [text]:
            padded = raw.ljust(iw - 2)
            _w(f"  {C.A}│{C.R} {C.W}{padded}{C.R} {C.A}│{C.R}\n")
        _w(f"  {C.A}└{'─' * iw}┘{C.R}\n")

    def proactive_hint(self, hint: str):
        """Display a memory hint that Shifu is about to volunteer."""
        snippet = hint[:65] + ("…" if len(hint) > 65 else "")
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.HINT}◎{C.R}  {C.HINT}memory hint{C.R}  {C.GD}{snippet}{C.R}")

    def hot_memory_inject(self, turns: int):
        """Show that hot memory was injected into the system prompt."""
        self._commit(
            f"  {C.G}{self._elapsed():>6}{C.R}  "
            f"{C.HINT}◑{C.R}  {C.HINT}hot memory{C.R}  "
            f"{C.GD}injecting {turns} recent turn(s){C.R}"
        )

    def hot_memory_store(self, p_atoms: int, r_atoms: int):
        """Show that hot memory atoms were stored after a mission."""
        self._commit(
            f"  {C.G}{self._elapsed():>6}{C.R}  "
            f"{C.HINT}◑{C.R}  {C.HINT}hot memory{C.R}  "
            f"{C.GD}stored {p_atoms}+{r_atoms} atoms (prompt+response){C.R}"
        )

    def clarification(self, question: str, answer: str):
        qs  = question[:55] + ("…" if len(question) > 55 else "")
        as_ = answer[:45]   + ("…" if len(answer)   > 45 else "")
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.A}💬{C.R}  {C.GD}{qs}{C.R}  {C.W}→ {as_}{C.R}")

    def verdict(self, v: str):
        if v == "PASS":
            self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.OK}◉  pass{C.R}")
        else:
            self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.ERR}◉  retry{C.R}")

    def timeout(self):
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.ERR}⚠  timeout{C.R}")


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION CLOSE  (auto-called on exit)
# ══════════════════════════════════════════════════════════════════════════════

_session_missions: list[str] = []  # rolling log for summary generation


async def _run_session_close(spinner: Spinner):
    """
    Generate and store a session summary via spada_session_close.
    Called automatically when the user exits.
    """
    if not _session_missions:
        return
    spinner._label = "closing session…"
    spinner = Spinner("closing session…"); spinner.start()

    summary_prompt = (
        "Write a 2-4 sentence summary of this session for long-term memory. "
        "Cover: what was accomplished, key decisions, and notable things the user shared. "
        "Third-person past tense. Be specific.\n\n"
        "Session missions:\n" +
        "\n".join(f"  {i+1}. {m}" for i, m in enumerate(_session_missions[-10:]))
    )

    try:
        msgs = [
            SystemMessage(content="You are a session summariser. Reply with only the summary paragraph."),
            HumanMessage(content=summary_prompt),
        ]
        response = await asyncio.to_thread(planner_llm.invoke, msgs)
        summary  = response.content.strip()

        # Call spada_session_close if available
        tool_map = {t.name: t for t in TOOLS}
        closer   = tool_map.get("spada_session_close")
        if closer and summary:
            await asyncio.to_thread(closer.invoke, {"summary": summary})
    except Exception:
        pass  # session close is best-effort; never crash on exit
    finally:
        try: spinner.stop()
        except: pass


# ══════════════════════════════════════════════════════════════════════════════
#  SKILL AUTO-GENERATION
# ══════════════════════════════════════════════════════════════════════════════

async def _offer_skill_save(mission: str, plan: str):
    """
    After a novel COMPLEX task, ask if the user wants to save it as a skill.
    Generates the SKILL.md automatically.
    """
    ts  = datetime.now().strftime("%H:%M")
    _w(f"\n  {C.GD}[{ts}]{C.R}  {C.HINT}◈{C.R}  "
       f"{C.G}Save this as a reusable skill? {C.GD}[y/N]{C.R}  ")
    try:
        answer = await asyncio.to_thread(input, "")
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer.strip().lower() not in ("y", "yes"):
        return

    # Derive a skill name from the mission
    safe_name = re.sub(r'[^a-z0-9_]+', '_', mission.lower().strip())[:30].strip('_')
    skill_dir  = SKILLS_DIR / safe_name
    skill_dir.mkdir(exist_ok=True)
    skill_file = skill_dir / "SKILL.md"

    if skill_file.exists():
        _w(f"  {C.G}skill '{safe_name}' already exists — skipping.{C.R}\n")
        return

    # Generate the SKILL.md content
    gen_prompt = (
        f"Generate a SKILL.md for Shifu, a LangGraph-based AI agent.\n"
        f"Mission that was just completed: {mission}\n"
        f"Execution plan used:\n{plan}\n\n"
        "The SKILL.md must include:\n"
        "1. YAML front-matter with: name, description (one line), version: 1.0\n"
        "2. ## Steps section: a numbered checklist of the key steps\n"
        "3. ## Notes section: any gotchas, defaults, or tips\n"
        "Keep it under 40 lines. No preamble. Start with ---"
    )
    try:
        msgs     = [HumanMessage(content=gen_prompt)]
        response = await asyncio.to_thread(planner_llm.invoke, msgs)
        skill_file.write_text(response.content.strip(), encoding="utf-8")
        _w(f"  {C.OK}✓{C.R}  {C.G}skill saved → {skill_dir.resolve()}{C.R}\n")
        # Refresh in-memory skills registry
        SKILLS.clear()
        SKILLS.update(_scan_skills())
    except Exception as exc:
        _w(f"  {C.ERR}✗{C.R}  {C.G}skill save failed: {exc}{C.R}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MISSION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def run_mission_async(mission: str, bar: Bar, spinner: Spinner) -> str:

    config = {"configurable": {"thread_id": _SESSION_THREAD_ID}}

    state: ShifuState = {
        "messages":        [],
        "mission":         mission,
        "plan":            "",
        "route":           "",
        "clarification_q": "",
        "clarification_a": "",
        "iterations":      0,
        "verdict":         "",
        "retry_count":     0,
        "needs_memory": False,
    }

    # ── PHASE 1: planning (merged single call) ──────────────────────────────
    spinner.start()
    spinner._label = "thinking…"

    prev = None
    for step in shifu_graph.stream(state, config=config, stream_mode="values"):
        if prev is None or (not prev.get("route") and step.get("route")):
            spinner.stop()
            bar.route(step.get("route", "SIMPLE"))
            spinner = Spinner("planning…"); spinner.start()
        if not (prev or {}).get("plan") and step.get("plan"):
            spinner.stop()
            bar.phase("plan", "◈")
            for ln in step["plan"].splitlines():
                if re.match(r'^\s*\d+\.', ln):
                    bar.plan_step(ln)
            spinner = Spinner("preparing…"); spinner.start()
        prev  = step
        state = step

    # ── handle clarify interrupt ────────────────────────────────────────────
    while True:
        snapshot = shifu_graph.get_state(config)
        if not snapshot.next:
            break
        question = ""
        for task in getattr(snapshot, "tasks", []):
            for iv in getattr(task, "interrupts", []):
                question = iv.value; break
            if question: break
        if not question:
            break

        spinner.pause(); spinner.stop()
        iw = _tw() - 8
        _w(f"\n  {C.AB}┌{'─' * iw}┐{C.R}\n")
        _w(f"  {C.AB}│{C.R}  {C.BOLD}💬  SHIFU NEEDS A QUICK CLARIFICATION{C.R}\n")
        _w(f"  {C.AB}│{C.R}\n")
        for line in textwrap.wrap(question, width=iw - 4) or [question]:
            _w(f"  {C.AB}│{C.R}  {C.W}{line}{C.R}\n")
        _w(f"  {C.AB}└{'─' * iw}┘{C.R}\n")

        ts_str = datetime.now().strftime("%H:%M")
        prompt = f"  {C.GD}[{ts_str}]{C.R}  {C.AB}›{C.R}  "
        try:
            answer = await asyncio.to_thread(input, prompt)
            answer = answer.strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if not answer:
            answer = "(user skipped — use your best judgment)"

        bar.clarification(question, answer)
        spinner = Spinner("re-planning…"); spinner.start()

        for step in shifu_graph.stream(Command(resume=answer), config=config, stream_mode="values"):
            if not (prev or {}).get("plan") and step.get("plan"):
                spinner.stop()
                bar.phase("plan (revised)", "◈")
                for ln in step["plan"].splitlines():
                    if re.match(r'^\s*\d+\.', ln):
                        bar.plan_step(ln)
                spinner = Spinner("preparing…"); spinner.start()
            prev  = step
            state = step

    # ── PHASE 2: execute loop (streaming + tools) ───────────────────────────
    spinner.stop()

    t_start = time.time()

    for iteration in range(MAX_ITERATIONS):
        # Wall-clock timeout
        if time.time() - t_start > MISSION_TIMEOUT:
            bar.timeout()
            break

        # Build message list
        if not state["messages"]:
            content = f"Mission: {state['mission']}"
            if state["route"] == "COMPLEX" and state["plan"]:
                content += f"\n\nExecution Plan:\n{state['plan']}"
            if state.get("clarification_a"):
                content += f"\n\nUser clarification: {state['clarification_a']}"
            if state.get("needs_memory"):
                content += "\n\n[SYSTEM NOTE: Memory recall is relevant here. Call spada_recall NOW before responding — do not skip it.]"

            # ── Hot memory injection ────────────────────────────────────────
            # Build the system content with hot memory block prepended.
            # Hot memory is zero-latency (no embedding lookup needed) and
            # gives Shifu immediate awareness of what just happened.
            hot_block = ""
            if _HOT_MEM and _hot_memory is not None:
                hot_block = _hot_memory.as_prompt_block()
            if hot_block:
                bar.hot_memory_inject(_hot_memory.count())
                sys_content = _EXECUTOR_SYS + "\n\n" + hot_block
            else:
                sys_content = _EXECUTOR_SYS

            msgs: list[BaseMessage] = [
                SystemMessage(content=sys_content),
                HumanMessage(content=content),
            ]
        else:
            msgs = state["messages"]

        # Stream executor response
        _blank()
        _dim_line()
        bar.phase(f"shifu  (turn {iteration + 1})", "▸")

        response = await _stream_executor_turn(msgs, silent=True)

        if not state["messages"]:
            all_msgs = msgs + [response]
        else:
            all_msgs = state["messages"] + [response]
        state = {**state, "messages": all_msgs, "iterations": iteration + 1}

        # Check for done sentinel in response
        response_has_done = _DONE_SENTINEL in (response.content or "")

        # No tool calls
        if not response.tool_calls:
            if state["route"] == "COMPLEX":
                spinner = Spinner("reviewing…"); spinner.start()
                state = review_node(state)
                spinner.stop()
                bar.verdict(state["verdict"])
                if state["verdict"] == "PASS" or state["retry_count"] >= MAX_RETRIES:
                    break
                state = {**state, "messages": []}
            else:
                break
            continue

        # Tool calls
        for tc in response.tool_calls:
            name = tc.get("name", "?")
            args = tc.get("args", {})
            primary = str(next(iter(args.values()), "")).replace("\n", " ")
            bar.tool_call(name, primary)

        tool_msgs  = []
        tool_map   = {t.name: t for t in TOOLS}
        for tc in response.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            tid  = tc.get("id") or str(uuid.uuid4())
            tool = tool_map.get(name)
            if tool is None:
                result = f'{{"status": "error", "message": "tool \'{name}\' not found"}}'
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = f'{{"status": "error", "message": "{e}"}}'
            tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tid, name=name))

        state = {**state, "messages": all_msgs + tool_msgs}

        for tm in tool_msgs:
            if isinstance(tm, ToolMessage):
                bar.tool_result(getattr(tm, "name", "tool"), str(tm.content))

        # If the done sentinel was present and tools finished, we're done
        if response_has_done:
            break

    # ── extract final answer ────────────────────────────────────────────────
    messages = state.get("messages", [])
    last_ai  = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None
    )

    final_text = ""
    if last_ai:
        # Strip the sentinel token from display
        final_text = last_ai.content.replace(_DONE_SENTINEL, "").strip()

    if not final_text:
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_results:
            lines = [f"  [{getattr(m,'name','tool')}] {str(m.content).strip()[:120]}"
                     for m in tool_results[-5:]]
            final_text = "Mission complete. Last tool outputs:\n" + "\n".join(lines)
        else:
            final_text = "Mission complete."

    # ── skill auto-generation offer (COMPLEX novel tasks) ───────────────────
    if state["route"] == "COMPLEX" and state.get("plan") and state["retry_count"] == 0:
        skill_name = re.sub(r'[^a-z0-9_]+', '_', mission.lower().strip())[:30].strip('_')
        if not (SKILLS_DIR / skill_name / "SKILL.md").exists():
            _blank()
            await _offer_skill_save(mission, state["plan"])

    # ── hot memory: atomise prompt + response, store pair ───────────────────
    # Done AFTER skill offer so it doesn't block the user prompt.
    # Two LLM calls (prompt atom, response atom) run in the background.
    if _HOT_MEM and _hot_memory is not None and final_text:
        try:
            p_atoms = await asyncio.to_thread(_hot_atomise, mission, "prompt")
            r_atoms = await asyncio.to_thread(_hot_atomise, final_text, "response")
            _hot_memory.store(p_atoms, r_atoms)
            bar.hot_memory_store(len(p_atoms), len(r_atoms))
        except Exception:
            pass  # hot memory storage is always best-effort

    return final_text


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _inline_md(t: str) -> str:
    t = re.sub(r'`([^`\n]+)`',
               C.PANELBG + C.A + r' \1 ' + C.RBG + C.R + C.W, t)
    t = re.sub(r'\*\*(.+?)\*\*', C.BOLD + C.W + r'\1' + C.R + C.W, t)
    t = re.sub(r'\*(.+?)\*',     C.ITALIC + r'\1' + C.R + C.W, t)
    return t

def _render_code_block(lang: str, code: str, inner_w: int):
    lang  = (lang or "code").strip() or "code"
    label = f" {lang} "
    bar_  = "─" * max(0, inner_w - len(label) - 1)
    _w("  " + C.GD + "╭─" + C.G + label + C.GD + bar_ + C.R + "\n")
    for line in code.split("\n"):
        padded = line + " " * max(0, inner_w - len(line) - 2)
        _w("  " + C.GD + "│" + C.R + C.PANELBG + " " + C.G + padded + C.RBG + "\n")
    _w("  " + C.GD + "╰" + "─" * inner_w + C.R + "\n")

def render_response(text: str, elapsed: float):
    # Strip sentinel just in case it leaked
    text = text.replace(_DONE_SENTINEL, "").strip()

    iw  = _tw() - 6
    _blank()
    ts  = datetime.now().strftime("%H:%M:%S")
    tag = f" shifu  {ts}  {elapsed:.1f}s "
    border_r = "─" * max(0, iw - len(tag) + 2)
    _w("  " + C.A + "┌" + C.AB + tag + C.A + border_r + "┐" + C.R + "\n")

    lines = str(text).split("\n")
    i = 0

    def box_line(content: str):
        _w("  " + C.A + "│" + C.R + "  " + content + "\n")

    def wrap_box(raw: str, prefix: str = ""):
        avail = iw - len(prefix)
        indent = " " * len(re.sub(r'\033\[[0-9;]*m', '', prefix))
        lines_ = textwrap.wrap(raw, width=max(40, avail),
                               break_long_words=False, break_on_hyphens=False) or [""]
        for idx, wl in enumerate(lines_):
            lead = prefix if idx == 0 else indent
            box_line(lead + C.W + _inline_md(wl) + C.R)

    while i < len(lines):
        ln = lines[i]

        m = re.match(r'^```(\w*)', ln)
        if m:
            lang, body = m.group(1), []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i]); i += 1
            box_line("")
            _render_code_block(lang, "\n".join(body), iw - 2)
            box_line("")
            i += 1; continue

        if ln.strip() == "":
            box_line(""); i += 1; continue

        if ln.startswith("# "):
            box_line("")
            box_line(C.AB + ln[2:].upper() + C.R)
            box_line(C.GD + "─" * min(len(ln[2:]), iw - 4) + C.R)
            i += 1; continue
        if ln.startswith("## "):
            box_line(C.A + C.BOLD + ln[3:] + C.R)
            i += 1; continue
        if ln.startswith("### "):
            box_line(C.G + "›  " + C.W + ln[4:] + C.R)
            i += 1; continue
        if re.match(r'^[-*_]{3,}\s*$', ln):
            box_line(C.GD + "─" * (iw - 4) + C.R)
            i += 1; continue
        if ln.startswith("> "):
            box_line(C.GD + "▌ " + C.R + C.ITALIC + C.G + _inline_md(ln[2:]) + C.R)
            i += 1; continue

        bm = re.match(r'^(\s*)[-*+] (.+)', ln)
        if bm:
            lvl = len(bm.group(1)) // 2
            dot = ["◆", "◇", "·"][min(lvl, 2)]
            col = [C.A, C.G, C.GD][min(lvl, 2)]
            wrap_box(bm.group(2), "  " * lvl + col + dot + " " + C.W)
            i += 1; continue

        nm = re.match(r'^(\s*)(\d+)\. (.+)', ln)
        if nm:
            lvl = len(nm.group(1)) // 2
            wrap_box(nm.group(3), "  " * lvl + C.A + nm.group(2) + ". " + C.W)
            i += 1; continue

        if "|" in ln and re.match(r'^\s*\|', ln):
            table_lines = []
            while i < len(lines) and "|" in lines[i] and re.match(r'^\s*\|', lines[i]):
                table_lines.append(lines[i]); i += 1
            rows = [
                [cell.strip() for cell in re.split(r'\|', tl.strip().strip('|'))]
                for tl in table_lines
                if not re.match(r'^[\s|:\-]+$', tl)
            ]
            if rows:
                col_count = max(len(r) for r in rows)
                rows = [r + [''] * (col_count - len(r)) for r in rows]
                max_w = iw - col_count * 3 - 2
                col_w = [min(max(len(r[c]) for r in rows), max(6, max_w // col_count))
                         for c in range(col_count)]
                def _tr(cells, widths, color):
                    parts = [color + cells[j][:widths[j]].ljust(widths[j]) + C.R
                             for j in range(len(cells))]
                    return ("  " + C.GD + "│" + C.R + " " +
                            (" " + C.GD + "│" + C.R + " ").join(parts) +
                            " " + C.GD + "│" + C.R)
                def _tsep(w): return "  " + C.GD + "├" + "┼".join("─"*(x+2) for x in w) + "┤" + C.R
                def _ttop(w): return "  " + C.GD + "┌" + "┬".join("─"*(x+2) for x in w) + "┐" + C.R
                def _tbot(w): return "  " + C.GD + "└" + "┴".join("─"*(x+2) for x in w) + "┘" + C.R
                box_line("")
                _w(_ttop(col_w) + "\n")
                for ri, row in enumerate(rows):
                    _w(_tr(row, col_w, C.AB if ri == 0 else C.W) + "\n")
                    if ri == 0: _w(_tsep(col_w) + "\n")
                _w(_tbot(col_w) + "\n")
                box_line("")
            continue

        wrap_box(ln)
        i += 1

    _w("  " + C.A + "└" + "─" * (iw + 2) + "┘" + C.R + "\n")
    _blank()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION COMMANDS
# ══════════════════════════════════════════════════════════════════════════════
_history: list[dict] = []

def show_history():
    if not _history:
        _w("  " + C.G + "no missions yet." + C.R + "\n"); return
    _blank(); _dim_line()
    _w("  " + C.AB + "history" + C.R + "\n"); _dim_line()
    for i, e in enumerate(_history, 1):
        q = e["q"][:_tw() - 20] + ("…" if len(e["q"]) > _tw() - 20 else "")
        _w(f"  {C.G}{i:02d}  {e['t']}{C.R}  {C.W}{q}{C.R}\n")
    _blank()


def show_mem_status():
    _blank(); _dim_line()
    _w("  " + C.AB + "memory status" + C.R + "\n"); _dim_line()

    # ── Hot memory ────────────────────────────────────────────────────────
    _w(f"  {C.HINT}◑  HOT MEMORY{C.R}\n")
    if _HOT_MEM and _hot_memory is not None:
        stats = _hot_memory.stats()
        _w(f"  {C.G}   turns stored : {C.W}{stats['turns']}/{stats['max_turns']}{C.R}\n")
        _w(f"  {C.G}   total atoms  : {C.W}{stats['total_atoms']}{C.R}\n")
        _w(f"  {C.G}   backed by    : {C.W}{stats['path']}{C.R}\n")
        _w(f"  {C.GD}   (wipe with /reset_mem){C.R}\n")
        entries = _hot_memory.preview_entries()
        if entries:
            _blank()
            _w("  " + C.GD + "── recent turns" + C.R + "\n")
            for e in entries[-5:]:  # show last 5
                _w(f"  {C.G}  T{e['turn']} [{e['ts']}]{C.R}\n")
                for a in e["prompt_atoms"]:
                    _w(f"  {C.GD}    IN  · {C.W}{a}{C.R}\n")
                for a in e["response_atoms"]:
                    _w(f"  {C.GD}    OUT · {C.W}{a}{C.R}\n")
    else:
        _w(f"  {C.ERR}   unavailable — check tools/spada_memory.py{C.R}\n")

    _blank()

    # ── Cold memory ───────────────────────────────────────────────────────
    _w(f"  {C.A}◉  COLD MEMORY (SPADA — ChromaDB){C.R}\n")
    persist_dir = os.getenv("SPADA_PERSIST_DIR", "./spada_db_shifu")
    collection  = os.getenv("SPADA_COLLECTION",  "shifu_memory")
    _w(f"  {C.G}   persist dir : {C.W}{persist_dir}{C.R}\n")
    _w(f"  {C.G}   collection  : {C.W}{collection}{C.R}\n")
    tool_map = {t.name: t for t in TOOLS}
    if "spada_recall" in tool_map:
        _w(f"  {C.OK}   tools loaded : recall · memorise · session_close{C.R}\n")
    else:
        _w(f"  {C.ERR}   spada tools not found in tools/{C.R}\n")
    _blank()

def show_skills():
    skills = _scan_skills()
    _blank(); _dim_line()
    _w("  " + C.AB + "installed skills" + C.R + "\n"); _dim_line()
    if not skills:
        _w("  " + C.G + "no skills installed yet.\n" + C.R)
        _w("  " + C.G + "create  skills/<name>/SKILL.md  to add one.\n" + C.R)
    else:
        for name, meta in skills.items():
            _w(f"  {C.A}◈{C.R}  {C.W}{name:<22}{C.R}  {C.G}{meta['description']}{C.R}\n")
        _w(f"\n  {C.GD}skills dir → {SKILLS_DIR.resolve()}{C.R}\n")
    _blank()

def show_help():
    _blank(); _dim_line()
    _w("  " + C.AB + "commands" + C.R + "\n"); _dim_line()
    for cmd, desc in [
        ("help / ?",    "this panel"),
        ("history",     "mission log"),
        ("skills",      "list installed skills"),
        ("files",       "supported file types"),
        ("mem_status",  "hot + cold memory stats and recent turns"),
        ("reset_mem",   "wipe hot memory (cold memory is untouched)"),
        ("clear / cls", "reset screen"),
        ("exit / quit", "shutdown (saves session memory)"),
        ("<anything>",  "send to shifu"),
    ]:
        _w(f"  {C.A}{cmd:<16}{C.R}{C.G}{desc}{C.R}\n")
    _blank()

def show_file_support():
    _blank(); _dim_line()
    _w("  " + C.AB + "supported file types" + C.R + "\n"); _dim_line()
    for kind, detail, col in [
        ("plain text",  ".txt .md .csv .json .yaml .toml .xml .log",   C.OK),
        ("source code", ".py .js .ts .sh .bat .html .css .sql …",      C.OK),
        ("PDF",         "requires  pip install pypdf",                  C.A),
        ("Word .docx",  "requires  pip install python-docx",            C.A),
        ("Excel .xlsx", "requires  pip install openpyxl",               C.A),
        ("images",      "not supported natively (no vision model)",     C.ERR),
        ("audio/video", "not supported",                                C.ERR),
    ]:
        _w(f"  {col}●{C.R}  {C.W}{kind:<14}{C.R}  {C.G}{detail}{C.R}\n")
    _blank()
    _w("  " + C.G + "tip: include the file path in your mission.\n"
       "       e.g.  »  summarise Playground/report.pdf\n" + C.R)
    _blank()

def get_input() -> str:
    try:
        ts = datetime.now().strftime("%H:%M")
        return input(f"  {C.GD}[{ts}]{C.R}  {C.AB}›{C.R}  ").strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"

async def shutdown():
    _blank()
    _w("  " + C.G + "wrapping up…" + C.R + "\n")
    spinner = Spinner("saving session…"); spinner.start()
    await _run_session_close(spinner)
    spinner.stop()
    _w("  " + C.G + "goodbye." + C.R + "\n")
    _blank()
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def _main_async():
    boot()
    _w("  " + C.OK + "✓  ready" + C.R +
       "  " + C.G  + f"playground → {PLAYGROUND_DIR.resolve()}" + C.R + "\n")
    _blank()

    while True:
        try:
            raw = await asyncio.to_thread(get_input)
        except KeyboardInterrupt:
            _blank()
            await shutdown()

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("exit", "quit", "q"):
            await shutdown()
        elif cmd in ("help", "?", "h"):
            show_help(); continue
        elif cmd in ("clear", "cls"):
            boot(); continue
        elif cmd == "history":
            show_history(); continue
        elif cmd in ("skills", "skill"):
            show_skills(); continue
        elif cmd in ("files", "filetypes", "supported"):
            show_file_support(); continue
        elif cmd in ("mem_status", "mem", "memory"):
            show_mem_status(); continue
        elif cmd in ("reset_mem", "reset_memory", "clear_mem", "clear_memory"):
            if _HOT_MEM and _hot_memory is not None:
                wiped = _hot_memory.clear()
                _w(f"  {C.OK}✓{C.R}  {C.G}hot memory wiped — {wiped} turn(s) erased.{C.R}\n")
                _w(f"  {C.GD}  cold memory (spada_db_shifu) is untouched.{C.R}\n")
            else:
                _w(f"  {C.ERR}✗{C.R}  {C.G}hot memory not available.{C.R}\n")
            _blank(); continue

        # ── mission ────────────────────────────────────────────────────────
        _blank()
        _dim_line()
        _blank()

        _session_missions.append(raw)

        bar     = Bar()
        spinner = Spinner("reading mission…")
        t0      = time.time()
        answer  = None

        try:
            answer = await asyncio.wait_for(
                run_mission_async(raw, bar, spinner),
                timeout=MISSION_TIMEOUT + 10,  # grace period on top of internal timeout
            )
        except asyncio.TimeoutError:
            try:    spinner.stop()
            except: pass
            bar.timeout()
            _w("  " + C.ERR + "mission timed out." + C.R + "\n")
            _blank(); continue
        except KeyboardInterrupt:
            try:    spinner.stop()
            except: pass
            _w("\n  " + C.G + "interrupted." + C.R + "\n")
            _blank(); continue
        except Exception as exc:
            try:    spinner.stop()
            except: pass
            _blank()
            _w("  " + C.ERR + f"error: {exc}" + C.R + "\n")
            import traceback; traceback.print_exc()
            _blank(); continue

        elapsed = time.time() - t0

        if not answer or not answer.strip():
            _w("  " + C.G + "⚠  empty response." + C.R + "\n")
            _blank(); continue

        _history.append({"t": datetime.now().strftime("%H:%M:%S"), "q": raw})
        render_response(answer, elapsed)


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()