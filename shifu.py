#!/usr/bin/env python3
"""
shifu.py — One-file Shifu: Agent + Terminal UI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The flow that modern AI companies use:

  1.  User sends mission
  2.  Shifu *streams* a conversational pre-flight brief
  3.  Tool calls appear inline with live status
  4.  After each tool result, Shifu *streams* a short observation
  5.  A warm closing summary *streams* token-by-token
  6.  Clean, minimal, amber-on-dark terminal UI

Key design decisions
────────────────────
• Real token-level streaming via .astream_events() — every word appears
  as it is generated, just like Claude / GPT products.
• The "talk → work → talk" rhythm is baked into the executor prompt so
  the model narrates naturally instead of going silent during tool use.
• Clarification uses LangGraph interrupt (graph suspends, terminal asks,
  graph resumes) — no threading hacks needed.
• Single file: no import-shifu dance, no split-brain config.
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

# ── third-party ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
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

MODEL_NAME     = "gpt-oss:120b-cloud"
BASE_URL       = "https://ollama.com/v1"
MAX_RPM        = 20
MAX_ITERATIONS = 50
MAX_RETRIES    = 10


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL PALETTE & PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

class C:
    R       = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    A       = "\033[38;5;214m"          # amber
    AB      = "\033[1;38;5;214m"        # amber bold
    ABGF    = "\033[38;5;0m\033[48;5;214m"  # black-on-amber (logo fill)
    W       = "\033[38;5;252m"          # near-white
    G       = "\033[38;5;240m"          # mid grey
    GD      = "\033[38;5;236m"          # dark grey
    OK      = "\033[38;5;71m"           # muted green
    ERR     = "\033[38;5;167m"          # muted red
    TOOL    = "\033[38;5;67m"           # steel-blue for tool names
    PLAN    = "\033[38;5;139m"          # muted violet for plan steps
    STREAM  = "\033[38;5;223m"          # soft cream for streamed tokens
    PANELBG = "\033[48;5;234m"
    RBG     = "\033[49m"


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
            if ch == " ":
                filled += " "
            else:
                filled += C.AB + ch + C.R
        _w("  " + raw[:first] + filled + "\n")
        time.sleep(0.018)
    _blank()
    ts = datetime.now().strftime("%A, %d %B %Y  ·  %H:%M")
    _w("  " + C.G + ts + C.R + "\n")
    _dim_line()
    _w("  " + C.G + "help  ·  history  ·  skills  ·  exit" + C.R + "\n")
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
        streaming=True,         # ← must be True for astream_events
    )

planner_llm  = _make_llm("OLLAMA_API_KEY_PLANNER")
executor_llm = _make_llm("OLLAMA_API_KEY_EXECUTOR")

_last_call: float = 0.0

def _invoke(llm, messages):
    """Synchronous rate-limited invoke (used for planner nodes)."""
    global _last_call
    gap = 60.0 / MAX_RPM
    wait = gap - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    result = llm.invoke(messages)
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

_TOOL_ICONS = {
    "web_search":           ("⌕",  "search"),
    "file_write":           ("↓",  "write"),
    "file_read":            ("↑",  "read"),
    "directory_read":       ("⊞",  "ls"),
    "terminal_command":     ("$",  "shell"),
    "load_skill":           ("◈",  "skill"),
    "browser_task":         ("◉",  "browse"),
    "browser_screenshot":   ("⊡",  "screen"),
    "browser_extract_text": ("≡",  "extract"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_PLANNER_SYS = f"""You are Shifu's strategic mind — sharp, concise, honest.

AVAILABLE SKILLS:
{_skills_index()}

[CLASSIFY]  Reply with exactly one word: SIMPLE or COMPLEX.
  SIMPLE  = SIMPLE  = one tool call or one obvious step, including casual chat or
            emotional input that just needs a memory recall + warm reply.
  COMPLEX = multiple steps, ambiguity, or research needed.

[NEEDS_CLARIFICATION]  Reply with either:
  NO_CLARIFICATION
or
  CLARIFY: <one focused question>
Only ask if the answer changes *how* the task is done. Never ask for
things you can default. Prefer one good question over several small ones.

[PLAN]  Write a numbered execution plan.
  - Each step is one atomic action.
  - If a skill matches, put load_skill("<exact-name>") as step 1.
  - Step 2 (or step 1 for simple tasks): spada_recall relevant context.
  - LAST step always: spada_memorise anything new learned this session.
  - All files go inside Playground/.
  - Max 10 steps. End with: PLAN COMPLETE.
"""

_EXECUTOR_SYS = f"""You are Shifu — a capable, talkative AI agent who gets things done with tools.

SKILLS (copy verbatim into load_skill):
{_skills_index()}
══ MEMORY RULES ══════════════════════════════════════════════════════════
══ MEMORY ════════════════════════════════════════════════════════════════
M1. RECALL FIRST — always run spada_recall before anything else, even for
    casual messages. Query intent + likely context (hobbies, mood, projects).

M2. TANGENTIAL IS FINE — recalled atoms are context, not just answers.
    Chess memory + "I'm bored" → "hop on chess.com?" Connect the dots.

M3. STORE — after every exchange, call spada_memorise
    on anything worth keeping. No permission needed. Categories to watch:
    personal facts · preferences · hobbies · ongoing projects · decisions
    · emotional patterns · research results · task outcomes. Remember your memory
    is your temple, don't polute it with unnecessary things, it will make 
    things difficult for you later

M4. LOW SCORES ARE USABLE — score < 0.5 → surface softly:
    "I vaguely remember you mentioned X — still true?"
    Never hallucinate. Always flag uncertainty.

M5. NO EXCUSES — never say "I don't remember" without calling spada_recall
    first. Skipping recall or storage = task failure.
═════════════════════════════════════════════════════════════════════════
══ CRITICAL RULES ════════════════════════════════════════════════════════
1. ALWAYS USE TOOLS TO CREATE FILES.
   Never write file content in your response text. Call the file-writing
   tool for every file. Writing code in your response = task failure.

1a. NEVER CREATE FILES FOR CONVERSATIONAL RESPONSES.
    Recommendations, explanations, lists, opinions, casual chat, and
    memory-based answers must be written directly in your response text —
    NOT saved to Playground/. Ask yourself: did the user ask you to build
    or save something? If no, do not touch the file system.
    Creating an unrequested file = task failure.

2. ALL files go inside Playground/ → {PLAYGROUND_DIR.resolve()}
   Always use full paths like "Playground/todo_app/app.py".

3. CREATE DIRECTORIES BEFORE FILES.
   Use the terminal or mkdir tool before writing files into subdirectories.

4. NEVER ASK THE USER QUESTIONS MID-TASK.
   If something is ambiguous, make a reasonable assumption, state it
   briefly ("I'll assume you want X — if not, just ask me to redo it"),
   and continue working. The time for questions was before execution.
   Asking a question instead of acting = task failure.
═════════════════════════════════════════════════════════════════════════

FLOW — this is how you must sound, every single response:

  Before your first tool call:
    Write 2-3 sentences telling the user what you understand the mission to
    be, what you are about to do first, and why. Natural English. First
    person. No bullet lists. No "Task received." Sound like a friend who
    actually knows what they are doing.

  After each tool result:
    Write 1-2 sentences commenting on what just came back. Be honest about
    surprises, errors, partial results. Never just say "Done." or be silent.
    Example: "Interesting — the API returned 47 results but most are
    duplicates. I'll deduplicate by URL before writing the file."

  When you are finished (no more tool calls needed):
    Start your final message with "Alright, I'm done —" and write a warm
    closing paragraph: what you built, any surprises you hit, and how to
    run / use / open the result. No bullet lists. No "✅ DONE:".

Max {MAX_ITERATIONS} tool-call iterations.
"""

_REVIEWER_SYS = """You are a quality reviewer.

PASS if ANY of these are true:
  • The agent output contains "Alright, I'm done"
  • A tool completed the core task with no error keywords in its result
  • The output clearly describes a successful outcome
  • The agent asked a clarifying question (that is valid behaviour, not failure)
  • The agent produced a response of any kind without a tool error

RETRY only if ALL of these are true:
  • A tool call explicitly returned an error, exception, or traceback
  • AND the task is clearly not complete

NEVER retry for: terse output, missing details, asking questions, or
responses that don't start with "Alright, I'm done".

Reply exactly:
VERDICT: PASS
or
VERDICT: RETRY — <one-sentence reason, must name the specific tool error>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════════════

class ShifuState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    mission:           str
    plan:              str
    complexity:        Literal["SIMPLE", "COMPLEX", ""]
    clarification_q:   str
    clarification_a:   str
    iterations:        int
    verdict:           Literal["PASS", "RETRY", ""]
    retry_count:       int


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def classify_node(state: ShifuState) -> ShifuState:
    resp = _invoke(planner_llm, [
        SystemMessage(content=_PLANNER_SYS),
        HumanMessage(content=f"[CLASSIFY]\nMission: {state['mission']}"),
    ])
    c: Literal["SIMPLE", "COMPLEX"] = (
        "COMPLEX" if "COMPLEX" in resp.content.upper() else "SIMPLE"
    )
    return {**state, "complexity": c}


def clarify_check_node(state: ShifuState) -> ShifuState:
    resp = _invoke(planner_llm, [
        SystemMessage(content=_PLANNER_SYS),
        HumanMessage(content=(
            f"[NEEDS_CLARIFICATION]\nMission: {state['mission']}\n"
            f"Complexity: {state['complexity']}"
        )),
    ])
    content = resp.content.strip()
    if content.upper().startswith("CLARIFY:"):
        return {**state, "clarification_q": content[len("CLARIFY:"):].strip()}
    return {**state, "clarification_q": ""}


def clarify_node(state: ShifuState) -> ShifuState:
    """Suspend the graph; caller injects the answer via Command(resume=...)."""
    answer = interrupt(state["clarification_q"])
    if not answer or not str(answer).strip():
        answer = "(user skipped — use your best judgment)"
    return {**state, "clarification_a": str(answer).strip()}


def plan_node(state: ShifuState) -> ShifuState:
    extra = f"\nUser clarification: {state['clarification_a']}" if state.get("clarification_a") else ""
    resp = _invoke(planner_llm, [
        SystemMessage(content=_PLANNER_SYS),
        HumanMessage(content=(
            f"[PLAN]\nMission: {state['mission']}{extra}\n"
            f"Playground: {PLAYGROUND_DIR.resolve()}"
        )),
    ])
    return {**state, "plan": resp.content}


def execute_node(state: ShifuState) -> ShifuState:
    if state["iterations"] >= MAX_ITERATIONS:
        bail = AIMessage(content="⚠️ Max iterations reached. Stopping.")
        return {**state, "messages": state["messages"] + [bail]}

    if not state["messages"]:
        content = f"Mission: {state['mission']}"
        if state["complexity"] == "COMPLEX" and state["plan"]:
            content += f"\n\nExecution Plan:\n{state['plan']}"
        if state.get("clarification_a"):
            content += f"\n\nUser clarification: {state['clarification_a']}"
        msgs: list[BaseMessage] = [
            SystemMessage(content=_EXECUTOR_SYS),
            HumanMessage(content=content),
        ]
    else:
        msgs = state["messages"]

    response = _invoke(executor_with_tools, msgs)
    new_msgs = (msgs if not state["messages"] else []) + [response]
    return {**state,
            "messages":   state["messages"] + new_msgs,
            "iterations": state["iterations"] + 1}


def tools_node_fn(state: ShifuState) -> ShifuState:
    result = tool_node.invoke({"messages": state["messages"]})
    return {**state, "messages": result["messages"]}


def review_node(state: ShifuState) -> ShifuState:
    last_tool = next((m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)), None)
    last_ai   = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),   None)

    # fast-path: no error keywords → PASS immediately
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

def _route_classify(state):       return "clarify_check"
def _route_clarify_check(state):
    if state["clarification_q"]:  return "clarify"
    return "plan" if state["complexity"] == "COMPLEX" else "execute"
def _route_clarify(state):
    return "plan" if state["complexity"] == "COMPLEX" else "execute"
def _route_plan(state):           return "execute"
def _route_execute(state):
    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "review" if state["complexity"] == "COMPLEX" else END
def _route_tools(state):
    return END if state["iterations"] >= MAX_ITERATIONS else "execute"
def _route_review(state):
    if state["verdict"] == "PASS" or state["retry_count"] >= MAX_RETRIES:
        return END
    return "execute"


# ── build ──────────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(ShifuState)
    for name, fn in [
        ("classify",      classify_node),
        ("clarify_check", clarify_check_node),
        ("clarify",       clarify_node),
        ("plan",          plan_node),
        ("execute",       execute_node),
        ("tools",         tools_node_fn),
        ("review",        review_node),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify",      _route_classify,       {"clarify_check": "clarify_check"})
    g.add_conditional_edges("clarify_check", _route_clarify_check,  {"clarify": "clarify", "plan": "plan", "execute": "execute"})
    g.add_conditional_edges("clarify",       _route_clarify,        {"plan": "plan", "execute": "execute"})
    g.add_conditional_edges("plan",          _route_plan,           {"execute": "execute"})
    g.add_conditional_edges("execute",       _route_execute,        {"tools": "tools", "review": "review", END: END})
    g.add_conditional_edges("tools",         _route_tools,          {"execute": "execute", END: END})
    g.add_conditional_edges("review",        _route_review,         {"execute": "execute", END: END})

    return g.compile(checkpointer=MemorySaver(), interrupt_before=["clarify"])

shifu_graph = _build_graph()


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMING EXECUTOR
#  This is the heart of the "modern AI" feel:
#  instead of polling graph.stream(), we run the executor LLM directly
#  with astream_events and print tokens live, then let the graph handle
#  routing.  Non-executor nodes run silently and fast.
# ══════════════════════════════════════════════════════════════════════════════

async def _stream_executor_turn(
    messages: list[BaseMessage],
    silent: bool = False,
    stop_on: str | None = None,
) -> AIMessage:
    full_text       = ""
    tool_calls      = []
    in_text         = False
    printing_paused = False

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
                    if not silent and not printing_paused:
                        # stop printing once the marker appears in accumulated text
                        if stop_on and stop_on in full_text.lower():
                            printing_paused = True
                        else:
                            in_text = True
                            _w(C.STREAM + token + C.R)
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
#  SPINNER  (runs while planner / reviewer nodes work)
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
#  CHECKPOINT DISPLAY  (the live progress trail)
# ══════════════════════════════════════════════════════════════════════════════

class Bar:
    """Prints a committed trail of checkpoints above the spinner line."""

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

    def complexity(self, c: str):
        col = C.OK if c == "SIMPLE" else C.A
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {col}◈{C.R}  {col}{c.lower()}{C.R}")

    def plan_step(self, line: str):
        clean = line.strip()
        if clean:
            self._commit(f"  {C.G}{'':>6}{C.R}  {C.PLAN}›{C.R}  {C.G}{clean[:70]}{C.R}")

    def tool_call(self, name: str, arg: str = ""):
        icon, label = _TOOL_ICONS.get(name, ("·", name))
        snippet = arg.replace("\n", " ")[:55]
        if len(arg) > 55: snippet += "…"
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.TOOL}{icon}  {label}{C.R}  {C.GD}{snippet}{C.R}")

    def tool_result(self, name: str, result: str):
        _, label = _TOOL_ICONS.get(name, ("·", name))
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

    def clarification(self, question: str, answer: str):
        qs = question[:55] + ("…" if len(question) > 55 else "")
        as_ = answer[:45] + ("…" if len(answer) > 45 else "")
        self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.A}💬{C.R}  {C.GD}{qs}{C.R}  {C.W}→ {as_}{C.R}")

    def verdict(self, v: str):
        if v == "PASS":
            self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.OK}◉  pass{C.R}")
        else:
            self._commit(f"  {C.G}{self._elapsed():>6}{C.R}  {C.ERR}◉  retry{C.R}")


# ══════════════════════════════════════════════════════════════════════════════
#  MISSION RUNNER  (the main async loop)
# ══════════════════════════════════════════════════════════════════════════════

async def run_mission_async(mission: str, bar: Bar, spinner: Spinner) -> str:
    """
    Drive the graph with streaming executor turns.

    Architecture:
      • Planner/reviewer nodes run via synchronous invoke (fast, no streaming needed)
      • Executor turns run via astream_events so tokens print live
      • Tool results are printed by bar.tool_result()
      • The graph state is managed manually so we can intercept execute_node
    """
    tid    = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    state: ShifuState = {
        "messages":        [],
        "mission":         mission,
        "plan":            "",
        "complexity":      "",
        "clarification_q": "",
        "clarification_a": "",
        "iterations":      0,
        "verdict":         "",
        "retry_count":     0,
    }

    # ── helper: run everything up to (but not including) execute, then return ──
    # We use graph.stream() for all non-execute nodes (they're fast and don't
    # stream), and handle execute ourselves so we can stream tokens.

    def _run_planning_nodes(input_val):
        """Run graph until it hits execute or END, return final state."""
        nonlocal state
        for step in shifu_graph.stream(input_val, config=config, stream_mode="values"):
            state = step
        return state

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 1: planning (classify → clarify? → plan → execute entry)
    # ────────────────────────────────────────────────────────────────────────
    spinner.start()
    spinner._label = "reading mission…"

    # Stream graph until we'd hit execute
    prev = None
    for step in shifu_graph.stream(state, config=config, stream_mode="values"):
        # update bar
        if prev is None or (not prev.get("complexity") and step.get("complexity")):
            spinner.stop()
            bar.complexity(step.get("complexity", ""))
            spinner = Spinner("planning…"); spinner.start()
        if not (prev or {}).get("plan") and step.get("plan"):
            spinner.stop()
            bar.phase("plan", "◈")
            for ln in step["plan"].splitlines():
                if re.match(r'^\s*\d+\.', ln):
                    bar.plan_step(ln)
            spinner = Spinner("preparing…"); spinner.start()
        prev = step
        state = step

    # ── handle clarify interrupt ───────────────────────────────────────────
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

        # ask user
        spinner.pause()
        spinner.stop()
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
            answer = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if not answer:
            answer = "(user skipped — use your best judgment)"

        bar.clarification(question, answer)
        spinner = Spinner("planning…"); spinner.start()

        for step in shifu_graph.stream(Command(resume=answer), config=config, stream_mode="values"):
            prev_c = prev.get("plan") if prev else None
            if not prev_c and step.get("plan"):
                spinner.stop()
                bar.phase("plan", "◈")
                for ln in step["plan"].splitlines():
                    if re.match(r'^\s*\d+\.', ln):
                        bar.plan_step(ln)
                spinner = Spinner("preparing…"); spinner.start()
            prev  = step
            state = step

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 2: execute loop  (token streaming + tool calls)
    # ────────────────────────────────────────────────────────────────────────
    spinner.stop()

    for iteration in range(MAX_ITERATIONS):
        # Build message list
        if not state["messages"]:
            content = f"Mission: {state['mission']}"
            if state["complexity"] == "COMPLEX" and state["plan"]:
                content += f"\n\nExecution Plan:\n{state['plan']}"
            if state.get("clarification_a"):
                content += f"\n\nUser clarification: {state['clarification_a']}"
            msgs: list[BaseMessage] = [
                SystemMessage(content=_EXECUTOR_SYS),
                HumanMessage(content=content),
            ]
        else:
            msgs = state["messages"]

        # ── STREAM the executor response ────────────────────────────────────
        _blank()
        _dim_line()
        bar.phase(f"shifu  (turn {iteration + 1})", "▸")

        _FINAL_MARKER = "Alright, I'm done"
        response = await _stream_executor_turn(msgs, silent=False, stop_on=_FINAL_MARKER)

        # merge into state manually
        if not state["messages"]:
            all_msgs = msgs + [response]
        else:
            all_msgs = state["messages"] + [response]
        state = {**state, "messages": all_msgs, "iterations": iteration + 1}
                # ── no tool calls → we're done or need review ───────────────────────
        if not response.tool_calls:
            if state["complexity"] == "COMPLEX":
                spinner = Spinner("reviewing…"); spinner.start()
                state = review_node(state)
                spinner.stop()
                bar.verdict(state["verdict"])
                if state["verdict"] == "PASS" or state["retry_count"] >= MAX_RETRIES:
                    break
                # RETRY — clear messages so executor gets a clean slate
                # Keep the mission/plan/clarification but drop the failed exchange
                state = {**state, "messages": []}
            else:
                break  # SIMPLE: done
            continue

        # ── tool calls present ──────────────────────────────────────────────
        for tc in response.tool_calls:
            name = tc.get("name", "?")
            args = tc.get("args", {})
            primary = str(next(iter(args.values()), "")).replace("\n", " ")
            bar.tool_call(name, primary)

        # run tools
        tool_msgs = []
        tool_map = {t.name: t for t in TOOLS}
        for tc in response.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            tid  = tc.get("id") or str(uuid.uuid4())
            tool = tool_map.get(name)
            if tool is None:
                result = f"Error: tool '{name}' not found."
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = f"Error: {e}"
            tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tid, name=name))
        state = {**state, "messages": all_msgs + tool_msgs}
        
        # display tool results
        for tm in tool_msgs:
            if isinstance(tm, ToolMessage):
                bar.tool_result(getattr(tm, "name", "tool"), str(tm.content))

    # ────────────────────────────────────────────────────────────────────────
    # EXTRACT FINAL RESPONSE
    # ────────────────────────────────────────────────────────────────────────
    messages = state.get("messages", [])
    last_ai = next(
    (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None
)
    if last_ai:
        content = last_ai.content
        idx = content.lower().find("alright, i'm done")
        # send only the "Alright, I'm done —" onwards to the amber box
        return content[idx:] if idx != -1 else content

    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    if tool_results:
        lines = [f"  [{getattr(m,'name','tool')}] {str(m.content).strip()[:120]}"
                 for m in tool_results[-5:]]
        return "Mission complete. Last tool outputs:\n" + "\n".join(lines)

    return "Mission complete (no textual output)."


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
    """
    Render only the *final* assistant message in the amber box.
    (The streaming already printed intermediate tokens inline.)
    """
    iw = _tw() - 6
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
            # continuation lines indent to align with text, not repeat the bullet
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
        ("clear / cls", "reset screen"),
        ("exit / quit", "shutdown"),
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

def shutdown():
    _blank()
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
            raw = get_input()
        except KeyboardInterrupt:
            _blank(); shutdown()

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("exit", "quit", "q"):
            shutdown()
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

        # ── mission ────────────────────────────────────────────────────────
        _blank()
        _dim_line()
        _blank()

        bar     = Bar()
        spinner = Spinner("reading mission…")
        t0      = time.time()
        answer  = None

        try:
            answer = await run_mission_async(raw, bar, spinner)
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

        # The final "Alright, I'm done —" summary gets rendered in the amber box
        render_response(answer, elapsed)


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()