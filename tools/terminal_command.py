"""
shifu.py — Shifu, the One-Agent LangGraph System  (Conversational Edition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Key upgrades over the robotic version
──────────────────────────────────────
1. NARRATE node  — Shifu thinks aloud *before* executing:
     "Alright, this looks like a web-scraping task. Let me first check
      if the URL is reachable, then decide the best extraction approach."

2. CLARIFY node  — planner decides whether it needs human input
   before committing. Graph pauses, prints a question, reads stdin,
   then resumes with the answer baked into the plan.

3. COMMENTARY in execute_node — after each tool call Shifu produces a
   short "mid-flight" observation ("Found 12 results — filtering for
   the three most recent ones.") instead of silent spinning.

4. FINISH node   — replaces the robotic "✅ DONE:" with a warm closing
   summary that mentions what happened, any surprises, and next steps.

5. Richer system prompts  — every prompt is written in first-person,
   conversational English and explicitly forbids bullet-point
   "task completed" endings.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

import os
import time
import textwrap
import shutil
import importlib
import pkgutil
from pathlib import Path
from typing import Annotated, TypedDict, Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
import tools

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

PLAYGROUND_DIR = Path("Playground");  PLAYGROUND_DIR.mkdir(exist_ok=True)
SKILLS_DIR     = Path("skills");       SKILLS_DIR.mkdir(exist_ok=True)
MAX_RPM        = 20
MODEL_NAME     = "gpt-oss:120b-cloud"
BASE_URL       = "https://ollama.com/v1"
MAX_ITERATIONS = 50
MAX_RETRIES    = 10

W      = min(shutil.get_terminal_size().columns, 80)
_RESET = "\033[0m";  _BOLD = "\033[1m";  _DIM = "\033[2m"
_GREEN = "\033[32m"; _RED  = "\033[31m"; _YEL = "\033[33m"
_BLU   = "\033[34m"; _MAG  = "\033[35m"; _CYN = "\033[36m"

# ── LLMs ──────────────────────────────────────────────────────────────────────

def _make_llm(api_key_env: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME, base_url=BASE_URL,
        api_key=os.getenv(api_key_env),
        temperature=0.4,      # slightly higher → more natural phrasing
        max_tokens=4096,
    )

planner_llm  = _make_llm("OLLAMA_API_KEY_PLANNER")
executor_llm = _make_llm("OLLAMA_API_KEY_EXECUTOR")

_last_call_time: float = 0.0

def _rate_limited_invoke(llm, messages):
    global _last_call_time
    gap = 60.0 / MAX_RPM
    elapsed = time.time() - _last_call_time
    if elapsed < gap:
        time.sleep(gap - elapsed)
    result = llm.invoke(messages)
    _last_call_time = time.time()
    return result

# ── Skills ────────────────────────────────────────────────────────────────────

def _scan_skills() -> dict[str, dict]:
    import re as _re
    registry: dict[str, dict] = {}
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        name = skill_md.parent.name
        text = skill_md.read_text(encoding="utf-8")
        fm: dict = {}
        m = _re.match(r'^---\s*\n(.*?)\n---', text, _re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    fm[k.strip()] = v.strip().strip('"').strip("'")
        registry[name] = {
            "name": fm.get("name", name),
            "description": fm.get("description", "(no description)"),
            "path": str(skill_md),
        }
    return registry

SKILLS = _scan_skills()

def _skills_index() -> str:
    if not SKILLS:
        return "  (no skills installed)"
    return "\n".join(
        f'  • load_skill("{n}")  —  {m["description"]}' for n, m in SKILLS.items()
    )

# ── Tools ─────────────────────────────────────────────────────────────────────

def _load_pkg_tools(package) -> list[BaseTool]:
    found = []
    for loader, mod_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        mod = importlib.import_module(mod_name)
        for obj in vars(mod).values():
            if isinstance(obj, BaseTool):
                found.append(obj)
    return found

TOOLS             = _load_pkg_tools(tools)
tool_node         = ToolNode(TOOLS)
executor_with_tools = executor_llm.bind_tools(TOOLS)

# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = f"""You are Shifu's strategic mind — sharp, concise, and honest.

AVAILABLE SKILLS:
{_skills_index()}

[CLASSIFY]
Reply with exactly one word: SIMPLE or COMPLEX.
  SIMPLE  = one tool call or one obvious step.
  COMPLEX = multiple steps, ambiguity, or research needed.

[NEEDS_CLARIFICATION]
Reply with either:
  NO_CLARIFICATION
or
  CLARIFY: <one focused question to ask the user>

Rules for asking:
- Only ask if the answer genuinely changes *how* the task is done.
- Never ask for things you can default (e.g. missing end time → 1 hour).
- Prefer one good question over multiple small ones.

[PLAN]
Write a numbered execution plan. Rules:
- Each step is one atomic action.
- If a skill matches, put load_skill("<exact-name>") as step 1.
- All files go inside Playground/.
- Maximum 10 steps.
- End with: PLAN COMPLETE.
"""

NARRATOR_SYSTEM = """You are Shifu — a capable, friendly AI agent.
Before you execute anything you give the user a short *pre-flight briefing*
in natural, conversational English.

Rules:
- 2–4 sentences max.
- Mention what you *understand* the mission to be.
- Mention the first thing you are about to do and why.
- If you spot any possible gotcha, name it briefly.
- Write in first person. No bullet lists. No bold headers.
- Do NOT say "I will now execute" or "Task received". Sound like a colleague, not a robot.

Example tone:
  "Alright, looks like you want me to scrape the pricing page and turn it into
   a comparison table. I'll start by fetching the raw HTML — the tricky part will
   be figuring out which CSS selectors hold the actual prices, so I'll inspect the
   structure first before writing anything to disk."
"""

def _build_executor_system() -> str:
    return f"""You are Shifu — an autonomous, talkative agent who gets things done using tools.

SKILLS (copy the string verbatim into load_skill):
{_skills_index()}

══ CRITICAL RULES ══════════════════════════════════════════════════════
1. ALWAYS USE TOOLS TO CREATE FILES.
   Never write file contents in plain text as part of your response.
   If the task involves creating files (code, HTML, CSS, configs, etc.),
   you MUST call the appropriate file-writing tool for EVERY file.
   Writing code in your response instead of to disk = task failure.

2. ALL files go inside Playground/ → {PLAYGROUND_DIR.resolve()}
   Never use relative paths like "./app.py". Always use the full path
   like "Playground/todo_app/app.py".

3. CREATE DIRECTORIES BEFORE FILES.
   Use the terminal tool or directory-creation tool before writing files
   into a subdirectory. Don't assume directories exist.
════════════════════════════════════════════════════════════════════════

NARRATION:
- After each tool call write 1–2 sentences describing what just happened.
  Good: "Created app.py — now writing the HTML template."
  Bad : silence, or "Tool executed successfully."
- If something fails, say so clearly before retrying.
- When done, write a warm closing paragraph starting with "Alright, I'm done —"
  covering what you built, any surprises, and how to run it.
- Never say "✅ DONE:". Never describe files without also writing them to disk.
- Max {MAX_ITERATIONS} tool-call iterations.
"""

REVIEWER_SYSTEM = """You are Shifu's quality reviewer.

PASS if ANY of these are true:
  • The agent output contains "Alright, I'm done"
  • A tool completed the core task with no error keywords in its result
  • The output clearly describes a successful outcome

RETRY only on genuine tool ERROR or unhandled exception.
Do NOT retry for terse output.

Reply exactly:
VERDICT: PASS
or
VERDICT: RETRY — <one-sentence reason>
"""

COMMENTARY_SYSTEM = """You are Shifu mid-task.
Given the result of the last tool call, write 1–2 sentences of natural commentary.
Be honest about surprises, failures, or progress.
No bullet points. No "Task completed." sound.
Example: "Interesting — the API returned 47 results but most are duplicates.
I'll deduplicate by URL before writing the file."
"""

# ── State ─────────────────────────────────────────────────────────────────────

class ShifuState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    mission:           str
    plan:              str
    complexity:        Literal["SIMPLE", "COMPLEX", ""]
    clarification_q:   str          # question posed to user, or ""
    clarification_a:   str          # user's answer, or ""
    iterations:        int
    verdict:           Literal["PASS", "RETRY", ""]
    retry_count:       int
    narration:         str          # pre-flight briefing text

# ── Nodes ──────────────────────────────────────────────────────────────────────

# 1. Classify ──────────────────────────────────────────────────────────────────
def classify_node(state: ShifuState) -> ShifuState:
    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"[CLASSIFY]\nMission: {state['mission']}"),
    ])
    complexity: Literal["SIMPLE", "COMPLEX"] = (
        "COMPLEX" if "COMPLEX" in resp.content.upper() else "SIMPLE"
    )
    return {**state, "complexity": complexity}


# 2. Check whether clarification is needed ─────────────────────────────────────
def clarify_check_node(state: ShifuState) -> ShifuState:
    """Ask the planner if it needs one clarification before planning."""
    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=(
            f"[NEEDS_CLARIFICATION]\nMission: {state['mission']}\n"
            f"Complexity: {state['complexity']}"
        )),
    ])
    content = resp.content.strip()
    if content.upper().startswith("CLARIFY:"):
        question = content[len("CLARIFY:"):].strip()
        return {**state, "clarification_q": question}
    return {**state, "clarification_q": ""}


# 3. Human clarification — uses LangGraph interrupt (NO stdin here) ────────────
def clarify_node(state: ShifuState) -> ShifuState:
    """
    Suspend the graph and surface the question to the caller.
    The terminal catches the interrupt, stops the spinner, asks the user,
    then resumes the graph via Command(resume=answer).
    This node never touches stdin or stdout directly.
    """
    answer = interrupt(state["clarification_q"])   # suspends; answer injected on resume
    if not answer or not str(answer).strip():
        answer = "(user skipped — use your best judgment)"
    return {**state, "clarification_a": str(answer).strip()}


# 4. Plan ──────────────────────────────────────────────────────────────────────
def plan_node(state: ShifuState) -> ShifuState:
    extra = ""
    if state.get("clarification_a"):
        extra = f"\nUser clarification: {state['clarification_a']}"
    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=(
            f"[PLAN]\nMission: {state['mission']}{extra}\n"
            f"Playground: {PLAYGROUND_DIR.resolve()}"
        )),
    ])
    return {**state, "plan": resp.content}


# 5. Narrate (pre-flight) ──────────────────────────────────────────────────────
def narrate_node(state: ShifuState) -> ShifuState:
    """Shifu thinks aloud before any tool is called."""
    plan_snippet = state["plan"][:600] if state["plan"] else ""
    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=NARRATOR_SYSTEM),
        HumanMessage(content=(
            f"Mission: {state['mission']}\n"
            f"Plan sketch:\n{plan_snippet}"
        )),
    ])
    return {**state, "narration": resp.content.strip()}


# 6. Execute ───────────────────────────────────────────────────────────────────
def execute_node(state: ShifuState) -> ShifuState:
    if state["iterations"] >= MAX_ITERATIONS:
        bail = AIMessage(content="⚠️ Max iterations reached. Stopping.")
        return {**state, "messages": state["messages"] + [bail]}

    # Build the initial message list on first call
    if not state["messages"]:
        user_content = f"Mission: {state['mission']}"
        if state["complexity"] == "COMPLEX" and state["plan"]:
            user_content += f"\n\nExecution Plan:\n{state['plan']}"
        if state.get("clarification_a"):
            user_content += f"\n\nUser clarification: {state['clarification_a']}"
        msgs: list[BaseMessage] = [
            SystemMessage(content=_build_executor_system()),
            HumanMessage(content=user_content),
        ]
    else:
        msgs = state["messages"]

    response = _rate_limited_invoke(executor_with_tools, msgs)
    new_msgs  = (msgs if not state["messages"] else []) + [response]
    return {**state,
            "messages":   state["messages"] + new_msgs,
            "iterations": state["iterations"] + 1}


# 7. Tools ─────────────────────────────────────────────────────────────────────
def tools_node_fn(state: ShifuState) -> ShifuState:
    result = tool_node.invoke({"messages": state["messages"]})
    return {**state, "messages": result["messages"]}


# 8. Commentary (mid-flight) ───────────────────────────────────────────────────
def commentary_node(state: ShifuState) -> ShifuState:
    """After a tool result lands, Shifu narrates what it found."""
    last_tool = next(
        (m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)), None
    )
    if not last_tool:
        return state

    snippet = str(last_tool.content).strip()[:500]
    resp = _rate_limited_invoke(executor_llm, [
        SystemMessage(content=COMMENTARY_SYSTEM),
        HumanMessage(content=f"Tool result:\n{snippet}"),
    ])
    comment = AIMessage(content=resp.content.strip())
    return {**state, "messages": state["messages"] + [comment]}


# 9. Review ────────────────────────────────────────────────────────────────────
def review_node(state: ShifuState) -> ShifuState:
    last_tool = next(
        (m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)), None
    )
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
    )
    agent_output = ""
    if last_ai:
        agent_output += last_ai.content or ""
    if last_tool:
        agent_output += "\n[Tool result]: " + str(last_tool.content)[:500]

    # fast-path
    if last_tool:
        err_kw = ("error", "exception", "failed", "traceback", "timeout")
        if not any(k in str(last_tool.content).lower() for k in err_kw):
            return {**state, "verdict": "PASS"}

    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"Mission: {state['mission']}\n\nAgent output:\n{agent_output}"),
    ])
    verdict: Literal["PASS", "RETRY"] = (
        "PASS" if "PASS" in resp.content.upper() else "RETRY"
    )
    return {**state,
            "verdict":     verdict,
            "retry_count": state["retry_count"] + (1 if verdict == "RETRY" else 0)}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_classify(state: ShifuState) -> str:
    # Always check for clarification first, regardless of complexity
    return "clarify_check"

def route_after_clarify_check(state: ShifuState) -> str:
    if state["clarification_q"]:
        return "clarify"
    # No clarification needed — go to plan (complex) or narrate (simple)
    return "plan" if state["complexity"] == "COMPLEX" else "narrate"

def route_after_clarify(state: ShifuState) -> str:
    return "plan" if state["complexity"] == "COMPLEX" else "narrate"

def route_after_plan(state: ShifuState) -> str:
    return "narrate"

def route_after_narrate(state: ShifuState) -> str:
    return "execute"

def route_after_execute(state: ShifuState) -> str:
    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "review" if state["complexity"] == "COMPLEX" else END

def route_after_tools(state: ShifuState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        return "review" if state["complexity"] == "COMPLEX" else END
    return "commentary"

def route_after_commentary(state: ShifuState) -> str:
    return "execute"

def route_after_review(state: ShifuState) -> str:
    if state["verdict"] == "PASS" or state["retry_count"] >= MAX_RETRIES:
        return END
    return "execute"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ShifuState)
    g.add_node("classify",      classify_node)
    g.add_node("clarify_check", clarify_check_node)
    g.add_node("clarify",       clarify_node)
    g.add_node("plan",          plan_node)
    g.add_node("narrate",       narrate_node)
    g.add_node("execute",       execute_node)
    g.add_node("tools",         tools_node_fn)
    g.add_node("commentary",    commentary_node)
    g.add_node("review",        review_node)

    g.set_entry_point("classify")

    g.add_conditional_edges("classify",      route_after_classify,       {"clarify_check": "clarify_check"})
    g.add_conditional_edges("clarify_check", route_after_clarify_check,  {"clarify": "clarify", "plan": "plan", "narrate": "narrate"})
    g.add_conditional_edges("clarify",       route_after_clarify,        {"plan": "plan", "narrate": "narrate"})
    g.add_conditional_edges("plan",          route_after_plan,           {"narrate": "narrate"})
    g.add_conditional_edges("narrate",       route_after_narrate,        {"execute": "execute"})
    g.add_conditional_edges("execute",       route_after_execute,        {"tools": "tools", "review": "review", END: END})
    g.add_conditional_edges("tools",         route_after_tools,          {"commentary": "commentary", "review": "review", END: END})
    g.add_conditional_edges("commentary",    route_after_commentary,     {"execute": "execute"})
    g.add_conditional_edges("review",        route_after_review,         {"execute": "execute", END: END})

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer, interrupt_before=["clarify"])

shifu_graph = build_graph()


# ── Pretty Printer ────────────────────────────────────────────────────────────

def _divider(char="━", color=""):
    print(f"{color}{char * W}{_RESET}")

def _wrap(text: str, indent: int = 3) -> str:
    pad = " " * indent
    return textwrap.fill(text, width=W - indent,
                         initial_indent=pad, subsequent_indent=pad)

_NODE_META = {
    "classify":      ("🔍", _CYN, "READING THE MISSION"),
    "clarify_check": ("🤔", _MAG, "CHECKING IF I NEED MORE INFO"),
    "clarify":       ("💬", _CYN, "CLARIFICATION"),
    "plan":          ("📋", _MAG, "DRAFTING EXECUTION PLAN"),
    "narrate":       ("🗣 ", _YEL, "SHIFU SPEAKS"),
    "execute":       ("⚙️ ", _YEL, "EXECUTING"),
    "tools":         ("🔧", _BLU, "TOOL CALL"),
    "commentary":    ("💭", _CYN, "SHIFU OBSERVES"),
    "review":        ("🔎", _MAG, "REVIEWING OUTPUT"),
}

def _header(node: str):
    icon, color, label = _NODE_META.get(node, ("▶", "", node.upper()))
    _divider("─", _DIM)
    print(f"{color}{_BOLD} {icon}  {label}{_RESET}")

# --- per-node printers ---

def _print_classify(state):
    c = state["complexity"]
    color = _GREEN if c == "SIMPLE" else _MAG
    print(f"   Complexity : {color}{_BOLD}{c}{_RESET}")

def _print_clarify_check(state):
    q = state.get("clarification_q", "")
    if q:
        print(f"{_MAG}   → Will ask: {q}{_RESET}")
    else:
        print(f"{_DIM}   → No clarification needed, proceeding.{_RESET}")

def _print_narrate(state):
    text = state.get("narration", "")
    if text:
        print(f"\n{_YEL}{_wrap(text)}{_RESET}\n")

def _print_plan(state):
    for line in state["plan"].splitlines():
        if line.strip():
            print(f"   {_DIM}{line}{_RESET}")

def _print_execute(state):
    if not state["messages"]:
        return
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            name  = tc.get("name", "?")
            args  = tc.get("args", {})
            primary = str(next(iter(args.values()), "")).replace("\n", " ")[:80]
            print(f"   {_BOLD}→ {name}{_RESET}  {_DIM}{primary}{_RESET}")
    elif isinstance(last, AIMessage) and last.content:
        snippet = last.content.strip()[:400]
        print(f"{_DIM}{_wrap(snippet)}{_RESET}")

def _print_tools(state):
    tool_msgs = []
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage):
            tool_msgs.insert(0, m)
        else:
            break
    for tm in tool_msgs:
        snippet = str(tm.content).strip()[:200].replace("\n", " ")
        print(f"   {_DIM}↩  {snippet}{_RESET}")

def _print_commentary(state):
    last = next(
        (m for m in reversed(state["messages"])
         if isinstance(m, AIMessage) and m.content), None
    )
    if last:
        print(f"\n{_CYN}{_wrap(last.content.strip()[:300])}{_RESET}\n")

def _print_review(state):
    verdict = state.get("verdict", "")
    retry   = state.get("retry_count", 0)
    if verdict == "PASS":
        print(f"   Verdict    : {_GREEN}{_BOLD}✅ PASS{_RESET}")
    elif verdict == "RETRY":
        print(f"   Verdict    : {_RED}{_BOLD}⚠️  RETRY  (attempt {retry}){_RESET}")
    else:
        print(f"   Verdict    : {_DIM}(pending){_RESET}")

_PRINTERS = {
    "classify":      _print_classify,
    "clarify_check": _print_clarify_check,
    "clarify":       lambda s: None,   # clarify prints itself interactively
    "plan":          _print_plan,
    "narrate":       _print_narrate,
    "execute":       _print_execute,
    "tools":         _print_tools,
    "commentary":    _print_commentary,
    "review":        _print_review,
}

# ── State-diff node detector ──────────────────────────────────────────────────

def _detect_node(prev, curr) -> str | None:
    if prev is None:
        return "classify"
    # check each transition signature
    if not prev.get("complexity") and curr.get("complexity"):
        return "classify"
    if prev.get("clarification_q") != curr.get("clarification_q") and curr.get("clarification_q") == "":
        # clarify_check just ran and set empty string (no question)
        return "clarify_check"
    if not prev.get("clarification_q") and curr.get("clarification_q"):
        return "clarify_check"
    if not prev.get("clarification_a") and curr.get("clarification_a"):
        return "clarify"
    if not prev.get("narration") and curr.get("narration"):
        return "narrate"
    if not prev.get("plan") and curr.get("plan"):
        return "plan"
    if prev.get("verdict") != curr.get("verdict"):
        return "review"
    prev_msgs = prev.get("messages", [])
    curr_msgs = curr.get("messages", [])
    if len(curr_msgs) > len(prev_msgs):
        new_msgs = curr_msgs[len(prev_msgs):]
        # commentary = new AIMessage that is NOT a tool_call response
        if any(isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) for m in new_msgs):
            last_before = prev_msgs[-1] if prev_msgs else None
            if isinstance(last_before, ToolMessage):
                return "commentary"
        if any(isinstance(m, ToolMessage) for m in new_msgs):
            return "tools"
        return "execute"
    return None


# ── Runner ────────────────────────────────────────────────────────────────────

def run_mission(mission: str,
                thread_id: str | None = None,
                clarify_callback=None) -> str:
    """
    Run a mission to completion.

    clarify_callback: optional callable(question: str) -> str
        Called when the graph hits the clarify interrupt.
        If None, uses a plain input() prompt (CLI mode).
        The terminal passes its own callback that stops the spinner first.
    """
    import uuid
    tid    = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    initial_state: ShifuState = {
        "messages":        [],
        "mission":         mission,
        "plan":            "",
        "complexity":      "",
        "clarification_q": "",
        "clarification_a": "",
        "iterations":      0,
        "verdict":         "",
        "retry_count":     0,
        "narration":       "",
    }

    _divider("━")
    print(f"{_BOLD}🐾  Shifu on it …{_RESET}\n")

    prev_state  = None
    final_state = None
    t0          = time.time()

    def _stream(input_val):
        nonlocal prev_state, final_state
        for step in shifu_graph.stream(input_val, config=config, stream_mode="values"):
            node = _detect_node(prev_state, step)
            if node:
                _header(node)
                printer = _PRINTERS.get(node)
                if printer:
                    printer(step)
            prev_state  = step
            final_state = step

    # ── first pass ────────────────────────────────────────────────────────────
    _stream(initial_state)

    # ── handle interrupt loop ─────────────────────────────────────────────────
    while True:
        snapshot = shifu_graph.get_state(config)
        if not snapshot.next:
            break   # graph finished normally

        # Extract the interrupt value (the question) from pending tasks
        question = ""
        for task in getattr(snapshot, "tasks", []):
            for iv in getattr(task, "interrupts", []):
                question = iv.value
                break
            if question:
                break

        if not question:
            break   # safety: no interrupt value found

        # Ask the user via callback (terminal) or fallback plain input (CLI)
        if clarify_callback:
            answer = clarify_callback(question)
        else:
            _divider("─", _CYN)
            print(f"{_CYN}{_BOLD} 💬  SHIFU NEEDS A QUICK CLARIFICATION{_RESET}")
            print(f"\n   {question}\n")
            _divider("─", _CYN)
            answer = input("   Your answer ..> ").strip()
            if not answer:
                answer = "(user skipped — use your best judgment)"

        # Resume the graph with the user's answer
        _stream(Command(resume=answer))

    elapsed = time.time() - t0
    _divider("━", _GREEN)
    print(f"{_GREEN}{_BOLD} ✅  MISSION COMPLETE  ({elapsed:.1f}s){_RESET}")
    _divider("━", _GREEN)

    if not final_state:
        return "No output produced."

    messages = final_state.get("messages", [])
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    if last_ai:
        return last_ai.content

    # Fallback: surface tool results so CLI users see something meaningful
    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    if tool_results:
        lines = [f"  [{getattr(m,'name','tool')}] {str(m.content).strip()[:120]}"
                 for m in tool_results[-5:]]
        return "Mission complete. Last tool outputs:\n" + "\n".join(lines)

    return "Mission complete (no textual output)."


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{_BOLD}🐾  Shifu — Conversational Edition{_RESET}")
    print(f"   Playground: {PLAYGROUND_DIR.resolve()}")
    print("━" * W)
    mission = input("Mission ..> ").strip()
    if not mission:
        print("No mission. Shifu meditates.")
    else:
        result = run_mission(mission)
        print()
        print(result)
        print()