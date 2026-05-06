"""
shifu.py — Shifu, the One-Agent LangGraph System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One agent. All tools. Two LLMs — planner for pre-flight reasoning,
executor for action, writing, code, and search.

Simple mission → direct execution.
Complex mission → planner drafts a step-by-step plan →
                  executor carries it out → reviewer signs off.

All generated files land in ./Playground/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import subprocess
import time
from pathlib import Path
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import importlib
import pkgutil
import tools
from langchain_core.tools import BaseTool

def load_all_tools_from_package(package):
    all_tools = []
    for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            if isinstance(obj, BaseTool):
                all_tools.append(obj)
    return all_tools

load_dotenv()

# ── Playground ────────────────────────────────────────────────────────────────
PLAYGROUND_DIR = Path("Playground")
PLAYGROUND_DIR.mkdir(exist_ok=True)

MODEL_NAME = "gpt-oss:120b-cloud"
BASE_URL   = "https://ollama.com/v1"

def _make_llm(api_key_env: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=os.getenv(api_key_env),
        temperature=0.2,
        max_tokens=4096,
    )

planner_llm  = _make_llm("OLLAMA_API_KEY_PLANNER")
executor_llm = _make_llm("OLLAMA_API_KEY_EXECUTOR")

# ── Rate-limit shim ───────────────────────────────────────────────────────────
MAX_RPM = 20
_last_call_time: float = 0.0

def _rate_limited_invoke(llm, messages):
    global _last_call_time
    min_gap = 60.0 / MAX_RPM
    elapsed = time.time() - _last_call_time
    if elapsed < min_gap:
        time.sleep(min_gap - elapsed)
    result = llm.invoke(messages)
    _last_call_time = time.time()
    return result

# ── Skills registry ───────────────────────────────────────────────────────────
SKILLS_DIR = Path("skills")
SKILLS_DIR.mkdir(exist_ok=True)
MAX_ITERATIONS = 50
MAX_RETRIES    = 10

def _scan_skills() -> dict[str, dict]:
    import re as _re
    registry: dict[str, dict] = {}
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        skill_name = skill_md.parent.name
        text = skill_md.read_text(encoding="utf-8")
        fm: dict = {}
        m = _re.match(r'^---\s*\n(.*?)\n---', text, _re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    fm[k.strip()] = v.strip().strip('"').strip("'")
        registry[skill_name] = {
            "name":        fm.get("name", skill_name),
            "description": fm.get("description", "(no description)"),
            "tags":        fm.get("tags", ""),
            "path":        str(skill_md),
        }
    return registry

SKILLS: dict[str, dict] = _scan_skills()

def _skills_index() -> str:
    if not SKILLS:
        return "  (no skills installed)"
    lines = []
    for name, meta in SKILLS.items():
        lines.append(f"  • {name:<26} {meta['description']}")
    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────────────────────
_package_tools = load_all_tools_from_package(tools)
TOOLS = _package_tools
tool_node = ToolNode(TOOLS)
executor_with_tools = executor_llm.bind_tools(TOOLS)

# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are Shifu's strategic mind.

[CLASSIFY]
Reply with exactly one word — SIMPLE or COMPLEX.
- SIMPLE: single-step, one-tool, or trivial.
- COMPLEX: multi-step, multi-tool, or open-ended.

[PLAN]
Write a numbered execution plan for the executor.
Rules:
- Each step is one atomic action.
- If the mission matches a skill, put load_skill(<name>) as step 1.
- All file paths must be inside Playground/.
- 10 steps maximum.
- End with: PLAN COMPLETE.
"""

def _build_executor_system() -> str:
    return f"""You are Shifu — an autonomous agent.

SKILLS (call load_skill(<name>) to read full instructions before starting):
{_skills_index()}

CONVENTIONS:
- All files go inside Playground/  →  {PLAYGROUND_DIR.resolve()}
- Use tools — never hallucinate outputs.
- End every mission with a summary starting: ✅ DONE:
- Max {MAX_ITERATIONS} tool-call iterations.
"""

REVIEWER_SYSTEM = """You are Shifu's quality reviewer.

Given the mission and the agent's output, reply in exactly this format:
VERDICT: PASS
or
VERDICT: RETRY — <one-sentence reason>
"""

# ── Graph State ───────────────────────────────────────────────────────────────

class ShifuState(TypedDict):
    messages:    Annotated[list[BaseMessage], add_messages]
    mission:     str
    plan:        str
    complexity:  Literal["SIMPLE", "COMPLEX", ""]
    iterations:  int
    verdict:     Literal["PASS", "RETRY", ""]
    retry_count: int

# ── Nodes ─────────────────────────────────────────────────────────────────────

def classify_node(state: ShifuState) -> ShifuState:
    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"[CLASSIFY]\nMission: {state['mission']}"),
    ])
    complexity: Literal["SIMPLE", "COMPLEX"] = (
        "COMPLEX" if "COMPLEX" in resp.content.upper() else "SIMPLE"
    )
    return {**state, "complexity": complexity}

def plan_node(state: ShifuState) -> ShifuState:
    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=(
            f"[PLAN]\nMission: {state['mission']}\n"
            f"Playground: {PLAYGROUND_DIR.resolve()}"
        )),
    ])
    return {**state, "plan": resp.content}

def execute_node(state: ShifuState) -> ShifuState:
    if state["iterations"] >= MAX_ITERATIONS:
        bail = AIMessage(content="⚠️ Max iterations reached. Stopping.")
        return {**state, "messages": state["messages"] + [bail]}

    if state["complexity"] == "COMPLEX" and state["plan"]:
        user_content = (
            f"Mission: {state['mission']}\n\n"
            f"Execution Plan:\n{state['plan']}\n\n"
            "Carry out the plan step by step using your tools."
        )
    else:
        user_content = f"Mission: {state['mission']}"

    if not state["messages"]:
        msgs: list[BaseMessage] = [
            SystemMessage(content=_build_executor_system()),
            HumanMessage(content=user_content),
        ]
    else:
        msgs = state["messages"]

    response = _rate_limited_invoke(executor_with_tools, msgs)
    new_messages = (msgs if not state["messages"] else []) + [response]
    return {**state,
            "messages":   state["messages"] + new_messages,
            "iterations": state["iterations"] + 1}

def tools_node(state: ShifuState) -> ShifuState:
    result = tool_node.invoke({"messages": state["messages"]})
    return {**state, "messages": result["messages"]}

def review_node(state: ShifuState) -> ShifuState:
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    agent_output = last_ai.content if last_ai else "(no output)"

    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=(
            f"Mission: {state['mission']}\n\nAgent output:\n{agent_output}"
        )),
    ])
    verdict: Literal["PASS", "RETRY"] = (
        "PASS" if "PASS" in resp.content.upper() else "RETRY"
    )
    return {**state,
            "verdict":     verdict,
            "retry_count": state["retry_count"] + (1 if verdict == "RETRY" else 0)}

# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_classify(state: ShifuState) -> str:
    return "plan" if state["complexity"] == "COMPLEX" else "execute"

def route_after_execute(state: ShifuState) -> str:
    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "review" if state["complexity"] == "COMPLEX" else END

def route_after_tools(state: ShifuState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        return "review" if state["complexity"] == "COMPLEX" else END
    return "execute"

def route_after_review(state: ShifuState) -> str:
    if state["verdict"] == "PASS" or state["retry_count"] >= MAX_RETRIES:
        return END
    return "execute"

# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ShifuState)
    g.add_node("classify", classify_node)
    g.add_node("plan",     plan_node)
    g.add_node("execute",  execute_node)
    g.add_node("tools",    tools_node)
    g.add_node("review",   review_node)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route_after_classify, {"plan": "plan", "execute": "execute"})
    g.add_edge("plan", "execute")
    g.add_conditional_edges("execute", route_after_execute, {"tools": "tools", "review": "review", END: END})
    g.add_conditional_edges("tools", route_after_tools, {"execute": "execute", "review": "review", END: END})
    g.add_conditional_edges("review", route_after_review, {"execute": "execute", END: END})
    return g.compile()

shifu_graph = build_graph()

# ── Pretty printer (unchanged) ────────────────────────────────────────────────

import textwrap, shutil

W = min(shutil.get_terminal_size().columns, 72)

_NODE_META = {
    "classify": ("🔍", "\033[36m",  "CLASSIFYING MISSION"),
    "plan":     ("📋", "\033[35m",  "DRAFTING EXECUTION PLAN"),
    "execute":  ("⚙️ ", "\033[33m",  "EXECUTING"),
    "tools":    ("🔧", "\033[34m",  "TOOL CALL"),
    "review":   ("🔎", "\033[35m",  "REVIEWING OUTPUT"),
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_RED   = "\033[31m"
_DIM   = "\033[2m"

def _divider(char="━", color=""):
    print(f"{color}{char * W}{_RESET}")

def _header(node: str):
    icon, color, label = _NODE_META.get(node, ("▶", "", node.upper()))
    _divider("─", _DIM)
    print(f"{color}{_BOLD} {icon}  {label}{_RESET}")

def _print_classify(state):
    c = state["complexity"]
    color = _GREEN if c == "SIMPLE" else "\033[35m"
    print(f"   Complexity : {color}{_BOLD}{c}{_RESET}")

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
            name = tc.get("name", "?")
            args = tc.get("args", {})
            primary = str(next(iter(args.values()), "")).replace("\n", " ")[:80]
            print(f"   {_BOLD}→ {name}{_RESET}  {_DIM}{primary}{_RESET}")
    elif isinstance(last, AIMessage) and last.content:
        snippet = last.content.strip()[:300]
        wrapped = textwrap.fill(snippet, width=W - 5, initial_indent="   ", subsequent_indent="   ")
        print(f"{_DIM}{wrapped}{_RESET}")

def _print_tools(state):
    from langchain_core.messages import ToolMessage
    tool_msgs = []
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage):
            tool_msgs.insert(0, m)
        else:
            break
    for tm in tool_msgs:
        snippet = str(tm.content).strip()[:200].replace("\n", " ")
        print(f"   {_DIM}↩  {snippet}{_RESET}")

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
    "classify": _print_classify,
    "plan":     _print_plan,
    "execute":  _print_execute,
    "tools":    _print_tools,
    "review":   _print_review,
}

def _detect_node(prev, curr) -> str | None:
    if prev is None:
        return "classify"
    if not prev.get("complexity") and curr.get("complexity"):
        return "classify"
    if not prev.get("plan") and curr.get("plan"):
        return "plan"
    if prev.get("verdict") != curr.get("verdict"):
        return "review"
    prev_msgs = prev.get("messages", [])
    curr_msgs = curr.get("messages", [])
    if len(curr_msgs) > len(prev_msgs):
        from langchain_core.messages import ToolMessage
        new_msgs = curr_msgs[len(prev_msgs):]
        if any(isinstance(m, ToolMessage) for m in new_msgs):
            return "tools"
        return "execute"
    return None

# ── Runner ────────────────────────────────────────────────────────────────────

def run_mission(mission: str) -> str:
    import time as _time
    initial_state: ShifuState = {
        "messages":    [],
        "mission":     mission,
        "plan":        "",
        "complexity":  "",
        "iterations":  0,
        "verdict":     "",
        "retry_count": 0,
    }

    _divider("━")
    print(f"{_BOLD} 🐾  SHIFU  —  Mission received{_RESET}")
    print(f"   {_DIM}{mission}{_RESET}")
    _divider("━")

    prev_state = None
    final_state = None
    t0 = _time.time()

    for step in shifu_graph.stream(initial_state, stream_mode="values"):
        node = _detect_node(prev_state, step)
        if node:
            _header(node)
            printer = _PRINTERS.get(node)
            if printer:
                printer(step)
        prev_state  = step
        final_state = step

    elapsed = _time.time() - t0
    _divider("━", _GREEN)
    print(f"{_GREEN}{_BOLD} ✅  MISSION COMPLETE  ({elapsed:.1f}s){_RESET}")
    _divider("━", _GREEN)

    if not final_state:
        return "No output produced."

    last_ai = next(
        (m for m in reversed(final_state.get("messages", []))
         if isinstance(m, AIMessage) and m.content),
        None,
    )
    return last_ai.content if last_ai else "Mission complete (no textual output)."

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{_BOLD}🐾  Shifu's playground :{_RESET}  {PLAYGROUND_DIR.resolve()}")
    print("━" * W)
    mission = input("Mission ..> ").strip()
    if not mission:
        print("No mission. Shifu meditates.")
    else:
        result = run_mission(mission)
        print()
        print(result)
        print()