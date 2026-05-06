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
from langchain_openai import ChatOpenAI          # ← speaks OpenAI protocol; handles auth correctly
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import importlib
import pkgutil
import tools  # Import your tools package
from langchain_core.tools import BaseTool

def load_all_tools_from_package(package):
    all_tools = []
    
    # Iterate through all modules in the 'tools' folder
    for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        # Import the module dynamically
        module = importlib.import_module(module_name)
        
        # Look for everything in the module that is a LangChain tool
        for name, obj in vars(module).items():
            if isinstance(obj, BaseTool):
                all_tools.append(obj)
                
    return all_tools

load_dotenv()

# ── Playground ────────────────────────────────────────────────────────────────
PLAYGROUND_DIR = Path("Playground")
PLAYGROUND_DIR.mkdir(exist_ok=True)


MODEL_NAME   = "gpt-oss:120b-cloud"          # strip the "ollama/" prefix here
BASE_URL     = "https://ollama.com/v1"        # OpenAI-compat path

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

# ── Tools ─────────────────────────────────────────────────────────────────────

TOOLS = load_all_tools_from_package(tools)
tool_node = ToolNode(TOOLS)
executor_with_tools = executor_llm.bind_tools(TOOLS)

# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are Shifu's strategic mind — a senior planning LLM.

Your job has two phases depending on the flag you receive:

[CLASSIFY]
Decide if the user's mission is SIMPLE or COMPLEX.
- SIMPLE: single-step, one-tool, factual lookup, or trivial file op.
- COMPLEX: multi-step, requires code + search + file ops, or open-ended research.
Reply with exactly one word: SIMPLE or COMPLEX.

[PLAN]
Draft a concise, numbered execution plan for the executor agent.
Rules:
- Each step is a single, atomic action.
- Reference specific tools: web_search, terminal_command, file_write, file_read, directory_read.
- All file paths must live inside Playground/.
- No more than 10 steps. Ruthlessly consolidate.
- End with: "PLAN COMPLETE."
"""

EXECUTOR_SYSTEM = f"""You are Shifu — a single autonomous agent that gets things done.

You have these tools:
  • web_search        — search the internet
  • terminal_command  — run shell commands
  • file_write        — write files (auto-scoped to Playground/)
  • file_read         — read files (auto-scoped to Playground/)
  • directory_read    — list directory contents (auto-scoped to Playground/)

Ground rules:
  1. ALL files you create must go inside  Playground/  (relative paths are auto-scoped).
  2. Follow the execution plan step-by-step if one is provided.
  3. Use tools — do NOT hallucinate file contents or command outputs.
  4. After completing all steps, write a concise final summary starting with "✅ DONE:".
  5. Max 15 tool-call iterations per mission.

Playground path: {PLAYGROUND_DIR.resolve()}
"""

REVIEWER_SYSTEM = """You are Shifu's quality reviewer.

Given the original mission and the agent's final output, decide:
  PASS  — mission accomplished; output is correct and complete.
  RETRY — something is wrong or missing; explain what needs fixing in one sentence.

Reply in exactly this format:
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

MAX_ITERATIONS = 50
MAX_RETRIES    = 10

# ── Node: Classify ────────────────────────────────────────────────────────────

def classify_node(state: ShifuState) -> ShifuState:
    """Planner LLM decides SIMPLE vs COMPLEX."""
    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"[CLASSIFY]\nMission: {state['mission']}"),
    ])
    complexity: Literal["SIMPLE", "COMPLEX"] = (
        "COMPLEX" if "COMPLEX" in resp.content.upper() else "SIMPLE"
    )
    return {**state, "complexity": complexity}

# ── Node: Plan ────────────────────────────────────────────────────────────────

def plan_node(state: ShifuState) -> ShifuState:
    """Planner LLM drafts a step-by-step execution plan (COMPLEX path only)."""
    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=(
            f"[PLAN]\nMission: {state['mission']}\n"
            f"Playground directory: {PLAYGROUND_DIR.resolve()}"
        )),
    ])
    return {**state, "plan": resp.content}

# ── Node: Execute ─────────────────────────────────────────────────────────────

def execute_node(state: ShifuState) -> ShifuState:
    """Executor LLM drives the ReAct tool-call loop."""
    if state["iterations"] >= MAX_ITERATIONS:
        bail = AIMessage(content="⚠️ Max iterations reached. Stopping.")
        return {**state, "messages": state["messages"] + [bail]}

    # Build context message
    if state["complexity"] == "COMPLEX" and state["plan"]:
        user_content = (
            f"Mission: {state['mission']}\n\n"
            f"Execution Plan:\n{state['plan']}\n\n"
            "Carry out the plan step by step using your tools."
        )
    else:
        user_content = f"Mission: {state['mission']}"

    # Prepend system + user turn if first call
    if not state["messages"]:
        msgs: list[BaseMessage] = [
            SystemMessage(content=EXECUTOR_SYSTEM),
            HumanMessage(content=user_content),
        ]
    else:
        msgs = state["messages"]

    response = _rate_limited_invoke(executor_with_tools, msgs)
    return {**state,
            "messages":   state["messages"] + ([] if state["messages"] else [SystemMessage(content=EXECUTOR_SYSTEM), HumanMessage(content=user_content)]) + [response],
            "iterations": state["iterations"] + 1}

# ── Node: Tools ───────────────────────────────────────────────────────────────

def tools_node(state: ShifuState) -> ShifuState:
    """Execute whatever tool calls the executor requested."""
    result = tool_node.invoke({"messages": state["messages"]})
    return {**state, "messages": result["messages"]}

# ── Node: Review ──────────────────────────────────────────────────────────────

def review_node(state: ShifuState) -> ShifuState:
    """Planner LLM reviews the final output (COMPLEX path only)."""
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    agent_output = last_ai.content if last_ai else "(no output)"

    resp = _rate_limited_invoke(planner_llm, [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=(
            f"Mission: {state['mission']}\n\n"
            f"Agent output:\n{agent_output}"
        )),
    ])

    verdict: Literal["PASS", "RETRY"] = (
        "PASS" if "PASS" in resp.content.upper() else "RETRY"
    )
    return {**state, "verdict": verdict, "retry_count": state["retry_count"] + (1 if verdict == "RETRY" else 0)}

# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_classify(state: ShifuState) -> str:
    return "plan" if state["complexity"] == "COMPLEX" else "execute"

def route_after_execute(state: ShifuState) -> str:
    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    # If COMPLEX, go to review; otherwise we're done
    return "review" if state["complexity"] == "COMPLEX" else END

def route_after_tools(state: ShifuState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        return "review" if state["complexity"] == "COMPLEX" else END
    return "execute"

def route_after_review(state: ShifuState) -> str:
    if state["verdict"] == "PASS" or state["retry_count"] >= MAX_RETRIES:
        return END
    return "execute"  # loop back for a retry pass

# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ShifuState)

    g.add_node("classify", classify_node)
    g.add_node("plan",     plan_node)
    g.add_node("execute",  execute_node)
    g.add_node("tools",    tools_node)
    g.add_node("review",   review_node)

    g.set_entry_point("classify")

    g.add_conditional_edges("classify", route_after_classify, {
        "plan":    "plan",
        "execute": "execute",
    })
    g.add_edge("plan", "execute")
    g.add_conditional_edges("execute", route_after_execute, {
        "tools":  "tools",
        "review": "review",
        END:      END,
    })
    g.add_conditional_edges("tools", route_after_tools, {
        "execute": "execute",
        "review":  "review",
        END:       END,
    })
    g.add_conditional_edges("review", route_after_review, {
        "execute": "execute",
        END:       END,
    })

    return g.compile(checkpointer=MemorySaver())

shifu_graph = build_graph()


import textwrap, shutil

W = min(shutil.get_terminal_size().columns, 72)  # respect narrow terminals

# Label config per node  {node_name: (icon, colour_code, label)}
_NODE_META = {
    "classify": ("🔍", "\033[36m",  "CLASSIFYING MISSION"),
    "plan":     ("📋", "\033[35m",  "DRAFTING EXECUTION PLAN"),
    "execute":  ("⚙️ ", "\033[33m",  "EXECUTING"),
    "tools":    ("🔧", "\033[34m",  "TOOL CALL"),
    "review":   ("🔎", "\033[35m",  "REVIEWING OUTPUT"),
}
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_DIM    = "\033[2m"

def _divider(char="━", color=""):
    print(f"{color}{char * W}{_RESET}")

def _header(node: str):
    icon, color, label = _NODE_META.get(node, ("▶", "", node.upper()))
    _divider("─", _DIM)
    print(f"{color}{_BOLD} {icon}  {label}{_RESET}")

def _print_classify(state: ShifuState):
    c = state["complexity"]
    color = _GREEN if c == "SIMPLE" else "\033[35m"
    print(f"   Complexity : {color}{_BOLD}{c}{_RESET}")

def _print_plan(state: ShifuState):
    if not state["plan"]:
        return
    lines = state["plan"].splitlines()
    for line in lines:
        if line.strip():
            print(f"   {_DIM}{line}{_RESET}")

def _print_execute(state: ShifuState):
    if not state["messages"]:
        return
    last = state["messages"][-1]
    # Tool-call request from the executor
    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            name = tc.get("name", "?")
            args = tc.get("args", {})
            # Pretty-print the primary argument (first value, truncated)
            primary = next(iter(args.values()), "") if args else ""
            primary_str = str(primary).replace("\n", " ")[:80]
            if len(str(primary)) > 80:
                primary_str += "…"
            print(f"   {_BOLD}→ {name}{_RESET}  {_DIM}{primary_str}{_RESET}")
    # Plain text thought from the executor
    elif isinstance(last, AIMessage) and last.content:
        snippet = last.content.strip()[:300]
        wrapped = textwrap.fill(snippet, width=W - 5,
                                initial_indent="   ", subsequent_indent="   ")
        print(f"{_DIM}{wrapped}{_RESET}")

def _print_tools(state: ShifuState):
    """Print tool results (ToolMessage objects appended after tool_node runs)."""
    from langchain_core.messages import ToolMessage
    # Find the last batch of ToolMessages (everything after the last AIMessage)
    tool_msgs = []
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage):
            tool_msgs.insert(0, m)
        else:
            break
    for tm in tool_msgs:
        result_str = str(tm.content).strip()
        snippet = result_str[:200].replace("\n", " ")
        if len(result_str) > 200:
            snippet += "…"
        print(f"   {_DIM}↩  {snippet}{_RESET}")

def _print_review(state: ShifuState):
    verdict = state.get("verdict", "")
    retry   = state.get("retry_count", 0)
    if verdict == "PASS":
        print(f"   Verdict    : {_GREEN}{_BOLD}✅ PASS{_RESET}")
    elif verdict == "RETRY":
        print(f"   Verdict    : {_RED}{_BOLD}⚠️  RETRY  (attempt {retry}){_RESET}")
    else:
        print(f"   Verdict    : {_DIM}(pending){_RESET}")

# Map node name → printer
_PRINTERS = {
    "classify": _print_classify,
    "plan":     _print_plan,
    "execute":  _print_execute,
    "tools":    _print_tools,
    "review":   _print_review,
}

def _detect_node(prev: ShifuState | None, curr: ShifuState) -> str | None:
    """
    LangGraph streams full state snapshots; we infer which node just ran by
    comparing what changed between the previous snapshot and the current one.
    """
    if prev is None:
        return "classify"  # first snapshot is always after classify

    # complexity appeared → classify just ran
    if not prev.get("complexity") and curr.get("complexity"):
        return "classify"
    # plan appeared → plan just ran
    if not prev.get("plan") and curr.get("plan"):
        return "plan"
    # verdict appeared → review just ran
    if not prev.get("verdict") and curr.get("verdict"):
        return "review"
    if prev.get("verdict") != curr.get("verdict"):
        return "review"
    # more messages than before → execute or tools ran
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
    """
    Synchronous entry point — mirrors ShifuCrew().crew().kickoff().
    Prints a live checkpoint banner for every graph node as it completes.
    """
    import time as _time
    config = {"configurable": {"thread_id": "shifu-session"}}
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

    prev_state: ShifuState | None = None
    final_state: ShifuState | None = None
    t0 = _time.time()

    for step in shifu_graph.stream(initial_state, config=config, stream_mode="values"):
        node = _detect_node(prev_state, step)
        if node:
            _header(node)
            printer = _PRINTERS.get(node)
            if printer:
                printer(step)
        prev_state  = step
        final_state = step

    # ── Final result banner ───────────────────────────────────────────────────
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


# ── CLI entry point ───────────────────────────────────────────────────────────

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