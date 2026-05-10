"""
automator.py — Shifu Automation Tool  (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Drop in  tools/  alongside every other tool.
Shifu's _load_tools() auto-discovers it at boot.

What this file does
───────────────────
1.  Exposes  create_automation  — a LangChain BaseTool Shifu calls when the
    user wants something done at a future time or condition.

2.  Scans the  tools/  package at import time to build an authoritative tool
    index.  The Automation LLM sees EXACTLY what Shifu sees — no manual list.
    Because this file lives inside tools/ it cannot import shifu.py directly,
    so it re-scans tools/ independently (same pkgutil walk Shifu uses, zero
    logic duplication).

3.  Sends the natural-language instruction to a dedicated Automation LLM
    (uses OLLAMA_API_KEY_AUTOMATION) which structures it into validated JSON.

4.  Writes the result to  .shifu/automations/<id>.yaml  and injects a hot
    memory atom so Shifu can recall "what automations have I set up?" in
    future turns without a cold-memory embedding search.

5.  Touches  .shifu/daemon.reload  to signal the daemon for a hot-reload.

Hot memory injection
────────────────────
After every successful automation creation, a compact atom is pushed into
hot memory:
    "Automation '<name>' created: <trigger summary> → <action chain>"

This lives in the rolling hot-memory window so Shifu knows about it in the
current session without any embedding lookup.  The daemon results file acts
as the longer-term store (read by Shifu on next boot).

Trigger types
─────────────
  schedule   — cron string  ("0 18 * * *")
  delay      — relative     ("in 10 minutes"  →  converted to one-shot cron)
  file_watch — fires when a path appears / changes
  startup    — once when the daemon next starts
  event      — named event, e.g. "user_arrives_home"  (future / extensible)

Template variables in action args
──────────────────────────────────
  {{date}}             current date when the action fires
  {{time}}             current time when the action fires
  {{trigger_value}}    value that fired the trigger (e.g. dropped file path)
  {{previous_result}}  output of the preceding action in the chain
  {{<store_as_name>}}  any name declared in a prior  store_as  field
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Type

import yaml
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT           = Path(__file__).resolve().parent.parent   # one level above tools/
_DATA_DIR       = _ROOT / ".shifu"
_AUTOMATIONS    = _DATA_DIR / "automations"
_RELOAD_FILE    = _DATA_DIR / "daemon.reload"
_HOT_MEM_PATH   = Path(os.getenv("HOT_MEMORY_PATH", str(_DATA_DIR / "hot_memory.json")))

for _p in (_DATA_DIR, _AUTOMATIONS):
    _p.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────

_MODEL_NAME = os.getenv("OLLAMA_MODEL",    "gpt-oss:120b-cloud")
_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
_API_KEY    = os.getenv("OLLAMA_API_KEY_AUTOMATION") or os.getenv("OLLAMA_API_KEY", "ollama")

# ── Tool index ─────────────────────────────────────────────────────────────────

def _build_tool_index() -> str:
    """
    Scan tools/ and return a name + first-line-docstring index.
    Called once at module import — same pkgutil walk as Shifu's _load_tools().
    """
    try:
        # tools/ is our own package — import relative to this file's parent
        import tools as _tools_pkg
    except ImportError:
        return "  (tools/ package not importable — check sys.path)"

    lines: list[str] = []
    seen:  set[str]  = set()

    for _, mod_name, _ in pkgutil.walk_packages(
        _tools_pkg.__path__, _tools_pkg.__name__ + "."
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for obj in vars(mod).values():
            if isinstance(obj, BaseTool) and obj.name not in seen:
                seen.add(obj.name)
                doc = (obj.description or "").strip().splitlines()[0][:90]
                lines.append(f"  {obj.name} — {doc}")

    return "\n".join(sorted(lines)) if lines else "  (no tools found)"


_TOOL_INDEX = _build_tool_index()   # built once at import

# ── Automation LLM system prompt ───────────────────────────────────────────────

def _make_automation_sys() -> str:
    now = datetime.now().isoformat(timespec="seconds")
    return f"""You are an automation planner for an AI personal assistant called Shifu.
Your ONLY job: convert a natural-language automation instruction into a strict JSON object.

Current local datetime: {now}

Output ONLY valid JSON — no markdown fences, no explanation, no preamble.

JSON schema (all fields required unless marked optional):
{{
  "id":   "<slug, lowercase+underscores, max 40 chars, unique>",
  "name": "<human-readable name ≤ 60 chars>",

  "trigger": {{
    "type": "<schedule | delay | file_watch | startup | event>",

    // schedule / delay  →  standard 5-field cron string
    "cron":     "<minute hour dom month dow>",
    "one_shot": true,          // optional — include only for one-time runs

    // file_watch  →  path to watch
    "path":  "<file or directory path>",
    "event": "<created | modified | any>",  // default: any

    // event  →  named event
    "source":    "<location | clipboard | custom>",
    "condition": "<plain-english condition>"
    // (startup has no extra fields)
  }},

  "actions": [
    {{
      "tool":     "<exact tool name from the list below>",
      "args":     {{ ... }},       // use {{{{variable}}}} for template substitution
      "store_as": "<name>"         // optional — stores result for use in later steps
    }}
  ],

  "on_error": "<notify | skip | retry_once>",  // default: notify
  "notify":   "<terminal | file | none>",       // default: terminal
  "enabled":  true
}}

AVAILABLE TOOLS (use exact names from this list):
{_TOOL_INDEX}

CONVERSION RULES:
- "in X minutes/hours"  →  compute absolute cron from {now}, set one_shot: true
- "at HH:MM" no date    →  daily cron  "MM HH * * *",  one_shot omitted (repeating)
- "every day at HH:MM"  →  daily cron, no one_shot
- If action B needs action A's output, set store_as in action A, use {{{{name}}}} in action B args
- id must be slug-safe (lowercase, underscores only, no spaces)
- If the user supplied an id, use it exactly
- Keep action args flat — no nested dicts deeper than 1 level
"""


# ── LLM call ───────────────────────────────────────────────────────────────────

def _call_automation_llm(instruction: str, aid: str) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)
    user_msg = f"Instruction: {instruction}\nRequested id: {aid}"

    resp = client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": _make_automation_sys()},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",       "", raw)
    return json.loads(raw.strip())


# ── Schema validation ──────────────────────────────────────────────────────────

_TRIGGER_TYPES   = {"schedule", "delay", "file_watch", "startup", "event"}
_ON_ERROR_VALUES = {"notify", "skip", "retry_once"}
_NOTIFY_VALUES   = {"terminal", "file", "none"}


def _validate(spec: dict) -> list[str]:
    errors: list[str] = []
    for key in ("id", "name", "trigger", "actions"):
        if key not in spec:
            errors.append(f"Missing required key: '{key}'")
    if errors:
        return errors

    t = spec["trigger"]
    if not isinstance(t, dict):
        errors.append("trigger must be a dict")
    else:
        ttype = t.get("type", "")
        if ttype not in _TRIGGER_TYPES:
            errors.append(f"trigger.type must be one of {_TRIGGER_TYPES}, got {ttype!r}")
        if ttype in ("schedule", "delay") and not t.get("cron"):
            errors.append("trigger.cron required for schedule/delay triggers")
        if ttype == "file_watch" and not t.get("path"):
            errors.append("trigger.path required for file_watch triggers")

    if not isinstance(spec.get("actions"), list) or not spec["actions"]:
        errors.append("actions must be a non-empty list")
    else:
        for i, a in enumerate(spec["actions"]):
            if not isinstance(a, dict) or not a.get("tool"):
                errors.append(f"actions[{i}].tool is required")

    if spec.get("on_error") and spec["on_error"] not in _ON_ERROR_VALUES:
        errors.append(f"on_error must be one of {_ON_ERROR_VALUES}")
    if spec.get("notify") and spec["notify"] not in _NOTIFY_VALUES:
        errors.append(f"notify must be one of {_NOTIFY_VALUES}")

    return errors


# ── YAML writer ────────────────────────────────────────────────────────────────

def _write_yaml(spec: dict) -> Path:
    spec.setdefault("on_error", "notify")
    spec.setdefault("notify",   "terminal")
    spec.setdefault("enabled",  True)
    spec["_created"] = datetime.now().isoformat(timespec="seconds")

    path = _AUTOMATIONS / f"{spec['id']}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return path


# ── Hot memory injection ───────────────────────────────────────────────────────

def _inject_hot_memory(spec: dict):
    """
    Push a compact atom into hot memory so Shifu can recall this automation
    immediately in the current session without any cold-memory lookup.

    Hot memory format: {entries: [{turn, ts, prompt_atoms, response_atoms}]}
    We append a synthetic "response atom" entry.
    """
    try:
        import threading

        # Build the atom string
        t       = spec.get("trigger", {})
        ttype   = t.get("type", "?")
        trigger_summary = (
            f"cron {t['cron']}" if ttype in ("schedule", "delay") and t.get("cron")
            else f"file_watch {t.get('path','?')}" if ttype == "file_watch"
            else ttype
        )
        action_chain = " → ".join(
            a.get("tool", "?") for a in spec.get("actions", [])
        )
        atom = (
            f"Automation '{spec.get('name', spec.get('id', '?'))}' registered: "
            f"{trigger_summary} → {action_chain}"
        )

        # Read current hot memory file
        hot: dict = {"entries": []}
        if _HOT_MEM_PATH.exists():
            try:
                hot = json.loads(_HOT_MEM_PATH.read_text(encoding="utf-8"))
            except Exception:
                hot = {"entries": []}

        entries = hot.get("entries", [])
        max_turn = max((e.get("turn", 0) for e in entries), default=0)

        entries.append({
            "turn":           max_turn + 1,
            "ts":             datetime.now().strftime("%H:%M"),
            "prompt_atoms":   [f"User set up automation: {spec.get('id', '?')}"],
            "response_atoms": [atom],
        })

        # Trim to HOT_MEMORY_MAX_TURNS
        max_turns = int(os.getenv("HOT_MEMORY_MAX_TURNS", "15"))
        if len(entries) > max_turns:
            entries = entries[-max_turns:]

        hot["entries"] = entries
        _HOT_MEM_PATH.write_text(
            json.dumps(hot, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    except Exception:
        pass   # hot memory injection is best-effort — never block tool execution


# ── Daemon ping ────────────────────────────────────────────────────────────────

def _ping_daemon():
    try:
        _RELOAD_FILE.touch()
    except Exception:
        pass


# ── Human-readable confirmation ────────────────────────────────────────────────

def _confirmation(spec: dict, yaml_path: Path) -> str:
    t     = spec["trigger"]
    ttype = t.get("type", "?")

    if ttype in ("schedule", "delay"):
        when = f"cron `{t.get('cron', '?')}`"
        if t.get("one_shot"):
            when += "  (one-shot — runs once then removed)"
    elif ttype == "file_watch":
        when = f"when `{t.get('path','?')}` {t.get('event','changes')}"
    elif ttype == "startup":
        when = "on next daemon startup"
    elif ttype == "event":
        when = f"event: {t.get('condition', t.get('source', '?'))}"
    else:
        when = ttype

    chain = " → ".join(a.get("tool", "?") for a in spec.get("actions", []))

    return (
        f"Automation `{spec['id']}` registered.\n"
        f"  Name    : {spec.get('name', '?')}\n"
        f"  Trigger : {when}\n"
        f"  Actions : {chain}\n"
        f"  File    : {yaml_path}\n"
        f"  The daemon will execute this headlessly. "
        f"Ask me 'what automations are set up?' to review them."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LangChain tool
# ══════════════════════════════════════════════════════════════════════════════

class _AutomationInput(BaseModel):
    instruction: str = Field(
        description=(
            "Full natural-language description of what to do and when/why. "
            "Include tool names if known and any variables that need to be "
            "passed between steps. Examples: "
            "'Create a Google Meet at 6pm and open it in the browser', "
            "'Summarise any PDF dropped in Playground/inbox/ and save to Playground/summaries/', "
            "'Turn off the heater in 20 minutes using smart_home_set'."
        )
    )
    automation_id: str = Field(
        default="",
        description=(
            "Optional slug for the YAML file (lowercase, underscores). "
            "Auto-generated if omitted. Existing file with the same id is overwritten."
        ),
    )


class CreateAutomationTool(BaseTool):
    """
    Declare a deferred or conditional automation from a natural-language instruction.

    Call this when the user wants a tool invoked at a future time or when a
    condition is met — e.g. 'open the meet link at 6pm', 'summarise any PDF
    dropped in inbox/', 'turn the heater off in 20 minutes'.

    A dedicated Automation LLM structures the instruction into a validated YAML
    file that the shifu_daemon process picks up and executes headlessly.
    Shifu does NOT wait for it — the response is immediate.

    After creation the automation is injected into hot memory so you can answer
    'what automations are set up?' instantly without a cold-memory lookup.
    """

    name:          str            = "create_automation"
    description:   str            = (
        "Schedule a tool call or action chain to run at a future time or when "
        "a condition is met. Use for: 'do X at Y time', 'do X every day at Y', "
        "'do X when file appears', 'do X in N minutes'. "
        "Shifu does not wait — the automation runs headlessly via the daemon. "
        "Provide a full natural-language description including which tools to call "
        "and any data that needs to flow between steps."
    )
    args_schema:   Type[BaseModel] = _AutomationInput
    return_direct: bool            = False

    def _run(self, instruction: str, automation_id: str = "") -> str:
        aid = automation_id.strip() or f"auto_{uuid.uuid4().hex[:8]}"

        # 1 — Ask Automation LLM to structure the instruction
        try:
            spec = _call_automation_llm(instruction, aid)
        except json.JSONDecodeError as e:
            return f"[automator] Automation LLM returned invalid JSON: {e}"
        except Exception as e:
            return f"[automator] Automation LLM call failed: {e}"

        # Enforce the requested id (LLM may have drifted)
        spec["id"] = aid

        # 2 — Validate
        errors = _validate(spec)
        if errors:
            return (
                f"[automator] Validation failed ({len(errors)} error(s)):\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

        # 3 — Write YAML
        try:
            yaml_path = _write_yaml(spec)
        except Exception as e:
            return f"[automator] Failed to write YAML: {e}"

        # 4 — Inject into hot memory (best-effort, never blocks)
        _inject_hot_memory(spec)

        # 5 — Signal daemon
        _ping_daemon()

        # 6 — Return human-readable confirmation
        return _confirmation(spec, yaml_path)

    async def _arun(self, instruction: str, automation_id: str = "") -> str:
        import asyncio
        return await asyncio.to_thread(self._run, instruction, automation_id)


# ── Singleton export (auto-discovered by _load_tools) ─────────────────────────

create_automation = CreateAutomationTool()