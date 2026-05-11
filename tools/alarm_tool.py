"""
tools/alarm.py — Alarm tool for Shifu
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Drop in tools/ alongside every other tool.
Auto-discovered by _load_tools() at boot, and also by automator.py's
_build_tool_index() — so Shifu knows it can schedule alarms via create_automation.

How the full alarm flow works
──────────────────────────────
  1.  User says "set an alarm for 7:30 AM called Morning Standup"
  2.  Shifu calls create_automation with the alarm instruction.
  3.  automator.py writes a YAML like:
        trigger:  { type: schedule, cron: "30 7 * * *" }
        actions:  [ { tool: trigger_alarm, args: { label: "Morning Standup" } } ]
  4.  shifu_daemon.py picks up the YAML and at 7:30 AM calls trigger_alarm().
  5.  trigger_alarm() launches alarm_gui.py as a subprocess — a Tkinter popup
      appears on screen with the alarm label, time, snooze, and dismiss.

trigger_alarm is the action tool used in YAML.
set_alarm is a convenience wrapper Shifu may call directly for immediate
scheduling (it internally calls create_automation).

YAML usage (daemon calls this directly)
────────────────────────────────────────
  actions:
    - tool: trigger_alarm
      args:
        label: "Morning Standup"
        sound: true          # optional, default true

Debugging
─────────
  If the popup doesn't appear, check: .shifu/alarm_errors.log
  That file captures stderr from every alarm_gui.py subprocess launch.
  Common causes logged there:
    - "No module named 'tkinter'"  → sudo apt install python3-tk
    - "no display name"            → DISPLAY env var not set (Linux/WSL)
    - any other crash traceback    → something in alarm_gui.py itself
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# ── Paths ──────────────────────────────────────────────────────────────────────
# alarm_gui.py sits alongside shifu.py in the project root (one level above tools/)
_ROOT        = Path(__file__).resolve().parent.parent
_GUI_SCRIPT  = _ROOT / "alarm_gui.py"
_DATA_DIR    = _ROOT / ".shifu"
_ERR_LOG     = _DATA_DIR / "alarm_errors.log"   # stderr from every GUI subprocess


def _get_launch_env() -> dict:
    """
    Build an environment dict for the alarm_gui.py subprocess.

    On Linux/WSL the daemon may have been started in a session that inherited
    DISPLAY from the user's terminal — but because we use start_new_session=True
    the env is still inherited, so we just pass it through explicitly to be safe.
    If DISPLAY is missing entirely (headless / pure SSH) we attempt ':0' as a
    last-ditch guess; the error log will make it obvious if that also fails.
    """
    env = os.environ.copy()
    if os.name != "nt":
        if not env.get("DISPLAY"):
            env["DISPLAY"] = ":0"           # best-guess fallback for Linux
        # WSLg exposes WAYLAND_DISPLAY; keep it if present
    return env


def _open_err_log():
    """Open the error log in append mode, creating parent dirs if needed."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return open(_ERR_LOG, "a", encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  trigger_alarm  — the action the DAEMON calls
# ══════════════════════════════════════════════════════════════════════════════

class _TriggerAlarmInput(BaseModel):
    label: str = Field(
        default="Alarm",
        description="Short human-readable label shown on the alarm popup."
    )
    sound: bool = Field(
        default=True,
        description="Whether to play a sound when the alarm fires."
    )
    message: str = Field(
        default="",
        description="Optional extra message displayed on the popup."
    )


class TriggerAlarmTool(BaseTool):
    """
    Fire an alarm popup immediately.

    This is the ACTION called by the daemon when a scheduled alarm triggers.
    It launches alarm_gui.py as a detached subprocess so the popup appears
    even if Shifu's main terminal is in the background.

    Do NOT call this directly to SET an alarm — use create_automation with a
    natural-language instruction like 'set an alarm for 7 AM every weekday'.
    Call this tool only when you want an alarm to fire RIGHT NOW.
    """

    name:          str             = "trigger_alarm"
    description:   str             = (
        "Fire an alarm popup window immediately. "
        "Used by the daemon as the action step of a scheduled alarm automation. "
        "Provide a label (shown on popup) and optional message. "
        "To SCHEDULE an alarm for a future time, use create_automation instead."
    )
    args_schema:   Type[BaseModel] = _TriggerAlarmInput
    return_direct: bool            = False

    def _run(
        self,
        label:   str  = "Alarm",
        sound:   bool = True,
        message: str  = "",
    ) -> str:
        if not _GUI_SCRIPT.exists():
            return (
                f"[alarm] alarm_gui.py not found at {_GUI_SCRIPT}. "
                "Make sure alarm_gui.py is in the Shifu project root."
            )

        cmd = [
            sys.executable,
            str(_GUI_SCRIPT),
            "--label",   label,
            "--message", message or "",
            "--sound",   "default" if sound else "silent",
        ]

        try:
            err_log = _open_err_log()

            # Write a timestamped header so each launch attempt is easy to find
            err_log.write(
                f"\n── alarm launch {datetime.now().isoformat(timespec='seconds')} "
                f"label='{label}' ──\n"
            )
            err_log.flush()

            if os.name == "nt":
                # Windows: DO NOT use DETACHED_PROCESS for a GUI app.
                # DETACHED_PROCESS detaches from the console, which is correct
                # for CLI tools, but on some Windows configurations it also
                # strips the process's access to the interactive window station,
                # which prevents Tkinter from creating any windows at all.
                # CREATE_NEW_PROCESS_GROUP alone is enough to stop Ctrl-C
                # propagation without touching the window station.
                kwargs: dict = {
                    "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
                    "stdin":  subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL,
                    "stderr": err_log,      # capture crash tracebacks
                    "env":    os.environ.copy(),
                }
            else:
                # Linux / macOS: pass the full environment explicitly so DISPLAY
                # (X11) and WAYLAND_DISPLAY survive into the child process even
                # though it runs in a new session.
                kwargs = {
                    "start_new_session": True,
                    "stdin":  subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL,
                    "stderr": err_log,      # capture crash tracebacks
                    "env":    _get_launch_env(),
                }

            proc = subprocess.Popen(cmd, **kwargs)

            return (
                f"[alarm] GUI launched — PID {proc.pid}, label: '{label}', "
                f"time: {datetime.now().strftime('%H:%M')}. "
                f"If nothing appears, check: {_ERR_LOG}"
            )

        except Exception as e:
            return f"[alarm] Failed to launch alarm_gui.py: {e}"

    async def _arun(
        self,
        label:   str  = "Alarm",
        sound:   bool = True,
        message: str  = "",
    ) -> str:
        import asyncio
        return await asyncio.to_thread(self._run, label, sound, message)


# ══════════════════════════════════════════════════════════════════════════════
#  set_alarm  — convenience tool Shifu can call to schedule via automator
# ══════════════════════════════════════════════════════════════════════════════

class _SetAlarmInput(BaseModel):
    label: str = Field(
        description="Short name for the alarm, e.g. 'Morning Standup', 'Medication', 'Tea timer'."
    )
    when: str = Field(
        description=(
            "When the alarm should fire. Natural language is fine: "
            "'at 7:30 AM', 'in 20 minutes', 'every day at 8 PM', "
            "'weekdays at 9 AM'. The automator will convert this to a cron schedule."
        )
    )
    message: str = Field(
        default="",
        description="Optional extra message shown on the alarm popup."
    )
    sound: bool = Field(
        default=True,
        description="Whether to play a sound when the alarm fires."
    )


class SetAlarmTool(BaseTool):
    """
    Schedule an alarm for a future time or recurring schedule.

    Shifu calls this when the user wants an alarm set — e.g.:
      'set an alarm for 7:30 AM'
      'remind me in 20 minutes'
      'alarm every weekday at 9 AM called standup'

    Internally this calls create_automation to write the YAML that the daemon
    will execute.  The daemon then calls trigger_alarm when the time comes,
    which pops up the alarm_gui.py window.

    Prefer this over calling create_automation manually for alarm use-cases —
    it sets sensible defaults (one_shot for absolute times, repeating for
    'every day' schedules).
    """

    name:          str             = "set_alarm"
    description:   str             = (
        "Schedule an alarm to fire at a future time or on a recurring schedule. "
        "Use when the user says 'set an alarm', 'remind me at X', 'wake me up at Y', "
        "or 'alarm every Z'. Provide label (name) and when (natural-language time). "
        "The alarm pops up a GUI window when it fires. "
        "For immediate firing use trigger_alarm instead."
    )
    args_schema:   Type[BaseModel] = _SetAlarmInput
    return_direct: bool            = False

    def _run(
        self,
        label:   str  = "Alarm",
        when:    str  = "",
        message: str  = "",
        sound:   bool = True,
    ) -> str:
        if not when.strip():
            return "[alarm] 'when' is required — e.g. 'at 7:30 AM' or 'in 20 minutes'."

        # Build a natural-language instruction for create_automation
        sound_note  = "with sound" if sound else "silently (no sound)"
        msg_note    = f" with message '{message}'" if message else ""
        instruction = (
            f"Set an alarm called '{label}'{msg_note} to fire {when} {sound_note}. "
            f"Use the trigger_alarm tool with args: label='{label}', "
            f"sound={'true' if sound else 'false'}, message='{message}'."
        )

        # Delegate to create_automation
        try:
            from tools.automator import create_automation  # type: ignore[import]
        except ImportError:
            return (
                "[alarm] create_automation tool not found. "
                "Make sure automator.py is in the tools/ package."
            )

        slug = "alarm_" + label.lower().replace(" ", "_")[:28]
        return create_automation._run(instruction=instruction, automation_id=slug)

    async def _arun(
        self,
        label:   str  = "Alarm",
        when:    str  = "",
        message: str  = "",
        sound:   bool = True,
    ) -> str:
        import asyncio
        return await asyncio.to_thread(self._run, label, when, message, sound)


# ── Singleton exports (auto-discovered by _load_tools) ────────────────────────

trigger_alarm = TriggerAlarmTool()
set_alarm     = SetAlarmTool()