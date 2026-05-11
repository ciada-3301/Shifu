#!/usr/bin/env python3
"""
shifu_daemon.py — Shifu Automation Daemon
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standalone process.  Spawned by Shifu on boot as a subprocess (or run manually).
Watches .shifu/automations/ for YAML files and executes them headlessly.

How to run
──────────
  python shifu_daemon.py          # foreground (you'll see logs)
  python shifu_daemon.py --quiet  # suppress stdout

How Shifu spawns it automatically
──────────────────────────────────
Add this near the top of shifu.py (after DATA_DIR is defined):

    import subprocess, sys as _sys
    _daemon_proc = subprocess.Popen(
        [_sys.executable, "shifu_daemon.py", "--quiet"],
        start_new_session=True,   # survives Shifu dying
    )

Architecture
────────────
  • APScheduler (BackgroundScheduler) handles cron/delay triggers.
  • watchdog ObserverThread handles file_watch triggers.
  • A polling loop watches .shifu/daemon.reload for hot-reload signals.
  • Each automation YAML maps to an AutomationSpec dataclass.
  • Actions execute by importing the shared tools/ package directly — zero
    tool logic is duplicated.
  • Every run (pass or fail) is appended to .shifu/automation_log.jsonl.

Dependencies
────────────
  pip install apscheduler watchdog pyyaml
  (langchain-core and langchain-openai already required by Shifu)
"""

from __future__ import annotations
from tzlocal import get_localzone
import argparse
import importlib
import json
import logging
import os
import pkgutil
import re
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ── Third-party ────────────────────────────────────────────────────────────────
try:
    import yaml
except ImportError:
    sys.exit("shifu_daemon: pyyaml not installed — run: pip install pyyaml")

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:
    sys.exit("shifu_daemon: apscheduler not installed — run: pip install apscheduler")

try:
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    from watchdog.observers import Observer
except ImportError:
    sys.exit("shifu_daemon: watchdog not installed — run: pip install watchdog")

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR        = Path(".shifu")
AUTOMATIONS_DIR = DATA_DIR / "automations"
LOG_FILE        = DATA_DIR / "automation_log.jsonl"
RESULTS_FILE    = DATA_DIR / "automation_results.jsonl"
RELOAD_SENTINEL = DATA_DIR / "daemon.reload"
PID_FILE        = DATA_DIR / "daemon.pid"

for _p in (DATA_DIR, AUTOMATIONS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────

_log = logging.getLogger("shifu_daemon")


def _setup_logging(quiet: bool):
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="  %(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class ActionSpec:
    tool:     str
    args:     dict[str, Any]     = field(default_factory=dict)
    store_as: Optional[str]      = None


@dataclass
class TriggerSpec:
    type:      str                          # schedule | delay | file_watch | startup | event
    cron:      Optional[str]      = None
    one_shot:  bool               = False
    path:      Optional[str]      = None    # file_watch
    event:     str                = "any"   # file_watch: created | modified | any
    source:    Optional[str]      = None    # event trigger
    condition: Optional[str]      = None    # event trigger


@dataclass
class AutomationSpec:
    id:       str
    name:     str
    trigger:  TriggerSpec
    actions:  list[ActionSpec]
    on_error: str    = "notify"
    notify:   str    = "terminal"
    enabled:  bool   = True
    one_shot: bool   = False    # mirrored from trigger for convenience
    _path:    Optional[Path] = field(default=None, repr=False)


# ── YAML loader ────────────────────────────────────────────────────────────────

def _load_spec(path: Path) -> Optional[AutomationSpec]:
    """Parse a YAML file into an AutomationSpec.  Returns None on parse error."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        _log.warning("Failed to parse %s: %s", path.name, e)
        return None

    if not isinstance(raw, dict):
        _log.warning("%s: top level must be a dict", path.name)
        return None

    # Trigger
    t = raw.get("trigger", {})
    if not isinstance(t, dict):
        _log.warning("%s: trigger must be a dict", path.name)
        return None

    trigger = TriggerSpec(
        type      = t.get("type", ""),
        cron      = t.get("cron"),
        one_shot  = bool(t.get("one_shot", False)),
        path      = t.get("path"),
        event     = t.get("event", "any"),
        source    = t.get("source"),
        condition = t.get("condition"),
    )

    # Actions
    raw_actions = raw.get("actions", [])
    if not isinstance(raw_actions, list):
        _log.warning("%s: actions must be a list", path.name)
        return None

    actions = []
    for a in raw_actions:
        if not isinstance(a, dict) or not a.get("tool"):
            continue
        actions.append(ActionSpec(
            tool     = a["tool"],
            args     = a.get("args", {}),
            store_as = a.get("store_as"),
        ))

    if not actions:
        _log.warning("%s: no valid actions", path.name)
        return None

    return AutomationSpec(
        id       = raw.get("id", path.stem),
        name     = raw.get("name", path.stem),
        trigger  = trigger,
        actions  = actions,
        on_error = raw.get("on_error", "notify"),
        notify   = raw.get("notify",   "terminal"),
        enabled  = bool(raw.get("enabled", True)),
        one_shot = trigger.one_shot,
        _path    = path,
    )


# ── Tools loader ───────────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, Any] = {}   # name → callable (the underlying function)


def _load_tools():
    """
    Import the shared tools/ package and build a name→callable registry.
    Calls the public BaseTool.run() so argument validation still applies.
    """
    global _TOOL_REGISTRY
    _TOOL_REGISTRY = {}

    try:
        import tools as tools_pkg
    except ImportError:
        _log.error("Cannot import tools/ package — is the daemon running from the Shifu root?")
        return

    from langchain_core.tools import BaseTool

    seen_tools: set[str] = set()   # deduplicate across re-exports

    for _, mod_name, _ in pkgutil.walk_packages(tools_pkg.__path__, tools_pkg.__name__ + "."):
        try:
            # Force a fresh load to avoid stale cached modules from automator's
            # own scan at import time (which can leave singletons un-initialized).
            mod = importlib.import_module(mod_name)
            # Reload only tool modules (not automator, to avoid re-entrant LLM calls)
            if "automator" not in mod_name:
                mod = importlib.reload(mod)
        except Exception as e:
            _log.warning("Could not import %s: %s", mod_name, e)
            continue

        for attr_name, obj in vars(mod).items():
            if not isinstance(obj, BaseTool):
                continue
            if obj.name in seen_tools:
                continue
            # Use the public .run() method — goes through LangChain validation.
            # Wrap it so the daemon can call tool_fn(**kwargs) directly.
            _TOOL_REGISTRY[obj.name] = obj._run   # bound to the singleton instance
            seen_tools.add(obj.name)
            _log.info("Registered tool: %s (from %s.%s)", obj.name, mod_name, attr_name)

    _log.info("Loaded %d tools: %s", len(_TOOL_REGISTRY), sorted(_TOOL_REGISTRY))


# ── Template variable substitution ────────────────────────────────────────────

def _resolve_templates(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Replace {{variable}} placeholders in string arg values."""
    resolved = {}
    for k, v in args.items():
        if isinstance(v, str):
            def _sub(m: re.Match) -> str:
                key = m.group(1).strip()
                return str(context.get(key, m.group(0)))   # keep original if not found
            resolved[k] = re.sub(r"\{\{(.+?)\}\}", _sub, v)
        else:
            resolved[k] = v
    return resolved


# ── Action executor ────────────────────────────────────────────────────────────

def _execute_actions(spec: AutomationSpec, trigger_value: str = "") -> dict:
    """
    Run the action chain for one automation.  Returns a result dict.
    """
    context: dict[str, Any] = {
        "date":          datetime.now().strftime("%Y-%m-%d"),
        "time":          datetime.now().strftime("%H:%M"),
        "trigger_value": trigger_value,
        "mission":       spec.name,
    }

    results   = []
    success   = True
    last_result = ""

    for i, action in enumerate(spec.actions):
        tool_fn = _TOOL_REGISTRY.get(action.tool)
        if tool_fn is None:
            msg = f"Tool '{action.tool}' not found in registry"
            _log.warning("[%s] %s", spec.id, msg)
            results.append({"tool": action.tool, "status": "error", "output": msg})
            success = False

            if spec.on_error == "retry_once" and i == 0:
                # Reload tools and try once more
                _load_tools()
                tool_fn = _TOOL_REGISTRY.get(action.tool)

            if tool_fn is None:
                if spec.on_error == "skip":
                    continue
                else:
                    break   # notify / retry_once exhausted — stop chain

        # Resolve template variables
        context["previous_result"] = last_result
        resolved_args = _resolve_templates(action.args, context)

        try:
            _log.info("[%s] Calling %s(%s)", spec.id, action.tool,
                      str(resolved_args)[:80])
            output = tool_fn(**resolved_args)
            last_result = str(output) if output is not None else ""

            # Store result under store_as name if declared
            if action.store_as:
                context[action.store_as] = last_result

            results.append({"tool": action.tool, "status": "ok", "output": last_result[:500]})
            _log.info("[%s] %s → %s", spec.id, action.tool, last_result[:80])

        except Exception as exc:
            tb  = traceback.format_exc()
            msg = str(exc)
            _log.error("[%s] %s raised: %s", spec.id, action.tool, msg)
            results.append({"tool": action.tool, "status": "error", "output": msg, "traceback": tb})
            success = False

            if spec.on_error == "retry_once":
                try:
                    output = tool_fn(**resolved_args)
                    last_result = str(output) if output is not None else ""
                    if action.store_as:
                        context[action.store_as] = last_result
                    results[-1] = {"tool": action.tool, "status": "ok (retry)", "output": last_result[:500]}
                    success = True
                    _log.info("[%s] %s succeeded on retry", spec.id, action.tool)
                except Exception as exc2:
                    results[-1]["retry_error"] = str(exc2)

            if spec.on_error == "skip":
                continue
            elif not success:
                break   # stop chain on error unless skip

    return {
        "automation_id": spec.id,
        "name":          spec.name,
        "success":       success,
        "actions":       results,
        "context":       {k: str(v)[:200] for k, v in context.items()},
    }


# ── Logging ────────────────────────────────────────────────────────────────────

def _log_run(spec: AutomationSpec, result: dict):
    """Append run result to .shifu/automation_log.jsonl and results file."""
    entry = {
        "ts":      datetime.now().isoformat(timespec="seconds"),
        **result,
    }
    line = json.dumps(entry) + "\n"

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line)

    # Results file is what Shifu reads on next boot to summarise
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(line)


def _notify(spec: AutomationSpec, result: dict):
    """Notify on terminal if configured (daemon may be backgrounded — best effort)."""
    if spec.notify != "terminal":
        return
    status = "✓" if result["success"] else "✗"
    ts     = datetime.now().strftime("%H:%M:%S")
    print(f"\n  [{ts}] automation {status}  {spec.name}")
    for a in result.get("actions", []):
        mark = "✓" if a["status"].startswith("ok") else "✗"
        print(f"    {mark} {a['tool']}  {str(a.get('output',''))[:80]}")


# ── Run wrapper ────────────────────────────────────────────────────────────────

def _run_automation(spec: AutomationSpec, trigger_value: str = ""):
    """Top-level entry point called by scheduler / watchdog."""
    _log.info("Running automation: %s", spec.id)
    result = _execute_actions(spec, trigger_value)
    _log_run(spec, result)
    _notify(spec, result)

    # One-shot: remove YAML after first run
    if spec.one_shot and spec._path and spec._path.exists():
        try:
            spec._path.unlink()
            _log.info("One-shot automation %s: YAML removed", spec.id)
        except Exception as e:
            _log.warning("Could not remove one-shot YAML %s: %s", spec._path, e)


# ── Daemon ─────────────────────────────────────────────────────────────────────

class ShifuDaemon:
    def __init__(self):
        self._scheduler   = BackgroundScheduler(timezone="Asia/Kolkata")
        self._observer    = Observer()
        self._specs:      dict[str, AutomationSpec] = {}   # id → spec
        self._reload_mtime: float = 0.0

    # ── YAML registry ──────────────────────────────────────────────────────────

    def _scan_automations(self) -> dict[str, AutomationSpec]:
        specs = {}
        for yaml_path in sorted(AUTOMATIONS_DIR.glob("*.yaml")):
            spec = _load_spec(yaml_path)
            if spec and spec.enabled:
                specs[spec.id] = spec
        return specs

    def _reload_all(self):
        """Hot-reload: diff current specs against running ones and update scheduler."""
        new_specs = self._scan_automations()

        # Remove automations that no longer exist or were disabled
        for aid in list(self._specs.keys()):
            if aid not in new_specs:
                self._unregister(aid)

        # Add / update automations
        for aid, spec in new_specs.items():
            self._register(spec)

        self._specs = new_specs
        _log.info("Reloaded automations: %d active", len(self._specs))

    # ── Scheduler registration ─────────────────────────────────────────────────

    def _register(self, spec: AutomationSpec):
        """Register or re-register a spec with APScheduler / watchdog."""
        # Remove old job if exists
        self._unregister(spec.id)

        t = spec.trigger

        if t.type in ("schedule", "delay") and t.cron:
            try:
                parts = t.cron.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid cron '{t.cron}'")
                minute, hour, dom, month, dow = parts
                cron_trigger = CronTrigger(
                    minute=minute, hour=hour,
                    day=dom, month=month, day_of_week=dow,
                    timezone="Asia/Kolkata",
                )
                self._scheduler.add_job(
                    _run_automation,
                    trigger=cron_trigger,
                    args=[spec],
                    id=spec.id,
                    replace_existing=True,
                    name=spec.name,
                )
                _log.info("Scheduled '%s' at cron %s", spec.id, t.cron)
            except Exception as e:
                _log.error("Failed to schedule '%s': %s", spec.id, e)

        elif t.type == "startup":
            # Run immediately (daemon just started or hot-reloaded)
            _log.info("Startup automation '%s': running now", spec.id)
            _run_automation(spec)

        elif t.type == "file_watch":
            # Handled by watchdog ObserverThread (see _FileWatchHandler)
            _log.info("File-watch automation '%s' watching %s", spec.id, t.path)
            # The observer handler already has a reference to self._specs;
            # re-registering just updates the dict (done above).

        elif t.type == "event":
            # Future: location/clipboard/custom events — no-op for now
            _log.info("Event automation '%s' registered (handler not yet implemented)", spec.id)

    def _unregister(self, aid: str):
        try:
            if self._scheduler.get_job(aid):
                self._scheduler.remove_job(aid)
        except Exception:
            pass

    # ── File-watch handler ─────────────────────────────────────────────────────

    def _make_file_watch_handler(self) -> "FileSystemEventHandler":
        specs_ref = self._specs   # live reference

        class _Handler(FileSystemEventHandler):
            def on_any_event(self_, ev: FileSystemEvent):
                # Check if any file_watch automation matches this path
                for spec in list(specs_ref.values()):
                    if spec.trigger.type != "file_watch":
                        continue
                    watch_path = spec.trigger.path
                    if not watch_path:
                        continue
                    watch_path = Path(watch_path)
                    ev_path    = Path(ev.src_path)

                    # Match: exact file, or any file under a watched directory
                    matches = (
                        ev_path == watch_path or
                        (watch_path.is_dir() and watch_path in ev_path.parents)
                    )
                    if not matches:
                        continue

                    ev_type = spec.trigger.event
                    if ev_type == "created"  and ev.event_type != "created":  continue
                    if ev_type == "modified" and ev.event_type != "modified": continue
                    # "any" matches everything

                    _run_automation(spec, trigger_value=str(ev_path))

        return _Handler()

    # ── Hot-reload poller ──────────────────────────────────────────────────────

    def _check_reload_sentinel(self):
        """Called in the main loop.  Reloads if sentinel was touched."""
        if not RELOAD_SENTINEL.exists():
            return
        mtime = RELOAD_SENTINEL.stat().st_mtime
        if mtime > self._reload_mtime:
            self._reload_mtime = mtime
            _log.info("Reload sentinel detected — hot-reloading automations")
            self._reload_all()

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        _log.info("Shifu daemon starting — PID %d", os.getpid())

        # Write PID file so Shifu can check if daemon is running
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

        # Load tools
        _load_tools()

        # Initial scan
        self._reload_all()

        # Start APScheduler
        self._scheduler.start()
        _log.info("APScheduler started")

        # Start watchdog observer (watches automations dir for YAML changes
        # AND handles file_watch triggers for user-specified paths)
        handler = self._make_file_watch_handler()
        # Watch the automations directory for YAML hot-reload
        self._observer.schedule(
            _AutomationsWatcher(self),
            str(AUTOMATIONS_DIR),
            recursive=False,
        )
        # Also register the file-watch handler for any path watchdog can observe
        # We watch the working directory broadly; the handler filters by spec.path
        self._observer.schedule(handler, ".", recursive=True)
        self._observer.start()
        _log.info("File-watch observer started")

        try:
            while True:
                time.sleep(2)
                self._check_reload_sentinel()
        except KeyboardInterrupt:
            _log.info("Shutdown signal received")
        finally:
            self._observer.stop()
            self._observer.join()
            self._scheduler.shutdown(wait=False)
            try:
                PID_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            _log.info("Shifu daemon stopped")


class _AutomationsWatcher(FileSystemEventHandler):
    """Watches .shifu/automations/ for new/changed YAMLs and triggers reload."""

    def __init__(self, daemon: ShifuDaemon):
        self._daemon = daemon
        self._last   = 0.0

    def on_any_event(self, ev: FileSystemEvent):
        if not str(ev.src_path).endswith(".yaml"):
            return
        now = time.time()
        if now - self._last < 0.5:   # debounce
            return
        self._last = now
        _log.info("YAML change detected: %s", ev.src_path)
        self._daemon._reload_all()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Shifu Automation Daemon")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress INFO logs (only warnings and errors)")
    args = parser.parse_args()

    _setup_logging(quiet=args.quiet)
    ShifuDaemon().run()


if __name__ == "__main__":
    main()