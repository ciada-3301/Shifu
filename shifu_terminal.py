#!/usr/bin/env python3
"""
shifu_terminal.py
─────────────────
Terminal UI for the LangGraph Shifu agent.
Attach this to shifu.py — it never touches the agent internals.

Design: ink-on-dark-paper. One accent colour (amber). No rainbow.
Every pixel earns its place.
"""

import os, sys, time, threading, textwrap, re, shutil
from datetime import datetime
from pathlib import Path
import subprocess

# ── palette ───────────────────────────────────────────────────────────────────
class C:
    R       = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    # ONE accent: warm amber
    A       = "\033[38;5;214m"   # amber
    AB      = "\033[1;38;5;214m" # amber bold
    # neutrals
    W       = "\033[38;5;252m"   # near-white text
    G       = "\033[38;5;240m"   # mid grey (secondary)
    GD      = "\033[38;5;236m"   # dark grey (borders)
    # semantic
    OK      = "\033[38;5;71m"    # muted green
    ERR     = "\033[38;5;167m"   # muted red
    TOOL    = "\033[38;5;67m"    # steel blue for tool names
    PLAN    = "\033[38;5;139m"   # muted violet for plan steps
    # box-drawing bg for response panel
    PANELBG = "\033[48;5;234m"   # very dark bg
    RBG     = "\033[49m"

def tw() -> int:
    try:    return min(shutil.get_terminal_size().columns, 110)
    except: return 90

def W_inner() -> int:   # inner width of response panel
    return tw() - 6

# ── primitives ────────────────────────────────────────────────────────────────
def blank(n=1):
    print("\n" * (n - 1))

def dim_line(ch="─"):
    w = tw() - 4
    sys.stdout.write("  " + C.GD + ch * w + C.R + "\n")

def _write(s: str):
    sys.stdout.write(s); sys.stdout.flush()

# ── logo ──────────────────────────────────────────────────────────────────────
_LOGO = [
    "  ░░░   ░░  ░░  ░░  ░░░░░  ░░   ░░",
    "  ▒▒▒   ▒▒  ▒▒  ▒▒  ▒▒     ▒▒   ▒▒",
    "  ▓███  ▓▓▓▓▓▓  ▓▓  ▓▓▓▓▓  ▓▓   ▓▓",
    "    ▓▓  ██  ██  ██  ██       ██ ██ ",
    "  ████  ██  ██  ██  ██        ███  ",
]

_LOGO_ART = r"""
      _____ _    _ _____  ______ _    _
     / ____| |  | |_   _||  ____| |  | |
    | (___ | |__| | | |  | |__  | |  | |
     \___ \|  __  | | |  |  __| | |  | |
     ____) | |  | |_| |_ | |    | |__| |
    |_____/|_|  |_|_____||_|     \____/
"""

def boot():
    os.system("cls" if os.name == "nt" else "clear")
    blank()
    for line in _LOGO_ART.strip("\n").split("\n"):
        _write("  " + C.AB + line + C.R + "\n")
        time.sleep(0.018)
    blank()
    ts = datetime.now().strftime("%A, %d %B %Y  ·  %H:%M")
    _write("  " + C.G + ts + C.R + "\n")
    _write("  " + C.GD + "─" * (tw() - 4) + C.R + "\n")
    _write("  " + C.G + "help  ·  history  ·  exit" + C.R + "\n")
    blank()


# ── live checkpoint strip ─────────────────────────────────────────────────────
# Printed as a single compact line that updates in-place until the node
# completes, then gets committed to a permanent summary line.

_TOOL_ICONS = {
    "web_search":       ("⌕", "search"),
    "file_write":       ("↓", "write"),
    "file_read":        ("↑", "read"),
    "directory_read":   ("⊞", "ls"),
    "terminal_command": ("$", "shell"),
    "load_skill":       ("◈", "skill"),   # ← progressive disclosure
}

class CheckpointBar:
    """
    Manages a live one-line status bar + a committed checkpoint trail.
    Call .node(name) when a graph node begins.
    Call .tool(name, arg) when a tool fires.
    Call .done() to clear the live line.
    """
    _NODE_LABELS = {
        "classify": ("·", "classifying"),
        "plan":     ("◈", "planning"),
        "execute":  ("▸", "thinking"),
        "tools":    ("⚙", "tool"),
        "review":   ("◎", "reviewing"),
    }

    def __init__(self):
        self._t0      = time.time()
        self._live    = False   # is a live line currently showing?
        self._lock    = threading.Lock()
        self._committed: list[str] = []

    def _elapsed(self) -> str:
        return f"{time.time() - self._t0:5.1f}s"

    def _clear_live(self):
        if self._live:
            _write("\r" + " " * (tw() - 2) + "\r")
            self._live = False

    def node(self, name: str):
        with self._lock:
            sym, label = self._NODE_LABELS.get(name, ("·", name))
            self._clear_live()
            line = (f"  {C.G}{self._elapsed()}{C.R}  "
                    f"{C.A}{sym}{C.R}  {C.W}{label}{C.G} …{C.R}")
            _write(line)
            self._live = True

    def tool(self, name: str, arg: str = ""):
        with self._lock:
            sym, label = _TOOL_ICONS.get(name, ("·", name))
            snippet = arg.replace("\n", " ")[:50]
            if len(arg) > 50: snippet += "…"
            self._clear_live()
            line = (f"  {C.G}{self._elapsed()}{C.R}  "
                    f"{C.TOOL}{sym}  {label}{C.R}  {C.GD}{snippet}{C.R}")
            _write(line)
            self._live = True

    def tool_result(self, name: str, result: str):
        """Commit a completed tool call as a permanent checkpoint line."""
        with self._lock:
            self._clear_live()
            sym, label = _TOOL_ICONS.get(name, ("·", name))
            snippet = result.replace("\n", " ")[:60]
            if len(result) > 60: snippet += "…"
            line = (f"  {C.G}{self._elapsed()}{C.R}  "
                    f"{C.OK}✓{C.R}  {C.TOOL}{label}{C.R}  {C.GD}{snippet}{C.R}\n")
            _write(line)
            self._committed.append(line)

    def verdict(self, v: str):
        with self._lock:
            self._clear_live()
            if v == "PASS":
                _write(f"  {C.G}{self._elapsed()}{C.R}  {C.OK}◉  pass{C.R}\n")
            else:
                _write(f"  {C.G}{self._elapsed()}{C.R}  {C.ERR}◉  retry{C.R}\n")

    def done(self):
        with self._lock:
            self._clear_live()

    def complexity(self, c: str):
        with self._lock:
            self._clear_live()
            col = C.G if c == "SIMPLE" else C.A
            _write(f"  {C.G}{self._elapsed()}{C.R}  {col}◈  {c.lower()}{C.R}\n")

    def plan_step(self, line: str):
        with self._lock:
            self._clear_live()
            clean = line.strip()
            if clean:
                _write(f"  {C.G}       {C.R}  {C.PLAN}›{C.R}  {C.G}{clean[:70]}{C.R}\n")


# ── idle spinner (shown when no checkpoints for > 1.5s) ──────────────────────
class IdleSpinner:
    _FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, bar: CheckpointBar):
        self._bar  = bar
        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._run, daemon=True)

    def start(self): self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1)
        _write("\r" + " " * 24 + "\r")

    def _run(self):
        import itertools
        for f in itertools.cycle(self._FRAMES):
            if self._stop.is_set(): break
            _write(f"\r  {C.GD}{f}{C.R}  ")
            sys.stdout.flush()
            time.sleep(0.09)


# ── response renderer ─────────────────────────────────────────────────────────
# Rule: put the answer inside a clean box. Markdown rendered inline.
# Code blocks get a subtle dark panel. No colour fireworks.

def _inline_md(t: str) -> str:
    """Render inline markdown: bold, italic, inline-code."""
    t = re.sub(r'`([^`\n]+)`',
               C.PANELBG + C.A + r' \1 ' + C.RBG + C.W, t)
    t = re.sub(r'\*\*(.+?)\*\*',
               C.BOLD + C.W + r'\1' + C.R + C.W, t)
    t = re.sub(r'\*(.+?)\*',
               C.ITALIC + r'\1' + C.R + C.W, t)
    return t

def _render_code_block(lang: str, code: str, inner_w: int):
    lang  = (lang or "code").strip() or "code"
    label = f" {lang} "
    bar   = "─" * max(0, inner_w - len(label) - 1)
    _write("  " + C.GD + "╭─" + C.G + label + C.GD + bar + C.R + "\n")
    for line in code.split("\n"):
        padded = line + " " * max(0, inner_w - len(line) - 2)
        _write("  " + C.GD + "│" + C.R + C.PANELBG + " " + C.G + padded + C.RBG + "\n")
    _write("  " + C.GD + "╰" + "─" * inner_w + C.R + "\n")

def render_response(text: str, elapsed: float):
    """Render the final answer inside a clean box."""
    iw   = W_inner()
    cols = tw()

    blank()
    # ── top bar ──
    ts   = datetime.now().strftime("%H:%M:%S")
    tag  = f" shifu  {ts}  {elapsed:.1f}s "
    border_r = "─" * max(0, iw - len(tag) + 2)
    _write("  " + C.A + "┌" + C.AB + tag + C.A + border_r + "┐" + C.R + "\n")

    # ── body ──
    lines = str(text).split("\n")
    i = 0
    PAD = "    "

    def box_line(content: str):
        _write("  " + C.A + "│" + C.R + "  " + content + "\n")

    def wrap_and_box(raw: str, prefix=""):
        available = iw - len(prefix)
        for wl in textwrap.wrap(raw, width=max(40, available)) or [""]:
            box_line(prefix + C.W + _inline_md(wl) + C.R)

    while i < len(lines):
        ln = lines[i]

        # fenced code block
        m = re.match(r'^```(\w*)', ln)
        if m:
            lang, body = m.group(1), []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i]); i += 1
            # emit code block inline inside the box
            _write("  " + C.A + "│" + C.R + "\n")
            _render_code_block(lang, "\n".join(body), iw - 2)
            _write("  " + C.A + "│" + C.R + "\n")
            i += 1; continue

        if ln.strip() == "":
            box_line(""); i += 1; continue

        # headings
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

        # horizontal rule
        if re.match(r'^[-*_]{3,}\s*$', ln):
            box_line(C.GD + "─" * (iw - 4) + C.R)
            i += 1; continue

        # blockquote
        if ln.startswith("> "):
            box_line(C.GD + "▌ " + C.R + C.ITALIC + C.G + _inline_md(ln[2:]) + C.R)
            i += 1; continue

        # bullet
        bm = re.match(r'^(\s*)[-*+] (.+)', ln)
        if bm:
            lvl = len(bm.group(1)) // 2
            dot = ["◆", "◇", "·"][min(lvl, 2)]
            col = [C.A, C.G, C.GD][min(lvl, 2)]
            wrap_and_box(bm.group(2), "  " * lvl + col + dot + " " + C.W)
            i += 1; continue

        # numbered list
        nm = re.match(r'^(\s*)(\d+)\. (.+)', ln)
        if nm:
            lvl = len(nm.group(1)) // 2
            wrap_and_box(nm.group(3), "  " * lvl + C.A + nm.group(2) + ". " + C.W)
            i += 1; continue

        # normal paragraph
        wrap_and_box(ln)
        i += 1

    # ── bottom bar ──
    _write("  " + C.A + "└" + "─" * (iw + 2) + "┘" + C.R + "\n")
    blank()


# ── session history ───────────────────────────────────────────────────────────
_history: list[dict] = []

def show_history():
    if not _history:
        _write("  " + C.G + "no missions yet." + C.R + "\n"); return
    blank()
    dim_line()
    _write("  " + C.AB + "history" + C.R + "\n")
    dim_line()
    for i, e in enumerate(_history, 1):
        q = e["q"][:tw() - 20] + ("…" if len(e["q"]) > tw() - 20 else "")
        _write(f"  {C.G}{i:02d}  {e['t']}{C.R}  {C.W}{q}{C.R}\n")
    blank()

def show_skills():
    """List all skills currently installed in the skills/ directory."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        import shifu as _s
        skills = _s._scan_skills()
    except Exception as e:
        _write("  " + C.ERR + f"⚠  could not read skills: {e}" + C.R + "\n")
        return

    blank(); dim_line()
    _write("  " + C.AB + "installed skills" + C.R + "\n"); dim_line()

    if not skills:
        _write("  " + C.G + "no skills installed yet.\n" + C.R)
        _write("  " + C.G + "create  skills/<name>/SKILL.md  to add one.\n" + C.R)
    else:
        for name, meta in skills.items():
            tags = f"  {C.GD}[{meta['tags']}]{C.R}" if meta['tags'] else ""
            _write(f"  {C.A}◈{C.R}  {C.W}{name:<22}{C.R}  {C.G}{meta['description']}{C.R}{tags}\n")
        _write(f"\n  {C.GD}skills dir → {_s.SKILLS_DIR.resolve()}{C.R}\n")
    blank()


def show_help():
    blank(); dim_line()
    _write("  " + C.AB + "commands" + C.R + "\n"); dim_line()
    for cmd, desc in [
        ("help / ?",    "this panel"),
        ("history",     "mission log"),
        ("skills",      "list installed skills"),
        ("files",       "supported file types"),
        ("clear / cls", "reset screen"),
        ("exit / quit", "shutdown"),
        ("<anything>",  "send to shifu"),
    ]:
        _write(f"  {C.A}{cmd:<16}{C.R}{C.G}{desc}{C.R}\n")
    blank()

def show_file_support():
    blank(); dim_line()
    _write("  " + C.AB + "supported file types" + C.R + "\n"); dim_line()
    types = [
        ("plain text",  ".txt .md .csv .json .yaml .toml .xml .log",   C.OK),
        ("source code", ".py .js .ts .sh .bat .html .css .sql …",      C.OK),
        ("PDF",         "requires  pip install pypdf",                  C.A),
        ("Word .docx",  "requires  pip install python-docx",            C.A),
        ("Excel .xlsx", "requires  pip install openpyxl",               C.A),
        ("images",      "not supported natively (no vision model)",     C.ERR),
        ("audio/video", "not supported",                                C.ERR),
    ]
    for kind, detail, col in types:
        _write(f"  {col}{'●'}{C.R}  {C.W}{kind:<14}{C.R}  {C.G}{detail}{C.R}\n")
    blank()
    _write("  " + C.G + "tip: to give shifu a file just include the path in your mission.\n"
           "       e.g.  »  summarise Playground/report.pdf\n" + C.R)
    blank()


# ── input prompt ──────────────────────────────────────────────────────────────
def get_input() -> str:
    try:
        ts = datetime.now().strftime("%H:%M")
        prompt = (f"  {C.GD}[{ts}]{C.R}  {C.AB}›{C.R}  ")
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"


# ── shutdown ──────────────────────────────────────────────────────────────────
def shutdown():
    blank()
    _write("  " + C.G + "goodbye." + C.R + "\n")
    blank()
    sys.exit(0)


# ── LangGraph integration ─────────────────────────────────────────────────────
# We hook into shifu.py's run_mission by monkey-patching the graph's stream
# so the checkpoint bar updates in real time.

def _load_shifu():
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        import shifu as _s
        return _s, None
    except Exception as e:
        return None, str(e)


def _run_with_checkpoints(shifu_mod, mission: str, bar: CheckpointBar) -> str:
    """
    Stream shifu_graph manually so we can fire checkpoint events.
    Duplicates run_mission() logic but feeds bar as it goes.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    initial = {
        "messages":    [],
        "mission":     mission,
        "plan":        "",
        "complexity":  "",
        "iterations":  0,
        "verdict":     "",
        "retry_count": 0,
    }

    prev   = None
    final  = None

    for step in shifu_mod.shifu_graph.stream(initial, stream_mode="values"):
        # ── detect which node just ran ────────────────────────────────────
        if prev is None:
            bar.node("classify")
        elif not prev.get("complexity") and step.get("complexity"):
            bar.complexity(step["complexity"])
            if step["complexity"] == "COMPLEX":
                bar.node("plan")
        elif not prev.get("plan") and step.get("plan"):
            # emit plan steps
            for ln in step["plan"].splitlines():
                if re.match(r'^\s*\d+\.', ln):
                    bar.plan_step(ln)
            bar.node("execute")
        elif not prev.get("verdict") and step.get("verdict"):
            bar.verdict(step["verdict"])
        elif prev.get("verdict") != step.get("verdict") and step.get("verdict"):
            bar.verdict(step["verdict"])
        else:
            prev_n = len((prev or {}).get("messages", []))
            curr_n = len(step.get("messages", []))
            if curr_n > prev_n:
                new_msgs = step["messages"][prev_n:]
                has_tool_result = any(isinstance(m, ToolMessage) for m in new_msgs)
                has_tool_call   = any(
                    hasattr(m, "tool_calls") and m.tool_calls
                    for m in new_msgs if isinstance(m, AIMessage)
                )
                if has_tool_result:
                    for m in new_msgs:
                        if isinstance(m, ToolMessage):
                            # find matching tool name from previous AI tool_calls
                            tool_name = m.name if hasattr(m, "name") and m.name else "tool"
                            bar.tool_result(tool_name, str(m.content))
                    bar.node("execute")
                elif has_tool_call:
                    for m in new_msgs:
                        if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
                            for tc in m.tool_calls:
                                primary = next(iter(tc.get("args", {}).values()), "")
                                bar.tool(tc["name"], str(primary))

        prev  = step
        final = step

    bar.done()

    if not final:
        return "no output."
    last_ai = next(
        (m for m in reversed(final.get("messages", []))
         if isinstance(m, AIMessage) and m.content),
        None,
    )
    return last_ai.content if last_ai else "mission complete."


# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    boot()

    shifu_mod, err = _load_shifu()
    if err:
        _write("  " + C.ERR + f"⚠  could not load shifu.py: {err}" + C.R + "\n")
        _write("  " + C.G  + "   fix shifu.py and restart." + C.R + "\n")
    else:
        _write("  " + C.OK + "✓  ready" + C.R +
               "  " + C.G  + f"playground → {shifu_mod.PLAYGROUND_DIR.resolve()}" + C.R + "\n")
    blank()

    while True:
        try:
            raw = get_input()
        except KeyboardInterrupt:
            blank(); shutdown()

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("exit", "quit", "q"):
            shutdown()
        elif cmd in ("help", "?", "h"):
            show_help()
        elif cmd in ("clear", "cls"):
            boot()
        elif cmd == "history":
            show_history()
        elif cmd in ("skills", "skill"):
            show_skills()
        elif cmd in ("files", "filetypes", "supported"):
            show_file_support()
        else:
            if shifu_mod is None:
                _write("  " + C.ERR + "⚠  shifu.py not loaded." + C.R + "\n")
                blank(); continue

            blank()
            dim_line()
            _write("  " + C.W + raw[:tw() - 6] + C.R + "\n")
            dim_line()
            blank()

            bar     = CheckpointBar()
            spinner = IdleSpinner(bar)
            t0      = time.time()
            answer  = None
            spinner.start()

            try:
                answer = _run_with_checkpoints(shifu_mod, raw, bar)
            except KeyboardInterrupt:
                spinner.stop(); bar.done()
                _write("\n  " + C.G + "interrupted." + C.R + "\n")
                blank(); continue
            except Exception as exc:
                spinner.stop(); bar.done()
                blank()
                _write("  " + C.ERR + f"error: {exc}" + C.R + "\n")
                blank(); continue

            spinner.stop()
            elapsed = time.time() - t0

            if not answer or not answer.strip():
                _write("  " + C.G + "⚠  empty response." + C.R + "\n")
                blank(); continue

            _history.append({
                "t": datetime.now().strftime("%H:%M:%S"),
                "q": raw,
            })
            render_response(answer, elapsed)


if __name__ == "__main__":
    main()