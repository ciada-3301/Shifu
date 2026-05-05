#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════╗
║         S H I F U   T E R M I N A L          ║
║         Powered by CrewAI + Ollama            ║
╚═══════════════════════════════════════════════╝

A Jarvis-style terminal interface for the Shifu AI Crew.
Drop this file alongside your crew.py and run it directly.
"""

import sys
import os
import time
import threading
import itertools
import textwrap
import random
from datetime import datetime

# ── Colour palette (ANSI) ───────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"

    # Cyan / teal family  →  primary "arc-reactor" glow
    CYAN    = "\033[96m"
    DCYAN   = "\033[36m"

    # Accent colours
    GOLD    = "\033[93m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"

    # Compound helpers
    PROMPT  = f"\033[96m\033[1m"          # bold cyan for the input caret
    SHIFU   = f"\033[93m\033[1m"          # gold-bold for Shifu's replies
    SYS     = f"\033[90m"                  # dim grey for system/metadata lines
    ERR     = f"\033[91m\033[1m"          # bold red for errors
    OK      = f"\033[92m\033[1m"          # bold green for success


# ── Terminal dimensions ──────────────────────────────────────────────────────
def term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 100


# ── Typewriter printer ───────────────────────────────────────────────────────
def typewrite(text: str, colour: str = C.WHITE, delay: float = 0.012, indent: int = 0) -> None:
    prefix = " " * indent
    for ch in text:
        sys.stdout.write(colour + ch + C.RESET)
        sys.stdout.flush()
        if ch not in (" ", "\n"):
            time.sleep(delay * random.uniform(0.5, 1.5))
    sys.stdout.write("\n")


def print_line(text: str = "", colour: str = C.WHITE, indent: int = 2) -> None:
    print(" " * indent + colour + text + C.RESET)


def divider(char: str = "─", colour: str = C.DCYAN) -> None:
    w = term_width() - 4
    print(" " * 2 + colour + char * w + C.RESET)


def blank() -> None:
    print()


# ── Animated spinner ─────────────────────────────────────────────────────────
class Spinner:
    FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    TAGS   = [
        "ANALYZING REQUEST",
        "QUERYING KNOWLEDGE BASE",
        "CROSS-REFERENCING DATA",
        "RUNNING AGENT PIPELINE",
        "SYNTHESIZING RESPONSE",
        "CALIBRATING INFERENCE",
    ]

    def __init__(self, message: str = "PROCESSING"):
        self.message   = message
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        cycle  = itertools.cycle(self.FRAMES)
        tags   = itertools.cycle(self.TAGS)
        tag    = next(tags)
        t_next = time.time() + 2.5
        while not self._stop_evt.is_set():
            frame = next(cycle)
            line  = (f"  {C.CYAN}{frame}{C.RESET}  "
                     f"{C.DIM}{C.CYAN}[ {tag} ]{C.RESET}   ")
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
            time.sleep(0.08)
            if time.time() >= t_next:
                tag    = next(tags)
                t_next = time.time() + 2.5

    def start(self) -> "Spinner":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * (term_width() - 2) + "\r")
        sys.stdout.flush()


# ── Boot sequence ─────────────────────────────────────────────────────────────
BOOT_LINES = [
    ("SHIFU NEURAL CORE", C.CYAN + C.BOLD,  0.04),
    ("Initializing CrewAI runtime …",        C.GREY,  0.02),
    ("Loading agent configuration …",        C.GREY,  0.02),
    ("Connecting to Ollama endpoint …",      C.GREY,  0.02),
    ("Mounting tool suite  [SERPER / PDF / FS]", C.GREY, 0.02),
    ("All systems nominal.",                 C.GREEN + C.BOLD, 0.03),
]

ASCII_LOGO = r"""
  ███████╗██╗  ██╗██╗███████╗██╗   ██╗
  ██╔════╝██║  ██║██║██╔════╝██║   ██║
  ███████╗███████║██║█████╗  ██║   ██║
  ╚════██║██╔══██║██║██╔══╝  ██║   ██║
  ███████║██║  ██║██║██║     ╚██████╔╝
  ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═════╝
"""

def boot_sequence() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    blank()

    for line in ASCII_LOGO.split("\n"):
        print(C.CYAN + C.BOLD + "  " + line + C.RESET)
        time.sleep(0.03)

    blank()
    divider("═")
    blank()

    for text, colour, spd in BOOT_LINES:
        sys.stdout.write("  " + colour)
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(spd)
        sys.stdout.write(C.RESET + "\n")
        time.sleep(0.12)

    blank()
    divider()
    blank()

    ts = datetime.now().strftime("%A, %d %b %Y  •  %H:%M:%S")
    print_line(f"SESSION STARTED  {ts}", C.SYS)
    print_line("Type  'help'  for commands  •  'exit' / 'quit' to terminate", C.SYS)
    blank()
    divider("═")
    blank()


# ── Help panel ────────────────────────────────────────────────────────────────
def show_help() -> None:
    blank()
    divider()
    print_line("AVAILABLE COMMANDS", C.GOLD + C.BOLD)
    blank()
    cmds = [
        ("help",           "Display this panel"),
        ("clear  /  cls",  "Clear the terminal"),
        ("history",        "Show conversation history"),
        ("status",         "Print crew / LLM status"),
        ("exit  /  quit",  "Shut down Shifu"),
        ("<any text>",     "Send a query to the Shifu agent"),
    ]
    for cmd, desc in cmds:
        print_line(f"  {C.CYAN}{cmd:<22}{C.RESET}{C.WHITE}{desc}", indent=2)
    blank()
    divider()
    blank()


# ── Status panel ──────────────────────────────────────────────────────────────
def show_status(crew_loaded: bool) -> None:
    blank()
    divider()
    print_line("SYSTEM STATUS", C.GOLD + C.BOLD)
    blank()

    rows = [
        ("Agent",    "Shifu",               C.CYAN),
        ("Model",    "ollama/gemma4:31b",   C.CYAN),
        ("Process",  "Sequential",          C.CYAN),
        ("Tools",    "Serper · PDF · FS",   C.CYAN),
        ("Crew",     "LOADED" if crew_loaded else "NOT LOADED",
                     C.GREEN if crew_loaded else C.RED),
    ]
    for label, val, col in rows:
        print_line(f"  {C.GREY}{label:<14}{col}{val}", indent=2)

    blank()
    divider()
    blank()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING MARKDOWN RENDERER
# Parses the agent's output token-by-token and applies rich ANSI styling.
# Handles: code blocks (with language tag + syntax highlight), inline code,
#          bold, italic, headers (H1-H3), bullet/numbered lists, blockquotes,
#          horizontal rules — all streamed character-by-character like AI sites.
# ══════════════════════════════════════════════════════════════════════════════

import re

# ── Tiny Python syntax highlighter (ANSI) ────────────────────────────────────
_PY_KEYWORDS   = {"def","class","return","import","from","as","if","elif","else",
                  "for","while","with","in","not","and","or","is","None","True",
                  "False","try","except","finally","raise","pass","break",
                  "continue","lambda","yield","async","await","global","nonlocal",
                  "del","assert","print"}
_PY_BUILTINS   = {"len","range","print","input","type","str","int","float","list",
                  "dict","set","tuple","open","enumerate","zip","map","filter",
                  "sorted","reversed","any","all","hasattr","getattr","setattr",
                  "isinstance","issubclass","super","self","cls"}

# ANSI for code-block interior
class _K:                                # code colour palette
    BG        = "\033[48;5;235m"         # dark grey background
    RESET_BG  = "\033[49m"
    KW        = "\033[38;5;81m"          # bright sky-blue  → keywords
    BUILTIN   = "\033[38;5;150m"         # soft green       → builtins
    STRING    = "\033[38;5;215m"         # warm orange      → strings
    COMMENT   = "\033[38;5;244m\033[3m"  # grey italic      → comments
    NUMBER    = "\033[38;5;141m"         # lavender         → numbers
    DECORATOR = "\033[38;5;213m"         # pink             → decorators
    PLAIN     = "\033[38;5;253m"         # near-white       → everything else
    RESET     = "\033[0m"


def _highlight_python(code: str) -> str:
    """Return ANSI-coloured Python code string (no streaming, called per block)."""
    lines_out = []
    for raw_line in code.split("\n"):
        line = raw_line

        # comments
        if re.match(r"\s*#", line):
            lines_out.append(_K.COMMENT + line + _K.RESET)
            continue

        # tokenise with regex, process left→right
        result = ""
        pos = 0
        token_re = re.compile(
            r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\\n]*\')'  # strings
            r'|(\b\d+\.?\d*\b)'            # numbers
            r'|(@\w+)'                      # decorators
            r'|(\b\w+\b)'                   # identifiers / keywords
        )
        for m in token_re.finditer(line):
            # literal text before this token
            result += _K.PLAIN + line[pos:m.start()]
            tok = m.group()
            if m.group(1):                  result += _K.STRING    + tok
            elif m.group(2):               result += _K.NUMBER    + tok
            elif m.group(3):               result += _K.DECORATOR + tok
            elif tok in _PY_KEYWORDS:      result += _K.KW        + tok
            elif tok in _PY_BUILTINS:      result += _K.BUILTIN   + tok
            else:                          result += _K.PLAIN     + tok
            pos = m.end()
        result += _K.PLAIN + line[pos:]
        lines_out.append(result + _K.RESET)

    return "\n".join(lines_out)


def _highlight_generic(code: str) -> str:
    """Light highlighting for non-Python code (strings, numbers, comments)."""
    result = ""
    for line in code.split("\n"):
        if re.match(r"\s*(#|//|--)", line):
            result += _K.COMMENT + line + _K.RESET + "\n"
        else:
            # strings
            line2 = re.sub(r'(".*?"|\'.*?\')', _K.STRING + r'\1' + _K.PLAIN, line)
            # numbers
            line2 = re.sub(r'\b(\d+\.?\d*)\b', _K.NUMBER + r'\1' + _K.PLAIN, line2)
            result += _K.PLAIN + line2 + _K.RESET + "\n"
    return result.rstrip("\n")


# ── Stream a styled string token-by-token ────────────────────────────────────
_TOKEN_DELAY  = 0.008   # seconds per character (prose)
_CODE_DELAY   = 0.004   # faster inside code blocks

def _stream(text: str, delay: float = _TOKEN_DELAY) -> None:
    """Write text to stdout one character at a time."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if ch not in (" ", "\n", "\t"):
            time.sleep(delay * random.uniform(0.6, 1.3))


def _stream_line(text: str, delay: float = _TOKEN_DELAY) -> None:
    _stream(text, delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


# ── Inline markdown → ANSI (bold, italic, inline-code) ───────────────────────
def _inline(text: str) -> str:
    """Convert inline markdown spans to ANSI escape sequences."""
    # inline code  `...`
    text = re.sub(r'`([^`]+)`',
                  _K.BG + _K.STRING + r' \1 ' + C.RESET + C.WHITE, text)
    # bold  **...**  or  __...__
    text = re.sub(r'\*\*(.+?)\*\*|__(.+?)__',
                  C.BOLD + C.WHITE + r'\1\2' + C.RESET + C.WHITE, text)
    # italic  *...*  or  _..._
    text = re.sub(r'\*(.+?)\*|_(.+?)_',
                  C.ITALIC + C.CYAN + r'\1\2' + C.RESET + C.WHITE, text)
    return text


# ── Code-block renderer ───────────────────────────────────────────────────────
_CB_TOP    = "╭"
_CB_BOT    = "╰"
_CB_SIDE   = "│"
_CB_FILL   = "─"

def _render_code_block(lang: str, code: str) -> None:
    """Pretty-print a fenced code block with language tag and syntax colours."""
    w       = min(term_width() - 6, 90)
    lang    = (lang or "code").strip().lower()
    label   = f" {lang} "
    top_bar = (_CB_TOP + _CB_FILL * 2 + label
               + _CB_FILL * max(0, w - len(label) - 3) + " ")

    # header
    sys.stdout.write("\n  " + C.DCYAN + top_bar + C.RESET + "\n")

    # syntax-highlight
    if lang in ("python", "py"):
        highlighted = _highlight_python(code)
    else:
        highlighted = _highlight_generic(code)

    # stream each line with side-bar
    for line in highlighted.split("\n"):
        sys.stdout.write("  " + C.DCYAN + _CB_SIDE + C.RESET
                         + _K.BG + "  " + line + "  " + _K.RESET + "\n")
        sys.stdout.flush()
        time.sleep(_CODE_DELAY)

    # footer
    sys.stdout.write("  " + C.DCYAN + _CB_BOT + _CB_FILL * (w) + C.RESET + "\n\n")
    sys.stdout.flush()


# ── Main markdown stream renderer ─────────────────────────────────────────────
def _render_markdown_stream(text: str) -> None:
    """
    Parse and stream `text` as markdown, applying rich ANSI styling.
    Processes line-by-line; handles fenced code blocks as atomic units.
    """
    lines       = text.split("\n")
    i           = 0
    indent      = "    "
    width       = term_width() - 10

    while i < len(lines):
        line = lines[i]

        # ── fenced code block  ```lang ... ``` ────────────────────────────
        fence_open = re.match(r'^```(\w*)', line)
        if fence_open:
            lang        = fence_open.group(1)
            code_lines  = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            _render_code_block(lang, "\n".join(code_lines))
            i += 1
            continue

        # ── blank line ────────────────────────────────────────────────────
        if line.strip() == "":
            sys.stdout.write("\n")
            sys.stdout.flush()
            i += 1
            continue

        # ── H1  #  ────────────────────────────────────────────────────────
        if line.startswith("# "):
            content = line[2:].strip()
            sys.stdout.write("\n")
            _stream_line(indent + C.BOLD + C.CYAN + "  ◈  " + content.upper() + "  ◈" + C.RESET)
            sys.stdout.write("\n")
            i += 1
            continue

        # ── H2  ##  ───────────────────────────────────────────────────────
        if line.startswith("## "):
            content = line[3:].strip()
            sys.stdout.write("\n")
            _stream_line(indent + C.BOLD + C.GOLD + "  ▸  " + content + C.RESET)
            sys.stdout.write("  " + C.DCYAN + "─" * (len(content) + 6) + C.RESET + "\n")
            i += 1
            continue

        # ── H3  ###  ──────────────────────────────────────────────────────
        if line.startswith("### "):
            content = line[4:].strip()
            _stream_line(indent + C.BOLD + C.MAGENTA + "  ›  " + content + C.RESET)
            i += 1
            continue

        # ── horizontal rule  ---  or  ***  ───────────────────────────────
        if re.match(r'^[-*_]{3,}\s*$', line):
            sys.stdout.write("  " + C.DCYAN + "━" * (term_width() - 6) + C.RESET + "\n")
            i += 1
            continue

        # ── blockquote  >  ────────────────────────────────────────────────
        if line.startswith("> "):
            content = _inline(line[2:])
            _stream_line("  " + C.DCYAN + "▌ " + C.RESET + C.ITALIC + C.GREY + content + C.RESET)
            i += 1
            continue

        # ── unordered bullet  - / * / +  ─────────────────────────────────
        bullet_m = re.match(r'^(\s*)[-*+] (.+)', line)
        if bullet_m:
            lvl     = len(bullet_m.group(1)) // 2
            content = _inline(bullet_m.group(2))
            dot     = ["◆", "◇", "·"][min(lvl, 2)]
            colour  = [C.CYAN, C.GOLD, C.GREY][min(lvl, 2)]
            prefix  = indent + "  " * lvl + colour + dot + " " + C.WHITE
            # wrap long bullets
            raw_len = len(bullet_m.group(2))
            if raw_len > width - 6:
                wrapped = textwrap.fill(bullet_m.group(2), width=width - 6)
                first   = True
                for wl in wrapped.split("\n"):
                    if first:
                        _stream_line(prefix + _inline(wl) + C.RESET)
                        first = False
                    else:
                        _stream_line(indent + "  " * lvl + "  " + C.WHITE + _inline(wl) + C.RESET)
            else:
                _stream_line(prefix + content + C.RESET)
            i += 1
            continue

        # ── ordered list  1.  2.  …  ─────────────────────────────────────
        num_m = re.match(r'^(\s*)(\d+)\. (.+)', line)
        if num_m:
            lvl     = len(num_m.group(1)) // 2
            num     = num_m.group(2)
            content = _inline(num_m.group(3))
            prefix  = indent + "  " * lvl + C.GOLD + f"{num}." + C.WHITE + " "
            _stream_line(prefix + content + C.RESET)
            i += 1
            continue

        # ── plain paragraph ───────────────────────────────────────────────
        content = _inline(line)
        wrapped = textwrap.fill(line, width=width)   # wrap on raw text first
        for wl in wrapped.split("\n"):
            _stream_line(indent + C.WHITE + _inline(wl) + C.RESET, delay=_TOKEN_DELAY)
        i += 1


# ── Public render entry point ─────────────────────────────────────────────────
def render_response(result: str, elapsed: float) -> None:
    blank()
    divider("─", C.GOLD)
    print_line(
        f"SHIFU  ›  {datetime.now().strftime('%H:%M:%S')}  ({elapsed:.1f}s)",
        C.GOLD + C.BOLD
    )
    divider("─", C.GOLD)
    blank()

    _render_markdown_stream(str(result))

    blank()
    divider("─", C.GOLD)
    blank()


# ── History storage ───────────────────────────────────────────────────────────
history: list[dict] = []

def show_history() -> None:
    if not history:
        blank()
        print_line("No queries yet this session.", C.GREY)
        blank()
        return
    blank()
    divider()
    print_line("CONVERSATION HISTORY", C.GOLD + C.BOLD)
    blank()
    for i, entry in enumerate(history, 1):
        ts  = entry["time"]
        usr = entry["query"][:80] + ("…" if len(entry["query"]) > 80 else "")
        print_line(f"  [{i:02d}]  {C.GREY}{ts}{C.RESET}  {C.WHITE}{usr}", indent=2)
    blank()
    divider()
    blank()


# ── Crew loader (lazy import so startup is fast) ──────────────────────────────
def load_crew():
    """
    Import ShifuAssistantCrew from crew.py in the same directory.
    Returns (crew_instance, None) on success or (None, error_message) on failure.
    """
    # Add current working directory to path so 'crew' is importable
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        from crew import ShifuAssistantCrew  # type: ignore
        return ShifuAssistantCrew(), None
    except ImportError as e:
        return None, f"Could not import crew.py → {e}"
    except Exception as e:
        return None, f"Error initialising crew → {e}"


# ── Input prompt ──────────────────────────────────────────────────────────────
def get_input() -> str:
    try:
        ts = datetime.now().strftime("%H:%M")
        raw = input(f"  {C.GREY}[{ts}]{C.RESET} {C.PROMPT}YOU ›{C.RESET}  ")
        return raw.strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"


# ── Shutdown ──────────────────────────────────────────────────────────────────
def shutdown() -> None:
    blank()
    divider("═")
    typewrite("  Shutting down Shifu …  Standing by.", colour=C.CYAN, delay=0.04)
    time.sleep(0.4)
    divider("═")
    blank()
    sys.exit(0)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    boot_sequence()

    # Lazy-load the crew
    spin = Spinner("LOADING CREW").start()
    crew_instance, err = load_crew()
    spin.stop()

    if err:
        print_line(f"⚠  {err}", C.ERR)
        print_line("   Shifu will still run but agent queries won't work.", C.GREY)
        blank()
        crew_instance = None
    else:
        print_line("✓  Crew loaded successfully.", C.OK)
        blank()

    while True:
        try:
            user_input = get_input()
        except KeyboardInterrupt:
            blank()
            shutdown()

        if not user_input:
            continue

        lower = user_input.lower()

        # ── Built-in commands ──────────────────────────────────────────────
        if lower in ("exit", "quit", "q"):
            shutdown()

        elif lower in ("help", "?", "h"):
            show_help()
            continue

        elif lower in ("clear", "cls"):
            os.system("cls" if os.name == "nt" else "clear")
            boot_sequence()
            continue

        elif lower == "history":
            show_history()
            continue

        elif lower == "status":
            show_status(crew_instance is not None)
            continue

        # ── Agent query ────────────────────────────────────────────────────
        else:
            if crew_instance is None:
                print_line("⚠  No crew loaded. Check that crew.py is in the current directory.", C.ERR)
                blank()
                continue

            blank()
            print_line(f"Dispatching to Shifu agent …", C.SYS)
            blank()

            spinner = Spinner().start()
            t_start = time.time()

            try:
                result = crew_instance.crew().kickoff(inputs={
                    "user_input":        user_input,
                    "playground_dir":    r"C:\Users\arkad\OneDrive\Documents\Codes\Shifu\Playground",
                    "planning_output":   "",
                    "research_output":   "",
                    "filesystem_output": "",
                    "execution_output":  "",
                })
                elapsed = time.time() - t_start
                spinner.stop()

                history.append({
                    "time":  datetime.now().strftime("%H:%M:%S"),
                    "query": user_input,
                })

                render_response(result, elapsed)

            except KeyboardInterrupt:
                spinner.stop()
                blank()
                print_line("  Query interrupted by user.", C.GREY)
                blank()

            except Exception as exc:
                elapsed = time.time() - t_start
                spinner.stop()
                blank()
                print_line(f"  ERROR  ({elapsed:.1f}s):  {exc}", C.ERR)
                blank()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()