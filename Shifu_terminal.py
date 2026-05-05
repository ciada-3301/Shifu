#!/usr/bin/env python3
"""
╔══════════════════════════════╗
║   S H I F U  T E R M I N A L ║
╚══════════════════════════════╝
"""
from nosier import  _patch_rich
import sys, os, time, threading, itertools, textwrap, re, random
from datetime import datetime

PLAYGROUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "playground")
os.makedirs(PLAYGROUND_DIR, exist_ok=True)


# ── Colours ────────────────────────────────────────────────────────────────────
class C:
    R       = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    CYAN    = "\033[96m"
    DCYAN   = "\033[36m"
    GOLD    = "\033[93m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"


def tw() -> int:
    try:    return os.get_terminal_size().columns
    except: return 100

def div(ch="─", col=C.DCYAN):
    print("  " + col + ch * (tw() - 4) + C.R)

def blank(): print()

def pl(text="", col=C.WHITE, indent=2):
    print(" " * indent + col + text + C.R)


# ── ASCII boot ─────────────────────────────────────────────────────────────────
LOGO = r"""
  ███████╗██╗  ██╗██╗███████╗██╗   ██╗
  ██╔════╝██║  ██║██║██╔════╝██║   ██║
  ███████╗███████║██║█████╗  ██║   ██║
  ╚════██║██╔══██║██║██╔══╝  ██║   ██║
  ███████║██║  ██║██║██║     ╚██████╔╝
  ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═════╝
"""

def boot():
    os.system("cls" if os.name == "nt" else "clear")
    blank()
    for line in LOGO.split("\n"):
        print(C.CYAN + C.BOLD + "  " + line + C.R)
        time.sleep(0.025)
    blank()
    div("═")
    blank()
    ts = datetime.now().strftime("%a %d %b %Y  •  %H:%M:%S")
    pl(f"{C.GREY}{ts}", indent=2)
    pl(f"{C.GREY}type  help  •  exit to quit", indent=2)
    blank()
    div("═")
    blank()


# ── Live action display ────────────────────────────────────────────────────────
# Instead of a spinner with generic rotating tags, we show what the agent is
# actually doing — tool calls, file writes, code runs — as they happen.
# CrewAI doesn't expose a streaming callback natively, so we hook into stdout.

class ActionStream:
    """
    Wraps the real stdout/stderr during the crew run.
    Intercepts CrewAI's internal print() calls and surfaces
    meaningful action lines — everything else is suppressed.

    Acts as a full stdout proxy so nothing crashes on writelines/isatty/etc.
    """

    _PATTERNS = [
        (re.compile(r'serper|web.?search|search.?query', re.I),  "Searching the web"),
        (re.compile(r'fetch|http|request|url', re.I),            "Fetching URL"),
        (re.compile(r'(writ|creat).{0,10}(file|path)', re.I),    "Writing file"),
        (re.compile(r'(read|open|load).{0,10}(file|path)', re.I),"Reading file"),
        (re.compile(r'execut|terminal|bash|subprocess|pip\b', re.I), "Running code"),
        (re.compile(r'director|scandir|listdir|tree\b', re.I),   "Scanning directory"),
        (re.compile(r'\bplanning\b|\bplan\b.*task', re.I),        "Planning"),
        (re.compile(r'agent.*start|starting.*agent|task.*start', re.I), "Agent working"),
    ]

    def __init__(self, t0: float, real_stdout):
        self._real   = real_stdout
        self._t0     = t0
        self._last   = ""
        self._lock   = threading.Lock()
        self._buf    = ""          # line buffer

    def _show(self, label: str):
        if label == self._last:
            return
        self._last = label
        elapsed = time.time() - self._t0
        line = (f"\r  {C.GREY}{elapsed:5.1f}s{C.R}  "
                f"{C.DCYAN}·{C.R}  {C.WHITE}{label}{C.R}          \n")
        with self._lock:
            self._real.write(line)
            self._real.flush()

    def _process(self, text: str):
        for pattern, label in self._PATTERNS:
            if pattern.search(text):
                self._show(label + " …")
                return
        m = re.search(r'(?:tool|using)[:\s]+([^\n\r]{3,40})', text, re.I)
        if m:
            self._show(f"Tool  {m.group(1).strip()} …")
            return
        m = re.search(r'[\w./\\-]+\.(py|js|ts|json|csv|txt|md|yaml|sh)\b', text)
        if m:
            self._show(f"↳  {m.group(0)}")

    def write(self, data: str):
        # Buffer until newline so we process whole lines
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self._process(line)

    # ── Full stdout proxy — must implement everything ──────────────────────
    def flush(self):                    pass
    def isatty(self):                   return False
    def writelines(self, lines):
        for l in lines: self.write(l)
    def fileno(self):                   return self._real.fileno()
    def readable(self):                 return False
    def writable(self):                 return True
    def seekable(self):                 return False
    @property
    def encoding(self):                 return self._real.encoding
    @property
    def errors(self):                   return self._real.errors


# ── Minimal inline markdown renderer ──────────────────────────────────────────
_DELAY_PROSE = 0.007
_DELAY_CODE  = 0.003

class _K:
    BG       = "\033[48;5;235m"
    RBGR     = "\033[49m"
    KW       = "\033[38;5;81m"
    STR      = "\033[38;5;215m"
    CMT      = "\033[38;5;244m\033[3m"
    NUM      = "\033[38;5;141m"
    PLAIN    = "\033[38;5;253m"
    RESET    = "\033[0m"

_PY_KW = {"def","class","return","import","from","as","if","elif","else",
          "for","while","with","in","not","and","or","is","None","True",
          "False","try","except","finally","raise","pass","break","continue",
          "lambda","yield","async","await","global","nonlocal"}

def _hl_python(code: str) -> str:
    out = []
    for raw in code.split("\n"):
        if re.match(r"\s*#", raw):
            out.append(_K.CMT + raw + _K.RESET); continue
        res, pos = "", 0
        for m in re.finditer(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\\n]*\')'
                              r'|(\b\d+\.?\d*\b)|(\b\w+\b)', raw):
            res += _K.PLAIN + raw[pos:m.start()]
            t = m.group()
            if   m.group(1): res += _K.STR + t
            elif m.group(2): res += _K.NUM + t
            elif t in _PY_KW: res += _K.KW + t
            else:             res += _K.PLAIN + t
            pos = m.end()
        res += _K.PLAIN + raw[pos:]
        out.append(res + _K.RESET)
    return "\n".join(out)

def _inline(t: str) -> str:
    t = re.sub(r'`([^`]+)`',   _K.BG + _K.STR + r' \1 ' + C.R + C.WHITE, t)
    t = re.sub(r'\*\*(.+?)\*\*', C.BOLD + C.WHITE + r'\1' + C.R + C.WHITE, t)
    t = re.sub(r'\*(.+?)\*',   C.ITALIC + C.CYAN + r'\1' + C.R + C.WHITE, t)
    return t

def _stream(text: str, delay: float = _DELAY_PROSE):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush()
        if ch not in (" ", "\n", "\t"):
            time.sleep(delay * random.uniform(0.6, 1.3))

def _sline(text: str, delay=_DELAY_PROSE):
    _stream(text, delay); sys.stdout.write("\n"); sys.stdout.flush()

def _code_block(lang: str, code: str):
    w     = min(tw() - 6, 88)
    lang  = (lang or "code").strip().lower()
    label = f" {lang} "
    top   = "╭──" + label + "─" * max(0, w - len(label) - 3)
    sys.stdout.write("\n  " + C.DCYAN + top + C.R + "\n")
    hl = _hl_python(code) if lang in ("python", "py") else code
    for line in hl.split("\n"):
        sys.stdout.write("  " + C.DCYAN + "│" + C.R + _K.BG + "  " + line + "  " + _K.RESET + "\n")
        sys.stdout.flush(); time.sleep(_DELAY_CODE)
    sys.stdout.write("  " + C.DCYAN + "╰" + "─" * w + C.R + "\n\n"); sys.stdout.flush()

def render(text: str, elapsed: float):
    blank()
    div("─", C.GOLD)
    pl(f"shifu  ›  {datetime.now().strftime('%H:%M:%S')}  ({elapsed:.1f}s)", C.GOLD + C.BOLD)
    div("─", C.GOLD)
    blank()

    lines = str(text).split("\n")
    W     = tw() - 10
    i     = 0
    PAD   = "    "
    while i < len(lines):
        ln = lines[i]

        if re.match(r'^```(\w*)', ln):
            lang, body = re.match(r'^```(\w*)', ln).group(1), []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i]); i += 1
            _code_block(lang, "\n".join(body)); i += 1; continue

        if ln.strip() == "":
            sys.stdout.write("\n"); sys.stdout.flush(); i += 1; continue

        if ln.startswith("# "):
            _sline("\n" + PAD + C.BOLD + C.CYAN + "◈  " + ln[2:].upper() + "  ◈" + C.R + "\n"); i += 1; continue
        if ln.startswith("## "):
            _sline("\n" + PAD + C.BOLD + C.GOLD + "▸  " + ln[3:] + C.R); i += 1; continue
        if ln.startswith("### "):
            _sline(PAD + C.BOLD + C.MAGENTA + "›  " + ln[4:] + C.R); i += 1; continue
        if re.match(r'^[-*_]{3,}\s*$', ln):
            sys.stdout.write("  " + C.DCYAN + "━" * (tw() - 6) + C.R + "\n"); i += 1; continue
        if ln.startswith("> "):
            _sline("  " + C.DCYAN + "▌ " + C.R + C.ITALIC + C.GREY + _inline(ln[2:]) + C.R); i += 1; continue

        bm = re.match(r'^(\s*)[-*+] (.+)', ln)
        if bm:
            lvl = len(bm.group(1)) // 2
            dot = ["◆","◇","·"][min(lvl, 2)]
            col = [C.CYAN, C.GOLD, C.GREY][min(lvl, 2)]
            _sline(PAD + "  " * lvl + col + dot + " " + C.WHITE + _inline(bm.group(2)) + C.R); i += 1; continue

        nm = re.match(r'^(\s*)(\d+)\. (.+)', ln)
        if nm:
            lvl = len(nm.group(1)) // 2
            _sline(PAD + "  " * lvl + C.GOLD + nm.group(2) + "." + C.WHITE + " " + _inline(nm.group(3)) + C.R); i += 1; continue

        for wl in textwrap.fill(ln, width=W).split("\n"):
            _sline(PAD + C.WHITE + _inline(wl) + C.R)
        i += 1

    blank(); div("─", C.GOLD); blank()


# ── Crew loader ────────────────────────────────────────────────────────────────
def load_crew():
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        from crew import ShifuCrew  # type: ignore
        return ShifuCrew(), None
    except Exception as e:
        return None, str(e)


# ── Input prompt ───────────────────────────────────────────────────────────────
def get_input() -> str:
    try:
        ts = datetime.now().strftime("%H:%M")
        return input(f"  {C.GREY}[{ts}]{C.R}  {C.CYAN}{C.BOLD}›{C.R}  ").strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"


# ── Help ───────────────────────────────────────────────────────────────────────
def show_help():
    blank(); div()
    pl("commands", C.GOLD + C.BOLD)
    blank()
    for cmd, desc in [
        ("help / ?",     "this panel"),
        ("clear / cls",  "clear screen"),
        ("history",      "session history"),
        ("exit / quit",  "shutdown"),
        ("<message>",    "send to shifu"),
    ]:
        pl(f"  {C.CYAN}{cmd:<16}{C.R}{C.WHITE}{desc}", indent=2)
    blank(); div(); blank()


# ── History ────────────────────────────────────────────────────────────────────
_history: list[dict] = []

def show_history():
    if not _history:
        pl("no queries yet.", C.GREY); return
    blank(); div()
    pl("history", C.GOLD + C.BOLD); blank()
    for i, e in enumerate(_history, 1):
        q = e["q"][:72] + ("…" if len(e["q"]) > 72 else "")
        pl(f"  [{i:02d}]  {C.GREY}{e['t']}{C.R}  {C.WHITE}{q}", indent=2)
    blank(); div(); blank()


# ── Shutdown ───────────────────────────────────────────────────────────────────
def shutdown():
    blank(); div("═")
    pl("shutting down …", C.CYAN)
    div("═"); blank()
    sys.exit(0)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    boot()

    # Load crew quietly
    crew_inst, err = load_crew()
    if err:
        pl(f"⚠  crew load failed: {err}", C.RED + C.BOLD)
        pl("   queries won't work until crew.py is fixed.", C.GREY)
    else:
        pl("✓  ready.", C.GREEN + C.BOLD)
    blank()

    while True:
        try:
            raw = get_input()
        except KeyboardInterrupt:
            blank(); shutdown()

        if not raw: continue
        cmd = raw.lower()

        if cmd in ("exit", "quit", "q"):          shutdown()
        elif cmd in ("help", "?", "h"):           show_help()
        elif cmd in ("clear", "cls"):             boot()
        elif cmd == "history":                    show_history()
        else:
            if crew_inst is None:
                pl("⚠  no crew loaded.", C.RED); blank(); continue

            blank()
            real_stdout   = sys.stdout
            real_stderr   = sys.stderr
            spinner_stop  = threading.Event()
            t0            = time.time()
            action        = ActionStream(t0, real_stdout)

            def _idle_dot():
                chars = itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"])
                while not spinner_stop.is_set():
                    real_stdout.write(f"\r  {C.GREY}{next(chars)}{C.R}  ")
                    real_stdout.flush()
                    time.sleep(0.09)
                real_stdout.write("\r" + " " * 24 + "\r")
                real_stdout.flush()

            dot_thread = threading.Thread(target=_idle_dot, daemon=True)
            dot_thread.start()

            # Redirect — crew output goes to ActionStream, render uses real_stdout
            sys.stdout = action
            sys.stderr = action
            try:
                _patch_rich() 
                result  = crew_inst.crew().kickoff(
                    inputs={
                        "user_input":     raw,
                        "playground_dir": PLAYGROUND_DIR,
                    }
                )
                elapsed = time.time() - t0
            except KeyboardInterrupt:
                sys.stdout = real_stdout; sys.stderr = real_stderr
                spinner_stop.set(); dot_thread.join()
                pl("  interrupted.", C.GREY); blank(); continue
            except Exception as exc:
                sys.stdout = real_stdout; sys.stderr = real_stderr
                elapsed = time.time() - t0
                spinner_stop.set(); dot_thread.join()
                pl(f"  error ({elapsed:.1f}s):", C.RED + C.BOLD)
                pl(f"  {exc}", C.RED); blank(); continue
            finally:
                sys.stdout = real_stdout
                sys.stderr = real_stderr

            spinner_stop.set()
            dot_thread.join()

            # Extract result text — CrewOutput can nest the answer in multiple places
            answer = None
            if result is not None:
                for attr in ("raw", "output", "result", "final_output"):
                    val = getattr(result, attr, None)
                    if val and str(val).strip():
                        answer = str(val).strip(); break
                if answer is None:
                    answer = str(result).strip() or None

            if not answer:
                pl("  ⚠  shifu returned an empty response.", C.GOLD)
                pl(f"  raw result object: {repr(result)[:200]}", C.GREY)
                blank(); continue

            _history.append({"t": datetime.now().strftime("%H:%M:%S"), "q": raw})
            render(answer, elapsed)


if __name__ == "__main__":
    main()