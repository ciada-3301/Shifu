# ── Noise filter & live action display ────────────────────────────────────────
import logging
import re

# ── 1. Silence every noisy logger before importing crewai ─────────────────────
# CrewAI, LiteLLM, and OpenAI SDK all log at INFO/DEBUG by default.
# Silence them globally; only WARNING+ from *our* code gets through.
for _noisy in (
    "crewai", "litellm", "openai", "httpx",
    "httpcore", "urllib3", "requests", "langchain",
    "langchain_core", "langchain_community",
):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# Root logger: suppress everything below WARNING
logging.basicConfig(level=logging.WARNING, force=True)


class ActionStream:
    """
    Wraps the real stdout/stderr during the crew run.

    Four layers of noise filtering:
      1. Logger suppression above (kills structured log output).
      2. Rich console patch (kills Rich's live / progress output).
      3. Line-level pattern matching (surfaces meaningful action lines).
      4. Hard-drop list (swallows known-useless CrewAI chatter verbatim).
    """

    # ── Lines to surface ───────────────────────────────────────────────────
    _PATTERNS = [
        (re.compile(r'serper|web.?search|search.?query',  re.I), "Searching the web"),
        (re.compile(r'fetch|http[s]?://|curl\b',          re.I), "Fetching URL"),
        (re.compile(r'(writ|creat).{0,12}(file|path)',    re.I), "Writing file"),
        (re.compile(r'(read|open|load).{0,12}(file|path)',re.I), "Reading file"),
        (re.compile(r'execut|terminal|bash|subprocess|pip\b', re.I), "Running code"),
        (re.compile(r'director|scandir|listdir|tree\b',   re.I), "Scanning directory"),
        (re.compile(r'\bplanning\b|\bplan\b.*task',        re.I), "Planning"),
        (re.compile(r'agent.*start|starting.*agent|task.*start', re.I), "Agent working"),
        (re.compile(r'tool.*call|calling.*tool|invoking',  re.I), "Calling tool"),
        (re.compile(r'llm.*request|sending.*prompt|api.*call', re.I), "Thinking"),
    ]

    # ── Lines to silently swallow ──────────────────────────────────────────
    # Exact substrings; if any match, the line is dropped entirely.
    _DROP = {
        "Working Agent:", "Task output:", "Agent stopped",
        "Entering new", "chain", "Finished chain",
        "tokens used", "Prompt tokens", "Completion tokens",
        "> Entering", "> Finished", "Retrying",
        "verbose", "DEBUG", "INFO ", " INFO",
        "HTTP Request:", "HTTP Response:",
        "model_name", "temperature", "max_tokens",
    }

    def __init__(self, t0: float, real_stdout):
        self._real  = real_stdout
        self._t0    = t0
        self._last  = ""
        self._lock  = threading.Lock()
        self._buf   = ""

    def _elapsed(self) -> str:
        return f"{time.time() - self._t0:5.1f}s"

    def _show(self, label: str):
        """Print a single deduplicated action line."""
        if label == self._last:
            return
        self._last = label
        line = (
            f"\r  {C.GREY}{self._elapsed()}{C.R}  "
            f"{C.DCYAN}·{C.R}  {C.WHITE}{label}{C.R}          \n"
        )
        with self._lock:
            self._real.write(line)
            self._real.flush()

    def _process(self, text: str):
        # Hard-drop first — cheapest check
        tl = text.lower()
        for drop in self._DROP:
            if drop.lower() in tl:
                return

        # Pattern match — surface a labelled action line
        for pattern, label in self._PATTERNS:
            if pattern.search(text):
                self._show(label + " …")
                return

        # Tool name heuristic
        m = re.search(r'(?:tool|using|action)[:\s]+([^\n\r]{3,40})', text, re.I)
        if m:
            self._show(f"Tool  ›  {m.group(1).strip()} …")
            return

        # File reference heuristic (only if line is short — avoids matching
        # giant LLM output dumps that happen to contain a filename)
        if len(text) < 120:
            m = re.search(r'[\w./\\-]+\.(py|js|ts|json|csv|txt|md|yaml|sh)\b', text)
            if m:
                self._show(f"↳  {m.group(0)}")

    def write(self, data: str):
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self._process(line)

    # ── Full stdout proxy ──────────────────────────────────────────────────
    def flush(self):            pass
    def isatty(self):           return False
    def writelines(self, lines):
        for l in lines: self.write(l)
    def fileno(self):           return self._real.fileno()
    def readable(self):         return False
    def writable(self):         return True
    def seekable(self):         return False
    @property
    def encoding(self):         return self._real.encoding
    @property
    def errors(self):           return self._real.errors


def _patch_rich():
    """
    Replace Rich's Console with a no-op so CrewAI's progress bars,
    live spinners, and panel prints never reach the terminal.
    Called once, right before kickoff().
    """
    try:
        import rich.console as _rc

        class _SilentConsole:
            def __init__(self, *a, **kw): pass
            def print(self, *a, **kw):    pass
            def log(self, *a, **kw):      pass
            def rule(self, *a, **kw):     pass
            def status(self, *a, **kw):   return _NullCtx()
            def __enter__(self):          return self
            def __exit__(self, *_):       pass

        class _NullCtx:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def update(self, *a, **kw): pass

        _rc.Console = _SilentConsole

        # Also patch crewai's own console handle if it already imported it
        import crewai.utilities.printer as _cp   # noqa: F401
        _cp.Printer = type("_NoPrinter", (), {
            "print": staticmethod(lambda *a, **kw: None),
        })
    except Exception:
        pass   # if the internal path changes, just continue — logging is enough