import os
import re
import sys
import subprocess
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYGROUND_DIR = os.path.join(_PROJECT_ROOT, "Playground")
os.makedirs(PLAYGROUND_DIR, exist_ok=True)

MAX_OUTPUT_CHARS = 2000

# ── Patterns blocked anywhere in the command string ───────────────────────────
_BLOCKED_PATTERNS = re.compile(
    r"rm\s+-rf\s+[/~]"          # rm -rf / or rm -rf ~
    r"|:\(\)\s*\{[^}]*\}"       # fork bomb :(){ :|:& };:
    r"|mkfs\b"                  # format filesystem
    r"|dd\s+if="                # raw disk write
    r"|\b(shutdown|reboot|halt|poweroff)\b"  # system control
    r"|>\s*/dev/[sh]d[a-z]",    # writing to raw block devices
    re.IGNORECASE,
)

# ── Flags injected silently so interactive prompts never hang ─────────────────
_NON_INTERACTIVE = {
    "pip":     ["--quiet", "--no-input"],
    "apt-get": ["-y", "-q"],
    "apt":     ["-y", "-q"],
    "conda":   ["--yes"],
}

# ── Commands whose zero-output success has a specific meaning ─────────────────
_SEMANTIC_EMPTY = {
    "mkdir":  "Directory created.",
    "touch":  "File created.",
    "cp":     "Copy complete.",
    "mv":     "Move complete.",
    "chmod":  "Permissions updated.",
    "chown":  "Ownership updated.",
    "ln":     "Link created.",
    "rm":     "File(s) removed.",
    "git":    "Git command succeeded with no output.",
    "pip":    "Package operation succeeded.",
}


class TerminalInput(BaseModel):
    command: str = Field(
        ...,
        description=(
            "The shell command to execute. "
            "Working directory is always the Playground/ sandbox. "
            "Supports persistent 'cd <dir>' across calls. "
            "Use 'pip install <pkg>' to install libraries on the fly. "
            "Examples: 'ls -la', 'cd src', 'pip install numpy', 'python script.py'"
        )
    )


class TerminalTool(BaseTool):
    name: str = "Terminal"
    description: str = ""      # built dynamically in __init__
    args_schema: type[BaseModel] = TerminalInput

    # ── Persisted state ───────────────────────────────────────────────────────
    _cwd: str = PLAYGROUND_DIR

    def __init__(self, **data):
        super().__init__(**data)
        self.description = self._build_description()

    # ── 4 · Build description with live sandbox context ───────────────────────
    def _build_description(self) -> str:
        try:
            entries = os.listdir(PLAYGROUND_DIR)
            listing = ", ".join(sorted(entries)) if entries else "(empty)"
        except OSError:
            listing = "(could not list)"

        return (
            "Execute any shell / terminal command inside the Playground/ sandbox directory. "
            "Use this to run scripts, install Python packages with pip, create files, "
            "inspect the environment, or do anything you'd do in a real terminal. "
            "All commands are sandboxed to Playground/ for safety. "
            f"Current Playground contents: [{listing}]."
        )

    # ── 1 · Security checks ───────────────────────────────────────────────────
    def _check_safety(self, command: str) -> None:
        """Raise ValueError with a descriptive message on any blocked command."""
        if _BLOCKED_PATTERNS.search(command):
            raise ValueError(f"[BLOCKED] Dangerous pattern detected in: {command!r}")

        if ".." in command:
            raise ValueError(
                "[BLOCKED] Path traversal ('..') is not allowed. "
                "All paths must stay within Playground/."
            )

    # ── 1 · Inject non-interactive flags ─────────────────────────────────────
    @staticmethod
    def _make_non_interactive(command: str) -> str:
        """Prepend safety flags for known interactive package managers."""
        first_token = command.strip().split()[0].lower()
        extra = _NON_INTERACTIVE.get(first_token, [])
        if not extra:
            return command
        # Insert after the first token so flags come before sub-command args
        rest = command.strip()[len(first_token):].lstrip()
        return f"{first_token} {' '.join(extra)} {rest}"

    # ── 2 · Directory state management ────────────────────────────────────────
    def _resolve_cd(self, command: str) -> str | None:
        """
        If the command is a pure 'cd <dir>', resolve and update self._cwd.
        Returns a status string if handled, None otherwise.
        """
        stripped = command.strip()
        if not re.match(r"^cd(\s|$)", stripped):
            return None

        parts = stripped.split(None, 1)
        target = parts[1] if len(parts) > 1 else os.path.expanduser("~")

        candidate = os.path.normpath(
            os.path.join(self._cwd, target)
            if not os.path.isabs(target)
            else target
        )

        # Enforce sandbox boundary
        if not candidate.startswith(PLAYGROUND_DIR):
            return (
                f"[BLOCKED] Cannot navigate outside Playground/. "
                f"Attempted: {candidate}"
            )

        if not os.path.isdir(candidate):
            return f"[ERROR] No such directory: {candidate}"

        self._cwd = candidate
        return f"[cd] Now in: {os.path.relpath(self._cwd, PLAYGROUND_DIR) or '.'}"

    # ── 3 · Truncate long output ──────────────────────────────────────────────
    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        kept = text[:MAX_OUTPUT_CHARS]
        omitted = len(text) - MAX_OUTPUT_CHARS
        return f"{kept}\n[Output truncated — {omitted} chars omitted]"

    # ── 3 · Semantic empty-output message ─────────────────────────────────────
    @staticmethod
    def _empty_message(command: str) -> str:
        first_token = command.strip().split()[0].lower()
        return _SEMANTIC_EMPTY.get(
            first_token,
            "[Command completed with no output]",
        )

    # ── Main entry point ──────────────────────────────────────────────────────
    def _run(self, command: str) -> str:
        # 1. Safety gate
        try:
            self._check_safety(command)
        except ValueError as exc:
            return str(exc)

        # 2. Handle pure 'cd' without spawning a shell
        cd_result = self._resolve_cd(command)
        if cd_result is not None:
            return cd_result

        # 1. Non-interactive flag injection
        command = self._make_non_interactive(command)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self._cwd,       # 2. Persisted directory
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )

            # 3. Build and truncate output
            parts = []
            if result.stdout:
                parts.append(f"[stdout]\n{self._truncate(result.stdout.rstrip())}")
            if result.stderr:
                parts.append(f"[stderr]\n{self._truncate(result.stderr.rstrip())}")
            if result.returncode != 0:
                parts.append(f"[exit code: {result.returncode}]")

            if not parts:
                # 3. Semantic empty message
                return self._empty_message(command)

            return "\n".join(parts)

        except subprocess.TimeoutExpired:
            return "[ERROR] Command timed out after 120 seconds."
        except Exception as exc:
            return f"[ERROR] {type(exc).__name__}: {exc}"