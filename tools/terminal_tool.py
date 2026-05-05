import os
import sys
import subprocess
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


# ── Playground sandbox path ───────────────────────────────────────────────────
PLAYGROUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Playground")
os.makedirs(PLAYGROUND_DIR, exist_ok=True)


# ── Input schema ──────────────────────────────────────────────────────────────
class TerminalInput(BaseModel):
    command: str = Field(
        ...,
        description=(
            "The shell command to execute. "
            "Working directory is always the playground/ sandbox. "
            "Use 'pip install <pkg>' to install libraries on the fly. "
            "Examples: 'ls -la', 'pip install numpy', 'python script.py'"
        )
    )


# ── Tool ──────────────────────────────────────────────────────────────────────
class TerminalTool(BaseTool):
    name: str = "Terminal"
    description: str = (
        "Execute any shell / terminal command inside the playground/ sandbox directory. "
        "Use this to run scripts, install Python packages with pip, create files, "
        "inspect the environment, or do anything you'd do in a real terminal. "
        "All commands are sandboxed to playground/ for safety."
    )
    args_schema: type[BaseModel] = TerminalInput

    _BLOCKED_PREFIXES: tuple = (
        "rm -rf /", "rm -rf ~", ":(){ :|:& };:", "mkfs", "dd if=",
        "shutdown", "reboot", "halt", "poweroff",
    )

    def _run(self, command: str) -> str:
        # Safety gate
        cmd_lower = command.strip().lower()
        for blocked in self._BLOCKED_PREFIXES:
            if cmd_lower.startswith(blocked):
                return f"[BLOCKED] Dangerous command refused: '{command}'"

        # Inherit current venv so pip installs land in the right place
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=PLAYGROUND_DIR,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            output_parts = []
            if result.stdout:
                output_parts.append(f"[stdout]\n{result.stdout.rstrip()}")
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr.rstrip()}")
            if result.returncode != 0:
                output_parts.append(f"[exit code: {result.returncode}]")

            return "\n".join(output_parts) if output_parts else "[Command completed with no output]"

        except subprocess.TimeoutExpired:
            return "[ERROR] Command timed out after 120 seconds."
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {e}"