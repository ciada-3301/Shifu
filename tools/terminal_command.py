from langchain_core.tools import tool
from pathlib import Path

PLAYGROUND_DIR = Path("Playground")
PLAYGROUND_DIR.mkdir(exist_ok=True)

@tool
def terminal_command(command: str) -> str:
    """
    Execute a shell command (mkdir, pip install, python script.py, etc.).
    Always scope file-creation commands to the Playground/ directory.
    Returns combined stdout + stderr.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        parts = []
        if stdout:
            parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        return "\n".join(parts) if parts else "(command completed with no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds."
    except Exception as e:
        return f"Error executing command: {e}"