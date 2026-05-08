from langchain_core.tools import tool
from pathlib import Path

PLAYGROUND_DIR = Path("")
PLAYGROUND_DIR.mkdir(exist_ok=True)

@tool
def file_write(filepath: str, content: str) -> str:
    """
    Write text content to a file.
    Paths that are NOT absolute are resolved relative to Playground/.
    Parent directories are created automatically.
    """
    path = Path(filepath)
    if not path.is_absolute():
        path = PLAYGROUND_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"✅ Written: {path.resolve()}"