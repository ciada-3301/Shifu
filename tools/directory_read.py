from langchain_core.tools import tool
from pathlib import Path

PLAYGROUND_DIR = Path("Playground")
PLAYGROUND_DIR.mkdir(exist_ok=True)

@tool
def directory_read(dirpath: str = ".") -> str:
    """
    List all files and subdirectories under a directory (recursive).
    Relative paths are resolved against Playground/.
    """
    path = Path(dirpath)
    if not path.is_absolute():
        path = PLAYGROUND_DIR / path
    if not path.exists():
        return f"Error: directory not found — {path}"
    lines = []
    for item in sorted(path.rglob("*")):
        indent = "  " * (len(item.relative_to(path).parts) - 1)
        lines.append(f"{indent}{'📁' if item.is_dir() else '📄'} {item.name}")
    return "\n".join(lines) if lines else "(empty directory)"