import os
from pathlib import Path
from langchain_core.tools import tool

HONCHO_FILE = Path(".shifu/honcho.txt")

@tool
def honcho_update(content: str, mode: str = "append") -> str:
    """
    Updates the Honcho memory file (.shifu/honcho.txt) which is injected into your system prompt on every run.
    Use this to save core personality traits, permanent preferences, or instructions the user wants you to ALWAYS remember.
    
    Args:
        content: The text to write or append to the honcho file.
        mode: Either "append" (default) to add to existing rules, or "overwrite" to replace everything.
    """
    HONCHO_FILE.parent.mkdir(exist_ok=True)
    
    if mode == "overwrite":
        HONCHO_FILE.write_text(content, encoding="utf-8")
        return "Honcho memory overwritten successfully. I will remember this forever."
    else:
        existing = ""
        if HONCHO_FILE.exists():
            existing = HONCHO_FILE.read_text(encoding="utf-8")
        
        new_content = existing + ("\n" if existing else "") + content
        HONCHO_FILE.write_text(new_content, encoding="utf-8")
        return "Honcho memory appended successfully. I will remember this forever."
