from langchain_core.tools import tool
from pathlib import Path

SKILLS_DIR = Path("skills")
SKILLS_DIR.mkdir(exist_ok=True)

@tool
def load_skill(skill_name: str) -> str:
    """
    Load the full instructions for a skill by name.
    Call this before starting any task that matches a skill's domain.
    Returns the complete SKILL.md content so you can follow its guidance.

    To see available skills and their descriptions, check the AVAILABLE SKILLS
    section in your system prompt, then call this tool with the skill folder name.
    """
    # Refresh registry in case new skills were dropped in
    global SKILLS
    SKILLS = _scan_skills()

    entry = SKILLS.get(skill_name)
    if not entry:
        available = ", ".join(SKILLS.keys()) or "none"
        return (f"Skill '{skill_name}' not found.\n"
                f"Available skills: {available}")

    path = Path(entry["path"])
    if not path.exists():
        return f"Error: SKILL.md missing at {path}"

    content = path.read_text(encoding="utf-8")
    return f"=== SKILL: {entry['name']} ===\n\n{content}"