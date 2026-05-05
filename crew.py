"""
crew.py — Shifu's Optimized AI Crew
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Smart routing — not every query needs the full pipeline:
  DIRECT   → Simple Q&A, conversational      → Single LLM call
  RESEARCH → Needs web search + write-up      → Viper → Po
  FILE_OPS → File read/write/organize         → Po only
  CODE     → Write + execute code             → Viper → Po → Tigress

Agents:
  Viper   — Web researcher (search only)
  Po      — Filesystem + content writer
  Tigress — Code executor (only when code is needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import re
from dotenv import load_dotenv

from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    SerperDevTool,
    FileWriterTool,
    FileReadTool,
    DirectoryReadTool,
)

from tools.terminal_tool import TerminalTool

load_dotenv()

# ── Playground ────────────────────────────────────────────────────────────────
PLAYGROUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Playground")
os.makedirs(PLAYGROUND_DIR, exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────────
shifu_llm = LLM(
    model="ollama/gpt-oss:120b-cloud",
    base_url="https://ollama.com",
    api_key=os.getenv("OLLAMA_API_KEY_SHIFU"),
)

tigress_llm = LLM(
    model="ollama/gpt-oss:120b-cloud",
    base_url="https://ollama.com",
    api_key=os.getenv("OLLAMA_API_KEY_TIGRESS"),
)


# ── Route Classification ─────────────────────────────────────────────────────
# Keywords/patterns that signal each route type.

_CODE_SIGNALS = re.compile(
    r'\b(write\s+(a\s+)?(python|script|code|program|function|class|app|application|bot|cli|tool))'
    r'|\b(build|create|develop|implement|code|program|automate|scrape|crawl|parse|execute|run|compile|deploy)\b'
    r'|\b(pip\s+install|import\s+\w|def\s+\w|\.py|javascript|html|css|sql|api|endpoint|server|flask|django|fastapi)\b',
    re.IGNORECASE
)

_RESEARCH_SIGNALS = re.compile(
    r'\b(search|research|find\s+(out|info|information|details|about))'
    r'|\b(look\s+up|what\s+is|who\s+is|when\s+did|where\s+is|how\s+does|how\s+do|explain|latest|current|recent|news)\b'
    r'|\b(compare|difference\s+between|vs\.?|versus|pros\s+and\s+cons)\b',
    re.IGNORECASE
)

_FILE_SIGNALS = re.compile(
    r'\b(read|open|list|show|display|cat|view|browse|tree|ls|dir)\s+(file|folder|directory|path)'
    r'|\b(write\s+to\s+file|save\s+(to|as|into)|create\s+(a\s+)?file|organize|move|rename|copy|delete)\b'
    r'|\b(summarise|summarize)\s+(the\s+)?(files?|folder|directory|content)\b',
    re.IGNORECASE
)


def classify_route(user_input: str) -> str:
    """
    Classify user input into a route type.
    Priority: CODE > RESEARCH+FILE > FILE > RESEARCH > DIRECT
    """
    text = user_input.strip()
    has_code = bool(_CODE_SIGNALS.search(text))
    has_research = bool(_RESEARCH_SIGNALS.search(text))
    has_file = bool(_FILE_SIGNALS.search(text))

    if has_code:
        return "CODE"
    if has_research and has_file:
        return "RESEARCH"   # Research that results in a file — Viper → Po
    if has_file:
        return "FILE_OPS"
    if has_research:
        return "RESEARCH"
    return "DIRECT"


# ── Crew ──────────────────────────────────────────────────────────────────────
@CrewBase
class ShifuAssistantCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ── Agents ────────────────────────────────────────────────────────────────

    @agent
    def viper(self) -> Agent:
        return Agent(
            config=self.agents_config["viper"],
            tools=[SerperDevTool()],
            llm=shifu_llm,
            verbose=False,
            allow_delegation=False,
            max_iter=5,
            memory=False,
        )

    @agent
    def po(self) -> Agent:
        return Agent(
            config=self.agents_config["po"],
            tools=[
                FileReadTool(),
                FileWriterTool(),
                DirectoryReadTool(),
            ],
            llm=shifu_llm,
            verbose=False,
            allow_delegation=False,
            max_iter=8,
            memory=False,
        )

    @agent
    def tigress(self) -> Agent:
        return Agent(
            config=self.agents_config["tigress"],
            tools=[
                TerminalTool(),
                FileWriterTool(),
                FileReadTool(),
                DirectoryReadTool(),
            ],
            llm=tigress_llm,
            verbose=False,
            allow_delegation=False,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_iter=15,
            memory=False,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.viper(),
        )

    @task
    def filesystem_task(self) -> Task:
        return Task(
            config=self.tasks_config["filesystem_task"],
            agent=self.po(),
            context=[self.research_task()],
        )

    @task
    def execution_task(self) -> Task:
        return Task(
            config=self.tasks_config["execution_task"],
            agent=self.tigress(),
            context=[
                self.research_task(),
                self.filesystem_task(),
            ],
        )

    # ── Crew Builders (one per route) ─────────────────────────────────────────

    def _build_crew(self, agents: list, tasks: list) -> Crew:
        """Build a crew with the given agents and tasks."""
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            memory=False,
            share_crew=False,
        )

    def direct_crew(self) -> Crew:
        """Single agent for simple Q&A — Po answers directly."""
        direct_task = Task(
            description=(
                "Answer the following user request directly and concisely. "
                "Do not use any tools unless absolutely necessary. "
                "Give a clear, helpful, well-formatted response.\n\n"
                "USER REQUEST: {user_input}"
            ),
            expected_output="A clear, helpful, well-formatted answer to the user's question.",
            agent=self.po(),
        )
        return self._build_crew([self.po()], [direct_task])

    def research_crew(self) -> Crew:
        """Viper researches, Po writes the output."""
        return self._build_crew(
            [self.viper(), self.po()],
            [self.research_task(), self.filesystem_task()],
        )

    def fileops_crew(self) -> Crew:
        """Po handles file operations solo."""
        return self._build_crew(
            [self.po()],
            [self.filesystem_task()],
        )

    def code_crew(self) -> Crew:
        """Full pipeline: Viper → Po → Tigress."""
        return self._build_crew(
            [self.viper(), self.po(), self.tigress()],
            [self.research_task(), self.filesystem_task(), self.execution_task()],
        )

    # ── Default crew (backwards compatibility) ────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Default crew — uses full pipeline. Use route_and_run() for smart routing."""
        return self._build_crew(
            [self.viper(), self.po(), self.tigress()],
            [self.research_task(), self.filesystem_task(), self.execution_task()],
        )

    # ── Smart Router ──────────────────────────────────────────────────────────

    def route_and_run(self, user_input: str, playground_dir: str = None):
        """
        Classify the user's request and run only the agents needed.
        Returns (result, route_type, crew_used).
        """
        if playground_dir is None:
            playground_dir = PLAYGROUND_DIR

        route = classify_route(user_input)
        inputs = {
            "user_input": user_input,
            "playground_dir": playground_dir,
        }

        if route == "DIRECT":
            crew_obj = self.direct_crew()
        elif route == "RESEARCH":
            crew_obj = self.research_crew()
        elif route == "FILE_OPS":
            crew_obj = self.fileops_crew()
        else:  # CODE
            crew_obj = self.code_crew()

        result = crew_obj.kickoff(inputs=inputs)
        return result, route


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🐾  Playground sandbox : {PLAYGROUND_DIR}")
    print("━" * 60)
    prompt = input("Mission for the crew ..> ").strip()

    if not prompt:
        print("No mission given. Shifu rests.")
    else:
        crew_instance = ShifuAssistantCrew()
        result, route = crew_instance.route_and_run(prompt)
        print(f"\n{'━' * 60}")
        print(f"  ROUTE: {route}")
        print(f"  MISSION COMPLETE — SHIFU'S REPORT")
        print(f"{'━' * 60}")
        print(result.raw)
        print("━" * 60 + "\n")