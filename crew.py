"""
crew.py — Shifu, the One-Agent Crew
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One agent. All tools. Two LLMs — planner for complex, executor for direct.
Simple mission → direct answer. Complex mission → plan → execute → review.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, FileWriterTool, FileReadTool, DirectoryReadTool
from tools.terminal_tool import TerminalTool

load_dotenv()
os.environ["OPENAI_API_KEY"] = "NA" 
PLAYGROUND_DIR = Path("playground")
PLAYGROUND_DIR.mkdir(exist_ok=True)

# ── LLMs ─────────────────────────────────────────────────────────────────────
# planner_llm  → reasoning, classification, review  (heavier, used sparingly)
# executor_llm → action, writing, code, search      (default workhorse)

planner_llm = LLM(
    model="ollama/gemma4:31b-cloud",
    base_url="https://ollama.com",
    api_key=os.getenv("OLLAMA_API_KEY_PLANNER"),
)

executor_llm = LLM(
    model="ollama/gemma4:31b-cloud",
    base_url="https://ollama.com",
    api_key=os.getenv("OLLAMA_API_KEY_EXECUTOR"),
)


# ── Crew ─────────────────────────────────────────────────────────────────────
@CrewBase
class ShifuCrew:
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    @agent
    def shifu(self) -> Agent:
        return Agent(
            config=self.agents_config["shifu"],
            tools=[
                SerperDevTool(),
                FileReadTool(),
                FileWriterTool(),
                DirectoryReadTool(),
                TerminalTool(),
            ],
            llm=executor_llm,
            verbose=False,
            allow_delegation=False,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_iter=15,
            memory=False,
        )

    @task
    def mission_task(self) -> Task:
        return Task(
            config=self.tasks_config["mission_task"],
            agent=self.shifu(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            # planning=True kicks in only when mission_task signals COMPLEX
            # via the task description — we let the task itself decide whether
            # to invoke the planner_llm for a pre-flight plan.
            planning=True,
            planning_llm=planner_llm,
            verbose=False,
            memory=False,
            max_rpm=20,
            share_crew=False,
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🐾  Shifu's playground : {PLAYGROUND_DIR.resolve()}")
    print("━" * 55)
    prompt = input("Mission ..> ").strip()

    if not prompt:
        print("No mission. Shifu meditates.")
    else:
        result = ShifuCrew().crew().kickoff(
            inputs={
                "user_input":     prompt,
                "playground_dir": str(PLAYGROUND_DIR.resolve()),
            }
        )
        print("\n" + "━" * 55)
        print(result.raw)
        print("━" * 55 + "\n")