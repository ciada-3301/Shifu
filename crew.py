"""
crew.py — Shifu's Kung Fu AI Crew
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agents:
  Shifu   — Hierarchical manager / planner / reviewer (no direct tools)
  Viper   — Researcher (web search only)
  Po      — Filesystem expert + content writer for non-coding missions
  Tigress — Coder / executor (only engages when code is actually needed)

Mission Types handled:
  RESEARCH_AND_WRITE  → Viper researches, Po writes the file. Tigress stands down.
  CODE_AND_EXECUTE    → Full pipeline: Viper → Po (scaffold) → Tigress (execute)
  FILESYSTEM_ONLY     → Po only. No research, no code.
  MIXED               → Full pipeline.

Routing is driven by Shifu's planning_task output. Every downstream task
reads TIGRESS REQUIRED from the plan and self-scopes accordingly.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
from textwrap import dedent
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

# ── LLM ───────────────────────────────────────────────────────────────────────
shifu_llm = LLM(
    model="gpt-oss:120b-cloud",
    base_url="https://ollama.com",
    api_key=os.getenv("OLLAMA_API_KEY_SHIFU"),
)

tigress_llm = LLM(
    model="ollama/gpt-oss:120b-cloud",
    base_url="https://ollama.com",
    api_key=os.getenv("OLLAMA_API_KEY_TIGRESS"),
)


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
            tools=[SerperDevTool()],        # Web search only — no PDFs, no files
            llm=shifu_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=False,
        )

    @agent
    def po(self) -> Agent:
        return Agent(
            config=self.agents_config["po"],
            tools=[
                FileReadTool(),             # Read existing files for context
                FileWriterTool(),           # Write final deliverables or scaffolding
                DirectoryReadTool(),        # Map directory structures
            ],
            llm=shifu_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=8,
            memory=False,
        )

    @agent
    def tigress(self) -> Agent:
        return Agent(
            config=self.agents_config["tigress"],
            tools=[
                TerminalTool(),             # Sandboxed terminal — pip, run scripts
                FileWriterTool(),           # Write code files to playground/
                FileReadTool(),             # Read output files and logs
                DirectoryReadTool(),        # Inspect playground/ structure
            ],
            llm=tigress_llm,
            verbose=True,
            allow_delegation=False,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_iter=15,
            memory=False,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @task
    def planning_task(self) -> Task:
        """
        Shifu classifies the mission type and builds an execution plan.
        The TIGRESS REQUIRED field in the output is the routing signal
        that all downstream tasks read to self-scope.
        """
        return Task(
            config=self.tasks_config["planning_task"],
            agent=None,  # Handled by Shifu (manager) in hierarchical mode
        )

    @task
    def research_task(self) -> Task:
        """
        Viper's research step. Self-scopes: outputs a STATUS-only block
        if no [Viper] steps appear in Shifu's plan.
        """
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.viper(),
            context=[self.planning_task()],
        )

    @task
    def filesystem_task(self) -> Task:
        """
        Po's task. In RESEARCH_AND_WRITE missions Po is the final executor
        and writes the deliverable himself. In coding missions Po scaffolds
        for Tigress. The task description encodes both modes.
        """
        return Task(
            config=self.tasks_config["filesystem_task"],
            agent=self.po(),
            context=[self.planning_task(), self.research_task()],
        )

    @task
    def execution_task(self) -> Task:
        """
        Tigress's task. Reads TIGRESS REQUIRED from the plan. If it's 'no',
        she outputs a one-liner and stops — no code, no terminal commands.
        If it's 'yes', she runs the full Antigravity loop.
        """
        return Task(
            config=self.tasks_config["execution_task"],
            agent=self.tigress(),
            context=[
                self.planning_task(),
                self.research_task(),
                self.filesystem_task(),
            ],
        )

    @task
    def review_task(self) -> Task:
        """
        Shifu's final review. Audits against SUCCESS CRITERIA, routes
        the correct quality checks based on MISSION TYPE, and either
        accepts the output or issues targeted feedback.
        """
        return Task(
            config=self.tasks_config["review_task"],
            agent=None,  # Shifu (manager)
            context=[
                self.planning_task(),
                self.filesystem_task(),
                self.execution_task(),
            ],
        )

    # ── Crew ──────────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=shifu_llm,
            verbose=True,
            memory=False,
            planning=True,
            planning_llm=shifu_llm,
            max_rpm=20,
            share_crew=False,
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🐾  Playground sandbox : {PLAYGROUND_DIR}")
    print("━" * 60)
    prompt = input("Mission for the Five ..> ").strip()

    if not prompt:
        print("No mission given. Shifu rests.")
    else:
        result = ShifuAssistantCrew().crew().kickoff(
            inputs={
                "user_input":        prompt,
                "playground_dir":    str(PLAYGROUND_DIR),
                # Populated at runtime by task outputs:
                "planning_output":   "",
                "research_output":   "",
                "filesystem_output": "",
                "execution_output":  "",
            }
        )
        print("\n" + "━" * 60)
        print("  MISSION COMPLETE — SHIFU'S FINAL REPORT")
        print("━" * 60)
        print(result.raw)
        print("━" * 60 + "\n")