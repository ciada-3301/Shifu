from crewai import Agent, Crew, Task,  Process
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM
import os
from dotenv import load_dotenv
from crewai_tools import (
    SerperDevTool,
    FileWriterTool,
    FileReadTool,
    DirectoryReadTool,
    DirectorySearchTool,
    PDFSearchTool,
    CodeInterpreterTool
)

load_dotenv()

# Setup the cloud LLM using your Ollama key
ollama_api_key = os.getenv("OLLAMA_API_KEY")

shifu_llm = LLM(
    model="ollama/gemma4:31b",
    base_url="https://ollama.com",
    api_key=ollama_api_key
)

@CrewBase
class ShifuAssistantCrew():
    """Shifu: Personal AI Assistant Crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def shifu(self) -> Agent:
        return Agent(
            config=self.agents_config['shifu'], # Update your agents.yaml to have a 'shifu' section
            tools=[
                SerperDevTool(),
                FileReadTool(),
                FileWriterTool(),
                DirectoryReadTool(),
                DirectorySearchTool(),
                PDFSearchTool(),
                CodeInterpreterTool()
            ],
            llm=shifu_llm,
            verbose=True,
            allow_delegation=False
        )

    @task
    def assistant_task(self) -> Task:
        return Task(
            config=self.tasks_config['assistant_task'], # Update your tasks.yaml to have an 'assistant_task'
            agent=self.shifu()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.shifu()],
            tasks=[self.assistant_task()],
            process=Process.sequential,
            verbose=True
        )

# Main execution loop for an interactive assistant feel
if __name__ == "__main__":
    prompt = input("How can Shifu help you today? ..> ")
    inputs = {
        "user_input": prompt
    }

    result = ShifuAssistantCrew().crew().kickoff(inputs=inputs)
    print("\n--- Shifu's Response ---")
    print(result)
