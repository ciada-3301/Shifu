from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM
import os
from dotenv import load_dotenv
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
load_dotenv()

ollama_api_key = os.getenv("OLLAMA_API_KEY")



cloud_llm = LLM(
    model="ollama/gemma4:31b",
    base_url="https://ollama.com",
    api_key=ollama_api_key
)

@CrewBase
class MyCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[SerperDevTool()],
            llm=cloud_llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            agent=self.researcher()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher()],
            tasks=[self.research_task()],
            verbose=True
        )

prompt = input("..> ")
inputs = {
    "user_input": prompt
}

result = MyCrew().crew().kickoff(inputs=inputs)

print(result)