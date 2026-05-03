from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import SerperDevTool
from langchain_ollama import ChatOllama

local_llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434"
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
            llm=local_llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher()],
            tasks=[self.research_task()],
            verbose=True
        )


inputs = {
    "user_input": "Latest AI news"
}

result = MyCrew().crew().kickoff(inputs=inputs)

print(result)