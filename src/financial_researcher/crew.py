from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool

researcher_llm = LLM(
    model = "groq/llama-3.3-70b-versatile"
)

analyst_llm = LLM(model="ollama/qwen3.5:4b", base_url="http://localhost:11434")





@CrewBase
class FinancialResearcher():
    """FinancialResearcher crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config['researcher'], llm=researcher_llm, verbose=True, tools=[SerperDevTool()])
    
    @agent
    def analyst(self) -> Agent:
        return Agent(config=self.agents_config['analyst'], llm=analyst_llm, verbose=True)
    
    @task
    def research_task(self) -> Task:
        return Task(config = self.tasks_config['research_task'])
    
    @task
    def analysis_task(self) -> Task:
        return Task(config = self.tasks_config['analysis_task'])
    

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents = self.agents,
            tasks = self.tasks,
            process = Process.sequential,
            verbose=True
        )