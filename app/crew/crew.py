from crewai import Crew, Process
from crewai.project import CrewBase, crew
from crewai import Task
from crewai.project import task
from crewai import Agent
from crewai.project import agent
from langchain_openai import ChatOpenAI
from .tools import QueridoDiarioTools, querido_diario_advanced_search_context_tool
from fastapi import HTTPException
from crew.errors import NoGazettesFoundError, CityNotFoundError

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

@CrewBase
class QueridoDiarioCrew():
	"""Querido diario crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	
	def __init__(self) -> None:
		self.llm = llm

	@agent
	def subject_creator(self) -> Agent:
		return Agent(
			config = self.agents_config['subject_creator'],
			llm = self.llm,
			tools = [querido_diario_advanced_search_context_tool]
		)

	@agent
	def gazette_data_retrieval(self) -> Agent:
		return Agent(
			config = self.agents_config['gazette_data_retrieval'],
			llm = self.llm,
			tools = [QueridoDiarioTools.querido_diario_fetch]
		)
	
	@agent
	def data_analyst(self) -> Agent:
		return Agent(
			config = self.agents_config['data_analyst'],
			llm = self.llm,
			tools = [QueridoDiarioTools.gazette_search_context],
		)

	@agent
	def gazette_data_analyst(self) -> Agent:
		return Agent(
			config = self.agents_config['gazette_data_analyst'],
			llm = self.llm,
		)

	@agent
	def fact_checker(self) -> Agent:
		return Agent(
			config = self.agents_config['fact_checker'],
			llm = self.llm,
		)
	
	@task
	def create_claim_subject(self) -> Task:
		return Task(
			config = self.tasks_config['create_claim_subject'],
			agent = self.subject_creator()
		)
	
	@task
	def research_gazettes(self) -> Task:
		return Task(
			config = self.tasks_config['research_gazettes'],
   			agent = self.gazette_data_retrieval()
		)
	
	@task
	def collect_gazette_relevant_data(self) -> Task:
		return Task(
			config = self.tasks_config['collect_gazette_relevant_data'],
			agent = self.data_analyst()
		)
  
	@task
	def cross_check_information(self) -> Task:
		return Task(
			config = self.tasks_config['cross_check_collected_data_and_claim'],
   			agent = self.gazette_data_analyst()
		)
	
	@task
	def create_fact_checking_report(self) -> Task:
		return Task(
			config = self.tasks_config['create_fact_checking_report'],
   			agent = self.fact_checker()
		)

	def task_call_back_error_handle(self, state):
		if "not found" in state.exported_output.lower():
			raise CityNotFoundError()
		if "invalid" and "url" in state.exported_output.lower() or "http error" in state.exported_output.lower():
			raise HTTPException(status_code=400, detail="Invalid request: The AI Agent task parameters are incorrect.")
		if "no public gazettes" in state.exported_output.lower():
			raise NoGazettesFoundError()
	
	@crew
	def crew(self) -> Crew:
		"""Creates the Querido diario crew"""
		return Crew(
			agents =  self.agents,
			tasks = self.tasks,
			process = Process.sequential,
			verbose = 2,
			task_callback=self.task_call_back_error_handle
		)
  
	def kickoff(self, state) -> Crew:
		crew = self.crew()
		context = state["context"]
		state = {
			**state,
   			"context": {
				"city": context.get("city", None),
				"published_since": context.get("published_since", None),
				"published_until": context.get("published_until", None),
				"sources": context.get("sources", []),
			}
		}
		result = crew.kickoff(state)
  
		return {
			**state,
			"messages": result
		}