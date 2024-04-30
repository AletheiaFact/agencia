from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from tools.querido_diario_tool import QueridoDiarioTools
from crewai_tools.tools import FileReadTool

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

querido_diario_advanced_search_context_tool = FileReadTool(
	file_path='querido_diario_search_context.txt',
	description="A tool designed to efficiently read and interpret search operators context to construct the subject"
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
	def fact_checker(self) -> Agent:
		return Agent(
			config = self.agents_config['fact_checker'],
			llm = self.llm,
		)

	@task
	def create_claim_subject(self) -> Task:
		return Task(
			config = self.tasks_config['create_claim_subject'],
			agent = self.subject_creator(),
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
	def create_fact_checking_report(self) -> Task:
		return Task(
			config = self.tasks_config['create_fact_checking_report'],
			agent = self.fact_checker()
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Querido diario crew"""
		return Crew(
			agents =  self.agents,
			tasks = self.tasks,
			process = Process.sequential,
			verbose = 2
		)