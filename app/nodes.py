from crew.agents import Agents
from langchain_community.document_loaders import WebBaseLoader

class Nodes():
	def __init__(self) -> None:
		self.listQuestionsAgent = Agents.listQuestionsAgent()
		self.researcherAgent = Agents.researcherAgent()
		self.factCheckerAgent = Agents.factCheckerAgent()
  
	def list_questions(self, state):
		claim = state["claim"]
		result = self.listQuestionsAgent.invoke({"claim": claim, "language": state["language"] })

		return {
			**state,
			"questions": result,
		}
  
	def search_online(self, state):
		claim = state["claim"]
		context = state["context"]
		sources = context["sources"]
		language = state["language"]
		doc_context = []

		if sources:
			loader = WebBaseLoader(sources)
			load_document = loader.load()
			doc_context = load_document[0].page_content

		result = self.researcherAgent.invoke({
      		"claim": claim,
      		"sources": sources,
      		"language": language,
			"context": doc_context
      	})
  
		return {
			**state,
			"messages": [result],
		}
  
	def create_report(self, state):
		claim = state["claim"]
		messages = state["messages"]
		questions = state["questions"]
		language = state["language"]

		result = self.factCheckerAgent.invoke({
      		"claim": claim,
			"messages": messages,
			"questions": questions,
   			"language": language
		})
  
		return {
			**state,
			"messages": result,
		}
  
	def router(self, state):
		search_type = state["search_type"]

		if search_type == "gazettes":
			return "continue"
		if search_type == "online":
			return "end"
		return "end"