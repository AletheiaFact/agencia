from crew.agents import CheckVerifiabilityAgent, ResearcherAgent, ListQuestionsAgent

class Nodes():
	def __init__(self) -> None:
		self.checkVerifiabilityAgent = CheckVerifiabilityAgent.agent()
		self.listQuestionsAgent = ListQuestionsAgent.agent()
		self.researcherAgent = ResearcherAgent.agent()

	def check_claim_node(self, state):
		can_be_fact_checked = False
		claim = state["claim"]
		result = self.checkVerifiabilityAgent.invoke({"claim": claim, "language": state["claim"]})

		if "YES" in result:
			can_be_fact_checked = True  
  
		return {
			**state,
			"messages": result,
			"can_be_fact_checked": can_be_fact_checked
		}
  
	def list_questions(self, state):
		claim = state["claim"]
		result = self.listQuestionsAgent.invoke({"claim": claim })

		return {
			**state,
			"questions": result,
		}
  
	def search_online(self, state):
		claim = state["claim"]
		context = state["context"]
		result = self.researcherAgent.invoke({"claim": claim, "sources": context["sources"] })
  
		return {
			**state,
			"messages": [*state["messages"], result],
		}
  
	def router(self, state):
		messages = state["messages"]

		if "YES" in messages:
			return "continue"
		if "NO" in messages:
			return "end"
		return "end"