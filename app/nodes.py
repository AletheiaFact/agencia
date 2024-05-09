from crew.agents import Agents

class Nodes():
	def __init__(self) -> None:
		self.checkVerifiabilityAgent = Agents.checkVerifiabilityAgent()
		self.listQuestionsAgent = Agents.listQuestionsAgent()
		self.researcherAgent = Agents.researcherAgent()
		self.factCheckerAgent = Agents.factCheckerAgent()

	def check_claim_node(self, state):
		can_be_fact_checked = False
		claim = state["claim"]
		result = self.checkVerifiabilityAgent.invoke({"claim": claim })

		if "YES" in result:
			can_be_fact_checked = True  
  
		return {
			**state,
			"messages": [result],
			"can_be_fact_checked": can_be_fact_checked
		}
  
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
		result = self.researcherAgent.invoke({"claim": claim, "sources": context["sources"] })
  
		return {
			**state,
			"messages": [result],
		}
  
	def create_report(self, state):
		claim = state["claim"]
		messages = state["messages"]
		questions = state["questions"]

		result = self.factCheckerAgent.invoke({
      		"claim": claim,
			"messages": messages,
			"questions": questions
		})
  
		return {
			**state,
			"messages": result,
		}
  
	def router(self, state):
		messages = state["messages"]

		if "YES" in messages[0]:
			return "continue"
		if "NO" in messages[0]:
			return "end"
		return "end"