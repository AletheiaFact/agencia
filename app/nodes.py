from crew.agents import CheckVerifiabilityAgent

class Nodes():
	def __init__(self) -> None:
		self.agent = CheckVerifiabilityAgent.agent()

	def check_claim_node(self, state):
		can_be_fact_checked = False
		claim = state["claim"]
		print("self.agent", self.agent)
		result = self.agent.invoke({"claim": claim })

		if "YES" in result:
			can_be_fact_checked = True  
  
		return {
			**state,
			"messages": result,
			"can_be_fact_checked": can_be_fact_checked
		}
  
	def router(self, state):
		messages = state["messages"]

		if "YES" in messages:
			return "continue"
		if "NO" in messages:
			return "end"
		return "end"