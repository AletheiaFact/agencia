from langgraph.graph import END, StateGraph
from state import AgentState
from nodes import Nodes
from crew.crew import QueridoDiarioCrew
from nodes import Nodes

class WorkFlow():
	def __init__(self):
		nodes = Nodes()
		workflow = StateGraph(AgentState)
    
		workflow.add_node("check_claim", nodes.check_claim_node)
		workflow.add_node("search_online", nodes.search_online)
		workflow.add_node("list_questions", nodes.list_questions)
		workflow.add_node("start_fact_checking", QueridoDiarioCrew().kickoff)
		workflow.set_entry_point("check_claim")
  
		workflow.add_conditional_edges(
      		"check_claim",
			nodes.router,
			{
       			"continue": "list_questions",
				"end": "search_online"
          	}
		)
		workflow.add_edge("search_online", END)
		workflow.add_edge("list_questions", "start_fact_checking")
		workflow.add_edge("start_fact_checking", END)

		self.app = workflow.compile()