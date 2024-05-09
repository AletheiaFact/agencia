from langgraph.graph import END, StateGraph
from state import AgentState
from nodes import Nodes
from crew.crew import QueridoDiarioCrew
from nodes import Nodes

class WorkFlow():
	def __init__(self):
		nodes = Nodes()
		workflow = StateGraph(AgentState)
    
		workflow.add_node("check_claim", nodes.check_claim_node) #check_claim and list_questions can be a crew
		workflow.add_node("list_questions", nodes.list_questions)
		workflow.add_node("search_online", nodes.search_online)
		workflow.add_node("create_report", nodes.create_report)
		workflow.add_node("start_fact_checking", QueridoDiarioCrew().kickoff)
		workflow.set_entry_point("check_claim")
  
		workflow.add_edge("check_claim", "list_questions")
		workflow.add_conditional_edges(
      		"list_questions",
			nodes.router,
			{
       			"continue": "start_fact_checking",
				"end": "search_online"
          	}
		)
		workflow.add_edge("start_fact_checking", END)
		workflow.add_edge("search_online", "create_report")
		workflow.add_edge("create_report", END)

		self.app = workflow.compile()