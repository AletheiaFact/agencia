"""Main workflow graph using LangGraph 0.6+."""

from langgraph.graph import END, StateGraph

from state import AgentState
from nodes.questions import list_questions
from nodes.router import router
from nodes.online_research import search_online
from nodes.report import create_report
from nodes.gazette.subgraph import build_gazette_subgraph


def build_workflow():
    """Build the main fact-checking workflow.

    Flow:
      list_questions -> router
        -> "gazettes": gazette subgraph -> END
        -> "online":  search_online -> create_report -> END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("list_questions", list_questions)
    workflow.add_node("search_online", search_online)
    workflow.add_node("create_report", create_report)
    workflow.add_node("fact_check_gazettes", build_gazette_subgraph())

    # Entry point
    workflow.set_entry_point("list_questions")

    # Conditional routing after question generation
    workflow.add_conditional_edges(
        "list_questions",
        router,
        {
            "continue": "fact_check_gazettes",
            "end": "search_online",
        },
    )

    # Online path
    workflow.add_edge("search_online", "create_report")
    workflow.add_edge("create_report", END)

    # Gazette path
    workflow.add_edge("fact_check_gazettes", END)

    return workflow.compile()
