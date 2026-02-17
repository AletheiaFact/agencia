"""Main workflow graph using LangGraph 0.6+."""

from langgraph.graph import END, StateGraph

from state import AgentState
from nodes.questions import list_questions
from nodes.factcheck_lookup import check_existing_factchecks, factcheck_router
from nodes.online_research import search_online
from nodes.report import create_report
from nodes.gazette.subgraph import build_gazette_subgraph


def build_workflow():
    """Build the main fact-checking workflow.

    Flow:
      list_questions -> check_existing_factchecks -> factcheck_router
        -> "found":    create_report -> END   (short-circuit for known claims)
        -> "gazettes": gazette subgraph -> END
        -> "online":   search_online -> create_report -> END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("list_questions", list_questions)
    workflow.add_node("check_existing_factchecks", check_existing_factchecks)
    workflow.add_node("search_online", search_online)
    workflow.add_node("create_report", create_report)
    workflow.add_node("fact_check_gazettes", build_gazette_subgraph())

    # Entry point
    workflow.set_entry_point("list_questions")

    # After questions, check existing fact-checks first
    workflow.add_edge("list_questions", "check_existing_factchecks")

    # 3-way routing after fact-check lookup
    workflow.add_conditional_edges(
        "check_existing_factchecks",
        factcheck_router,
        {
            "found": "create_report",
            "gazettes": "fact_check_gazettes",
            "online": "search_online",
        },
    )

    # Online path
    workflow.add_edge("search_online", "create_report")
    workflow.add_edge("create_report", END)

    # Gazette path
    workflow.add_edge("fact_check_gazettes", END)

    return workflow.compile()
