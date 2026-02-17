"""Gazette subgraph: Sequential pipeline replacing the CrewAI crew."""

from langgraph.graph import StateGraph, END

from state import AgentState
from nodes.gazette.subject_creator import create_subject
from nodes.gazette.gazette_fetcher import fetch_gazettes
from nodes.gazette.gazette_analyzer import analyze_gazette
from nodes.gazette.cross_checker import cross_check
from nodes.gazette.gazette_reporter import create_gazette_report


def build_gazette_subgraph() -> StateGraph:
    """Build the gazette fact-checking subgraph.

    Pipeline:
      create_subject -> fetch_gazettes -> analyze_gazette -> cross_check -> gazette_report -> END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("create_subject", create_subject)
    workflow.add_node("fetch_gazettes", fetch_gazettes)
    workflow.add_node("analyze_gazette", analyze_gazette)
    workflow.add_node("cross_check", cross_check)
    workflow.add_node("gazette_report", create_gazette_report)

    workflow.set_entry_point("create_subject")
    workflow.add_edge("create_subject", "fetch_gazettes")
    workflow.add_edge("fetch_gazettes", "analyze_gazette")
    workflow.add_edge("analyze_gazette", "cross_check")
    workflow.add_edge("cross_check", "gazette_report")
    workflow.add_edge("gazette_report", END)

    return workflow.compile()
