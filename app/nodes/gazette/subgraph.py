"""Gazette subgraph: Adaptive search loop for gazette fact-checking.

Replaces the linear pipeline with a loop that:
1. Plans diverse search strategies
2. Fetches and scores gazette candidates by excerpt relevance
3. Evaluates evidence sufficiency
4. Refines queries if insufficient (max 3 iterations)
5. Downloads top gazettes for deep FAISS analysis
6. Cross-checks evidence against the claim
7. Produces the final report
"""

from langgraph.graph import StateGraph, END

from state import AgentState
from nodes.gazette.search_planner import plan_search
from nodes.gazette.fetch_and_score import fetch_and_score
from nodes.gazette.evidence_evaluator import evaluate_evidence
from nodes.gazette.query_refiner import refine_queries
from nodes.gazette.deep_analyzer import download_and_analyze
from nodes.gazette.cross_checker import cross_check
from nodes.gazette.gazette_reporter import create_gazette_report


def _should_continue_search(state: AgentState) -> str:
    """Conditional edge: proceed to download or refine queries."""
    if state.get("evidence_sufficient", False):
        return "download"
    return "refine"


def build_gazette_subgraph() -> StateGraph:
    """Build the gazette fact-checking subgraph with adaptive search loop.

    Pipeline:
      plan_search
        -> fetch_and_score -> evaluate_evidence
          -> (sufficient) -> download_and_analyze -> cross_check -> gazette_report -> END
          -> (insufficient) -> refine_queries -> fetch_and_score -> ... (loop)
    """
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("plan_search", plan_search)
    workflow.add_node("fetch_and_score", fetch_and_score)
    workflow.add_node("evaluate_evidence", evaluate_evidence)
    workflow.add_node("refine_queries", refine_queries)
    workflow.add_node("download_and_analyze", download_and_analyze)
    workflow.add_node("cross_check", cross_check)
    workflow.add_node("gazette_report", create_gazette_report)

    # Entry point
    workflow.set_entry_point("plan_search")

    # Linear edges
    workflow.add_edge("plan_search", "fetch_and_score")
    workflow.add_edge("fetch_and_score", "evaluate_evidence")

    # Conditional loop edge
    workflow.add_conditional_edges(
        "evaluate_evidence",
        _should_continue_search,
        {
            "download": "download_and_analyze",
            "refine": "refine_queries",
        },
    )

    # Refinement loops back to fetch
    workflow.add_edge("refine_queries", "fetch_and_score")

    # Post-download linear path
    workflow.add_edge("download_and_analyze", "cross_check")
    workflow.add_edge("cross_check", "gazette_report")
    workflow.add_edge("gazette_report", END)

    return workflow.compile()
