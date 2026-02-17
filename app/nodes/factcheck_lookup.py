"""Node: Check existing fact-check databases before running expensive pipelines."""

import logging

from state import AgentState
from plugins.registry import get as get_plugin

logger = logging.getLogger(__name__)


def check_existing_factchecks(state: AgentState) -> dict:
    """Query existing fact-check databases for prior coverage of this claim.

    If the Google Fact Check plugin is available, searches for existing
    verifications. Results are stored in state for the router to evaluate.
    """
    claim = state["claim"]
    language = state.get("language", "pt")
    logger.info("[check_existing_factchecks] claim='%s'", claim[:80])

    plugin = get_plugin("google_factcheck")
    if plugin is None or not plugin.is_available():
        logger.info("[check_existing_factchecks] Google Fact Check plugin not available, skipping")
        return {
            "existing_factchecks": [],
            "reasoning_log": [
                "[check_existing_factchecks] Google Fact Check plugin not available, skipped"
            ],
        }

    result = plugin.search(query=claim, language_code=language)

    if result.error:
        logger.warning("[check_existing_factchecks] Plugin error: %s", result.error)
        return {
            "existing_factchecks": [],
            "reasoning_log": [
                f"[check_existing_factchecks] Plugin error: {result.error}"
            ],
        }

    logger.info("[check_existing_factchecks] Found %d existing fact-checks", result.result_count)
    publishers = [fc.get("publisher", "?") for fc in result.results[:5]]
    return {
        "existing_factchecks": result.results,
        "reasoning_log": [
            f"[check_existing_factchecks] Found {result.result_count} existing fact-checks "
            f"from: {', '.join(publishers)}"
        ],
    }


def factcheck_router(state: AgentState) -> str:
    """Route based on fact-check lookup results and search_type.

    Returns:
        "found" — existing fact-check found, short-circuit to report
        "gazettes" — no match, route to gazette pipeline
        "online" — no match, route to online search pipeline
    """
    existing = state.get("existing_factchecks", [])
    if existing:
        for fc in existing:
            if fc.get("rating") and fc.get("url"):
                logger.info(
                    "[factcheck_router] Existing fact-check found (publisher=%s rating=%s), short-circuiting",
                    fc.get("publisher", "unknown"),
                    fc.get("rating", "unknown"),
                )
                return "found"

    # Fall through to existing routing logic
    search_type = state.get("search_type", "online")
    route = "gazettes" if search_type == "gazettes" else "online"
    logger.info("[factcheck_router] No existing fact-check found, routing to %s", route)
    return route
