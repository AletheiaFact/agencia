"""Node: Route claims to the appropriate search pipeline."""

import logging

from state import AgentState

logger = logging.getLogger(__name__)


def router(state: AgentState) -> str:
    search_type = state.get("search_type", "online")
    route = "continue" if search_type == "gazettes" else "end"
    logger.info("[router] search_type=%s → route=%s", search_type, route)
    return route
