"""Node: Route claims to the appropriate search pipeline."""

from state import AgentState


def router(state: AgentState) -> str:
    search_type = state.get("search_type", "online")
    if search_type == "gazettes":
        return "continue"
    return "end"
