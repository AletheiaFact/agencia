from typing import TypedDict

class AgentState(TypedDict):
    claim: str
    context: object
    messages: dict
    can_be_fact_checked: bool
    language: str
    #steps_description