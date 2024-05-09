from typing import TypedDict

class AgentState(TypedDict):
    claim: str
    context: object
    messages: dict = []
    questions: dict = []
    can_be_fact_checked: bool = None
    language: str = "Portuguese"