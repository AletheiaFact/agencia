from typing import TypedDict, List

class AgentState(TypedDict):
    claim: str
    context: object
    messages: dict = []
    questions: List[str] = []
    can_be_fact_checked: bool = None
    language: str = "Portuguese"