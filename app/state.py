from typing import TypedDict, List, Optional

class Context(TypedDict, total=False):
    published_since: Optional[str]
    published_until: Optional[str]
    city: Optional[str]
    sources: Optional[List[str]]

class AgentState(TypedDict):
    claim: str
    context: Context
    messages: dict = []
    questions: List[str] = []
    can_be_fact_checked: bool = None
    language: str = "Portuguese"