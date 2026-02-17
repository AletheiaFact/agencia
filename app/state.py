from __future__ import annotations

from typing import TypedDict, Optional


class Context(TypedDict, total=False):
    published_since: Optional[str]
    published_until: Optional[str]
    city: Optional[str]
    sources: Optional[list[str]]
    search_type: Optional[str]


class AgentState(TypedDict, total=False):
    # Core fields (from original)
    claim: str
    context: Context
    messages: dict
    questions: list[str]
    can_be_fact_checked: bool
    search_type: str
    language: str

    # Gazette pipeline intermediate fields (new - replaces CrewAI internal state)
    search_subject: Optional[str]
    gazette_data: Optional[dict]
    gazette_url: Optional[str]
    gazette_analysis: Optional[str]
    cross_check_result: Optional[str]
