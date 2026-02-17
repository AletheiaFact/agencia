from __future__ import annotations

import operator
from typing import Annotated, TypedDict, Optional


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

    # Gazette pipeline — adaptive search loop fields
    search_strategies: Optional[list[str]]
    gazette_candidates: Optional[list[dict]]
    evidence_sufficient: Optional[bool]
    search_iteration: Optional[int]
    selected_gazettes: Optional[list[dict]]
    evidence_summary: Optional[str]
    gazette_analysis: Optional[str]
    cross_check_result: Optional[str]
    reasoning_log: Annotated[list[str], operator.add]

    # Existing fact-check lookup results (Phase 1 tooling expansion)
    existing_factchecks: Optional[list[dict]]
