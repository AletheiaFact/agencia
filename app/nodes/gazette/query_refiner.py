"""Gazette node: Refine search queries based on what was/wasn't found.

Adjusts the search strategies for the next loop iteration, using knowledge
of what the previous queries returned.
"""

import logging
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState

logger = logging.getLogger(__name__)

_context_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "querido_diario_search_context.txt"
)
with open(_context_path, "r", encoding="utf-8") as f:
    _search_context = f.read()

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a search refinement expert for Brazilian municipal gazettes.

The previous search queries did not return sufficient evidence for the claim.
Analyze what went wrong and generate 3 improved queries.

Claim: {claim}
City: {city}

Previous queries that were tried:
{previous_strategies}

What was found (top excerpts):
{evidence_summary}

Search operator reference:
{search_context}

REFINEMENT STRATEGIES:
- If previous queries were too specific → broaden with | (OR) operators, use prefix wildcards (*)
- If previous queries were too broad → add more specific terms with + (AND), use exact phrases
- If wrong concepts → try different government terminology (portaria, decreto, edital, ata, resolução)
- If no results at all → try shorter queries with just 1-2 key terms
- Consider: the gazette might use formal/legal language different from the claim

Generate exactly 3 NEW queries (different from previous), one per line, NO numbering or bullets.
All queries must be in Portuguese.""",
    ),
])


def refine_queries(state: AgentState) -> dict:
    """Refine search queries for the next iteration."""
    claim = state["claim"]
    context = state.get("context", {})
    previous = state.get("search_strategies", [])
    evidence = state.get("evidence_summary", "No results found.")
    iteration = state.get("search_iteration", 0)

    logger.info(
        "[refine_queries] Starting — iteration=%d, previous_strategies=%d",
        iteration, len(previous),
    )

    llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=1)
    chain = _prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": claim,
        "city": context.get("city", ""),
        "previous_strategies": "\n".join(f"- {s}" for s in previous),
        "evidence_summary": evidence[:2000],
        "search_context": _search_context,
    })

    new_strategies = [line.strip() for line in result.strip().split("\n") if line.strip()]
    new_strategies = new_strategies[:3]

    logger.info(
        "[refine_queries] Generated %d refined strategies: %s",
        len(new_strategies), new_strategies,
    )

    strategies_display = "; ".join(f'"{s}"' for s in new_strategies)
    return {
        "search_strategies": new_strategies,
        "search_iteration": iteration + 1,
        "reasoning_log": [
            f"[query_refiner] Refined to {len(new_strategies)} new strategies: {strategies_display}"
        ],
    }
