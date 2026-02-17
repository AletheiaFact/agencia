"""Gazette node: Plan diverse search strategies from a claim.

Generates 2-3 Querido Diário search queries using Elasticsearch simple syntax
operators, designed to maximize the chance of finding relevant gazette evidence.
"""

import logging
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState

logger = logging.getLogger(__name__)

# Load search context (advanced search operators guide)
_context_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "querido_diario_search_context.txt"
)
with open(_context_path, "r", encoding="utf-8") as f:
    _search_context = f.read()

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert search strategist for Brazilian municipal gazettes (Diários Oficiais).

Your goal: Generate 2-3 diverse search queries to find gazette evidence about a claim.
Each query should use a DIFFERENT search strategy to maximize recall.

Here is the Querido Diário advanced search operator reference:
{search_context}

IMPORTANT RULES:
- All queries must be in Portuguese
- Use the Elasticsearch simple query syntax operators from the reference above
- Generate exactly 3 queries, one per line, with NO numbering, bullets, or other formatting
- Each query should target a different aspect or use different operators

STRATEGY GUIDE:
- Query 1 (Exact): Use exact phrases with quotes and + (AND) for specific entities or terms from the claim
- Query 2 (Broad): Use | (OR) operators with related synonyms and broader terms
- Query 3 (Contextual): Target the government process that would produce the gazette entry (e.g., licitação, decreto, portaria, contrato)

Example for a claim about hospital medication procurement:
"licitacao" + "medicamentos" + "hospital"
medicamentos | remedios | farmacos + saude
"pregao eletronico" + (medicamentos | insumos) + hospital

Claim: {claim}
City: {city}
Date range: {published_since} to {published_until}""",
    ),
])


def plan_search(state: AgentState) -> dict:
    """Generate 2-3 search strategies from the claim and context."""
    context = state.get("context", {})
    city = context.get("city", "")
    claim = state["claim"]

    logger.info("[plan_search] Starting — claim='%s' city='%s'", claim[:80], city)

    llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=1)
    chain = _prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": claim,
        "city": city,
        "published_since": context.get("published_since", ""),
        "published_until": context.get("published_until", ""),
        "search_context": _search_context,
    })

    # Parse output into list of queries (one per line, skip empty)
    strategies = [line.strip() for line in result.strip().split("\n") if line.strip()]
    # Limit to 3 strategies max
    strategies = strategies[:3]

    logger.info("[plan_search] Generated %d strategies: %s", len(strategies), strategies)
    strategies_display = "; ".join(f'"{s}"' for s in strategies)
    return {
        "search_strategies": strategies,
        "search_iteration": 0,
        "evidence_sufficient": False,
        "reasoning_log": [
            f"[plan_search] Generated {len(strategies)} search strategies: {strategies_display}"
        ],
    }
