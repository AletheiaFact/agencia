"""Gazette node: Evaluate whether collected evidence is sufficient.

Loop control node — decides whether to proceed to deep analysis
or refine the search queries for another iteration.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an evidence sufficiency evaluator for a fact-checking pipeline.

Given a claim and the top gazette excerpts found so far, decide:
- SUFFICIENT: The excerpts contain information directly relevant to verifying or refuting the claim.
- INSUFFICIENT: The excerpts are irrelevant, too vague, or don't address the claim at all.

Claim: {claim}

Search iteration: {search_iteration} of 3 (max)
Search strategies used: {strategies}

Top gazette excerpts found:
{evidence_summary}

Respond with EXACTLY one word: SUFFICIENT or INSUFFICIENT

Guidelines:
- If excerpts mention the same topic, entities, or events as the claim → SUFFICIENT
- If excerpts are about unrelated topics or only tangentially related → INSUFFICIENT
- If no excerpts were found at all → INSUFFICIENT
- On iteration 3, always respond SUFFICIENT (we must proceed with whatever we have)""",
    ),
])


def evaluate_evidence(state: AgentState) -> dict:
    """Evaluate whether current evidence is sufficient to proceed."""
    claim = state["claim"]
    iteration = state.get("search_iteration", 0)
    evidence_summary = state.get("evidence_summary", "")
    strategies = state.get("search_strategies", [])
    candidates = state.get("gazette_candidates", [])

    logger.info(
        "[evaluate_evidence] Starting — iteration=%d candidates=%d",
        iteration, len(candidates),
    )

    # Force sufficient on max iteration or if we have high-scoring candidates
    if iteration >= 2:
        logger.info("[evaluate_evidence] Max iterations reached, forcing SUFFICIENT")
        return {
            "evidence_sufficient": True,
            "reasoning_log": [
                f"[evaluate_evidence] Iteration {iteration + 1}/3 — max reached, proceeding with available evidence"
            ],
        }

    # If no candidates at all, definitely insufficient
    if not candidates:
        logger.info("[evaluate_evidence] No candidates found, INSUFFICIENT")
        return {
            "evidence_sufficient": False,
            "reasoning_log": [
                f"[evaluate_evidence] Iteration {iteration + 1}/3 — no candidates found, refining queries"
            ],
        }

    # Check if top candidate has a strong relevance score
    top_score = candidates[0].get("_relevance_score", 0) if candidates else 0
    if top_score >= 7:
        logger.info("[evaluate_evidence] Top score %d >= 7, SUFFICIENT without LLM", top_score)
        return {
            "evidence_sufficient": True,
            "reasoning_log": [
                f"[evaluate_evidence] Iteration {iteration + 1}/3 — top score {top_score}/10, evidence sufficient"
            ],
        }

    # Use LLM for borderline cases
    llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=1)
    chain = _prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": claim,
        "search_iteration": iteration + 1,
        "strategies": "\n".join(strategies),
        "evidence_summary": evidence_summary[:3000],
    })

    is_sufficient = "SUFFICIENT" in result.strip().upper() and "INSUFFICIENT" not in result.strip().upper()
    logger.info("[evaluate_evidence] LLM verdict: '%s' → sufficient=%s", result.strip(), is_sufficient)

    verdict = "sufficient — proceeding to analysis" if is_sufficient else "insufficient — refining queries"
    return {
        "evidence_sufficient": is_sufficient,
        "reasoning_log": [
            f"[evaluate_evidence] Iteration {iteration + 1}/3 — LLM assessed evidence as {verdict}"
        ],
    }
