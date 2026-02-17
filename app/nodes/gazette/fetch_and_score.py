"""Gazette node: Fetch gazettes using search strategies and score by excerpt relevance.

Calls the querido_diario_search tool with all planned strategies, then ranks
gazette candidates by how relevant their excerpts are to the claim.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState
from tools.querido_diario_search import querido_diario_search
from utils.llm_retry import invoke_with_retry

logger = logging.getLogger(__name__)

_scoring_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a relevance scoring expert. Given a fact-checking claim and gazette excerpts,
score each gazette's relevance from 0-10.

Claim: {claim}

For each gazette below, output ONLY a single integer score (0-10) on a separate line.
Score 0 = completely irrelevant, 10 = directly addresses the claim with strong evidence.

{gazette_summaries}

Output one score per line, in the same order as the gazettes above. Nothing else.""",
    ),
])


def _format_gazette_for_scoring(gazette: dict, index: int) -> str:
    """Format a gazette's metadata and excerpts for LLM scoring."""
    excerpts = gazette.get("excerpts", [])
    excerpt_text = "\n".join(f"  - {e[:300]}" for e in excerpts) if excerpts else "  (no excerpts)"
    return (
        f"Gazette {index + 1}: {gazette.get('territory_name', '?')} — "
        f"{gazette.get('date', '?')} — edition {gazette.get('edition', '?')}\n"
        f"  Matched query: {gazette.get('_matched_query', '?')}\n"
        f"  Excerpts:\n{excerpt_text}"
    )


def fetch_and_score(state: AgentState) -> dict:
    """Fetch gazettes for all search strategies and score by excerpt relevance."""
    strategies = state.get("search_strategies", [])
    context = state.get("context", {})
    claim = state["claim"]

    logger.info(
        "[fetch_and_score] Starting — %d strategies, city='%s'",
        len(strategies), context.get("city", ""),
    )

    if not strategies:
        logger.warning("[fetch_and_score] No search strategies provided")
        return {"gazette_candidates": [], "evidence_sufficient": False}

    # Call the multi-query search tool directly (no agent wrapper needed)
    search_result = querido_diario_search.invoke({
        "queries": strategies,
        "city": context.get("city"),
        "published_since": context.get("published_since"),
        "published_until": context.get("published_until"),
    })

    gazettes = search_result.get("gazettes", [])
    query_stats = search_result.get("query_stats", [])

    logger.info(
        "[fetch_and_score] Got %d unique gazettes from %d queries. Stats: %s",
        len(gazettes), len(strategies), query_stats,
    )

    if not gazettes:
        return {
            "gazette_candidates": [],
            "evidence_sufficient": False,
            "evidence_summary": "No gazettes found for any search strategy.",
            "reasoning_log": [
                f"[fetch_and_score] Searched with {len(strategies)} strategies — no gazettes found"
            ],
        }

    # Score gazettes by excerpt relevance using LLM — in batches of 10
    SCORING_BATCH_SIZE = 10
    gazettes_to_score = gazettes[:20]

    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1)
    chain = _scoring_prompt | llm | StrOutputParser()

    all_scores: list[int] = []
    for batch_start in range(0, len(gazettes_to_score), SCORING_BATCH_SIZE):
        batch = gazettes_to_score[batch_start:batch_start + SCORING_BATCH_SIZE]
        batch_summaries = "\n\n".join(
            _format_gazette_for_scoring(g, batch_start + i)
            for i, g in enumerate(batch)
        )

        logger.info(
            "[fetch_and_score] Scoring batch %d-%d of %d gazettes",
            batch_start + 1, batch_start + len(batch), len(gazettes_to_score),
        )

        scores_raw = invoke_with_retry(
            chain,
            params={"claim": claim, "gazette_summaries": batch_summaries},
            truncatable_keys=["gazette_summaries"],
        )

        # Parse scores for this batch
        score_lines = [line.strip() for line in scores_raw.strip().split("\n") if line.strip()]
        batch_scores: list[int] = []
        for line in score_lines:
            try:
                score = int(line.split()[0])
                batch_scores.append(max(0, min(10, score)))
            except (ValueError, IndexError):
                batch_scores.append(0)

        # Pad if LLM returned fewer scores than gazettes in this batch
        while len(batch_scores) < len(batch):
            batch_scores.append(0)

        all_scores.extend(batch_scores[:len(batch)])

    # Attach scores and sort
    for gazette, score in zip(gazettes_to_score, all_scores):
        gazette["_relevance_score"] = score

    scored_gazettes = sorted(gazettes_to_score, key=lambda g: g.get("_relevance_score", 0), reverse=True)

    # Build evidence summary from top excerpts
    top_excerpts = []
    for g in scored_gazettes[:5]:
        for exc in g.get("excerpts", [])[:2]:
            top_excerpts.append(f"[{g.get('territory_name', '?')} {g.get('date', '?')}] {exc[:200]}")

    evidence_summary = "\n".join(top_excerpts) if top_excerpts else "No relevant excerpts found."

    logger.info(
        "[fetch_and_score] Scored %d gazettes. Top scores: %s",
        len(scored_gazettes),
        [g.get("_relevance_score", 0) for g in scored_gazettes[:5]],
    )

    top_scores = [g.get("_relevance_score", 0) for g in scored_gazettes[:5]]
    return {
        "gazette_candidates": scored_gazettes,
        "evidence_summary": evidence_summary,
        "reasoning_log": [
            f"[fetch_and_score] Found {len(gazettes)} gazettes, scored {len(gazettes_to_score)}. "
            f"Top 5 scores: {top_scores}"
        ],
    }
