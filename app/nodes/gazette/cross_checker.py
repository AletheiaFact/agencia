"""Gazette node: Cross-check collected gazette data against the claim.

Upgraded to use gpt-5 for stronger reasoning and accept richer evidence
including evidence_summary and selected_gazettes metadata.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from state import AgentState
from utils.llm_retry import invoke_with_retry

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an independent fact-checking evaluator. You receive evidence from multiple sources
and must cross-reference them to reach an impartial classification.

CRITICAL: You receive three types of evidence below. The AI-generated analysis may contain bias.
You MUST cross-reference it against the raw gazette passages. If the analysis claims something
not supported by the raw passages, note the discrepancy. If the raw passages contain relevant
information the analysis omitted, factor it into your classification.

Claim: {claim}

--- RAW GAZETTE PASSAGES (not filtered by AI — review independently) ---
{raw_passages}

--- AI-GENERATED ANALYSIS (may contain bias — cross-reference with raw passages above) ---
{gazette_analysis}

--- CONTRADICTORY EVIDENCE (independently extracted — weigh carefully) ---
{contradictory_evidence}

--- EVIDENCE SUMMARY (key search excerpts) ---
{evidence_summary}

--- GAZETTES ANALYZED ---
{gazette_metadata}

CLASSIFICATION — Assign one of the following labels:

- Not Fact: The information lacks evidence or a factual basis.
- Trustworthy: The information is reliable, backed by evidence or reputable sources.
- Trustworthy, but: Generally reliable, albeit with minor inaccuracies or caveats.
- Arguable: Subject to debate or different interpretations.
- Misleading: Distorts the facts, causing potential misunderstandings.
- False: Demonstrably incorrect or untrue.
- Unsustainable: Lacks long-term viability or feasibility.
- Exaggerated: Contains elements of truth but is overstated or embellished.
- Unverifiable: Cannot be substantiated through the gazette sources searched.

REASONING GUIDELINES:
- Pay careful attention to dates, names, and specific details in both the claim and evidence
- Distinguish between similar but different events (e.g., "conducted exam" vs. "called approved candidates")
- If evidence partially supports the claim, use "Trustworthy, but" or "Arguable" rather than "Trustworthy"
- If no relevant evidence was found after searching, use "Unverifiable"
- Cite specific gazette data (dates, contract numbers, names) to support your classification
- Note any discrepancies between the AI analysis and the raw passages

CONFIDENCE ASSESSMENT — After your classification, provide:
- Evidence quality: high/medium/low (how relevant are the raw passages to the claim?)
- Analysis accuracy: high/medium/low (does the AI analysis faithfully represent the raw passages?)
- Overall confidence: high/medium/low (how confident are you in this classification?)

Compile a comprehensive report detailing your investigative process, key findings, and evidence.""",
    ),
])


def cross_check(state: AgentState) -> dict:
    """Cross-check gazette evidence against the claim with independent verification.

    Receives raw passages alongside LLM summaries so the classifier can
    cross-reference and detect bias in the AI-generated analysis.
    """
    claim = state["claim"]
    gazette_analysis = state.get("gazette_analysis", "")
    evidence_summary = state.get("evidence_summary", "")
    raw_passages = state.get("raw_gazette_passages", "")
    contradictory_evidence = state.get("contradictory_evidence", "")
    selected = state.get("selected_gazettes", [])

    # Format gazette metadata
    if selected:
        metadata_lines = []
        for g in selected:
            metadata_lines.append(
                f"- {g.get('territory_name', '?')} ({g.get('date', '?')}) "
                f"— relevance: {g.get('relevance_score', '?')}/10 "
                f"— {g.get('txt_url', 'no URL')}"
            )
        gazette_metadata = "\n".join(metadata_lines)
    else:
        gazette_metadata = "No gazette documents were analyzed."

    # Truncate inputs for context window management
    raw_passages_truncated = raw_passages[:4000] if raw_passages else "No raw passages available."
    contra_truncated = contradictory_evidence[:2000] if contradictory_evidence else "No contradictory evidence extracted."

    logger.info(
        "[cross_check] Starting — claim='%s' analysis_len=%d raw_len=%d contra_len=%d gazettes=%d",
        claim[:80], len(gazette_analysis), len(raw_passages_truncated),
        len(contra_truncated), len(selected),
    )

    llm = get_llm()
    chain = _prompt | llm | StrOutputParser()
    result = invoke_with_retry(
        chain,
        params={
            "claim": claim,
            "gazette_analysis": gazette_analysis[:5000],
            "evidence_summary": evidence_summary[:2000],
            "gazette_metadata": gazette_metadata,
            "raw_passages": raw_passages_truncated,
            "contradictory_evidence": contra_truncated,
        },
        truncatable_keys=["gazette_analysis", "evidence_summary", "raw_passages"],
    )

    logger.info("[cross_check] Completed — result length=%d chars", len(result))
    # Extract a brief snippet for the reasoning log
    snippet = result[:150].replace("\n", " ").strip()
    return {
        "cross_check_result": result,
        "reasoning_log": [
            f"[cross_checker] Cross-checked evidence against claim ({len(selected)} gazettes, "
            f"{len(gazette_analysis)} chars of analysis): {snippet}..."
        ],
    }
