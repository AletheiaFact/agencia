"""Gazette node: Cross-check collected gazette data against the claim.

Upgraded to use gpt-5 for stronger reasoning and accept richer evidence
including evidence_summary and selected_gazettes metadata.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState
from utils.llm_retry import invoke_with_retry

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert data analyst specialized in creating fact-checking reports.

Your goal: Perform a deep and thorough analysis to derive an informed conclusion by comparing
collected gazette evidence with the claim. Use critical thinking and analytical skills to assess
the accuracy and relevance of the data.

Claim: {claim}

Gazette deep analysis (from multiple documents):
{gazette_analysis}

Evidence summary (key excerpts from search):
{evidence_summary}

Gazettes analyzed:
{gazette_metadata}

CLASSIFICATION — Assign one of the following labels based on your analysis:

- Not Fact: The information lacks evidence or a factual basis.
- Trustworthy: The information is reliable, backed by evidence or reputable sources.
- Trustworthy, but: Generally reliable, albeit with minor inaccuracies or caveats.
- Arguable: Subject to debate or different interpretations.
- Misleading: Distorts the facts, causing potential misunderstandings.
- False: Demonstrably incorrect or untrue.
- Unsustainable: Lacks long-term viability or feasibility.
- Exaggerated: Contains elements of truth but is overstated or embellished.
- Unverifiable: Cannot be substantiated through the gazette sources searched.

IMPORTANT REASONING GUIDELINES:
- Pay careful attention to dates, names, and specific details in both the claim and evidence
- Distinguish between similar but different events (e.g., "conducted exam" vs. "called approved candidates")
- If evidence partially supports the claim, use "Trustworthy, but" or "Arguable" rather than "Trustworthy"
- If no relevant evidence was found after searching, use "Unverifiable"
- Cite specific gazette data (dates, contract numbers, names) to support your classification

Compile a comprehensive report detailing your investigative process, key findings, and evidence.""",
    ),
])


def cross_check(state: AgentState) -> dict:
    """Cross-check gazette evidence against the claim using gpt-5."""
    claim = state["claim"]
    gazette_analysis = state.get("gazette_analysis", "")
    evidence_summary = state.get("evidence_summary", "")
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

    logger.info(
        "[cross_check] Starting — claim='%s' analysis_len=%d gazettes=%d",
        claim[:80], len(gazette_analysis), len(selected),
    )

    llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=1)
    chain = _prompt | llm | StrOutputParser()
    result = invoke_with_retry(
        chain,
        params={
            "claim": claim,
            "gazette_analysis": gazette_analysis[:6000],
            "evidence_summary": evidence_summary[:3000],
            "gazette_metadata": gazette_metadata,
        },
        truncatable_keys=["gazette_analysis", "evidence_summary"],
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
