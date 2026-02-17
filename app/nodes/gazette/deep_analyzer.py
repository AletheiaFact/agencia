"""Gazette node: Download top gazettes and perform deep FAISS analysis.

Uses map-reduce to handle large documents without losing evidence:
1. MAP: FAISS returns relevant passages, split into batches
2. Each batch gets a mini-summary from the LLM
3. REDUCE: All mini-summaries are merged into a final synthesis
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState
from tools.gazette_deep_search import gazette_deep_search
from utils.llm_retry import invoke_with_retry

logger = logging.getLogger(__name__)

MAX_GAZETTES = 3
BATCH_CHAR_LIMIT = 4000  # chars per batch sent to LLM

_batch_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert analyst for Brazilian municipal gazettes.

Given a claim and a batch of gazette passages, extract ALL relevant evidence:
- Names, dates, contract numbers, monetary values, decree numbers
- Specific facts that support OR contradict the claim
- Context that helps evaluate the claim's accuracy

Be thorough — do not omit any detail that could be relevant.

Claim: {claim}
Questions: {questions}

Passages (batch {batch_number} of {total_batches}):
{passages}

Extract all relevant evidence from these passages. Be specific and cite details.""",
    ),
])

_merge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert data analyst for Brazilian municipal gazettes.

You have received evidence summaries extracted from multiple batches of gazette passages.
Synthesize them into a single comprehensive analysis covering:

1. What specific evidence was found in the gazettes
2. Which gazette(s) contain the most relevant information
3. Key facts: names, dates, contract numbers, monetary values, decree numbers
4. Whether the evidence supports, contradicts, or is ambiguous about the claim
5. Any contradictions or inconsistencies across different passages

Claim: {claim}
Questions: {questions}

Evidence from all batches:
{batch_summaries}

Provide a thorough, evidence-based analysis. Cite specific details.
If the evidence is insufficient, say so clearly.""",
    ),
])


def _split_into_batches(text: str, limit: int) -> list[str]:
    """Split text into batches respecting the char limit, splitting on passage boundaries."""
    if not text:
        return []

    # Split on passage boundaries (the "---" separator from gazette_deep_search)
    passages = text.split("\n\n---\n\n")
    batches = []
    current_batch = []
    current_len = 0

    for passage in passages:
        passage_len = len(passage) + 10  # account for separators
        if current_len + passage_len > limit and current_batch:
            batches.append("\n\n---\n\n".join(current_batch))
            current_batch = [passage]
            current_len = passage_len
        else:
            current_batch.append(passage)
            current_len += passage_len

    if current_batch:
        batches.append("\n\n---\n\n".join(current_batch))

    return batches


def download_and_analyze(state: AgentState) -> dict:
    """Download top gazettes and perform deep multi-document analysis with map-reduce."""
    candidates = state.get("gazette_candidates", [])
    claim = state["claim"]
    questions = state.get("questions", [])

    logger.info(
        "[download_and_analyze] Starting — %d candidates available",
        len(candidates),
    )

    if not candidates:
        logger.warning("[download_and_analyze] No candidates to analyze")
        return {
            "gazette_analysis": "No gazette documents were available for analysis.",
            "selected_gazettes": [],
            "reasoning_log": ["[deep_analyzer] No candidates available for analysis"],
        }

    # Select top N gazettes by relevance score
    top_gazettes = sorted(
        candidates,
        key=lambda g: g.get("_relevance_score", 0),
        reverse=True,
    )[:MAX_GAZETTES]

    urls = [g.get("txt_url", "") for g in top_gazettes if g.get("txt_url")]

    if not urls:
        logger.warning("[download_and_analyze] No valid txt_urls in top candidates")
        return {
            "gazette_analysis": "No valid gazette URLs found for download.",
            "selected_gazettes": top_gazettes,
        }

    logger.info(
        "[download_and_analyze] Downloading %d gazettes for deep analysis",
        len(urls),
    )

    # Call the multi-document FAISS search tool
    questions_str = "\n".join(questions) if questions else ""
    passages = gazette_deep_search.invoke({
        "claim": claim,
        "urls": urls,
        "questions": questions_str,
    })

    if passages.startswith("Error:"):
        logger.error("[download_and_analyze] Deep search failed: %s", passages)
        return {
            "gazette_analysis": passages,
            "selected_gazettes": top_gazettes,
        }

    # MAP phase: split passages into batches and extract evidence from each
    batches = _split_into_batches(passages, BATCH_CHAR_LIMIT)
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1)
    questions_display = questions_str or "No specific questions provided."

    analysis_method = ""
    if len(batches) <= 1:
        # Single batch — no need for map-reduce, just synthesize directly
        logger.info("[download_and_analyze] Single batch (%d chars), direct synthesis", len(passages))
        analysis_method = "direct synthesis"
        chain = _merge_prompt | llm | StrOutputParser()
        analysis = invoke_with_retry(
            chain,
            params={
                "claim": claim,
                "questions": questions_display,
                "batch_summaries": passages,
            },
            truncatable_keys=["batch_summaries"],
        )
    else:
        logger.info(
            "[download_and_analyze] Map-reduce: %d batches from %d chars of passages",
            len(batches), len(passages),
        )

        # MAP: extract evidence from each batch
        batch_chain = _batch_prompt | llm | StrOutputParser()
        batch_summaries = []

        for i, batch_text in enumerate(batches):
            logger.info(
                "[download_and_analyze] Processing batch %d/%d (%d chars)",
                i + 1, len(batches), len(batch_text),
            )
            summary = invoke_with_retry(
                batch_chain,
                params={
                    "claim": claim,
                    "questions": questions_display,
                    "passages": batch_text,
                    "batch_number": str(i + 1),
                    "total_batches": str(len(batches)),
                },
                truncatable_keys=["passages"],
            )
            batch_summaries.append(f"=== Batch {i + 1}/{len(batches)} ===\n{summary}")

        # REDUCE: merge all batch summaries into final analysis
        all_summaries = "\n\n".join(batch_summaries)
        logger.info(
            "[download_and_analyze] Reduce phase: merging %d summaries (%d chars)",
            len(batch_summaries), len(all_summaries),
        )

        analysis_method = f"map-reduce ({len(batches)} batches)"
        merge_chain = _merge_prompt | llm | StrOutputParser()
        analysis = invoke_with_retry(
            merge_chain,
            params={
                "claim": claim,
                "questions": questions_display,
                "batch_summaries": all_summaries,
            },
            truncatable_keys=["batch_summaries"],
        )

    logger.info(
        "[download_and_analyze] Completed — analysis length=%d chars, gazettes=%d",
        len(analysis), len(top_gazettes),
    )

    gazette_names = [f"{g.get('territory_name', '?')} ({g.get('date', '?')})" for g in top_gazettes]
    return {
        "gazette_analysis": analysis,
        "selected_gazettes": [
            {
                "territory_name": g.get("territory_name", ""),
                "date": g.get("date", ""),
                "txt_url": g.get("txt_url", ""),
                "url": g.get("url", ""),
                "relevance_score": g.get("_relevance_score", 0),
            }
            for g in top_gazettes
        ],
        "reasoning_log": [
            f"[deep_analyzer] Downloaded and analyzed {len(urls)} gazettes via {analysis_method}: "
            + ", ".join(gazette_names)
        ],
    }
