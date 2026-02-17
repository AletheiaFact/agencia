"""Gazette node: Create the final fact-checking report from gazette analysis.

Enhanced to include gazette source URLs, search strategies used,
confidence level, and a reasoning_log timeline in the structured JSON output.
"""

import json
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
        """You are an expert journalist specialized in fact-checking and creating detailed reports.

Create a fact-checking report in structured JSON format:

{{
"classification": "Classification in English — one of: Not Fact, Trustworthy, Trustworthy but, Arguable, Misleading, False, Unsustainable, Exaggerated, Unverifiable",
"summary": "A concise overview reflecting your classification",
"questions": {questions},
"report": "A detailed narrative of your findings and evidence",
"verification": "Methods and tools used for verification",
"sources": {gazette_sources},
"search_strategies": {search_strategies},
"confidence": "high/medium/low — based on evidence quality and quantity"
}}

Field guidelines:
- classification: MUST be in English, one of the listed labels
- summary: Succinct summary supporting the classification
- report: Comprehensive account of the investigative process, key findings, and evidence
- verification: Specific methodologies and tools used (Querido Diário API, FAISS similarity search, multi-document analysis)
- sources: List of gazette URLs that were analyzed
- search_strategies: The search queries used to find evidence
- confidence: "high" if strong direct evidence, "medium" if partial evidence, "low" if weak/no evidence

Cross-check analysis: {cross_check_result}
Claim: {claim}

Compile your response in {language}, however the classification field must remain in English.""",
    ),
])


def create_gazette_report(state: AgentState) -> dict:
    """Create the final structured fact-checking report with reasoning timeline."""
    claim = state["claim"]
    selected = state.get("selected_gazettes", [])
    strategies = state.get("search_strategies", [])
    reasoning_log = state.get("reasoning_log", [])

    gazette_sources = [
        f"{g.get('territory_name', '?')} ({g.get('date', '?')}): {g.get('txt_url', 'N/A')}"
        for g in selected
    ] if selected else ["No gazette sources analyzed"]

    logger.info(
        "[create_gazette_report] Starting — claim='%s' sources=%d reasoning_steps=%d",
        claim[:80], len(gazette_sources), len(reasoning_log),
    )

    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1)
    chain = _prompt | llm | StrOutputParser()
    result = invoke_with_retry(
        chain,
        params={
            "claim": claim,
            "questions": state.get("questions", []),
            "cross_check_result": state.get("cross_check_result", ""),
            "language": state.get("language", "pt"),
            "gazette_sources": gazette_sources,
            "search_strategies": strategies,
        },
        truncatable_keys=["cross_check_result"],
    )

    # Inject reasoning_log into the JSON response programmatically
    # (not via LLM, so the timeline is exact)
    result = _inject_reasoning_log(result, reasoning_log)

    logger.info("[create_gazette_report] Completed — report length=%d chars", len(result))
    return {
        "messages": result,
        "reasoning_log": [
            f"[gazette_reporter] Final report generated ({len(result)} chars)"
        ],
    }


def _inject_reasoning_log(llm_output: str, reasoning_log: list[str]) -> str:
    """Inject the reasoning_log timeline into the LLM-generated JSON report.

    Tries to parse the LLM output as JSON and add the field cleanly.
    If parsing fails, appends it as a separate JSON block.
    """
    # Try to find the JSON object in the LLM output (may have markdown fences)
    json_str = llm_output.strip()
    if json_str.startswith("```"):
        # Strip markdown code fences
        lines = json_str.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_str = "\n".join(lines).strip()

    try:
        report = json.loads(json_str)
        report["reasoning_log"] = reasoning_log
        return json.dumps(report, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        # If LLM output isn't valid JSON, append the log as a separate section
        log_json = json.dumps({"reasoning_log": reasoning_log}, ensure_ascii=False, indent=2)
        return f"{llm_output}\n\n--- Reasoning Timeline ---\n{log_json}"
