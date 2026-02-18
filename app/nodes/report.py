"""Node: Create fact-checking report from gathered evidence.

Injects the accumulated reasoning_log timeline into the structured JSON output
so callers receive a full trace of the verification pipeline.
"""

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a journalist specialized in fact-checking creates a fact-checking report relying the context gathered from the web: {messages}.

{existing_factchecks_section}

{source_confidence_section}

Use critical thinking and analytical skills to assess the accuracy and relevance of the data in relation to the following claim: {claim}.

The report should be presented in a structured JSON format. Below is the format you are to follow:

{{
"classification": "Classification from the previous user, this must be in English",
"summary": "A concise overview reflecting your classification",
"questions": {questions},
"report": "A detailed narrative of your findings and evidence",
"verification": "A detailed description of the methods and tools used for verification",
}}

classification: Assign one of the following labels based on your analysis:

- Not Fact: The information lacks evidence or a factual basis.
- Trustworthy: The information is reliable, backed by evidence or reputable sources.
- Trustworthy, but: Generally reliable, albeit with minor inaccuracies.
- Arguable: Subject to debate or different interpretations.
- Misleading: Distorts the facts, causing potential misunderstandings.
- False: Demonstrably incorrect or untrue.
- Unsustainable: Lacks long-term viability or feasibility.
- Exaggerated: Contains elements of truth but is overstated or embellished.
- Unverifiable: Cannot be substantiated through reliable sources.

- summary: Provide a succinct summary that directly supports your classification,
offering insight into your analytical reasoning.

- questions: Enumerate essential questions that were crucial in guiding your
analysis and verification process, this MUST contain at least one question.

- report: Document a comprehensive account detailing the investigative process, key findings,
and evidence that supports your conclusion.

- verification: Explain the specific methodologies and tools you employed to verify the
information, highlighting your systematic approach to substantiating the claim.

compile your response in {language}, however the classification field must remain in English""",
    ),
])

def _build_existing_factchecks_section(state: AgentState) -> str:
    """Build a prompt section summarizing existing fact-checks, if any."""
    factchecks = state.get("existing_factchecks") or []
    if not factchecks:
        return ""
    lines = ["Existing fact-check results found for this claim:"]
    for fc in factchecks:
        lines.append(
            f"- Publisher: {fc.get('publisher', 'N/A')}, "
            f"Rating: {fc.get('rating', 'N/A')}, "
            f"URL: {fc.get('url', 'N/A')}"
        )
    lines.append(
        "\nIncorporate these existing fact-checks into your analysis. "
        "If they provide a clear consensus, weigh that heavily in your classification."
    )
    return "\n".join(lines)


def _build_source_confidence_section(state: AgentState) -> str:
    """Build a prompt section with source reliability and relevance scores."""
    confidence = state.get("source_confidence") or {}
    selected = state.get("selected_sources") or []
    if not confidence and not selected:
        return ""
    lines = ["Source reliability and relevance scores:"]
    for src in selected:
        reliability = confidence.get(src.get("plugin_name", ""), "N/A")
        lines.append(
            f"- {src.get('plugin_name', 'unknown')}: reliability={reliability}, "
            f"relevance={src.get('relevance', 'N/A')} ({src.get('reason', '')})"
        )
    lines.append(
        "\nWeigh higher-reliability sources more heavily in your analysis."
    )
    return "\n".join(lines)


def _inject_reasoning_log(llm_output: str, reasoning_log: list[str]) -> str:
    """Inject the reasoning_log timeline into the LLM-generated JSON report.

    Tries to parse the LLM output as JSON and add the field cleanly.
    If parsing fails, appends it as a separate JSON block.
    """
    json_str = llm_output.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_str = "\n".join(lines).strip()

    try:
        report = json.loads(json_str)
        report["reasoning_log"] = reasoning_log
        return json.dumps(report, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        log_json = json.dumps({"reasoning_log": reasoning_log}, ensure_ascii=False, indent=2)
        return f"{llm_output}\n\n--- Reasoning Timeline ---\n{log_json}"


def create_report(state: AgentState) -> dict:
    logger.info("[create_report] Starting — claim='%s'", state["claim"][:80])
    llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=1)
    chain = _prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": state["claim"],
        "messages": state.get("messages", []),
        "questions": state.get("questions", []),
        "language": state.get("language", "pt"),
        "existing_factchecks_section": _build_existing_factchecks_section(state),
        "source_confidence_section": _build_source_confidence_section(state),
    })

    # Collect the reasoning_log accumulated so far and add our own entry
    reasoning_log = state.get("reasoning_log", [])
    reasoning_log = reasoning_log + [
        f"[create_report] Final report generated ({len(result)} chars)"
    ]

    # Inject reasoning_log into the JSON response programmatically
    # (not via LLM, so the timeline is exact)
    result = _inject_reasoning_log(result, reasoning_log)

    logger.info("[create_report] Completed — report length=%d chars", len(result))
    return {
        "messages": result,
        "reasoning_log": [
            f"[create_report] Final report generated ({len(result)} chars)"
        ],
    }
