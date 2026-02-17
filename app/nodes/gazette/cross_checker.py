"""Gazette node: Cross-check collected gazette data against the claim."""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert data analyst specialized in creating fact-checking reports.

Your goal: Perform a deep and thorough analysis to derive an informed report with the received
information using critical thinking and analytical skills to assess the accuracy and relevance
of the data.

Claim: {claim}
Gazette analysis: {gazette_analysis}

Your task is to perform a deep and thorough analysis to derive an informed conclusion by comparing
the collected gazette data with the specific claim provided.

Use critical thinking and analytical skills to assess the accuracy and relevance of the data
in relation to the claim.

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

Compile a comprehensive report detailing the investigative process, key findings, and evidence
that supports your conclusion containing all relevant data between the claim and the gazette.""",
    ),
])

def cross_check(state: AgentState) -> dict:
    logger.info("[cross_check] Starting — claim='%s' analysis_length=%d", state["claim"][:80], len(state.get("gazette_analysis", "")))
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1)
    chain = _prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": state["claim"],
        "gazette_analysis": state.get("gazette_analysis", ""),
    })
    logger.info("[cross_check] Completed — result length=%d chars", len(result))
    return {"cross_check_result": result}
