"""Gazette node: Create the final fact-checking report from gazette analysis."""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert journalist specialized in fact-checking and creating detailed reports.

Upon receiving data from the Data analyst, create a fact-checking report.
The report should be presented in a structured JSON format. Below is the format you are to follow:

{{
"classification": "Classification from the previous user, this MUST be in English",
"summary": "A concise overview reflecting your classification",
"questions": {questions},
"report": "A detailed narrative of your findings and evidence",
"verification": "A detailed description of the methods and tools used for verification",
}}

classification: Assign one of the following labels based on your analysis:
- ["Not Fact", "Trustworthy", "Trustworthy, but", "Arguable", "Misleading", "False", "Unsustainable", "Exaggerated", "Unverifiable"]

- summary: Provide a succinct summary that directly supports your classification,
offering insight into your analytical reasoning.

- report: Document a comprehensive account detailing the investigative process, key findings,
and evidence that supports your conclusion.

- verification: Explain the specific methodologies and tools you employed to verify the
information, highlighting your systematic approach to substantiating the claim.

Cross-check analysis: {cross_check_result}
Claim: {claim}

compile your response in {language}, however the classification field must remain in English""",
    ),
])

def create_gazette_report(state: AgentState) -> dict:
    logger.info("[create_gazette_report] Starting — claim='%s'", state["claim"][:80])
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1)
    chain = _prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": state["claim"],
        "questions": state.get("questions", []),
        "cross_check_result": state.get("cross_check_result", ""),
        "language": state.get("language", "pt"),
    })
    logger.info("[create_gazette_report] Completed — report length=%d chars", len(result))
    return {"messages": result}
