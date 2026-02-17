"""Node: Create fact-checking report from gathered evidence."""

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

def create_report(state: AgentState) -> dict:
    logger.info("[create_report] Starting — claim='%s'", state["claim"][:80])
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1)
    chain = _prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": state["claim"],
        "messages": state.get("messages", []),
        "questions": state.get("questions", []),
        "language": state.get("language", "pt"),
    })
    logger.info("[create_report] Completed — report length=%d chars", len(result))
    return {"messages": result}
