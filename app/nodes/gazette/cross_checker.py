"""Gazette node: Cross-check collected gazette data against the claim."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState

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
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = _prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": state["claim"],
        "gazette_analysis": state.get("gazette_analysis", ""),
    })
    return {"cross_check_result": result}
