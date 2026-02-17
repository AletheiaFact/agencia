"""Gazette node: Create an optimized search subject from the claim."""

import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load search context (advanced search operators guide)
_context_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "querido_diario_search_context.txt"
)
with open(_context_path, "r", encoding="utf-8") as f:
    _search_context = f.read()

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert writer skilled in condensing extensive claims into a single,
optimized sentence for search purposes.

Your goal: Create highly effective and precise search subjects for claims in Portuguese
using a variety of search operators to maximize retrieval of relevant and accurate information.

Here is a guide on advanced search operators you can use:
{search_context}

Analyze the claim provided and construct a subject line in {language}. This subject should
concentrate on the main keywords and phrases most likely to retrieve accurate and relevant information.

Incorporate appropriate search operators to refine search results efficiently.

Claim: {claim}""",
    ),
])

_chain = _prompt | llm | StrOutputParser()


def create_subject(state: AgentState) -> dict:
    result = _chain.invoke({
        "claim": state["claim"],
        "language": state.get("language", "pt"),
        "search_context": _search_context,
    })
    return {"search_subject": result}
