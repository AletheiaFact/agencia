"""Gazette node: Analyze gazette content using FAISS similarity search."""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

from state import AgentState
from tools.gazette_search import gazette_search_context

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert data analyst specialized in extracting data from gazette archives.

Your goal: Use the information gathered by the data researcher to extract important data
from the relevant gazette, and create a comprehensive view of the claim.

Use the gazette_search tool with the txt_url source provided to collect all relevant documents.

Structure your query with:
- claim: {claim}
- url: {gazette_url}
- questions: {questions}

Gather and compile a detailed response that comprehensively addresses all pertinent gazette
document data related to the questions. Ensure that each response is thorough, providing
extensive and accurate information to cover all relevant aspects.
{agent_scratchpad}""",
    ),
])


def analyze_gazette(state: AgentState) -> dict:
    gazette_url = state.get("gazette_url", "")
    logger.info("[analyze_gazette] Starting — claim='%s' url='%s'", state["claim"][:80], gazette_url[:120] if gazette_url else "N/A")

    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1)
    tools = [gazette_search_context]
    agent = create_tool_calling_agent(llm, tools, _prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    result = executor.invoke({
        "claim": state["claim"],
        "gazette_url": gazette_url,
        "questions": state.get("questions", []),
    })

    output = result.get("output", "")
    logger.info("[analyze_gazette] Completed — analysis length=%d chars", len(output))
    return {"gazette_analysis": output}
