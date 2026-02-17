"""Gazette node: Fetch gazette data from the Querido Diario API."""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

from state import AgentState
from tools.querido_diario import querido_diario_fetch

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an automated data retrieval agent. Your ONLY job is to call the fetch_querido_diario_api tool and return its results. Do NOT ask for confirmation. Do NOT ask clarifying questions. Execute the tool call IMMEDIATELY.

You MUST call fetch_querido_diario_api with these exact parameters:
- subject: "{search_subject}"
- city: "{city}"
- published_since: "{published_since}"
- published_until: "{published_until}"

If any parameter is empty, pass it as-is (empty string or None). Do NOT wait for user input.

After receiving the tool response, extract and return the txt_url field from the first gazette result. If the tool returns an error, return the error message.""",
    ),
    ("human", "Execute the gazette fetch now. Call the tool immediately."),
    ("placeholder", "{agent_scratchpad}"),
])


def fetch_gazettes(state: AgentState) -> dict:
    context = state.get("context", {})
    city = context.get("city", "")
    search_subject = state.get("search_subject", state["claim"])
    logger.info("[fetch_gazettes] Starting — city=%s subject='%s'", city, search_subject[:80])

    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1)
    tools = [querido_diario_fetch]
    agent = create_tool_calling_agent(llm, tools, _prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    result = executor.invoke({
        "search_subject": search_subject,
        "city": city,
        "published_since": context.get("published_since", ""),
        "published_until": context.get("published_until", ""),
    })

    # Extract gazette data and URL from agent output
    output = result.get("output", "")
    logger.info("[fetch_gazettes] Completed — output length=%d chars", len(output))
    return {
        "gazette_data": result,
        "gazette_url": output,
    }
