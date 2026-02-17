"""Gazette node: Fetch gazette data from the Querido Diario API."""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

from state import AgentState
from tools.querido_diario import querido_diario_fetch

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert data researcher in navigating and extracting data from extensive digital gazette archives.

Your goal: Use the search subject to fetch comprehensive and accurate data from gazettes,
ensuring the information meets user-defined parameters and research needs.

Use the querido_diario_fetch tool to gather data from gazettes.

Format your query using the following parameters:
- city: {city}
- subject: {search_subject}
- published_since: {published_since}
- published_until: {published_until}

Note: The subject should not be changed once set. Ensure the initial subject is comprehensive
and precisely targets the needed information.

Provide the txt_url source from the results.
{agent_scratchpad}""",
    ),
])


def fetch_gazettes(state: AgentState) -> dict:
    context = state.get("context", {})
    tools = [querido_diario_fetch]
    agent = create_tool_calling_agent(llm, tools, _prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    result = executor.invoke({
        "search_subject": state.get("search_subject", state["claim"]),
        "city": context.get("city", ""),
        "published_since": context.get("published_since", ""),
        "published_until": context.get("published_until", ""),
    })

    # Extract gazette data and URL from agent output
    output = result.get("output", "")
    return {
        "gazette_data": result,
        "gazette_url": output,
    }
