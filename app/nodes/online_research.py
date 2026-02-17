"""Node: Search online sources for fact-checking evidence."""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import WebBaseLoader

from state import AgentState
from tools.web_search import get_search_tool

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Reseacher Journalist with a specific role: Search content on the internet related to the claim and the provided sources

Claim: {claim}
Sources: {sources}
Context: {context}

In case sources is provided utilize the context to gather relevant data.
In case there is not any sources, create your query using the claim.

Compiles a comprehensive response containing all relevant data from the tool
Provide your response in {language}.
{agent_scratchpad}""",
    ),
])


def search_online(state: AgentState) -> dict:
    claim = state["claim"]
    context = state.get("context", {})
    sources = context.get("sources", [])
    language = state.get("language", "pt")
    doc_context = []

    logger.info("[search_online] Starting — claim='%s' sources=%d language=%s", claim[:80], len(sources), language)

    if sources:
        logger.info("[search_online] Loading %d source URLs", len(sources))
        loader = WebBaseLoader(sources)
        load_document = loader.load()
        doc_context = load_document[0].page_content
        logger.info("[search_online] Loaded source content (length=%d chars)", len(doc_context))

    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1)
    search_tool = get_search_tool()
    tools = [search_tool]
    agent = create_tool_calling_agent(llm, tools, _prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    result = executor.invoke({
        "claim": claim,
        "sources": sources,
        "language": language,
        "context": doc_context,
    })

    logger.info("[search_online] Completed — output length=%d chars", len(str(result.get("output", ""))))
    return {"messages": [result]}
