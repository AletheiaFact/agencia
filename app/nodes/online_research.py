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
        """You are an expert Researcher Journalist. Your job is to search the internet for evidence related to a claim using the search tool. Do NOT ask for confirmation. Execute the search IMMEDIATELY.

Claim: {claim}
Sources: {sources}
Context: {context}

If sources are provided, use the context to gather relevant data.
If there are no sources, create your search query using the claim.

Compile a comprehensive response containing all relevant data from the search results.
Provide your response in {language}.""",
    ),
    ("human", "Search for evidence about this claim now."),
    ("placeholder", "{agent_scratchpad}"),
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

    llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=1)
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
