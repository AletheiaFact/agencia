"""Node: Search online sources for fact-checking evidence."""

import logging
import re
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import WebBaseLoader

from state import AgentState
from tools.web_search import get_search_tool
from plugins.registry import get_langchain_tools, get_tools_for_selection

logger = logging.getLogger(__name__)
_URL_PATTERN = re.compile(r"(https?://[^\s'\"<>()]+)", re.IGNORECASE)

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
    seed_sources = context.get("sources", [])
    language = state.get("language", "pt")
    doc_context = []

    logger.info("[search_online] Starting — claim='%s' sources=%d language=%s", claim[:80], len(seed_sources), language)

    if seed_sources:
        logger.info("[search_online] Loading %d source URLs", len(seed_sources))
        loader = WebBaseLoader(seed_sources)
        load_document = loader.load()
        doc_context = load_document[0].page_content
        logger.info("[search_online] Loaded source content (length=%d chars)", len(doc_context))

    llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=1)
    search_tool = get_search_tool()
    tools = [search_tool]
    # Use source selection results if available, otherwise fall back to all plugins
    selected = state.get("selected_sources")
    if selected:
        tools.extend(get_tools_for_selection(selected))
    else:
        tools.extend(get_langchain_tools())
    logger.info("[search_online] Agent tools: %s", [t.name for t in tools])
    agent = create_tool_calling_agent(llm, tools, _prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
    )

    result = executor.invoke({
        "claim": claim,
        "sources": seed_sources,
        "language": language,
        "context": doc_context,
    })

    collected_urls = _collect_urls(result, seed_sources)
    structured_sources = _build_structured_sources(collected_urls)
    output_len = len(str(result.get("output", "")))
    tool_names = [t.name for t in tools]
    logger.info("[search_online] Completed — output length=%d chars", output_len)
    return {
        "messages": [result],
        "sources": structured_sources,
        "reasoning_log": [
            f"[search_online] Researched claim online using tools {tool_names}, "
            f"collected {output_len} chars of evidence and {len(collected_urls)} unique URLs"
        ],
    }


def _extract_urls(text: str) -> list[str]:
    """Extract URLs from free text and normalize trailing punctuation."""
    if not text:
        return []
    cleaned = []
    for url in _URL_PATTERN.findall(text):
        cleaned.append(url.rstrip(".,);:!?"))
    return cleaned


def _collect_urls(result: dict, seed_urls: list[str]) -> list[str]:
    """Collect and deduplicate source URLs from agent output and tool traces."""
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(url: str) -> None:
        if not url or url in seen:
            return
        seen.add(url)
        ordered.append(url)

    for url in seed_urls or []:
        _add(url)

    # Defensive fallback in case LLM outputs URLs directly
    for url in _extract_urls(str(result.get("output", ""))):
        _add(url)

    for step in result.get("intermediate_steps", []) or []:
        step_text = str(step)
        for url in _extract_urls(step_text):
            _add(url)

    return ordered


def _build_structured_sources(urls: list[str]) -> list[dict]:
    """Convert URL list into structured source objects for report output."""
    sources: list[dict] = []
    for url in urls:
        host = urlparse(url).netloc or url
        sources.append({
            "title": host,
            "url": url,
            "type": "web",
        })
    return sources
