"""Web search tool — uses Tavily (primary) with SerpAPI fallback."""

import logging
import os

from langchain.agents import Tool

logger = logging.getLogger(__name__)


def _create_tavily_tool() -> Tool:
    """Create a Tavily-backed search tool."""
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)

    def _search(query: str) -> str:
        response = client.search(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_answer=True,
        )
        # Combine answer + results into a single string for the agent
        parts = []
        if response.get("answer"):
            parts.append(f"Answer: {response['answer']}")
        for item in response.get("results", []):
            parts.append(f"- {item.get('title', '')}: {item.get('content', '')[:300]}")
        return "\n".join(parts) if parts else "No results found."

    return Tool(
        name="search_tool",
        func=_search,
        description="useful for when you need to ask with search",
    )


def _create_serpapi_tool() -> Tool:
    """Create a SerpAPI-backed search tool (fallback)."""
    from langchain_community.utilities import SerpAPIWrapper

    api_key = os.getenv("SERPAPI_API_KEY")
    search = SerpAPIWrapper(
        params={"engine": "google", "gl": "br", "hl": "pt"},
        serpapi_api_key=api_key,
    )
    return Tool(
        name="search_tool",
        func=search.run,
        description="useful for when you need to ask with search",
    )


def get_search_tool() -> Tool:
    """Create the web search tool. Prefers Tavily, falls back to SerpAPI."""
    if os.getenv("TAVILY_API_KEY"):
        try:
            tool = _create_tavily_tool()
            logger.info("[web_search] Using Tavily search")
            return tool
        except ImportError:
            logger.warning("[web_search] tavily-python not installed, falling back to SerpAPI")

    if os.getenv("SERPAPI_API_KEY"):
        logger.info("[web_search] Using SerpAPI search (fallback)")
        return _create_serpapi_tool()

    logger.warning("[web_search] No search API key configured, returning stub tool")
    return Tool(
        name="search_tool",
        func=lambda q: "No search API configured. Set TAVILY_API_KEY or SERPAPI_API_KEY.",
        description="useful for when you need to ask with search",
    )
