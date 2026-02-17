"""Tavily Search API plugin — replaces SerpAPI for web search.

Tavily provides search results optimized for LLM consumption with
built-in content extraction and relevance scoring.
"""

import logging
import os

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult

logger = logging.getLogger(__name__)


class TavilySearchPlugin(DataSourcePlugin):

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="tavily_search",
            display_name="Tavily Search",
            description="Search the web using Tavily, an API optimized for LLM-based "
            "applications. Returns relevant web results with content extraction.",
            category=PluginCategory.WEB_SEARCH,
            required_env_vars=["TAVILY_API_KEY"],
        )

    def is_available(self) -> bool:
        return bool(os.getenv("TAVILY_API_KEY"))

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_answer: bool = True,
        **kwargs,
    ) -> PluginResult:
        """Search the web using Tavily.

        Args:
            query: Search query string.
            max_results: Maximum number of results (default: 5).
            search_depth: "basic" or "advanced" (default: "advanced").
            include_answer: Include a direct AI-generated answer (default: True).
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return PluginResult(
                source="tavily_search",
                query=query,
                error="TAVILY_API_KEY not set",
            )

        logger.info(
            "[tavily_search] Searching — query='%s' depth=%s max=%d",
            query[:80],
            search_depth,
            max_results,
        )

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
            )
        except ImportError:
            return PluginResult(
                source="tavily_search",
                query=query,
                error="tavily-python package not installed",
            )
        except Exception as e:
            logger.error("[tavily_search] Error: %s", e)
            return PluginResult(
                source="tavily_search",
                query=query,
                error=f"Tavily search failed: {e}",
            )

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            })

        metadata = {}
        if response.get("answer"):
            metadata["answer"] = response["answer"]

        logger.info("[tavily_search] Found %d results", len(results))
        return PluginResult(
            source="tavily_search",
            query=query,
            results=results,
            result_count=len(results),
            metadata=metadata,
        )
