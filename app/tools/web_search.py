import os

from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()


def _create_search_wrapper() -> SerpAPIWrapper:
    api_key = os.getenv("SERPAPI_API_KEY")
    return SerpAPIWrapper(
        params={
            "engine": "google",
            "gl": "br",
            "hl": "pt",
        },
        serpapi_api_key=api_key,
    )


def get_search_tool() -> Tool:
    """Create the SerpAPI search tool. Call at runtime when API key is available."""
    search = _create_search_wrapper()
    return Tool(
        name="search_tool",
        func=search.run,
        description="useful for when you need to ask with search",
    )
