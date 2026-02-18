"""Agencia data source plugin system.

Provides a pluggable architecture for integrating external data sources
into the fact-checking pipeline.
"""

from plugins.registry import register, get, get_available, get_all, get_langchain_tools, get_tools_for_selection, clear  # noqa: F401


def register_all_plugins() -> None:
    """Register all available plugins. Called once at server startup."""
    from plugins.claim_databases.google_factcheck import GoogleFactCheckPlugin
    from plugins.government_data.transparencia import PortalTransparenciaPlugin
    from plugins.government_data.ibge_sidra import IBGESidraPlugin
    from plugins.web_search.tavily_search import TavilySearchPlugin
    from plugins.electoral.tse import TSEPlugin
    from plugins.government_data.bacen import BACENPlugin
    from plugins.knowledge_bases.wikipedia import WikipediaPlugin

    register(GoogleFactCheckPlugin())
    register(PortalTransparenciaPlugin())
    register(IBGESidraPlugin())
    register(TavilySearchPlugin())
    register(TSEPlugin())
    register(BACENPlugin())
    register(WikipediaPlugin())
