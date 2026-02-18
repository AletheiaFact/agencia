"""Plugin base classes for Agencia data source integrations."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PluginCategory(str, Enum):
    CLAIM_DATABASE = "claim_databases"
    GOVERNMENT_DATA = "government_data"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_BASE = "knowledge_bases"
    FACT_CHECK_ORG = "fact_check_orgs"
    ACADEMIC = "academic"
    LEGISLATION = "legislation"
    MULTIMEDIA = "multimedia"


class PluginMetadata(BaseModel):
    """Immutable plugin descriptor."""

    name: str
    display_name: str
    description: str
    category: PluginCategory
    required_env_vars: list[str] = []
    optional_env_vars: list[str] = []
    auth_env_var_groups: list[list[str]] = []
    rate_limit_rpm: Optional[int] = None
    version: str = "0.1.0"
    reliability_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Source trustworthiness")


class PluginResult(BaseModel):
    """Standardized result envelope from any plugin search."""

    source: str
    query: str
    results: list[dict] = []
    result_count: int = 0
    metadata: dict = {}
    error: Optional[str] = None


class DataSourcePlugin(ABC):
    """Abstract base for all data source plugins.

    Plugins provide a consistent interface for querying external data sources.
    Each plugin must implement get_metadata(), is_available(), and search().
    """

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata. Must not require env vars or network."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if required env vars are set and the plugin can operate.

        Must not make network calls. Used for lazy initialization gating.
        """
        ...

    def health_check(self) -> bool:
        """Make a lightweight check to verify the plugin can operate.

        Default returns is_available(). Override for real API ping.
        """
        return self.is_available()

    @abstractmethod
    def search(self, query: str, **kwargs) -> PluginResult:
        """Execute the plugin's primary search/lookup operation.

        Implementations should handle their own error wrapping and
        return PluginResult with error field set on failure.
        """
        ...

    def as_langchain_tool(self):
        """Return a LangChain Tool wrapping this plugin's search method.

        Default implementation creates a Tool from self.search.
        Plugins may override to provide richer tool schemas.
        """
        from langchain.agents import Tool

        meta = self.get_metadata()
        return Tool(
            name=meta.name,
            func=lambda q: self.search(q).model_dump_json(),
            description=meta.description,
        )
