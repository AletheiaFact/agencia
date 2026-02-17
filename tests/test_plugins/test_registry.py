"""Tests for the plugin registry."""

import pytest

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult
from plugins import registry


# --- Test fixtures ---


class FakePlugin(DataSourcePlugin):
    def __init__(self, name: str, category: PluginCategory, available: bool = True):
        self._name = name
        self._category = category
        self._available = available

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self._name,
            display_name=f"Fake {self._name}",
            description=f"Fake plugin: {self._name}",
            category=self._category,
        )

    def is_available(self) -> bool:
        return self._available

    def search(self, query: str, **kwargs) -> PluginResult:
        return PluginResult(source=self._name, query=query, results=[], result_count=0)


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    registry.clear()
    yield
    registry.clear()


# --- Registration tests ---


class TestRegistration:
    def test_register_and_get(self):
        plugin = FakePlugin("test", PluginCategory.WEB_SEARCH)
        registry.register(plugin)
        assert registry.get("test") is plugin

    def test_get_nonexistent_returns_none(self):
        assert registry.get("nonexistent") is None

    def test_duplicate_registration_overwrites(self):
        p1 = FakePlugin("test", PluginCategory.WEB_SEARCH, available=True)
        p2 = FakePlugin("test", PluginCategory.WEB_SEARCH, available=False)
        registry.register(p1)
        registry.register(p2)
        assert registry.get("test") is p2


# --- Listing tests ---


class TestListing:
    def test_get_all(self):
        registry.register(FakePlugin("a", PluginCategory.WEB_SEARCH))
        registry.register(FakePlugin("b", PluginCategory.GOVERNMENT_DATA))
        assert len(registry.get_all()) == 2

    def test_get_available_filters_unavailable(self):
        registry.register(FakePlugin("available", PluginCategory.WEB_SEARCH, available=True))
        registry.register(FakePlugin("unavailable", PluginCategory.WEB_SEARCH, available=False))
        available = registry.get_available()
        assert len(available) == 1
        assert available[0].get_metadata().name == "available"

    def test_get_available_filters_by_category(self):
        registry.register(FakePlugin("web", PluginCategory.WEB_SEARCH))
        registry.register(FakePlugin("gov", PluginCategory.GOVERNMENT_DATA))
        registry.register(FakePlugin("claim", PluginCategory.CLAIM_DATABASE))

        web_plugins = registry.get_available(PluginCategory.WEB_SEARCH)
        assert len(web_plugins) == 1
        assert web_plugins[0].get_metadata().name == "web"

        gov_plugins = registry.get_available(PluginCategory.GOVERNMENT_DATA)
        assert len(gov_plugins) == 1
        assert gov_plugins[0].get_metadata().name == "gov"

    def test_get_available_empty_category(self):
        registry.register(FakePlugin("web", PluginCategory.WEB_SEARCH))
        assert registry.get_available(PluginCategory.ACADEMIC) == []


# --- LangChain tool integration tests ---


class TestLangChainTools:
    def test_get_langchain_tools_returns_tools(self):
        registry.register(FakePlugin("a", PluginCategory.WEB_SEARCH))
        registry.register(FakePlugin("b", PluginCategory.WEB_SEARCH))
        tools = registry.get_langchain_tools()
        assert len(tools) == 2
        assert all(hasattr(t, "name") for t in tools)

    def test_get_langchain_tools_filters_by_category(self):
        registry.register(FakePlugin("web", PluginCategory.WEB_SEARCH))
        registry.register(FakePlugin("gov", PluginCategory.GOVERNMENT_DATA))
        tools = registry.get_langchain_tools(PluginCategory.GOVERNMENT_DATA)
        assert len(tools) == 1
        assert tools[0].name == "gov"

    def test_get_langchain_tools_excludes_unavailable(self):
        registry.register(FakePlugin("ok", PluginCategory.WEB_SEARCH, available=True))
        registry.register(FakePlugin("no", PluginCategory.WEB_SEARCH, available=False))
        tools = registry.get_langchain_tools()
        assert len(tools) == 1


# --- Clear tests ---


class TestClear:
    def test_clear_removes_all(self):
        registry.register(FakePlugin("a", PluginCategory.WEB_SEARCH))
        registry.register(FakePlugin("b", PluginCategory.GOVERNMENT_DATA))
        assert len(registry.get_all()) == 2
        registry.clear()
        assert len(registry.get_all()) == 0
