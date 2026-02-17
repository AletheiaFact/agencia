"""Tests for the plugin base classes."""

import pytest

from plugins.base import (
    DataSourcePlugin,
    PluginCategory,
    PluginMetadata,
    PluginResult,
)


# --- Concrete test implementation ---


class DummyPlugin(DataSourcePlugin):
    """Minimal concrete plugin for testing the ABC contract."""

    def __init__(self, available: bool = True):
        self._available = available

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="dummy",
            display_name="Dummy Plugin",
            description="A test plugin",
            category=PluginCategory.WEB_SEARCH,
            required_env_vars=["DUMMY_KEY"],
        )

    def is_available(self) -> bool:
        return self._available

    def search(self, query: str, **kwargs) -> PluginResult:
        return PluginResult(
            source="dummy",
            query=query,
            results=[{"title": "Test", "url": "https://example.com"}],
            result_count=1,
        )


# --- ABC contract tests ---


class TestDataSourcePluginABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            DataSourcePlugin()

    def test_must_implement_get_metadata(self):
        class Incomplete(DataSourcePlugin):
            def is_available(self):
                return True

            def search(self, query, **kwargs):
                return PluginResult(source="x", query=query)

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_is_available(self):
        class Incomplete(DataSourcePlugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="x", display_name="X", description="x",
                    category=PluginCategory.WEB_SEARCH,
                )

            def search(self, query, **kwargs):
                return PluginResult(source="x", query=query)

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_search(self):
        class Incomplete(DataSourcePlugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="x", display_name="X", description="x",
                    category=PluginCategory.WEB_SEARCH,
                )

            def is_available(self):
                return True

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_implementation_works(self):
        plugin = DummyPlugin()
        assert plugin.get_metadata().name == "dummy"
        assert plugin.is_available() is True
        result = plugin.search("test query")
        assert result.result_count == 1


# --- PluginMetadata tests ---


class TestPluginMetadata:
    def test_required_fields(self):
        meta = PluginMetadata(
            name="test",
            display_name="Test",
            description="Test plugin",
            category=PluginCategory.CLAIM_DATABASE,
        )
        assert meta.name == "test"
        assert meta.category == PluginCategory.CLAIM_DATABASE
        assert meta.required_env_vars == []
        assert meta.version == "0.1.0"

    def test_all_fields(self):
        meta = PluginMetadata(
            name="test",
            display_name="Test Plugin",
            description="Full test",
            category=PluginCategory.GOVERNMENT_DATA,
            required_env_vars=["KEY_1"],
            optional_env_vars=["KEY_2"],
            rate_limit_rpm=100,
            version="1.0.0",
        )
        assert meta.rate_limit_rpm == 100
        assert meta.version == "1.0.0"
        assert meta.optional_env_vars == ["KEY_2"]

    def test_auth_env_var_groups_default_empty(self):
        meta = PluginMetadata(
            name="test",
            display_name="Test",
            description="Test plugin",
            category=PluginCategory.CLAIM_DATABASE,
        )
        assert meta.auth_env_var_groups == []

    def test_auth_env_var_groups_with_values(self):
        meta = PluginMetadata(
            name="test",
            display_name="Test",
            description="Test plugin",
            category=PluginCategory.CLAIM_DATABASE,
            auth_env_var_groups=[["KEY_A"], ["KEY_B", "KEY_C"]],
        )
        assert len(meta.auth_env_var_groups) == 2
        assert meta.auth_env_var_groups[0] == ["KEY_A"]
        assert meta.auth_env_var_groups[1] == ["KEY_B", "KEY_C"]

    def test_invalid_category_rejected(self):
        with pytest.raises(ValueError):
            PluginMetadata(
                name="test",
                display_name="Test",
                description="Test",
                category="invalid_category",
            )


# --- PluginResult tests ---


class TestPluginResult:
    def test_minimal_result(self):
        result = PluginResult(source="test", query="q")
        assert result.results == []
        assert result.result_count == 0
        assert result.error is None

    def test_result_with_data(self):
        result = PluginResult(
            source="test",
            query="query",
            results=[{"key": "value"}],
            result_count=1,
            metadata={"page": 1},
        )
        assert result.result_count == 1
        assert result.results[0]["key"] == "value"

    def test_error_result(self):
        result = PluginResult(
            source="test",
            query="query",
            error="API timeout",
        )
        assert result.error == "API timeout"
        assert result.results == []

    def test_serialization(self):
        result = PluginResult(
            source="test",
            query="q",
            results=[{"a": 1}],
            result_count=1,
        )
        json_str = result.model_dump_json()
        assert "test" in json_str
        restored = PluginResult.model_validate_json(json_str)
        assert restored.source == "test"


# --- PluginCategory tests ---


class TestPluginCategory:
    def test_all_categories_exist(self):
        expected = [
            "claim_databases", "government_data", "web_search",
            "knowledge_bases", "fact_check_orgs", "academic",
            "legislation", "multimedia",
        ]
        actual = [c.value for c in PluginCategory]
        assert sorted(actual) == sorted(expected)


# --- Default method tests ---


class TestDefaultMethods:
    def test_health_check_defaults_to_is_available(self):
        plugin = DummyPlugin(available=True)
        assert plugin.health_check() is True

        plugin_unavailable = DummyPlugin(available=False)
        assert plugin_unavailable.health_check() is False

    def test_as_langchain_tool_returns_tool(self):
        plugin = DummyPlugin()
        tool = plugin.as_langchain_tool()
        assert tool.name == "dummy"
        assert "test plugin" in tool.description.lower()
        # Tool func should be callable and return JSON
        output = tool.func("test")
        assert "test" in output
