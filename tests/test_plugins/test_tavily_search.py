"""Tests for the Tavily Search plugin."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from plugins.web_search.tavily_search import TavilySearchPlugin


@pytest.fixture
def plugin():
    return TavilySearchPlugin()


def _mock_tavily_module(search_return_value=None, search_side_effect=None):
    """Create a mock tavily module with a mock TavilyClient."""
    mock_client = MagicMock()
    if search_side_effect:
        mock_client.search.side_effect = search_side_effect
    elif search_return_value is not None:
        mock_client.search.return_value = search_return_value

    mock_tavily = MagicMock()
    mock_tavily.TavilyClient.return_value = mock_client
    return mock_tavily, mock_client


# --- Metadata ---

class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "tavily_search"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.WEB_SEARCH

    def test_requires_api_key(self, plugin):
        assert "TAVILY_API_KEY" in plugin.get_metadata().required_env_vars


# --- Availability ---

class TestAvailability:
    def test_available_when_key_set(self, plugin, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        assert plugin.is_available() is True

    def test_unavailable_when_key_missing(self, plugin, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert plugin.is_available() is False


# --- Search ---

class TestSearch:
    def test_returns_error_when_no_api_key(self, plugin, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = plugin.search("test query")
        assert result.error is not None
        assert "TAVILY_API_KEY" in result.error

    def test_successful_search(self, plugin, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        mock_tavily, mock_client = _mock_tavily_module(search_return_value={
            "answer": "The earth is round.",
            "results": [
                {
                    "title": "Shape of Earth",
                    "url": "https://example.com/earth",
                    "content": "Earth is an oblate spheroid.",
                    "score": 0.95,
                },
                {
                    "title": "Earth Facts",
                    "url": "https://example.com/facts",
                    "content": "More facts about earth.",
                    "score": 0.88,
                },
            ],
        })

        with patch.dict(sys.modules, {"tavily": mock_tavily}):
            result = plugin.search("Is the earth round?")

        assert result.error is None
        assert result.result_count == 2
        assert result.results[0]["title"] == "Shape of Earth"
        assert result.results[0]["url"] == "https://example.com/earth"
        assert result.results[0]["score"] == 0.95
        assert result.metadata.get("answer") == "The earth is round."

    def test_empty_results(self, plugin, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        mock_tavily, _ = _mock_tavily_module(search_return_value={"results": []})

        with patch.dict(sys.modules, {"tavily": mock_tavily}):
            result = plugin.search("extremely obscure nonsense query xyz123")

        assert result.error is None
        assert result.result_count == 0
        assert result.metadata.get("answer") is None

    def test_handles_api_error(self, plugin, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        mock_tavily, _ = _mock_tavily_module(
            search_side_effect=Exception("Rate limit exceeded")
        )

        with patch.dict(sys.modules, {"tavily": mock_tavily}):
            result = plugin.search("test")

        assert result.error is not None
        assert "Tavily search failed" in result.error

    def test_handles_missing_package(self, plugin, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

        with patch.dict(sys.modules, {"tavily": None}):
            result = plugin.search("test")

        assert result.error is not None
        assert "not installed" in result.error
