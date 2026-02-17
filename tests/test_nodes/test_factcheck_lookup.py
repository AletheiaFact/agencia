"""Tests for the factcheck_lookup node."""

from unittest.mock import MagicMock, patch

import pytest

from plugins.base import PluginResult
from nodes.factcheck_lookup import check_existing_factchecks, factcheck_router


# --- Helper to build minimal state ---

def _state(**overrides):
    base = {
        "claim": "The earth is round",
        "language": "pt",
    }
    base.update(overrides)
    return base


# --- check_existing_factchecks ---

class TestCheckExistingFactchecks:
    @patch("nodes.factcheck_lookup.get_plugin", return_value=None)
    def test_returns_empty_when_plugin_not_registered(self, mock_get):
        result = check_existing_factchecks(_state())
        assert result["existing_factchecks"] == []
        assert "reasoning_log" in result
        assert any("not available" in entry for entry in result["reasoning_log"])

    @patch("nodes.factcheck_lookup.get_plugin")
    def test_returns_empty_when_plugin_unavailable(self, mock_get):
        plugin = MagicMock()
        plugin.is_available.return_value = False
        mock_get.return_value = plugin

        result = check_existing_factchecks(_state())
        assert result["existing_factchecks"] == []
        assert "reasoning_log" in result
        assert any("not available" in entry for entry in result["reasoning_log"])

    @patch("nodes.factcheck_lookup.get_plugin")
    def test_returns_empty_when_plugin_errors(self, mock_get):
        plugin = MagicMock()
        plugin.is_available.return_value = True
        plugin.search.return_value = PluginResult(
            source="google_factcheck", query="test", error="API failed"
        )
        mock_get.return_value = plugin

        result = check_existing_factchecks(_state())
        assert result["existing_factchecks"] == []
        assert "reasoning_log" in result
        assert any("API failed" in entry for entry in result["reasoning_log"])

    @patch("nodes.factcheck_lookup.get_plugin")
    def test_returns_results_on_success(self, mock_get):
        plugin = MagicMock()
        plugin.is_available.return_value = True
        plugin.search.return_value = PluginResult(
            source="google_factcheck",
            query="test",
            results=[
                {"rating": "True", "publisher": "Lupa", "url": "https://lupa.news/1"}
            ],
            result_count=1,
        )
        mock_get.return_value = plugin

        result = check_existing_factchecks(_state())
        assert len(result["existing_factchecks"]) == 1
        assert result["existing_factchecks"][0]["publisher"] == "Lupa"

    @patch("nodes.factcheck_lookup.get_plugin")
    def test_passes_claim_and_language(self, mock_get):
        plugin = MagicMock()
        plugin.is_available.return_value = True
        plugin.search.return_value = PluginResult(
            source="google_factcheck", query="test"
        )
        mock_get.return_value = plugin

        check_existing_factchecks(_state(claim="Custom claim", language="en"))
        plugin.search.assert_called_once_with(query="Custom claim", language_code="en")


# --- factcheck_router ---

class TestFactcheckRouter:
    def test_returns_found_when_existing_factcheck_with_rating_and_url(self):
        state = _state(
            existing_factchecks=[
                {"rating": "False", "url": "https://example.com/check", "publisher": "Org"}
            ]
        )
        assert factcheck_router(state) == "found"

    def test_skips_factcheck_without_rating(self):
        state = _state(
            existing_factchecks=[
                {"rating": "", "url": "https://example.com/check"}
            ],
            search_type="online",
        )
        assert factcheck_router(state) == "online"

    def test_skips_factcheck_without_url(self):
        state = _state(
            existing_factchecks=[
                {"rating": "False", "url": ""}
            ],
            search_type="gazettes",
        )
        assert factcheck_router(state) == "gazettes"

    def test_routes_to_gazettes_when_no_factchecks(self):
        state = _state(existing_factchecks=[], search_type="gazettes")
        assert factcheck_router(state) == "gazettes"

    def test_routes_to_online_when_no_factchecks(self):
        state = _state(existing_factchecks=[], search_type="online")
        assert factcheck_router(state) == "online"

    def test_defaults_to_online_when_no_search_type(self):
        state = _state(existing_factchecks=[])
        assert factcheck_router(state) == "online"

    def test_returns_found_when_any_factcheck_has_rating_and_url(self):
        """Even if first result has no rating, second should trigger found."""
        state = _state(
            existing_factchecks=[
                {"rating": "", "url": ""},
                {"rating": "Misleading", "url": "https://org.com/fc"},
            ]
        )
        assert factcheck_router(state) == "found"
