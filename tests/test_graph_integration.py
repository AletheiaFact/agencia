"""Integration tests for the updated LangGraph workflow.

Tests the full graph flow including the new check_existing_factchecks node
and factcheck_router 3-way routing.
"""

from unittest.mock import MagicMock, patch

import pytest

from plugins.base import PluginResult
from plugins import registry


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure clean registry for each test."""
    registry.clear()
    yield
    registry.clear()


# --- Graph structure tests ---

class TestGraphStructure:
    @patch("nodes.factcheck_lookup.get_plugin", return_value=None)
    def test_graph_compiles(self, _):
        """The graph should compile without errors."""
        from graph import build_workflow
        workflow = build_workflow()
        assert workflow is not None

    @patch("nodes.factcheck_lookup.get_plugin", return_value=None)
    def test_graph_has_expected_nodes(self, _):
        """Verify all expected nodes are present in the graph."""
        from graph import build_workflow
        workflow = build_workflow()
        graph = workflow.get_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "list_questions",
            "check_existing_factchecks",
            "select_sources",
            "search_online",
            "create_report",
            "fact_check_gazettes",
            "__start__",
            "__end__",
        }
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"


# --- Routing tests ---

class TestRouting:
    """Test the factcheck_router routes correctly based on state."""

    def test_found_route_when_factcheck_exists(self):
        from nodes.factcheck_lookup import factcheck_router
        state = {
            "claim": "Test claim",
            "existing_factchecks": [
                {"rating": "False", "url": "https://example.com/fc", "publisher": "Org"}
            ],
        }
        assert factcheck_router(state) == "found"

    def test_online_route_when_no_factcheck(self):
        from nodes.factcheck_lookup import factcheck_router
        state = {
            "claim": "Test claim",
            "existing_factchecks": [],
            "search_type": "online",
        }
        assert factcheck_router(state) == "online"

    def test_gazettes_route_when_no_factcheck(self):
        from nodes.factcheck_lookup import factcheck_router
        state = {
            "claim": "Test claim",
            "existing_factchecks": [],
            "search_type": "gazettes",
        }
        assert factcheck_router(state) == "gazettes"


# --- Plugin registration tests ---

class TestPluginRegistration:
    def test_register_all_plugins_handles_missing_env(self):
        """register_all_plugins should not raise even without API keys."""
        from plugins import register_all_plugins
        # Should not raise — plugins just register as unavailable
        register_all_plugins()
        all_plugins = registry.get_all()
        assert len(all_plugins) == 4

    def test_registered_plugin_names(self):
        from plugins import register_all_plugins
        register_all_plugins()
        names = {p.get_metadata().name for p in registry.get_all()}
        assert names == {
            "google_factcheck",
            "portal_transparencia",
            "ibge_sidra",
            "tavily_search",
        }

    def test_government_data_tools_available(self):
        """Government data plugins should be available (no API keys needed)."""
        from plugins import register_all_plugins
        from plugins.base import PluginCategory
        register_all_plugins()
        gov_plugins = registry.get_available(PluginCategory.GOVERNMENT_DATA)
        # portal_transparencia is always available, ibge_sidra if sidrapy installed
        assert len(gov_plugins) >= 1
        names = {p.get_metadata().name for p in gov_plugins}
        assert "portal_transparencia" in names


# --- Check existing factchecks node ---

class TestCheckExistingFactchecksNode:
    @patch("nodes.factcheck_lookup.get_plugin", return_value=None)
    def test_gracefully_skips_when_no_plugin(self, _):
        from nodes.factcheck_lookup import check_existing_factchecks
        state = {"claim": "Test claim", "language": "pt"}
        result = check_existing_factchecks(state)
        assert result["existing_factchecks"] == []
        assert "reasoning_log" in result

    @patch("nodes.factcheck_lookup.get_plugin")
    def test_returns_factchecks_when_plugin_has_results(self, mock_get):
        from nodes.factcheck_lookup import check_existing_factchecks
        plugin = MagicMock()
        plugin.is_available.return_value = True
        plugin.search.return_value = PluginResult(
            source="google_factcheck",
            query="test",
            results=[{"rating": "True", "publisher": "Org", "url": "https://org.com/fc"}],
            result_count=1,
        )
        mock_get.return_value = plugin

        state = {"claim": "Test claim", "language": "pt"}
        result = check_existing_factchecks(state)
        assert len(result["existing_factchecks"]) == 1


# --- Report node ---

class TestReportNode:
    def test_build_existing_factchecks_section_empty(self):
        from nodes.report import _build_existing_factchecks_section
        state = {"claim": "test"}
        assert _build_existing_factchecks_section(state) == ""

    def test_build_existing_factchecks_section_with_data(self):
        from nodes.report import _build_existing_factchecks_section
        state = {
            "claim": "test",
            "existing_factchecks": [
                {"publisher": "Lupa", "rating": "False", "url": "https://lupa.news/1"}
            ],
        }
        section = _build_existing_factchecks_section(state)
        assert "Lupa" in section
        assert "False" in section
        assert "Incorporate" in section


class TestSourceSelectionIntegration:
    @patch("nodes.source_selection.get_classifier")
    @patch("nodes.source_selection.get_available", return_value=[])
    @patch("nodes.source_selection.get", return_value=None)
    def test_select_sources_node_runs_on_online_path(
        self, mock_get, mock_avail, mock_get_clf
    ):
        from plugins.source_selector import (
            SourceSelectionResult,
            SourceRecommendation,
        )

        clf = MagicMock()
        clf.classify.return_value = SourceSelectionResult(
            claim_type="general",
            selected_sources=[
                SourceRecommendation(
                    plugin_name="tavily_search",
                    relevance=0.8,
                    reason="general web search",
                )
            ],
            classification_method="llm",
        )
        mock_get_clf.return_value = clf

        from nodes.source_selection import select_sources

        state = {"claim": "Test claim", "language": "pt"}
        result = select_sources(state)
        assert result["selected_sources"][0]["plugin_name"] == "tavily_search"
