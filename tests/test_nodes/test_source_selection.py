"""Tests for the select_sources node."""

from unittest.mock import patch, MagicMock

import pytest

from plugins.source_selector import SourceSelectionResult, SourceRecommendation
from nodes.source_selection import select_sources, set_classifier


@pytest.fixture(autouse=True)
def _reset_classifier():
    yield
    set_classifier(None)


def _state(**overrides):
    base = {"claim": "O governo gastou R$10 milhões em obras", "language": "pt"}
    base.update(overrides)
    return base


def _mock_selection_result():
    return SourceSelectionResult(
        claim_type="government_spending",
        selected_sources=[
            SourceRecommendation(
                plugin_name="portal_transparencia",
                relevance=0.95,
                reason="Government spending claim",
            ),
            SourceRecommendation(
                plugin_name="tavily_search",
                relevance=0.7,
                reason="Web search for corroboration",
            ),
        ],
        classification_method="llm",
    )


class TestSelectSources:
    @patch("nodes.source_selection.get_classifier")
    @patch("nodes.source_selection.get_available")
    @patch("nodes.source_selection.get")
    def test_returns_selected_sources(self, mock_get, mock_avail, mock_get_clf):
        clf = MagicMock()
        clf.classify.return_value = _mock_selection_result()
        mock_get_clf.return_value = clf
        mock_avail.return_value = []

        # Mock plugin lookups for reliability scores
        mock_plugin = MagicMock()
        mock_plugin.get_metadata.return_value = MagicMock(reliability_score=0.85)
        mock_get.return_value = mock_plugin

        result = select_sources(_state())
        assert "selected_sources" in result
        assert len(result["selected_sources"]) == 2
        assert result["selected_sources"][0]["plugin_name"] == "portal_transparencia"

    @patch("nodes.source_selection.get_classifier")
    @patch("nodes.source_selection.get_available")
    @patch("nodes.source_selection.get")
    def test_builds_confidence_map(self, mock_get, mock_avail, mock_get_clf):
        clf = MagicMock()
        clf.classify.return_value = _mock_selection_result()
        mock_get_clf.return_value = clf
        mock_avail.return_value = []

        mock_plugin = MagicMock()
        mock_plugin.get_metadata.return_value = MagicMock(reliability_score=0.85)
        mock_get.return_value = mock_plugin

        result = select_sources(_state())
        assert "source_confidence" in result
        assert "portal_transparencia" in result["source_confidence"]
        assert result["source_confidence"]["portal_transparencia"] == 0.85

    @patch("nodes.source_selection.get_classifier")
    @patch("nodes.source_selection.get_available")
    @patch("nodes.source_selection.get")
    def test_emits_reasoning_log(self, mock_get, mock_avail, mock_get_clf):
        clf = MagicMock()
        clf.classify.return_value = _mock_selection_result()
        mock_get_clf.return_value = clf
        mock_avail.return_value = []
        mock_get.return_value = MagicMock(
            get_metadata=MagicMock(return_value=MagicMock(reliability_score=0.5))
        )

        result = select_sources(_state())
        assert "reasoning_log" in result
        assert any("select_sources" in entry for entry in result["reasoning_log"])
        assert any("government_spending" in entry for entry in result["reasoning_log"])

    @patch("nodes.source_selection.get_classifier")
    @patch("nodes.source_selection.get_available")
    @patch("nodes.source_selection.get")
    def test_handles_unknown_plugin_gracefully(self, mock_get, mock_avail, mock_get_clf):
        clf = MagicMock()
        clf.classify.return_value = _mock_selection_result()
        mock_get_clf.return_value = clf
        mock_avail.return_value = []

        # plugin lookup returns None for unknown plugins
        mock_get.return_value = None

        result = select_sources(_state())
        # Should still return selected_sources, just no confidence scores
        assert "selected_sources" in result
        assert result["source_confidence"] == {}
