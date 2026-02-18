"""Tests for source confidence integration in create_report."""

from nodes.report import _build_source_confidence_section


class TestBuildSourceConfidenceSection:
    def test_returns_empty_when_no_data(self):
        state = {}
        assert _build_source_confidence_section(state) == ""

    def test_returns_empty_when_both_empty(self):
        state = {"source_confidence": {}, "selected_sources": []}
        assert _build_source_confidence_section(state) == ""

    def test_formats_single_source(self):
        state = {
            "selected_sources": [
                {"plugin_name": "portal_transparencia", "relevance": 0.95, "reason": "Spending claim"}
            ],
            "source_confidence": {"portal_transparencia": 0.85},
        }
        section = _build_source_confidence_section(state)
        assert "portal_transparencia" in section
        assert "0.85" in section
        assert "0.95" in section
        assert "Spending claim" in section

    def test_formats_multiple_sources(self):
        state = {
            "selected_sources": [
                {"plugin_name": "ibge_sidra", "relevance": 0.9, "reason": "Stats"},
                {"plugin_name": "tavily_search", "relevance": 0.6, "reason": "Web"},
            ],
            "source_confidence": {"ibge_sidra": 0.9, "tavily_search": 0.6},
        }
        section = _build_source_confidence_section(state)
        assert "ibge_sidra" in section
        assert "tavily_search" in section
        assert "Weigh higher-reliability" in section

    def test_handles_missing_confidence_for_source(self):
        state = {
            "selected_sources": [
                {"plugin_name": "unknown_plugin", "relevance": 0.5, "reason": "test"}
            ],
            "source_confidence": {},
        }
        section = _build_source_confidence_section(state)
        assert "N/A" in section
