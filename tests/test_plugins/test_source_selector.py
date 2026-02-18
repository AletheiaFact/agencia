"""Tests for source classifier strategy."""

from unittest.mock import patch, MagicMock

import pytest
from pydantic import ValidationError

from plugins.base import PluginMetadata, PluginCategory
from plugins.source_selector import (
    LLMSourceClassifier,
    SourceClassifier,
    SourceSelectionResult,
    SourceRecommendation,
)


class TestSourceRecommendation:
    def test_valid_recommendation(self):
        rec = SourceRecommendation(
            plugin_name="portal_transparencia",
            relevance=0.9,
            reason="Claim is about government spending",
        )
        assert rec.plugin_name == "portal_transparencia"
        assert rec.relevance == 0.9

    def test_relevance_must_be_between_0_and_1(self):
        # Valid boundary values
        SourceRecommendation(plugin_name="x", relevance=0.0, reason="low")
        SourceRecommendation(plugin_name="x", relevance=1.0, reason="high")

    def test_relevance_below_range_rejected(self):
        with pytest.raises(ValidationError):
            SourceRecommendation(plugin_name="x", relevance=-0.1, reason="too low")

    def test_relevance_above_range_rejected(self):
        with pytest.raises(ValidationError):
            SourceRecommendation(plugin_name="x", relevance=1.1, reason="too high")

    def test_missing_fields_rejected(self):
        with pytest.raises(ValidationError):
            SourceRecommendation(plugin_name="x")  # missing relevance and reason

    def test_serialization(self):
        rec = SourceRecommendation(
            plugin_name="ibge_sidra",
            relevance=0.85,
            reason="Statistical claim",
        )
        d = rec.model_dump()
        assert d["plugin_name"] == "ibge_sidra"
        assert d["relevance"] == 0.85


class TestSourceSelectionResult:
    def test_valid_result(self):
        result = SourceSelectionResult(
            claim_type="political_spending",
            selected_sources=[
                SourceRecommendation(
                    plugin_name="portal_transparencia",
                    relevance=0.95,
                    reason="Government spending claim",
                ),
            ],
            classification_method="llm",
        )
        assert result.claim_type == "political_spending"
        assert len(result.selected_sources) == 1
        assert result.classification_method == "llm"

    def test_empty_sources_allowed(self):
        result = SourceSelectionResult(
            claim_type="general",
            selected_sources=[],
            classification_method="rules",
        )
        assert result.selected_sources == []


class TestSourceClassifierABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SourceClassifier()

    def test_concrete_subclass_works(self):
        class DummyClassifier(SourceClassifier):
            def classify(self, claim, available_plugins):
                return SourceSelectionResult(
                    claim_type="test",
                    selected_sources=[],
                    classification_method="dummy",
                )

        c = DummyClassifier()
        result = c.classify("test claim", [])
        assert result.claim_type == "test"



def _sample_plugins():
    """Return metadata for the 4 Phase 1 plugins."""
    return [
        PluginMetadata(
            name="google_factcheck",
            display_name="Google Fact Check API",
            description="Search existing fact-checks from Google's ClaimReview database.",
            category=PluginCategory.CLAIM_DATABASE,
            reliability_score=0.9,
        ),
        PluginMetadata(
            name="portal_transparencia",
            display_name="Portal da Transparência",
            description="Search Brazilian federal government spending data.",
            category=PluginCategory.GOVERNMENT_DATA,
            reliability_score=0.85,
        ),
        PluginMetadata(
            name="ibge_sidra",
            display_name="IBGE SIDRA",
            description="Query official Brazilian statistics (population, GDP, inflation).",
            category=PluginCategory.GOVERNMENT_DATA,
            reliability_score=0.9,
        ),
        PluginMetadata(
            name="tavily_search",
            display_name="Tavily Search",
            description="Search the web for relevant information.",
            category=PluginCategory.WEB_SEARCH,
            reliability_score=0.6,
        ),
    ]


class TestLLMSourceClassifier:
    @patch("plugins.source_selector.ChatOpenAI")
    def test_returns_valid_selection_result(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # Simulate LLM returning valid JSON
        mock_llm.invoke.return_value = MagicMock(
            content='{"claim_type": "government_spending", "sources": [{"plugin_name": "portal_transparencia", "relevance": 0.95, "reason": "Claim about federal spending"}, {"plugin_name": "tavily_search", "relevance": 0.7, "reason": "General web search for corroboration"}]}'
        )

        classifier = LLMSourceClassifier()
        result = classifier.classify("O governo gastou R$10 milhões", _sample_plugins())

        assert isinstance(result, SourceSelectionResult)
        assert result.claim_type == "government_spending"
        assert result.classification_method == "llm"
        assert len(result.selected_sources) >= 1
        assert result.selected_sources[0].plugin_name == "portal_transparencia"

    @patch("plugins.source_selector.ChatOpenAI")
    def test_falls_back_to_all_sources_on_parse_error(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="not valid json")

        classifier = LLMSourceClassifier()
        plugins = _sample_plugins()
        result = classifier.classify("some claim", plugins)

        assert result.claim_type == "general"
        assert result.classification_method == "llm_fallback"
        # Should include all available plugins as fallback
        assert len(result.selected_sources) == len(plugins)

    @patch("plugins.source_selector.ChatOpenAI")
    def test_falls_back_on_llm_exception(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("API timeout")

        classifier = LLMSourceClassifier()
        plugins = _sample_plugins()
        result = classifier.classify("some claim", plugins)

        assert result.claim_type == "general"
        assert result.classification_method == "llm_fallback"
        assert len(result.selected_sources) == len(plugins)

    @patch("plugins.source_selector.ChatOpenAI")
    def test_passes_plugin_descriptions_in_prompt(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(
            content='{"claim_type": "general", "sources": []}'
        )

        classifier = LLMSourceClassifier()
        classifier.classify("test claim", _sample_plugins())

        # Verify the prompt includes plugin descriptions
        call_args = mock_llm.invoke.call_args
        prompt_content = str(call_args)
        assert "portal_transparencia" in prompt_content
        assert "ibge_sidra" in prompt_content

    @patch("plugins.source_selector.ChatOpenAI")
    def test_filters_unknown_plugin_names(self, mock_llm_cls):
        """LLM may hallucinate plugin names — those should be filtered out."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(
            content='{"claim_type": "general", "sources": [{"plugin_name": "nonexistent_plugin", "relevance": 0.9, "reason": "hallucinated"}, {"plugin_name": "tavily_search", "relevance": 0.8, "reason": "real"}]}'
        )

        classifier = LLMSourceClassifier()
        plugins = _sample_plugins()
        result = classifier.classify("test", plugins)

        plugin_names = [s.plugin_name for s in result.selected_sources]
        assert "nonexistent_plugin" not in plugin_names
        assert "tavily_search" in plugin_names

    @patch("plugins.source_selector.ChatOpenAI")
    def test_strips_markdown_fences_from_response(self, mock_llm_cls):
        """LLMs sometimes wrap JSON in ```json ... ``` fences."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(
            content='```json\n{"claim_type": "demographics", "sources": [{"plugin_name": "ibge_sidra", "relevance": 0.9, "reason": "Population data"}]}\n```'
        )

        classifier = LLMSourceClassifier()
        result = classifier.classify("população do Brasil", _sample_plugins())

        assert result.claim_type == "demographics"
        assert result.classification_method == "llm"
        assert result.selected_sources[0].plugin_name == "ibge_sidra"
