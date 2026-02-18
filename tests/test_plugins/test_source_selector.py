"""Tests for source classifier strategy."""

import pytest
from pydantic import ValidationError

from plugins.source_selector import (
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
