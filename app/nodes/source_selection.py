"""Node: Select the most relevant data sources for a claim.

Uses a swappable SourceClassifier strategy to classify the claim type
and recommend data source plugins. The default is LLMSourceClassifier
(gpt-5-mini). See plugins/source_selector.py for future alternatives.
"""

import logging

from state import AgentState
from plugins.registry import get_available, get
from plugins.source_selector import SourceClassifier, LLMSourceClassifier

logger = logging.getLogger(__name__)

# Module-level classifier instance — swappable via set_classifier()
_classifier: SourceClassifier | None = None


def get_classifier() -> SourceClassifier:
    """Return the current classifier strategy. Defaults to LLMSourceClassifier."""
    global _classifier
    if _classifier is None:
        _classifier = LLMSourceClassifier()
    return _classifier


def set_classifier(classifier: SourceClassifier) -> None:
    """Swap the classifier strategy at runtime."""
    global _classifier
    _classifier = classifier


def select_sources(state: AgentState) -> dict:
    """Classify claim and select relevant data source plugins.

    Writes selected_sources and source_confidence into state for
    downstream nodes (search_online, create_report) to consume.
    """
    claim = state["claim"]
    logger.info("[select_sources] Classifying claim='%s'", claim[:80])

    available = get_available()
    classifier = get_classifier()
    result = classifier.classify(claim, [p.get_metadata() for p in available])

    # Build confidence map from plugin reliability scores
    source_confidence = {}
    for rec in result.selected_sources:
        plugin = get(rec.plugin_name)
        if plugin:
            source_confidence[rec.plugin_name] = plugin.get_metadata().reliability_score

    logger.info(
        "[select_sources] claim_type='%s' sources=%s method=%s",
        result.claim_type,
        [r.plugin_name for r in result.selected_sources],
        result.classification_method,
    )

    return {
        "selected_sources": [r.model_dump() for r in result.selected_sources],
        "source_confidence": source_confidence,
        "reasoning_log": [
            f"[select_sources] Classified claim as '{result.claim_type}', "
            f"selected {len(result.selected_sources)} sources: "
            f"{[r.plugin_name for r in result.selected_sources]} "
            f"(method={result.classification_method})"
        ],
    }
