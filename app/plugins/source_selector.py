"""Source classifier strategy — classifies claims and selects data sources.

Provides a swappable strategy interface (SourceClassifier ABC) so the
classification logic can be upgraded from LLM → rules → ML without
changing the rest of the pipeline.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from plugins.base import PluginMetadata


class SourceRecommendation(BaseModel):
    """A single source recommended for a claim."""

    plugin_name: str
    relevance: float = Field(ge=0.0, le=1.0)  # how relevant this source is for the claim
    reason: str  # brief justification


class SourceSelectionResult(BaseModel):
    """Result of claim classification and source selection."""

    claim_type: str  # e.g., "political_spending", "health_statistics", "general"
    selected_sources: list[SourceRecommendation]
    classification_method: str  # "llm", "rules", "ml" — for observability


class SourceClassifier(ABC):
    """Strategy interface for claim classification and source selection.

    Implementations decide which data source plugins are most relevant
    for a given claim. The interface is intentionally simple to allow
    swapping between LLM-based, rule-based, and ML-based strategies.

    Future strategies (not implemented):
    - RuleBasedClassifier: keyword/regex mapping (cheapest, ~0ms)
    - EmbeddingClassifier: sentence embeddings + cosine similarity (~10ms)
    """

    @abstractmethod
    def classify(
        self, claim: str, available_plugins: list[PluginMetadata]
    ) -> SourceSelectionResult:
        """Classify claim type and recommend data sources.

        Args:
            claim: The fact-checking claim text.
            available_plugins: Metadata for all available plugins.

        Returns:
            SourceSelectionResult with claim_type and ranked source list.
        """
        ...
