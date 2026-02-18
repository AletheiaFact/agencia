"""Source classifier strategy — classifies claims and selects data sources.

Provides a swappable strategy interface (SourceClassifier ABC) so the
classification logic can be upgraded from LLM → rules → ML without
changing the rest of the pipeline.
"""

import json
import logging
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from plugins.base import PluginMetadata

logger = logging.getLogger(__name__)


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


_CLASSIFIER_SYSTEM_PROMPT = """You are a source selection engine for a fact-checking system.
Given a claim and a list of available data source plugins, classify the claim type and select
the most relevant sources for verification.

Available plugins:
{plugins_description}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "claim_type": "<category like political_spending, health_statistics, demographics, economic_data, general>",
  "sources": [
    {{"plugin_name": "<exact plugin name>", "relevance": <0.0-1.0>, "reason": "<brief justification>"}}
  ]
}}

Rules:
- Always include at least one source
- Only use plugin names from the available list above
- relevance is how useful this source is for THIS specific claim (0.0 = irrelevant, 1.0 = perfect match)
- For general/unclear claims, include the web search plugin with moderate relevance
"""


class LLMSourceClassifier(SourceClassifier):
    """LLM-based claim classifier using gpt-4o-mini.

    Classifies claims by type and recommends the most relevant data sources.
    Uses a cheap, fast model to minimize latency and cost.

    This is the MVP strategy. See SourceClassifier docstring for future
    alternatives (RuleBasedClassifier, EmbeddingClassifier).
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        self._model = model
        self._temperature = temperature

    def classify(
        self, claim: str, available_plugins: list[PluginMetadata]
    ) -> SourceSelectionResult:
        plugins_desc = "\n".join(
            f"- {p.name}: {p.description}" for p in available_plugins
        )

        try:
            llm = ChatOpenAI(model=self._model, temperature=self._temperature)
            response = llm.invoke([
                SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT.format(
                    plugins_description=plugins_desc
                )),
                HumanMessage(content=f"Classify this claim and select sources:\n\n{claim}"),
            ])

            return self._parse_response(response.content, available_plugins)
        except Exception as e:
            logger.warning("[LLMSourceClassifier] LLM call failed: %s, using fallback", e)
            return self._fallback_result(available_plugins)

    def _parse_response(
        self, content: str, available_plugins: list[PluginMetadata]
    ) -> SourceSelectionResult:
        valid_names = {p.name for p in available_plugins}

        try:
            raw = content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines).strip()

            data = json.loads(raw)
            claim_type = data.get("claim_type", "general")

            sources = []
            for src in data.get("sources", []):
                name = src.get("plugin_name", "")
                if name in valid_names:
                    sources.append(SourceRecommendation(
                        plugin_name=name,
                        relevance=float(src.get("relevance", 0.5)),
                        reason=src.get("reason", ""),
                    ))

            if not sources:
                return self._fallback_result(available_plugins)

            return SourceSelectionResult(
                claim_type=claim_type,
                selected_sources=sources,
                classification_method="llm",
            )
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("[LLMSourceClassifier] Parse error: %s", e)
            return self._fallback_result(available_plugins)

    @staticmethod
    def _fallback_result(available_plugins: list[PluginMetadata]) -> SourceSelectionResult:
        return SourceSelectionResult(
            claim_type="general",
            selected_sources=[
                SourceRecommendation(
                    plugin_name=p.name,
                    relevance=0.5,
                    reason="Fallback: included all sources",
                )
                for p in available_plugins
            ],
            classification_method="llm_fallback",
        )
