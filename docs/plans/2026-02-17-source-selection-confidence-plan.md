# Phase 2: Source Selection & Confidence Scoring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Classify claims by type, select the most relevant data source plugins, and assign confidence scores — with a swappable classifier strategy (LLM MVP, future rules/ML).

**Architecture:** New `select_sources` LangGraph node between `factcheck_router` and `search_online`. A `SourceClassifier` ABC with an `LLMSourceClassifier` implementation uses gpt-4o-mini to classify claims and recommend plugins. `PluginMetadata` gains a `reliability_score` field. Both source reliability and selection relevance are surfaced in the final report prompt.

**Tech Stack:** LangGraph, LangChain, OpenAI (gpt-4o-mini for classifier), Pydantic, pytest

**Design doc:** `docs/plans/2026-02-17-source-selection-confidence-design.md`

---

### Task 1: Add `reliability_score` to PluginMetadata

**Files:**
- Modify: `app/plugins/base.py:21-32` (PluginMetadata model)
- Modify: `app/plugins/claim_databases/google_factcheck.py:59-70`
- Modify: `app/plugins/government_data/transparencia.py:20-30`
- Modify: `app/plugins/government_data/ibge_sidra.py:27-36`
- Modify: `app/plugins/web_search/tavily_search.py:17-25`
- Test: `tests/test_plugins/test_plugin_base.py`

**Step 1: Write the failing test**

Add to `tests/test_plugins/test_plugin_base.py`:

```python
class TestReliabilityScore:
    def test_default_reliability_score(self):
        meta = PluginMetadata(
            name="test",
            display_name="Test",
            description="A test plugin",
            category=PluginCategory.WEB_SEARCH,
        )
        assert meta.reliability_score == 0.5

    def test_custom_reliability_score(self):
        meta = PluginMetadata(
            name="test",
            display_name="Test",
            description="A test plugin",
            category=PluginCategory.WEB_SEARCH,
            reliability_score=0.9,
        )
        assert meta.reliability_score == 0.9

    def test_reliability_score_bounds(self):
        """Score must be between 0.0 and 1.0."""
        meta = PluginMetadata(
            name="test",
            display_name="Test",
            description="A test plugin",
            category=PluginCategory.WEB_SEARCH,
            reliability_score=0.0,
        )
        assert meta.reliability_score == 0.0
```

**Step 2: Run test to verify it fails**

Run: `cd app && python -m pytest ../tests/test_plugins/test_plugin_base.py::TestReliabilityScore -v`
Expected: FAIL — `reliability_score` not a field on PluginMetadata

**Step 3: Implement — add field to PluginMetadata**

In `app/plugins/base.py`, add to the `PluginMetadata` class after line 32:

```python
    reliability_score: float = 0.5  # 0.0-1.0, source trustworthiness
```

**Step 4: Set scores on all 4 plugins**

- `google_factcheck.py` `get_metadata()`: add `reliability_score=0.9`
- `transparencia.py` `get_metadata()`: add `reliability_score=0.85`
- `ibge_sidra.py` `get_metadata()`: add `reliability_score=0.9`
- `tavily_search.py` `get_metadata()`: add `reliability_score=0.6`

**Step 5: Run tests to verify they pass**

Run: `cd app && python -m pytest ../tests/ -v --tb=short`
Expected: all 196+ tests PASS (no regressions)

**Step 6: Commit**

```bash
git add app/plugins/base.py app/plugins/claim_databases/google_factcheck.py \
  app/plugins/government_data/transparencia.py app/plugins/government_data/ibge_sidra.py \
  app/plugins/web_search/tavily_search.py tests/test_plugins/test_plugin_base.py
git commit -m "feat: add reliability_score to PluginMetadata"
```

---

### Task 2: Create SourceClassifier ABC and models

**Files:**
- Create: `app/plugins/source_selector.py`
- Test: `tests/test_plugins/test_source_selector.py`

**Step 1: Write the failing tests**

Create `tests/test_plugins/test_source_selector.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd app && python -m pytest ../tests/test_plugins/test_source_selector.py -v`
Expected: FAIL — `plugins.source_selector` module not found

**Step 3: Create the module**

Create `app/plugins/source_selector.py`:

```python
"""Source classifier strategy — classifies claims and selects data sources.

Provides a swappable strategy interface (SourceClassifier ABC) so the
classification logic can be upgraded from LLM → rules → ML without
changing the rest of the pipeline.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from plugins.base import PluginMetadata


class SourceRecommendation(BaseModel):
    """A single source recommended for a claim."""

    plugin_name: str
    relevance: float  # 0.0-1.0, how relevant this source is for the claim
    reason: str       # brief justification


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
```

**Step 4: Run tests to verify they pass**

Run: `cd app && python -m pytest ../tests/test_plugins/test_source_selector.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add app/plugins/source_selector.py tests/test_plugins/test_source_selector.py
git commit -m "feat: add SourceClassifier ABC and Pydantic models"
```

---

### Task 3: Implement LLMSourceClassifier

**Files:**
- Modify: `app/plugins/source_selector.py`
- Test: `tests/test_plugins/test_source_selector.py`

**Step 1: Write the failing tests**

Append to `tests/test_plugins/test_source_selector.py`:

```python
from unittest.mock import patch, MagicMock

from plugins.base import PluginMetadata, PluginCategory
from plugins.source_selector import LLMSourceClassifier


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
```

**Step 2: Run test to verify it fails**

Run: `cd app && python -m pytest ../tests/test_plugins/test_source_selector.py::TestLLMSourceClassifier -v`
Expected: FAIL — `LLMSourceClassifier` not defined

**Step 3: Implement LLMSourceClassifier**

Append to `app/plugins/source_selector.py`:

```python
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

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
```

**Step 4: Run tests to verify they pass**

Run: `cd app && python -m pytest ../tests/test_plugins/test_source_selector.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add app/plugins/source_selector.py tests/test_plugins/test_source_selector.py
git commit -m "feat: implement LLMSourceClassifier with mocked tests"
```

---

### Task 4: Add `get_tools_for_selection` to registry

**Files:**
- Modify: `app/plugins/registry.py:47-52`
- Modify: `app/plugins/__init__.py:7`
- Test: `tests/test_plugins/test_registry.py`

**Step 1: Write the failing test**

Add to `tests/test_plugins/test_registry.py`:

```python
class TestGetToolsForSelection:
    def test_returns_tools_for_selected_names(self):
        p1 = _make_plugin("alpha", available=True)
        p2 = _make_plugin("beta", available=True)
        p3 = _make_plugin("gamma", available=True)
        registry.register(p1)
        registry.register(p2)
        registry.register(p3)

        selection = [
            {"plugin_name": "alpha", "relevance": 0.9, "reason": "relevant"},
            {"plugin_name": "gamma", "relevance": 0.7, "reason": "related"},
        ]
        tools = registry.get_tools_for_selection(selection)
        tool_names = [t.name for t in tools]
        assert "alpha" in tool_names
        assert "gamma" in tool_names
        assert "beta" not in tool_names

    def test_excludes_unavailable_plugins(self):
        p1 = _make_plugin("available_one", available=True)
        p2 = _make_plugin("unavailable_one", available=False)
        registry.register(p1)
        registry.register(p2)

        selection = [
            {"plugin_name": "available_one", "relevance": 0.9, "reason": "ok"},
            {"plugin_name": "unavailable_one", "relevance": 0.9, "reason": "ok"},
        ]
        tools = registry.get_tools_for_selection(selection)
        assert len(tools) == 1
        assert tools[0].name == "available_one"

    def test_handles_unknown_plugin_names(self):
        p1 = _make_plugin("known", available=True)
        registry.register(p1)

        selection = [
            {"plugin_name": "known", "relevance": 0.9, "reason": "ok"},
            {"plugin_name": "does_not_exist", "relevance": 0.9, "reason": "ok"},
        ]
        tools = registry.get_tools_for_selection(selection)
        assert len(tools) == 1

    def test_returns_empty_for_empty_selection(self):
        p1 = _make_plugin("something", available=True)
        registry.register(p1)
        tools = registry.get_tools_for_selection([])
        assert tools == []
```

Note: This test file already has a `_make_plugin` helper — check the existing test file. If it doesn't, create a simple factory that returns a MagicMock plugin with `get_metadata()` and `is_available()`.

**Step 2: Run test to verify it fails**

Run: `cd app && python -m pytest ../tests/test_plugins/test_registry.py::TestGetToolsForSelection -v`
Expected: FAIL — `get_tools_for_selection` not defined

**Step 3: Implement in registry.py**

Add to `app/plugins/registry.py` after the `get_langchain_tools` function:

```python
def get_tools_for_selection(selection: list[dict]) -> list:
    """Get LangChain Tool objects for specifically selected plugins.

    Used by the source selection node to provide only relevant tools
    to the research agent.

    Args:
        selection: List of dicts with at least a "plugin_name" key.
    """
    selected_names = {s["plugin_name"] for s in selection if s.get("plugin_name")}
    return [
        p.as_langchain_tool()
        for p in _registry.values()
        if p.get_metadata().name in selected_names and p.is_available()
    ]
```

Update `app/plugins/__init__.py` to export the new function:

```python
from plugins.registry import register, get, get_available, get_all, get_langchain_tools, get_tools_for_selection, clear  # noqa: F401
```

**Step 4: Run tests to verify they pass**

Run: `cd app && python -m pytest ../tests/test_plugins/test_registry.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add app/plugins/registry.py app/plugins/__init__.py tests/test_plugins/test_registry.py
git commit -m "feat: add get_tools_for_selection to plugin registry"
```

---

### Task 5: Create `select_sources` node + classifier factory

**Files:**
- Create: `app/nodes/source_selection.py`
- Test: `tests/test_nodes/test_source_selection.py`

**Step 1: Write the failing tests**

Create `tests/test_nodes/test_source_selection.py`:

```python
"""Tests for the select_sources node."""

from unittest.mock import patch, MagicMock

from plugins.source_selector import SourceSelectionResult, SourceRecommendation
from nodes.source_selection import select_sources


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
```

**Step 2: Run test to verify it fails**

Run: `cd app && python -m pytest ../tests/test_nodes/test_source_selection.py -v`
Expected: FAIL — module not found

**Step 3: Implement the node**

Create `app/nodes/source_selection.py`:

```python
"""Node: Select the most relevant data sources for a claim.

Uses a swappable SourceClassifier strategy to classify the claim type
and recommend data source plugins. The default is LLMSourceClassifier
(gpt-4o-mini). See plugins/source_selector.py for future alternatives.
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
```

**Step 4: Run tests to verify they pass**

Run: `cd app && python -m pytest ../tests/test_nodes/test_source_selection.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add app/nodes/source_selection.py tests/test_nodes/test_source_selection.py
git commit -m "feat: add select_sources node with classifier factory"
```

---

### Task 6: Wire into LangGraph + update state

**Files:**
- Modify: `app/state.py:36-37`
- Modify: `app/graph.py`
- Modify: `app/nodes/online_research.py:55-57`

**Step 1: Write the failing test**

Add to `tests/test_graph_integration.py`:

```python
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
```

**Step 2: Run to verify it fails**

Run: `cd app && python -m pytest ../tests/test_graph_integration.py::TestSourceSelectionIntegration -v`
Expected: FAIL — import error or assertion failure

**Step 3: Wire the graph**

Update `app/state.py` — add after line 37:

```python
    # Phase 2: Source selection & confidence scoring
    selected_sources: Optional[list[dict]]
    source_confidence: Optional[dict]
```

Update `app/graph.py`:

1. Add import: `from nodes.source_selection import select_sources`
2. Add node: `workflow.add_node("select_sources", select_sources)`
3. Change conditional edge routing: `"online": "select_sources"` (was `"online": "search_online"`)
4. Add new edge: `workflow.add_edge("select_sources", "search_online")`
5. Update docstring to reflect new flow

The full updated `app/graph.py`:

```python
"""Main workflow graph using LangGraph 0.6+."""

from langgraph.graph import END, StateGraph

from state import AgentState
from nodes.questions import list_questions
from nodes.factcheck_lookup import check_existing_factchecks, factcheck_router
from nodes.source_selection import select_sources
from nodes.online_research import search_online
from nodes.report import create_report
from nodes.gazette.subgraph import build_gazette_subgraph


def build_workflow():
    """Build the main fact-checking workflow.

    Flow:
      list_questions -> check_existing_factchecks -> factcheck_router
        -> "found":    create_report -> END   (short-circuit for known claims)
        -> "gazettes": gazette subgraph -> END
        -> "online":   select_sources -> search_online -> create_report -> END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("list_questions", list_questions)
    workflow.add_node("check_existing_factchecks", check_existing_factchecks)
    workflow.add_node("select_sources", select_sources)
    workflow.add_node("search_online", search_online)
    workflow.add_node("create_report", create_report)
    workflow.add_node("fact_check_gazettes", build_gazette_subgraph())

    # Entry point
    workflow.set_entry_point("list_questions")

    # After questions, check existing fact-checks first
    workflow.add_edge("list_questions", "check_existing_factchecks")

    # 3-way routing after fact-check lookup
    workflow.add_conditional_edges(
        "check_existing_factchecks",
        factcheck_router,
        {
            "found": "create_report",
            "gazettes": "fact_check_gazettes",
            "online": "select_sources",
        },
    )

    # Online path: select sources → search → report
    workflow.add_edge("select_sources", "search_online")
    workflow.add_edge("search_online", "create_report")
    workflow.add_edge("create_report", END)

    # Gazette path
    workflow.add_edge("fact_check_gazettes", END)

    return workflow.compile()
```

Update `app/nodes/online_research.py` lines 55-57 — replace:

```python
    tools.extend(get_langchain_tools(PluginCategory.GOVERNMENT_DATA))
```

With:

```python
    # Use source selection results if available, otherwise fall back to all plugins
    selected = state.get("selected_sources")
    if selected:
        from plugins.registry import get_tools_for_selection
        tools.extend(get_tools_for_selection(selected))
    else:
        tools.extend(get_langchain_tools())
```

And remove the unused import `from plugins.base import PluginCategory` if it's no longer used.

**Step 4: Run all tests**

Run: `cd app && python -m pytest ../tests/ -v --tb=short`
Expected: all tests PASS (previous + new)

**Step 5: Commit**

```bash
git add app/state.py app/graph.py app/nodes/online_research.py \
  tests/test_graph_integration.py
git commit -m "feat: wire select_sources into LangGraph, update online path"
```

---

### Task 7: Add source confidence to report prompt

**Files:**
- Modify: `app/nodes/report.py:18-63` (prompt template) and `105-133` (create_report function)
- Test: `tests/test_nodes/test_report_confidence.py`

**Step 1: Write the failing test**

Create `tests/test_nodes/test_report_confidence.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd app && python -m pytest ../tests/test_nodes/test_report_confidence.py -v`
Expected: FAIL — `_build_source_confidence_section` not found

**Step 3: Implement the helper and wire into prompt**

Add to `app/nodes/report.py`, after `_build_existing_factchecks_section`:

```python
def _build_source_confidence_section(state: AgentState) -> str:
    """Build a prompt section with source reliability and relevance scores."""
    confidence = state.get("source_confidence") or {}
    selected = state.get("selected_sources") or []
    if not confidence and not selected:
        return ""
    lines = ["Source reliability and relevance scores:"]
    for src in selected:
        reliability = confidence.get(src.get("plugin_name", ""), "N/A")
        lines.append(
            f"- {src.get('plugin_name', 'unknown')}: reliability={reliability}, "
            f"relevance={src.get('relevance', 'N/A')} ({src.get('reason', '')})"
        )
    lines.append(
        "\nWeigh higher-reliability sources more heavily in your analysis."
    )
    return "\n".join(lines)
```

Add `{source_confidence_section}` placeholder to the prompt template (line 23, after `{existing_factchecks_section}`):

```
{source_confidence_section}
```

In `create_report()`, add to the `chain.invoke()` call:

```python
"source_confidence_section": _build_source_confidence_section(state),
```

**Step 4: Run all tests**

Run: `cd app && python -m pytest ../tests/ -v --tb=short`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add app/nodes/report.py tests/test_nodes/test_report_confidence.py
git commit -m "feat: surface source confidence in report prompt"
```

---

### Task 8: Final integration test + cleanup

**Files:**
- Test: `tests/test_graph_integration.py`
- Verify: full test suite

**Step 1: Add end-to-end integration test**

Add to `tests/test_graph_integration.py`:

```python
class TestPhase2EndToEnd:
    """Verify the full online path with source selection."""

    @patch("nodes.source_selection.get_classifier")
    @patch("nodes.source_selection.get_available", return_value=[])
    @patch("nodes.source_selection.get", return_value=None)
    def test_online_path_includes_source_selection(
        self, mock_get, mock_avail, mock_get_clf
    ):
        """Source selection node runs and its output reaches search_online."""
        from plugins.source_selector import (
            SourceSelectionResult,
            SourceRecommendation,
        )

        clf = MagicMock()
        clf.classify.return_value = SourceSelectionResult(
            claim_type="government_spending",
            selected_sources=[
                SourceRecommendation(
                    plugin_name="portal_transparencia",
                    relevance=0.95,
                    reason="Spending claim",
                )
            ],
            classification_method="llm",
        )
        mock_get_clf.return_value = clf

        from nodes.source_selection import select_sources

        state = {
            "claim": "O governo gastou R$10 milhões",
            "language": "pt",
        }
        result = select_sources(state)

        # Verify state shape
        assert "selected_sources" in result
        assert "source_confidence" in result
        assert "reasoning_log" in result
        assert result["selected_sources"][0]["plugin_name"] == "portal_transparencia"
        assert result["selected_sources"][0]["relevance"] == 0.95
```

**Step 2: Run full test suite**

Run: `cd app && python -m pytest ../tests/ -v --tb=short`
Expected: all tests PASS, no regressions

**Step 3: Commit**

```bash
git add tests/test_graph_integration.py
git commit -m "test: add Phase 2 end-to-end integration tests"
```

**Step 4: Push and update PR**

```bash
git push origin claude/competent-clarke
```
