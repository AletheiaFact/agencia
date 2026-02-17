# Phase 2: Source Selection & Confidence Scoring

**Date:** 2026-02-17
**Branch:** `claude/competent-clarke`
**Depends on:** Phase 1 (PR #16 — plugin architecture + 4 integrations)

## Problem

Phase 1 loads all available plugins as LangChain tools and lets the LLM agent decide which to call at runtime. This works but has no intelligence about claim type — a claim about government spending gets the same tool set as a claim about population statistics. There's also no confidence scoring, so the final report treats all sources equally.

## Goals

1. Classify claims by type and select the most relevant data sources
2. Assign reliability scores to sources and relevance scores to selections
3. Make the classification strategy swappable (LLM now, rules/ML later)
4. Surface confidence information in the final report

## Non-Goals

- Multi-source parallel orchestration (Phase 2 PRD scope, deferred)
- Result synthesis/merging across sources (deferred)
- Performance tracking / metrics dashboard (deferred)
- ML or embedding-based classifiers (documented as future work)

## Architecture

### Updated Graph Flow

```
list_questions → check_existing_factchecks → factcheck_router
  → "found":    create_report → END
  → "gazettes": gazette subgraph → END
  → "online":   select_sources → search_online → create_report → END
```

`select_sources` is a new LangGraph node on the online path, between `factcheck_router` and `search_online`.

### New State Fields

```python
class AgentState(TypedDict, total=False):
    # ... existing fields ...

    # Phase 2: Source selection & confidence
    selected_sources: Optional[list[dict]]   # [{name, relevance, reason}]
    source_confidence: Optional[dict]        # {source_name: reliability_score}
```

### New Modules

#### `app/plugins/source_selector.py`

Swappable classifier strategy:

```python
class SourceClassifier(ABC):
    @abstractmethod
    def classify(self, claim: str, available_plugins: list[PluginMetadata]) -> SourceSelectionResult:
        ...

class SourceSelectionResult(BaseModel):
    claim_type: str                              # "political_spending", "health_statistics", "general"
    selected_sources: list[SourceRecommendation]
    classification_method: str                   # "llm", "rules", "ml"

class SourceRecommendation(BaseModel):
    plugin_name: str
    relevance: float    # 0.0-1.0
    reason: str         # brief justification
```

**LLMSourceClassifier** (MVP implementation):
- Uses gpt-4o-mini with structured output
- Receives claim + list of available plugin names/descriptions
- Returns JSON with claim_type + source recommendations
- ~200ms latency, ~$0.0001 per classification

#### `app/nodes/source_selection.py`

New LangGraph node:

```python
def select_sources(state: AgentState) -> dict:
    claim = state["claim"]
    available = get_available()  # all registered + available plugins
    classifier = get_classifier()  # returns current SourceClassifier impl
    result = classifier.classify(claim, [p.get_metadata() for p in available])

    # Build confidence map from plugin reliability scores
    source_confidence = {}
    for rec in result.selected_sources:
        plugin = get(rec.plugin_name)
        if plugin:
            source_confidence[rec.plugin_name] = plugin.get_metadata().reliability_score

    return {
        "selected_sources": [r.model_dump() for r in result.selected_sources],
        "source_confidence": source_confidence,
        "reasoning_log": [
            f"[select_sources] Classified claim as '{result.claim_type}', "
            f"selected {len(result.selected_sources)} sources: "
            f"{[r.plugin_name for r in result.selected_sources]}"
        ],
    }
```

### Modified Modules

#### `app/plugins/base.py`

Add `reliability_score` to `PluginMetadata`:

```python
class PluginMetadata(BaseModel):
    # ... existing fields ...
    reliability_score: float = 0.5  # 0.0-1.0, static per source
```

Scores:
- Google Fact Check API: 0.9 (curated organizations)
- Portal da Transparencia: 0.85 (official government data)
- IBGE SIDRA: 0.9 (official statistics bureau)
- Tavily Search: 0.6 (web search, mixed quality)

#### `app/plugins/registry.py`

New function:

```python
def get_tools_for_selection(selection: list[dict]) -> list:
    """Get LangChain tools only for selected plugin names."""
    selected_names = {s["plugin_name"] for s in selection if s.get("plugin_name")}
    return [
        p.as_langchain_tool()
        for p in _registry.values()
        if p.get_metadata().name in selected_names and p.is_available()
    ]
```

#### `app/nodes/online_research.py`

Read selected sources from state:

```python
# Replace: tools.extend(get_langchain_tools(PluginCategory.GOVERNMENT_DATA))
# With:
selected = state.get("selected_sources")
if selected:
    tools.extend(get_tools_for_selection(selected))
else:
    tools.extend(get_langchain_tools())  # fallback: all available
```

#### `app/nodes/report.py`

Add source confidence context to the report prompt:

```python
def _build_source_confidence_section(state: AgentState) -> str:
    confidence = state.get("source_confidence") or {}
    selected = state.get("selected_sources") or []
    if not confidence and not selected:
        return ""
    lines = ["Source reliability and relevance scores:"]
    for src in selected:
        reliability = confidence.get(src["plugin_name"], "N/A")
        lines.append(
            f"- {src['plugin_name']}: reliability={reliability}, "
            f"relevance={src.get('relevance', 'N/A')} ({src.get('reason', '')})"
        )
    lines.append("\nWeigh higher-reliability sources more heavily in your analysis.")
    return "\n".join(lines)
```

#### `app/graph.py`

Add the new node and edge:

```python
workflow.add_node("select_sources", select_sources)

# Change: "online" now routes to select_sources instead of search_online
# "online": "select_sources"
workflow.add_edge("select_sources", "search_online")
```

#### `app/state.py`

Add new fields:

```python
selected_sources: Optional[list[dict]]
source_confidence: Optional[dict]
```

### Confidence Scoring Model

**Two-level scoring:**

1. **Source reliability** (static, per plugin): Set in `PluginMetadata.reliability_score`. Reflects the inherent trustworthiness of the data source.

2. **Selection relevance** (dynamic, per claim): The classifier rates how relevant each source is for the specific claim. A claim about public spending gets `transparencia: relevance=0.95` but `ibge_sidra: relevance=0.2`.

Both scores are passed to `create_report` in the prompt, so the LLM can weigh evidence proportionally.

## Testing Strategy

- **Unit**: SourceClassifier ABC contract, LLMSourceClassifier with mocked LLM response
- **Node**: select_sources node with mocked classifier, verify state updates
- **Integration**: Full graph flow — verify selected_sources propagates to search_online and create_report
- **Confidence**: Verify reliability scores from PluginMetadata land in report prompt
- **Fallback**: search_online still works when selected_sources is empty (backward compat)

## Future Considerations

### Alternative Classifier Strategies (Not Implemented)

**RuleBasedClassifier**: Keyword/regex mapping for known patterns. Example: claim contains "orcamento", "verba", "gasto" → transparencia + tavily. Cheapest (~0ms), most predictable, but brittle for novel claim types.

**EmbeddingClassifier**: Pre-compute embeddings for each plugin's description. At classification time, embed the claim and use cosine similarity to rank plugins. ~10ms per classification, no LLM cost. Requires a sentence-transformer model.

### Optimization Opportunities

- **Caching**: Cache claim_type → source_selection for repeated/similar claims (e.g., TTL-based with claim embedding as key)
- **A/B testing**: Run multiple classifier strategies in parallel, track which selection leads to better report quality (measured by classification confidence or user feedback)
- **Cost tracking**: Monitor LLM classifier costs per invocation and trigger switch to RuleBasedClassifier when costs exceed threshold
- **Adaptive thresholds**: Adjust minimum relevance threshold based on historical success rates per source
