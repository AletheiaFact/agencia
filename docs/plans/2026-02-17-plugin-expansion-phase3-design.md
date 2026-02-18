# Phase 3: Plugin Expansion — TSE, BACEN, ClaimBuster, Wikipedia

**Date:** 2026-02-17
**Branch:** `claude/competent-clarke`
**Depends on:** Phase 1 (plugin architecture + 4 integrations), Phase 2 (source selection + confidence scoring)

## Problem

The pipeline currently has 4 data source plugins covering fact-check databases, federal spending, statistics, and web search. Major claim domains remain uncovered: electoral data, economic/financial indicators, claim-worthiness scoring, and encyclopedic facts. The 2026 Brazilian elections make electoral and economic coverage especially urgent.

## Goals

1. Add 4 new plugins following the established `DataSourcePlugin` pattern
2. Cover electoral claims (TSE), economic claims (BACEN), claim prioritization (ClaimBuster), and encyclopedic verification (Wikipedia/Wikidata)
3. No graph changes — plugins integrate automatically via the existing source selector
4. Two new plugin categories: `ELECTORAL` and `KNOWLEDGE_BASE` (directory only; enum value already exists)

## Non-Goals

- Graph modifications or new nodes
- State changes to `AgentState`
- ClaimBuster as a pipeline gate (plugin-only integration)
- DivulgaCandContas unofficial API stability guarantees
- Real-time election-night polling infrastructure

## Architecture

### Approach: Four Independent Plugins

Each plugin follows the established `DataSourcePlugin` ABC pattern. The source selector routes claims automatically based on plugin metadata descriptions. No graph or state changes needed.

### Plugin Inventory

| Plugin | Class | Category | Directory | Auth | Deps | Reliability |
|--------|-------|----------|-----------|------|------|-------------|
| TSE Open Data | `TSEPlugin` | `ELECTORAL` | `plugins/electoral/` | None | `ckanapi`, `requests` | 0.95 |
| BACEN | `BACENPlugin` | `GOVERNMENT_DATA` | `plugins/government_data/` | None | `python-bcb` | 0.95 |
| ClaimBuster | `ClaimBusterPlugin` | `CLAIM_DATABASE` | `plugins/claim_databases/` | `CLAIMBUSTER_API_KEY` | `requests` | 0.7 |
| Wikipedia/Wikidata | `WikipediaPlugin` | `KNOWLEDGE_BASE` | `plugins/knowledge_bases/` | None | `wikipedia-api` | 0.6 |

### New Directories

```
plugins/
├── electoral/
│   ├── __init__.py
│   └── tse.py
├── knowledge_bases/
│   ├── __init__.py
│   └── wikipedia.py
```

---

## Plugin Designs

### 1. TSE Plugin (`plugins/electoral/tse.py`)

Three sub-APIs behind one `search()` interface, routed by `endpoint` parameter (same pattern as Portal da Transparencia).

**Sub-APIs:**

| Endpoint | Base URL | Purpose |
|----------|----------|---------|
| `datasets` (default) | `dadosabertos.tse.jus.br/api/3/` | Search 166 CKAN datasets |
| `candidates` | `divulgacandcontas.tse.jus.br/` | Candidate profiles, assets, campaign finance |
| `results` | `resultados.tse.jus.br/` | Election results by year/office/region |

**`search()` signature:**

```python
def search(
    self,
    query: str,
    endpoint: str = "datasets",    # "datasets" | "candidates" | "results"
    election_year: str = "2024",
    state: str = "",               # UF code (e.g. "SP", "RJ") — empty = national
    office: str = "",              # "presidente", "governador", "prefeito", etc.
    **kwargs,
) -> PluginResult:
```

**Internal routing:**
- `datasets` -> `_search_ckan(query)` — `package_search?q={query}` via `ckanapi`
- `candidates` -> `_search_candidates(query, election_year, state)` — DivulgaCandContas REST
- `results` -> `_search_results(election_year, state, office)` — predictable JSON URLs

**Error handling:**
- DivulgaCandContas is undocumented — wrap all calls, return `PluginResult(error=...)` on non-200
- CKAN is stable — standard `ckanapi` error handling
- Results endpoint returns 404 for non-existent elections — return empty result, not error

**LangChain tool description:**
> "Search Brazilian electoral data from TSE (Superior Electoral Court). Covers election results, candidate registrations, campaign finance, and voter statistics. Use endpoint='datasets' for general search, 'candidates' for specific candidate lookups, 'results' for vote tallies."

### 2. BACEN Plugin (`plugins/government_data/bacen.py`)

Wraps `python-bcb` for Central Bank time series data. Same pattern as IBGE SIDRA.

**Common series:**

```python
COMMON_SERIES = {
    "selic": 432,           # SELIC target rate
    "ipca": 433,            # IPCA monthly inflation
    "igpm": 189,            # IGP-M monthly inflation
    "cambio_dolar": 1,      # USD/BRL exchange rate
    "cambio_euro": 21619,   # EUR/BRL exchange rate
    "pib_mensal": 4380,     # Monthly GDP
    "divida_publica": 4513, # Net public debt (% GDP)
    "desemprego": 24369,    # Unemployment rate (PNAD)
}
```

**`search()` signature:**

```python
def search(
    self,
    query: str,
    series_code: int = 0,          # SGS series code — 0 = auto-detect
    start_date: str = "",          # "YYYY-MM-DD" — empty = last 12 months
    end_date: str = "",
    **kwargs,
) -> PluginResult:
```

**Behavior:**
- If `series_code` provided, fetch via `bcb.sgs.get()`
- If not, `_detect_series(query)` — keyword matching against `COMMON_SERIES`
- Returns `[{"date": ..., "value": ..., "series": ...}]`
- `is_available()` checks `import bcb` succeeds

**LangChain tool description:**
> "Query Brazilian Central Bank (BACEN) economic data. Covers SELIC interest rate, inflation (IPCA, IGP-M), exchange rates (USD, EUR), GDP, public debt, and unemployment. Use for verifying claims about Brazilian economic indicators."

### 3. ClaimBuster Plugin (`plugins/claim_databases/claimbuster.py`)

Single REST endpoint, returns check-worthiness scores.

**API:** `POST https://idir.uta.edu/claimbuster/api/v2/score/text/`
- Input: `input_text` parameter
- Output: `[{"text": sentence, "score": 0.0-1.0}]`

**`search()` signature:**

```python
def search(
    self,
    query: str,       # Claim text to score
    **kwargs,
) -> PluginResult:
```

**Behavior:**
- POST claim text, get per-sentence scores
- `is_available()` checks `CLAIMBUSTER_API_KEY` env var
- Returns `[{"sentence": ..., "score": ..., "check_worthy": bool}]` where `check_worthy = score >= 0.5`

**LangChain tool description:**
> "Score how check-worthy a claim is using ClaimBuster AI. Returns a 0-1 score indicating how important it is to fact-check the statement. Useful for prioritizing which claims to investigate."

### 4. Wikipedia/Wikidata Plugin (`plugins/knowledge_bases/wikipedia.py`)

Two knowledge sources behind one plugin, routed by `endpoint` parameter.

| Endpoint | API | Purpose |
|----------|-----|---------|
| `wikipedia` (default) | MediaWiki API (`pt.wikipedia.org`) | Article summaries |
| `wikidata` | SPARQL (`query.wikidata.org`) | Structured entity data |

**`search()` signature:**

```python
def search(
    self,
    query: str,
    endpoint: str = "wikipedia",   # "wikipedia" | "wikidata"
    language: str = "pt",
    **kwargs,
) -> PluginResult:
```

**Wikipedia (`_search_wikipedia`):**
- Uses `wikipedia-api` library
- Returns top 3 article summaries: `{"title": ..., "summary": ..., "url": ...}`
- Targets `pt.wikipedia.org` by default

**Wikidata (`_search_wikidata`):**
- SPARQL query via `requests` to `query.wikidata.org/sparql`
- Returns entity properties: `{"entity": ..., "description": ..., "properties": {...}}`
- Use case: "Person X holds position Y", entity disambiguation

**`is_available()`:** Checks `import wikipediaapi` succeeds.

**LangChain tool description:**
> "Search Wikipedia and Wikidata for encyclopedic facts. Use endpoint='wikipedia' for article summaries (Portuguese by default), 'wikidata' for structured entity data (politicians, cities, organizations, dates). Useful for verifying general knowledge claims and disambiguating entities."

---

## Registration Changes

**`plugins/base.py`** — add `ELECTORAL` to `PluginCategory`:

```python
class PluginCategory(str, Enum):
    CLAIM_DATABASE = "claim_databases"
    GOVERNMENT_DATA = "government_data"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_BASE = "knowledge_bases"
    FACT_CHECK_ORG = "fact_check_orgs"
    ACADEMIC = "academic"
    LEGISLATION = "legislation"
    MULTIMEDIA = "multimedia"
    ELECTORAL = "electoral"          # NEW
```

**`plugins/__init__.py`** — register new plugins:

```python
from plugins.electoral.tse import TSEPlugin
from plugins.government_data.bacen import BACENPlugin
from plugins.claim_databases.claimbuster import ClaimBusterPlugin
from plugins.knowledge_bases.wikipedia import WikipediaPlugin

register(TSEPlugin())
register(BACENPlugin())
register(ClaimBusterPlugin())
register(WikipediaPlugin())
```

---

## Dependencies

New entries in `requirements.txt`:

```
# Electoral data
ckanapi>=0.1.0

# Central Bank data
python-bcb>=0.3.0

# Wikipedia
wikipedia-api>=0.6.0
```

`requests` already available. ClaimBuster uses it directly.

---

## Testing Strategy

**Test files:**

```
tests/test_plugins/
├── test_tse.py
├── test_bacen.py
├── test_claimbuster.py
├── test_wikipedia.py
```

**Per-plugin test coverage:**
- Metadata tests (name, category, reliability_score)
- `is_available()` tests (with/without dependencies)
- `search()` happy path with mocked HTTP responses
- `search()` error handling (API failures, timeouts, empty results)
- Keyword/endpoint detection tests (TSE, BACEN routing)
- Multi-endpoint routing (TSE: datasets/candidates/results, Wikipedia: wikipedia/wikidata)

**Integration test additions** (`test_graph_integration.py`):
- All 8 plugins register successfully (up from 4)
- Source selector routes electoral claims to `tse`
- Source selector routes economic claims to `bacen`

All tests mock external HTTP calls — no real API calls in CI.

---

## Files Summary

### New Files
| File | Purpose |
|------|---------|
| `app/plugins/electoral/__init__.py` | Package init |
| `app/plugins/electoral/tse.py` | TSE plugin (CKAN + DivulgaCandContas + results) |
| `app/plugins/government_data/bacen.py` | BACEN plugin |
| `app/plugins/claim_databases/claimbuster.py` | ClaimBuster plugin |
| `app/plugins/knowledge_bases/__init__.py` | Package init |
| `app/plugins/knowledge_bases/wikipedia.py` | Wikipedia/Wikidata plugin |
| `tests/test_plugins/test_tse.py` | TSE tests |
| `tests/test_plugins/test_bacen.py` | BACEN tests |
| `tests/test_plugins/test_claimbuster.py` | ClaimBuster tests |
| `tests/test_plugins/test_wikipedia.py` | Wikipedia tests |

### Modified Files
| File | Changes |
|------|---------|
| `app/plugins/base.py` | Add `ELECTORAL` to `PluginCategory` enum |
| `app/plugins/__init__.py` | Register 4 new plugins in `register_all_plugins()` |
| `requirements.txt` | Add `ckanapi`, `python-bcb`, `wikipedia-api` |
| `tests/test_graph_integration.py` | Verify 8 plugins register, routing tests |
