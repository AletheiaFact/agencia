# PRD: Improve Gazette Search Reasoning and Accuracy

**Date:** 2026-02-17
**Status:** Implemented
**Author:** Claude (with human direction)

---

## Implementation Notes (Post-Implementation)

> These notes document changes made during implementation that differ from the original design below.

### Model Changes
- **gpt-5 → gpt-5.2-2025-12-11**: All reasoning nodes (search_planner, evidence_evaluator, query_refiner, cross_checker) use `gpt-5.2-2025-12-11`
- **gpt-5-mini → gpt-5-mini-2025-08-07**: All extraction/synthesis nodes (fetch_and_score, deep_analyzer, gazette_reporter)
- **Temperature constraint**: Both models only support `temperature=1` (default). No other value is accepted.

### Resilience: Map-Reduce Batching (deep_analyzer.py)
The deep analyzer uses a **map-reduce** pattern to handle large gazette documents without losing evidence:
1. **MAP**: FAISS returns relevant passages, split into batches of `BATCH_CHAR_LIMIT = 4000` chars
2. Each batch gets a mini-summary from the LLM extracting all evidence
3. **REDUCE**: All mini-summaries are merged into a final synthesis
- Single-batch case skips the MAP step and goes directly to synthesis
- This avoids token-limit errors while preserving ALL evidence (no truncation)

### Resilience: invoke_with_retry (utils/llm_retry.py)
Safety-net utility for LLM calls that exceed token limits:
- Catches OpenAI `BadRequestError` with token-limit messages
- Retries up to 3 times, shrinking the largest truncatable field by 50% each retry
- Only used as a last resort (map-reduce is the primary strategy)
- Applied in: deep_analyzer, fetch_and_score, cross_checker, gazette_reporter

### Resilience: FAISS Embedding Retry (gazette_deep_search.py)
- If building the FAISS index fails on token limits, halves the chunk count and retries (up to 3 attempts)
- `MAX_CHARS_PER_GAZETTE = 80,000` — per-gazette text truncation before chunking

### Parameter Adjustments
- `MAX_GAZETTES = 3` (originally 3-5 in design, fixed at 3 for token safety)
- `top_k = 10` (originally 15 in design, reduced for token safety)
- `SCORING_BATCH_SIZE = 10` — fetch_and_score processes gazettes in batches of 10

### Reasoning Log Timeline
- `reasoning_log: Annotated[list[str], operator.add]` added to `AgentState` — uses LangGraph reducer for automatic list concatenation across nodes
- Every gazette subgraph node appends descriptive log entries
- `gazette_reporter.py` programmatically injects the full `reasoning_log` into the final JSON output (not via LLM, so the timeline is exact)

### Batch Scoring (fetch_and_score.py)
- Instead of scoring all gazettes at once, scores in batches of 10 to avoid token limits
- Each batch uses `invoke_with_retry` as additional safety

---

## Context

The gazette fact-checking pipeline currently has three compounding accuracy problems:

1. **Only 1 gazette is ever analyzed** — `querido_diario.py:71` takes `gazettes[0]` and discards all other results
2. **No query adaptation** — a single search query is generated; if it's too specific/broad, there's no recovery
3. **Unused API capabilities** — the Querido Diario API supports `size`, `offset`, `excerpt_size`, `number_of_excerpts`, and highlighting, none of which are used

These issues mean the system frequently returns irrelevant results, misses relevant gazettes that exist in the database, and draws conclusions from a single document when evidence may be spread across multiple publications.

## Goal

Redesign the gazette subgraph into an **adaptive search loop** that:
- Generates multiple search strategies per claim
- Fetches and scores many gazette candidates using excerpts
- Downloads and analyzes the top 3-5 most relevant gazettes
- Can refine its search if initial results are insufficient
- Achieves measurable accuracy improvements on golden test cases

## Success Criteria

- Pipeline finds relevant gazette(s) for all 4 golden test scenarios (see below)
- Correct classification on >= 3 of 4 golden test cases
- Graceful "Unverifiable" classification when no data exists (scenario 5)
- No regression in pipeline latency beyond 2x current (adaptive loop has max 3 iterations)

---

## Architecture: Adaptive Search Loop

### Current Pipeline (linear, no recovery)

```
create_subject → fetch_gazettes(top 1) → analyze_gazette(1 doc, FAISS k=10) → cross_check → report
```

### Proposed Pipeline

```
plan_search_strategies
  → [LOOP max 3 iterations]:
      fetch_and_score_gazettes(size=30, excerpts=3, excerpt_size=500)
      → evaluate_evidence_sufficiency
        → if sufficient: BREAK
        → if insufficient: refine_queries
  → download_top_gazettes(top 3-5 by excerpt relevance)
  → deep_analyze(FAISS across all downloaded docs, k=15)
  → cross_check
  → report
```

### New Subgraph Node Map

| Node | Purpose | Model | Tools |
|------|---------|-------|-------|
| `plan_search` | Generate 2-3 search query strategies from claim + context | gpt-5.2-2025-12-11 | none (pure LLM) |
| `fetch_and_score` | Execute queries, score results by excerpt relevance | gpt-5-mini-2025-08-07 | `querido_diario_search` (new) |
| `evaluate_evidence` | Decide: sufficient evidence or refine? (loop control) | gpt-5.2-2025-12-11 | none (pure LLM) |
| `refine_queries` | Adjust queries based on what was/wasn't found | gpt-5.2-2025-12-11 | none (pure LLM) |
| `download_and_analyze` | Fetch full text of top gazettes, map-reduce FAISS analysis | gpt-5-mini-2025-08-07 | `gazette_deep_search` (new) |
| `cross_check` | Classify claim against evidence (existing, upgraded) | gpt-5.2-2025-12-11 | none (pure LLM) |
| `gazette_report` | Format final JSON report with reasoning_log timeline | gpt-5-mini-2025-08-07 | none (pure LLM) |

### State Changes

New fields added to `AgentState`:

```python
class AgentState(TypedDict, total=False):
    # ... existing fields ...
    search_strategies: Optional[list[str]]       # Multiple query strings from planner
    gazette_candidates: Optional[list[dict]]     # All gazette results with relevance scores
    evidence_sufficient: Optional[bool]          # Loop control flag
    search_iteration: Optional[int]              # Current loop iteration (0-2)
    selected_gazettes: Optional[list[dict]]      # Top gazettes chosen for deep analysis
    evidence_summary: Optional[str]              # Accumulated evidence across iterations
    gazette_analysis: Optional[str]              # Deep analysis result
    cross_check_result: Optional[str]            # Cross-check classification result
    reasoning_log: Annotated[list[str], operator.add]  # Timeline of reasoning steps (auto-appended via LangGraph reducer)
```

### LangGraph Loop Implementation

The loop uses a conditional edge from `evaluate_evidence`:

```python
def should_continue_search(state: AgentState) -> str:
    if state.get("evidence_sufficient", False):
        return "download"        # break out of loop
    if state.get("search_iteration", 0) >= 3:
        return "download"        # max iterations reached
    return "refine"              # loop back to refine queries

# Edges:
# plan_search → fetch_and_score → evaluate_evidence
# evaluate_evidence → (conditional) → refine_queries → fetch_and_score  (loop)
#                                    → download_and_analyze              (break)
# download_and_analyze → cross_check → gazette_report → END
```

---

## Component Details

### 1. Query Planner (`plan_search`)

**Input:** claim, context (city, dates), questions
**Output:** `search_strategies: list[str]` (2-3 queries)

The planner generates diverse search strategies:
- **Strategy 1 (Exact):** Key entities and specific terms from the claim, using `+` AND operators and exact phrases
- **Strategy 2 (Broad):** Core topic with related terms using `|` OR operators
- **Strategy 3 (Contextual):** Related government processes (e.g., claim about hospital → "licitacao" + "medicamentos")

Uses the existing `querido_diario_search_context.txt` for Elasticsearch operator reference.

**Model:** gpt-5 — query planning requires strong reasoning to decompose claims into effective search strategies.

### 2. Fetch and Score (`fetch_and_score`)

**Input:** `search_strategies`, context
**Output:** `gazette_candidates: list[dict]` (scored and ranked)

Calls the **new** `querido_diario_search` tool for EACH strategy:
- `size=30` results per query
- `excerpt_size=500` characters per excerpt
- `number_of_excerpts=3` excerpts per gazette
- `sort_by=relevance`

After fetching, merges results across queries (dedup by `txt_url`), then scores each gazette:
- Excerpt text relevance to claim (simple keyword/entity overlap scoring)
- Date proximity to claim's date range
- Territory match confidence

**Model:** gpt-5-mini — scoring is a simpler extraction/comparison task.

### 3. Evidence Evaluator (`evaluate_evidence`)

**Input:** `gazette_candidates`, claim, `search_iteration`
**Output:** `evidence_sufficient: bool`, `evidence_summary: str`

Reviews the top-scored excerpts and decides:
- Do the excerpts contain information directly relevant to the claim?
- Is there enough to support or refute the claim?
- Or should we try different search terms?

**Model:** gpt-5 — this is a critical judgment call that determines pipeline behavior.

### 4. Query Refiner (`refine_queries`)

**Input:** `search_strategies` (previous), `gazette_candidates` (what was found), claim
**Output:** Updated `search_strategies`, incremented `search_iteration`

Analyzes why previous queries may have failed:
- Too specific? → broaden with OR operators, prefix wildcards
- Too broad? → add city-specific terms, narrow date range
- Wrong concepts? → try related government terminology

**Model:** gpt-5 — needs reasoning about what went wrong and how to fix it.

### 5. Download and Deep Analyze (`download_and_analyze`)

**Input:** `gazette_candidates` (top 3 by score), claim, questions
**Output:** `gazette_analysis: str`

- Downloads full `.txt` for top 3 gazettes (MAX_GAZETTES=3)
- Each gazette truncated to MAX_CHARS_PER_GAZETTE=80,000 chars
- Chunks all documents together using `RecursiveCharacterTextSplitter`
- Creates a single FAISS index across all documents (with embedding retry on failure)
- Retrieves top `k=10` chunks most relevant to the claim
- **Map-reduce pattern** to avoid token limits while preserving all evidence:
  - MAP: Split passages into batches of 4,000 chars, extract evidence from each batch
  - REDUCE: Merge all batch summaries into final synthesis
  - Single-batch case skips MAP step
- Safety net: `invoke_with_retry` for each LLM call

**Model:** gpt-5-mini-2025-08-07 — extraction and synthesis, not high-level reasoning.

### 6. Cross-Check (upgraded existing)

**Input:** claim, `gazette_analysis`, `evidence_summary`
**Output:** `cross_check_result: str`

Same classification categories as current, but now receives richer evidence from multiple gazettes. Uses the evidence summary from the evaluator to understand what was and wasn't found.

**Model:** gpt-5 (upgraded from gpt-5-mini) — classification accuracy is the most important output.

### 7. Report (minor changes to existing)

Same as current but includes:
- Which gazettes were analyzed (URLs, dates, territories)
- Search strategies that were tried
- Confidence level based on evidence sufficiency

---

## Tool Changes

### New: `querido_diario_search` (replaces `querido_diario_fetch`)

```python
@tool("querido_diario_search")
def querido_diario_search(
    queries: list[str],                    # Multiple search strings
    city: Optional[str] = None,
    published_since: Optional[str] = None,
    published_until: Optional[str] = None,
    size: int = 30,                        # Results per query
    excerpt_size: int = 500,               # Characters per excerpt
    number_of_excerpts: int = 3,           # Excerpts per gazette
) -> dict:
    """Search Querido Diario API with multiple queries, returning scored results."""
```

Key differences from current tool:
- Accepts **list of queries** and merges results
- Returns **all matching gazettes** (not just first)
- Requests **rich excerpts** for pre-filtering
- Deduplicates by `txt_url`
- Returns `total_gazettes` count for each query (useful for evaluator)

### Updated: `gazette_deep_search` (replaces `gazette_search_context`)

```python
@tool("gazette_deep_search")
def gazette_deep_search(
    claim: str,
    urls: list[str],                       # Multiple gazette URLs
    questions: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 15,                       # More results from FAISS
) -> str:
    """Download and search across multiple gazette documents."""
```

Key differences:
- Accepts **list of URLs** (multi-document)
- `top_k` default: 10 (reduced from original 15 for token safety)
- Builds single FAISS index across all documents
- Per-gazette truncation: `MAX_CHARS_PER_GAZETTE = 80,000` chars
- FAISS embedding retry: halves chunks on token-limit error (up to 3 attempts)

---

## Golden Test Scenarios

### Scenario 1: Porto Alegre Emergency Flood Contracts (TRUE)

**Claim:** "A Prefeitura de Porto Alegre firmou contratos emergenciais com empresas privadas para limpeza urbana apos as enchentes de maio de 2024."

**Context:** city=Porto Alegre, published_since=2024-05-01, published_until=2024-07-31

**API Evidence:**
- Query `"contrato emergencial"` with territory_id `4314902` returns **20 gazettes**
- Top result (2024-05-17): "EXTRATO DO CONTRATO EMERGENCIAL 002/2024 PROCESSO 24.17.000001778-5" — DMLU contracted LOCAR VEICULOS E EQUIPAMENTOS LTDA
- Second result (2024-06-11): contracts 004, 006, 007/2024 with AGUIAR SERVICOS, MECANICAPINA, FG SOLUCOES AMBIENTAIS
- Supporting query `"estado de calamidade"` returns **49 gazettes** including Decreto No 22.647

**Expected Classification:** Trustworthy
**Expected Behavior:** Pipeline should find emergency contracts and calamity decree, confirm the claim.

### Scenario 2: Rio de Janeiro Hospital Medication Procurement (TRUE)

**Claim:** "A Prefeitura do Rio de Janeiro realizou licitacoes para aquisicao de medicamentos para hospitais municipais em 2024."

**Context:** city=Rio de Janeiro, published_since=2024-01-01, published_until=2024-12-31

**API Evidence:**
- Query `"licitacao" + "medicamentos"` with territory_id `3304557` returns **239 gazettes**
- Multiple procurement records with company names (LEMAN MEDICAMENTOS, ANJOMEDI, ANTIBIOTICOS DO BRASIL), process numbers, and monetary values
- txt_url: `https://data.queridodiario.ok.org.br/3304557/2024-05-14/8b2cb5ceefd60b30d7b4352db186ff10a2bdf336.txt`

**Expected Classification:** Trustworthy
**Expected Behavior:** Abundant evidence, pipeline should confirm quickly without needing query refinement.

### Scenario 3: Belo Horizonte Guarda Municipal Concurso (NUANCED)

**Claim:** "A Prefeitura de Belo Horizonte realizou concurso publico para a Guarda Municipal em 2024."

**Context:** city=Belo Horizonte, published_since=2024-01-01, published_until=2024-12-31

**API Evidence:**
- Query `"concurso publico" + "guarda municipal"` with territory_id `3106200` returns **77 gazettes**
- But: the concurso edital was from 2022 (Edital 02/2022). In 2024, gazettes show convocations of approved candidates, salary progressions, and one revocation
- No NEW concurso was opened in 2024

**Expected Classification:** Trustworthy but / Arguable
**Expected Behavior:** Pipeline must distinguish between "realizou concurso" (conducted a new exam) vs. "convocou aprovados" (called candidates from existing exam). This tests reasoning depth.

### Scenario 4: Curitiba Nurse Hiring (TRUE)

**Claim:** "A Prefeitura de Curitiba nomeou enfermeiros aprovados em concurso publico em 2024."

**Context:** city=Curitiba, published_since=2024-01-01, published_until=2024-12-31

**API Evidence:**
- Query `"concurso publico" + "enfermeiro"` with territory_id `4106902` returns **129 gazettes**
- Results show specific nurse names, registration numbers (matricula 113.079), salary pattern codes (padrao 4130)
- txt_url: `https://data.queridodiario.ok.org.br/4106902/2024-04-18/f236e7a372015829a44b72044627b5880131ec3f.txt`

**Expected Classification:** Trustworthy
**Expected Behavior:** Clear evidence with specific employee data. Straightforward verification.

### Scenario 5: Absurd Claim — No Results (NEGATIVE)

**Claim:** "Houve invasao alienigena registrada no diario oficial de Sao Paulo em 2024."

**Context:** city=Sao Paulo, published_since=2024-01-01, published_until=2024-12-31

**API Evidence:**
- Query returns `total_gazettes: 0`, empty results array

**Expected Classification:** Unverifiable
**Expected Behavior:** Pipeline should exhaust search strategies (up to 3 iterations), then classify as Unverifiable with clear explanation that no relevant gazette data was found.

---

## Test Implementation

### Test Structure

```
tests/
  test_phase1_foundation.py          # Existing
  test_gazette_accuracy.py            # NEW: Golden test cases
  fixtures/
    gazette_api_responses/
      porto_alegre_flood.json         # Cached API response
      rio_medicamentos.json
      bh_guarda_municipal.json
      curitiba_enfermeiros.json
      empty_result.json
    gazette_texts/
      porto_alegre_flood.txt          # Cached gazette full text
      rio_medicamentos.txt
      bh_guarda_municipal.txt
      curitiba_enfermeiros.txt
```

### Test Categories

**1. Tool-level tests (no LLM needed):**
- `test_querido_diario_search_multi_query`: Multiple queries return merged, deduplicated results
- `test_querido_diario_search_pagination`: `size` and `excerpt` params are passed correctly
- `test_querido_diario_search_empty`: Empty results return proper error structure
- `test_gazette_deep_search_multi_doc`: FAISS index built across multiple documents
- `test_city_filter_graceful_degradation`: Unknown city searches without filter

**2. Integration tests (requires OPENAI_API_KEY):**
- `test_plan_search_generates_diverse_strategies`: Query planner produces 2-3 distinct queries
- `test_evaluate_evidence_sufficient`: Evaluator correctly identifies sufficient evidence
- `test_evaluate_evidence_insufficient`: Evaluator triggers refinement on poor results
- `test_cross_check_classification`: Cross-checker produces correct classification category

**3. End-to-end accuracy tests (requires OPENAI_API_KEY + API access):**
- `test_golden_porto_alegre_flood`: Full pipeline → Trustworthy
- `test_golden_rio_medicamentos`: Full pipeline → Trustworthy
- `test_golden_bh_guarda_municipal`: Full pipeline → Trustworthy but / Arguable
- `test_golden_curitiba_enfermeiros`: Full pipeline → Trustworthy
- `test_golden_negative_no_results`: Full pipeline → Unverifiable

**Test fixtures:** Cached API responses and gazette texts for offline tool-level tests. Integration and E2E tests hit real APIs (marked with `@pytest.mark.integration` and `@pytest.mark.e2e`).

---

## Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `app/nodes/gazette/search_planner.py` | Query strategy planner node |
| `app/nodes/gazette/fetch_and_score.py` | Multi-query fetch + excerpt scoring (batch scoring) |
| `app/nodes/gazette/evidence_evaluator.py` | Evidence sufficiency evaluator (loop control) |
| `app/nodes/gazette/query_refiner.py` | Query refinement for retry loop |
| `app/nodes/gazette/deep_analyzer.py` | Multi-document map-reduce FAISS analysis |
| `app/tools/querido_diario_search.py` | New multi-query search tool |
| `app/tools/gazette_deep_search.py` | New multi-document search tool (with embedding retry) |
| `app/utils/__init__.py` | Utils package |
| `app/utils/llm_retry.py` | invoke_with_retry safety-net utility |
| `tests/test_gazette_accuracy.py` | Golden test cases |
| `tests/fixtures/gazette_api_responses/*.json` | Cached API fixtures (5 files) |
| `tests/fixtures/gazette_texts/*.txt` | Cached gazette texts (4 files) |

### Modified Files
| File | Changes |
|------|---------|
| `app/nodes/gazette/subgraph.py` | Rewrite with loop architecture |
| `app/nodes/gazette/cross_checker.py` | Upgrade model, accept richer evidence |
| `app/nodes/gazette/gazette_reporter.py` | Include gazette URLs and search strategies in report |
| `app/state.py` | Add new state fields |
| `requirements.txt` | Add pytest if missing |

### Deprecated (to remove)
| File | Reason |
|------|--------|
| `app/tools/querido_diario.py` | Replaced by `querido_diario_search.py` |
| `app/tools/gazette_search.py` | Replaced by `gazette_deep_search.py` |
| `app/nodes/gazette/subject_creator.py` | Replaced by `search_planner.py` |
| `app/nodes/gazette/gazette_fetcher.py` | Replaced by `fetch_and_score.py` |
| `app/nodes/gazette/gazette_analyzer.py` | Replaced by `deep_analyzer.py` |

---

## API Parameter Reference

Parameters now used by the new tool:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `querystring` | from planner | Search query (Elasticsearch simple syntax) |
| `territory_ids` | from IBGE codes | City filter |
| `published_since` | from context | Date range start |
| `published_until` | from context | Date range end |
| `size` | 30 | Results per query (was default 10) |
| `sort_by` | relevance | Relevance ranking |
| `excerpt_size` | 500 | Characters per excerpt (was default/unused) |
| `number_of_excerpts` | 3 | Excerpts per gazette (was default 1/unused) |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Higher token cost (gpt-5.2 for 4 reasoning stages) | Loop max 3 iterations; gpt-5-mini for extraction tasks |
| Longer latency | Early break when evidence is sufficient (score ≥7 skips LLM evaluation) |
| API rate limits | Sequential queries with small delay; respect API behavior |
| Token limit exceeded on large gazettes | Map-reduce batching (primary), invoke_with_retry (safety net), MAX_CHARS_PER_GAZETTE=80K |
| FAISS embedding failure | Embedding retry: halves chunks on token error (up to 3 attempts) |
| FAISS memory with multiple docs | Limit to top 3 gazettes; chunk size keeps memory bounded |
| Query planner generates bad queries | Fallback: if all 3 iterations fail, report as Unverifiable |
| Temperature parameter rejection | Both gpt-5.2 and gpt-5-mini require temperature=1 (only default supported) |

---

## Verification Plan

1. **Unit tests pass:** `python -m pytest tests/test_phase1_foundation.py -v`
2. **Tool tests pass:** `python -m pytest tests/test_gazette_accuracy.py -v -k "not integration and not e2e"`
3. **Integration tests pass:** `OPENAI_API_KEY=xxx python -m pytest tests/test_gazette_accuracy.py -v -m integration`
4. **E2E golden tests pass:** `OPENAI_API_KEY=xxx python -m pytest tests/test_gazette_accuracy.py -v -m e2e`
5. **Manual smoke test:** `curl -X POST http://localhost:8080/invoke -H "Content-Type: application/json" -d '{"input":{"claim":"A Prefeitura de Porto Alegre firmou contratos emergenciais...","search_type":"gazettes","language":"pt","context":{"city":"Porto Alegre","published_since":"2024-05-01","published_until":"2024-07-31"}}}'`
