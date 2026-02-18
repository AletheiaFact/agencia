# Phase 3: Plugin Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 4 new data source plugins (TSE, BACEN, ClaimBuster, Wikipedia/Wikidata) following the established `DataSourcePlugin` pattern.

**Architecture:** Each plugin implements the `DataSourcePlugin` ABC, registers in `plugins/__init__.py`, and integrates automatically via the existing source selector. No graph or state changes. TSE and Wikipedia use multi-endpoint routing (same as Portal da Transparencia). BACEN uses keyword detection (same as IBGE SIDRA). ClaimBuster is a simple REST wrapper (same as Google Fact Check).

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, `ckanapi`, `python-bcb`, `wikipedia-api`, `requests`

**Working directory:** `/Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke`

**Branch:** `claude/competent-clarke`

**SECURITY:** NEVER read `.env` files.

---

## Task 1: Add ELECTORAL Category to PluginCategory Enum

**Files:**
- Modify: `app/plugins/base.py`
- Test: `tests/test_plugins/test_plugin_base.py`

**Step 1: Write the failing test**

Add to `tests/test_plugins/test_plugin_base.py`:

```python
def test_electoral_category_exists():
    """ELECTORAL category should exist for TSE plugin."""
    from plugins.base import PluginCategory
    assert PluginCategory.ELECTORAL == "electoral"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_plugin_base.py::test_electoral_category_exists -v`

Expected: FAIL with `AttributeError: ELECTORAL`

**Step 3: Write minimal implementation**

In `app/plugins/base.py`, add to the `PluginCategory` enum after `MULTIMEDIA`:

```python
    ELECTORAL = "electoral"
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_plugin_base.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add app/plugins/base.py tests/test_plugins/test_plugin_base.py
git commit -m "feat: add ELECTORAL category to PluginCategory enum"
```

---

## Task 2: Create Electoral Plugin Directory Structure

**Files:**
- Create: `app/plugins/electoral/__init__.py`

**Step 1: Create the directory and init file**

```python
# app/plugins/electoral/__init__.py
```

(Empty file — just the package init.)

**Step 2: Verify import works**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -c "import plugins.electoral; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add app/plugins/electoral/__init__.py
git commit -m "chore: create electoral plugin directory"
```

---

## Task 3: TSE Plugin — Tests

**Files:**
- Create: `tests/test_plugins/test_tse.py`

**Step 1: Write all TSE tests**

Create `tests/test_plugins/test_tse.py`:

```python
"""Tests for the TSE (Superior Electoral Court) plugin."""

from unittest.mock import MagicMock, patch

import pytest

from plugins.electoral.tse import TSEPlugin


@pytest.fixture
def plugin():
    return TSEPlugin()


# --- Metadata ---

class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "tse"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.ELECTORAL

    def test_no_api_key_required(self, plugin):
        assert plugin.get_metadata().required_env_vars == []

    def test_reliability_score(self, plugin):
        assert plugin.get_metadata().reliability_score == 0.95


# --- Availability ---

class TestAvailability:
    def test_available_when_ckanapi_installed(self, plugin):
        assert plugin.is_available() is True

    def test_unavailable_when_ckanapi_missing(self, plugin):
        import sys
        with patch.dict(sys.modules, {"ckanapi": None}):
            assert plugin.is_available() is False


# --- CKAN dataset search ---

class TestSearchDatasets:
    @patch("plugins.electoral.tse.ckanapi.RemoteCKAN")
    def test_successful_dataset_search(self, mock_ckan_cls, plugin):
        mock_ckan = MagicMock()
        mock_ckan.action.package_search.return_value = {
            "results": [
                {
                    "id": "ds-001",
                    "title": "Candidatos 2024",
                    "notes": "Lista de candidatos eleicoes 2024",
                    "metadata_modified": "2024-08-15",
                }
            ]
        }
        mock_ckan_cls.return_value = mock_ckan

        result = plugin.search("candidatos 2024", endpoint="datasets")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["title"] == "Candidatos 2024"
        assert result.metadata["endpoint"] == "datasets"

    @patch("plugins.electoral.tse.ckanapi.RemoteCKAN")
    def test_empty_dataset_search(self, mock_ckan_cls, plugin):
        mock_ckan = MagicMock()
        mock_ckan.action.package_search.return_value = {"results": []}
        mock_ckan_cls.return_value = mock_ckan

        result = plugin.search("nonexistent xyz", endpoint="datasets")
        assert result.error is None
        assert result.result_count == 0

    @patch("plugins.electoral.tse.ckanapi.RemoteCKAN")
    def test_ckan_api_error(self, mock_ckan_cls, plugin):
        mock_ckan = MagicMock()
        mock_ckan.action.package_search.side_effect = Exception("CKAN timeout")
        mock_ckan_cls.return_value = mock_ckan

        result = plugin.search("test", endpoint="datasets")
        assert result.error is not None
        assert "CKAN" in result.error


# --- Candidate search ---

class TestSearchCandidates:
    @patch("plugins.electoral.tse.requests.get")
    def test_successful_candidate_search(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidatos": [
                {
                    "id": 12345,
                    "nomeUrna": "CANDIDATO TESTE",
                    "numero": 13,
                    "partido": {"sigla": "PT"},
                    "cargo": {"nome": "Prefeito"},
                }
            ]
        }
        mock_get.return_value = mock_response

        result = plugin.search("candidato teste", endpoint="candidates", election_year="2024", state="SP")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["name"] == "CANDIDATO TESTE"
        assert result.results[0]["party"] == "PT"
        assert result.metadata["endpoint"] == "candidates"

    @patch("plugins.electoral.tse.requests.get")
    def test_candidate_http_error(self, mock_get, plugin):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("timeout")

        result = plugin.search("test", endpoint="candidates")
        assert result.error is not None
        assert "HTTP request failed" in result.error

    @patch("plugins.electoral.tse.requests.get")
    def test_candidate_non_200(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response

        result = plugin.search("test", endpoint="candidates")
        assert result.error is not None


# --- Results search ---

class TestSearchResults:
    @patch("plugins.electoral.tse.requests.get")
    def test_successful_results_search(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "abrangencia": [
                {
                    "candidatos": [
                        {
                            "nomeUrna": "CANDIDATO A",
                            "numero": 13,
                            "partido": {"sigla": "PT"},
                            "totVotos": "50000",
                            "resultado": "Eleito",
                        }
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response

        result = plugin.search("", endpoint="results", election_year="2022", state="SP", office="governador")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["name"] == "CANDIDATO A"
        assert result.results[0]["votes"] == "50000"
        assert result.metadata["endpoint"] == "results"

    @patch("plugins.electoral.tse.requests.get")
    def test_results_404_returns_empty(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        result = plugin.search("", endpoint="results", election_year="1800")
        assert result.error is None
        assert result.result_count == 0


# --- Default endpoint ---

class TestDefaultEndpoint:
    @patch("plugins.electoral.tse.ckanapi.RemoteCKAN")
    def test_default_is_datasets(self, mock_ckan_cls, plugin):
        mock_ckan = MagicMock()
        mock_ckan.action.package_search.return_value = {"results": []}
        mock_ckan_cls.return_value = mock_ckan

        result = plugin.search("test")
        assert result.metadata["endpoint"] == "datasets"

    def test_invalid_endpoint(self, plugin):
        result = plugin.search("test", endpoint="invalid_xyz")
        assert result.error is not None
        assert "Unknown endpoint" in result.error
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_tse.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.electoral.tse'`

**Step 3: Commit test file**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add tests/test_plugins/test_tse.py
git commit -m "test: add TSE plugin test suite (red phase)"
```

---

## Task 4: TSE Plugin — Implementation

**Files:**
- Create: `app/plugins/electoral/tse.py`

**Step 1: Implement the TSE plugin**

Create `app/plugins/electoral/tse.py`:

```python
"""TSE Open Data plugin — Brazilian electoral data.

Queries TSE (Superior Electoral Court) open data APIs for electoral
information: CKAN datasets, candidate registrations (DivulgaCandContas),
and election results. No API key required.
"""

import logging

import requests

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult

logger = logging.getLogger(__name__)

CKAN_BASE = "https://dadosabertos.tse.jus.br/api/3"
CANDIDATES_BASE = "https://divulgacandcontas.tse.jus.br/divulga/rest/v1"
RESULTS_BASE = "https://resultados.tse.jus.br/oficial"

VALID_ENDPOINTS = {"datasets", "candidates", "results"}


class TSEPlugin(DataSourcePlugin):

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="tse",
            display_name="TSE Open Data",
            description="Search Brazilian electoral data from TSE (Superior Electoral Court). "
            "Covers election results, candidate registrations, campaign finance, "
            "and voter statistics. Use endpoint='datasets' for general search, "
            "'candidates' for specific candidate lookups, 'results' for vote tallies.",
            category=PluginCategory.ELECTORAL,
            required_env_vars=[],
            reliability_score=0.95,
        )

    def is_available(self) -> bool:
        try:
            import ckanapi  # noqa: F401
            return True
        except ImportError:
            return False

    def search(
        self,
        query: str,
        endpoint: str = "datasets",
        election_year: str = "2024",
        state: str = "",
        office: str = "",
        **kwargs,
    ) -> PluginResult:
        """Search TSE open data.

        Args:
            query: Search term.
            endpoint: "datasets" (CKAN), "candidates" (DivulgaCandContas),
                      or "results" (election results).
            election_year: Election year (default: "2024").
            state: UF code (e.g. "SP", "RJ") — empty = national.
            office: Office name (e.g. "presidente", "governador", "prefeito").
        """
        if endpoint not in VALID_ENDPOINTS:
            return PluginResult(
                source="tse",
                query=query,
                error=f"Unknown endpoint '{endpoint}'. Valid: {', '.join(sorted(VALID_ENDPOINTS))}",
            )

        logger.info(
            "[tse] Searching — query='%s' endpoint=%s year=%s",
            query[:80],
            endpoint,
            election_year,
        )

        if endpoint == "datasets":
            return self._search_ckan(query)
        elif endpoint == "candidates":
            return self._search_candidates(query, election_year, state)
        else:
            return self._search_results(election_year, state, office)

    def _search_ckan(self, query: str) -> PluginResult:
        """Search CKAN datasets on dadosabertos.tse.jus.br."""
        try:
            import ckanapi

            ckan = ckanapi.RemoteCKAN(CKAN_BASE.rsplit("/api/3", 1)[0])
            data = ckan.action.package_search(q=query, rows=10)
        except ImportError:
            return PluginResult(
                source="tse", query=query,
                error="ckanapi package not installed",
            )
        except Exception as e:
            logger.error("[tse] CKAN error: %s", e)
            return PluginResult(
                source="tse", query=query,
                error=f"CKAN query failed: {e}",
            )

        results = []
        for pkg in data.get("results", []):
            results.append({
                "id": pkg.get("id", ""),
                "title": pkg.get("title", ""),
                "description": pkg.get("notes", ""),
                "modified": pkg.get("metadata_modified", ""),
                "raw": pkg,
            })

        logger.info("[tse] CKAN found %d datasets", len(results))
        return PluginResult(
            source="tse", query=query,
            results=results, result_count=len(results),
            metadata={"endpoint": "datasets"},
        )

    def _search_candidates(
        self, query: str, election_year: str, state: str
    ) -> PluginResult:
        """Search DivulgaCandContas for candidate registrations."""
        url = f"{CANDIDATES_BASE}/candidatura/listar/{election_year}"
        params = {"nomeUrnaCandidato": query}
        if state:
            params["sgUe"] = state

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error("[tse] DivulgaCandContas error: %s", e)
            return PluginResult(
                source="tse", query=query,
                error=f"HTTP request failed: {e}",
            )
        except Exception as e:
            logger.error("[tse] DivulgaCandContas error: %s", e)
            return PluginResult(
                source="tse", query=query,
                error=f"Candidate query failed: {e}",
            )

        candidates = data.get("candidatos", [])
        results = []
        for c in candidates:
            results.append({
                "id": c.get("id", ""),
                "name": c.get("nomeUrna", ""),
                "number": c.get("numero", ""),
                "party": c.get("partido", {}).get("sigla", "") if isinstance(c.get("partido"), dict) else "",
                "office": c.get("cargo", {}).get("nome", "") if isinstance(c.get("cargo"), dict) else "",
                "raw": c,
            })

        logger.info("[tse] Found %d candidates", len(results))
        return PluginResult(
            source="tse", query=query,
            results=results, result_count=len(results),
            metadata={"endpoint": "candidates"},
        )

    def _search_results(
        self, election_year: str, state: str, office: str
    ) -> PluginResult:
        """Search TSE election results."""
        # Build the results API URL — simplified path
        url = f"{RESULTS_BASE}/{election_year}/dados-simplificados"
        params = {}
        if state:
            params["sg_ue"] = state
        if office:
            params["cargo"] = office

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 404:
                logger.info("[tse] No results for year=%s", election_year)
                return PluginResult(
                    source="tse", query="",
                    results=[], result_count=0,
                    metadata={"endpoint": "results"},
                )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error("[tse] Results error: %s", e)
            return PluginResult(
                source="tse", query="",
                error=f"HTTP request failed: {e}",
            )
        except Exception as e:
            logger.error("[tse] Results error: %s", e)
            return PluginResult(
                source="tse", query="",
                error=f"Results query failed: {e}",
            )

        results = []
        for scope in data.get("abrangencia", []):
            for c in scope.get("candidatos", []):
                results.append({
                    "name": c.get("nomeUrna", ""),
                    "number": c.get("numero", ""),
                    "party": c.get("partido", {}).get("sigla", "") if isinstance(c.get("partido"), dict) else "",
                    "votes": c.get("totVotos", ""),
                    "result": c.get("resultado", ""),
                    "raw": c,
                })

        logger.info("[tse] Found %d result entries", len(results))
        return PluginResult(
            source="tse", query="",
            results=results, result_count=len(results),
            metadata={"endpoint": "results"},
        )
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_tse.py -v`

Expected: ALL PASS

**Step 3: Run full test suite to verify no regressions**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/ -v`

Expected: ALL PASS

**Step 4: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add app/plugins/electoral/tse.py
git commit -m "feat: add TSE electoral data plugin (CKAN + candidates + results)"
```

---

## Task 5: BACEN Plugin — Tests

**Files:**
- Create: `tests/test_plugins/test_bacen.py`

**Step 1: Write all BACEN tests**

Create `tests/test_plugins/test_bacen.py`:

```python
"""Tests for the BACEN (Central Bank) plugin."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from plugins.government_data.bacen import BACENPlugin, COMMON_SERIES


@pytest.fixture
def plugin():
    return BACENPlugin()


# --- Metadata ---

class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "bacen"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.GOVERNMENT_DATA

    def test_no_api_key_required(self, plugin):
        assert plugin.get_metadata().required_env_vars == []

    def test_reliability_score(self, plugin):
        assert plugin.get_metadata().reliability_score == 0.95


# --- Availability ---

class TestAvailability:
    def test_available_when_bcb_installed(self, plugin):
        # python-bcb should be installed in test env
        assert plugin.is_available() is True

    def test_unavailable_when_bcb_missing(self, plugin):
        with patch.dict(sys.modules, {"bcb": None}):
            assert plugin.is_available() is False


# --- Series detection ---

class TestSeriesDetection:
    def test_detects_selic(self):
        assert BACENPlugin._detect_series("taxa selic atual") == 432

    def test_detects_ipca(self):
        assert BACENPlugin._detect_series("inflação IPCA mensal") == 433

    def test_detects_igpm(self):
        assert BACENPlugin._detect_series("IGP-M acumulado") == 189

    def test_detects_dollar(self):
        assert BACENPlugin._detect_series("cotação do dólar") == 1

    def test_detects_euro(self):
        assert BACENPlugin._detect_series("câmbio euro") == 21619

    def test_detects_pib(self):
        assert BACENPlugin._detect_series("PIB mensal") == 4380

    def test_detects_divida(self):
        assert BACENPlugin._detect_series("dívida pública líquida") == 4513

    def test_detects_desemprego(self):
        assert BACENPlugin._detect_series("taxa de desemprego") == 24369

    def test_returns_zero_for_unknown(self):
        assert BACENPlugin._detect_series("random unrelated topic xyz") == 0


# --- Search ---

class TestSearch:
    def test_returns_error_when_series_not_detected(self, plugin):
        result = plugin.search("completely unrelated topic xyz123")
        assert result.error is not None
        assert "Could not determine" in result.error

    @patch("plugins.government_data.bacen.sgs")
    def test_successful_search(self, mock_sgs, plugin):
        import pandas as pd

        mock_df = pd.DataFrame(
            {"432": [13.75, 13.25, 12.75]},
            index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        )
        mock_sgs.get.return_value = mock_df

        result = plugin.search("taxa selic")
        assert result.error is None
        assert result.result_count == 3
        assert result.metadata["series_code"] == 432

    @patch("plugins.government_data.bacen.sgs")
    def test_explicit_series_code(self, mock_sgs, plugin):
        import pandas as pd

        mock_df = pd.DataFrame(
            {"9999": [1.5]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        mock_sgs.get.return_value = mock_df

        result = plugin.search("custom query", series_code=9999)
        assert result.error is None
        assert result.metadata["series_code"] == 9999
        mock_sgs.get.assert_called_once()

    @patch("plugins.government_data.bacen.sgs")
    def test_api_error_handled(self, mock_sgs, plugin):
        mock_sgs.get.side_effect = Exception("BCB API timeout")

        result = plugin.search("taxa selic")
        assert result.error is not None
        assert "BCB query failed" in result.error

    def test_missing_bcb_handled(self, plugin):
        with patch.dict(sys.modules, {"bcb": None, "bcb.sgs": None}):
            result = plugin.search("selic", series_code=432)
            assert result.error is not None
            assert "not installed" in result.error

    @patch("plugins.government_data.bacen.sgs")
    def test_date_range_params(self, mock_sgs, plugin):
        import pandas as pd

        mock_df = pd.DataFrame(
            {"432": [13.75]},
            index=pd.to_datetime(["2024-06-01"]),
        )
        mock_sgs.get.return_value = mock_df

        plugin.search("selic", series_code=432, start_date="2024-06-01", end_date="2024-06-30")
        call_kwargs = mock_sgs.get.call_args[1]
        assert call_kwargs["start"] == "2024-06-01"
        assert call_kwargs["end"] == "2024-06-30"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_bacen.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.government_data.bacen'`

**Step 3: Commit test file**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add tests/test_plugins/test_bacen.py
git commit -m "test: add BACEN plugin test suite (red phase)"
```

---

## Task 6: BACEN Plugin — Implementation

**Files:**
- Create: `app/plugins/government_data/bacen.py`

**Step 1: Implement the BACEN plugin**

Create `app/plugins/government_data/bacen.py`:

```python
"""BACEN plugin — Brazilian Central Bank economic data.

Queries the BCB SGS (Time Series Management System) for official economic
indicators: SELIC rate, inflation (IPCA, IGP-M), exchange rates, GDP,
public debt, and unemployment. No API key required.

Uses the python-bcb library for convenient access.
"""

import logging
from datetime import datetime, timedelta

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult

logger = logging.getLogger(__name__)

# Common BCB SGS series codes
COMMON_SERIES = {
    "selic": 432,
    "ipca": 433,
    "igpm": 189,
    "igp-m": 189,
    "cambio_dolar": 1,
    "dolar": 1,
    "cambio_euro": 21619,
    "euro": 21619,
    "pib_mensal": 4380,
    "pib": 4380,
    "divida_publica": 4513,
    "divida": 4513,
    "desemprego": 24369,
}


class BACENPlugin(DataSourcePlugin):

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="bacen",
            display_name="BACEN (Central Bank)",
            description="Query Brazilian Central Bank (BACEN) economic data. "
            "Covers SELIC interest rate, inflation (IPCA, IGP-M), "
            "exchange rates (USD, EUR), GDP, public debt, and unemployment. "
            "Use for verifying claims about Brazilian economic indicators.",
            category=PluginCategory.GOVERNMENT_DATA,
            required_env_vars=[],
            reliability_score=0.95,
        )

    def is_available(self) -> bool:
        try:
            import bcb  # noqa: F401
            return True
        except ImportError:
            return False

    def search(
        self,
        query: str,
        series_code: int = 0,
        start_date: str = "",
        end_date: str = "",
        **kwargs,
    ) -> PluginResult:
        """Query BACEN SGS for economic time series data.

        Args:
            query: Natural language description (used to auto-detect series).
            series_code: SGS series code. If 0, tries to detect from query.
            start_date: Start date "YYYY-MM-DD" — empty = last 12 months.
            end_date: End date "YYYY-MM-DD" — empty = today.
        """
        if not series_code:
            series_code = self._detect_series(query)

        if not series_code:
            return PluginResult(
                source="bacen",
                query=query,
                error=f"Could not determine BCB series for query: '{query}'. "
                f"Available topics: {', '.join(sorted(set(COMMON_SERIES.keys())))}"
            )

        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            "[bacen] Querying series=%d period=%s to %s",
            series_code, start_date, end_date,
        )

        try:
            from bcb import sgs

            data = sgs.get(
                codes={str(series_code): series_code},
                start=start_date,
                end=end_date,
            )
        except ImportError:
            return PluginResult(
                source="bacen",
                query=query,
                error="python-bcb package not installed",
            )
        except Exception as e:
            logger.error("[bacen] Error querying series %d: %s", series_code, e)
            return PluginResult(
                source="bacen",
                query=query,
                error=f"BCB query failed: {e}",
            )

        results = []
        try:
            col = str(series_code)
            for date_idx, row in data.iterrows():
                value = row.get(col, row.iloc[0] if len(row) > 0 else None)
                results.append({
                    "date": str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx),
                    "value": float(value) if value is not None else None,
                    "series": series_code,
                })
        except Exception as e:
            logger.warning("[bacen] Error parsing results: %s", e)
            results = [{"raw": data.to_dict()}] if hasattr(data, "to_dict") else []

        logger.info("[bacen] Retrieved %d records for series %d", len(results), series_code)
        return PluginResult(
            source="bacen",
            query=query,
            results=results,
            result_count=len(results),
            metadata={"series_code": series_code},
        )

    @staticmethod
    def _detect_series(query: str) -> int:
        """Try to detect the appropriate SGS series from a natural language query."""
        query_lower = query.lower()
        for keyword, code in COMMON_SERIES.items():
            if keyword in query_lower:
                return code

        # Additional keyword mappings
        keyword_map = {
            "juro": 432,
            "taxa básica": 432,
            "inflac": 433,
            "preco": 433,
            "câmbio": 1,
            "moeda": 1,
            "produto interno": 4380,
            "gdp": 4380,
            "dívid": 4513,
            "empreg": 24369,
            "trabalh": 24369,
        }
        for kw, code in keyword_map.items():
            if kw in query_lower:
                return code

        return 0
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_bacen.py -v`

Expected: ALL PASS

**Step 3: Run full test suite**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/ -v`

Expected: ALL PASS

**Step 4: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add app/plugins/government_data/bacen.py
git commit -m "feat: add BACEN Central Bank economic data plugin"
```

---

## Task 7: ClaimBuster Plugin — Tests

**Files:**
- Create: `tests/test_plugins/test_claimbuster.py`

**Step 1: Write all ClaimBuster tests**

Create `tests/test_plugins/test_claimbuster.py`:

```python
"""Tests for the ClaimBuster plugin."""

from unittest.mock import MagicMock, patch

import pytest

from plugins.claim_databases.claimbuster import ClaimBusterPlugin


@pytest.fixture
def plugin():
    return ClaimBusterPlugin()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure env var is cleared before each test."""
    monkeypatch.delenv("CLAIMBUSTER_API_KEY", raising=False)


# --- Metadata ---

class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "claimbuster"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.CLAIM_DATABASE

    def test_reliability_score(self, plugin):
        assert plugin.get_metadata().reliability_score == 0.7

    def test_requires_api_key(self, plugin):
        assert plugin.get_metadata().required_env_vars == ["CLAIMBUSTER_API_KEY"]


# --- Availability ---

class TestAvailability:
    def test_available_when_key_set(self, plugin, monkeypatch):
        monkeypatch.setenv("CLAIMBUSTER_API_KEY", "test-key")
        assert plugin.is_available() is True

    def test_unavailable_when_key_missing(self, plugin):
        assert plugin.is_available() is False


# --- Search ---

class TestSearch:
    def test_returns_error_when_no_key(self, plugin):
        result = plugin.search("test claim")
        assert result.error is not None
        assert "API key" in result.error

    @patch("plugins.claim_databases.claimbuster.requests.post")
    def test_successful_search(self, mock_post, plugin, monkeypatch):
        monkeypatch.setenv("CLAIMBUSTER_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"text": "The economy grew 5%.", "score": 0.82},
                {"text": "It was a sunny day.", "score": 0.15},
            ]
        }
        mock_post.return_value = mock_response

        result = plugin.search("The economy grew 5%. It was a sunny day.")
        assert result.error is None
        assert result.result_count == 2
        assert result.results[0]["sentence"] == "The economy grew 5%."
        assert result.results[0]["score"] == 0.82
        assert result.results[0]["check_worthy"] is True
        assert result.results[1]["check_worthy"] is False

    @patch("plugins.claim_databases.claimbuster.requests.post")
    def test_passes_api_key_header(self, mock_post, plugin, monkeypatch):
        monkeypatch.setenv("CLAIMBUSTER_API_KEY", "my-key-123")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response

        plugin.search("test")
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["x-api-key"] == "my-key-123"

    @patch("plugins.claim_databases.claimbuster.requests.post")
    def test_empty_response(self, mock_post, plugin, monkeypatch):
        monkeypatch.setenv("CLAIMBUSTER_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response

        result = plugin.search("test")
        assert result.error is None
        assert result.result_count == 0

    @patch("plugins.claim_databases.claimbuster.requests.post")
    def test_http_error_handled(self, mock_post, plugin, monkeypatch):
        import requests as req
        monkeypatch.setenv("CLAIMBUSTER_API_KEY", "test-key")
        mock_post.side_effect = req.exceptions.ConnectionError("timeout")

        result = plugin.search("test")
        assert result.error is not None
        assert "HTTP request failed" in result.error

    @patch("plugins.claim_databases.claimbuster.requests.post")
    def test_check_worthy_threshold(self, mock_post, plugin, monkeypatch):
        """Score >= 0.5 is check-worthy, < 0.5 is not."""
        monkeypatch.setenv("CLAIMBUSTER_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"text": "Borderline.", "score": 0.5},
                {"text": "Not worthy.", "score": 0.49},
            ]
        }
        mock_post.return_value = mock_response

        result = plugin.search("Borderline. Not worthy.")
        assert result.results[0]["check_worthy"] is True
        assert result.results[1]["check_worthy"] is False
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_claimbuster.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.claim_databases.claimbuster'`

**Step 3: Commit test file**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add tests/test_plugins/test_claimbuster.py
git commit -m "test: add ClaimBuster plugin test suite (red phase)"
```

---

## Task 8: ClaimBuster Plugin — Implementation

**Files:**
- Create: `app/plugins/claim_databases/claimbuster.py`

**Step 1: Implement the ClaimBuster plugin**

Create `app/plugins/claim_databases/claimbuster.py`:

```python
"""ClaimBuster plugin — Claim check-worthiness scoring.

Queries the ClaimBuster API to score how check-worthy a claim is.
Returns per-sentence scores from 0 to 1. Requires CLAIMBUSTER_API_KEY.
"""

import logging
import os

import requests

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult

logger = logging.getLogger(__name__)

API_URL = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
CHECK_WORTHY_THRESHOLD = 0.5


class ClaimBusterPlugin(DataSourcePlugin):

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="claimbuster",
            display_name="ClaimBuster",
            description="Score how check-worthy a claim is using ClaimBuster AI. "
            "Returns a 0-1 score indicating how important it is to fact-check "
            "the statement. Useful for prioritizing which claims to investigate.",
            category=PluginCategory.CLAIM_DATABASE,
            required_env_vars=["CLAIMBUSTER_API_KEY"],
            reliability_score=0.7,
        )

    def is_available(self) -> bool:
        return bool(os.getenv("CLAIMBUSTER_API_KEY"))

    def search(self, query: str, **kwargs) -> PluginResult:
        """Score claim text for check-worthiness.

        Args:
            query: Claim text to score. Can contain multiple sentences.
        """
        api_key = os.getenv("CLAIMBUSTER_API_KEY")
        if not api_key:
            return PluginResult(
                source="claimbuster",
                query=query,
                error="No API key configured. Set CLAIMBUSTER_API_KEY.",
            )

        logger.info("[claimbuster] Scoring — text='%s'", query[:80])

        try:
            response = requests.post(
                API_URL,
                json={"input_text": query},
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error("[claimbuster] HTTP error: %s", e)
            return PluginResult(
                source="claimbuster",
                query=query,
                error=f"HTTP request failed: {e}",
            )

        sentences = data.get("results", [])
        results = []
        for s in sentences:
            score = s.get("score", 0.0)
            results.append({
                "sentence": s.get("text", ""),
                "score": score,
                "check_worthy": score >= CHECK_WORTHY_THRESHOLD,
            })

        logger.info("[claimbuster] Scored %d sentences", len(results))
        return PluginResult(
            source="claimbuster",
            query=query,
            results=results,
            result_count=len(results),
        )
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_claimbuster.py -v`

Expected: ALL PASS

**Step 3: Run full test suite**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/ -v`

Expected: ALL PASS

**Step 4: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add app/plugins/claim_databases/claimbuster.py
git commit -m "feat: add ClaimBuster claim-worthiness scoring plugin"
```

---

## Task 9: Create Knowledge Bases Plugin Directory

**Files:**
- Create: `app/plugins/knowledge_bases/__init__.py`

**Step 1: Create the directory and init file**

```python
# app/plugins/knowledge_bases/__init__.py
```

**Step 2: Verify import works**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -c "import plugins.knowledge_bases; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add app/plugins/knowledge_bases/__init__.py
git commit -m "chore: create knowledge_bases plugin directory"
```

---

## Task 10: Wikipedia/Wikidata Plugin — Tests

**Files:**
- Create: `tests/test_plugins/test_wikipedia.py`

**Step 1: Write all Wikipedia tests**

Create `tests/test_plugins/test_wikipedia.py`:

```python
"""Tests for the Wikipedia/Wikidata plugin."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from plugins.knowledge_bases.wikipedia import WikipediaPlugin


@pytest.fixture
def plugin():
    return WikipediaPlugin()


# --- Metadata ---

class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "wikipedia"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.KNOWLEDGE_BASE

    def test_no_api_key_required(self, plugin):
        assert plugin.get_metadata().required_env_vars == []

    def test_reliability_score(self, plugin):
        assert plugin.get_metadata().reliability_score == 0.6


# --- Availability ---

class TestAvailability:
    def test_available_when_wikipediaapi_installed(self, plugin):
        assert plugin.is_available() is True

    def test_unavailable_when_missing(self, plugin):
        with patch.dict(sys.modules, {"wikipediaapi": None}):
            assert plugin.is_available() is False


# --- Wikipedia search ---

class TestSearchWikipedia:
    @patch("plugins.knowledge_bases.wikipedia.wikipediaapi")
    def test_successful_search(self, mock_wikiapi, plugin):
        mock_wiki = MagicMock()
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.title = "Brasil"
        mock_page.summary = "Brasil é o maior país da América do Sul."
        mock_page.fullurl = "https://pt.wikipedia.org/wiki/Brasil"
        mock_wiki.page.return_value = mock_page

        mock_wikiapi.Wikipedia.return_value = mock_wiki

        result = plugin.search("Brasil", endpoint="wikipedia")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["title"] == "Brasil"
        assert "América do Sul" in result.results[0]["summary"]
        assert result.metadata["endpoint"] == "wikipedia"

    @patch("plugins.knowledge_bases.wikipedia.wikipediaapi")
    def test_page_not_found(self, mock_wikiapi, plugin):
        mock_wiki = MagicMock()
        mock_page = MagicMock()
        mock_page.exists.return_value = False
        mock_wiki.page.return_value = mock_page

        mock_wikiapi.Wikipedia.return_value = mock_wiki

        result = plugin.search("xyznonexistent123", endpoint="wikipedia")
        assert result.error is None
        assert result.result_count == 0

    @patch("plugins.knowledge_bases.wikipedia.wikipediaapi")
    def test_api_error(self, mock_wikiapi, plugin):
        mock_wikiapi.Wikipedia.side_effect = Exception("API error")

        result = plugin.search("test", endpoint="wikipedia")
        assert result.error is not None
        assert "Wikipedia query failed" in result.error

    @patch("plugins.knowledge_bases.wikipedia.wikipediaapi")
    def test_language_param(self, mock_wikiapi, plugin):
        mock_wiki = MagicMock()
        mock_page = MagicMock()
        mock_page.exists.return_value = False
        mock_wiki.page.return_value = mock_page
        mock_wikiapi.Wikipedia.return_value = mock_wiki

        plugin.search("test", endpoint="wikipedia", language="en")
        mock_wikiapi.Wikipedia.assert_called_once()
        call_kwargs = mock_wikiapi.Wikipedia.call_args
        assert call_kwargs[1].get("language") == "en" or call_kwargs[0][0] == "en"


# --- Wikidata search ---

class TestSearchWikidata:
    @patch("plugins.knowledge_bases.wikipedia.requests.get")
    def test_successful_wikidata_search(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "bindings": [
                    {
                        "item": {"value": "http://www.wikidata.org/entity/Q155"},
                        "itemLabel": {"value": "Brasil"},
                        "itemDescription": {"value": "country in South America"},
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        result = plugin.search("Brasil", endpoint="wikidata")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["entity"] == "Brasil"
        assert result.results[0]["description"] == "country in South America"
        assert result.metadata["endpoint"] == "wikidata"

    @patch("plugins.knowledge_bases.wikipedia.requests.get")
    def test_empty_wikidata_results(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": {"bindings": []}}
        mock_get.return_value = mock_response

        result = plugin.search("xyznonexistent123", endpoint="wikidata")
        assert result.error is None
        assert result.result_count == 0

    @patch("plugins.knowledge_bases.wikipedia.requests.get")
    def test_wikidata_http_error(self, mock_get, plugin):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("timeout")

        result = plugin.search("test", endpoint="wikidata")
        assert result.error is not None
        assert "HTTP request failed" in result.error


# --- Default endpoint ---

class TestDefaultEndpoint:
    @patch("plugins.knowledge_bases.wikipedia.wikipediaapi")
    def test_default_is_wikipedia(self, mock_wikiapi, plugin):
        mock_wiki = MagicMock()
        mock_page = MagicMock()
        mock_page.exists.return_value = False
        mock_wiki.page.return_value = mock_page
        mock_wikiapi.Wikipedia.return_value = mock_wiki

        result = plugin.search("test")
        assert result.metadata["endpoint"] == "wikipedia"

    def test_invalid_endpoint(self, plugin):
        result = plugin.search("test", endpoint="invalid_xyz")
        assert result.error is not None
        assert "Unknown endpoint" in result.error
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_wikipedia.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.knowledge_bases.wikipedia'`

**Step 3: Commit test file**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add tests/test_plugins/test_wikipedia.py
git commit -m "test: add Wikipedia/Wikidata plugin test suite (red phase)"
```

---

## Task 11: Wikipedia/Wikidata Plugin — Implementation

**Files:**
- Create: `app/plugins/knowledge_bases/wikipedia.py`

**Step 1: Implement the Wikipedia plugin**

Create `app/plugins/knowledge_bases/wikipedia.py`:

```python
"""Wikipedia/Wikidata plugin — encyclopedic facts and structured data.

Queries Wikipedia for article summaries and Wikidata for structured
entity data via SPARQL. No API key required.

Uses the wikipedia-api library for Wikipedia and requests for Wikidata.
"""

import logging

import requests

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
VALID_ENDPOINTS = {"wikipedia", "wikidata"}


class WikipediaPlugin(DataSourcePlugin):

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="wikipedia",
            display_name="Wikipedia / Wikidata",
            description="Search Wikipedia and Wikidata for encyclopedic facts. "
            "Use endpoint='wikipedia' for article summaries (Portuguese by default), "
            "'wikidata' for structured entity data (politicians, cities, organizations, "
            "dates). Useful for verifying general knowledge claims and disambiguating entities.",
            category=PluginCategory.KNOWLEDGE_BASE,
            required_env_vars=[],
            reliability_score=0.6,
        )

    def is_available(self) -> bool:
        try:
            import wikipediaapi  # noqa: F401
            return True
        except ImportError:
            return False

    def search(
        self,
        query: str,
        endpoint: str = "wikipedia",
        language: str = "pt",
        **kwargs,
    ) -> PluginResult:
        """Search Wikipedia or Wikidata.

        Args:
            query: Search term.
            endpoint: "wikipedia" for article summaries, "wikidata" for SPARQL.
            language: Wikipedia language code (default: "pt").
        """
        if endpoint not in VALID_ENDPOINTS:
            return PluginResult(
                source="wikipedia",
                query=query,
                error=f"Unknown endpoint '{endpoint}'. Valid: {', '.join(sorted(VALID_ENDPOINTS))}",
            )

        logger.info(
            "[wikipedia] Searching — query='%s' endpoint=%s lang=%s",
            query[:80], endpoint, language,
        )

        if endpoint == "wikipedia":
            return self._search_wikipedia(query, language)
        else:
            return self._search_wikidata(query)

    def _search_wikipedia(self, query: str, language: str) -> PluginResult:
        """Search Wikipedia for article summaries."""
        try:
            import wikipediaapi

            wiki = wikipediaapi.Wikipedia(
                user_agent="Agencia-FactChecker/1.0 (fact-checking tool)",
                language=language,
            )
            page = wiki.page(query)

            if not page.exists():
                return PluginResult(
                    source="wikipedia", query=query,
                    results=[], result_count=0,
                    metadata={"endpoint": "wikipedia"},
                )

            results = [{
                "title": page.title,
                "summary": page.summary[:1000],  # Limit summary length
                "url": page.fullurl,
            }]
        except ImportError:
            return PluginResult(
                source="wikipedia", query=query,
                error="wikipedia-api package not installed",
            )
        except Exception as e:
            logger.error("[wikipedia] Error: %s", e)
            return PluginResult(
                source="wikipedia", query=query,
                error=f"Wikipedia query failed: {e}",
            )

        logger.info("[wikipedia] Found page: %s", results[0]["title"])
        return PluginResult(
            source="wikipedia", query=query,
            results=results, result_count=len(results),
            metadata={"endpoint": "wikipedia"},
        )

    def _search_wikidata(self, query: str) -> PluginResult:
        """Search Wikidata via SPARQL for structured entity data."""
        # Escape single quotes in query to prevent SPARQL injection
        safe_query = query.replace("'", "\\'")
        sparql = f"""
        SELECT ?item ?itemLabel ?itemDescription WHERE {{
            ?item rdfs:label "{safe_query}"@pt .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
        }}
        LIMIT 5
        """

        try:
            response = requests.get(
                WIKIDATA_SPARQL,
                params={"query": sparql, "format": "json"},
                headers={"User-Agent": "Agencia-FactChecker/1.0"},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error("[wikipedia] Wikidata SPARQL error: %s", e)
            return PluginResult(
                source="wikipedia", query=query,
                error=f"HTTP request failed: {e}",
            )
        except Exception as e:
            logger.error("[wikipedia] Wikidata error: %s", e)
            return PluginResult(
                source="wikipedia", query=query,
                error=f"Wikidata query failed: {e}",
            )

        bindings = data.get("results", {}).get("bindings", [])
        results = []
        for b in bindings:
            results.append({
                "entity": b.get("itemLabel", {}).get("value", ""),
                "description": b.get("itemDescription", {}).get("value", ""),
                "wikidata_id": b.get("item", {}).get("value", "").split("/")[-1],
                "url": b.get("item", {}).get("value", ""),
            })

        logger.info("[wikipedia] Wikidata found %d entities", len(results))
        return PluginResult(
            source="wikipedia", query=query,
            results=results, result_count=len(results),
            metadata={"endpoint": "wikidata"},
        )
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_plugins/test_wikipedia.py -v`

Expected: ALL PASS

**Step 3: Run full test suite**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/ -v`

Expected: ALL PASS

**Step 4: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add app/plugins/knowledge_bases/wikipedia.py
git commit -m "feat: add Wikipedia/Wikidata encyclopedic data plugin"
```

---

## Task 12: Register All New Plugins

**Files:**
- Modify: `app/plugins/__init__.py`

**Step 1: Update registration**

In `app/plugins/__init__.py`, add the 4 new plugin imports and registrations inside `register_all_plugins()`:

```python
def register_all_plugins() -> None:
    """Register all available plugins. Called once at server startup."""
    from plugins.claim_databases.google_factcheck import GoogleFactCheckPlugin
    from plugins.government_data.transparencia import PortalTransparenciaPlugin
    from plugins.government_data.ibge_sidra import IBGESidraPlugin
    from plugins.web_search.tavily_search import TavilySearchPlugin
    from plugins.electoral.tse import TSEPlugin
    from plugins.government_data.bacen import BACENPlugin
    from plugins.claim_databases.claimbuster import ClaimBusterPlugin
    from plugins.knowledge_bases.wikipedia import WikipediaPlugin

    register(GoogleFactCheckPlugin())
    register(PortalTransparenciaPlugin())
    register(IBGESidraPlugin())
    register(TavilySearchPlugin())
    register(TSEPlugin())
    register(BACENPlugin())
    register(ClaimBusterPlugin())
    register(WikipediaPlugin())
```

**Step 2: Verify registration works**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -c "from plugins import register_all_plugins; from plugins import registry; register_all_plugins(); print(len(registry.get_all()), 'plugins'); print([p.get_metadata().name for p in registry.get_all()])"`

Expected: `8 plugins` followed by all 8 plugin names

**Step 3: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add app/plugins/__init__.py
git commit -m "feat: register 4 new plugins (TSE, BACEN, ClaimBuster, Wikipedia)"
```

---

## Task 13: Update Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add new dependencies**

Add to `requirements.txt` after the existing `sidrapy` line:

```
# Electoral data
ckanapi>=0.1.0

# Central Bank data
python-bcb>=0.3.0

# Wikipedia
wikipedia-api>=0.6.0
```

**Step 2: Install new dependencies**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke && pip install ckanapi python-bcb wikipedia-api`

Expected: Successfully installed

**Step 3: Run full test suite to verify no regressions**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/ -v`

Expected: ALL PASS

**Step 4: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add requirements.txt
git commit -m "deps: add ckanapi, python-bcb, wikipedia-api"
```

---

## Task 14: Update Integration Tests

**Files:**
- Modify: `tests/test_graph_integration.py`

**Step 1: Update plugin count and names assertions**

In `tests/test_graph_integration.py`, update `TestPluginRegistration`:

Change `test_register_all_plugins_handles_missing_env`:
```python
assert len(all_plugins) == 8
```

Change `test_registered_plugin_names`:
```python
names = {p.get_metadata().name for p in registry.get_all()}
assert names == {
    "google_factcheck",
    "portal_transparencia",
    "ibge_sidra",
    "tavily_search",
    "tse",
    "bacen",
    "claimbuster",
    "wikipedia",
}
```

**Step 2: Add routing tests for new plugin categories**

Add to `TestPluginRegistration` class:

```python
def test_electoral_tools_available(self):
    """TSE plugin should be available (no API keys needed, ckanapi installed)."""
    from plugins import register_all_plugins
    from plugins.base import PluginCategory
    register_all_plugins()
    electoral_plugins = registry.get_available(PluginCategory.ELECTORAL)
    assert len(electoral_plugins) >= 1
    names = {p.get_metadata().name for p in electoral_plugins}
    assert "tse" in names

def test_knowledge_base_tools_available(self):
    """Wikipedia plugin should be available (no API keys needed)."""
    from plugins import register_all_plugins
    from plugins.base import PluginCategory
    register_all_plugins()
    kb_plugins = registry.get_available(PluginCategory.KNOWLEDGE_BASE)
    assert len(kb_plugins) >= 1
    names = {p.get_metadata().name for p in kb_plugins}
    assert "wikipedia" in names
```

**Step 3: Run integration tests**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/test_graph_integration.py -v`

Expected: ALL PASS

**Step 4: Run full test suite**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/ -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke
git add tests/test_graph_integration.py
git commit -m "test: update integration tests for 8-plugin registry"
```

---

## Task 15: Final Verification

**Step 1: Run full test suite one final time**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/ -v`

Expected: ALL PASS — no regressions, all new tests green

**Step 2: Verify file structure**

Run: `find /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app/plugins -name "*.py" | sort`

Expected:
```
app/plugins/__init__.py
app/plugins/base.py
app/plugins/claim_databases/__init__.py
app/plugins/claim_databases/claimbuster.py
app/plugins/claim_databases/google_factcheck.py
app/plugins/electoral/__init__.py
app/plugins/electoral/tse.py
app/plugins/government_data/__init__.py
app/plugins/government_data/bacen.py
app/plugins/government_data/ibge_sidra.py
app/plugins/government_data/transparencia.py
app/plugins/knowledge_bases/__init__.py
app/plugins/knowledge_bases/wikipedia.py
app/plugins/registry.py
app/plugins/source_selector.py
app/plugins/web_search/__init__.py
app/plugins/web_search/tavily_search.py
```

**Step 3: Check total test count**

Run: `cd /Users/mbsantos/workspace/aletheia_fact/agencia/.claude/worktrees/competent-clarke/app && python -m pytest ../tests/ --co -q | tail -1`

Expected: Should show ~280+ tests collected (was 234 before, adding ~50 new tests)

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | Add ELECTORAL enum | `base.py` | 1 |
| 2 | Electoral directory | `electoral/__init__.py` | — |
| 3 | TSE tests (red) | `test_tse.py` | 12 |
| 4 | TSE implementation | `electoral/tse.py` | — |
| 5 | BACEN tests (red) | `test_bacen.py` | 11 |
| 6 | BACEN implementation | `government_data/bacen.py` | — |
| 7 | ClaimBuster tests (red) | `test_claimbuster.py` | 10 |
| 8 | ClaimBuster implementation | `claim_databases/claimbuster.py` | — |
| 9 | Knowledge bases directory | `knowledge_bases/__init__.py` | — |
| 10 | Wikipedia tests (red) | `test_wikipedia.py` | 11 |
| 11 | Wikipedia implementation | `knowledge_bases/wikipedia.py` | — |
| 12 | Register all plugins | `__init__.py` | — |
| 13 | Update dependencies | `requirements.txt` | — |
| 14 | Integration tests | `test_graph_integration.py` | 2 |
| 15 | Final verification | — | — |

**Total new tests:** ~47
**Total commits:** 14
**Estimated time:** 60-90 minutes with subagent-driven execution
