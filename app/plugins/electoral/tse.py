"""TSE Open Data plugin — Brazilian electoral data.

Queries TSE (Superior Electoral Court) open data APIs for electoral
information: CKAN datasets, candidate registrations (DivulgaCandContas),
and election results. No API key required.
"""

import logging

import requests

try:
    import ckanapi
except ImportError:  # pragma: no cover
    ckanapi = None  # type: ignore[assignment]

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
        return ckanapi is not None

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
            query[:80] if query else "",
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
        if ckanapi is None:
            return PluginResult(
                source="tse", query=query,
                error="ckanapi package not installed",
            )
        try:
            ckan = ckanapi.RemoteCKAN(CKAN_BASE.rsplit("/api/3", 1)[0])
            data = ckan.action.package_search(q=query, rows=10)
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
