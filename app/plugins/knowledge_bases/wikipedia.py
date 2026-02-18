"""Wikipedia/Wikidata plugin — encyclopedic facts and structured data.

Queries Wikipedia for article summaries and Wikidata for structured
entity data via SPARQL. No API key required.

Uses the wikipedia-api library for Wikipedia and requests for Wikidata.
"""

import logging

import requests

try:
    import wikipediaapi
except ImportError:  # pragma: no cover
    wikipediaapi = None  # type: ignore[assignment]

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
        return wikipediaapi is not None

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
        if wikipediaapi is None:
            return PluginResult(
                source="wikipedia", query=query,
                error="wikipedia-api package not installed",
            )
        try:
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
                "summary": page.summary[:1000],
                "url": page.fullurl,
            }]
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
        safe_query = query.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
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
