"""Multi-query search tool for Querido Diário API.

Replaces the single-query querido_diario_fetch tool with support for:
- Multiple search queries merged and deduplicated
- Configurable result count (size), excerpt size, and excerpt count
- Excerpt-based pre-filtering before full-text download
"""

import json
import logging
import os
from typing import Optional

import requests
from langchain.tools import tool
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Load IBGE city codes mapping
_json_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "ibge_cities_code.json")
with open(_json_file_path, "r", encoding="utf-8") as f:
    _cities = json.load(f)

API_URL = "https://api.queridodiario.ok.org.br/gazettes"


def _resolve_territory_ids(city: str | None) -> list[str]:
    """Resolve a city name to IBGE territory IDs."""
    if not city:
        return []
    city_clean = city.strip()
    if not city_clean or city_clean not in _cities:
        return []
    tid = _cities[city_clean]
    if isinstance(tid, list):
        return [str(t) for t in tid]
    return [str(tid)]


def _fetch_single_query(
    query: str,
    territory_ids: list[str],
    published_since: Optional[str],
    published_until: Optional[str],
    size: int,
    excerpt_size: int,
    number_of_excerpts: int,
) -> dict:
    """Execute a single API query and return raw response data."""
    params: dict = {
        "querystring": query,
        "sort_by": "relevance",
        "size": size,
        "excerpt_size": excerpt_size,
        "number_of_excerpts": number_of_excerpts,
    }
    if territory_ids:
        params["territory_ids"] = territory_ids
    if published_since and published_since.strip().lower() not in ("", "none"):
        params["published_since"] = published_since.strip()
    if published_until and published_until.strip().lower() not in ("", "none"):
        params["published_until"] = published_until.strip()

    query_string = urlencode(params, doseq=True)
    full_url = f"{API_URL}?{query_string}"
    logger.info("[querido_diario_search] GET %s", full_url)

    response = requests.get(full_url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return {
        "total_gazettes": data.get("total_gazettes", 0),
        "gazettes": data.get("gazettes", []),
        "query": query,
    }


@tool("querido_diario_search")
def querido_diario_search(
    queries: list[str],
    city: Optional[str] = None,
    published_since: Optional[str] = None,
    published_until: Optional[str] = None,
    size: int = 30,
    excerpt_size: int = 500,
    number_of_excerpts: int = 3,
) -> dict:
    """Search Querido Diário API with multiple queries, returning merged scored results.

    Executes each query against the API, merges results, deduplicates by txt_url,
    and returns all gazette candidates with their excerpts for scoring.

    Args:
        queries: List of search query strings (Elasticsearch simple syntax).
        city: Brazilian city name to filter results (must match IBGE codes database).
        published_since: Start date filter in YYYY-MM-DD format.
        published_until: End date filter in YYYY-MM-DD format.
        size: Number of results per query (default 30).
        excerpt_size: Characters per excerpt snippet (default 500).
        number_of_excerpts: Number of excerpts per gazette (default 3).
    """
    logger.info(
        "[querido_diario_search] queries=%s city='%s' since='%s' until='%s' size=%d",
        [q[:60] for q in queries], city, published_since, published_until, size,
    )

    territory_ids = _resolve_territory_ids(city) if city else []
    if city and not territory_ids:
        logger.warning("[querido_diario_search] City '%s' not in IBGE database, searching without city filter", city)

    all_gazettes: list[dict] = []
    seen_urls: set[str] = set()
    query_stats: list[dict] = []

    for query in queries:
        try:
            result = _fetch_single_query(
                query=query,
                territory_ids=territory_ids,
                published_since=published_since,
                published_until=published_until,
                size=size,
                excerpt_size=excerpt_size,
                number_of_excerpts=number_of_excerpts,
            )
            query_stats.append({
                "query": query,
                "total_gazettes": result["total_gazettes"],
                "returned": len(result["gazettes"]),
            })
            logger.info(
                "[querido_diario_search] query='%s' total=%d returned=%d",
                query[:60], result["total_gazettes"], len(result["gazettes"]),
            )

            for gazette in result["gazettes"]:
                txt_url = gazette.get("txt_url", "")
                if txt_url and txt_url not in seen_urls:
                    seen_urls.add(txt_url)
                    gazette["_matched_query"] = query
                    all_gazettes.append(gazette)

        except requests.exceptions.RequestException as e:
            logger.error("[querido_diario_search] HTTP error for query '%s': %s", query[:60], e)
            query_stats.append({"query": query, "error": str(e)})
        except Exception as e:
            logger.error("[querido_diario_search] Unexpected error for query '%s': %s", query[:60], e)
            query_stats.append({"query": query, "error": str(e)})

    logger.info(
        "[querido_diario_search] Total unique gazettes: %d from %d queries",
        len(all_gazettes), len(queries),
    )

    return {
        "gazettes": all_gazettes,
        "total_unique": len(all_gazettes),
        "query_stats": query_stats,
    }
