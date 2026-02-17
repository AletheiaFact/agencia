import json
import logging
import os
from typing import Optional

import requests
from langchain.tools import tool
from urllib.parse import urlencode

from errors import NoGazettesFoundError, CityNotFoundError

logger = logging.getLogger(__name__)

# Load IBGE city codes mapping
_json_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "ibge_cities_code.json")
with open(_json_file_path, "r", encoding="utf-8") as f:
    _cities = json.load(f)


@tool("fetch_querido_diario_api")
def querido_diario_fetch(
    subject: str,
    city: Optional[str] = None,
    published_since: Optional[str] = None,
    published_until: Optional[str] = None,
):
    """Fetches data from Querido Diario API to search about municipal gazettes.
    Returns a gazette object with fields including txt_url for the full text.

    Args:
        subject: Search query string for gazette content.
        city: Brazilian city name to filter results (must match IBGE codes database).
        published_since: Start date filter in YYYY-MM-DD format.
        published_until: End date filter in YYYY-MM-DD format.
    """
    logger.info("[fetch_querido_diario_api] subject='%s' city='%s' since='%s' until='%s'", subject[:80], city, published_since, published_until)
    try:
        api_url = "https://api.queridodiario.ok.org.br/gazettes"

        query_params = {"querystring": subject, "sort_by": "relevance"}

        # Filter by city if provided and found in IBGE codes
        city_clean = (city or "").strip()
        if city_clean:
            if city_clean in _cities:
                territory_id = _cities[city_clean]
                if isinstance(territory_id, list):
                    query_params["territory_ids[]"] = territory_id
                else:
                    query_params["territory_ids[]"] = [territory_id]
                logger.info("[fetch_querido_diario_api] Filtering by city=%s territory_id=%s", city_clean, territory_id)
            else:
                logger.warning("[fetch_querido_diario_api] City '%s' not in IBGE database, searching without city filter", city_clean)

        if published_since and published_since.strip().lower() not in ("", "none"):
            query_params["published_since"] = published_since.strip()
        if published_until and published_until.strip().lower() not in ("", "none"):
            query_params["published_until"] = published_until.strip()

        query_string = urlencode(query_params, doseq=True)
        full_url = f"{api_url}?{query_string}"
        logger.info("[fetch_querido_diario_api] GET %s", full_url)
        response = requests.get(full_url)
        response.raise_for_status()
        gazettes = response.json().get("gazettes", [])
        logger.info("[fetch_querido_diario_api] API returned %d gazettes", len(gazettes))

        if not gazettes:
            return {"error": "No public gazettes found for this search query."}

        gazette = gazettes[0]
        txt_url = gazette.get("txt_url", "")
        logger.info("[fetch_querido_diario_api] Top gazette txt_url=%s", txt_url)
        return gazette

    except requests.exceptions.RequestException as e:
        logger.error("[fetch_querido_diario_api] HTTP error: %s", e)
        return {"error": f"HTTP request failed: {e}"}
    except Exception as e:
        logger.error("[fetch_querido_diario_api] Unexpected error: %s", e)
        return {"error": f"An unexpected error occurred: {e}"}
