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
    """Fetches data from Querido Diario API to search about municipal gazettes."""
    logger.info("[fetch_querido_diario_api] subject=%s city=%s since=%s until=%s", subject[:80], city, published_since, published_until)
    try:
        api_url = "https://api.queridodiario.ok.org.br/gazettes"

        query_params = {"querystring": subject, "sort_by": "relevance"}

        if city and city in _cities:
            territory_id = _cities[city]
            # New API expects territory_ids[] as array parameter
            if isinstance(territory_id, list):
                query_params["territory_ids[]"] = territory_id
            else:
                query_params["territory_ids[]"] = [territory_id]
        else:
            raise CityNotFoundError()

        if published_since and published_since.lower() != "none":
            query_params["published_since"] = published_since
        if published_until and published_until.lower() != "none":
            query_params["published_until"] = published_until

        query_string = urlencode(query_params, doseq=True)
        full_url = f"{api_url}?{query_string}"
        logger.info("[fetch_querido_diario_api] GET %s", full_url)
        response = requests.get(full_url)
        gazettes = response.json().get("gazettes", [])
        logger.info("[fetch_querido_diario_api] API returned %d gazettes", len(gazettes))

        if not gazettes:
            raise NoGazettesFoundError()

        # TODO: Return and handle the 5 most relevant public gazettes
        return gazettes[0]

    except NoGazettesFoundError:
        logger.error("[fetch_querido_diario_api] No gazettes found")
        return {"error": "No public gazettes found"}
    except CityNotFoundError:
        logger.error("[fetch_querido_diario_api] City not found: %s", city)
        return {"error": "City not found in our database"}
    except Exception as e:
        logger.error("[fetch_querido_diario_api] Unexpected error: %s", e)
        return {"error": "An unexpected error occurred."}
