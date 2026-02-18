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
