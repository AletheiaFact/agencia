"""Google Fact Check Tools API plugin.

Queries Google's ClaimReview database to find existing fact-checks
from hundreds of organizations worldwide (including Brazilian ones
like Lupa and Aos Fatos).

Supports two authentication modes:
  1. API key: Set GOOGLE_FACTCHECK_API_KEY (simplest)
  2. Service Account: Set GOOGLE_APPLICATION_CREDENTIALS to a JSON key file path
"""

import logging
import os
from typing import Optional

import requests
from langchain.tools import tool

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult

logger = logging.getLogger(__name__)

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
_SCOPES = ["https://www.googleapis.com/auth/factchecktools"]


def _get_auth_mode() -> Optional[str]:
    """Determine which auth mode is available.

    Returns "api_key", "service_account", or None.
    API key takes precedence when both are set.
    """
    if os.getenv("GOOGLE_FACTCHECK_API_KEY"):
        return "api_key"
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return "service_account"
    return None


def _build_service_account_session():
    """Build an AuthorizedSession using default credentials.

    google.auth.default() picks up GOOGLE_APPLICATION_CREDENTIALS
    automatically. The session auto-refreshes expired tokens.
    """
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _project = google.auth.default(scopes=_SCOPES)
    return AuthorizedSession(credentials)


class GoogleFactCheckPlugin(DataSourcePlugin):

    def __init__(self):
        self._session = None  # lazy-initialized AuthorizedSession

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="google_factcheck",
            display_name="Google Fact Check API",
            description="Search existing fact-checks from Google's ClaimReview database. "
            "Returns prior fact-checks from reputable organizations worldwide.",
            category=PluginCategory.CLAIM_DATABASE,
            required_env_vars=[],
            auth_env_var_groups=[
                ["GOOGLE_FACTCHECK_API_KEY"],
                ["GOOGLE_APPLICATION_CREDENTIALS"],
            ],
            reliability_score=0.9,
        )

    def is_available(self) -> bool:
        return _get_auth_mode() is not None

    def search(
        self,
        query: str,
        language_code: str = "pt",
        page_size: int = 10,
        **kwargs,
    ) -> PluginResult:
        """Search for existing fact-checks matching a claim.

        Args:
            query: The claim text to search for.
            language_code: BCP-47 language code (default: "pt" for Portuguese).
            page_size: Max results to return (default: 10).
        """
        auth_mode = _get_auth_mode()
        if auth_mode is None:
            return PluginResult(
                source="google_factcheck",
                query=query,
                error="No auth configured: set GOOGLE_FACTCHECK_API_KEY or GOOGLE_APPLICATION_CREDENTIALS",
            )

        logger.info(
            "[google_factcheck] Searching — query='%s' lang=%s auth=%s",
            query[:80],
            language_code,
            auth_mode,
        )

        try:
            if auth_mode == "api_key":
                response = requests.get(
                    API_URL,
                    params={
                        "query": query,
                        "languageCode": language_code,
                        "pageSize": page_size,
                        "key": os.getenv("GOOGLE_FACTCHECK_API_KEY"),
                    },
                    timeout=10,
                )
            else:
                # Service account: use AuthorizedSession (lazy init)
                if self._session is None:
                    self._session = _build_service_account_session()
                response = self._session.get(
                    API_URL,
                    params={
                        "query": query,
                        "languageCode": language_code,
                        "pageSize": page_size,
                    },
                    timeout=10,
                )

            if not response.ok:
                logger.error(
                    "[google_factcheck] HTTP %d — body: %s",
                    response.status_code,
                    response.text[:500],
                )
            response.raise_for_status()
            data = response.json()
        except ImportError:
            logger.error("[google_factcheck] google-auth package not installed")
            return PluginResult(
                source="google_factcheck",
                query=query,
                error="Service account auth requires 'google-auth' package. "
                "Install with: pip install google-auth",
            )
        except requests.exceptions.RequestException as e:
            logger.error("[google_factcheck] HTTP error: %s", e)
            return PluginResult(
                source="google_factcheck",
                query=query,
                error=f"HTTP request failed: {e}",
            )
        except Exception as e:
            logger.error("[google_factcheck] Auth error: %s", e)
            return PluginResult(
                source="google_factcheck",
                query=query,
                error=f"Authentication failed: {e}",
            )

        claims = data.get("claims", [])
        results = []
        for claim_item in claims:
            for review in claim_item.get("claimReview", []):
                results.append({
                    "claim_text": claim_item.get("text", ""),
                    "claimant": claim_item.get("claimant", ""),
                    "rating": review.get("textualRating", ""),
                    "publisher": review.get("publisher", {}).get("name", ""),
                    "url": review.get("url", ""),
                    "review_title": review.get("title", ""),
                    "review_date": review.get("reviewDate", ""),
                })

        logger.info("[google_factcheck] Found %d fact-check reviews", len(results))
        return PluginResult(
            source="google_factcheck",
            query=query,
            results=results,
            result_count=len(results),
        )

    def as_langchain_tool(self):
        """Return a LangChain tool with proper schema for claim search."""
        plugin = self

        @tool("search_existing_factchecks")
        def search_existing_factchecks(
            query: str, language_code: str = "pt"
        ) -> str:
            """Search Google's fact-check database for existing verifications of a claim.

            Args:
                query: The claim text to search for existing fact-checks.
                language_code: Language code for results (default: pt for Portuguese).
            """
            result = plugin.search(query=query, language_code=language_code)
            return result.model_dump_json()

        return search_existing_factchecks
