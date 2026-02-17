"""Tests for the Google Fact Check plugin — API key and Service Account auth."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from plugins.claim_databases.google_factcheck import (
    GoogleFactCheckPlugin,
    _get_auth_mode,
)


@pytest.fixture
def plugin():
    return GoogleFactCheckPlugin()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure both auth env vars are cleared before each test."""
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)


# --- Metadata ---


class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "google_factcheck"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.CLAIM_DATABASE

    def test_required_env_vars_empty(self, plugin):
        assert plugin.get_metadata().required_env_vars == []

    def test_auth_env_var_groups(self, plugin):
        groups = plugin.get_metadata().auth_env_var_groups
        assert ["GOOGLE_FACTCHECK_API_KEY"] in groups
        assert ["GOOGLE_APPLICATION_CREDENTIALS"] in groups


# --- Auth Mode Detection ---


class TestAuthMode:
    def test_api_key_mode(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
        assert _get_auth_mode() == "api_key"

    def test_service_account_mode(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        assert _get_auth_mode() == "service_account"

    def test_api_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        assert _get_auth_mode() == "api_key"

    def test_neither_set(self):
        assert _get_auth_mode() is None


# --- Availability ---


class TestAvailability:
    def test_available_when_api_key_set(self, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
        assert plugin.is_available() is True

    def test_available_when_credentials_set(self, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        assert plugin.is_available() is True

    def test_unavailable_when_neither_set(self, plugin):
        assert plugin.is_available() is False


# --- Search with API Key ---


class TestSearchApiKey:
    def test_returns_error_when_no_auth(self, plugin):
        result = plugin.search("test claim")
        assert result.error is not None
        assert "No auth configured" in result.error

    @patch("plugins.claim_databases.google_factcheck.requests.get")
    def test_successful_search(self, mock_get, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "claims": [
                {
                    "text": "Earth is flat",
                    "claimant": "Internet User",
                    "claimReview": [
                        {
                            "textualRating": "False",
                            "publisher": {"name": "FactCheck.org"},
                            "url": "https://factcheck.org/earth",
                            "title": "Earth is not flat",
                            "reviewDate": "2024-01-15",
                        }
                    ],
                }
            ]
        }
        mock_get.return_value = mock_response

        result = plugin.search("Earth is flat")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["claim_text"] == "Earth is flat"
        assert result.results[0]["rating"] == "False"
        assert result.results[0]["publisher"] == "FactCheck.org"
        assert result.results[0]["url"] == "https://factcheck.org/earth"

    @patch("plugins.claim_databases.google_factcheck.requests.get")
    def test_passes_api_key_as_param(self, mock_get, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "my-key-123")
        mock_response = MagicMock()
        mock_response.json.return_value = {"claims": []}
        mock_get.return_value = mock_response

        plugin.search("test")
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["key"] == "my-key-123"

    @patch("plugins.claim_databases.google_factcheck.requests.get")
    def test_empty_response(self, mock_get, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        result = plugin.search("unique untested claim xyz123")
        assert result.error is None
        assert result.result_count == 0
        assert result.results == []

    @patch("plugins.claim_databases.google_factcheck.requests.get")
    def test_http_error_handled_gracefully(self, mock_get, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")

        result = plugin.search("test claim")
        assert result.error is not None
        assert "HTTP request failed" in result.error

    @patch("plugins.claim_databases.google_factcheck.requests.get")
    def test_multiple_reviews_per_claim(self, mock_get, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "claims": [
                {
                    "text": "Claim A",
                    "claimReview": [
                        {"textualRating": "True", "publisher": {"name": "Org1"}, "url": "u1"},
                        {"textualRating": "False", "publisher": {"name": "Org2"}, "url": "u2"},
                    ],
                }
            ]
        }
        mock_get.return_value = mock_response

        result = plugin.search("Claim A")
        assert result.result_count == 2

    @patch("plugins.claim_databases.google_factcheck.requests.get")
    def test_passes_language_and_page_size(self, mock_get, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"claims": []}
        mock_get.return_value = mock_response

        plugin.search("test", language_code="en", page_size=5)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["languageCode"] == "en"
        assert kwargs["params"]["pageSize"] == 5


# --- Search with Service Account ---


class TestSearchServiceAccount:
    @patch("plugins.claim_databases.google_factcheck._build_service_account_session")
    def test_successful_search(self, mock_build, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "claims": [
                {
                    "text": "Test claim",
                    "claimReview": [
                        {
                            "textualRating": "True",
                            "publisher": {"name": "FactChecker"},
                            "url": "https://example.com",
                        }
                    ],
                }
            ]
        }
        mock_session.get.return_value = mock_response
        mock_build.return_value = mock_session

        result = plugin.search("Test claim")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["rating"] == "True"

    @patch("plugins.claim_databases.google_factcheck._build_service_account_session")
    def test_no_api_key_param_in_sa_mode(self, mock_build, plugin, monkeypatch):
        """Service account uses Bearer token, not key= query param."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"claims": []}
        mock_session.get.return_value = mock_response
        mock_build.return_value = mock_session

        plugin.search("test")
        _, kwargs = mock_session.get.call_args
        assert "key" not in kwargs.get("params", {})

    @patch("plugins.claim_databases.google_factcheck._build_service_account_session")
    def test_session_is_cached(self, mock_build, plugin, monkeypatch):
        """Session should be created once and reused across searches."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"claims": []}
        mock_session.get.return_value = mock_response
        mock_build.return_value = mock_session

        plugin.search("query 1")
        plugin.search("query 2")
        assert mock_build.call_count == 1

    @patch("plugins.claim_databases.google_factcheck._build_service_account_session")
    def test_auth_error_handled_gracefully(self, mock_build, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/bad.json")
        mock_build.side_effect = Exception("Could not load credentials file")

        result = plugin.search("test claim")
        assert result.error is not None
        assert "Authentication failed" in result.error

    @patch("plugins.claim_databases.google_factcheck._build_service_account_session")
    def test_import_error_handled(self, mock_build, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        mock_build.side_effect = ImportError("No module named 'google.auth'")

        result = plugin.search("test claim")
        assert result.error is not None
        assert "google-auth" in result.error

    @patch("plugins.claim_databases.google_factcheck._build_service_account_session")
    def test_http_error_in_sa_mode(self, mock_build, plugin, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")

        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.ConnectionError("timeout")
        mock_build.return_value = mock_session

        result = plugin.search("test claim")
        assert result.error is not None
        assert "HTTP request failed" in result.error


# --- LangChain tool ---


class TestLangChainTool:
    def test_tool_name(self, plugin):
        tool = plugin.as_langchain_tool()
        assert tool.name == "search_existing_factchecks"

    def test_tool_description(self, plugin):
        tool = plugin.as_langchain_tool()
        assert "fact-check" in tool.description.lower()
