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
