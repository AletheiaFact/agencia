"""Tests for the Wikipedia/Wikidata plugin."""

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
        with patch("plugins.knowledge_bases.wikipedia.wikipediaapi", None):
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
        call_args = mock_wikiapi.Wikipedia.call_args
        # The language param should be passed — check positional or keyword args
        args, kwargs = call_args
        assert "en" in args or kwargs.get("language") == "en"


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
