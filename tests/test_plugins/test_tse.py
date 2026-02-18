"""Tests for the TSE (Superior Electoral Court) plugin."""

from unittest.mock import MagicMock, patch

import pytest

from plugins.electoral.tse import TSEPlugin


@pytest.fixture
def plugin():
    return TSEPlugin()


# --- Metadata ---

class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "tse"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.ELECTORAL

    def test_no_api_key_required(self, plugin):
        assert plugin.get_metadata().required_env_vars == []

    def test_reliability_score(self, plugin):
        assert plugin.get_metadata().reliability_score == 0.95


# --- Availability ---

class TestAvailability:
    def test_available_when_ckanapi_installed(self, plugin):
        assert plugin.is_available() is True

    def test_unavailable_when_ckanapi_missing(self, plugin):
        with patch("plugins.electoral.tse.ckanapi", None):
            assert plugin.is_available() is False


# --- CKAN dataset search ---

class TestSearchDatasets:
    @patch("plugins.electoral.tse.ckanapi.RemoteCKAN")
    def test_successful_dataset_search(self, mock_ckan_cls, plugin):
        mock_ckan = MagicMock()
        mock_ckan.action.package_search.return_value = {
            "results": [
                {
                    "id": "ds-001",
                    "title": "Candidatos 2024",
                    "notes": "Lista de candidatos eleicoes 2024",
                    "metadata_modified": "2024-08-15",
                }
            ]
        }
        mock_ckan_cls.return_value = mock_ckan

        result = plugin.search("candidatos 2024", endpoint="datasets")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["title"] == "Candidatos 2024"
        assert result.metadata["endpoint"] == "datasets"

    @patch("plugins.electoral.tse.ckanapi.RemoteCKAN")
    def test_empty_dataset_search(self, mock_ckan_cls, plugin):
        mock_ckan = MagicMock()
        mock_ckan.action.package_search.return_value = {"results": []}
        mock_ckan_cls.return_value = mock_ckan

        result = plugin.search("nonexistent xyz", endpoint="datasets")
        assert result.error is None
        assert result.result_count == 0

    @patch("plugins.electoral.tse.ckanapi.RemoteCKAN")
    def test_ckan_api_error(self, mock_ckan_cls, plugin):
        mock_ckan = MagicMock()
        mock_ckan.action.package_search.side_effect = Exception("CKAN timeout")
        mock_ckan_cls.return_value = mock_ckan

        result = plugin.search("test", endpoint="datasets")
        assert result.error is not None
        assert "CKAN" in result.error


# --- Candidate search ---

class TestSearchCandidates:
    @patch("plugins.electoral.tse.requests.get")
    def test_successful_candidate_search(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidatos": [
                {
                    "id": 12345,
                    "nomeUrna": "CANDIDATO TESTE",
                    "numero": 13,
                    "partido": {"sigla": "PT"},
                    "cargo": {"nome": "Prefeito"},
                }
            ]
        }
        mock_get.return_value = mock_response

        result = plugin.search("candidato teste", endpoint="candidates", election_year="2024", state="SP")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["name"] == "CANDIDATO TESTE"
        assert result.results[0]["party"] == "PT"
        assert result.metadata["endpoint"] == "candidates"

    @patch("plugins.electoral.tse.requests.get")
    def test_candidate_http_error(self, mock_get, plugin):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("timeout")

        result = plugin.search("test", endpoint="candidates")
        assert result.error is not None
        assert "HTTP request failed" in result.error

    @patch("plugins.electoral.tse.requests.get")
    def test_candidate_non_200(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response

        result = plugin.search("test", endpoint="candidates")
        assert result.error is not None


# --- Results search ---

class TestSearchResults:
    @patch("plugins.electoral.tse.requests.get")
    def test_successful_results_search(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "abrangencia": [
                {
                    "candidatos": [
                        {
                            "nomeUrna": "CANDIDATO A",
                            "numero": 13,
                            "partido": {"sigla": "PT"},
                            "totVotos": "50000",
                            "resultado": "Eleito",
                        }
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response

        result = plugin.search("", endpoint="results", election_year="2022", state="SP", office="governador")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["name"] == "CANDIDATO A"
        assert result.results[0]["votes"] == "50000"
        assert result.metadata["endpoint"] == "results"

    @patch("plugins.electoral.tse.requests.get")
    def test_results_404_returns_empty(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        result = plugin.search("", endpoint="results", election_year="1800")
        assert result.error is None
        assert result.result_count == 0


# --- Default endpoint ---

class TestDefaultEndpoint:
    @patch("plugins.electoral.tse.ckanapi.RemoteCKAN")
    def test_default_is_datasets(self, mock_ckan_cls, plugin):
        mock_ckan = MagicMock()
        mock_ckan.action.package_search.return_value = {"results": []}
        mock_ckan_cls.return_value = mock_ckan

        result = plugin.search("test")
        assert result.metadata["endpoint"] == "datasets"

    def test_invalid_endpoint(self, plugin):
        result = plugin.search("test", endpoint="invalid_xyz")
        assert result.error is not None
        assert "Unknown endpoint" in result.error
