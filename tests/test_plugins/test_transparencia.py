"""Tests for the Portal da Transparência plugin."""

from unittest.mock import MagicMock, patch

import pytest

from plugins.government_data.transparencia import PortalTransparenciaPlugin


@pytest.fixture
def plugin():
    return PortalTransparenciaPlugin()


# --- Metadata ---

class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "portal_transparencia"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.GOVERNMENT_DATA

    def test_no_api_key_required(self, plugin):
        assert plugin.get_metadata().required_env_vars == []

    def test_has_rate_limit(self, plugin):
        assert plugin.get_metadata().rate_limit_rpm == 30


# --- Availability ---

class TestAvailability:
    def test_always_available(self, plugin):
        assert plugin.is_available() is True


# --- Search ---

class TestSearch:
    @patch("plugins.government_data.transparencia.requests.get")
    def test_successful_search(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "12345",
                "objeto": "Contrato de serviços de TI",
                "valor": 50000.0,
                "orgaoVinculado": {"nome": "Ministério da Educação"},
                "dataInicioVigencia": "2024-01-01",
            }
        ]
        mock_get.return_value = mock_response

        result = plugin.search("serviços de TI")
        assert result.error is None
        assert result.result_count == 1
        assert result.results[0]["description"] == "Contrato de serviços de TI"
        assert result.results[0]["value"] == 50000.0
        assert result.results[0]["entity"] == "Ministério da Educação"
        assert result.metadata["endpoint"] == "contratos"

    @patch("plugins.government_data.transparencia.requests.get")
    def test_empty_response(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        result = plugin.search("nonexistent query xyz")
        assert result.error is None
        assert result.result_count == 0

    @patch("plugins.government_data.transparencia.requests.get")
    def test_http_error(self, mock_get, plugin):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("timeout")

        result = plugin.search("test")
        assert result.error is not None
        assert "HTTP request failed" in result.error

    @patch("plugins.government_data.transparencia.requests.get")
    def test_passes_endpoint_param(self, mock_get, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        plugin.search("test", endpoint="licitacoes")
        call_args = mock_get.call_args
        assert "licitacoes" in call_args[0][0] or "licitacoes" in str(call_args)

    @patch("plugins.government_data.transparencia.requests.get")
    def test_handles_dict_response(self, mock_get, plugin):
        """Some endpoints return a dict with 'results' key instead of a list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": "1", "descricao": "Item A", "valor": 100}
            ]
        }
        mock_get.return_value = mock_response

        result = plugin.search("test")
        assert result.result_count == 1
        assert result.results[0]["description"] == "Item A"
