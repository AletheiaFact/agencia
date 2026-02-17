"""Tests for the IBGE SIDRA plugin."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from plugins.government_data.ibge_sidra import IBGESidraPlugin, COMMON_TABLES


@pytest.fixture
def plugin():
    return IBGESidraPlugin()


# --- Metadata ---

class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "ibge_sidra"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.GOVERNMENT_DATA

    def test_no_api_key_required(self, plugin):
        assert plugin.get_metadata().required_env_vars == []


# --- Availability ---

class TestAvailability:
    def test_available_when_sidrapy_installed(self, plugin):
        assert plugin.is_available() is True

    def test_unavailable_when_sidrapy_missing(self, plugin):
        with patch.dict(sys.modules, {"sidrapy": None}):
            assert plugin.is_available() is False


# --- Table detection ---

class TestTableDetection:
    def test_detects_population(self, plugin):
        assert IBGESidraPlugin._detect_table("população brasileira") == "6579"

    def test_detects_pib(self, plugin):
        assert IBGESidraPlugin._detect_table("PIB do Brasil") == "5932"

    def test_detects_ipca(self, plugin):
        assert IBGESidraPlugin._detect_table("inflação IPCA") == "1737"

    def test_detects_unemployment(self, plugin):
        assert IBGESidraPlugin._detect_table("taxa de desemprego") == "6381"

    def test_detects_salary(self, plugin):
        assert IBGESidraPlugin._detect_table("salário mínimo") == "1619"

    def test_returns_empty_for_unknown(self, plugin):
        assert IBGESidraPlugin._detect_table("random unrelated topic xyz") == ""


# --- Search ---

class TestSearch:
    def test_returns_error_when_table_not_detected(self, plugin):
        result = plugin.search("completely unrelated topic xyz123")
        assert result.error is not None
        assert "Could not determine" in result.error

    def test_successful_search(self, plugin):
        import pandas as pd

        mock_df = pd.DataFrame({
            "V": ["Valor", "12345"],
            "MN": ["Unidade", "Pessoas"],
            "D2C": ["Período", "2024"],
            "D1N": ["Local", "Brasil"],
            "D4N": ["Variável", "População residente"],
        })
        mock_sidrapy = MagicMock()
        mock_sidrapy.get_table.return_value = mock_df

        with patch.dict(sys.modules, {"sidrapy": mock_sidrapy}):
            result = plugin.search("população brasileira")

        assert result.error is None
        assert result.result_count == 1
        assert result.metadata["table_code"] == "6579"

    def test_explicit_table_code(self, plugin):
        import pandas as pd

        mock_df = pd.DataFrame({"V": ["Valor", "100"]})
        mock_sidrapy = MagicMock()
        mock_sidrapy.get_table.return_value = mock_df

        with patch.dict(sys.modules, {"sidrapy": mock_sidrapy}):
            result = plugin.search("custom query", table_code="9999")

        mock_sidrapy.get_table.assert_called_once()
        call_kwargs = mock_sidrapy.get_table.call_args
        assert call_kwargs[1]["table_code"] == "9999"

    def test_api_error_handled(self, plugin):
        mock_sidrapy = MagicMock()
        mock_sidrapy.get_table.side_effect = Exception("SIDRA API timeout")

        with patch.dict(sys.modules, {"sidrapy": mock_sidrapy}):
            result = plugin.search("população do Brasil")

        assert result.error is not None
        assert "SIDRA query failed" in result.error

    def test_missing_sidrapy_handled(self, plugin):
        with patch.dict(sys.modules, {"sidrapy": None}):
            result = plugin.search("população", table_code="6579")
            assert result.error is not None
            assert "not installed" in result.error
