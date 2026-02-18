"""Tests for the BACEN (Central Bank) plugin."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from plugins.government_data.bacen import BACENPlugin, COMMON_SERIES


@pytest.fixture
def plugin():
    return BACENPlugin()


# --- Metadata ---

class TestMetadata:
    def test_name(self, plugin):
        assert plugin.get_metadata().name == "bacen"

    def test_category(self, plugin):
        from plugins.base import PluginCategory
        assert plugin.get_metadata().category == PluginCategory.GOVERNMENT_DATA

    def test_no_api_key_required(self, plugin):
        assert plugin.get_metadata().required_env_vars == []

    def test_reliability_score(self, plugin):
        assert plugin.get_metadata().reliability_score == 0.95


# --- Availability ---

class TestAvailability:
    def test_available_when_bcb_installed(self, plugin):
        assert plugin.is_available() is True

    def test_unavailable_when_bcb_missing(self, plugin):
        with patch("plugins.government_data.bacen.sgs", None):
            assert plugin.is_available() is False


# --- Series detection ---

class TestSeriesDetection:
    def test_detects_selic(self):
        assert BACENPlugin._detect_series("taxa selic atual") == 432

    def test_detects_ipca(self):
        assert BACENPlugin._detect_series("inflacao IPCA mensal") == 433

    def test_detects_igpm(self):
        assert BACENPlugin._detect_series("IGP-M acumulado") == 189

    def test_detects_dollar(self):
        assert BACENPlugin._detect_series("cotacao do dolar") == 1

    def test_detects_euro(self):
        assert BACENPlugin._detect_series("cambio euro") == 21619

    def test_detects_pib(self):
        assert BACENPlugin._detect_series("PIB mensal") == 4380

    def test_detects_divida(self):
        assert BACENPlugin._detect_series("divida publica liquida") == 4513

    def test_detects_desemprego(self):
        assert BACENPlugin._detect_series("taxa de desemprego") == 24369

    def test_returns_zero_for_unknown(self):
        assert BACENPlugin._detect_series("random unrelated topic xyz") == 0


# --- Search ---

class TestSearch:
    def test_returns_error_when_series_not_detected(self, plugin):
        result = plugin.search("completely unrelated topic xyz123")
        assert result.error is not None
        assert "Could not determine" in result.error

    @patch("plugins.government_data.bacen.sgs")
    def test_successful_search(self, mock_sgs, plugin):
        import pandas as pd

        mock_df = pd.DataFrame(
            {"432": [13.75, 13.25, 12.75]},
            index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        )
        mock_sgs.get.return_value = mock_df

        result = plugin.search("taxa selic")
        assert result.error is None
        assert result.result_count == 3
        assert result.metadata["series_code"] == 432

    @patch("plugins.government_data.bacen.sgs")
    def test_explicit_series_code(self, mock_sgs, plugin):
        import pandas as pd

        mock_df = pd.DataFrame(
            {"9999": [1.5]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        mock_sgs.get.return_value = mock_df

        result = plugin.search("custom query", series_code=9999)
        assert result.error is None
        assert result.metadata["series_code"] == 9999
        mock_sgs.get.assert_called_once()

    @patch("plugins.government_data.bacen.sgs")
    def test_api_error_handled(self, mock_sgs, plugin):
        mock_sgs.get.side_effect = Exception("BCB API timeout")

        result = plugin.search("taxa selic")
        assert result.error is not None
        assert "BCB query failed" in result.error

    def test_missing_bcb_handled(self, plugin):
        with patch("plugins.government_data.bacen.sgs", None):
            result = plugin.search("selic", series_code=432)
            assert result.error is not None
            assert "not installed" in result.error

    @patch("plugins.government_data.bacen.sgs")
    def test_date_range_params(self, mock_sgs, plugin):
        import pandas as pd

        mock_df = pd.DataFrame(
            {"432": [13.75]},
            index=pd.to_datetime(["2024-06-01"]),
        )
        mock_sgs.get.return_value = mock_df

        plugin.search("selic", series_code=432, start_date="2024-06-01", end_date="2024-06-30")
        call_kwargs = mock_sgs.get.call_args[1]
        assert call_kwargs["start"] == "2024-06-01"
        assert call_kwargs["end"] == "2024-06-30"
