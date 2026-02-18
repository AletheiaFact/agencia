"""BACEN plugin — Brazilian Central Bank economic data.

Queries the BCB SGS (Time Series Management System) for official economic
indicators: SELIC rate, inflation (IPCA, IGP-M), exchange rates, GDP,
public debt, and unemployment. No API key required.

Uses the python-bcb library for convenient access.
"""

import logging
from datetime import datetime, timedelta

try:
    from bcb import sgs
except ImportError:  # pragma: no cover
    sgs = None  # type: ignore[assignment]

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult

logger = logging.getLogger(__name__)

# Common BCB SGS series codes
COMMON_SERIES = {
    "selic": 432,
    "ipca": 433,
    "igpm": 189,
    "igp-m": 189,
    "cambio_dolar": 1,
    "dolar": 1,
    "cambio_euro": 21619,
    "euro": 21619,
    "pib_mensal": 4380,
    "pib": 4380,
    "divida_publica": 4513,
    "divida": 4513,
    "desemprego": 24369,
}


class BACENPlugin(DataSourcePlugin):

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="bacen",
            display_name="BACEN (Central Bank)",
            description="Query Brazilian Central Bank (BACEN) economic data. "
            "Covers SELIC interest rate, inflation (IPCA, IGP-M), "
            "exchange rates (USD, EUR), GDP, public debt, and unemployment. "
            "Use for verifying claims about Brazilian economic indicators.",
            category=PluginCategory.GOVERNMENT_DATA,
            required_env_vars=[],
            reliability_score=0.95,
        )

    def is_available(self) -> bool:
        return sgs is not None

    def search(
        self,
        query: str,
        series_code: int = 0,
        start_date: str = "",
        end_date: str = "",
        **kwargs,
    ) -> PluginResult:
        """Query BACEN SGS for economic time series data.

        Args:
            query: Natural language description (used to auto-detect series).
            series_code: SGS series code. If 0, tries to detect from query.
            start_date: Start date "YYYY-MM-DD" — empty = last 12 months.
            end_date: End date "YYYY-MM-DD" — empty = today.
        """
        if sgs is None:
            return PluginResult(
                source="bacen",
                query=query,
                error="python-bcb package not installed",
            )

        if not series_code:
            series_code = self._detect_series(query)

        if not series_code:
            return PluginResult(
                source="bacen",
                query=query,
                error=f"Could not determine BCB series for query: '{query}'. "
                f"Available topics: {', '.join(sorted(set(COMMON_SERIES.keys())))}"
            )

        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            "[bacen] Querying series=%d period=%s to %s",
            series_code, start_date, end_date,
        )

        try:
            data = sgs.get(
                codes={str(series_code): series_code},
                start=start_date,
                end=end_date,
            )
        except Exception as e:
            logger.error("[bacen] Error querying series %d: %s", series_code, e)
            return PluginResult(
                source="bacen",
                query=query,
                error=f"BCB query failed: {e}",
            )

        results = []
        try:
            col = str(series_code)
            for date_idx, row in data.iterrows():
                value = row.get(col, row.iloc[0] if len(row) > 0 else None)
                results.append({
                    "date": str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx),
                    "value": float(value) if value is not None else None,
                    "series": series_code,
                })
        except Exception as e:
            logger.warning("[bacen] Error parsing results: %s", e)
            results = [{"raw": data.to_dict()}] if hasattr(data, "to_dict") else []

        logger.info("[bacen] Retrieved %d records for series %d", len(results), series_code)
        return PluginResult(
            source="bacen",
            query=query,
            results=results,
            result_count=len(results),
            metadata={"series_code": series_code},
        )

    @staticmethod
    def _detect_series(query: str) -> int:
        """Try to detect the appropriate SGS series from a natural language query."""
        query_lower = query.lower()
        for keyword, code in COMMON_SERIES.items():
            if keyword in query_lower:
                return code

        # Additional keyword mappings
        keyword_map = {
            "juro": 432,
            "taxa basica": 432,
            "inflac": 433,
            "preco": 433,
            "cambio": 1,
            "moeda": 1,
            "produto interno": 4380,
            "gdp": 4380,
            "divid": 4513,
            "empreg": 24369,
            "trabalh": 24369,
        }
        for kw, code in keyword_map.items():
            if kw in query_lower:
                return code

        return 0
