"""IBGE SIDRA plugin — Brazilian official statistics.

Queries the IBGE SIDRA API for official Brazilian statistics
(population, GDP, inflation, employment, etc.). No API key required.

Uses the sidrapy library for convenient access.
"""

import logging

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult

logger = logging.getLogger(__name__)

# Common IBGE table IDs for fact-checking
COMMON_TABLES = {
    "populacao": "6579",       # Population estimates
    "pib": "5932",             # GDP
    "ipca": "1737",            # Consumer Price Index (IPCA)
    "desemprego": "6381",      # Unemployment rate (PNAD)
    "salario_minimo": "1619",  # Minimum wage history
}


class IBGESidraPlugin(DataSourcePlugin):

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="ibge_sidra",
            display_name="IBGE SIDRA",
            description="Query official Brazilian statistics from IBGE SIDRA. "
            "Covers population, GDP, inflation (IPCA), unemployment, and more. "
            "Useful for verifying claims involving Brazilian economic or demographic data.",
            category=PluginCategory.GOVERNMENT_DATA,
            required_env_vars=[],
            reliability_score=0.9,
        )

    def is_available(self) -> bool:
        try:
            import sidrapy  # noqa: F401
            return True
        except ImportError:
            return False

    def search(
        self,
        query: str,
        table_code: str = "",
        territorial_level: str = "1",
        ibge_territorial_code: str = "all",
        period: str = "last",
        **kwargs,
    ) -> PluginResult:
        """Query IBGE SIDRA for statistics.

        Args:
            query: Natural language description of what to search (used to
                   auto-detect table if table_code is not provided).
            table_code: SIDRA table code. If empty, tries to detect from query.
            territorial_level: "1" = Brazil, "2" = Region, "3" = State,
                               "6" = Municipality (default: "1").
            ibge_territorial_code: Territory code or "all" (default: "all").
            period: Time period — "last", "last 5", "202301", etc. (default: "last").
        """
        if not table_code:
            table_code = self._detect_table(query)

        if not table_code:
            return PluginResult(
                source="ibge_sidra",
                query=query,
                error=f"Could not determine IBGE table for query: '{query}'. "
                f"Available topics: {', '.join(COMMON_TABLES.keys())}",
            )

        logger.info(
            "[ibge_sidra] Querying table=%s period=%s level=%s",
            table_code,
            period,
            territorial_level,
        )

        try:
            import sidrapy

            data = sidrapy.get_table(
                table_code=table_code,
                territorial_level=territorial_level,
                ibge_territorial_code=ibge_territorial_code,
                period=period,
            )
        except ImportError:
            return PluginResult(
                source="ibge_sidra",
                query=query,
                error="sidrapy package not installed",
            )
        except Exception as e:
            logger.error("[ibge_sidra] Error querying table %s: %s", table_code, e)
            return PluginResult(
                source="ibge_sidra",
                query=query,
                error=f"SIDRA query failed: {e}",
            )

        # sidrapy returns a pandas DataFrame
        results = []
        try:
            # Skip header row (row 0 contains column descriptions)
            for _, row in data.iloc[1:].iterrows():
                results.append({
                    "variable": row.get("V", ""),
                    "value": row.get("V", ""),
                    "unit": row.get("MN", ""),
                    "period": row.get("D2C", row.get("D3C", "")),
                    "location": row.get("D1N", ""),
                    "variable_name": row.get("D4N", row.get("D3N", "")),
                })
        except Exception as e:
            logger.warning("[ibge_sidra] Error parsing results: %s", e)
            # Return raw data as fallback
            results = [{"raw": data.to_dict()}] if hasattr(data, "to_dict") else []

        logger.info("[ibge_sidra] Retrieved %d records from table %s", len(results), table_code)
        return PluginResult(
            source="ibge_sidra",
            query=query,
            results=results,
            result_count=len(results),
            metadata={"table_code": table_code},
        )

    @staticmethod
    def _detect_table(query: str) -> str:
        """Try to detect the appropriate SIDRA table from a natural language query."""
        query_lower = query.lower()
        for keyword, table_id in COMMON_TABLES.items():
            if keyword in query_lower:
                return table_id

        # Additional keyword mappings
        keyword_map = {
            "popul": "6579",
            "habitant": "6579",
            "demogra": "6579",
            "pib": "5932",
            "gdp": "5932",
            "produto interno": "5932",
            "inflac": "1737",
            "ipca": "1737",
            "preco": "1737",
            "desempreg": "6381",
            "emprego": "6381",
            "trabalh": "6381",
            "salario": "1619",
            "salário": "1619",
        }
        for kw, table_id in keyword_map.items():
            if kw in query_lower:
                return table_id

        return ""
