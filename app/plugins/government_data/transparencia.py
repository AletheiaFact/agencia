"""Portal da Transparência plugin — Brazilian federal government spending data.

Queries the Portal da Transparência API for federal contracts, agreements,
direct spending, and other government financial data. No API key required.
"""

import logging

import requests

from plugins.base import DataSourcePlugin, PluginCategory, PluginMetadata, PluginResult

logger = logging.getLogger(__name__)

API_BASE = "https://api.portaldatransparencia.gov.br/api-de-dados"


class PortalTransparenciaPlugin(DataSourcePlugin):

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="portal_transparencia",
            display_name="Portal da Transparência",
            description="Search Brazilian federal government spending data including "
            "contracts, agreements, and direct spending. Useful for verifying "
            "claims about government expenditures.",
            category=PluginCategory.GOVERNMENT_DATA,
            required_env_vars=[],
            rate_limit_rpm=30,
            reliability_score=0.85,
        )

    def is_available(self) -> bool:
        return True

    def search(
        self,
        query: str,
        endpoint: str = "contratos",
        page: int = 1,
        page_size: int = 10,
        **kwargs,
    ) -> PluginResult:
        """Search Portal da Transparência for government data.

        Args:
            query: Search term (used as keyword filter).
            endpoint: API endpoint — "contratos", "convenios", "despesas",
                      "licitacoes", etc. (default: "contratos").
            page: Page number (default: 1).
            page_size: Results per page (default: 10).
        """
        logger.info(
            "[portal_transparencia] Searching — query='%s' endpoint=%s",
            query[:80],
            endpoint,
        )

        url = f"{API_BASE}/{endpoint}"
        params = {
            "termoBusca": query,
            "pagina": page,
            "tamanhoPagina": page_size,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error("[portal_transparencia] HTTP error: %s", e)
            return PluginResult(
                source="portal_transparencia",
                query=query,
                error=f"HTTP request failed: {e}",
            )

        # API returns a list of records directly
        records = data if isinstance(data, list) else data.get("results", [])

        results = []
        for record in records[:page_size]:
            results.append({
                "id": record.get("id", ""),
                "description": record.get("objeto", record.get("descricao", "")),
                "value": record.get("valor", record.get("valorInicial", "")),
                "entity": record.get("orgaoVinculado", {}).get("nome", "")
                if isinstance(record.get("orgaoVinculado"), dict) else "",
                "date": record.get("dataInicioVigencia", record.get("dataCompra", "")),
                "raw": record,
            })

        logger.info("[portal_transparencia] Found %d records", len(results))
        return PluginResult(
            source="portal_transparencia",
            query=query,
            results=results,
            result_count=len(results),
            metadata={"endpoint": endpoint},
        )
