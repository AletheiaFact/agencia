"""Plugin registry for managing data source plugins."""

import logging
from typing import Optional

from plugins.base import DataSourcePlugin, PluginCategory

logger = logging.getLogger(__name__)

_registry: dict[str, DataSourcePlugin] = {}


def register(plugin: DataSourcePlugin) -> None:
    """Register a plugin instance. Logs availability status."""
    meta = plugin.get_metadata()
    _registry[meta.name] = plugin
    available = plugin.is_available()
    logger.info(
        "Plugin registered: %s [%s] available=%s",
        meta.display_name,
        meta.category.value,
        available,
    )


def get(name: str) -> Optional[DataSourcePlugin]:
    """Get a plugin by name. Returns None if not registered."""
    return _registry.get(name)


def get_available(category: Optional[PluginCategory] = None) -> list[DataSourcePlugin]:
    """List all registered plugins that are currently available.

    Optionally filter by category.
    """
    plugins = [p for p in _registry.values() if p.is_available()]
    if category:
        plugins = [p for p in plugins if p.get_metadata().category == category]
    return plugins


def get_all() -> list[DataSourcePlugin]:
    """List all registered plugins regardless of availability."""
    return list(_registry.values())


def get_langchain_tools(category: Optional[PluginCategory] = None) -> list:
    """Get LangChain Tool objects for all available plugins.

    Useful for passing to create_tool_calling_agent().
    """
    return [p.as_langchain_tool() for p in get_available(category)]


def clear() -> None:
    """Clear all registered plugins. Used in tests."""
    _registry.clear()
