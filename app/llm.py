"""Centralized LLM factory with provider abstraction.

Supports OpenAI and Anthropic (Claude) models via environment-driven configuration.
All nodes should use get_llm() instead of instantiating ChatOpenAI directly.

Configuration via environment variables:
    LLM_PROVIDER: "openai" (default) or "anthropic"
    LLM_MODEL: Model name override (e.g., "gpt-5.2-2025-12-11", "claude-sonnet-4-6")
    LLM_MODEL_MINI: Model name override for cheap/fast tasks
    LLM_TEMPERATURE: Temperature override (default: 1)

Provider-specific keys:
    OPENAI_API_KEY: Required when LLM_PROVIDER=openai
    ANTHROPIC_API_KEY: Required when LLM_PROVIDER=anthropic
"""

import os
import logging

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

_PROVIDER_DEFAULTS = {
    "openai": {
        "model": "gpt-5.2-2025-12-11",
        "model_mini": "gpt-5-mini-2025-08-07",
    },
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "model_mini": "claude-haiku-4-5-20251001",
    },
}

_VALID_PROVIDERS = set(_PROVIDER_DEFAULTS.keys())


def _get_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower().strip()
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"Invalid LLM_PROVIDER='{provider}'. Must be one of: {_VALID_PROVIDERS}"
        )
    return provider


def _get_temperature() -> float:
    return float(os.environ.get("LLM_TEMPERATURE", "1"))


def get_llm(*, mini: bool = False, temperature: float | None = None) -> BaseChatModel:
    """Create an LLM instance using the configured provider.

    Args:
        mini: If True, use the cheaper/faster model variant.
              Used for scoring, classification, and report formatting.
        temperature: Override the default temperature. If None, uses
                     LLM_TEMPERATURE env var (default: 1).

    Returns:
        A LangChain chat model instance (ChatOpenAI or ChatAnthropic).
    """
    provider = _get_provider()
    defaults = _PROVIDER_DEFAULTS[provider]
    temp = temperature if temperature is not None else _get_temperature()

    if mini:
        model = os.environ.get("LLM_MODEL_MINI", defaults["model_mini"])
    else:
        model = os.environ.get("LLM_MODEL", defaults["model"])

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temp)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temp)

    # Unreachable due to validation in _get_provider, but satisfies type checkers
    raise ValueError(f"Unsupported provider: {provider}")
