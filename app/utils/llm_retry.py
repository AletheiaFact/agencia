"""Retry-with-truncation utility for LLM calls that may exceed token limits.

When an LLM call fails due to token limits, this module provides helpers
that automatically truncate the largest text field and retry, up to a
configurable number of attempts.
"""

import logging
import re
from typing import Any

from openai import BadRequestError

logger = logging.getLogger(__name__)

# Pattern to detect token-limit errors in OpenAI responses
_TOKEN_LIMIT_PATTERN = re.compile(
    r"(max.tokens|token.limit|context.length|too many tokens)",
    re.IGNORECASE,
)


def is_token_limit_error(exc: Exception) -> bool:
    """Check if an exception is a token-limit error from OpenAI."""
    if not isinstance(exc, BadRequestError):
        return False
    msg = str(exc).lower()
    return bool(
        "token" in msg
        and ("max" in msg or "limit" in msg or "exceed" in msg or "requested" in msg)
    )


def invoke_with_retry(
    chain,
    params: dict[str, Any],
    truncatable_keys: list[str],
    max_retries: int = 3,
    shrink_factor: float = 0.5,
    min_length: int = 500,
) -> str:
    """Invoke a LangChain chain with automatic truncation retry on token-limit errors.

    On each retry, the largest truncatable field is shrunk by `shrink_factor`.
    This continues until the call succeeds or `max_retries` is exhausted.

    Args:
        chain: A LangChain chain (prompt | llm | parser).
        params: Dict of parameters to pass to chain.invoke().
        truncatable_keys: List of param keys whose values can be truncated.
        max_retries: Maximum number of retry attempts (default 3).
        shrink_factor: Multiply current length by this on each retry (default 0.5).
        min_length: Stop truncating below this char count (default 500).

    Returns:
        The chain output string.

    Raises:
        The original exception if all retries are exhausted.
    """
    attempt = 0
    last_error = None
    current_params = dict(params)

    while attempt <= max_retries:
        try:
            return chain.invoke(current_params)
        except (BadRequestError, Exception) as exc:
            if not is_token_limit_error(exc):
                raise

            attempt += 1
            last_error = exc

            if attempt > max_retries:
                logger.error(
                    "[llm_retry] All %d retries exhausted, raising error", max_retries
                )
                raise

            # Find the largest truncatable field and shrink it
            largest_key = None
            largest_len = 0
            for key in truncatable_keys:
                val = current_params.get(key, "")
                if isinstance(val, str) and len(val) > largest_len:
                    largest_key = key
                    largest_len = len(val)

            if largest_key is None or largest_len <= min_length:
                logger.error(
                    "[llm_retry] No field left to truncate (largest=%d, min=%d)",
                    largest_len, min_length,
                )
                raise

            new_len = max(int(largest_len * shrink_factor), min_length)
            current_params[largest_key] = current_params[largest_key][:new_len]

            logger.warning(
                "[llm_retry] Token limit hit (attempt %d/%d). "
                "Truncating '%s' from %d to %d chars and retrying.",
                attempt, max_retries, largest_key, largest_len, new_len,
            )

    raise last_error
