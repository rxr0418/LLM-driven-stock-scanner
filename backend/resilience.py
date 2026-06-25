"""
resilience.py - Retry, backoff, and timeout utilities.

Provides:
  - with_retry()     : decorator for functions that call external APIs
  - ticker_timeout() : async context manager for per-ticker deadline
  - safe_call()      : synchronous try/except wrapper that logs and returns fallback

Retry strategy (exponential backoff with jitter):
  Attempt 1: immediate
  Attempt 2: ~2s
  Attempt 3: ~4s
  Attempt 4: ~8s (max)
  Gives up after MAX_ATTEMPTS, re-raises the last exception.

Retryable errors:
  - anthropic.RateLimitError      (429)
  - anthropic.APIStatusError 529  (overload)
  - anthropic.APIConnectionError  (network)
  - anthropic.InternalServerError (500/503)
  - requests / httpx transient errors
  - yfinance / urllib3 connection errors
"""

import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional, Type

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger("scanner.resilience")

# ─────────────────────────────────────────────────────────────
# Retryable exception predicate
# ─────────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors that are safe to retry."""
    try:
        import anthropic
        if isinstance(exc, anthropic.RateLimitError):
            return True
        if isinstance(exc, anthropic.APIConnectionError):
            return True
        if isinstance(exc, anthropic.InternalServerError):
            return True
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code in {429, 500, 502, 503, 529}:
            return True
    except ImportError:
        pass

    # Network-level errors (yfinance, requests, urllib3)
    type_name = type(exc).__name__
    retryable_names = {
        "ConnectionError", "TimeoutError", "ReadTimeout",
        "ConnectTimeout", "ChunkedEncodingError", "RemoteDisconnected",
        "IncompleteRead", "ProtocolError",
    }
    if type_name in retryable_names:
        return True

    # urllib3 / requests
    exc_module = getattr(type(exc), "__module__", "")
    if "urllib3" in exc_module or "requests" in exc_module:
        return True

    return False


# ─────────────────────────────────────────────────────────────
# Retry decorator
# ─────────────────────────────────────────────────────────────

MAX_ATTEMPTS    = 4
WAIT_MIN        = 1.0   # seconds
WAIT_MAX        = 10.0  # seconds
WAIT_JITTER     = 2.0   # ± jitter


def with_retry(
    max_attempts: int = MAX_ATTEMPTS,
    wait_min: float = WAIT_MIN,
    wait_max: float = WAIT_MAX,
    label: str = "",
):
    """
    Decorator: retry a function on transient errors with exponential backoff.

    Usage:
        @with_retry(label="anthropic")
        def call_claude(...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=wait_min, max=wait_max, jitter=wait_jitter),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# tenacity's wait_exponential_jitter takes `jitter` not `wait_jitter`
# fix the import alias
from tenacity import wait_exponential_jitter as _wait_exp_jitter


def with_retry(
    max_attempts: int = MAX_ATTEMPTS,
    wait_min: float = WAIT_MIN,
    wait_max: float = WAIT_MAX,
    jitter: float = WAIT_JITTER,
    label: str = "",
):
    def decorator(fn: Callable) -> Callable:
        retrying = retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_attempts),
            wait=_wait_exp_jitter(initial=wait_min, max=wait_max, jitter=jitter),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

        @retrying
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────
# Per-ticker async timeout
# ─────────────────────────────────────────────────────────────

TICKER_TIMEOUT_SECONDS = 120   # max wall time per ticker in orchestrator


@asynccontextmanager
async def ticker_timeout(ticker: str, seconds: float = TICKER_TIMEOUT_SECONDS):
    """
    Async context manager that cancels the enclosed block after `seconds`.

    Usage:
        async with ticker_timeout("NVDA"):
            state, decision = await run_ticker_async(...)
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        logger.error(
            "ticker timed out",
            extra={"ticker": ticker, "timeout_s": seconds},
        )
        raise


# ─────────────────────────────────────────────────────────────
# Safe call wrapper (sync)
# ─────────────────────────────────────────────────────────────

def safe_call(
    fn: Callable,
    *args,
    fallback: Any = None,
    label: str = "",
    log_level: int = logging.ERROR,
    **kwargs,
) -> Any:
    """
    Call fn(*args, **kwargs), return fallback on any exception.
    Logs the error but does not propagate.

    Usage:
        events = safe_call(get_upcoming_events, ticker, fallback=[], label="events")
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.log(
            log_level,
            f"safe_call failed: {label or fn.__name__}: {exc}",
            exc_info=True,
        )
        return fallback
