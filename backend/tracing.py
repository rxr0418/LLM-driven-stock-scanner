"""
tracing.py - Langfuse tracing for the swing pipeline.

One trace per ticker per scan. Each agent gets a span.
Gracefully no-ops when Langfuse keys are absent.

Usage:
    from tracing import tracer
    with tracer.ticker_trace("NVDA", regime="TRENDING") as trace:
        with trace.span("search_agent") as span:
            result = run_search(...)
            span.end(output=result)
"""

import os
from contextlib import contextmanager
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Langfuse init — optional, no-ops if keys missing
# ─────────────────────────────────────────────────────────────

_langfuse = None
LANGFUSE_AVAILABLE = False

try:
    from langfuse import Langfuse
    _pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    _sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    _host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    if _pk and _sk:
        _langfuse = Langfuse(public_key=_pk, secret_key=_sk, host=_host)
        LANGFUSE_AVAILABLE = True
        log.info("Langfuse tracing enabled", extra={"host": _host})
    else:
        log.info("Langfuse keys not set — tracing disabled")
except Exception as e:
    log.warning("Langfuse init failed — tracing disabled", extra={"error": str(e)})


# ─────────────────────────────────────────────────────────────
# Span wrapper
# ─────────────────────────────────────────────────────────────

class _Span:
    """Wraps a Langfuse span or is a no-op."""

    def __init__(self, span=None, name: str = ""):
        self._span = span
        self.name  = name

    def end(self, output=None, metadata: Optional[dict] = None, level: str = "DEFAULT") -> None:
        if self._span:
            try:
                self._span.end(output=output, metadata=metadata, level=level)
            except Exception:
                pass

    def event(self, name: str, input=None, output=None, metadata: Optional[dict] = None) -> None:
        if self._span:
            try:
                self._span.event(name=name, input=input, output=output, metadata=metadata)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# Trace wrapper
# ─────────────────────────────────────────────────────────────

class _Trace:
    def __init__(self, trace=None):
        self._trace = trace

    @contextmanager
    def span(self, name: str, input=None, metadata: Optional[dict] = None):
        """Context manager for one agent span."""
        lf_span = None
        if self._trace:
            try:
                lf_span = self._trace.span(name=name, input=input, metadata=metadata)
            except Exception:
                pass
        s = _Span(span=lf_span, name=name)
        try:
            yield s
        except Exception as exc:
            if lf_span:
                try:
                    lf_span.end(level="ERROR", metadata={"error": str(exc)})
                except Exception:
                    pass
            raise
        # Caller is responsible for calling span.end() with output

    def score(self, name: str, value: float, comment: str = "") -> None:
        if self._trace:
            try:
                self._trace.score(name=name, value=value, comment=comment)
            except Exception:
                pass

    def update(self, output=None, metadata: Optional[dict] = None) -> None:
        if self._trace:
            try:
                self._trace.update(output=output, metadata=metadata)
            except Exception:
                pass

    def flush(self) -> None:
        if _langfuse:
            try:
                _langfuse.flush()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

class SwingTracer:
    """Entry point for swing pipeline tracing."""

    @contextmanager
    def ticker_trace(
        self,
        ticker: str,
        regime: str = "",
        signal_direction: str = "",
        factor_score: float = 0.0,
        scan_date: str = "",
    ):
        """
        Open a top-level Langfuse trace for one ticker.

        Usage:
            with tracer.ticker_trace("NVDA", regime="TRENDING") as trace:
                with trace.span("search_agent") as span:
                    ...
                    span.end(output=result)
        """
        lf_trace = None
        if _langfuse:
            try:
                lf_trace = _langfuse.trace(
                    name=f"swing/{ticker}",
                    input={
                        "ticker": ticker,
                        "regime": regime,
                        "signal_direction": signal_direction,
                        "factor_score": factor_score,
                        "scan_date": scan_date,
                    },
                    tags=["swing", regime, signal_direction],
                )
            except Exception as e:
                log.warning("trace creation failed", extra={"ticker": ticker, "error": str(e)})

        t = _Trace(trace=lf_trace)
        try:
            yield t
        finally:
            if lf_trace:
                try:
                    lf_trace.update(metadata={"ticker": ticker})
                except Exception:
                    pass
            t.flush()


tracer = SwingTracer()
