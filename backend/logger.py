"""
logger.py - Central structured logger for the stock scanner.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("scan started", extra={"ticker": "NVDA", "regime": "TRENDING"})

Format:
  - Production (LOG_FORMAT=json): JSON lines — parseable by Railway / Datadog
  - Development (default): human-readable with color-coded levels
"""

import logging
import os
import sys
import json
from datetime import datetime, timezone


LOG_LEVEL  = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("LOG_FORMAT", "text")   # "json" in production


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        # Attach any extra fields passed via extra={}
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                payload[key] = val

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    """Human-readable format with module context."""
    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        ts    = datetime.now(timezone.utc).strftime("%H:%M:%S")
        base  = f"{color}[{record.levelname[0]}]{self.RESET} {ts} [{record.name}] {record.getMessage()}"

        # Append extra fields inline
        extras = []
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, val in record.__dict__.items():
            if key not in skip:
                extras.append(f"{key}={val!r}")
        if extras:
            base += "  " + " ".join(extras)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    if LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_TextFormatter())
    return handler


# Root scanner logger — all child loggers inherit this handler
_root = logging.getLogger("scanner")
_root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if not _root.handlers:
    _root.addHandler(_build_handler())
_root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'scanner' namespace.

    Args:
        name: typically __name__, e.g. 'swing.agents.orchestrator'
               Strip 'backend.' prefix if present.
    """
    # Normalise: 'backend.swing.foo' → 'scanner.swing.foo'
    clean = name.removeprefix("backend.").replace(".", ".")
    return _root.getChild(clean)
