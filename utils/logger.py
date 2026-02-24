"""Structured logging for the patent-search pipeline.

Usage
-----
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.debug("fetched record", extra={"patent_id": pid})
    log.info("search stage complete", extra={"count": len(results)})

Convention:
  DEBUG — individual records / low-level detail
  INFO  — stage-level summaries (start, finish, counts)
  WARNING / ERROR — unexpected conditions
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    # Fields to pull from the LogRecord that are always present
    _RESERVED = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Base payload
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach any extra fields passed via extra={...}
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value

        # Append exception info if present
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _build_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    return handler


# Module-level cache so we don't add duplicate handlers
_configured_names: set[str] = set()


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Return a JSON-structured logger for *name*.

    The root logger is left untouched; only the named logger is configured,
    which avoids duplicate output if the caller also configures the root.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    level:
        Minimum log level; defaults to DEBUG so all messages are visible.
        Override with ``logging.INFO`` in production to reduce noise.
    """
    logger = logging.getLogger(name)

    if name not in _configured_names:
        logger.setLevel(level)
        logger.addHandler(_build_handler())
        # Prevent propagation to the root logger to avoid double-printing
        logger.propagate = False
        _configured_names.add(name)

    return logger
