"""Structured logging setup.

Memory operations are the thing worth tracing in this system — which memories
were recalled for a turn, which were written, and (later) which were updated or
forgotten. Structured events make that greppable instead of prose.
"""

from __future__ import annotations

import logging

import structlog

from engram.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog once, at process start."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=level)

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
