"""OpenTelemetry tracing, off unless asked for.

What is worth tracing in a memory system is not latency. It is **what the write
path decided and why**: this turn recalled four memories, extracted one fact,
found it already known, and reinforced it instead of writing a fifth. A store
that has drifted is otherwise impossible to explain after the fact, because the
evidence — the sequence of decisions that produced it — is gone.

The structured logs and the per-turn ``TurnTrace`` already carry that. Spans add
the dimension neither has: the shape of a turn over time, and a way to see many
turns at once.

**Why OpenTelemetry rather than LangSmith or Langfuse.** LangSmith is close to
free effort for a LangGraph app, but it is a third-party SaaS, and shipping a
user's stored personal facts to one in a project whose entire subject is careful
handling of that data is the wrong trade. OTel is vendor-neutral: the same spans
go to Jaeger, a self-hosted Langfuse, or CloudWatch via ADOT, and none of it
requires an account.

**Off by default, and genuinely off.** With no provider configured the OTel API
returns a no-op tracer, so the instrumentation costs an attribute lookup and
nothing else. Nothing leaves the process unless an exporter is configured, which
for a system holding personal data is the only sane default.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

log = structlog.get_logger(__name__)

_TRACER_NAME = "engram"
_configured = False

try:  # pragma: no cover - depends on the optional extra
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    OTEL_AVAILABLE = False


def configure_tracing(settings: Settings) -> bool:
    """Set up a tracer provider if one is configured. Returns whether it did.

    Idempotent: the API's lifespan and a CLI entry point may both call it.
    """
    global _configured
    if _configured or settings.otel_exporter == "none":
        return False
    if not OTEL_AVAILABLE:
        log.warning(
            "tracing.unavailable",
            hint='OpenTelemetry is an optional extra: pip install -e ".[otel]"',
        )
        return False

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )

    if settings.otel_exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        # Simple, not batched: when you are watching spans in a terminal you
        # want them as they happen, not when a buffer flushes.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = settings.otel_endpoint or None
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
            )
        )

    trace.set_tracer_provider(provider)
    _configured = True
    log.info(
        "tracing.configured",
        exporter=settings.otel_exporter,
        endpoint=settings.otel_endpoint or "(default)",
    )
    return True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span, or do nothing at all if tracing was never configured.

    Deliberately tolerant: an observability failure must never be the reason a
    memory write fails. If the SDK is absent, or a provider was never set, this
    is a no-op context manager.
    """
    if not OTEL_AVAILABLE:
        yield None
        return

    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def set_attributes(current: Any, attributes: Mapping[str, Any]) -> None:
    """Add attributes to a span that may be None."""
    if current is None:
        return
    for key, value in attributes.items():
        if value is not None:
            current.set_attribute(key, value)
