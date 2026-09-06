"""The FastAPI application.

The backend is opened once for the process lifetime rather than per request.
That matters more here than it would for a stateless service: with the Postgres
backend it is a connection pool and a schema migration, and doing either on
every turn would dominate the latency of the thing it is serving.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI

from engram.api.routes import router
from engram.config import Settings, get_settings
from engram.graph import Agent
from engram.logging import configure_logging
from engram.store import open_backend

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    configure_logging(settings)

    with ExitStack() as stack:
        backend = stack.enter_context(open_backend(settings))
        app.state.agent = Agent(
            settings,
            checkpointer=backend.checkpointer,
            store=backend.store,
            embeddings=backend.embeddings,
        )
        log.info(
            "api.started",
            store=settings.store_backend,
            writer=settings.memory_writer,
            provider=settings.chat_provider,
            embedder=settings.embedder,
        )
        yield
    log.info("api.stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings to keep it offline."""
    app = FastAPI(
        title="engram",
        version="0.1.0",
        summary="Long-term memory for conversational agents.",
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, Any]:
        active = getattr(app.state, "settings", None) or get_settings()
        return {
            "status": "ok",
            "store": active.store_backend,
            "writer": active.memory_writer,
            "provider": active.chat_provider,
        }

    app.include_router(router)
    return app


app = create_app()
