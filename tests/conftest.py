"""Shared fixtures. Everything here is offline: stub model, in-process stores."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from engram.config import Settings
from engram.graph import Agent
from engram.store import Backend, open_backend


@pytest.fixture
def settings() -> Settings:
    """Deterministic, offline settings. Never reads the developer's .env."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        chat_provider="stub",
        embedder="hashing",
        store_backend="memory",
        recall_top_k=5,
    )


@pytest.fixture
def backend(settings: Settings) -> Iterator[Backend]:
    with open_backend(settings) as be:
        yield be


@pytest.fixture
def agent(settings: Settings, backend: Backend) -> Agent:
    return Agent(settings, checkpointer=backend.checkpointer, store=backend.store)
