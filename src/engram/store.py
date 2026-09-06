"""Backend wiring: the two kinds of memory an agent needs, opened together.

LangGraph already draws the distinction this project is about, so we use its
primitives rather than inventing parallel ones:

* the **checkpointer** is *thread* memory — the running conversation, durable
  across process restarts. Nothing is extracted or judged; it is a transcript.
* the **store** is *long-term* memory — facts about a user, scoped to that user,
  retrieved by meaning rather than by recency, and shared across every thread
  that user ever opens.

``memory`` keeps both in-process (tests, quick demos, the stateless eval arm).
``postgres`` is the real deliverable: ``PostgresSaver`` for checkpoints and a
pgvector-backed ``PostgresStore`` for memories, both shipped by LangGraph — the
vector search is theirs, and the interesting custom code stays in the write path.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, NamedTuple, cast

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.store.memory import InMemoryStore

from engram.embeddings import build_embedder

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings


class Backend(NamedTuple):
    """What an agent runs against: both memories, and the embedder behind them.

    The embedder rides along because the store's vector index and the write
    path's dedupe check must use the *same* vectors, and because loading a
    sentence-embedding model twice per process is pure waste.
    """

    checkpointer: BaseCheckpointSaver  # type: ignore[type-arg]
    store: BaseStore
    embeddings: Embeddings


@contextmanager
def open_backend(settings: Settings, *, setup: bool = True) -> Iterator[Backend]:
    """Open the configured checkpointer and store as a context manager.

    ``setup`` runs the backend's schema migrations. They are idempotent, so a
    demo or a test can leave it on; a deployed service would run them once at
    release time instead of on every process start.
    """
    embedder = build_embedder(settings)
    # Index only the fact text. The rest of the record — provenance, expiry,
    # supersession — is metadata we filter on, and embedding it would blur the
    # vector that recall depends on.
    index: IndexConfig = {
        "dims": embedder.dims,
        "embed": embedder.embeddings,
        "fields": ["text"],
    }

    if settings.store_backend == "memory":
        yield Backend(
            checkpointer=InMemorySaver(),
            store=InMemoryStore(index=index),
            embeddings=embedder.embeddings,
        )
        return

    if settings.store_backend == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
        from langgraph.store.postgres.base import PostgresIndexConfig

        with ExitStack() as stack:
            checkpointer = stack.enter_context(
                PostgresSaver.from_conn_string(settings.postgres_dsn)
            )
            store = stack.enter_context(
                # PostgresIndexConfig only adds optional pgvector keys on top of
                # IndexConfig, but TypedDicts are invariant, so widen explicitly.
                PostgresStore.from_conn_string(
                    settings.postgres_dsn, index=cast(PostgresIndexConfig, index)
                )
            )
            if setup:
                checkpointer.setup()
                store.setup()
            yield Backend(checkpointer=checkpointer, store=store, embeddings=embedder.embeddings)
        return

    raise ValueError(f"Unknown store backend: {settings.store_backend!r}")
