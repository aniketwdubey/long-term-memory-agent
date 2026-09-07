"""Backend wiring, including the Postgres path.

The Postgres tests are the ones that prove durability is real rather than an
in-process illusion, so they run against an actual pgvector database. They are
skipped unless ``ENGRAM_TEST_DSN`` is set, which keeps CI hermetic:

    make up
    ENGRAM_TEST_DSN=postgresql://engram:engram@localhost:5432/engram pytest -m postgres
"""

from __future__ import annotations

import os

import pytest

from engram.config import Settings
from engram.graph import Agent
from engram.store import open_backend

TEST_DSN = os.environ.get("ENGRAM_TEST_DSN")
needs_postgres = pytest.mark.skipif(not TEST_DSN, reason="ENGRAM_TEST_DSN is not set")


def test_memory_backend_indexes_at_the_embedders_dimensions() -> None:
    settings = Settings(_env_file=None, store_backend="memory", hashing_embed_dims=64)  # type: ignore[call-arg]
    with open_backend(settings) as backend:
        assert backend.store.index_config is not None
        assert backend.store.index_config["dims"] == 64
        # Both vectors: the fact text, and the slot it occupies.
        assert backend.store.index_config["fields"] == ["text", "slot_text"]


def test_the_local_dsn_is_used_when_no_host_is_configured() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.resolved_dsn == settings.postgres_dsn


def test_a_dsn_is_assembled_from_parts_when_a_host_is_set() -> None:
    """Deployed, the password arrives on its own from Secrets Manager.

    Baking it into a connection string would put it anywhere a DSN gets logged
    or echoed, so the parts are configured separately and joined here.
    """
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        postgres_host="db.internal",
        postgres_port=5432,
        postgres_user="engram",
        postgres_password="secret",
        postgres_db="engram",
    )
    assert settings.resolved_dsn == "postgresql://engram:secret@db.internal:5432/engram"


def test_credentials_are_url_encoded() -> None:
    """RDS generates passwords containing characters that break a raw URL."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, postgres_host="db.internal", postgres_password="p@ss/w#rd"
    )
    assert "p%40ss%2Fw%23rd" in settings.resolved_dsn
    assert "@db.internal" in settings.resolved_dsn


def test_unknown_backend_is_rejected() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with (
        pytest.raises(ValueError, match="Unknown store backend"),
        open_backend(settings.model_copy(update={"store_backend": "nope"})),
    ):
        pass


@pytest.fixture
def pg_settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        chat_provider="stub",
        embedder="hashing",
        store_backend="postgres",
        postgres_dsn=TEST_DSN or "",
    )


@pytest.mark.postgres
@needs_postgres
def test_memories_survive_a_new_process_level_connection(pg_settings: Settings) -> None:
    """Long-term memory outlives the agent that wrote it."""
    user = "durability-user"

    with open_backend(pg_settings) as backend:
        agent = Agent(pg_settings, checkpointer=backend.checkpointer, store=backend.store)
        for memory in agent.reader.all_memories(user):  # start from a clean slate
            backend.store.delete(memory.namespace(), memory.id)
        agent.chat(user, "session-one", "I prefer pytest for everything I write.")

    # Everything above is closed. A completely fresh connection must still see it.
    with open_backend(pg_settings) as backend:
        agent = Agent(pg_settings, checkpointer=backend.checkpointer, store=backend.store)
        reply = agent.chat(user, "session-two", "Scaffold me a test for the endpoint.").reply
        assert "pytest" in reply


@pytest.mark.postgres
@needs_postgres
def test_thread_transcripts_survive_a_new_connection(pg_settings: Settings) -> None:
    """Thread memory is durable too — that is what the checkpointer buys."""
    with open_backend(pg_settings) as backend:
        agent = Agent(pg_settings, checkpointer=backend.checkpointer, store=backend.store)
        agent.chat("durability-user", "persistent-thread", "remember this turn")

    with open_backend(pg_settings) as backend:
        agent = Agent(pg_settings, checkpointer=backend.checkpointer, store=backend.store)
        assert any("remember this turn" in line for line in agent.history("persistent-thread"))
