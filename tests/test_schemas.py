"""The memory record: lifecycle, provenance, and store round-tripping."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from engram.schemas import (
    MEMORY_NAMESPACE,
    Memory,
    MemoryKind,
    Provenance,
    is_trusted,
    utcnow,
)


def test_defaults_are_a_durable_trusted_semantic_fact() -> None:
    m = Memory(user_id="u1", text="prefers pytest")
    assert m.kind is MemoryKind.SEMANTIC
    assert m.source is Provenance.USER
    assert m.expires_at is None
    assert m.superseded_by is None
    assert m.is_active()


def test_namespace_is_scoped_per_user() -> None:
    assert Memory(user_id="alice", text="x").namespace() == (MEMORY_NAMESPACE, "alice")
    assert Memory(user_id="bob", text="x").namespace() == (MEMORY_NAMESPACE, "bob")


def test_round_trips_through_the_store_representation() -> None:
    original = Memory(
        user_id="u1",
        text="lives in Berlin",
        kind=MemoryKind.EPISODIC,
        source=Provenance.TOOL,
        expires_at=utcnow() + timedelta(days=1),
        thread_id="t9",
    )
    assert Memory.from_value(original.to_value()) == original


def test_expired_memories_are_not_active() -> None:
    expired = Memory(
        user_id="u1",
        text="debugging a flaky test today",
        expires_at=utcnow() - timedelta(seconds=1),
    )
    assert not expired.is_active()

    future = Memory(user_id="u1", text="on call this week", expires_at=utcnow() + timedelta(days=1))
    assert future.is_active()


def test_superseded_memories_are_not_active_even_without_an_expiry() -> None:
    """Conflict resolution keeps history; history must not be recalled."""
    old = Memory(user_id="u1", text="lives in Mumbai", superseded_by="abc123")
    assert old.expires_at is None
    assert not old.is_active()


def test_provenance_decides_what_may_write_to_user_memory() -> None:
    assert is_trusted(Provenance.USER)
    assert is_trusted(Provenance.AGENT)
    # The injection gate turns on exactly this: text the agent merely read is
    # never allowed to become a fact about the user.
    assert not is_trusted(Provenance.TOOL)
    assert not is_trusted(Provenance.DOCUMENT)


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Memory(user_id="u1", text="")


def test_importance_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Memory(user_id="u1", text="x", importance=1.5)
