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


def test_slot_text_renders_the_slot_as_searchable_words() -> None:
    """Underscores become spaces so the slot key matches ordinary phrasing."""
    m = Memory(user_id="u1", text="I use Neovim", attribute="testing_framework", value="pytest")
    assert m.slot_text() == "testing framework pytest"


def test_unslotted_memories_have_no_slot_text() -> None:
    assert Memory(user_id="u1", text="Always show the SQL").slot_text() == ""


def test_slot_text_is_written_to_the_store_for_indexing() -> None:
    m = Memory(user_id="u1", text="I use Neovim", attribute="editor", value="neovim")
    assert m.to_value()["slot_text"] == "editor neovim"


def test_unslotted_memories_omit_the_field_rather_than_writing_it_empty() -> None:
    """An empty string still embeds.

    Written as "", every unslotted memory would share one identical slot vector
    for some unlucky query to collide with. Absent is skipped by the index.
    """
    assert "slot_text" not in Memory(user_id="u1", text="Always show the SQL").to_value()


def test_the_derived_field_does_not_break_round_tripping() -> None:
    m = Memory(user_id="u1", text="I use Neovim", attribute="editor", value="neovim")
    assert Memory.from_value(m.to_value()) == m


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
