"""The read and write paths against a real (in-process) LangGraph store."""

from __future__ import annotations

from datetime import timedelta

from engram.memory import MemoryReader, NaiveMemoryWriter
from engram.memory.write import MemoryWriter
from engram.schemas import Memory, Provenance, utcnow
from engram.store import Backend


def test_naive_writer_satisfies_the_writer_protocol(backend: Backend) -> None:
    """The seam a real memory manager drops into."""
    assert isinstance(NaiveMemoryWriter(backend.store), MemoryWriter)


def test_written_memories_come_back_from_recall(backend: Backend) -> None:
    writer = NaiveMemoryWriter(backend.store)
    reader = MemoryReader(backend.store, top_k=5)

    writer.write("alice", "I prefer pytest for everything I write.")
    recalls = reader.recall("alice", "what testing framework do I use?")

    assert [r.memory.text for r in recalls] == ["I prefer pytest for everything I write."]


def test_memories_are_isolated_per_user(backend: Backend) -> None:
    """A leak here would be a privacy incident, not a relevance bug."""
    writer = NaiveMemoryWriter(backend.store)
    reader = MemoryReader(backend.store, top_k=5)

    writer.write("alice", "I prefer pytest.")
    writer.write("bob", "I prefer unittest.")

    assert [r.memory.text for r in reader.recall("alice", "testing")] == ["I prefer pytest."]
    assert [r.memory.text for r in reader.recall("bob", "testing")] == ["I prefer unittest."]


def test_writer_records_provenance_and_thread(backend: Backend) -> None:
    written = NaiveMemoryWriter(backend.store).write(
        "alice", "a doc said something", thread_id="t1", source=Provenance.TOOL
    )
    assert written[0].source is Provenance.TOOL
    assert written[0].thread_id == "t1"


def test_writer_ignores_blank_turns(backend: Backend) -> None:
    assert NaiveMemoryWriter(backend.store).write("alice", "   ") == []


def test_recall_skips_expired_memories(backend: Backend) -> None:
    stale = Memory(
        user_id="alice",
        text="I am debugging a flaky test today",
        expires_at=utcnow() - timedelta(hours=1),
    )
    backend.store.put(stale.namespace(), stale.id, stale.to_value())

    reader = MemoryReader(backend.store, top_k=5)
    assert reader.recall("alice", "flaky test") == []
    # Still present in the store — expiry filters recall, it does not delete.
    assert len(reader.all_memories("alice")) == 1


def test_recall_skips_superseded_memories(backend: Backend) -> None:
    """A fact that has been replaced must never be surfaced again."""
    old = Memory(user_id="alice", text="I live in Mumbai", superseded_by="new-id")
    new = Memory(id="new-id", user_id="alice", text="I live in Berlin")
    for m in (old, new):
        backend.store.put(m.namespace(), m.id, m.to_value())

    texts = [r.memory.text for r in MemoryReader(backend.store).recall("alice", "where I live")]
    assert texts == ["I live in Berlin"]


def test_recall_respects_top_k(backend: Backend) -> None:
    writer = NaiveMemoryWriter(backend.store)
    for i in range(20):
        writer.write("alice", f"fact number {i} about deployment and testing")

    assert len(MemoryReader(backend.store, top_k=3).recall("alice", "testing")) == 3


def test_recall_overfetches_so_stale_matches_do_not_starve_results(
    backend: Backend,
) -> None:
    """The close matches are all superseded; a live one further down must survive.

    Without over-fetching, filtering after the store's own limit returns nothing.
    """
    for i in range(5):
        m = Memory(user_id="alice", text=f"deployment runbook {i}", superseded_by="x")
        backend.store.put(m.namespace(), m.id, m.to_value())
    live = Memory(user_id="alice", text="deployment happens with Terraform")
    backend.store.put(live.namespace(), live.id, live.to_value())

    recalls = MemoryReader(backend.store, top_k=2).recall("alice", "deployment")
    assert [r.memory.text for r in recalls] == ["deployment happens with Terraform"]


def test_blank_query_recalls_nothing(backend: Backend) -> None:
    NaiveMemoryWriter(backend.store).write("alice", "I prefer pytest.")
    assert MemoryReader(backend.store).recall("alice", "   ") == []
