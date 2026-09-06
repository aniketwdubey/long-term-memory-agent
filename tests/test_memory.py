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

    writer.apply("alice", "I prefer pytest for everything I write.")
    recalls = reader.recall("alice", "what testing framework do I use?")

    assert [r.memory.text for r in recalls] == ["I prefer pytest for everything I write."]


def test_memories_are_isolated_per_user(backend: Backend) -> None:
    """A leak here would be a privacy incident, not a relevance bug."""
    writer = NaiveMemoryWriter(backend.store)
    reader = MemoryReader(backend.store, top_k=5)

    writer.apply("alice", "I prefer pytest.")
    writer.apply("bob", "I prefer unittest.")

    assert [r.memory.text for r in reader.recall("alice", "testing")] == ["I prefer pytest."]
    assert [r.memory.text for r in reader.recall("bob", "testing")] == ["I prefer unittest."]


def test_writer_records_provenance_and_thread(backend: Backend) -> None:
    written = (
        NaiveMemoryWriter(backend.store)
        .apply("alice", "a doc said something", thread_id="t1", source=Provenance.TOOL)
        .written
    )
    assert written[0].source is Provenance.TOOL
    assert written[0].thread_id == "t1"


def test_writer_ignores_blank_turns(backend: Backend) -> None:
    assert NaiveMemoryWriter(backend.store).apply("alice", "   ").written == []


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
        writer.apply("alice", f"fact number {i} about deployment and testing")

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


def test_a_query_naming_the_slot_finds_a_fact_it_shares_no_words_with(
    backend: Backend,
) -> None:
    """The read path cashing in what the write path worked out.

    "How should I set up my editor?" and "I use Neovim" have no vocabulary in
    common, so text-only retrieval buries the answer under chatter. The slot key
    `editor` is the bridge — and it exists only because extraction produced it.
    """
    for text, attribute, value in [
        ("I use Neovim", "editor", "neovim"),
        ("I had a great coffee this morning near the office", "", ""),
        ("We finally shipped the dashboard on Friday", "", ""),
        ("My sister is visiting from Toronto", "", ""),
    ]:
        m = Memory(user_id="alice", text=text, attribute=attribute, value=value)
        backend.store.put(m.namespace(), m.id, m.to_value())

    recalls = MemoryReader(backend.store, top_k=1).recall(
        "alice", "How should I set up my editor for this project?"
    )
    assert [r.memory.text for r in recalls] == ["I use Neovim"]


def test_an_unslotted_store_gets_no_benefit_from_slot_search(backend: Backend) -> None:
    """Pins where the gain comes from.

    The same fact stored verbatim by the naive writer — no slot — is not found
    by the same query. The retrieval improvement is not free: it is paid for by
    the write path bothering to work out what the fact is about.
    """
    writer = NaiveMemoryWriter(backend.store)
    writer.apply("bob", "I use Neovim")
    for chatter in (
        "I had a great coffee this morning near the office",
        "We finally shipped the dashboard on Friday",
        "My sister is visiting from Toronto",
    ):
        writer.apply("bob", chatter)

    recalls = MemoryReader(backend.store, top_k=1).recall(
        "bob", "How should I set up my editor for this project?"
    )
    assert [r.memory.text for r in recalls] != ["I use Neovim"]


def test_blank_query_recalls_nothing(backend: Backend) -> None:
    NaiveMemoryWriter(backend.store).apply("alice", "I prefer pytest.")
    assert MemoryReader(backend.store).recall("alice", "   ") == []
