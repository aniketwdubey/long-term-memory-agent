"""The memory manager: dedupe, conflict resolution, decay.

Every decision here is policy code over structured facts, so these tests drive
the manager with hand-built candidates rather than through the extractor — that
keeps a failure attributable to the policy rather than to a regex.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from engram.embeddings import HashingEmbeddings
from engram.memory.extract import CandidateFact
from engram.memory.manager import MemoryManager
from engram.memory.read import MemoryReader
from engram.memory.write import MemoryOp, MemoryWriter
from engram.schemas import Memory, MemoryKind, Provenance, utcnow
from engram.store import Backend


class ScriptedExtractor:
    """Returns pre-built candidates, one list per call."""

    def __init__(self, *batches: list[CandidateFact]) -> None:
        self._batches = list(batches)

    def extract(self, turn: str) -> list[CandidateFact]:
        return self._batches.pop(0) if self._batches else []


def fact(text: str, attribute: str = "", value: str = "", **kw: object) -> CandidateFact:
    return CandidateFact(text=text, attribute=attribute, value=value, **kw)  # type: ignore[arg-type]


def manager(backend: Backend, *batches: list[CandidateFact]) -> MemoryManager:
    return MemoryManager(
        backend.store,
        extractor=ScriptedExtractor(*batches),
        embeddings=HashingEmbeddings(256),
    )


def active(backend: Backend, user: str = "alice") -> list[Memory]:
    return [m for m in MemoryReader(backend.store).all_memories(user) if m.is_active()]


def retired(backend: Backend, user: str = "alice") -> list[Memory]:
    return [m for m in MemoryReader(backend.store).all_memories(user) if not m.is_active()]


def test_satisfies_the_writer_protocol(backend: Backend) -> None:
    assert isinstance(manager(backend), MemoryWriter)


# --- writing -----------------------------------------------------------------


def test_writes_a_new_fact(backend: Backend) -> None:
    m = manager(backend, [fact("Prefers pytest", "testing_framework", "pytest")])
    report = m.apply("alice", "I prefer pytest")

    assert [d.op for d in report.decisions] == [MemoryOp.WRITE]
    assert [x.value for x in active(backend)] == ["pytest"]


def test_a_turn_with_no_facts_writes_nothing(backend: Backend) -> None:
    """The common case — most turns are chatter."""
    m = manager(backend, [])
    assert m.apply("alice", "the coffee machine is broken").written == []
    assert active(backend) == []


def test_a_blank_turn_is_ignored(backend: Backend) -> None:
    assert manager(backend).apply("alice", "   ").written == []


def test_several_facts_from_one_turn_are_all_written(backend: Backend) -> None:
    m = manager(
        backend, [fact("On payments", "team", "payments"), fact("Prefers pytest", "tf", "pytest")]
    )
    assert len(m.apply("alice", "...").written) == 2


# --- dedupe ------------------------------------------------------------------


def test_restating_a_known_fact_writes_nothing(backend: Backend) -> None:
    """Five mentions of the same team should leave one memory, not five."""
    f = fact("On the payments team", "team", "payments")
    m = manager(backend, [f], [f], [f])
    for _ in range(3):
        m.apply("alice", "I'm on the payments team")

    assert len(active(backend)) == 1


def test_dedupe_is_reported_and_reinforces_the_original(backend: Backend) -> None:
    """A fact restated is a fact that matters — bump it rather than copy it."""
    f = fact("On the payments team", "team", "payments", importance=0.5)
    m = manager(backend, [f], [f])
    first = m.apply("alice", "...").written[0]
    report = m.apply("alice", "...")

    assert [d.op for d in report.decisions] == [MemoryOp.DEDUPE]
    assert report.written == []
    survivor = active(backend)[0]
    assert survivor.id == first.id
    assert survivor.importance > first.importance


def test_dedupe_ignores_case_and_spacing_in_the_slot_value(backend: Backend) -> None:
    m = manager(
        backend,
        [fact("Runs Postgres", "database", "postgres")],
        [fact("Runs Postgres", "database", "  Postgres  ")],
    )
    m.apply("alice", "...")
    m.apply("alice", "...")
    assert len(active(backend)) == 1


def test_unslotted_near_duplicates_are_deduped_by_similarity(backend: Backend) -> None:
    """Facts the extractor could not slot still must not accumulate copies."""
    text = "Always show me the SQL before running it"
    m = manager(backend, [fact(text, kind=MemoryKind.PROCEDURAL)], [fact(text)])
    m.apply("alice", "...")
    report = m.apply("alice", "...")

    assert [d.op for d in report.decisions] == [MemoryOp.DEDUPE]
    assert len(active(backend)) == 1


def test_distinct_unslotted_facts_are_both_kept(backend: Backend) -> None:
    m = manager(
        backend,
        [fact("Always show the SQL before running it")],
        [fact("Never deploy on a Friday afternoon")],
    )
    m.apply("alice", "...")
    m.apply("alice", "...")
    assert len(active(backend)) == 2


def test_a_fact_stated_twice_in_one_turn_is_stored_once(backend: Backend) -> None:
    """Candidates within a turn must see each other, not just the store."""
    f = fact("On the payments team", "team", "payments")
    m = manager(backend, [f, f])
    report = m.apply("alice", "I'm on payments. Yes, payments.")

    assert [d.op for d in report.decisions] == [MemoryOp.WRITE, MemoryOp.DEDUPE]
    assert len(active(backend)) == 1


# --- conflict resolution -----------------------------------------------------


def test_a_contradiction_supersedes_rather_than_coexisting(backend: Backend) -> None:
    """The failure this prevents: an agent confidently answering with a stale fact."""
    m = manager(
        backend,
        [fact("On the payments team", "team", "payments")],
        [fact("On the platform team", "team", "platform")],
    )
    old = m.apply("alice", "I'm on payments").written[0]
    report = m.apply("alice", "I switched to platform")

    assert [d.op for d in report.decisions] == [MemoryOp.SUPERSEDE]
    assert [x.value for x in active(backend)] == ["platform"]
    assert report.decisions[0].superseded_ids == [old.id]


def test_the_superseded_fact_is_kept_as_history(backend: Backend) -> None:
    """Supersede *with history* — the old record is retired, never deleted."""
    m = manager(
        backend,
        [fact("Lives in Mumbai", "location", "mumbai")],
        [fact("Lives in Berlin", "location", "berlin")],
    )
    m.apply("alice", "...")
    new = m.apply("alice", "...").written[0]

    stale = retired(backend)
    assert [x.value for x in stale] == ["mumbai"]
    assert stale[0].superseded_by == new.id
    assert stale[0].superseded_at is not None


def test_superseded_facts_are_never_recalled(backend: Backend) -> None:
    m = manager(
        backend,
        [fact("Lives in Mumbai", "location", "mumbai")],
        [fact("Lives in Berlin", "location", "berlin")],
    )
    m.apply("alice", "...")
    m.apply("alice", "...")

    recalls = MemoryReader(backend.store).recall("alice", "where does the user live")
    assert [r.memory.value for r in recalls] == ["berlin"]


def test_facts_in_different_slots_do_not_conflict(backend: Backend) -> None:
    m = manager(
        backend,
        [fact("On payments", "team", "payments")],
        [fact("Prefers pytest", "testing_framework", "pytest")],
    )
    m.apply("alice", "...")
    m.apply("alice", "...")
    assert len(active(backend)) == 2


def test_scope_lets_two_values_of_one_attribute_coexist(backend: Backend) -> None:
    """A work address and a home address are not a contradiction."""
    m = manager(
        backend,
        [fact("Works in Berlin", "location", "berlin", scope="work")],
        [fact("Lives in Potsdam", "location", "potsdam", scope="home")],
    )
    m.apply("alice", "...")
    report = m.apply("alice", "...")

    assert [d.op for d in report.decisions] == [MemoryOp.WRITE]
    assert {x.value for x in active(backend)} == {"berlin", "potsdam"}


def test_a_contradiction_retires_every_stale_value_in_the_slot(backend: Backend) -> None:
    m = manager(
        backend,
        [fact("Lives in Mumbai", "location", "mumbai")],
        [fact("Lives in Berlin", "location", "berlin")],
        [fact("Lives in Lisbon", "location", "lisbon")],
    )
    for _ in range(3):
        m.apply("alice", "...")

    assert [x.value for x in active(backend)] == ["lisbon"]
    assert {x.value for x in retired(backend)} == {"mumbai", "berlin"}


# --- decay -------------------------------------------------------------------


def test_a_ttl_becomes_an_expiry(backend: Backend) -> None:
    m = manager(backend, [fact("Debugging a flaky test", ttl_days=7)])
    written = m.apply("alice", "...").written[0]

    assert written.expires_at is not None
    assert written.expires_at > utcnow() + timedelta(days=6)


def test_durable_facts_have_no_expiry(backend: Backend) -> None:
    m = manager(backend, [fact("Prefers pytest", "testing_framework", "pytest")])
    assert m.apply("alice", "...").written[0].expires_at is None


def test_an_expired_memory_stops_blocking_its_slot(backend: Backend) -> None:
    """Decay frees the slot: once a transient fact lapses, a new one just writes."""
    stale = Memory(
        user_id="alice",
        text="On call this week",
        attribute="oncall",
        value="yes",
        expires_at=utcnow() - timedelta(days=1),
    )
    backend.store.put(stale.namespace(), stale.id, stale.to_value())

    m = manager(backend, [fact("Not on call", "oncall", "no")])
    report = m.apply("alice", "...")

    assert [d.op for d in report.decisions] == [MemoryOp.WRITE]
    assert [x.value for x in active(backend)] == ["no"]


# --- provenance --------------------------------------------------------------


def test_provenance_is_recorded_on_every_write(backend: Backend) -> None:
    """Nothing acts on this yet — the injection gate is slice 3 — but a record
    written without it could never be gated retroactively."""
    m = manager(backend, [fact("Something a document claimed")])
    written = m.apply("alice", "...", source=Provenance.TOOL, thread_id="t9").written[0]

    assert written.source is Provenance.TOOL
    assert written.thread_id == "t9"


def test_memories_stay_scoped_to_one_user(backend: Backend) -> None:
    f = fact("On the payments team", "team", "payments")
    m = manager(backend, [f], [f])
    m.apply("alice", "...")
    m.apply("bob", "...")

    # Same slot, same value, two users: two memories, no cross-user dedupe.
    assert len(active(backend, "alice")) == 1
    assert len(active(backend, "bob")) == 1


@pytest.mark.parametrize("op", list(MemoryOp))
def test_every_decision_names_the_candidate_it_was_about(backend: Backend, op: MemoryOp) -> None:
    """Traces have to be readable after the fact, not just present."""
    f = fact("On the payments team", "team", "payments")
    other = fact("On the platform team", "team", "platform")
    m = manager(backend, [f], [f], [other])

    reports = [m.apply("alice", "...") for _ in range(3)]
    decisions = [d for r in reports for d in r.decisions]
    match = next(d for d in decisions if d.op is op)
    assert match.candidate
