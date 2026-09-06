"""The write-path contract, and the naive baseline it replaced.

Retrieval-augmented generation only ever *reads*. A memory system must also
decide what deserves to be stored at all, notice that a candidate fact is
something it already knows, work out whether a new fact updates an old one or
merely coexists with it, judge how long it should survive, and refuse to store
things that were never the user's to say. Those decisions are where memory
systems succeed or fail.

:class:`MemoryWriter` is that contract. :class:`NaiveMemoryWriter` is the
baseline — it stores each user turn verbatim, which is the obvious thing to
build first and is wrong in three measurable ways:

* **precision** — a whole turn is not a fact. "I'm on the payments team and I
  prefer pytest" is stored as one lump, alongside "sounds good, thanks", so the
  store fills with text that will never usefully answer anything.
* **duplication** — a user who mentions their team in five sessions gets five
  near-identical memories, all competing for the same top-k slots.
* **contradiction** — when the user moves from Mumbai to Berlin, both facts are
  stored and both are recalled. The agent is now *less* reliable than with no
  memory at all, because it will confidently surface the stale one.

It is kept, not deleted, because the benchmark runs it as an arm: the manager's
improvement is a measured delta rather than a claim. See
:mod:`engram.memory.manager` for the implementation that replaces it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

import structlog
from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict, Field

from engram.schemas import Memory, MemoryKind, Provenance

log = structlog.get_logger(__name__)


class MemoryOp(StrEnum):
    """What was decided about one candidate fact."""

    WRITE = "write"
    DEDUPE = "dedupe"
    SUPERSEDE = "supersede"


class MemoryDecision(BaseModel):
    """One write-path decision, recorded so a turn can be explained.

    "Which memories were retrieved, written, updated, forgotten" is the
    observability unit of a memory system; without it, a store that drifts is
    impossible to debug after the fact.
    """

    model_config = ConfigDict(frozen=True)

    op: MemoryOp
    candidate: str
    memory_id: str | None = None
    superseded_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class WriteReport(BaseModel):
    """Everything that happened on one turn's write path."""

    model_config = ConfigDict(frozen=True)

    written: list[Memory] = Field(default_factory=list)
    decisions: list[MemoryDecision] = Field(default_factory=list)


@runtime_checkable
class MemoryWriter(Protocol):
    """The seam between "something was said" and "something is remembered"."""

    def apply(
        self,
        user_id: str,
        text: str,
        *,
        thread_id: str | None = None,
        source: Provenance = Provenance.USER,
    ) -> WriteReport:
        """Consider ``text`` for storage; report what was decided and written.

        Returning a report rather than a memory is deliberate: one turn may
        yield several facts, none at all, or changes to records it did not
        write — a supersession is a write-path outcome with no new memory of
        its own to return.
        """
        ...


class NaiveMemoryWriter:
    """Stores every user turn verbatim. The baseline the manager must beat.

    Note what it does *not* do with ``source``: it records provenance faithfully
    but does not act on it, so untrusted content becomes a memory like anything
    else. That is the vulnerable baseline the injection gate is measured
    against; it is not a gap left by accident.
    """

    def __init__(self, store: BaseStore) -> None:
        self._store = store

    def apply(
        self,
        user_id: str,
        text: str,
        *,
        thread_id: str | None = None,
        source: Provenance = Provenance.USER,
    ) -> WriteReport:
        text = text.strip()
        if not text:
            return WriteReport()

        memory = Memory(
            user_id=user_id,
            text=text,
            kind=MemoryKind.SEMANTIC,
            source=source,
            thread_id=thread_id,
            evidence=text,
        )
        self._store.put(memory.namespace(), memory.id, memory.to_value())

        log.debug(
            "memory.write.naive",
            user_id=user_id,
            memory_id=memory.id,
            source=source.value,
            chars=len(text),
        )
        return WriteReport(
            written=[memory],
            decisions=[MemoryDecision(op=MemoryOp.WRITE, candidate=text, memory_id=memory.id)],
        )
