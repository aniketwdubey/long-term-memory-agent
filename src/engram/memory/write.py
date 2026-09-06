"""The write path — the part RAG does not have, and the part that is hard.

Retrieval-augmented generation only ever *reads*. A memory system must also
decide what deserves to be stored at all, notice that a candidate fact is
something it already knows, work out whether a new fact updates an old one or
merely coexists with it, judge how long it should survive, and refuse to store
things that were never the user's to say. Those decisions are where memory
systems succeed or fail.

This module currently contains the **baseline**, not that machinery.
:class:`NaiveMemoryWriter` stores each user turn verbatim, which is the obvious
thing to build first and is wrong in three specific, measurable ways:

* **precision** — a whole turn is not a fact. "I'm on the payments team and I
  prefer pytest" is stored as one lump, alongside "sounds good, thanks", so the
  store fills with text that will never usefully answer anything.
* **duplication** — a user who mentions their team in five sessions gets five
  near-identical memories, all of which then compete for the same top-k slots.
* **contradiction** — when the user moves from Mumbai to Berlin, both facts are
  stored and both are recalled. The agent is now *less* reliable than it was
  with no memory at all, because it will confidently surface the stale one.

Each of those is scored by the eval harness, so the memory manager that replaces
this has a number to beat rather than a claim to make. The
:class:`MemoryWriter` protocol is the seam it drops into.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog
from langgraph.store.base import BaseStore

from engram.schemas import Memory, MemoryKind, Provenance

log = structlog.get_logger(__name__)


@runtime_checkable
class MemoryWriter(Protocol):
    """The seam between "something was said" and "something is remembered"."""

    def write(
        self,
        user_id: str,
        text: str,
        *,
        thread_id: str | None = None,
        source: Provenance = Provenance.USER,
    ) -> list[Memory]:
        """Consider ``text`` for storage; return the memories actually written.

        Returning a list rather than a single record is deliberate: one turn may
        yield several facts, or — just as importantly — none.
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

    def write(
        self,
        user_id: str,
        text: str,
        *,
        thread_id: str | None = None,
        source: Provenance = Provenance.USER,
    ) -> list[Memory]:
        text = text.strip()
        if not text:
            return []

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
            "memory.write",
            user_id=user_id,
            memory_id=memory.id,
            source=source.value,
            chars=len(text),
        )
        return [memory]
