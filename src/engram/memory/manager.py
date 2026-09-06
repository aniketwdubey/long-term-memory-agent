"""The memory manager — the write path this project exists to build.

Four decisions, in order, for every candidate fact a turn produces:

1. **Dedupe.** Is this already known? A user who mentions their team in five
   sessions should end up with one memory, not five competing for the same
   top-k slots. Slotted facts dedupe on the slot's value; unslotted ones dedupe
   on embedding similarity.
2. **Resolve conflicts.** Does this contradict something already stored? Two
   facts conflict when they fill the same ``(attribute, scope)`` slot with
   different values. The policy is **supersede with history**: the new fact is
   written, the old one is marked superseded and kept. Never keep both live —
   that is the failure that makes an agent with memory *less* reliable than one
   without, because it will surface the stale fact with full confidence.
3. **Score importance and set decay.** Explicitly temporary states get an
   expiry; durable preferences do not.
4. Write what survives.

Deliberately, only step 1 of the pipeline as a whole uses a model — extraction
normalises messy phrasing into slots (see :mod:`engram.memory.extract`). Every
decision *here* is policy code over structured data, which means it is
deterministic, unit-testable, and identical whether the extractor was Claude or
the offline fixture. Asking a model "do these two facts contradict?" on every
write would be slower, costlier, and impossible to pin down in a test.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import TYPE_CHECKING

import structlog
from langchain_core.embeddings import Embeddings
from langgraph.store.base import BaseStore

from engram.memory.extract import CandidateFact, FactExtractor
from engram.memory.write import MemoryDecision, MemoryOp, WriteReport
from engram.schemas import MEMORY_NAMESPACE, Memory, Provenance, utcnow

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

log = structlog.get_logger(__name__)

_MAX_EXISTING = 500


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _norm(value: str) -> str:
    return " ".join(value.lower().split())


class MemoryManager:
    """Extract, dedupe, resolve conflicts, decay. Satisfies ``MemoryWriter``."""

    def __init__(
        self,
        store: BaseStore,
        *,
        extractor: FactExtractor,
        embeddings: Embeddings,
        dedupe_similarity: float = 0.9,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._embeddings = embeddings
        self._dedupe_similarity = dedupe_similarity

    # -- the pipeline ------------------------------------------------------

    def apply(
        self,
        user_id: str,
        text: str,
        *,
        thread_id: str | None = None,
        source: Provenance = Provenance.USER,
    ) -> WriteReport:
        """Run the full write path and report every decision taken."""
        text = text.strip()
        if not text:
            return WriteReport()

        candidates = self._extractor.extract(text)
        if not candidates:
            # The common case. Most turns are not facts, and a write path that
            # cannot say "nothing here" is the reason naive stores fill up.
            log.debug("memory.extract.empty", user_id=user_id, chars=len(text))
            return WriteReport()

        existing = self._active_memories(user_id)
        written: list[Memory] = []
        decisions: list[MemoryDecision] = []

        for candidate in candidates:
            decision, memory = self._resolve(candidate, existing, user_id, thread_id, source)
            decisions.append(decision)
            if memory is not None:
                written.append(memory)
                # Later candidates in the same turn must see earlier ones, or a
                # turn that states a fact twice would store it twice.
                existing.append(memory)

        log.info(
            "memory.write",
            user_id=user_id,
            candidates=len(candidates),
            written=[m.id for m in written],
            ops=[d.op.value for d in decisions],
        )
        return WriteReport(written=written, decisions=decisions)

    def _resolve(
        self,
        candidate: CandidateFact,
        existing: list[Memory],
        user_id: str,
        thread_id: str | None,
        source: Provenance,
    ) -> tuple[MemoryDecision, Memory | None]:
        """Decide this candidate's fate against what is already known."""
        if candidate.attribute:
            same_slot = [
                m
                for m in existing
                if m.attribute == candidate.attribute and m.scope == candidate.scope
            ]

            duplicate = next(
                (m for m in same_slot if _norm(m.value) == _norm(candidate.value)), None
            )
            if duplicate is not None:
                self._reinforce(duplicate, existing)
                return (
                    MemoryDecision(
                        op=MemoryOp.DEDUPE,
                        candidate=candidate.text,
                        memory_id=duplicate.id,
                        reason=f"already knew {candidate.attribute}={candidate.value}",
                    ),
                    None,
                )

            if same_slot:
                # Same slot, different value: a contradiction. The newer
                # statement wins — the user is the authority on their own facts
                # — and the old one is kept as history rather than deleted.
                memory = self._build(candidate, user_id, thread_id, source)
                self._store.put(memory.namespace(), memory.id, memory.to_value())
                for stale in same_slot:
                    self._supersede(stale, memory.id, existing)
                return (
                    MemoryDecision(
                        op=MemoryOp.SUPERSEDE,
                        candidate=candidate.text,
                        memory_id=memory.id,
                        superseded_ids=[m.id for m in same_slot],
                        reason=(
                            f"{candidate.attribute}: "
                            f"{', '.join(m.value for m in same_slot)} -> {candidate.value}"
                        ),
                    ),
                    memory,
                )
        else:
            # Unslotted: nothing to contradict, so the only question is whether
            # this is a restatement of something already stored.
            near = self._nearest(candidate.text, existing)
            if near is not None:
                self._reinforce(near, existing)
                return (
                    MemoryDecision(
                        op=MemoryOp.DEDUPE,
                        candidate=candidate.text,
                        memory_id=near.id,
                        reason="near-duplicate of an existing memory",
                    ),
                    None,
                )

        memory = self._build(candidate, user_id, thread_id, source)
        self._store.put(memory.namespace(), memory.id, memory.to_value())
        return (
            MemoryDecision(op=MemoryOp.WRITE, candidate=candidate.text, memory_id=memory.id),
            memory,
        )

    # -- helpers -----------------------------------------------------------

    def _build(
        self,
        candidate: CandidateFact,
        user_id: str,
        thread_id: str | None,
        source: Provenance,
    ) -> Memory:
        expires_at = (
            utcnow() + timedelta(days=candidate.ttl_days)
            if candidate.ttl_days is not None
            else None
        )
        return Memory(
            user_id=user_id,
            text=candidate.text,
            attribute=candidate.attribute,
            value=candidate.value,
            scope=candidate.scope,
            kind=candidate.kind,
            source=source,
            importance=candidate.importance,
            expires_at=expires_at,
            thread_id=thread_id,
            evidence=candidate.text,
        )

    def _nearest(self, text: str, existing: list[Memory]) -> Memory | None:
        """The existing memory this text duplicates, if any.

        Similarity is computed here rather than read off the store's search
        score, so the threshold means the same thing on every backend.
        """
        if not existing:
            return None
        query = self._embeddings.embed_query(text)
        best, best_score = None, 0.0
        for memory in existing:
            score = _cosine(query, self._embeddings.embed_query(memory.text))
            if score > best_score:
                best, best_score = memory, score
        return best if best_score >= self._dedupe_similarity else None

    def _reinforce(self, memory: Memory, existing: list[Memory]) -> None:
        """A fact restated is a fact that matters. Bump it instead of copying it."""
        updated = memory.model_copy(
            update={
                "importance": min(1.0, memory.importance + 0.1),
                "updated_at": utcnow(),
            }
        )
        self._store.put(updated.namespace(), updated.id, updated.to_value())
        self._replace(existing, updated)

    def _supersede(self, stale: Memory, replacement_id: str, existing: list[Memory]) -> None:
        """Retire a contradicted fact, keeping it as history."""
        retired = stale.model_copy(
            update={"superseded_by": replacement_id, "superseded_at": utcnow()}
        )
        self._store.put(retired.namespace(), retired.id, retired.to_value())
        existing[:] = [m for m in existing if m.id != stale.id]

    @staticmethod
    def _replace(existing: list[Memory], updated: Memory) -> None:
        for i, memory in enumerate(existing):
            if memory.id == updated.id:
                existing[i] = updated
                return

    def _active_memories(self, user_id: str) -> list[Memory]:
        items = self._store.search((MEMORY_NAMESPACE, user_id), limit=_MAX_EXISTING)
        memories = [Memory.from_value(dict(i.value)) for i in items]
        return [m for m in memories if m.is_active()]


def build_writer(
    settings: Settings,
    store: BaseStore,
    *,
    extractor: FactExtractor,
    embeddings: Embeddings,
) -> MemoryManager:
    """Construct the manager from settings."""
    return MemoryManager(
        store,
        extractor=extractor,
        embeddings=embeddings,
        dedupe_similarity=settings.dedupe_similarity,
    )
