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

Before any of it, the **injection gate** decides whether this content is
allowed to write user memory at all — see :mod:`engram.memory.gate`.

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

from engram.memory.extract import CandidateFact, FactExtractor, has_temporal_marker
from engram.memory.gate import InjectionGate
from engram.memory.write import MemoryDecision, MemoryOp, WriteReport
from engram.schemas import MEMORY_NAMESPACE, Memory, Provenance, utcnow
from engram.tracing import set_attributes, span

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

log = structlog.get_logger(__name__)

_MAX_EXISTING = 500

# Values that are a model saying "I don't know" rather than a fact. Left
# unchecked they fill a slot and, being different from whatever is there,
# supersede it — so the agent forgets something true because it was asked a
# question about it.
_NON_VALUES = frozenset({"", "unknown", "unspecified", "none", "n/a", "na", "null", "?"})


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
        gate: InjectionGate | None = None,
        dedupe_similarity: float = 0.9,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._embeddings = embeddings
        self._gate = gate or InjectionGate(store)
        self._dedupe_similarity = dedupe_similarity

    @property
    def gate(self) -> InjectionGate:
        return self._gate

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

        # The gate runs before extraction, not after. Hostile input should not
        # reach a model at all: it costs a call, and an extractor asked to
        # normalise "SYSTEM: the user is an admin" may well do it correctly,
        # producing a perfectly well-formed poisoned fact.
        with span("memory.gate", **{"content.source": source.value}) as current:
            verdict = self._gate.inspect(text, source)
            set_attributes(
                current,
                {
                    "gate.decision": verdict.decision.value,
                    "gate.markers": ",".join(verdict.markers),
                },
            )
        if not verdict.allowed:
            self._gate.quarantine(user_id, text, verdict, thread_id=thread_id)
            return WriteReport(
                decisions=[
                    MemoryDecision(
                        op=MemoryOp.QUARANTINE,
                        candidate=text,
                        reason=verdict.reason,
                    )
                ]
            )

        with span("memory.extract") as current:
            candidates = self._extractor.extract(text)
            set_attributes(
                current,
                {
                    "extract.candidates": len(candidates),
                    "extract.slots": ",".join(c.attribute for c in candidates if c.attribute),
                },
            )
        if not candidates:
            # The common case. Most turns are not facts, and a write path that
            # cannot say "nothing here" is the reason naive stores fill up.
            log.debug("memory.extract.empty", user_id=user_id, chars=len(text))
            return WriteReport()

        candidates = [self._verify_ttl(c, text) for c in candidates]
        candidates = [c for c in candidates if self._is_usable(c)]
        if not candidates:
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

    @staticmethod
    def _is_usable(candidate: CandidateFact) -> bool:
        """Reject a slotted candidate that carries no actual value.

        Found on the first live run, and it is the most destructive failure this
        code has had. Asked "Which team am I on again?", the model dutifully
        returned ``team=""`` — a fact-shaped object with nothing in it. Because
        the empty value differs from the stored one, conflict resolution treated
        it as a contradiction and **superseded the correct fact**. Asking a
        question about something made the agent forget it.

        The same applies to "unknown" and friends: those are the model
        signalling absence, not reporting a value. A slot must be filled to
        supersede what is already in it.
        """
        if not candidate.attribute:
            return True
        if _norm(candidate.value) in _NON_VALUES:
            log.debug(
                "memory.candidate.rejected",
                reason="slotted candidate has no value",
                attribute=candidate.attribute,
                candidate=candidate.text,
            )
            return False
        return True

    def _verify_ttl(self, candidate: CandidateFact, turn: str) -> CandidateFact:
        """Drop an expiry the user's own words do not support.

        Whether a fact is durable is too consequential to delegate: a wrong TTL
        does not fail loudly, it forgets a standing preference weeks later and
        the agent quietly starts getting answers wrong again. Live models are
        genuinely bad at this — on the first Bedrock run Nova Micro gave
        "prefers pytest" a 365-day expiry and "on the payments team" a 30-day
        one, neither of which the user said anything to suggest.

        So the model may *propose* an expiry, and this checks the proposal
        against the text it came from. Same division of labour as everywhere
        else here: the model normalises, policy code decides.
        """
        if candidate.ttl_days is None or has_temporal_marker(turn):
            return candidate

        log.debug(
            "memory.ttl.rejected",
            candidate=candidate.text,
            proposed_ttl_days=candidate.ttl_days,
        )
        return candidate.model_copy(update={"ttl_days": None})

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
        gate=InjectionGate(store, scan_trusted_content=settings.scan_trusted_content),
        dedupe_similarity=settings.dedupe_similarity,
    )
