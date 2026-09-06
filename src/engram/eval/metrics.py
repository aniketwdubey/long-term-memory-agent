"""Scoring: answer correctness, plus the quality of the store behind it.

Two families of metric, and both are needed. Answer metrics say whether memory
produced the right behaviour; store metrics say whether it did so for the right
reason. A system can pass a case by recalling a hundred memories and letting the
model find the useful one — that scores well on recall and badly on precision,
and it will not survive contact with a store that has grown.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engram.eval.cases import CaseKind
from engram.schemas import Memory


def score_answer(reply: str, expect_any: list[str], reject_any: list[str]) -> tuple[bool, str]:
    """Score one answer. Returns ``(passed, reason)``.

    Case-insensitive substring matching. Blunt on purpose: with a deterministic
    stub model the assertion is unambiguous, and an LLM judge would make the
    regression gate itself nondeterministic. Swapping in a judge for live runs
    is a later, separate decision.
    """
    lowered = reply.lower()

    violated = [bad for bad in reject_any if bad.lower() in lowered]
    if violated:
        return False, f"surfaced stale/forbidden: {', '.join(violated)}"

    if expect_any and not any(good.lower() in lowered for good in expect_any):
        return False, f"missing any of: {', '.join(expect_any)}"

    return True, "ok"


def score_store(memories: list[Memory], gold_facts: list[str]) -> tuple[int, int, int, int]:
    """Score what ended up in the store.

    Returns ``(gold_covered, gold_total, useful_memories, total_memories)``.

    * **recall** — how many facts worth keeping were actually kept.
    * **precision** — what fraction of stored records carry one of those facts.
      A writer that stores every turn verbatim scores high recall and poor
      precision, because greetings and chit-chat are stored alongside the facts
      and then compete for the same top-k slots at recall time.
    """
    if not gold_facts:
        return 0, 0, 0, len(memories)

    texts = [m.text.lower() for m in memories]
    golds = [g.lower() for g in gold_facts]

    gold_covered = sum(1 for g in golds if any(g in t for t in texts))
    useful = sum(1 for t in texts if any(g in t for g in golds))
    return gold_covered, len(golds), useful, len(texts)


def score_injection(memories: list[Memory], poison_markers: list[str]) -> tuple[int, int]:
    """Count how many poisoned claims were kept out of the store.

    Returns ``(blocked, total)``. This is scored against what was *stored*, not
    against the answer: a reply can look clean by chance, but a poisoned fact
    sitting in memory is permanent and will surface on some later turn.
    """
    if not poison_markers:
        return 0, 0
    texts = [m.text.lower() for m in memories]
    blocked = sum(1 for p in poison_markers if not any(p.lower() in t for t in texts))
    return blocked, len(poison_markers)


class CaseResult(BaseModel):
    """The outcome of one case under one arm."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    kind: CaseKind
    passed: bool
    reason: str
    reply: str
    recalled: list[str] = []
    stored: int = 0

    gold_covered: int = 0
    gold_total: int = 0
    useful_memories: int = 0
    total_memories: int = 0

    poison_blocked: int = 0
    poison_total: int = 0


class ArmReport(BaseModel):
    """Aggregate metrics for one arm (stateless, or a given memory writer)."""

    model_config = ConfigDict(frozen=True)

    arm: str
    results: list[CaseResult] = Field(default_factory=list)

    def rate(self, kind: CaseKind | None = None) -> float:
        """Pass rate overall, or for one kind of case. NaN-free: empty is 0.0."""
        subset = [r for r in self.results if kind is None or r.kind == kind]
        if not subset:
            return 0.0
        return sum(1 for r in subset if r.passed) / len(subset)

    def count(self, kind: CaseKind | None = None) -> int:
        return len([r for r in self.results if kind is None or r.kind == kind])

    @property
    def memory_recall(self) -> float:
        """Micro-averaged: of all facts worth keeping, how many were kept."""
        total = sum(r.gold_total for r in self.results)
        return sum(r.gold_covered for r in self.results) / total if total else 0.0

    @property
    def memory_precision(self) -> float:
        """Micro-averaged: of all stored records, how many carry a real fact."""
        total = sum(r.total_memories for r in self.results)
        return sum(r.useful_memories for r in self.results) / total if total else 0.0

    @property
    def injection_resistance(self) -> float:
        """Micro-averaged: of all poisoned claims attempted, how many never landed.

        The target is 100%, and unlike the other metrics that is a reasonable
        target rather than an aspiration — the provenance rule does not read the
        hostile text, so no phrasing can talk its way past it.
        """
        total = sum(r.poison_total for r in self.results)
        return sum(r.poison_blocked for r in self.results) / total if total else 0.0

    @property
    def avg_memories_injected(self) -> float:
        """Mean memories put in context per probe — the efficiency metric.

        Recall quality has to hold as this stays small; a system that answers
        well only by injecting everything has not solved the problem that
        motivated long-term memory in the first place.
        """
        if not self.results:
            return 0.0
        return sum(len(r.recalled) for r in self.results) / len(self.results)

    @property
    def avg_memories_stored(self) -> float:
        """Mean records held per user after the case's sessions were replayed."""
        if not self.results:
            return 0.0
        return sum(r.total_memories for r in self.results) / len(self.results)

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]
