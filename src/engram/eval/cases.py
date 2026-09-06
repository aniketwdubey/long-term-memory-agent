"""Eval case format and loader.

A case is a small piece of a user's history — a few prior sessions — followed by
a probe asked in a *new* session. What makes it a memory test rather than a
conversation test is that the probe's thread shares nothing with the sessions
that established the fact: the only path from one to the other is long-term
memory.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engram.schemas import Provenance


class CaseKind(StrEnum):
    """What a case is testing.

    PASSIVE_RECALL    the probe asks for the fact directly ("what framework do
                      I prefer?"). The easy baseline.
    DECISION_RELEVANT the probe never names the fact, but the right answer
                      depends on it ("scaffold me a test"). The headline metric.
    CONFLICT          a later session contradicts an earlier one. The right
                      answer uses the new fact and must not surface the stale
                      one — which a store that simply keeps both cannot do.
    INJECTION         the history contains hostile text — a poisoned document
                      the agent read, or hostile content pasted into a turn.
                      None of it may become a fact about the user.
    """

    PASSIVE_RECALL = "passive_recall"
    DECISION_RELEVANT = "decision_relevant"
    CONFLICT = "conflict"
    INJECTION = "injection"


class Turn(BaseModel):
    """One thing that happened in a session, and where it came from.

    A bare string parses as the user speaking, which keeps ordinary cases
    readable. Anything the agent merely *read* — a tool result, a retrieved
    document — must say so, because provenance is the whole subject of the
    injection cases.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    source: Provenance = Provenance.USER

    @model_validator(mode="before")
    @classmethod
    def _accept_a_bare_string(cls, value: object) -> object:
        return {"text": value} if isinstance(value, str) else value

    @property
    def spoken_by_user(self) -> bool:
        return self.source is Provenance.USER


class Session(BaseModel):
    """One prior conversation, replayed turn by turn to build up memory."""

    model_config = ConfigDict(frozen=True)

    thread_id: str
    turns: list[Turn] = Field(min_length=1)


class EvalCase(BaseModel):
    """One memory test."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: CaseKind
    user_id: str
    sessions: list[Session] = Field(min_length=1)
    probe: str
    probe_thread_id: str = "probe"

    # The answer must contain at least one `expect_any` string and none of the
    # `reject_any` ones. Matching is case-insensitive substring. `expect_any`
    # may be empty when a case only asserts that something is absent.
    expect_any: list[str] = []
    reject_any: list[str] = []

    # Substrings that must NOT appear in any *stored* memory. This is the
    # injection metric proper: an answer can be clean by luck, but a poisoned
    # fact sitting in the store will surface on some later turn.
    poison_markers: list[str] = []

    # Substrings a well-behaved store *should* hold for this user, used to score
    # memory precision and recall. These describe facts worth keeping — not the
    # filler turns that surround them.
    gold_facts: list[str] = []

    note: str = ""

    @model_validator(mode="after")
    def _must_assert_something(self) -> EvalCase:
        if not (self.expect_any or self.reject_any or self.poison_markers):
            raise ValueError(f"case {self.id!r}: asserts nothing")
        return self

    @model_validator(mode="after")
    def _probe_thread_must_be_new(self) -> EvalCase:
        """The probe must be asked in a session the facts were not stated in.

        Without this the checkpointer alone would answer the probe from the
        running transcript, long-term memory would never be consulted, and every
        arm — including the stateless control — would score perfectly. A guard
        rather than a comment, because that failure is silent and flattering.
        """
        established = {s.thread_id for s in self.sessions}
        if self.probe_thread_id in established:
            raise ValueError(
                f"case {self.id!r}: probe_thread_id {self.probe_thread_id!r} is also a "
                "session thread, so the probe could be answered from thread memory "
                "alone and would not test long-term memory"
            )
        return self

    @property
    def user_key(self) -> str:
        """Store-facing user id, namespaced by case so cases cannot leak."""
        return f"{self.id}:{self.user_id}"


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load cases from a JSONL file, skipping blank lines and ``#`` comments."""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = EvalCase.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases
