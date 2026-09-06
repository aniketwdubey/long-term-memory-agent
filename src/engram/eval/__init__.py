"""The eval harness — the reason to believe any of this works.

Memory's headline metric is not "can you retrieve a stored fact when asked
directly" (passive recall, which is easy and near-saturated in published
results) but "did the right answer change because the right fact was recalled"
(decision-relevant recall, where published systems fall to 40–60%). Both are
measured here, against the same cases, with a stateless control arm.
"""

from engram.eval.cases import CaseKind, EvalCase, Session, load_cases
from engram.eval.metrics import ArmReport, CaseResult, score_answer

__all__ = [
    "ArmReport",
    "CaseKind",
    "CaseResult",
    "EvalCase",
    "Session",
    "load_cases",
    "score_answer",
]
