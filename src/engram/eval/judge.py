"""Answer grading for free-form gold answers.

The hand-authored benchmark asserts substrings, which is blunt but unambiguous
and therefore safe to gate CI on. LoCoMo cannot work that way: its gold answers
are things like *"The sunday before 25 May 2023"* and *"Psychology, counseling
certification"*, and an agent that says "in late May 2023, the Sunday before the
25th" is right. Substring matching would score it wrong, over and over, and the
resulting number would measure phrasing rather than memory.

So LoCoMo runs are **judged**, and that has a cost worth stating plainly: a judge
is a model, so the score is no longer deterministic and this harness can never be
a regression gate. That is why it lives beside the hand-authored benchmark rather
than replacing it.

:class:`SubstringJudge` is the offline stand-in — same interface, no model, used
by tests and by anyone running the loader without credentials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

log = structlog.get_logger(__name__)


class Verdict(BaseModel):
    """One graded answer."""

    model_config = ConfigDict(frozen=True)

    correct: bool = Field(description="Whether the answer conveys the gold answer.")
    reason: str = Field(default="", description="One short sentence of justification.")


@runtime_checkable
class Judge(Protocol):
    def score(self, question: str, gold: str, answer: str) -> Verdict: ...


JUDGE_SYSTEM = """\
You grade an assistant's answer against a known-correct answer.

Mark it correct when the answer conveys the same fact, however it is worded. \
Paraphrase, extra detail, a different date format, or a fuller sentence are all \
fine — you are grading the information, not the phrasing.

Mark it incorrect when the answer states something that contradicts the gold \
answer, omits the fact entirely, hedges without committing to it, or says it \
does not know. An answer that lists the right fact among several guesses is \
incorrect: guessing broadly enough to include the truth is not remembering it.

Give one short sentence of justification.
"""


class LLMJudge:
    """Grades with a model. Non-deterministic by nature — see the module docstring."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(Verdict)

    def score(self, question: str, gold: str, answer: str) -> Verdict:
        prompt = f"Question: {question}\nKnown-correct answer: {gold}\nAssistant's answer: {answer}"
        try:
            result = self._model.invoke(
                [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]
            )
        except Exception as exc:
            # A judge that errors must not silently mark answers correct. Score
            # it wrong and say why, so the failure shows up in the report rather
            # than inflating it.
            log.warning("judge.failed", error=str(exc)[:200])
            return Verdict(correct=False, reason=f"judge error: {type(exc).__name__}")

        if isinstance(result, Verdict):
            return result
        return Verdict.model_validate(result)


class SubstringJudge:
    """Deterministic fallback: is the gold answer present verbatim?

    Strictly worse than a model at this job — it fails every correct paraphrase —
    and it exists so the loader and runner can be exercised without credentials.
    A number produced with this judge is a floor, not a score.
    """

    def score(self, question: str, gold: str, answer: str) -> Verdict:
        hit = gold.strip().lower() in answer.lower()
        return Verdict(
            correct=hit,
            reason="gold answer present verbatim" if hit else "gold answer not found",
        )


def build_judge(settings: Settings, model: BaseChatModel | None = None) -> Judge:
    """An LLM judge when a real model is configured, the substring floor otherwise."""
    if settings.chat_provider == "stub":
        return SubstringJudge()

    from engram.models import build_chat_model

    return LLMJudge(model or build_chat_model(settings))
