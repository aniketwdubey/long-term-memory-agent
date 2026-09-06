"""LoCoMo: the same memory system, measured on somebody else's benchmark.

Every other number in this project is scored on cases written here, which is
worth exactly as much as any exam written by the person sitting it. LoCoMo is
public, third-party and much longer — 10 conversations, ~5.9k turns, ~2k
questions, up to 19 sessions and 400+ turns per conversation — so it says
something the hand-authored set cannot.

It is an addition, not a replacement, for three reasons:

* **It is judged, so it cannot gate CI.** Gold answers are free-form
  ("The sunday before 25 May 2023"); grading them needs a model, and a model
  makes the score non-deterministic. The hand-authored benchmark stays the
  regression gate.
* **It has no conflict or injection questions.** The two properties this
  project is really about have no LoCoMo counterpart.
* **It annotates evidence turns, not facts worth keeping**, so the store-side
  precision/recall metrics have nothing to score against here.

**Speaker mapping.** LoCoMo is a dialogue between two people, not a user and an
assistant, and its questions ask about both of them. Rather than pick one as
"the user" and throw the other's facts away, the whole conversation is treated
as one rememberable subject: turns are ingested as ``"Caroline: ..."`` /
``"Melanie: ..."`` under a single user id, keeping attribution inside the text
where extraction can use it.

**Adversarial questions are graded as refusals, and their `adversarial_answer`
field is a trap rather than a gold answer.** Category 5 questions carry a false
premise — usually attributing something to the wrong speaker — and
`adversarial_answer` holds the plausible answer you produce if you fail to
notice ("What did *Caroline* realise after her charity race?" -> "self-care is
important", which is a thing *Melanie* realised). The correct behaviour is to
decline, and the official evaluation scores exactly that, checking the output for
"no information available" rather than matching any answer. Reading that field as
gold would invert the metric, so it is loaded as :attr:`LocomoQuestion.distractor`
and scored as a trap the answer should *not* contain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from engram.config import Settings
from engram.eval.judge import Judge, Verdict, build_judge
from engram.graph import Agent
from engram.logging import configure_logging
from engram.schemas import Provenance
from engram.store import open_backend

log = structlog.get_logger(__name__)

DEFAULT_PATH = Path("data/locomo10.json")

# Category numbers as used by the dataset. 1, 2 and 5 are pinned by the official
# evaluation code and the data itself; 3 and 4 are grouped together with 2 there
# ("single-hop, temporal, open-domain") without saying which is which, so they
# are reported by number rather than given a name this project invented.
ADVERSARIAL_CATEGORY = 5
CATEGORY_NAMES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "category-3",
    4: "category-4",
    5: "adversarial",
}

# The official evaluation scores an adversarial question correct when the answer
# declines to answer. These are its two markers plus the ordinary ways a model
# says the same thing.
_REFUSAL = re.compile(
    r"\b(?:no information available|not mentioned|no information|don'?t know|"
    r"do not know|isn'?t mentioned|was not mentioned|no record|nothing about|"
    r"cannot (?:tell|say|determine)|can'?t (?:tell|say|determine)|"
    r"not (?:stated|specified|available|provided|something I)|"
    r"nothing (?:stored|in my memory)|no memory of|never mentioned)\b",
    re.IGNORECASE,
)


class LocomoTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    speaker: str
    dia_id: str = ""
    text: str

    def rendered(self) -> str:
        """Attributed text, so extraction can tell whose fact it is."""
        return f"{self.speaker}: {self.text}"


class LocomoSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    date_time: str = ""
    turns: list[LocomoTurn]


class LocomoQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    #: The gold answer. Empty for adversarial questions, which have none — the
    #: right response there is to decline.
    answer: str = ""
    #: Adversarial only: the plausible-but-wrong answer the false premise
    #: invites. An answer containing this fell for the trap.
    distractor: str = ""
    category: int
    evidence: list[str] = Field(default_factory=list)

    @property
    def is_adversarial(self) -> bool:
        return self.category == ADVERSARIAL_CATEGORY

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"category-{self.category}")


class LocomoConversation(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    speaker_a: str = ""
    speaker_b: str = ""
    sessions: list[LocomoSession]
    questions: list[LocomoQuestion]

    @property
    def turn_count(self) -> int:
        return sum(len(s.turns) for s in self.sessions)


def _session_sort_key(key: str) -> int:
    match = re.search(r"(\d+)", key)
    return int(match.group(1)) if match else 0


def _parse_conversation(raw: dict[str, Any], index: int) -> LocomoConversation:
    conversation = raw.get("conversation", {})
    session_keys = sorted(
        (k for k in conversation if k.startswith("session_") and "date_time" not in k),
        key=_session_sort_key,
    )
    sessions = [
        LocomoSession(
            key=key,
            date_time=str(conversation.get(f"{key}_date_time", "")),
            turns=[
                LocomoTurn(
                    speaker=str(t.get("speaker", "")),
                    dia_id=str(t.get("dia_id", "")),
                    text=str(t.get("text", "")),
                )
                for t in conversation[key]
                if str(t.get("text", "")).strip()
            ],
        )
        for key in session_keys
    ]

    questions: list[LocomoQuestion] = []
    for item in raw.get("qa", []):
        category = int(item.get("category", 0))
        adversarial = category == ADVERSARIAL_CATEGORY
        # `adversarial_answer` is the trap, not the gold answer — see the module
        # docstring. Loading it as gold would score a correct refusal as wrong
        # and a credulous answer as right.
        questions.append(
            LocomoQuestion(
                question=str(item.get("question", "")),
                answer="" if adversarial else str(item.get("answer", "")),
                distractor=str(item.get("adversarial_answer", "")) if adversarial else "",
                category=category,
                evidence=[str(e) for e in item.get("evidence", [])],
            )
        )

    return LocomoConversation(
        sample_id=str(raw.get("sample_id") or f"conversation-{index}"),
        speaker_a=str(conversation.get("speaker_a", "")),
        speaker_b=str(conversation.get("speaker_b", "")),
        sessions=sessions,
        questions=questions,
    )


def load_locomo(path: str | Path = DEFAULT_PATH) -> list[LocomoConversation]:
    """Read the LoCoMo JSON into typed conversations."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"{file} not found. The dataset is not vendored here (it is ~2.8MB and "
            "carries its own licence terms). Fetch it with: python scripts/fetch_locomo.py"
        )
    raw = json.loads(file.read_text(encoding="utf-8"))
    return [_parse_conversation(item, i) for i, item in enumerate(raw)]


# --- scoring -----------------------------------------------------------------


def declines_to_answer(answer: str) -> bool:
    """Whether an answer admits it does not know.

    Mirrors the official evaluation's adversarial check, widened a little to
    cover the ordinary phrasings a model reaches for.
    """
    return bool(_REFUSAL.search(answer))


def score_locomo_answer(question: LocomoQuestion, answer: str, judge: Judge) -> Verdict:
    """Grade one answer, refusal-style for adversarial questions."""
    if question.is_adversarial:
        declined = declines_to_answer(answer)
        if declined:
            return Verdict(correct=True, reason="declined a question with a false premise")
        fell_for_it = bool(question.distractor) and question.distractor.lower() in answer.lower()
        return Verdict(
            correct=False,
            reason="repeated the trap answer" if fell_for_it else "answered anyway",
        )
    return judge.score(question.question, question.answer, answer)


class LocomoResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    question: str
    category: int
    gold: str
    answer: str
    correct: bool
    reason: str = ""


class LocomoReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    arm: str
    results: list[LocomoResult] = Field(default_factory=list)
    turns_ingested: int = 0
    memories_stored: int = 0

    def accuracy(self, category: int | None = None) -> float:
        subset = [r for r in self.results if category is None or r.category == category]
        return sum(1 for r in subset if r.correct) / len(subset) if subset else 0.0

    def count(self, category: int | None = None) -> int:
        return len([r for r in self.results if category is None or r.category == category])

    def categories(self) -> list[int]:
        return sorted({r.category for r in self.results})


# --- running -----------------------------------------------------------------


def run_conversation(
    agent: Agent,
    conversation: LocomoConversation,
    judge: Judge,
    *,
    max_questions: int | None = None,
) -> tuple[list[LocomoResult], int]:
    """Replay one conversation into memory, then ask its questions."""
    user = conversation.sample_id

    ingested = 0
    for session in conversation.sessions:
        for turn in session.turns:
            # Write-path only: no reply is generated during replay. Nothing is
            # scored on those replies, and at ~400 turns per conversation the
            # saved model calls are the difference between a runnable benchmark
            # and an unaffordable one. Provenance stays USER — this is the
            # conversation being remembered, not a document the agent read.
            agent.observe(user, turn.rendered(), thread_id=session.key, source=Provenance.USER)
            ingested += 1

    questions = conversation.questions[:max_questions] if max_questions else conversation.questions
    results: list[LocomoResult] = []
    for i, question in enumerate(questions):
        # A fresh thread per question, so an earlier answer cannot leak into a
        # later one through the transcript.
        trace = agent.chat(user, f"{user}:q{i}", question.question)
        verdict = score_locomo_answer(question, trace.reply, judge)
        results.append(
            LocomoResult(
                sample_id=user,
                question=question.question,
                category=question.category,
                gold=question.answer,
                answer=trace.reply,
                correct=verdict.correct,
                reason=verdict.reason,
            )
        )
    return results, ingested


def run_locomo(
    settings: Settings,
    conversations: Sequence[LocomoConversation],
    *,
    arm: str = "manager",
    max_questions: int | None = None,
) -> LocomoReport:
    """Run every conversation under one configuration."""
    settings = settings.model_copy(update={"store_backend": "memory", "memory_writer": arm})
    judge = build_judge(settings)

    results: list[LocomoResult] = []
    ingested = 0
    stored = 0
    with open_backend(settings) as backend:
        agent = Agent(
            settings,
            checkpointer=backend.checkpointer,
            store=backend.store,
            embeddings=backend.embeddings,
        )
        for conversation in conversations:
            conv_results, conv_ingested = run_conversation(
                agent, conversation, judge, max_questions=max_questions
            )
            results.extend(conv_results)
            ingested += conv_ingested
            stored += len(
                [m for m in agent.reader.all_memories(conversation.sample_id) if m.is_active()]
            )

    return LocomoReport(arm=arm, results=results, turns_ingested=ingested, memories_stored=stored)


def render_report(report: LocomoReport, conversations: Sequence[LocomoConversation]) -> str:
    lines = ["", f"LoCoMo · {len(conversations)} conversation(s) · arm: {report.arm}", ""]
    lines.append(
        f"{report.turns_ingested} turns ingested -> {report.memories_stored} memories kept"
    )
    lines.append("")
    lines.append(f"{'category':<22}{'n':>6}{'accuracy':>12}")
    lines.append("-" * 40)
    for category in report.categories():
        name = CATEGORY_NAMES.get(category, f"category-{category}")
        lines.append(
            f"{name:<22}{report.count(category):>6}{report.accuracy(category) * 100:>11.1f}%"
        )
    lines.append("-" * 40)
    lines.append(f"{'overall':<22}{report.count():>6}{report.accuracy() * 100:>11.1f}%")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="engram.eval.locomo", description="Run the LoCoMo benchmark."
    )
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--arm", default="manager", choices=["manager", "naive"])
    parser.add_argument(
        "--conversations", type=int, default=1, help="How many conversations to run."
    )
    parser.add_argument(
        "--questions", type=int, default=None, help="Cap questions per conversation."
    )
    parser.add_argument("--provider", choices=["stub", "bedrock"])
    parser.add_argument("--embedder", choices=["hashing", "fastembed", "bedrock"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.provider:
        overrides["chat_provider"] = args.provider
    if args.embedder:
        overrides["embedder"] = args.embedder
    settings = Settings(**overrides)  # type: ignore[arg-type]
    configure_logging(settings)

    conversations = load_locomo(args.path)[: args.conversations]
    report = run_locomo(settings, conversations, arm=args.arm, max_questions=args.questions)

    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        print(render_report(report, conversations))
        if settings.chat_provider == "stub":
            print(
                "NOTE: the stub model and substring judge are a floor, not a score. "
                "Run with --provider bedrock --embedder bedrock for a real number.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
