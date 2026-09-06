"""The LoCoMo harness: loading a third-party format, and grading it honestly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from engram.config import Settings
from engram.eval.judge import LLMJudge, SubstringJudge, Verdict, build_judge
from engram.eval.locomo import (
    ADVERSARIAL_CATEGORY,
    LocomoQuestion,
    declines_to_answer,
    load_locomo,
    run_locomo,
    score_locomo_answer,
)
from engram.models import StubChatModel

RAW: list[dict[str, Any]] = [
    {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "7 May 2023",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "I prefer pytest."},
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "I moved to Berlin."},
                {"speaker": "Melanie", "dia_id": "D1:3", "text": "   "},
            ],
            # Deliberately out of order in the file, to prove sorting is numeric.
            "session_10": [
                {"speaker": "Caroline", "dia_id": "D10:1", "text": "Still on payments."}
            ],
            "session_2": [{"speaker": "Caroline", "dia_id": "D2:1", "text": "I use Neovim."}],
        },
        "qa": [
            {
                "question": "What does Caroline prefer?",
                "answer": "pytest",
                "evidence": ["D1:1"],
                "category": 1,
            },
            {
                "question": "What did Caroline realize after her charity race?",
                "adversarial_answer": "self-care is important",
                "evidence": ["D2:3"],
                "category": 5,
            },
        ],
    }
]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(RAW), encoding="utf-8")
    return path


@pytest.fixture
def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None, chat_provider="stub", embedder="hashing", store_backend="memory"
    )


# --- loading -----------------------------------------------------------------


def test_loads_conversations_sessions_and_turns(dataset: Path) -> None:
    conv = load_locomo(dataset)[0]
    assert conv.sample_id == "conv-1"
    assert (conv.speaker_a, conv.speaker_b) == ("Caroline", "Melanie")
    assert conv.turn_count == 4


def test_sessions_are_ordered_numerically_not_lexically(dataset: Path) -> None:
    """ "session_10" sorts before "session_2" as a string.

    Getting this wrong would replay a conversation out of order, which for a
    system whose whole job is "what is true *now*" quietly corrupts every
    conflict the dialogue contains.
    """
    assert [s.key for s in load_locomo(dataset)[0].sessions] == [
        "session_1",
        "session_2",
        "session_10",
    ]


def test_blank_turns_are_dropped(dataset: Path) -> None:
    assert all(t.text.strip() for s in load_locomo(dataset)[0].sessions for t in s.turns)


def test_turns_keep_their_speaker_when_rendered(dataset: Path) -> None:
    """LoCoMo is two people talking, and its questions ask about both.

    Attribution has to survive into the text or extraction cannot tell whose
    fact it is.
    """
    first = load_locomo(dataset)[0].sessions[0].turns[0]
    assert first.rendered() == "Caroline: I prefer pytest."


def test_session_timestamps_are_kept(dataset: Path) -> None:
    assert load_locomo(dataset)[0].sessions[0].date_time == "7 May 2023"


def test_a_missing_dataset_says_how_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_locomo"):
        load_locomo(tmp_path / "absent.json")


# --- the adversarial trap ----------------------------------------------------


def test_adversarial_questions_have_no_gold_answer(dataset: Path) -> None:
    """`adversarial_answer` is the trap, not the answer.

    These questions carry a false premise — usually the wrong speaker — and that
    field holds the plausible answer you give if you fall for it. Loading it as
    gold would score a correct refusal wrong and a credulous answer right,
    inverting the metric across 446 of the dataset's ~2000 questions.
    """
    adversarial = [q for q in load_locomo(dataset)[0].questions if q.is_adversarial]
    assert len(adversarial) == 1
    assert adversarial[0].answer == ""
    assert adversarial[0].distractor == "self-care is important"


def test_ordinary_questions_keep_their_gold_answer(dataset: Path) -> None:
    normal = [q for q in load_locomo(dataset)[0].questions if not q.is_adversarial]
    assert normal[0].answer == "pytest"
    assert normal[0].distractor == ""


@pytest.mark.parametrize(
    "answer",
    [
        "No information available.",
        "That was not mentioned in our conversations.",
        "I don't know — you never told me that.",
        "I have no memory of that.",
    ],
)
def test_declining_is_the_correct_answer_to_a_false_premise(answer: str) -> None:
    question = LocomoQuestion(
        question="What did Caroline realize?", distractor="self-care", category=ADVERSARIAL_CATEGORY
    )
    assert score_locomo_answer(question, answer, SubstringJudge()).correct


def test_falling_for_the_trap_is_reported_as_such() -> None:
    question = LocomoQuestion(
        question="What did Caroline realize?",
        distractor="self-care is important",
        category=ADVERSARIAL_CATEGORY,
    )
    verdict = score_locomo_answer(
        question, "She realized self-care is important.", SubstringJudge()
    )
    assert not verdict.correct
    assert "trap" in verdict.reason


def test_answering_confidently_is_wrong_even_without_the_trap_answer() -> None:
    question = LocomoQuestion(
        question="What did Caroline realize?", distractor="self-care", category=ADVERSARIAL_CATEGORY
    )
    assert not score_locomo_answer(
        question, "She realized she loves running.", SubstringJudge()
    ).correct


def test_an_ordinary_question_goes_to_the_judge() -> None:
    question = LocomoQuestion(question="What does she prefer?", answer="pytest", category=1)
    assert score_locomo_answer(question, "She prefers pytest.", SubstringJudge()).correct
    assert not score_locomo_answer(question, "She prefers unittest.", SubstringJudge()).correct


def test_refusal_detection_does_not_fire_on_a_real_answer() -> None:
    assert not declines_to_answer("Caroline prefers pytest and lives in Berlin.")


# --- the judge ---------------------------------------------------------------


class _Structured:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def invoke(self, messages: Any) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class _FakeModel(StubChatModel):
    payload: Any = None

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return _Structured(self.payload)


def test_the_judge_accepts_a_paraphrase() -> None:
    """The entire reason a judge exists here."""
    judge = LLMJudge(_FakeModel(payload=Verdict(correct=True, reason="same date")))
    assert judge.score("When?", "7 May 2023", "It was the 7th of May, 2023.").correct


def test_a_broken_judge_scores_wrong_rather_than_right() -> None:
    """A judge that errors must not silently inflate the score."""
    judge = LLMJudge(_FakeModel(payload=RuntimeError("bedrock is down")))
    verdict = judge.score("q", "gold", "answer")
    assert not verdict.correct
    assert "judge error" in verdict.reason


def test_the_substring_judge_is_only_a_floor() -> None:
    """It fails correct paraphrases, which is why it is documented as a floor."""
    assert not SubstringJudge().score("When?", "7 May 2023", "The 7th of May, 2023").correct


def test_the_offline_config_gets_the_substring_judge(settings: Settings) -> None:
    assert isinstance(build_judge(settings), SubstringJudge)


# --- end to end --------------------------------------------------------------


def test_a_conversation_runs_end_to_end_offline(settings: Settings, dataset: Path) -> None:
    report = run_locomo(settings, load_locomo(dataset))
    assert report.turns_ingested == 4
    assert report.count() == 2
    assert 0.0 <= report.accuracy() <= 1.0
    assert report.categories() == [1, ADVERSARIAL_CATEGORY]
