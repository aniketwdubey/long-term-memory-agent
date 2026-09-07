"""Extraction: what becomes a candidate fact, and what is correctly ignored."""

from __future__ import annotations

from typing import Any

import pytest

from engram.config import Settings
from engram.memory.extract import (
    CandidateFact,
    ExtractionResult,
    FactExtractor,
    LLMFactExtractor,
    RuleFactExtractor,
    build_extractor,
    turn_date,
)
from engram.models import StubChatModel
from engram.schemas import MemoryKind


@pytest.fixture
def extractor() -> RuleFactExtractor:
    return RuleFactExtractor()


def test_satisfies_the_extractor_protocol(extractor: RuleFactExtractor) -> None:
    assert isinstance(extractor, FactExtractor)


# --- what must NOT be stored -------------------------------------------------


@pytest.mark.parametrize(
    "turn",
    [
        "Morning! The coffee machine is broken again.",
        "Did you see the standup notes from yesterday?",
        "We finally shipped the new dashboard on Friday.",
        "I'm going to grab lunch, back in twenty.",
        "My sister is visiting from Toronto next month.",
        "My neighbour is renovating, so there's drilling all day.",
        "Traffic was awful on the bridge this morning.",
        "Scaffold me a test for the refund endpoint.",
        "",
        "   ",
    ],
)
def test_ignores_everything_that_is_not_a_fact_about_the_user(
    extractor: RuleFactExtractor, turn: str
) -> None:
    """Precision is the whole reason extraction exists.

    Chatter, status updates, other people's activities and the user's own
    requests are not facts about the user. A write path that stores them is the
    naive baseline with extra steps.
    """
    assert extractor.extract(turn) == []


# --- what must be stored, and in what shape ----------------------------------


@pytest.mark.parametrize(
    ("turn", "attribute", "value"),
    [
        ("I prefer pytest for everything I write.", "testing_framework", "pytest"),
        ("I just joined the payments team.", "team", "payments"),
        ("I've relocated to Berlin.", "location", "Berlin"),
        ("I'm based in Lisbon.", "location", "Lisbon"),
        ("We migrated to Postgres over the weekend.", "database", "Postgres"),
        ("I switched to Zed a few weeks ago.", "editor", "Zed"),
        ("I'm vegetarian.", "diet", "vegetarian"),
        ("My dog Biscuit is a beagle.", "pet_name", "Biscuit"),
    ],
)
def test_normalises_a_fact_into_a_slot(
    extractor: RuleFactExtractor, turn: str, attribute: str, value: str
) -> None:
    """The slot is the point: it is what makes conflict detection a lookup."""
    facts = extractor.extract(turn)
    assert len(facts) == 1
    assert facts[0].attribute == attribute
    assert facts[0].value == value.lower()


def test_slot_values_are_case_normalised(extractor: RuleFactExtractor) -> None:
    """ "Postgres" and "postgres" must land in the same slot value, or dedupe fails."""
    a = extractor.extract("We run Postgres in production.")[0]
    b = extractor.extract("We run postgres in production.")[0]
    assert a.value == b.value


def test_standing_instructions_are_procedural(extractor: RuleFactExtractor) -> None:
    facts = extractor.extract("Always show me the SQL before running it.")
    assert len(facts) == 1
    assert facts[0].kind is MemoryKind.PROCEDURAL
    # Unslotted: a directive should never supersede an unrelated one.
    assert facts[0].attribute == ""


def test_directives_score_higher_importance(extractor: RuleFactExtractor) -> None:
    directive = extractor.extract("Don't send me Slack notifications after 6pm.")[0]
    assert directive.importance > 0.5


def test_a_turn_can_yield_several_facts(extractor: RuleFactExtractor) -> None:
    facts = extractor.extract("I'm on the payments team. I prefer pytest.")
    assert len(facts) == 2
    assert {f.attribute for f in facts} == {"team", "testing_framework"}


def test_unrecognised_vocabulary_still_becomes_an_unslotted_fact(
    extractor: RuleFactExtractor,
) -> None:
    """A lexicon can only know words someone thought of. The fallback keeps the
    fact rather than dropping it; it simply never supersedes anything."""
    facts = extractor.extract("I use a standing desk for most of the day.")
    assert len(facts) == 1
    assert facts[0].attribute == ""


# --- decay -------------------------------------------------------------------


def test_explicitly_temporary_states_get_an_expiry(extractor: RuleFactExtractor) -> None:
    facts = extractor.extract("I'm debugging a flaky test today.")
    assert len(facts) == 1
    assert facts[0].ttl_days is not None


def test_durable_preferences_never_expire(extractor: RuleFactExtractor) -> None:
    """The failure mode this guards: silently forgetting a standing preference."""
    assert extractor.extract("I prefer pytest for everything I write.")[0].ttl_days is None


def test_an_activity_without_a_time_marker_is_not_stored_at_all(
    extractor: RuleFactExtractor,
) -> None:
    assert extractor.extract("I'm reviewing the migration plan.") == []


# --- dated turns -------------------------------------------------------------


def test_the_bracketed_date_is_not_treated_as_part_of_the_sentence(
    extractor: RuleFactExtractor,
) -> None:
    facts = extractor.extract("[7 May 2023] I prefer pytest for everything I write.")
    assert facts[0].attribute == "testing_framework"


def test_turn_date_reads_the_prefix() -> None:
    assert turn_date("[7 May 2023] Caroline: hello") == "7 May 2023"
    assert turn_date("Caroline: hello") == ""


# --- the LLM extractor -------------------------------------------------------


class _Structured:
    """Stands in for ``model.with_structured_output(...)``."""

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


def test_llm_extractor_returns_the_models_facts() -> None:
    fact = CandidateFact(text="Prefers pytest", attribute="testing_framework", value="pytest")
    extractor = LLMFactExtractor(_FakeModel(payload=ExtractionResult(facts=[fact])))
    assert extractor.extract("I like pytest") == [fact]


def test_llm_extractor_accepts_a_plain_dict() -> None:
    """Some providers hand back a dict rather than the model instance."""
    extractor = LLMFactExtractor(_FakeModel(payload={"facts": [{"text": "Lives in Berlin"}]}))
    assert [f.text for f in extractor.extract("...")] == ["Lives in Berlin"]


def test_a_failed_extraction_never_breaks_the_conversation() -> None:
    """A turn that cannot be extracted is a turn that is not remembered.

    That is recoverable; a failed response is not. The write path must degrade
    to "no memory written", never to an exception reaching the user.
    """
    extractor = LLMFactExtractor(_FakeModel(payload=RuntimeError("bedrock is down")))
    assert extractor.extract("I prefer pytest") == []


# --- wiring ------------------------------------------------------------------


def test_stub_provider_gets_the_offline_extractor() -> None:
    settings = Settings(_env_file=None, chat_provider="stub")  # type: ignore[call-arg]
    assert isinstance(build_extractor(settings, StubChatModel()), RuleFactExtractor)
