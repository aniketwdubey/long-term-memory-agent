"""Extraction: turning a conversational turn into candidate facts.

This is the first decision in the write path, and the one that fixes precision:
most turns contain nothing worth remembering, and the ones that do usually bury
a fact inside chatter. A turn goes in; zero or more normalised candidates come
out.

The important output is not the text but the **slot** — the ``(attribute,
value)`` pair. "I switched to the platform team" becomes
``team = platform``, which is what later lets policy code recognise it as
contradicting ``team = payments`` without asking a model to adjudicate. Getting
messy human phrasing into that shape is exactly what a language model is good
at, so extraction is where the LLM belongs.

Two implementations, chosen by ``ENGRAM_CHAT_PROVIDER``:

``LLMFactExtractor``   the real one — structured output from Claude on Bedrock.
``RuleFactExtractor``  a deterministic offline stand-in. **It is a test fixture,
                       not a small language model**, and its job is to make CI
                       hermetic while holding extraction quality constant, so
                       that the numbers move only when the dedupe / conflict /
                       decay policy moves. Extraction quality itself is measured
                       by running the benchmark against Bedrock.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from engram.schemas import MemoryKind

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

log = structlog.get_logger(__name__)


class CandidateFact(BaseModel):
    """One fact proposed for storage, before dedupe and conflict resolution."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, description="The fact in canonical, self-contained form.")
    attribute: str = Field(
        default="",
        description=(
            "Normalised slot key such as 'team', 'location', 'editor'. Two facts "
            "with the same attribute and scope cannot both be true. Empty when "
            "the fact does not occupy a named slot."
        ),
    )
    value: str = Field(default="", description="The value filling that slot.")
    scope: str = Field(
        default="",
        description="Optional qualifier ('work', 'home') letting two values coexist.",
    )
    kind: MemoryKind = MemoryKind.SEMANTIC
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_days: int | None = Field(
        default=None,
        description=(
            "Days until this should be forgotten. Use null — never 0 — for "
            "durable facts that should never expire."
        ),
    )

    @field_validator("ttl_days")
    @classmethod
    def _zero_means_durable(cls, value: int | None) -> int | None:
        """Treat a non-positive TTL as "no expiry".

        Real models return ``ttl_days: 0`` for durable facts however plainly the
        prompt asks for null — both Nova Micro and Nova Lite did it on the first
        live run. Taken literally that is an expiry of *now*, so every standing
        preference would be written already dead. The prompt asks for null and
        this makes it not matter.
        """
        return None if value is not None and value <= 0 else value


class ExtractionResult(BaseModel):
    """The structured-output envelope for one turn."""

    facts: list[CandidateFact] = Field(
        default_factory=list,
        description="Facts worth remembering. Empty for small talk — most turns.",
    )


@runtime_checkable
class FactExtractor(Protocol):
    """Turn text in, candidate facts out."""

    def extract(self, turn: str) -> list[CandidateFact]: ...


EXTRACTION_SYSTEM = """\
You extract durable facts about a user from one conversational turn, so an \
assistant can remember them in later sessions.

Return facts ONLY when the turn states something about the user that would \
still be useful weeks from now: preferences, tools they use, where they live, \
their role or team, constraints, dietary needs, and standing instructions about \
how they want things done.

Most turns contain nothing. Small talk, status updates, complaints about the \
office, questions, and requests are NOT facts about the user:
  "Morning! The coffee machine is broken again."  -> no facts
  "Did you see the standup notes?"                -> no facts
  "Scaffold me a test for the refund endpoint."   -> no facts
  "We finally shipped the dashboard on Friday."   -> no facts

A QUESTION IS NEVER A FACT. When the user asks about something, they are \
telling you they do not know it — that is the opposite of stating it. Never \
turn the subject of a question into a slot:
  "Which team am I on again?"        -> no facts (NOT team="")
  "What access do I have?"           -> no facts (NOT access="unknown")
  "Where did I say I lived?"         -> no facts
  "Who should review my change?"     -> no facts

Never emit a fact whose `value` is empty, "unknown", or "unspecified". If you \
do not know the value, there is no fact — return nothing.

Returning an empty list is the common, correct answer. Inventing facts to seem \
useful is the main failure mode.

For each fact:
- `text` — a self-contained sentence in the third person, understandable with \
no other context. Write "Prefers pytest for testing", not "yeah I like it".

  If the turn begins with a date in brackets, that is when it was said. RESOLVE \
relative time against it and write the resolved date into the fact. A memory \
that still says "yesterday" is anchored to nothing and can never answer a \
question about when something happened:
    "[7 May 2023] Caroline: I went to the support group yesterday"
      -> "Caroline went to the LGBTQ support group on 6 May 2023"
    "[25 May 2023] Melanie: I ran a charity race last Sunday"
      -> "Melanie ran a charity race on Sunday 21 May 2023"
    "[3 June 2023] Sam: I'm moving to Berlin next month"
      -> "Sam is moving to Berlin in July 2023"
  Never carry "yesterday", "last week", "next month" or "recently" into a fact \
unresolved. Do not put the bracketed date in the fact verbatim — resolve it.
- `attribute` — a short snake_case slot key. Two facts that cannot both be true \
at once MUST share an attribute, and the same fact restated later MUST get the \
same key, so prefer one of these wherever it fits:
  team, role, employer, location, timezone, working_hours, language, \
  testing_framework, web_framework, database, editor, deploy_tool, cloud, os, \
  diet, allergy, accessibility, pet_name, communication_preference
  Invent a new snake_case key only when none of them fits, and keep it generic \
(`editor`, not `preferred_code_editor`). Leave it empty for standing \
instructions that do not name a property of the user.
- `value` — the value filling the slot, normalised and short: "platform", \
"Berlin", "pytest".
- `scope` — set only when two values of the same attribute can genuinely both \
be true (a work address and a home address). Otherwise leave it empty.
- `kind` — `semantic` for durable facts, `episodic` for things that happened, \
`procedural` for standing instructions about how to do things.
- `importance` — 0 to 1. Standing preferences and constraints score high.
- `ttl_days` — null for almost everything. Set a number ONLY when the user's \
own words mark the state as temporary. Do not invent an expiry for something \
that simply feels like it might change one day.
  "I'm debugging a flaky test today"   -> ttl_days: 1
  "I'm on call this week"              -> ttl_days: 7
  "I prefer pytest"                    -> ttl_days: null
  "I've relocated to Berlin"           -> ttl_days: null
  "I'm on the payments team"           -> ttl_days: null
  Never use 0. A 0 means "expired already" and throws the fact away.
"""


class LLMFactExtractor:
    """Extraction by structured output from the chat model."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(ExtractionResult)

    def extract(self, turn: str) -> list[CandidateFact]:
        try:
            result = self._model.invoke(
                [SystemMessage(content=EXTRACTION_SYSTEM), HumanMessage(content=turn)]
            )
        except Exception as exc:
            # Never let a memory write take down the conversation. A turn that
            # cannot be extracted is a turn that is not remembered, which is
            # recoverable; a failed response is not.
            log.warning("memory.extract.failed", error=str(exc), chars=len(turn))
            return []

        facts = (
            list(result.facts)
            if isinstance(result, ExtractionResult)
            else list(ExtractionResult.model_validate(result).facts)
        )
        return facts


# A leading "[7 May 2023] " marks when a turn was said. Conversation replayed
# from a transcript needs it: the record's created_at is when it was *ingested*,
# which for a dialogue from last year is not when anything happened.
_DATED_TURN = re.compile(r"^\s*\[([^\]]{3,40})\]\s*")


def turn_date(turn: str) -> str:
    """The bracketed date a turn was said on, if it carries one.

    The prompt asks the model to resolve relative time against this. An earlier
    version *also* appended it to any fact that still said "yesterday". That was
    reverted: it grew the store 32% (a date in the text makes near-duplicates
    look distinct, defeating dedupe) and moved no metric in two measured runs.
    Carrying it as a field the embedding never sees is the right shape if this
    is revisited — mutating the text that retrieval depends on is not.
    """
    match = _DATED_TURN.match(turn)
    return match.group(1).strip() if match else ""


# --- the offline stand-in ----------------------------------------------------

# Sentences that state something about the speaker. Deliberately about grammar
# rather than topic, so which *subjects* are recognised is not baked in here.
#
# The verb list is kept tight on purpose. Widening it to catch more facts costs
# precision fast — "I need to renew my passport" and "I'm going to grab lunch"
# are errands, not facts about the user, and a write path that stores them is
# the naive baseline wearing a costume.
_SUBJECT = r"\b(?:i|we)(?:'(?:ve|d|ll))?"
_STATIVE_VERBS = (
    r"prefer|like|love|hate|use|run|write|deploy|live|living|work|working|"
    r"joined|switched|moved|relocated|migrated|eat|am|are"
)

_FACT_PATTERNS = (
    re.compile(rf"{_SUBJECT}\s+(?:\w+\s+){{0,2}}?(?:{_STATIVE_VERBS})\b", re.IGNORECASE),
    # "I'm on the payments team", "I'm vegetarian", "I'm based in Lisbon" —
    # a predicate describing the speaker.
    re.compile(r"\bi'?m\s+(?:an?\s+)?(?!\w+ing\b)\w+", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:\w+\s+){1,2}(?:is|are)\b", re.IGNORECASE),
)

# A progressive predicate describes an activity, not a state: "I'm debugging a
# flaky test", "my sister is visiting". Neither is a fact about the user worth
# keeping — unless the user marked it as a temporary state, in which case it is
# exactly the kind of memory that should be stored *and then expire*.
_ACTIVITY_PATTERN = re.compile(r"\b(?:i'?m|is|are)\s+\w+ing\b", re.IGNORECASE)

# Standing instructions — how the user wants things done.
_DIRECTIVE_PATTERN = re.compile(
    r"^\s*(?:always|never|don'?t|do not|please\s+(?:always|never))\b", re.IGNORECASE
)

# Narrow: what makes an *ongoing activity* worth storing at all. Widening this
# is how "my sister is visiting next month" becomes a fact about the user.
_TRANSIENT_PATTERN = re.compile(
    r"\b(?:today|tonight|right now|this (?:morning|afternoon|week|sprint))\b",
    re.IGNORECASE,
)
_TRANSIENT_TTL_DAYS = 7

# Wide: what counts as the user having framed something as temporary at all.
# The two err in opposite directions on purpose. Storing an activity costs a
# little precision; wrongly rejecting a real expiry makes a temporary fact
# permanent, which is a silent, long-lived wrong answer.
_TEMPORAL_MARKER_PATTERN = re.compile(
    r"\b(?:today|tonight|tomorrow|right now|currently|for now|at the moment|"
    r"this (?:morning|afternoon|evening|week|month|sprint|quarter|year)|"
    r"next (?:week|month|quarter)|until \w+|for the next \w+|"
    r"temporarily|for a (?:while|bit)|these days|on call|this time)\b",
    re.IGNORECASE,
)


def has_temporal_marker(text: str) -> bool:
    """Whether the text frames itself as temporary in the user's own words.

    Used to check a model's proposed expiry against the turn it came from. Both
    extractors propose TTLs, but only the user can actually make a fact
    temporary — a model that decides "prefers pytest" lapses in a year has
    invented a deadline nobody set. See MemoryManager for the enforcement.
    """
    return bool(_TEMPORAL_MARKER_PATTERN.search(text))


# The slot lexicon. This is the part a real model does far better, and it is the
# reason the rule extractor is documented as a fixture: a lexicon can only
# recognise vocabulary someone thought of in advance. Anything it misses falls
# through as an unslotted fact — still stored, still deduplicated, just never
# superseding anything.
_SLOT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("team", re.compile(r"\b(?:the\s+)?([a-z]+)\s+team\b", re.IGNORECASE)),
    (
        "location",
        re.compile(
            r"\b(?i:live in|living in|based in|relocated to|moved to|i'?m in)\s+"
            r"([A-Z][a-zA-Z]+)"
        ),
    ),
    ("testing_framework", re.compile(r"\b(pytest|unittest|jest|vitest|junit|rspec)\b", re.I)),
    ("web_framework", re.compile(r"\b(fastapi|flask|django|express|rails)\b", re.I)),
    ("database", re.compile(r"\b(postgres|postgresql|mysql|sqlite|mongodb|dynamodb)\b", re.I)),
    ("editor", re.compile(r"\b(neovim|zed|emacs|vs ?code|pycharm|intellij|sublime)\b", re.I)),
    ("language", re.compile(r"\b(typescript|javascript|rust|kotlin|golang)\b", re.I)),
    ("deploy_tool", re.compile(r"\b(terraform|pulumi|ansible|cloudformation|helm)\b", re.I)),
    ("diet", re.compile(r"\b(vegetarian|vegan|pescatarian|halal|kosher)\b", re.I)),
    ("accessibility", re.compile(r"\b(colou?rblind|colou?r[- ]blind|dyslexic)\b", re.I)),
    ("pet_name", re.compile(r"\b(?i:my (?:dog|cat))\s+([A-Z][a-zA-Z]+)")),
)

# Split on sentence terminators only. An em dash *joins* clauses — splitting on
# it tears "I've moved off payments — I'm on the platform team now" into two
# candidates, and the dangling first half gets stored as an unslotted fact that
# still contains the stale value, so the contradiction survives resolution.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class RuleFactExtractor:
    """Deterministic offline extraction. A fixture, not a model — see the module docstring.

    It keeps sentences that grammatically state something about the speaker,
    drops everything else, and assigns a slot from a fixed lexicon. That is
    enough to hold extraction roughly constant so the policy downstream of it
    can be measured, and no more than that.
    """

    def extract(self, turn: str) -> list[CandidateFact]:
        facts: list[CandidateFact] = []
        body = _DATED_TURN.sub("", turn.strip())
        for raw in _SENTENCE_SPLIT.split(body):
            sentence = raw.strip().rstrip(".!?").strip()
            if not sentence:
                continue

            directive = bool(_DIRECTIVE_PATTERN.search(sentence))
            transient = bool(_TRANSIENT_PATTERN.search(sentence))

            if not directive and not self._is_fact(sentence, transient=transient):
                continue

            attribute, value = self._slot(sentence)
            facts.append(
                CandidateFact(
                    text=sentence,
                    attribute=attribute,
                    value=value,
                    kind=MemoryKind.PROCEDURAL if directive else MemoryKind.SEMANTIC,
                    importance=0.8 if directive or attribute else 0.5,
                    ttl_days=_TRANSIENT_TTL_DAYS if transient else None,
                )
            )
        return facts

    @staticmethod
    def _is_fact(sentence: str, *, transient: bool) -> bool:
        """Whether this sentence states something about the speaker worth keeping."""
        if _ACTIVITY_PATTERN.search(sentence):
            # An ongoing activity only earns a memory when the user marked it as
            # temporary — and it will then expire on its own.
            return transient
        return any(p.search(sentence) for p in _FACT_PATTERNS)

    @staticmethod
    def _slot(sentence: str) -> tuple[str, str]:
        """First lexicon hit wins; unslotted otherwise."""
        for attribute, pattern in _SLOT_PATTERNS:
            match = pattern.search(sentence)
            if match:
                value = (match.group(1) if match.groups() else match.group(0)).strip()
                return attribute, value.lower()
        return "", ""


def build_extractor(settings: Settings, model: BaseChatModel) -> FactExtractor:
    """Pick the extractor matching the configured chat provider."""
    if settings.chat_provider == "stub":
        return RuleFactExtractor()
    return LLMFactExtractor(model)
