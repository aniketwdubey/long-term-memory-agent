"""The benchmark itself — cases, scoring, and the guards that keep it honest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from engram.config import Settings
from engram.eval.cases import CaseKind, EvalCase, load_cases
from engram.eval.metrics import ArmReport, CaseResult, score_answer, score_store
from engram.eval.runner import MEMORY, STATELESS, run_arm
from engram.schemas import Memory

CASES_PATH = Path(__file__).resolve().parent.parent / "eval" / "cases" / "slice1.jsonl"


def _case(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "c1",
        "kind": "passive_recall",
        "user_id": "alice",
        "sessions": [{"thread_id": "s1", "turns": ["I prefer pytest."]}],
        "probe": "What framework do I prefer?",
        "probe_thread_id": "probe",
        "expect_any": ["pytest"],
    }
    return {**base, **overrides}


# --- case validity -----------------------------------------------------------


def test_probe_may_not_reuse_a_session_thread() -> None:
    """The guard against a benchmark that flatters every arm equally.

    If the probe is asked in a thread where the fact was stated, the checkpointer
    answers it and long-term memory is never consulted — so even the stateless
    control would score 100%.
    """
    with pytest.raises(ValidationError, match="thread memory"):
        EvalCase.model_validate(_case(probe_thread_id="s1"))


def test_case_needs_at_least_one_expectation() -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate(_case(expect_any=[]))


def test_user_key_namespaces_by_case_so_cases_cannot_leak() -> None:
    """Two cases reusing a name must not share a memory store."""
    a = EvalCase.model_validate(_case(id="c1", user_id="alice"))
    b = EvalCase.model_validate(_case(id="c2", user_id="alice"))
    assert a.user_key != b.user_key


# --- loader ------------------------------------------------------------------


def test_loader_skips_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(f"# a comment\n\n{json.dumps(_case())}\n", encoding="utf-8")
    assert len(load_cases(path)) == 1


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    line = json.dumps(_case())
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(path)


def test_loader_reports_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(f"{json.dumps(_case())}\n{{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_cases(path)


def test_loader_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no cases"):
        load_cases(path)


def test_the_committed_benchmark_loads_and_covers_every_kind() -> None:
    cases = load_cases(CASES_PATH)
    assert len(cases) >= 20
    assert {c.kind for c in cases} == set(CaseKind)


# --- answer scoring ----------------------------------------------------------


def test_answer_passes_on_any_expected_string() -> None:
    assert score_answer("use WEST time", ["Lisbon", "WEST"], [])[0]


def test_answer_matching_is_case_insensitive() -> None:
    assert score_answer("Use PyTest", ["pytest"], [])[0]


def test_a_forbidden_string_fails_even_when_the_expected_one_is_present() -> None:
    """The conflict case: surfacing the stale fact alongside the new one is wrong."""
    passed, reason = score_answer("you live in Berlin, formerly Mumbai", ["Berlin"], ["Mumbai"])
    assert not passed
    assert "Mumbai" in reason


def test_missing_expectation_reports_what_was_wanted() -> None:
    passed, reason = score_answer("no idea", ["pytest"], [])
    assert not passed
    assert "pytest" in reason


# --- store scoring -----------------------------------------------------------


def _mem(text: str) -> Memory:
    return Memory(user_id="alice", text=text)


def test_store_scoring_separates_recall_from_precision() -> None:
    memories = [_mem("I prefer pytest."), _mem("The coffee machine is broken.")]
    covered, total, useful, stored = score_store(memories, ["pytest"])
    assert (covered, total) == (1, 1)  # the fact was kept
    assert (useful, stored) == (1, 2)  # but so was junk


def test_store_scoring_handles_an_empty_store() -> None:
    assert score_store([], ["pytest"]) == (0, 1, 0, 0)


# --- aggregation -------------------------------------------------------------


def _result(kind: CaseKind, passed: bool) -> CaseResult:
    return CaseResult(case_id="x", kind=kind, passed=passed, reason="", reply="")


def test_rates_are_computed_per_kind() -> None:
    report = ArmReport(
        arm="t",
        results=[
            _result(CaseKind.PASSIVE_RECALL, True),
            _result(CaseKind.DECISION_RELEVANT, True),
            _result(CaseKind.DECISION_RELEVANT, False),
        ],
    )
    assert report.rate(CaseKind.PASSIVE_RECALL) == 1.0
    assert report.rate(CaseKind.DECISION_RELEVANT) == 0.5
    assert report.rate() == pytest.approx(2 / 3)


def test_rates_of_an_empty_arm_are_zero_not_undefined() -> None:
    empty = ArmReport(arm="t")
    assert empty.rate() == 0.0
    assert empty.memory_precision == 0.0
    assert empty.avg_memories_injected == 0.0


# --- end to end --------------------------------------------------------------


def test_memory_arm_beats_the_stateless_arm_on_the_real_benchmark(
    settings: Settings,
) -> None:
    """The claim the whole slice exists to support, asserted rather than asserted about."""
    cases = load_cases(CASES_PATH)[:6]

    stateless = run_arm(settings, cases, STATELESS, memory=False)
    memory = run_arm(settings, cases, MEMORY, memory=True)

    assert stateless.rate() == 0.0
    assert memory.rate() > stateless.rate()
    assert memory.avg_memories_injected <= settings.recall_top_k
