"""Run the benchmark: replay each case under each arm and report the deltas.

Every arm is built from the same graph code, so a difference between columns is
a difference in the write path and nothing else. Each case's history is replayed
turn by turn in its own threads, and only then is the probe asked in a fresh
thread.

    python -m engram.eval.runner eval/cases/slice1.jsonl

The exit code is a CI regression gate: the ``--fail-under-*`` floors fail the
build when the shipping arm regresses.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any, NamedTuple

from engram.config import Settings
from engram.eval.cases import CaseKind, EvalCase, load_cases
from engram.eval.metrics import (
    ArmReport,
    CaseResult,
    score_answer,
    score_injection,
    score_store,
)
from engram.graph import Agent
from engram.logging import configure_logging
from engram.store import open_backend


class Arm(NamedTuple):
    """One configuration of the agent to benchmark."""

    label: str
    memory: bool
    writer: str


# Three arms, all compiled from the same graph so the comparison isolates the
# write path: no memory at all, memory written naively, and memory written by
# the manager.
ARMS = (
    Arm("stateless", memory=False, writer="naive"),
    Arm("naive", memory=True, writer="naive"),
    Arm("manager", memory=True, writer="manager"),
    # Opt-in: mem0 calls a real LLM on every write, so it cannot be part of the
    # offline default run. Add it with `--arms manager mem0`.
    Arm("mem0", memory=True, writer="mem0"),
)

DEFAULT_ARMS = tuple(a.label for a in ARMS if a.writer != "mem0")


def run_case(agent: Any, case: EvalCase, *, memory: bool) -> CaseResult:
    """Replay one case's history, ask the probe in a new thread, and score it."""
    user = case.user_key

    for session in case.sessions:
        thread = f"{case.id}:{session.thread_id}"
        for turn in session.turns:
            if turn.spoken_by_user:
                agent.chat(user, thread, turn.text)
            else:
                # Content the agent read rather than content the user said. It
                # goes to the write path with its real provenance — which is the
                # entire question the injection cases ask.
                agent.observe(user, turn.text, thread_id=thread, source=turn.source)

    trace = agent.chat(user, f"{case.id}:{case.probe_thread_id}", case.probe)
    passed, reason = score_answer(trace.reply, case.expect_any, case.reject_any)

    # Score only memories that are still live. A record that was correctly
    # superseded or expired is not clutter — keeping it as history is the right
    # behaviour, and precision must not punish a system for doing that.
    stored = [m for m in agent.reader.all_memories(user) if m.is_active()] if memory else []
    gold_covered, gold_total, useful, total = score_store(stored, case.gold_facts)
    poison_blocked, poison_total = score_injection(stored, case.poison_markers)

    return CaseResult(
        case_id=case.id,
        kind=case.kind,
        passed=passed,
        reason=reason,
        reply=trace.reply,
        recalled=[r.memory.text for r in trace.recalled],
        stored=len(stored),
        gold_covered=gold_covered,
        gold_total=gold_total,
        useful_memories=useful,
        total_memories=total,
        poison_blocked=poison_blocked,
        poison_total=poison_total,
    )


def run_arm(settings: Settings, cases: Sequence[EvalCase], arm: Arm) -> ArmReport:
    """Run every case under one arm, in a store dedicated to that arm.

    The store is always in-process, whatever ``ENGRAM_STORE_BACKEND`` says. A
    benchmark has to start from an empty store every time: pointed at Postgres,
    a second run would replay every case's history on top of the first run's
    memories, quietly inflating the store and changing the numbers. What is
    being measured here is the memory *algorithms*, not the storage backend —
    the durability of that backend is what the Postgres tests are for.
    """
    if arm.writer == "mem0":
        # mem0 owns its own storage, so it does not use our backend at all.
        from engram.eval.baselines import build_mem0_agent

        mem0_agent = build_mem0_agent(settings)
        results = [run_case(mem0_agent, case, memory=True) for case in cases]
        return ArmReport(arm=arm.label, results=results)

    settings = settings.model_copy(update={"store_backend": "memory", "memory_writer": arm.writer})
    with open_backend(settings) as backend:
        agent = Agent(
            settings,
            checkpointer=backend.checkpointer,
            store=backend.store,
            embeddings=backend.embeddings,
            memory=arm.memory,
        )
        results = [run_case(agent, case, memory=arm.memory) for case in cases]
    return ArmReport(arm=arm.label, results=results)


def _sample_per_kind(cases: Sequence[EvalCase], limit: int) -> list[EvalCase]:
    """Up to ``limit`` cases of each kind, in file order.

    Balanced rather than a prefix: the case file is grouped by kind, so taking
    the first N would run nothing but passive recall and report a number that
    looks great and means nothing.
    """
    seen: dict[CaseKind, int] = {}
    out: list[EvalCase] = []
    for case in cases:
        count = seen.get(case.kind, 0)
        if count < limit:
            seen[case.kind] = count + 1
            out.append(case)
    return out


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def render_report(reports: Sequence[ArmReport], cases: Sequence[EvalCase]) -> str:
    """Render the comparison table."""
    kinds = [k for k in CaseKind if any(c.kind == k for c in cases)]
    width = max(max(len(r.arm) for r in reports) + 2, 11)

    lines: list[str] = []
    lines.append("")
    lines.append(f"{len(cases)} cases  ·  arms: {', '.join(r.arm for r in reports)}")
    lines.append("")

    header = f"{'metric':<34}" + "".join(f"{r.arm:>{width}}" for r in reports)
    lines.append(header)
    lines.append("-" * len(header))

    def row(label: str, values: list[str]) -> None:
        lines.append(f"{label:<34}" + "".join(f"{v:>{width}}" for v in values))

    for kind in kinds:
        n = reports[0].count(kind)
        label = kind.value.replace("_", "-")
        marker = "  <<<" if kind is CaseKind.DECISION_RELEVANT else ""
        row(f"{label} (n={n}){marker}", [_pct(r.rate(kind)) for r in reports])

    lines.append("")
    row(f"overall (n={len(cases)})", [_pct(r.rate()) for r in reports])
    lines.append("")
    if any(r.poison_total for report in reports for r in report.results):
        row("injection resistance", [_pct(r.injection_resistance) for r in reports])
    row("memory recall (kept the facts)", [_pct(r.memory_recall) for r in reports])
    row("memory precision (kept only)", [_pct(r.memory_precision) for r in reports])
    row("avg memories stored / user", [f"{r.avg_memories_stored:.1f}" for r in reports])
    row("avg memories injected / turn", [f"{r.avg_memories_injected:.1f}" for r in reports])
    lines.append("")

    for report in reports:
        failures = report.failures()
        if not failures:
            continue
        lines.append(f"failures — {report.arm}:")
        for f in failures:
            lines.append(f"  [{f.kind.value:<17}] {f.case_id:<28} {f.reason}")
        lines.append("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="engram.eval.runner", description="Benchmark long-term memory recall."
    )
    parser.add_argument("cases", help="Path to a JSONL case file.")
    parser.add_argument("--embedder", choices=["hashing", "fastembed", "bedrock"])
    parser.add_argument("--provider", choices=["stub", "bedrock"])
    parser.add_argument("--top-k", type=int, help="Memories injected per turn.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run at most N cases OF EACH KIND. For live runs, where every turn "
        "is an API call — a flat prefix would be all one kind and tell you nothing.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[a.label for a in ARMS],
        default=None,
        help="Run only these arms. Defaults to the three offline ones; `mem0` is "
        "opt-in because it calls a real LLM on every write.",
    )
    parser.add_argument(
        "--fail-under-decision",
        type=float,
        default=None,
        metavar="RATE",
        help="Exit non-zero if decision-relevant recall is below RATE (0-1).",
    )
    parser.add_argument(
        "--fail-under-conflict",
        type=float,
        default=None,
        metavar="RATE",
        help="Exit non-zero if conflict-resolution accuracy is below RATE (0-1).",
    )
    parser.add_argument(
        "--fail-under-injection",
        type=float,
        default=None,
        metavar="RATE",
        help="Exit non-zero if injection resistance is below RATE (0-1).",
    )
    parser.add_argument(
        "--fail-under-precision",
        type=float,
        default=None,
        metavar="RATE",
        help="Exit non-zero if memory precision is below RATE (0-1).",
    )
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.embedder:
        overrides["embedder"] = args.embedder
    if args.provider:
        overrides["chat_provider"] = args.provider
    if args.top_k:
        overrides["recall_top_k"] = args.top_k
    settings = Settings(**overrides)  # type: ignore[arg-type]
    configure_logging(settings)

    cases = load_cases(args.cases)
    if args.limit:
        cases = _sample_per_kind(cases, args.limit)
    selected = args.arms if args.arms is not None else DEFAULT_ARMS
    arms = [a for a in ARMS if a.label in selected]
    reports = [run_arm(settings, cases, arm) for arm in arms]

    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in reports], indent=2))
    else:
        print(render_report(reports, cases))

    # Gates apply to the last arm — the one that ships.
    shipped = reports[-1]
    gates = (
        ("decision-relevant recall", CaseKind.DECISION_RELEVANT, args.fail_under_decision),
        ("conflict resolution", CaseKind.CONFLICT, args.fail_under_conflict),
    )
    failed = False
    if (
        args.fail_under_injection is not None
        and shipped.injection_resistance < args.fail_under_injection
    ):
        print(
            f"FAIL: injection resistance {shipped.injection_resistance:.3f} "
            f"< {args.fail_under_injection:.3f}",
            file=sys.stderr,
        )
        failed = True
    for name, kind, floor in gates:
        if floor is None:
            continue
        actual = shipped.rate(kind)
        if actual < floor:
            print(f"FAIL: {name} {actual:.3f} < {floor:.3f}", file=sys.stderr)
            failed = True
    if (
        args.fail_under_precision is not None
        and shipped.memory_precision < args.fail_under_precision
    ):
        print(
            f"FAIL: memory precision {shipped.memory_precision:.3f} "
            f"< {args.fail_under_precision:.3f}",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
