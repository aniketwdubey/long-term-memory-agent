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
from typing import NamedTuple

from engram.config import Settings
from engram.eval.cases import CaseKind, EvalCase, load_cases
from engram.eval.metrics import ArmReport, CaseResult, score_answer, score_store
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
)


def run_case(agent: Agent, case: EvalCase, *, memory: bool) -> CaseResult:
    """Replay one case's history, ask the probe in a new thread, and score it."""
    user = case.user_key

    for session in case.sessions:
        for turn in session.turns:
            agent.chat(user, f"{case.id}:{session.thread_id}", turn)

    trace = agent.chat(user, f"{case.id}:{case.probe_thread_id}", case.probe)
    passed, reason = score_answer(trace.reply, case.expect_any, case.reject_any)

    # Score only memories that are still live. A record that was correctly
    # superseded or expired is not clutter — keeping it as history is the right
    # behaviour, and precision must not punish a system for doing that.
    stored = [m for m in agent.reader.all_memories(user) if m.is_active()] if memory else []
    gold_covered, gold_total, useful, total = score_store(stored, case.gold_facts)

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
    reports = [run_arm(settings, cases, arm) for arm in ARMS]

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
