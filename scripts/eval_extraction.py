"""Score live fact extraction on frozen, labeled synthetic turns.

Run before and after a prompt or policy change. The development split diagnoses
failures; holdout is for the final comparison. This measures turn-level selection,
not the memory-store precision reported by the three-arm benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from engram.config import Settings
from engram.memory.extract import EXTRACTION_SYSTEM, LLMFactExtractor
from engram.models import build_chat_model


class Probe(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    split: Literal["development", "holdout"]
    turn: str = Field(min_length=1)
    expected: str


def load_probes(path: Path) -> list[Probe]:
    probes: list[Probe] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        probe = Probe.model_validate_json(line)
        if probe.id in seen:
            raise ValueError(f"{path}:{number}: duplicate probe {probe.id!r}")
        seen.add(probe.id)
        probes.append(probe)
    if not probes or {p.split for p in probes} != {"development", "holdout"}:
        raise ValueError("Both development and holdout probes are required")
    return probes


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    positive = [r for r in rows if r["expected"]]
    negative = [r for r in rows if not r["expected"]]
    true_positive = sum(r["result"] == "correct" for r in positive)
    false_positive = sum(r["result"] == "false_positive" for r in negative)
    false_negative = len(positive) - true_positive
    return {
        "turns": len(rows),
        "positive_turns": len(positive),
        "negative_turns": len(negative),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "turn_precision": round(true_positive / (true_positive + false_positive), 4)
        if true_positive + false_positive
        else 0.0,
        "positive_recall": round(true_positive / len(positive), 4) if positive else 0.0,
        "negative_rejection": round((len(negative) - false_positive) / len(negative), 4)
        if negative
        else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Live extraction probe")
    parser.add_argument("--cases", type=Path, default=Path("eval/cases/extraction.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings(_env_file=None, chat_provider="bedrock", embedder="hashing")
    probes = load_probes(args.cases)
    extractor = LLMFactExtractor(build_chat_model(settings))
    rows: list[dict[str, object]] = []
    for probe in probes:
        facts = extractor.extract(probe.turn)
        texts = [f.text for f in facts]
        found = bool(probe.expected) and any(
            probe.expected.casefold() in f"{f.text} {f.value}".casefold() for f in facts
        )
        if not probe.expected:
            outcome = "false_positive" if facts else "correct"
        else:
            outcome = "correct" if found else "wrong_fact" if facts else "missed"
        rows.append(
            {
                "id": probe.id,
                "split": probe.split,
                "turn": probe.turn,
                "expected": probe.expected,
                "result": outcome,
                "facts": [f.model_dump(mode="json") for f in facts],
            }
        )
        print(f"{probe.id} {outcome} candidates={len(texts)}", flush=True)
    report = {
        "schema_version": 1,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(EXTRACTION_SYSTEM.encode()).hexdigest(),
        "model_id": settings.bedrock_model_id,
        "aws_region": settings.aws_region,
        "summary": {
            split: summarize([r for r in rows if r["split"] == split])
            for split in ("development", "holdout")
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
