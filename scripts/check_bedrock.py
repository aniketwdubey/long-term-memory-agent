"""Preflight for the live Bedrock path, and a bake-off between models.

Everything else in this project runs offline against the stub, which is what
makes CI trustworthy — and also means the live path can rot silently. It did:
the first real run found that models return ``ttl_days: 0`` for durable facts,
which taken literally expires them the instant they are written. No offline test
could have caught that, because the stub never emits a 0.

What it checks is the *system*, not raw model output: the manager's TTL guard is
applied here too, and the report says when it had to step in — a model proposing
expiries nobody asked for is worth knowing about even when policy code catches it.

Run this after changing the extraction prompt or the model:

    python scripts/check_bedrock.py
    python scripts/check_bedrock.py --models amazon.nova-micro-v1:0 amazon.nova-lite-v1:0

It costs a few cents. Defaults to Amazon's own models, which AWS credits cover.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_aws import ChatBedrockConverse  # noqa: E402

from engram.config import Settings  # noqa: E402
from engram.embeddings import build_embedder  # noqa: E402
from engram.memory.extract import LLMFactExtractor, has_temporal_marker  # noqa: E402

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"

DEFAULT_MODELS = ("amazon.nova-lite-v1:0", "amazon.nova-micro-v1:0")

ANY_SLOT = "*"

# (turn, expected slot, expectation) where expectation is one of:
#   "ignore"     nothing should be stored at all
#   "durable"    stored with no expiry
#   "temporary"  stored WITH an expiry — the user framed it as passing, and the
#                point of decay is that it lapses on its own
# Slot "*" means any slot will do.
PROBES: tuple[tuple[str, str, str], ...] = (
    ("Morning! The coffee machine is broken again.", "", "ignore"),
    ("Did you see the standup notes from yesterday?", "", "ignore"),
    ("I'm on the payments team.", "team", "durable"),
    ("I prefer pytest for everything I write.", "testing_framework", "durable"),
    ("I've relocated to Berlin.", "location", "durable"),
    ("We run Postgres in production.", "database", "durable"),
    ("Always show me the SQL before running it.", "", "durable"),
    ("I'm debugging a flaky test today.", ANY_SLOT, "temporary"),
    ("I'm on call this week.", ANY_SLOT, "temporary"),
)

# Dated turns, checked separately: the fact must carry a resolved date and must
# NOT carry an unanchored relative reference. A memory that says "yesterday"
# with nothing to anchor it can never answer a question about when.
DATED_PROBES: tuple[tuple[str, str], ...] = (
    ("[7 May 2023] Caroline: I went to the support group yesterday.", "6 May"),
    ("[25 May 2023] Melanie: I ran a charity race last Sunday.", "May"),
)
_RELATIVE = ("yesterday", "last sunday", "last week", "next month", "recently")


def check_embeddings(settings: Settings) -> bool:
    print(f"\n{BOLD}Embeddings{RESET}  {settings.bedrock_embed_model_id}")
    try:
        spec = build_embedder(settings.model_copy(update={"embedder": "bedrock"}))
        vector = spec.embeddings.embed_query("I prefer pytest")
    except Exception as exc:
        print(f"  {RED}✗{RESET} {type(exc).__name__}: {str(exc)[:150]}")
        return False
    print(f"  {GREEN}✓{RESET} {len(vector)} dimensions")
    return True


def check_model(settings: Settings, model_id: str) -> bool:
    print(f"\n{BOLD}Extraction{RESET}  {model_id}")
    try:
        model = ChatBedrockConverse(
            model=model_id,
            region_name=settings.aws_region,
            max_tokens=settings.max_tokens,
        )
        extractor = LLMFactExtractor(model)
    except Exception as exc:
        print(f"  {RED}✗{RESET} could not build: {type(exc).__name__}: {str(exc)[:150]}")
        return False

    ok = True
    for turn, expected_slot, expectation in PROBES:
        facts = extractor.extract(turn)

        # Apply the same TTL guard the manager applies, so this reports what the
        # *system* does rather than raw model output — and say when the guard
        # had to step in, because that is still a signal about the model.
        guard_fired = False
        if not has_temporal_marker(turn):
            kept = []
            for fact in facts:
                if fact.ttl_days is not None:
                    guard_fired = True
                    fact = fact.model_copy(update={"ttl_days": None})
                kept.append(fact)
            facts = kept

        if expectation == "ignore":
            good = not facts
            detail = "ignored" if good else f"stored {[f.text[:34] for f in facts]}"
        elif not facts:
            good, detail = False, "extracted nothing"
        else:
            fact = facts[0]
            slot_ok = expected_slot == ANY_SLOT or fact.attribute == expected_slot
            ttl_ok = (fact.ttl_days is None) == (expectation == "durable")
            good = slot_ok and ttl_ok
            detail = f"{fact.attribute or '(unslotted)'}={fact.value or '-'}"
            if not slot_ok:
                detail += f" (expected slot {expected_slot!r})"
            if not ttl_ok:
                detail += (
                    " (no expiry on a temporary state)"
                    if expectation == "temporary"
                    else f" ttl={fact.ttl_days}"
                )
        if guard_fired:
            detail += f" {DIM}[guard dropped an unsupported ttl]{RESET}"

        ok &= good
        mark = f"{GREEN}✓{RESET}" if good else f"{RED}✗{RESET}"
        print(f"  {mark} {turn[:44]:<46} {DIM}{detail}{RESET}")

    for turn, expected in DATED_PROBES:
        facts = extractor.extract(turn)
        if not facts:
            good, detail = False, "extracted nothing"
        else:
            text = facts[0].text
            dangling = [r for r in _RELATIVE if r in text.lower()]
            good = expected.lower() in text.lower() and not dangling
            detail = text[:52]
            if dangling:
                detail += f" (unanchored: {dangling[0]})"
            elif not good:
                detail += f" (no resolved date, wanted {expected!r})"
        ok &= good
        mark = f"{GREEN}✓{RESET}" if good else f"{RED}✗{RESET}"
        print(f"  {mark} {turn[:44]:<46} {DIM}{detail}{RESET}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the live Bedrock path.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    settings = Settings()
    if args.region:
        settings = settings.model_copy(update={"aws_region": args.region})

    print(f"{BOLD}Bedrock preflight{RESET}  ·  region={settings.aws_region}")
    results = {"embeddings": check_embeddings(settings)}
    for model_id in args.models:
        results[model_id] = check_model(settings, model_id)

    print(f"\n{BOLD}Summary{RESET}")
    for name, ok in results.items():
        print(f"  {GREEN}✓{RESET} {name}" if ok else f"  {RED}✗{RESET} {name}")
    print(
        f"\n{DIM}A ✗ on extraction is a prompt problem, not an outage — the model "
        f"answered, it just answered badly. Fix engram/memory/extract.py and "
        f"re-run.{RESET}"
    )
    # Only a hard failure (credentials, access, a model that will not respond)
    # is worth a non-zero exit; extraction quality is a judgement call.
    return 0 if results["embeddings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
