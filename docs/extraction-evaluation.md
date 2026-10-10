# Live extraction evaluation

[Documentation index](README.md) · [Evaluation](evaluation.md) · [Status](status.md)

On 2026-10-10, we labeled 104 synthetic turns: 52 explicit user-profile
facts and 52 questions, requests, third-party reports, errands, plans, or
one-off events. The [probe script](../scripts/eval_extraction.py) sends each
turn through the production `LLMFactExtractor` using Amazon Nova Lite and
saves every proposed fact in JSON. The original 60 turns have development and
holdout splits; two later challenge sets contain 20 and 24 turns.

```bash
.venv/bin/python scripts/eval_extraction.py \
  --output eval/results/extraction-run.json
```

**Turn-level selection precision** is the proportion of turns that produced a
candidate and contained the labeled fact. Positive recall is the fraction of
fact turns whose expected fact appeared. These measures differ from the
end-to-end benchmark's **memory-store precision**, which scores records after
extraction, deduplication, conflict resolution, and retrieval.

| Sample | Version | Expected facts kept | Nonfacts incorrectly extracted | Turn precision |
|---|---|---:|---:|---:|
| Original 60 | No selection guard | 30/30 | 25/30 | 54.5% |
| Original 60 | Current grammar guard | 30/30 | 0/30 | 100% |
| First extra 20 | First guard | 10/10 | 4/10 | 71.4% |
| First extra 20 | Current grammar guard | 10/10 | 0/10 | 100% |
| Broader extra 24 | Earlier guard | 12/12 | 6/12 | 66.7% |
| Broader extra 24 | Current grammar guard | 12/12 | 0/12 | 100% |

The end-to-end manager benchmark used the same balanced eight-case selection
before and after the grammar revision. The earlier selection guard scored
**48.8% store precision, 69.6% labeled-fact recall, and 75.0% answer accuracy**.
The grammar guard scored **81.1% store precision, 76.8% labeled-fact recall, and
100% answer accuracy**. Both runs blocked every labeled injection. The manager
stored 10.0 memories per user in the earlier run and 6.6 in the later one.
Raw machine-readable reports are in `eval/results/manager-eight-*.json`.

An older published run of the same case selection reported 31.6% store
precision. That run predated this evaluation and did not save per-case raw
results. Nova outputs vary between runs, so the eight-case comparison is a
useful signal rather than a stable improvement estimate.

The original holdout and both extra sets exposed failures that informed later
repairs. **None of the final 100% figures is an independent estimate.** They
are repeatable regression results on a fixed, labeled corpus. The first run of
each extra set is useful evidence of how the preceding version generalized.
A larger independently labeled sample from actual use remains necessary for a
production quality claim.

The guard first rejects clear questions, one-off requests, hypotheticals,
third-party reports, errands, and transient events. It then asks whether the
turn contains grammatical evidence of a user state, preference, standing
instruction, or lasting change. This reuses the offline fixture's broad
speaker grammar, with additional cues for needs, team technology, and dated
personal episodes. Turns with that evidence still go to Nova for normalization
and slot selection. The filter does not read the database or any other user's
text, and it only affects the live model path. The offline regression benchmark
retains its original deterministic extractor.

The [end-to-end comparison](evaluation.md#the-same-benchmark-on-real-models)
is needed to see whether better selection also reduces store clutter without
losing useful recall. Scores from different live runs are noisy at eight cases;
report both precision and recall and inspect individual retrieval failures.
