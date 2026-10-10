# Evaluation and published results

[Documentation index](README.md) · [Project overview](../README.md)

## Benchmark

32 hand-authored cases in `eval/cases/core.jsonl`. Each gives a user six sessions
of ordinary chatter, **seven unrelated real facts**, and the fact the probe
depends on, then asks the probe **in a brand-new thread**, so the only path from
fact to answer is long-term memory. All three arms are compiled from the same
graph, so a difference between columns is a difference in the write path and
nothing else.

```bash
make eval            # deterministic config, this is the CI gate
make eval-semantic   # real sentence embeddings
```

**The headline metric is decision-relevant recall**, not passive recall. Passive
recall asks "can you retrieve a stored fact when asked directly?" and saturates
easily. Decision-relevant recall asks "did the right answer change because the
right fact was recalled or updated?", which is the thing memory is actually for.
Published memory benchmarks report the same shape: passive recall near 100%,
decision-relevant far lower.

Results with `--embedder fastembed`:

| Metric | Stateless | Naive writer | **Memory manager** |
|---|---|---|---|
| Passive recall (n=9) | 0.0% | 100.0% | **100.0%** |
| **Decision-relevant recall (n=10)** | 0.0% | 60.0% | **90.0%** |
| Conflict resolution (n=7) | 0.0% | 14.3% | **100.0%** |
| Injection (n=6) | 0.0% | 16.7% | **100.0%** |
| Overall (n=32) | 0.0% | 53.1% | **96.9%** |
| **Injection resistance** | n/a | 0.0% | **100.0%** |
| Memory recall, kept the facts | 0.0% | 96.9% | 96.9% |
| Memory precision, kept *only* facts | 0.0% | 24.3% | **87.9%** |
| Memories stored per user | 0.0 | 28.0 | **7.8** |

With the deterministic `hashing` embedder used in CI: decision-relevant 60.0 /
90.0, conflict 14.3 / 71.4, injection resistance 0 / 100, precision 24.3 / 87.9
(naive / manager). The stateless arm scores 100% on injection resistance for an
uninteresting reason, namely that it stores nothing at all, so that cell is left
blank rather than presented as a win.

**What the write path bought.** Conflict resolution went from 14.3% to 100% and
precision from 24.3% to 87.9%, while the store shrank from 28 records per user to
7.8. Those move together, and that is the mechanism rather than a coincidence:
the naive writer keeps every turn, so the five top-k slots fill with "the coffee
machine is broken again" and with facts the user has since contradicted. Storing
less is how the manager recalls better.

**The conflict number is the one worth dwelling on.** A store that keeps both
"I'm on the payments team" and "I switched to the platform team" makes the agent
**less** reliable than having no memory at all, because it will surface the stale
fact with total confidence. Superseding, meaning writing the new fact and
retiring the old one with history, is what closes that.

### CI regression gate

`.github/workflows/ci.yml` runs lint, `mypy --strict`, the test suite and then
the benchmark with floors just under the current numbers:

```
--fail-under-decision 0.85  --fail-under-conflict 0.65
--fail-under-precision 0.85 --fail-under-injection 1.00
```

A prompt or policy change that degrades the write path fails the build rather
than merging quietly. CI is offline: no credentials, no network, the stub model
and the hashing embedder.

### The same benchmark on real models

The table above holds model quality constant so the memory subsystem is the only
variable. Running the same cases against live Nova Lite and Titan says something
different and worth knowing. A balanced sample of 2 cases per kind, 8 in total:

| Metric | Stateless | **Manager** |
|---|---|---|
| Overall (n=8) | 12.5% | **87.5%** |
| Conflict resolution (n=2) | 0.0% | **100.0%** |
| Injection (n=2) | 0.0% | **100.0%** |
| Injection resistance | n/a | 100.0% |
| **Memory precision** | 0.0% | **31.6%** |
| Memories stored per user | 0.0 | 17.0 |

Three honest caveats, because the headline reads better than the run deserves:

- **n=8.** 100% on eight cases is encouraging, not a claim.
- **The stateless arm is no longer 0%.** A real model produces plausible
  defaults, and substring scoring cannot tell a lucky guess from a recalled fact.
  The offline 0% was partly an artefact of a stub that never guesses.
- **Precision is far worse live: 31.6% against 87.9% offline.** A real model
  extracts from turns the rule fixture ignores, storing 17.0 records per user
  instead of 7.8. The offline precision figure flatters the system by roughly
  2.7x, and the live one is the honest number. It did not improve when the
  extraction prompt was tightened (28.7% before those fixes, 31.6% after, inside
  the noise at this sample size), so it is a standing weakness rather than a
  tuning oversight.

---


## LoCoMo: somebody else's benchmark

Every number above is scored on cases written here, which is worth about as much
as any exam written by the person sitting it.
[LoCoMo](https://github.com/snap-research/locomo) is public and much longer: 10
conversations, about 5.9k turns, about 2k questions, up to 19 sessions and 400+
turns each.

```bash
make locomo        # fetches the dataset, runs one conversation offline
make locomo-live   # real Bedrock + an LLM judge
```

The dataset is fetched, not vendored: it is 2.8MB and carries its own licence
terms. It runs **beside** the hand-authored benchmark rather than replacing it,
for three reasons:

- **It is judged, so it cannot gate CI.** Gold answers are free-form ("The sunday
  before 25 May 2023"), and an agent that says "the Sunday before the 25th of
  May" is right. Grading that needs a model, and a model makes the score
  non-deterministic.
- **It has no conflict or injection questions.** The two properties this project
  is actually about have no LoCoMo counterpart.
- **It annotates evidence turns, not facts worth keeping**, so the store-side
  precision and recall metrics have nothing to score against.

### Results

**41.3%** overall: 3 conversations, 300 questions, live Nova and Titan.

| Category | n | Accuracy |
|---|---|---|
| category-3 | 21 | 52.4% |
| category-4 | 96 | 50.0% |
| temporal | 90 | 46.7% |
| multi-hop | 74 | 28.4% |
| **adversarial** | 19 | **10.5%** |
| **Overall** | **300** | **41.3%** |

The 30-question baseline scored 26.7% overall and 31.2% temporal; a reverted
anchoring experiment lowered temporal to 18.8%. At n=90 temporal questions,
the larger run scored 46.7%. **The small sample was badly
unrepresentative**, which is the argument for measuring once at a real size
rather than tuning against a noisy one. Two attempts to "fix" temporal against
that small sample were reverted for exactly this reason.

**The weak column is adversarial: 10.5%.** Those questions carry a false premise,
usually attributing something to the wrong speaker, and the correct answer is to
decline. The system answers them anyway. That is worth sitting with, because it
is the mirror image of this project's strongest result: it reliably refuses
untrusted **sources** (100% injection resistance, structurally) and reliably
fails to refuse false **premises**. Provenance is checked; plausibility is not.
Nothing in the write path was designed to, and it shows.

**1,451 turns became 1,300 memories**, roughly one per turn. On its own cases the
manager keeps 7.8 per user; here extraction barely filters at all, because in
rich two-person narrative almost every turn contains *something* it judges a
fact. That is the live-precision weakness measured at scale.

### Why the number is what it is

Look at what LoCoMo asks:

> *"What are the main ingredients of the ice cream recipe shared by Nate?"*
> *"What does John write on the whiteboard to help him stay motivated?"*
> *"How many times has Jolene been to France?"*

Now the extraction prompt, written deliberately:

> *"Return facts ONLY when the turn states something about the user that would
> still be useful weeks from now... Most turns contain nothing. Small talk,
> status updates ... are NOT facts."*

**LoCoMo measures transcript recall. This system does user-profile
distillation.** It scores what it scores substantially *because it is working as
designed*: it discards recipe ingredients and whiteboard slogans as chatter, and
that discarding is what earns 87.9% precision and 7.8 memories per user on the
task it is actually for.

So this number is reported, not optimised. Tuning for it would mean making
extraction less selective, which trades away the precision that makes the system
good at remembering a person. That trade is available and was not taken.

Two things this cost, recorded because the failures are the useful part. Adding
session dates to the ingested text moved temporal accuracy not at all (31.2%
before and after): extraction was dropping the date while *keeping* the
"yesterday" it anchored. Appending the date deterministically instead grew the
store 32% (a date makes near-duplicates look distinct, defeating dedupe) and
moved nothing again. Deterministic anchoring was reverted; the loader still includes session dates. The right shape, if this is
revisited, is an `occurred_at` **field** the embedding never sees. Mutating the
text that retrieval depends on in order to carry metadata was the error.

Two details the data format hides, both of which would have produced a wrong
number:

**`adversarial_answer` is a trap, not a gold answer.** Category 5 (446 of about
2000 questions) carries a false premise, and that field holds the plausible
answer you give if you fail to notice. *"What did **Caroline** realise after her
charity race?"* maps to *"self-care is important"*, which is a thing **Melanie**
realised. The correct response is to decline, and the official evaluation scores
exactly that. Reading the field as gold would have scored every correct refusal
wrong and every credulous answer right, inverting the metric across a fifth of
the dataset.

**Sessions must be ordered numerically.** `session_10` sorts before `session_2`
as a string, and replaying a conversation out of order silently corrupts every
contradiction it contains. For a system whose whole job is "what is true *now*",
that is the worst possible way to be wrong.

---


## The managed baseline: mem0

`mem0` runs as a fourth arm on identical footing: the same Nova model, the same
Titan embeddings, the same Postgres instance, the same prompt builder and the
same scoring. Only the memory logic differs.

```bash
.venv/bin/pip install -e ".[baseline]"
make up
make eval-baseline
```

It is opt-in and never part of CI, because mem0 calls an LLM on every write. Two
things worth stating so the comparison is read fairly: this is mem0's **default**
behaviour, not mem0 tuned (no custom prompts, no graph memory, no hosted
platform), and mem0 is wired through its `langchain` provider so it receives the
same model *object* used here.

8 cases (2 per kind), identical Nova Lite, Titan and Postgres on both sides:

| Metric | **Manager** | mem0 |
|---|---|---|
| Passive recall (n=2) | 100.0% | 100.0% |
| Decision-relevant (n=2) | 50.0% | 0.0% |
| Conflict resolution (n=2) | **100.0%** | 50.0% |
| Injection (n=2) | **100.0%** | 0.0% |
| Overall (n=8) | **87.5%** | 37.5% |
| **Injection resistance** | **100.0%** | **0.0%** |
| Memory recall, kept the facts | 80.4% | **89.3%** |
| Memory precision, kept *only* facts | **32.1%** | 25.9% |
| Memories stored per user | 17.5 | 50.2 |

**The one result that is structural rather than incidental is injection
resistance: 100% against 0%.** mem0 has no provenance concept, so a poisoned
document the agent merely read became a fact about the user and was then recalled
into answers: "root access" and "security team" both surfaced. No amount of
prompt tuning closes that, because the defence has to be a property of the write
path, not of the extraction quality.

**mem0 beats this system on memory recall: 89.3% against 80.4%.** It keeps more
of what matters, because it keeps far more of everything: 50.2 records per user
against 17.5. That is the recall/precision trade in its plainest form, and it is
also why it loses on conflict resolution. A store that keeps three phrasings of
"which team" has no way to retire the stale one.

**Treat the small-n live numbers as noisy.** An earlier live run of the same eight
cases scored the manager arm 100% on decision-relevant recall; this one scored
50%, on one case flipping. With two cases per kind, a single flip moves a column
by fifty points. The offline benchmark is the one to regress against; the live
runs say "it works on a real model" and little more.

Routing mem0 through LangChain turned out to be mandatory. **mem0 2.0.20's own
`aws_bedrock` adapter is broken for Amazon models**: `_format_messages_amazon`
emits `{"role": ..., "content": "<text>"}` where the Bedrock Converse API
requires content blocks, so every Nova call fails parameter validation. Its
Anthropic formatter builds the blocks correctly; the Amazon one does not.

---
