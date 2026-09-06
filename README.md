# engram — long-term memory for conversational agents

A LangGraph agent that **remembers a user across sessions**, built on the premise
that the hard part of memory is not reading it back but deciding what to write.

> The public LLM APIs are **stateless**. Every call starts from zero. So anything
> built on them — an assistant, a copilot, a support bot — has to build its own
> memory layer, and that layer is an open engineering problem rather than a
> solved feature. This repository is that layer, built and benchmarked in the
> open, including where it fails.

**Status: slice 1 of 4.** The agent, both stores, the read path, and the
benchmark are built and measured. The memory manager that this slice exists to
justify — extraction, dedupe, conflict resolution, decay — is slice 2. The
numbers below are the baseline it has to beat, not a finished result.

---

## Why this is not RAG

RAG only ever **reads**. A memory system also has to write, and every write is a
decision:

| Decision | Question it answers |
|---|---|
| **Extract** | Is anything in this turn worth remembering at all? |
| **Dedupe** | Do I already know this? |
| **Resolve conflicts** | The user moved from Mumbai to Berlin. Update, supersede, or keep both? |
| **Decay** | "I'm debugging a flaky test today" should expire. "I prefer pytest" should not. |
| **Gate** | A document I *read* said "the user is an admin". That is not the user talking. |

Those write-side decisions are where memory systems succeed or fail, and RAG has
none of them.

---

## Architecture

```
  user message (session N)                      ┌──────────────────────────────┐
        │                                       │  THREAD MEMORY               │
        │  ◄────────────────────────────────────┤  LangGraph checkpointer      │
        │                                       │  this conversation, durable  │
        ▼                                       └──────────────────────────────┘
  ① RECALL ────────────► semantic search, scoped to this user,
        │                top-k only — never the whole store
        ▼
  ② RESPOND ───────────► answer grounded in the recalled memories + the turn
        │
        ▼
  ③ REMEMBER ──────────► the write path
        │                  ├─ extract candidate facts        ┐
        │                  ├─ dedupe against what is known   │ slice 2
        │                  ├─ resolve contradictions         │
        │                  ├─ score importance, set decay    ┘
        │                  └─ injection gate: only trusted     ─ slice 3
        │                     (user-authored) content may write
        ▼
  ┌──────────────────────────────────────────────────────────┐
  │  LONG-TERM MEMORY — LangGraph Store over pgvector         │
  │  namespace ("memories", user_id)                          │
  │  semantic · episodic · procedural                         │
  └──────────────────────────────────────────────────────────┘
```

The two memories are deliberately different things, and LangGraph already draws
the distinction, so the project uses its primitives rather than inventing
parallel ones:

* the **checkpointer** is *thread* memory — a transcript, durable across
  restarts, scoped to one conversation. Nothing is judged or extracted.
* the **store** is *long-term* memory — facts about a user, retrieved by meaning
  rather than recency, shared across every session that user ever opens.

`thread_id` and `user_id` are independent. That separation is the entire trick:
it is what makes a fact stated on Monday available on Thursday.

### Memory taxonomy

| Kind | What it holds | Example |
|---|---|---|
| **Semantic** | durable facts and preferences | "prefers pytest", "team = payments" |
| **Episodic** | past events and interactions | "last week we debugged the auth bug" |
| **Procedural** | how the user wants things done | "always show the SQL before running it" |

---

## Benchmark

26 hand-authored cases in `eval/cases/slice1.jsonl`. Each gives a user ~30
sessions' worth of ordinary chatter with a fact buried in it, then asks a probe
**in a brand-new thread** — so the only path from the fact to the answer is
long-term memory. Both arms are compiled from the same graph with `memory`
flipped, so the comparison isolates one variable.

```
make eval            # deterministic config — this is the CI gate
make eval-semantic   # real sentence embeddings
```

| Metric | Stateless | Memory (naive writer) | |
|---|---|---|---|
| Passive recall (n=9) | 0.0% | **100.0%** | probe names the fact |
| **Decision-relevant recall (n=10)** | 0.0% | **70.0%** | ← the headline |
| Conflict resolution (n=7) | 0.0% | **28.6%** | a later session contradicts an earlier one |
| Memory recall — kept the facts | 0.0% | 100.0% | |
| Memory precision — kept *only* facts | 0.0% | **3.2%** | |
| Avg memories injected / turn | 0.0 | 5.0 | out of ~31 stored |

<sub>`--embedder fastembed`. With the deterministic `hashing` embedder used in
CI: passive 66.7%, decision-relevant 50.0%, conflict 14.3%.</sub>

**Read the shape, not the headline.** Passive recall saturates at 100% while
decision-relevant recall drops to 70% — that is the same gap published memory
benchmarks report, reproduced here on a system whose internals are inspectable.
And two numbers are outright bad on purpose:

* **Conflict resolution, 28.6%.** The naive writer stores "I'm on the payments
  team" *and* "I switched to the platform team", recalls both, and answers with
  both. On a contradiction the agent is now **less** reliable than having no
  memory at all, because it surfaces the stale fact with total confidence.
* **Memory precision, 3.2%.** Storing every turn verbatim means ~31 records per
  user of which one carries a fact. The rest is "the coffee machine is broken
  again", competing for the same five top-k slots.

Both are what slice 2 fixes, and both now have a number attached rather than an
argument. That was the point of building the benchmark before the memory
manager: the improvement gets to be measured instead of asserted.

---

## Quickstart

Runs fully offline — no AWS account, no API key, no network.

```bash
make install    # venv + deps (uses uv when present)
make demo       # cross-session recall, in-process
make test       # 56 tests, no network
make eval       # the benchmark table above
```

Against real Postgres + pgvector:

```bash
make up                                            # docker compose: pgvector
ENGRAM_STORE_BACKEND=postgres make demo
ENGRAM_TEST_DSN=postgresql://engram:engram@localhost:5432/engram \
  .venv/bin/pytest -m postgres                     # durability tests
make down                                          # tears down the volume too
```

Talk to it yourself:

```bash
python scripts/chat.py --user alice --thread monday
python scripts/chat.py --user alice --thread thursday   # same user, new session
```

`/memories` lists what is stored, `/history` shows the thread, `/quit` exits.

---

## Configuration

All via environment (`ENGRAM_` prefix, `pydantic-settings`). See `.env.example`.
Every default is offline.

| Setting | Default | Notes |
|---|---|---|
| `ENGRAM_CHAT_PROVIDER` | `stub` | `stub` \| `bedrock` |
| `ENGRAM_EMBEDDER` | `hashing` | `hashing` \| `fastembed` \| `bedrock` |
| `ENGRAM_STORE_BACKEND` | `memory` | `memory` \| `postgres` |
| `ENGRAM_RECALL_TOP_K` | `5` | memories injected per turn |
| `ENGRAM_BEDROCK_MODEL_ID` | `us.anthropic.claude-haiku-4-5-…` | cheap tier: extraction, dedupe triage |
| `ENGRAM_BEDROCK_REASONING_MODEL_ID` | same | point at a larger model for conflict resolution |

### Why the defaults are what they are

**The stub chat model is an instrument, not a mock.** It answers by reporting
exactly the memories placed in its context — nothing more. So an assertion like
"the answer mentions pytest" passes if and only if retrieval surfaced that fact,
which makes the benchmark a measurement of the memory subsystem with model
quality held constant. That is what a regression gate needs. Set
`ENGRAM_CHAT_PROVIDER=bedrock` to measure memory and model together.

It is also honest about failure: when contradicting facts have both been stored,
both appear in the answer. The conflict score above is not simulated.

**Three embedders, because dedupe quality is the whole game.** `hashing` is
deterministic across processes (BLAKE2b, not Python's salted `hash`), so CI
numbers move only when behaviour moves. It is lexical, though — it will never
connect "Berlin" to "Germany's capital", and the gap between the two columns
above is exactly that limitation. `fastembed` gives real sentence semantics
offline via ONNX, no torch, ~130MB downloaded once.

---

## What slice 1 establishes

- [x] Thread memory + long-term memory via LangGraph, durable across process restarts (verified against pgvector, not just in-process)
- [x] Per-user memory isolation
- [x] Top-k recall — memory that outlives the context window
- [x] Memory-op tracing: what was recalled and written, per turn
- [x] A benchmark with a stateless control arm, and a CI regression gate
- [x] Provenance and decay fields on every record from the first write
- [x] Offline by default; `make up` for the real store; typed, `mypy --strict` clean

Provenance and expiry are recorded but **not yet enforced** — nothing consults
them at write time. They exist now because retrofitting provenance onto a store
already full of unattributed facts is a migration nobody wants to run.

### Not built yet

| Slice | Work |
|---|---|
| 2 | The memory manager: extraction → dedupe → **conflict resolution** → importance + decay |
| 3 | Injection gate (untrusted content may not write user memory), "forget me" hard delete |
| 4 | mem0 / LangMem as a managed baseline on the same benchmark |
| later | FastAPI + SSE, real LoCoMo loader, OpenTelemetry, teardownable CDK stack |

---

## Layout

```
src/engram/
  config.py       pydantic-settings, ENGRAM_ prefix
  schemas.py      the Memory record: kind, provenance, decay, supersession
  embeddings.py   hashing | fastembed | bedrock, behind LangChain's Embeddings
  models.py       the stub chat model, and Claude on Bedrock
  store.py        checkpointer + store, in-process or Postgres/pgvector
  prompts.py      the recalled-memory block
  graph.py        recall → respond → remember, and the Agent wrapper
  memory/
    read.py       semantic recall, filtered to live memories
    write.py      the naive baseline + the MemoryWriter seam slice 2 fills
  eval/           cases, metrics, runner
eval/cases/       the committed benchmark
tests/            56 offline tests + Postgres durability tests
```

---

## Design notes

**The benchmark guards itself.** An `EvalCase` refuses to validate if its probe
thread is also one of its session threads — in that shape the checkpointer would
answer the probe from the running transcript, long-term memory would never be
consulted, and *every* arm including the stateless control would score 100%.
That failure is silent and flattering, so it is a validator rather than a
comment.

**Cases are sized so ranking matters.** An earlier draft gave each user ~8
memories against a top-k of 5, which made retrieval nearly "inject everything"
and produced a meaningless 100% on decision-relevant recall. The committed cases
give each user ~31.

**Facts are not worded to match their probes.** An earlier draft phrased a fact
as "no meat at any *restaurant*" for a probe about picking a *restaurant* — which
tunes the benchmark to flatter a lexical retriever. The facts are now worded as a
user would state them, and the deterministic embedder scores worse for it.

**Recall over-fetches before filtering.** Expired and superseded memories are
removed after the store returns its matches, so a user whose closest matches are
all stale still gets live results instead of an empty recall.

**The write node runs after the response.** A fact learned from this turn should
not be recalled into the answer to that same turn.
