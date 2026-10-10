# Running and deploying Engram

[Documentation index](README.md) · [Project overview](../README.md)

## How to run it

There are **two independent choices**, and confusing them is the main source of
"which mode am I in?" trouble. Pick one from each column.

| Choice | Options | What it controls |
|---|---|---|
| **Where it runs** | in-process / local + Docker Postgres / all Docker / AWS | Durability and packaging |
| **Which brain it uses** | stub + hashing / real Bedrock | Whether an LLM is actually called |

So "in-process with a real model" and "Docker with the stub" are both valid and
both useful. The four run modes below are the *where*; the
[model matrix](#choosing-the-brain-stub-or-bedrock) is the *which brain*.

### Mode 1: everything in-process (the default)

Nothing external at all. Checkpoints and memories live in Python dictionaries,
the chat model is a deterministic stub, and embeddings are a local hash. This is
what CI runs and what a first-time reader should start with.

```bash
make install
make demo        # or: make test, make eval
make serve-demo       # HTTP API on :8000, interactive docs at /docs
.venv/bin/python scripts/chat.py --user alice --thread monday
```

Everything disappears when the process exits. That is fine for the demo (which
replays a week of conversation in one process) and wrong for anything else.

### Mode 2: local Python, Postgres in Docker

The application still runs on your laptop under `uvicorn` or `python`, but both
memories now live in a real database: LangGraph's `PostgresSaver` for
transcripts and its pgvector-backed `PostgresStore` for facts. Memory survives a
restart, which is the point of the system.

```bash
make up                                      # docker compose: pgvector/pgvector:pg17
ENGRAM_STORE_BACKEND=postgres make demo
ENGRAM_STORE_BACKEND=postgres make serve-demo

# the durability tests, which are skipped unless a DSN is present
ENGRAM_TEST_DSN=postgresql://engram:engram@localhost:5432/engram \
  .venv/bin/pytest -m postgres

make down                                    # stops the container and drops its volume
```

This is the mode to develop in. You get real persistence and a fast edit loop,
because the code is not inside an image.

### Mode 3: everything in Docker

Application and database both in containers, wired together by compose. This is
the closest local approximation of the deployed system, and it is what proves
the image is correct before a deploy.

```bash
docker compose build                         # after any code change

docker compose --profile api up              # API on :8000, against Postgres
docker compose --profile demo run --rm demo  # one-shot: the demo, against Postgres
```

The image is built from the repository, so it does **not** pick up local edits
until you rebuild. If the demo prints something that no longer matches the
source, that is why.

### Mode 4: deployed on AWS

The container image compose already builds, run by ECS Express Mode, against RDS
PostgreSQL with pgvector. `PostgresStore` and `PostgresSaver` are exercised by the optional PostgreSQL
durability tests. Default offline CI uses the in-memory backend. See [Deploying it](#deploying-it).

```bash
make -C infra install     # once
make cdk-synth               # render the template, creates nothing
make cdk-deploy              # ~11 minutes, RDS is the long pole
make -C infra outputs        # the service URL and other stack outputs
make cdk-destroy             # leaves nothing behind
```

This stack has standing cost (NAT gateway, RDS, a running ECS task). Destroy it
when you are not demoing.

### Mode summary

| | Mode 1: in-process | Mode 2: local + Docker DB | Mode 3: all Docker | Mode 4: AWS |
|---|---|---|---|---|
| App runs on | your Python | your Python | container | ECS Express Mode |
| Checkpointer | `InMemorySaver` | `PostgresSaver` | `PostgresSaver` | `PostgresSaver` |
| Store | `InMemoryStore` | `PostgresStore` (pgvector) | `PostgresStore` (pgvector) | `PostgresStore` on RDS |
| Survives restart | no | yes | yes | yes |
| Default model | stub | stub | stub | Bedrock (Nova) |
| Needs Docker | no | yes | yes | yes, to build the image |
| Needs AWS | no | no | no | yes |
| Cost | zero | zero | zero | standing |
| Start with | `make demo` | `make up` | `docker compose --profile api up` | `make cdk-deploy` |

### Choosing the brain: stub or Bedrock

Orthogonal to all four modes. Two environment variables switch it.

| | Offline default | Real models |
|---|---|---|
| `ENGRAM_CHAT_PROVIDER` | `stub` | `bedrock` (`amazon.nova-lite-v1:0`) |
| `ENGRAM_EMBEDDER` | `hashing` | `bedrock` (Titan v2), or `fastembed` for offline semantics |
| Credentials | none | `aws configure` |
| Cost | zero | a few cents per run |
| Determinism | exact | no |

```bash
.venv/bin/python scripts/check_bedrock.py                 # preflight: embeddings + a model bake-off
ENGRAM_CHAT_PROVIDER=bedrock ENGRAM_EMBEDDER=bedrock make demo
ENGRAM_CHAT_PROVIDER=bedrock ENGRAM_EMBEDDER=bedrock \
  ENGRAM_STORE_BACKEND=postgres make serve-demo      # Mode 2 with a real brain
```

**The stub is an instrument, not a mock.** It answers by reporting exactly the
memories placed in its context and nothing else. So an assertion like "the
answer mentions pytest" passes if and only if retrieval surfaced that fact,
which makes the benchmark a measurement of the memory subsystem with model
quality held constant. That is what a regression gate needs. It is also honest
about failure: when contradicting facts have both been stored, both appear in
the answer, so the conflict score is not simulated.

`scripts/check_bedrock.py` is worth running before any live work. It verifies
embeddings, runs the extraction prompt against Nova Lite and Nova Micro side by
side, and reports when a policy guard had to correct the model.

### The HTTP API

The examples below use local demo mode. Deployed calls require a Cognito access
token and `user_id` equal to its `sub`; see [authentication](authentication.md).

```bash
make serve-demo                                   # or docker compose --profile api up
open http://localhost:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat` | one turn: returns the reply **and** the memory trace |
| `POST /v1/observe` | feed it content it *read*; provenance is explicit at the call site |
| `GET /v1/memories/{user}` | everything stored, retired records included |
| `GET /v1/memories/{user}/history/{thread}` | one conversation's transcript, from the checkpointer |
| `GET /v1/quarantine/{user}` | what the gate refused |
| `DELETE /v1/memories/{user}` | forget me: memories, quarantine, transcripts |
| `DELETE /v1/memories/{user}/threads/{thread}` | forget one conversation |
| `DELETE /v1/memories/{user}/{memory_id}` | forget one fact |

Cross-session recall over HTTP, which is the whole system in two calls:

```bash
curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"user_id":"alice","thread_id":"monday","message":"I am on the payments team and I prefer pytest."}'

curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"user_id":"alice","thread_id":"thursday","message":"Scaffold me a test for the refund endpoint."}'
```

The inspection and deletion endpoints exist because a memory system whose
contents cannot be seen or removed is one users have to take on trust, and
"forget me" should not live only in a Python function.

### Tracing

```bash
make trace-demo    # spans to the console
```

Off by default and genuinely off: with no exporter configured the OpenTelemetry
API returns a no-op tracer, so nothing leaves the process. See
[Design notes](architecture.md#design-notes) for why OTel rather than LangSmith.

### Every make target

| Target | What it does |
|---|---|
| `make install` | create the venv, install with dev/embed/otel extras |
| `make lint` | ruff + `mypy --strict` |
| `make test` | offline test suite, no network or credentials |
| `make demo` | the 8-act cross-session demo, in-process |
| `make serve-demo` | local API without auth on 127.0.0.1:8000 |
| `make serve` | API requiring Cognito configuration by default |
| `make eval` | the three-arm benchmark, deterministic config (the CI gate) |
| `make eval-semantic` | the same benchmark with real sentence embeddings |
| `make eval-live` | a balanced live sample against Bedrock |
| `make eval-baseline` | manager vs mem0, identical models |
| `make locomo` | fetch LoCoMo, run one conversation offline |
| `make locomo-live` | LoCoMo against Bedrock with an LLM judge |
| `make check-bedrock` | preflight the live path, compare Nova models |
| `make trace-demo` | print OpenTelemetry spans |
| `make up` / `make down` / `make logs` | the Postgres container |
| `make docker-demo` | the demo, in Docker, against Postgres |
| `make cdk-synth` / `cdk-deploy` / `cdk-destroy` | the AWS stack |
| `make -C infra test` | offline assertions on the CDK template |

---


## Configuration

All settings come from the environment with an `ENGRAM_` prefix, read and
validated by `pydantic-settings`. See `.env.example`. The CLI model and storage defaults are
offline. HTTP authentication defaults to Cognito; `make serve-demo` explicitly
disables authentication for local use.

| Setting | Default | Notes |
|---|---|---|
| `ENGRAM_AUTH_MODE` | `cognito` | `disabled` only for local demos |
| `ENGRAM_COGNITO_USER_POOL_ID` | *(empty)* | required for authenticated API startup |
| `ENGRAM_COGNITO_CLIENT_ID` | *(empty)* | expected access-token app client |
| `ENGRAM_CHAT_PROVIDER` | `stub` | `stub` \| `bedrock` |
| `ENGRAM_EMBEDDER` | `hashing` | `hashing` \| `fastembed` \| `bedrock` |
| `ENGRAM_STORE_BACKEND` | `memory` | `memory` \| `postgres` |
| `ENGRAM_POSTGRES_DSN` | `postgresql://engram:engram@localhost:5432/engram` | used when no host is set |
| `ENGRAM_POSTGRES_HOST` | *(empty)* | set it and the DSN is assembled from parts instead |
| `ENGRAM_MEMORY_WRITER` | `manager` | `manager` \| `naive` (the benchmark's control arm) |
| `ENGRAM_RECALL_TOP_K` | `5` | memories injected per turn |
| `ENGRAM_DEDUPE_SIMILARITY` | `0.9` | cosine above which two unslotted facts are one fact |
| `ENGRAM_SCAN_TRUSTED_CONTENT` | `true` | the gate's second, heuristic layer only |
| `ENGRAM_OTEL_EXPORTER` | `none` | `none` \| `console` \| `otlp` |
| `ENGRAM_BEDROCK_MODEL_ID` | `amazon.nova-lite-v1:0` | extraction and responses |
| `ENGRAM_BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | embeddings |
| `ENGRAM_AWS_REGION` | `us-east-1` | |

### Why the defaults are what they are

**Amazon's own models on Bedrock.** Nova Lite for extraction and responses, Titan
v2 for embeddings. Amazon-published models are covered by AWS credits, while
marketplace models bill through AWS Marketplace and are not. Nova Micro is
cheaper still and was measurably worse at choosing slot keys (it filed `pytest`
under `editor`).

**Three embedders, because dedupe quality is the whole game.** `hashing` is
deterministic across processes (BLAKE2b, not Python's salted `hash`, which is
seeded per process and would make stored and query vectors disagree), so CI
numbers move only when behaviour moves. It is lexical, though: it will never
connect "Berlin" to "Germany's capital". `fastembed` gives real sentence
semantics offline via ONNX with no torch, about 130MB downloaded once.
`bedrock` uses Titan v2.

**The password never sits in a connection string.** Deployed, RDS generates it
into Secrets Manager and ECS injects it as `ENGRAM_POSTGRES_PASSWORD`; the DSN is
assembled in `config.py` so the secret never passes through anything that might
log a URL.

---


## Deploying it

For an existing database, read [upgrading existing data](#upgrading-existing-data)
first. Cognito configuration is created by the stack; account sign-in details
are in [authentication](authentication.md). Review `make -C infra diff` before
deployment.

The point of the deployment is that **nothing about the application changes**.
The container is the image `docker compose` already builds, and RDS runs the same
PostgreSQL with the same pgvector extension as the local container, so
`PostgresStore` and `PostgresSaver` are the official classes exercised by the
optional PostgreSQL durability tests. Default offline CI does not run those tests.

```bash
make -C infra install          # once
make cdk-synth                    # render the template, creates nothing
make cdk-deploy                   # provisions everything, ~11 min, RDS is the long pole
make -C infra outputs             # service URL, DB endpoint, log group
make cdk-destroy                  # leaves nothing behind
```

The original version was verified on **2026-09-08**. The authentication and
isolation update was deployed and verified with two accounts on **2026-10-10**.
The original verification covered: cross-session
recall, a contradiction superseded, a poisoned document quarantined, and the
retired record still sitting in RDS, all on live Nova and real pgvector through
the public endpoint.

| Local | Deployed | Why |
|---|---|---|
| Docker Postgres | **RDS PostgreSQL + pgvector** | replaces *both* jobs, memory store and thread checkpointer, with zero new code |
| uvicorn on a laptop | **ECS Express Mode** | managed HTTPS endpoint and autoscaling from the image we already build |
| Bedrock | Bedrock | unchanged, the models were always AWS |
| stub + hashing defaults | Nova + Titan | CI stays offline, free and credential-free; the deployed service runs the real thing |

`make cdk-destroy` when you are done. **This stack has standing cost**: a NAT
gateway, an RDS instance and at least one running ECS task.

### Three decisions worth the words

**Why RDS and not a vector database.** Postgres does two jobs here: the memory
store *and* the checkpointer. S3 Vectors and OpenSearch replace only the first,
leaving transcripts homeless and requiring a hand-written `BaseStore` on top.
LangGraph ships exactly three store implementations (`base`, `memory`,
`postgres`) and neither of those is one of them. Once a relational store has to
exist anyway, pgvector comes with it for nothing, and the deployed storage layer
stays the one the optional database tests exercise.

**Why not Lambda.** This app holds a Postgres connection pool for its process
lifetime. Lambda would rebuild that pool on every cold start and multiply
connections against the instance limit under concurrency. A long-running
container is the right shape; that it also avoids cold starts is a bonus.

**Why not AgentCore Runtime, and why AgentCore Memory is the more interesting
question.** Bedrock AgentCore is AWS's managed hosting for agents. Its *Runtime*
is invocation-shaped: you hand it an agent and it serves invocations. `POST
/v1/chat` fits that shape, but `GET /v1/memories`, `GET /v1/quarantine` and
`DELETE /v1/memories/{user}` do not, and those are exactly the endpoints that
make the memory system inspectable and deletable. Hosting the agent somewhere
that cannot express them would trade away the property the project is about.
AgentCore *Memory* is the different and more interesting question: it is the
AWS-native managed memory service this project implicitly competes with, which
makes it the most relevant benchmark arm still unbuilt. mem0 answers the same
question for a third-party service; AgentCore Memory would answer it for AWS.

### Two things the first deploy taught, both now in the stack

ECS Express Mode exposes no `runtimePlatform`, so tasks are x86_64 and the image
is pinned to `LINUX_AMD64`; an arm64 image built on an Apple Silicon laptop fails
at task start with "exec format error". And the infrastructure role's managed
policy lives under `service-role/`: the first deploy died on a 404 for that one
ARN, and every other resource in the stack was a cascade cancellation from it.
Confirming the policy *name* was not the same as confirming its *ARN*.

---


## Upgrading existing data

The conversation-isolation fix changes the internal checkpoint key from a raw
`thread_id` to a versioned hash of `(user_id, thread_id)`. New thread-index
records include that checkpoint ID. Public thread names and long-term memory
records retain their original shape.

An indexed legacy conversation returns HTTP **409** on chat, history, or thread
deletion. A whole-user deletion checks all indexed conversations before changing
anything and also returns 409 if any are legacy. This prevents both exposing a
possibly shared transcript and reporting a successful delete while leaving it
behind. There is no fallback to an old global checkpoint key.

Before updating a database created by an older version:

1. Stop application writers and take a backup if the data must be kept.
2. Inventory the old thread index and checkpoints with trusted operator access.
   A raw thread ID can belong to several users, and older `memory=False` runs
   may have checkpoints without an index entry. Do not assign those records
   based only on the last `user_id` in graph state.
3. For disposable demo data, explicitly reset the old database before running
   this version. `make down` deletes the local Compose volume; never use it to
   preserve data. Cloud deletion/recreation also destroys accounts and data.
4. For retained data, prepare an operator-reviewed migration: verify each
   transcript's owner, re-key its complete checkpoint history and pending writes,
   and update the corresponding index with `checkpoint_id`. Mixed-user histories
   must be reviewed or discarded; automated reassignment would preserve the leak.
5. Map old display-name user IDs to verified Cognito `sub` identities separately.
   Renaming checkpoint keys alone does not migrate long-term memory namespaces.
6. Verify recall, history, per-thread deletion, and whole-user deletion using two
   distinct accounts before reopening the service.

No migration or database reset is executed automatically. This change has been
validated locally and has not been applied to the previously documented AWS
stack. An automated retained-data migration is not included because the legacy
format does not reliably establish ownership.
