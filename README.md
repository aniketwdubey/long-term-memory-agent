# engram: long-term memory for conversational agents

A LangGraph agent that remembers users across conversations and decides what is
worth keeping. Its write path extracts facts, deduplicates repeats, supersedes
contradictions, expires temporary facts, and quarantines untrusted sources.

**Status:** the core agent, storage backends, API, tracing, evaluation harness,
and AWS infrastructure are implemented. User-scoped conversations and Cognito
API authorization were deployed and verified with two accounts on 2026-10-10.
Live extraction precision remains the largest quality weakness. See
[project status and remaining work](docs/status.md).

## Quickstart

Requires Python 3.11 or newer. Installation downloads dependencies; the default
demo, tests, and benchmark then run offline without AWS credentials.

```bash
make install
make demo          # eight acts: recall, update, injection checks, deletion
make test
make eval          # deterministic three-arm comparison
make serve-demo    # local API without auth, bound to 127.0.0.1:8000
```

Open [interactive API docs](http://localhost:8000/docs) after starting the API.
`make serve` requires Cognito configuration; `make serve-demo` explicitly disables
authentication for local use. See [authentication](docs/authentication.md).

The default backend is in-memory. For persistence, run `make up` and set
`ENGRAM_STORE_BACKEND=postgres`. See the [four run modes](docs/operations.md#how-to-run-it).

## What it does

| Capability | Behavior |
|---|---|
| Cross-session recall | A preference learned on Monday is available in Thursday's new conversation. |
| Selective memory | Extracts durable facts and preferences; live models still store too much chatter. |
| Deduplication | Repeated facts reinforce an existing memory. |
| Conflict resolution | New slot values supersede old ones, preserving retired history. |
| Expiry | Temporary facts receive a TTL and stop being recalled after it expires. |
| Injection checks | Tool and document content is quarantined before extraction. |
| Inspection and deletion | Users can inspect and remove their facts, conversations, and quarantine. |
| User isolation | Conversations are scoped by user and thread; authenticated API callers can access only their own data. |

## Architecture

![Recall, respond, remember](docs/architecture.svg)

Two stores serve different purposes: conversation checkpoints hold raw messages,
while long-term memory holds distilled facts. The graph runs **recall → respond
→ remember**. Only extraction uses a model within the write path; deduplication,
conflict handling, expiry, and provenance checks are policy code.

The same application supports in-memory storage or PostgreSQL with pgvector,
and a deterministic offline model or Amazon Bedrock. AWS deployment uses ECS
Express Mode, RDS PostgreSQL, and a Cognito user pool.

Read [architecture and design](docs/architecture.md) for the graph, memory types,
write policy, injection boundary, and deletion behavior.

## Results and limits

| Measurement | Result | Scope |
|---|---|---|
| Offline overall accuracy | 90.6% | 32 cases, stub + hashing; reproduced in the local audit |
| Offline memory precision | 87.9% | Deterministic extractor fixture |
| Semantic-embedding overall accuracy | 96.9% | Previously recorded, same 32 cases |
| Live memory precision | 31.6% | Previously recorded Nova/Titan sample, 8 cases |
| LoCoMo overall accuracy | 41.3% | Previously recorded live run, 300 questions |
| LoCoMo adversarial accuracy | 10.5% | False-premise questions remain a weakness |

The offline score does not predict live extraction quality. Provenance checks
also do not establish whether a user's claim is true. See [evaluation methods,
results, and caveats](docs/evaluation.md) and [remaining work](docs/status.md).

## Documentation

| Start here | Contents |
|---|---|
| [Documentation index](docs/README.md) | Reading order and where public/private documents live |
| [Project status](docs/status.md) | Completed work, priorities, verification limits |
| [Operations](docs/operations.md) | Installation, run modes, configuration, deployment, data upgrades |
| [Authentication](docs/authentication.md) | Cognito accounts, tokens, ownership checks, client integration |
| [Architecture](docs/architecture.md) | Memory design, write/read/delete paths, design decisions |
| [Evaluation](docs/evaluation.md) | Offline, live, LoCoMo, and mem0 benchmarks |
| [.env.example](.env.example) | Environment configuration |

`docs/PROJECT_CONTEXT.md` and `docs/blog.md` are private local files, deliberately
git-ignored. They are not available in a fresh clone. Public status and operating
instructions live in the tracked guides above.

## Repository layout

```text
src/engram/
  graph.py, threads.py        agent flow and user-scoped conversation keys
  memory/                    extraction, read/write policy, gate, deletion
  api/                       FastAPI routes and Cognito authorization
  eval/                      runner, metrics, LoCoMo, judge, mem0 comparison
  store.py, schemas.py        storage backends and memory records
  config.py, embeddings.py    settings and embedding providers
  models.py, prompts.py       chat providers and prompt construction
  logging.py, tracing.py      observability
tests/                       offline suite and optional PostgreSQL tests
eval/cases/core.jsonl         32 committed benchmark cases
infra/                       AWS CDK stack
scripts/                     demo, local chat, Bedrock checks, dataset fetch
docs/                        public guides and architecture diagram
```

Deployment has standing AWS costs. The [deployment guide](docs/operations.md#deploying-it)
includes teardown commands. Read [existing-data upgrade instructions](docs/operations.md#upgrading-existing-data)
before pointing another deployment at an older database. This deployment reset its
disposable demo data after taking a recovery snapshot.
