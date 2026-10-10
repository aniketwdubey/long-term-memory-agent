# Project status and remaining work

[Documentation index](README.md) · [Project overview](../README.md)

## Current status — 2026-10-10

The core memory API is deployed. On 2026-10-10, the AWS stack was updated with
user-scoped transcripts and Cognito access-token verification. Two temporary
accounts passed a live end-to-end acceptance test and were deleted afterward.

| Area | Status |
|---|---|
| Recall, extraction, dedupe, supersession, TTL | Implemented; live extraction still needs improvement |
| In-memory and Postgres/pgvector backends | Implemented; PostgreSQL durability checks are opt-in |
| Injection gate and quarantine | Implemented; observed HTTP content cannot claim trusted provenance |
| User/thread isolation and deletion | Deployed; verified with two live accounts and fresh PostgreSQL connections |
| API authentication and ownership | Deployed; all eight memory routes rejected anonymous and cross-user requests |
| Cognito infrastructure | Pool/client deployed; SRP sign-in and refresh-token rotation verified live |
| Demo, tracing, offline/live eval, mem0, LoCoMo | Implemented; historical live results retained with their caveats |
| AWS deployment | UPDATE_COMPLETE on 2026-10-10; new service healthy and authenticated |
| Documentation | Overview plus operations, authentication, architecture, evaluation, and this status page |

## Live deployment — 2026-10-10

The disposable legacy records were reset in a single database transaction after
snapshot `engram-before-auth-20261010` became available. A temporary maintenance
rule kept `/v1/*` closed during the reset and rollout; it has been removed.
The stack update added a Cognito pool/client and replaced the application image
without replacing the service or changing the RDS/network resource properties.
The stack reports UPDATE_COMPLETE and one healthy running task.

Real PostgreSQL checks confirmed memory and transcript persistence across fresh
connections, user isolation, and complete deletion. Two Cognito accounts then
passed live sign-in, authorization on all eight routes, cross-session Bedrock
recall, refresh-token rotation, quarantine, thread deletion, and whole-user
deletion. Test accounts and their application data were removed. The previously
unauthenticated memory endpoint now returns HTTP 401 without a bearer token.

## Remaining work, in order

1. **Operations:** decide when to remove the pre-reset recovery snapshot, and
   add a repeatable database backup policy before storing non-demo user data.
   The current database has zero automated backup retention. Keep the snapshot
   until recovery is no longer needed.
2. **Live extraction quality:** establish a larger held-out live sample, save
   machine-readable results, and evaluate changes against both precision and
   recall. The recorded 31.6% precision remains unresolved; this patch does not
   claim a model-quality gain.
3. **False-premise questions:** add independent evaluation coverage and measure
   refusal behavior. LoCoMo adversarial accuracy remains 10.5% in the recorded run.
4. **Evaluation gaps:** benchmark decay across time, expand live samples, and
   retain run metadata/results. GitHub Actions passed for deployed commit
   `0c3c494` on 2026-10-10.
5. **Optional extensions:** AgentCore Memory comparison and an event-time field
   separate from embedded text. A browser login/client UI is not part of the
   current API/CLI project.

## Verification scope

Verified locally on 2026-10-10: **287 tests passed, 3 skipped**, lint and strict
type checks passed, the eight-act demo passed, and the deterministic benchmark
passed its CI thresholds. The three infrastructure template tests passed and the
stack synthesized. Documentation links were checked. Cognito tests use real RSA
signatures and a local JWKS fixture. GitHub Actions passed for the deployed
commit, including the infrastructure and benchmark jobs. PostgreSQL durability, live Bedrock
recall, and end-to-end Cognito sign-in were verified against the deployment.
These checks do not measure improved live extraction precision.

## Where it fails

Stated rather than buried.

| Weakness | Detail |
|---|---|
| **Live extraction precision: 31.6%** | Against 87.9% offline. A real model extracts from turns the rule fixture ignores. Re-measured after prompt fixes and unchanged, so it is standing, not stale. **The single biggest thing to fix.** |
| **The offline benchmark cannot see that** | CI runs the rule fixture, which holds extraction at a quality real models do not reach. The gate protects the *policy*, not the extractor. |
| **Adversarial questions: 10.5% on LoCoMo** | The system refuses untrusted *sources* structurally, and fails to refuse false *premises*. Provenance is checked; plausibility is not. |
| **Semantic retrieval misses** | "Build me a chart of weekly signups" does not retrieve "I'm colorblind, so avoid red/green pairings". This is the recorded fastembed result. The current hashing run misses the SQL-confirmation case instead; failure identity depends on the embedder. |
| **The hand-authored benchmark is near saturation** | 96.9% on the manager arm. It still gates regressions but can barely measure improvements, which is the argument for more LoCoMo rather than more hand-authored cases. |
| **The content-marker gate layer is a heuristic** | A determined author can phrase around it. The provenance layer is what the 100% rests on. |
| **Decay is unit-tested, not benchmarked** | The benchmark has no time axis, so a 7-day TTL never lapses during a run. Claiming a decay score from it would be dishonest. |
| **Substring scoring can fail a correct answer** | If a new fact's own wording contains the old value, `reject_any` trips even though memory did the right thing. This is the reason an LLM judge eventually earns its place for live runs. |

### Deliberately not built

- **A DynamoDB or S3 Vectors store adapter.** See "Why RDS and not a vector
  database" above.
- **AgentCore Runtime as the host.** See above. AgentCore *Memory* as a benchmark
  arm is the interesting version of that idea and remains open.
- **SSE streaming on `/v1/chat`.** The interesting payload is the memory trace,
  which arrives with the reply.

---
