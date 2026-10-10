# Documentation index

Start with the [project overview](../README.md), then choose a guide:

1. [Status and remaining work](status.md): what is complete, what is verified, and what comes next.
2. [Operations](operations.md): run locally, configure storage/models, deploy, and handle old data.
3. [Authentication](authentication.md): separate Cognito accounts and API access.
4. [Architecture and design](architecture.md): graph, memory policy, and design decisions.
5. [Evaluation](evaluation.md): methods, reported results, and known limitations.

[architecture.svg](architecture.svg) is the diagram used in the README.
[.env.example](../.env.example), the [root Makefile](../Makefile), and the
[CI workflow](../.github/workflows/ci.yml) are the executable configuration references.

## Private local documents

`docs/PROJECT_CONTEXT.md` is the private handoff containing historical deployment
identifiers, debugging lessons, and working preferences. `docs/blog.md` is an
article draft. Both are git-ignored on purpose; do not publish them as part of a
documentation cleanup. Neither is required to run the project from a fresh clone.

The public status page is the source for current work. Older experiments in the
private handoff and blog are historical, not newer benchmark measurements.
