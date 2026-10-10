# Authentication and user accounts

[Documentation index](README.md) · [Operations](operations.md)

The HTTP API supports separate Amazon Cognito accounts. Every `/v1` endpoint
requires a bearer **access token** when authentication is enabled. The token's
`sub` is the user's immutable ID. Request-body and path `user_id` values must
match that subject; another user's ID returns HTTP 403 before any data is read
or changed. Missing or invalid credentials return 401.

The API validates the RS256 signature, issuer, expiry, issued-at time, token use,
app client, and subject. Public signing keys come only from the configured pool's
JWKS endpoint and are cached by PyJWT. ID tokens are rejected. A signing-key
service outage fails closed with 503. This follows
[AWS's JWT verification guidance](https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html).

## Local and deployed modes

- `make serve-demo`: authentication disabled explicitly, loopback only.
- `make serve`: Cognito authentication by default; supply the two identifiers below.
- Local Docker Compose: demo mode by default, host port bound to `127.0.0.1`.
- CDK deployment: creates a Cognito pool/client and enables authentication in ECS.

```bash
export ENGRAM_AUTH_MODE=cognito
export ENGRAM_COGNITO_USER_POOL_ID=us-east-1_YOURPOOL
export ENGRAM_COGNITO_CLIENT_ID=YOUR_APP_CLIENT_ID
make serve
```

Pool and client IDs are public configuration, not passwords. Missing Cognito
configuration prevents API startup. CLI demos and the evaluation harness do not
serve HTTP and remain usable offline.

## Sign-up and sign-in

The stack enables email sign-up and verification, a 12-character minimum
password, optional authenticator-app MFA, and SRP sign-in. The public app client
has no secret. Its access/ID tokens last five minutes; refresh tokens last seven
days with rotation and revocation enabled. The stack outputs `UserPoolId` and
`UserPoolClientId` for a client application to consume. The 2026-10-10 deployment
verified SRP sign-in and refresh-token rotation with two temporary accounts.

Use Cognito's client SDK (for example Amplify Auth) to sign up, confirm the email
code, sign in with SRP, and refresh the session. The project currently provides
an API and local CLI, not a browser login application. The local CLI calls the
Python agent directly; it is an operator tool, not a remote authenticated client.

After sign-in, send the session's **access token** in `Authorization: Bearer ...`
and use its `sub` for every `user_id`. The Swagger UI at `/docs` has an Authorize
control for an already-issued access token. Do not paste tokens into source,
commit them, or print them in request logs.

```http
POST /v1/chat
Authorization: Bearer <access-token>
Content-Type: application/json

{"user_id":"<token-sub>","thread_id":"monday","message":"I prefer pytest."}
```

`GET /health` stays public for the load balancer. `/docs` and the OpenAPI schema
are public metadata; their API calls still require authorization.

## Boundaries and existing data

A thread name is local to its owner. Two users may both use `default` without
sharing model context, history, or deletion. `Agent.history` now requires
`(user_id, thread_id)`. This storage boundary also protects direct Python callers
from accidental thread collisions; direct callers remain trusted operators.

`POST /v1/observe` accepts only `tool` and `document` provenance. User assertions
belong on `/v1/chat`; callers cannot relabel observed content as a trusted source.

Earlier data keyed by names such as `alice` is not automatically assigned to a
new Cognito subject. Verify ownership before migrating it. Old globally keyed
transcripts can contain mixed-user content; see the
[upgrade procedure](operations.md#upgrading-existing-data).

Offline JWT verification cannot detect revocation before token expiry. An
already-issued access token can remain usable for up to five minutes after
sign-out. Account deletion in Cognito and `DELETE /v1/memories/{user_id}` are
separate operations: the latter deletes application data, not the login account.
