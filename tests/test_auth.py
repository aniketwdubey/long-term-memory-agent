"""Real signed JWTs, offline JWKS, and authorization over every API route."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from engram.api.main import create_app
from engram.config import Settings

POOL = "us-east-1_TestPool"
CLIENT = "testclient"
ISSUER = f"https://cognito-idp.us-east-1.amazonaws.com/{POOL}"


@pytest.fixture(scope="module")
def key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def token(key: rsa.RSAPrivateKey, **overrides: Any) -> str:
    claims = {
        "iss": ISSUER,
        "sub": "alice",
        "client_id": CLIENT,
        "token_use": "access",
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
        **overrides,
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})


@pytest.fixture
def client(key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    public = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    public.update(kid="test-key", use="sig", alg="RS256")
    # Replace only the network boundary; real key selection and verification run.
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", lambda self: {"keys": [public]})
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        chat_provider="stub",
        embedder="hashing",
        store_backend="memory",
        auth_mode="cognito",
        cognito_user_pool_id=POOL,
        cognito_client_id=CLIENT,
    )
    with TestClient(create_app(settings)) as c:
        yield c


ROUTES = [
    ("POST", "/v1/chat", {"user_id": "bob", "message": "I use Neovim."}),
    ("POST", "/v1/observe", {"user_id": "bob", "text": "external text"}),
    ("GET", "/v1/memories/bob", None),
    ("GET", "/v1/memories/bob/history/default", None),
    ("GET", "/v1/quarantine/bob", None),
    ("DELETE", "/v1/memories/bob", None),
    ("DELETE", "/v1/memories/bob/threads/default", None),
    ("DELETE", "/v1/memories/bob/fact", None),
]


@pytest.mark.parametrize("method,path,body", ROUTES)
def test_all_memory_routes_require_a_token(
    client: TestClient, method: str, path: str, body: dict[str, str] | None
) -> None:
    response = client.request(method, path, json=body)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("method,path,body", ROUTES)
def test_all_memory_routes_enforce_the_signed_in_owner(
    client: TestClient,
    key: rsa.RSAPrivateKey,
    method: str,
    path: str,
    body: dict[str, str] | None,
) -> None:
    response = client.request(
        method, path, json=body, headers={"Authorization": f"Bearer {token(key)}"}
    )
    assert response.status_code == 403


def test_signed_in_users_can_chat_inspect_and_forget_their_own_data(
    client: TestClient, key: rsa.RSAPrivateKey
) -> None:
    headers = {"Authorization": f"Bearer {token(key)}"}
    response = client.post(
        "/v1/chat", json={"user_id": "alice", "message": "I use Neovim."}, headers=headers
    )
    assert response.status_code == 200
    assert client.get("/v1/memories/alice", headers=headers).json()["total"] == 1
    assert client.get("/v1/memories/alice/history/default", headers=headers).json()["history"]
    assert client.delete("/v1/memories/alice", headers=headers).status_code == 200
    assert client.get("/v1/memories/alice/history/default", headers=headers).json()["history"] == []


@pytest.mark.parametrize(
    "claims",
    [
        {"exp": 1},
        {"iss": "https://attacker.example"},
        {"client_id": "other-client"},
        {"token_use": "id"},
        {"sub": ""},
        {"iat": 9999999999},
        {"exp": None},
        {"sub": None},
    ],
)
def test_invalid_claims_are_rejected(
    client: TestClient, key: rsa.RSAPrivateKey, claims: dict[str, Any]
) -> None:
    response = client.get(
        "/v1/memories/alice", headers={"Authorization": f"Bearer {token(key, **claims)}"}
    )
    assert response.status_code == 401


def test_an_untrusted_signing_key_is_rejected(client: TestClient) -> None:
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert (
        client.get(
            "/v1/memories/alice", headers={"Authorization": f"Bearer {token(attacker_key)}"}
        ).status_code
        == 401
    )


@pytest.mark.parametrize("bad_token", ["not-a-jwt", "", "a.b.c"])
def test_malformed_credentials_are_rejected(client: TestClient, bad_token: str) -> None:
    assert (
        client.get(
            "/v1/memories/alice", headers={"Authorization": f"Bearer {bad_token}"}
        ).status_code
        == 401
    )


def test_health_remains_public(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_auth_service_failure_does_not_allow_access(
    client: TestClient, key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(self: Any) -> Any:
        raise jwt.PyJWKClientConnectionError("offline")

    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", unavailable)
    assert (
        client.get(
            "/v1/memories/alice", headers={"Authorization": f"Bearer {token(key)}"}
        ).status_code
        == 503
    )


def test_api_does_not_start_without_auth_configuration() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with (
        pytest.raises(ValueError, match="Cognito authentication requires"),
        TestClient(create_app(settings)),
    ):
        pass


@pytest.mark.parametrize("algorithm", ["none", "HS256"])
def test_other_signing_algorithms_are_rejected(client: TestClient, algorithm: str) -> None:
    forged = jwt.encode(
        {"sub": "alice"},
        "attacker-secret-at-least-32-bytes-long" if algorithm == "HS256" else None,
        algorithm=algorithm,
        headers={"kid": "test-key"},
    )
    assert (
        client.get("/v1/memories/alice", headers={"Authorization": f"Bearer {forged}"}).status_code
        == 401
    )


def test_unknown_signing_key_is_rejected(client: TestClient, key: rsa.RSAPrivateKey) -> None:
    unknown = jwt.encode({"sub": "alice"}, key, algorithm="RS256", headers={"kid": "unknown"})
    assert (
        client.get("/v1/memories/alice", headers={"Authorization": f"Bearer {unknown}"}).status_code
        == 401
    )
