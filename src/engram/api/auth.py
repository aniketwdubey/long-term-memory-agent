"""Cognito access-token verification and per-user authorization."""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from engram.config import Settings

bearer = HTTPBearer(auto_error=False)


class CognitoVerifier:
    def __init__(self, settings: Settings) -> None:
        if not settings.cognito_user_pool_id or not settings.cognito_client_id:
            raise ValueError(
                "Cognito authentication requires ENGRAM_COGNITO_USER_POOL_ID and "
                "ENGRAM_COGNITO_CLIENT_ID. For local demos only, set ENGRAM_AUTH_MODE=disabled."
            )
        region = settings.cognito_user_pool_id.split("_", 1)[0]
        self.issuer = f"https://cognito-idp.{region}.amazonaws.com/{settings.cognito_user_pool_id}"
        self.client_id = settings.cognito_client_id
        self.jwks = PyJWKClient(f"{self.issuer}/.well-known/jwks.json", timeout=5)

    def subject(self, token: str) -> str:
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
            raise jwt.InvalidTokenError("Invalid signing algorithm or key ID")
        key = self.jwks.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256"],
            issuer=self.issuer,
            options={
                # Cognito access tokens identify the app in client_id, not aud.
                "verify_aud": False,
                "require": ["exp", "iat", "iss", "sub", "client_id", "token_use"],
            },
        )
        if claims["token_use"] != "access" or claims["client_id"] != self.client_id:
            raise jwt.InvalidTokenError("Wrong token use or app client")
        subject = claims["sub"]
        if not isinstance(subject, str) or not subject:
            raise jwt.InvalidTokenError("Missing subject")
        return subject


def current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> str | None:
    if request.app.state.settings.auth_mode == "disabled":
        return None
    if credentials is None:
        raise HTTPException(
            401, "Bearer access token required", headers={"WWW-Authenticate": "Bearer"}
        )
    verifier: CognitoVerifier = request.app.state.verifier
    try:
        return verifier.subject(credentials.credentials)
    except jwt.PyJWKClientConnectionError as exc:
        raise HTTPException(503, "Authentication service unavailable") from exc
    except (jwt.InvalidTokenError, jwt.PyJWKClientError) as exc:
        raise HTTPException(
            401, "Invalid access token", headers={"WWW-Authenticate": "Bearer"}
        ) from exc


UserDep = Annotated[str | None, Depends(current_user)]


def require_owner(subject: str | None, user_id: str) -> None:
    if subject is not None and subject != user_id:
        raise HTTPException(403, "Cannot access another user's data")
