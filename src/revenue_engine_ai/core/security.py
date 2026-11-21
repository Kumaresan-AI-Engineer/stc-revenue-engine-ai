"""Service-to-service authentication. Only the backend calls this service, with a short-lived JWT
minted per call. The workspace comes only from the token, never from a body or a document."""

import uuid
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, Header

from revenue_engine_ai.core.config import Settings, get_settings
from revenue_engine_ai.core.errors import Unauthorized

REQUIRED_CLAIMS = ["exp", "aud", "iss", "workspace_id", "actor_id"]


@dataclass(frozen=True)
class ServiceContext:
    workspace_id: uuid.UUID
    actor_id: uuid.UUID
    can_see_restricted: bool
    request_id: str | None


def decode_service_token(token: str, settings: Settings) -> ServiceContext:
    secret = settings.service_token_secret.get_secret_value()
    if not secret:
        raise Unauthorized("Service authentication is not configured.")
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience=settings.service_token_audience,
            issuer=settings.service_token_issuer,
            options={"require": REQUIRED_CLAIMS},
        )
        return ServiceContext(
            workspace_id=uuid.UUID(claims["workspace_id"]),
            actor_id=uuid.UUID(claims["actor_id"]),
            can_see_restricted=bool(claims.get("can_see_restricted", False)),
            request_id=claims.get("request_id"),
        )
    except (jwt.PyJWTError, ValueError) as exc:
        raise Unauthorized("The service token is not valid.") from exc


def service_context(
    authorization: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> ServiceContext:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Unauthorized("A service token is required.")
    return decode_service_token(token, settings)
