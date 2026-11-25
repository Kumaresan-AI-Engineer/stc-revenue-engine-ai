"""Domain errors rendered as RFC 9457 problem+json, the same contract as the backend."""

import logging
from http import HTTPStatus
from typing import Any, ClassVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

PROBLEM_JSON = "application/problem+json"

log = logging.getLogger(__name__)


class DomainError(Exception):
    status = HTTPStatus.INTERNAL_SERVER_ERROR
    title = "Internal Server Error"
    default_headers: ClassVar[dict[str, str] | None] = None

    def __init__(self, detail: str | None = None, **extra: Any) -> None:
        super().__init__(detail or self.title)
        self.detail = detail or self.title
        self.extra = extra


class Unauthorized(DomainError):
    status = HTTPStatus.UNAUTHORIZED
    title = "Unauthorized"
    default_headers: ClassVar[dict[str, str] | None] = {"WWW-Authenticate": "Bearer"}


class NotFound(DomainError):
    status = HTTPStatus.NOT_FOUND
    title = "Not Found"


class RuleViolation(DomainError):
    status = HTTPStatus.UNPROCESSABLE_ENTITY
    title = "Rule Violation"

    def __init__(self, rule: str, detail: str, **extra: Any) -> None:
        super().__init__(detail, rule=rule, **extra)


class UpstreamError(DomainError):
    status = HTTPStatus.BAD_GATEWAY
    title = "Upstream service failed"


# The workspace's AI provider is the only source of AI credentials, so these two are rules the admin can fix
# (422 with a `rule`, which the backend passes on to the user as it is).
PROVIDER_NOT_SET = "ai-provider-not-set"
PROVIDER_CREDENTIAL_REJECTED = "ai-credential-rejected"


class ProviderNotConfigured(RuleViolation):
    def __init__(self) -> None:
        super().__init__(
            PROVIDER_NOT_SET,
            "No AI provider is set for this workspace. Ask an admin to add one in Admin > AI provider.",
        )


class ProviderCredentialError(RuleViolation):
    def __init__(self) -> None:
        super().__init__(
            PROVIDER_CREDENTIAL_REJECTED,
            "The AI provider did not accept this workspace's API key (it may have expired). "
            "Ask an admin to replace it in Admin > AI provider.",
        )


def _problem(status: int, title: str, detail: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "about:blank", "title": title, "status": status, "detail": detail, **(extra or {})}


def install_error_handlers(app: FastAPI) -> None:
    async def domain(_: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, DomainError)
        return JSONResponse(
            _problem(exc.status, exc.title, exc.detail, exc.extra),
            status_code=exc.status,
            media_type=PROBLEM_JSON,
            headers=exc.default_headers,
        )

    async def validation(_: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RequestValidationError)
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        body = _problem(422, "Validation Error", "The request is not valid.", {"errors": errors})
        return JSONResponse(body, status_code=422, media_type=PROBLEM_JSON)

    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", exc_info=exc)
        body = _problem(500, "Internal Server Error", "Something went wrong on our side.")
        return JSONResponse(body, status_code=500, media_type=PROBLEM_JSON)

    app.add_exception_handler(DomainError, domain)
    app.add_exception_handler(RequestValidationError, validation)
    app.add_exception_handler(Exception, unexpected)
