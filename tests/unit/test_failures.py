import httpx
import openai
import pytest

from revenue_engine_ai.core.errors import PROVIDER_CREDENTIAL_REJECTED, ProviderCredentialError, UpstreamError
from revenue_engine_ai.llm.failures import (
    MODEL_UNKNOWN,
    NO_ANSWER,
    NOT_REACHED,
    RATE_LIMITED,
    provider_failure,
)

REQUEST = httpx.Request("POST", "https://llm.example/v1/chat/completions")


def _status(error: type[openai.APIStatusError], code: int) -> openai.APIStatusError:
    return error(
        "secret-key-echoed-by-the-provider", response=httpx.Response(code, request=REQUEST), body=None
    )


def test_an_expired_or_refused_key_is_a_credential_error_the_admin_can_fix() -> None:
    for exc in (_status(openai.AuthenticationError, 401), _status(openai.PermissionDeniedError, 403)):
        failure = provider_failure(exc, "extraction")
        assert isinstance(failure, ProviderCredentialError)
        assert failure.extra["rule"] == PROVIDER_CREDENTIAL_REJECTED
        assert "Admin > AI provider" in failure.detail


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (_status(openai.RateLimitError, 429), RATE_LIMITED),
        (_status(openai.NotFoundError, 404), MODEL_UNKNOWN),
        (openai.APIConnectionError(request=REQUEST), NOT_REACHED),
        (openai.APITimeoutError(request=REQUEST), NOT_REACHED),
        (_status(openai.InternalServerError, 500), NO_ANSWER),
    ],
)
def test_other_failures_are_told_apart(exc: openai.APIError, message: str) -> None:
    failure = provider_failure(exc, "extraction")
    assert isinstance(failure, UpstreamError) and failure.detail == message


def test_nothing_from_the_provider_answer_reaches_the_user() -> None:
    failure = provider_failure(_status(openai.AuthenticationError, 401), "extraction")
    assert "secret-key-echoed" not in failure.detail
