import httpx
import openai
import pytest

from revenue_engine_ai.core.errors import PROVIDER_CREDENTIAL_REJECTED, ProviderCredentialError, UpstreamError
from revenue_engine_ai.llm.failures import (
    MODEL_UNKNOWN,
    NO_ANSWER,
    NO_STRUCTURED,
    NO_TOOLS,
    NOT_REACHED,
    OUTPUT_LIMIT,
    RATE_LIMITED,
    REFUSED,
    TOO_LARGE,
    provider_failure,
)

REQUEST = httpx.Request("POST", "https://llm.example/v1/chat/completions")


def _status(
    error: type[openai.APIStatusError], code: int, message: str = "secret-key-echoed-by-the-provider"
) -> openai.APIStatusError:
    return error(message, response=httpx.Response(code, request=REQUEST), body=None)


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


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("max_completion_tokens is too large: 32000. This model supports at most 16384", OUTPUT_LIMIT),
        ("Unsupported parameter: 'max_tokens' is not supported with this model", OUTPUT_LIMIT),
        ("This model's maximum context length is 8192 tokens", TOO_LARGE),
        ("'response_format.type' must be 'json_schema' or 'text'", NO_STRUCTURED),
        ("This model does not support tools", NO_TOOLS),
        ("something else is wrong with the request", REFUSED),
    ],
)
def test_a_refused_request_says_what_the_model_cannot_do(message: str, expected: str) -> None:
    for error in (openai.BadRequestError, openai.UnprocessableEntityError):
        code = 400 if error is openai.BadRequestError else 422
        failure = provider_failure(_status(error, code, message), "extraction")
        assert isinstance(failure, UpstreamError) and failure.detail == expected


def test_a_request_that_is_too_large_is_told_so() -> None:
    failure = provider_failure(_status(openai.APIStatusError, 413), "extraction")

    assert failure.detail == TOO_LARGE


def test_a_refusal_never_passes_on_the_providers_words() -> None:
    message = "max_tokens above the limit for key sk-live-abc123"

    failure = provider_failure(_status(openai.BadRequestError, 400, message), "extraction")

    assert "sk-live" not in failure.detail
