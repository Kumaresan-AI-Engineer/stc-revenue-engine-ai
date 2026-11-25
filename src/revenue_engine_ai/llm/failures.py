"""What a failed provider call means to the user, in plain words and with nothing from the provider's answer
(it can echo parts of the request, and the key).

Both LLM clients (`openai_provider`, `chat_completions_provider`) turn an `openai.APIError` into one of
these, so an expired key, a wrong model name, an unreachable server and a rate limit are told apart
instead of all being "did not answer".
"""

import logging

import openai

from revenue_engine_ai.core.errors import DomainError, ProviderCredentialError, UpstreamError

log = logging.getLogger(__name__)

NOT_REACHED = (
    "The AI provider could not be reached. "
    "Check its address in Admin > AI provider, and that the server is running."
)
RATE_LIMITED = "The AI provider is limiting requests or the account is out of quota. Try again in a moment."
MODEL_UNKNOWN = "The AI provider does not have the model chosen in Admin > AI provider. Check the model name."
NO_ANSWER = "The AI provider did not answer. Try again in a moment."


def provider_failure(exc: openai.APIError, agent: str) -> DomainError:
    """The error to raise for `exc`. Logs the class and HTTP status, never the message (it may echo a key)."""
    status = getattr(exc, "status_code", None)
    log.warning(
        "llm_call_failed",
        extra={
            "fields": {
                "agent": agent,
                "error": type(exc).__name__,
                "status": status,
                "key_source": "workspace",
            }
        },
    )
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return ProviderCredentialError()
    if isinstance(exc, openai.RateLimitError):
        return UpstreamError(RATE_LIMITED)
    if isinstance(exc, openai.NotFoundError):
        return UpstreamError(MODEL_UNKNOWN)
    if isinstance(exc, openai.APIConnectionError):  # includes timeouts
        return UpstreamError(NOT_REACHED)
    return UpstreamError(NO_ANSWER)
