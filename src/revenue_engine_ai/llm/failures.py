"""What a failed provider call means to the user, in plain words and with nothing from the provider's answer
(it can echo parts of the request, and the key).

Both LLM clients (`openai_provider`, `chat_completions_provider`) turn an `openai.APIError` into one of
these, so an expired key, a wrong model name, an unreachable server and a rate limit are told apart
instead of all being "did not answer".
"""

import logging
from http import HTTPStatus

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
OUTPUT_LIMIT = (
    "The model cannot write as long an answer as this service asks for. "
    "Lower RE_AI_MAX_OUTPUT_TOKENS, or choose a model with a higher limit."
)
TOO_LARGE = (
    "The request is larger than the provider allows for this model or account. "
    "Try a smaller document, or a model with a higher limit."
)
NO_STRUCTURED = (
    "The model or server does not support the structured (JSON) answers this system needs. "
    "Choose another model."
)
NO_TOOLS = "The model or server does not support tool calling, which this system needs. Choose another model."
REFUSED = "The AI provider refused the request. Check the model name and the API mode."

# What a refusal (HTTP 400/422) is about, told from the words in the provider's answer: the first match wins.
# The words only pick a message of ours; none of the answer is passed on.
_REFUSALS = (
    (("max_tokens", "max_completion_tokens", "max_output_tokens"), OUTPUT_LIMIT),
    (("context length", "context_length", "context window"), TOO_LARGE),
    (("response_format", "json_object", "json_schema"), NO_STRUCTURED),
    (("tool", "function"), NO_TOOLS),
)


def _refusal(exc: openai.APIError) -> str:
    text = str(getattr(exc, "message", "") or exc).lower()
    return next((message for words, message in _REFUSALS if any(w in text for w in words)), REFUSED)


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
    if status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
        return UpstreamError(TOO_LARGE)
    if isinstance(exc, openai.BadRequestError | openai.UnprocessableEntityError):
        return UpstreamError(_refusal(exc))
    return UpstreamError(NO_ANSWER)
