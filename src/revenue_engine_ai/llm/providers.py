"""FastAPI dependencies that hand out the configured providers; tests override them."""

from functools import lru_cache

from fastapi import Depends
from openai import AsyncOpenAI

from revenue_engine_ai.core.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_DIMENSIONS, get_settings
from revenue_engine_ai.core.errors import UpstreamError
from revenue_engine_ai.core.security import ServiceContext, service_context
from revenue_engine_ai.llm.chat_completions_provider import ChatCompletionsLLM
from revenue_engine_ai.llm.local_embedder import LocalEmbedder
from revenue_engine_ai.llm.openai_provider import OpenAILLM
from revenue_engine_ai.llm.ports import LLM, Embedder
from revenue_engine_ai.llm.workspace_llm import WorkspaceLLMs


@lru_cache
def _client() -> AsyncOpenAI:
    settings = get_settings()
    key = settings.openai_api_key.get_secret_value()
    if not key:
        raise UpstreamError("The AI provider is not configured (RE_AI_OPENAI_API_KEY).")
    return AsyncOpenAI(
        api_key=key,
        base_url=settings.openai_base_url or None,
        timeout=settings.request_timeout_s,
        max_retries=2,
    )


PROVIDERS: dict[str, type[OpenAILLM] | type[ChatCompletionsLLM]] = {
    "responses": OpenAILLM,
    "chat_completions": ChatCompletionsLLM,
}


@lru_cache
def _own_llm() -> LLM:
    """The service's own account (RE_AI_OPENAI_API_KEY): for a workspace that has not set up its own."""
    settings = get_settings()
    return PROVIDERS[settings.llm_api](_client(), settings)


@lru_cache
def get_llm_resolver() -> WorkspaceLLMs:
    return WorkspaceLLMs(get_settings(), _own_llm)


def get_llm(
    ctx: ServiceContext = Depends(service_context), resolver: WorkspaceLLMs = Depends(get_llm_resolver)
) -> LLM:
    """The LLM for this call's workspace: its own provider and key when an admin configured one."""
    return resolver.get(ctx)


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if EMBEDDING_MODEL_DIMENSIONS.get(settings.embedding_model) != EMBEDDING_DIMENSIONS:
        raise UpstreamError(
            f"Embedding model '{settings.embedding_model}' does not match the index size "
            f"({EMBEDDING_DIMENSIONS}); a new migration and re-index are needed."
        )
    return LocalEmbedder(settings)
