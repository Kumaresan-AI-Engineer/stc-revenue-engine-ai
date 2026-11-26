"""FastAPI dependencies that hand out the configured providers; tests override them."""

from functools import lru_cache
from typing import Protocol

from fastapi import Depends

from revenue_engine_ai.core.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_DIMENSIONS, get_settings
from revenue_engine_ai.core.errors import UpstreamError
from revenue_engine_ai.core.security import ServiceContext, service_context
from revenue_engine_ai.llm.local_embedder import LocalEmbedder
from revenue_engine_ai.llm.ports import LLM, Embedder
from revenue_engine_ai.llm.workspace_llm import NO_KEY_PLACEHOLDER, WorkspaceLLMs, make_llm


@lru_cache
def get_llm_resolver() -> WorkspaceLLMs:
    return WorkspaceLLMs(get_settings())


def get_llm(
    ctx: ServiceContext = Depends(service_context), resolver: WorkspaceLLMs = Depends(get_llm_resolver)
) -> LLM:
    """The LLM for this call's workspace: the provider and key its admin saved (admin > AI provider).
    A workspace without one gets a "provider not set" error; there is no service-level key."""
    return resolver.get(ctx)


class ProbeLLMBuilder(Protocol):
    def __call__(self, *, api_mode: str, base_url: str | None, model: str, api_key: str | None) -> LLM: ...


def get_probe_llm_builder() -> ProbeLLMBuilder:
    """Builds the client for a model being tried before it is saved: the settings and key come with the
    request, since nothing is stored yet. One try only (no retries), so a failure is reported at once."""
    settings = get_settings()

    def build(*, api_mode: str, base_url: str | None, model: str, api_key: str | None) -> LLM:
        return make_llm(
            settings,
            api_mode=api_mode,
            base_url=base_url,
            model=model,
            api_key=api_key or NO_KEY_PLACEHOLDER,
            max_retries=0,
        )

    return build


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if EMBEDDING_MODEL_DIMENSIONS.get(settings.embedding_model) != EMBEDDING_DIMENSIONS:
        raise UpstreamError(
            f"Embedding model '{settings.embedding_model}' does not match the index size "
            f"({EMBEDDING_DIMENSIONS}); a new migration and re-index are needed."
        )
    return LocalEmbedder(settings)
