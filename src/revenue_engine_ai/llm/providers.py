"""FastAPI dependencies that hand out the configured providers; tests override them."""

from functools import lru_cache

from fastapi import Depends

from revenue_engine_ai.core.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_DIMENSIONS, get_settings
from revenue_engine_ai.core.errors import UpstreamError
from revenue_engine_ai.core.security import ServiceContext, service_context
from revenue_engine_ai.llm.local_embedder import LocalEmbedder
from revenue_engine_ai.llm.ports import LLM, Embedder
from revenue_engine_ai.llm.workspace_llm import WorkspaceLLMs


@lru_cache
def get_llm_resolver() -> WorkspaceLLMs:
    return WorkspaceLLMs(get_settings())


def get_llm(
    ctx: ServiceContext = Depends(service_context), resolver: WorkspaceLLMs = Depends(get_llm_resolver)
) -> LLM:
    """The LLM for this call's workspace: the provider and key its admin saved (admin > AI provider).
    A workspace without one gets a "provider not set" error; there is no service-level key."""
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
