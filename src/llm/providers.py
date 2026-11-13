"""FastAPI dependencies that hand out the configured providers; tests override them."""

from functools import lru_cache

from openai import AsyncOpenAI

from core.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_DIMENSIONS, get_settings
from core.errors import UpstreamError
from llm.local_embedder import LocalEmbedder
from llm.openai_provider import OpenAILLM
from llm.ports import LLM, Embedder


@lru_cache
def _client() -> AsyncOpenAI:
    settings = get_settings()
    key = settings.openai_api_key.get_secret_value()
    if not key:
        raise UpstreamError("The AI provider is not configured (RE_AI_OPENAI_API_KEY).")
    return AsyncOpenAI(api_key=key, timeout=settings.request_timeout_s, max_retries=2)


@lru_cache
def get_llm() -> LLM:
    return OpenAILLM(_client(), get_settings())


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if EMBEDDING_MODEL_DIMENSIONS.get(settings.embedding_model) != EMBEDDING_DIMENSIONS:
        raise UpstreamError(
            f"Embedding model '{settings.embedding_model}' does not match the index size "
            f"({EMBEDDING_DIMENSIONS}); a new migration and re-index are needed."
        )
    return LocalEmbedder(settings)
