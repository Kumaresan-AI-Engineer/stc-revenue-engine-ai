"""Embeddings through the cache: a text is embedded once per model, then always gets the same vector."""

from fastapi import Depends

from revenue_engine_ai.llm.ports import Embedder
from revenue_engine_ai.llm.providers import get_embedder
from revenue_engine_ai.rules.text import stable_hash
from revenue_engine_ai.store.ports import EmbeddingCache
from revenue_engine_ai.store.postgres import embedding_cache


class Vectorizer:
    def __init__(self, embedder: Embedder, cache: EmbeddingCache) -> None:
        self._embedder = embedder
        self._cache = cache

    async def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [stable_hash(self._embedder.model, t) for t in texts]
        known = await self._cache.get_many(list(dict.fromkeys(keys)))
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in known}
        if missing:
            fresh = await self._embedder.embed(list(missing.values()))
            new = dict(zip(missing, fresh, strict=True))
            await self._cache.put_many(new)
            known.update(new)
        return [known[k] for k in keys]


def vectorizer(
    embedder: Embedder = Depends(get_embedder), cache: EmbeddingCache = Depends(embedding_cache)
) -> Vectorizer:
    return Vectorizer(embedder, cache)
