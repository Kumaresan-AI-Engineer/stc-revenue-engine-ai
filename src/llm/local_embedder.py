"""Free, local embeddings with fastembed (ONNX on CPU, no API key, no data leaves the service).

The model is downloaded once to `cache_dir` and loaded on first use. Inference is CPU-bound, so it runs
in a worker thread and the event loop keeps serving other requests.
"""

import asyncio
from functools import cached_property

from fastembed import TextEmbedding

from core.config import Settings

EMBEDDING_BATCH = 64


class LocalEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.embedding_model
        self._cache_dir = settings.embedding_cache_dir
        # One model instance is not safe to run from several threads at once.
        self._lock = asyncio.Lock()

    @cached_property
    def _engine(self) -> TextEmbedding:
        return TextEmbedding(self.model, cache_dir=self._cache_dir)

    def _run(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._engine.embed(texts, batch_size=EMBEDDING_BATCH)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._run, texts)
