"""Pin the environment before the app is imported, and wire every external dependency to a fake."""

import os

os.environ.update(
    {
        "RE_AI_ENV": "test",
        "RE_AI_SERVICE_TOKEN_SECRET": "test-secret-that-is-long-enough-for-hs256",
        "RE_AI_OPENAI_API_KEY": "",
    }
)

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import jwt
import pytest
from fakes import FakeEmbedder, FakeLLM, InMemoryEmbeddingCache, InMemoryIndex, InMemoryResultCache

from core.config import get_settings
from llm.providers import get_embedder, get_llm
from main import create_app
from services.vectors import Vectorizer, vectorizer
from store.postgres import library_index, result_cache


def token(workspace_id: uuid.UUID, **overrides: object) -> str:
    settings = get_settings()
    claims: dict[str, object] = {
        "workspace_id": str(workspace_id),
        "actor_id": str(uuid.uuid4()),
        "aud": settings.service_token_audience,
        "exp": int(time.time()) + 60,
        "can_see_restricted": False,
    }
    claims.update(overrides)
    return jwt.encode(claims, settings.service_token_secret.get_secret_value(), algorithm="HS256")


@dataclass
class Env:
    client: httpx.AsyncClient
    llm: FakeLLM
    embedder: FakeEmbedder
    index: InMemoryIndex
    results: InMemoryResultCache
    workspace_id: uuid.UUID

    def auth(self, workspace_id: uuid.UUID | None = None) -> dict[str, str]:
        return {"Authorization": f"Bearer {token(workspace_id or self.workspace_id)}"}


@pytest.fixture
async def env() -> AsyncIterator[Env]:
    app = create_app()
    llm, embedder = FakeLLM(), FakeEmbedder()
    index, embeddings, results = InMemoryIndex(), InMemoryEmbeddingCache(), InMemoryResultCache()
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[library_index] = lambda: index
    app.dependency_overrides[result_cache] = lambda: results
    app.dependency_overrides[vectorizer] = lambda: Vectorizer(embedder, embeddings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ai") as client:
        yield Env(client, llm, embedder, index, results, uuid.uuid4())
