"""Library passages and their vectors in Qdrant.

Embedded by default (a local folder, no server); a Qdrant server when RE_AI_QDRANT_URL is set. Search is
exact (no approximate index) inside one workspace, so the same query over the same library always returns
the same passages in the same order.
"""

import asyncio
import uuid
from functools import lru_cache

from qdrant_client import AsyncQdrantClient, models

from revenue_engine_ai.core.config import EMBEDDING_DIMENSIONS, get_settings
from revenue_engine_ai.documents.chunking import Passage
from revenue_engine_ai.store.ports import Hit, StoredPassage

COLLECTION = "library_passage"
SIMILARITY_DECIMALS = 4


@lru_cache
def get_qdrant() -> AsyncQdrantClient:
    settings = get_settings()
    if settings.qdrant_url:
        return AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key.get_secret_value() or None
        )
    return AsyncQdrantClient(path=settings.qdrant_path)


def _point_id(workspace_id: uuid.UUID, passage_id: str) -> str:
    # Stable per (workspace, passage): re-indexing overwrites, and workspaces never collide.
    return str(uuid.uuid5(workspace_id, passage_id))


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _match_any(key: str, values: list[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def _stored(payload: dict[str, object]) -> StoredPassage:
    return StoredPassage(
        str(payload["passage_id"]),
        str(payload["entry_id"]),
        str(payload["section"]),
        int(str(payload["page"])),
        str(payload["text"]),
    )


class QdrantPassages:
    def __init__(
        self, client: AsyncQdrantClient, *, server: bool = False, collection: str = COLLECTION
    ) -> None:
        self._client = client
        # Embedded Qdrant always scans exactly; a server is told to, instead of using its approximate index.
        self._params = models.SearchParams(exact=True) if server else None
        self._collection = collection
        self._ready = False
        self._lock = asyncio.Lock()

    async def _ensure_collection(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if not await self._client.collection_exists(self._collection):
                await self._client.create_collection(
                    self._collection,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE
                    ),
                )
            self._ready = True

    async def add(
        self, workspace_id: uuid.UUID, entry_id: str, passages: list[Passage], vectors: list[list[float]]
    ) -> None:
        await self._ensure_collection()
        points = [
            models.PointStruct(
                id=_point_id(workspace_id, p.id),
                vector=v,
                payload={
                    "workspace_id": str(workspace_id),
                    "entry_id": entry_id,
                    "passage_id": p.id,
                    "section": p.section,
                    "page": p.page,
                    "ordinal": p.ordinal,
                    "text": p.text,
                },
            )
            for p, v in zip(passages, vectors, strict=True)
        ]
        if points:
            await self._client.upsert(self._collection, points=points, wait=True)

    async def delete(self, workspace_id: uuid.UUID, entry_id: str) -> None:
        await self._ensure_collection()
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[_match("workspace_id", str(workspace_id)), _match("entry_id", entry_id)]
            )
        )
        await self._client.delete(self._collection, points_selector=selector, wait=True)

    async def search(
        self,
        workspace_id: uuid.UUID,
        vector: list[float],
        *,
        entry_ids: list[str] | None,
        sections: list[str] | None,
        limit: int,
    ) -> list[Hit]:
        await self._ensure_collection()
        must: list[models.Condition] = [_match("workspace_id", str(workspace_id))]
        if entry_ids is not None:
            if not entry_ids:
                return []
            must.append(_match_any("entry_id", entry_ids))
        if sections is not None:
            must.append(_match_any("section", sections))
        result = await self._client.query_points(
            self._collection,
            query=vector,
            query_filter=models.Filter(must=must),
            limit=limit,
            with_payload=True,
            search_params=self._params,
        )
        hits = [Hit(_stored(p.payload or {}), round(p.score, SIMILARITY_DECIMALS)) for p in result.points]
        return sorted(hits, key=lambda h: (-h.similarity, h.passage.id))

    async def get(self, workspace_id: uuid.UUID, passage_id: str) -> StoredPassage | None:
        await self._ensure_collection()
        found = await self._client.retrieve(
            self._collection, ids=[_point_id(workspace_id, passage_id)], with_payload=True
        )
        if not found or (found[0].payload or {}).get("workspace_id") != str(workspace_id):
            return None
        return _stored(found[0].payload or {})


@lru_cache
def get_passages() -> QdrantPassages:
    return QdrantPassages(get_qdrant(), server=bool(get_settings().qdrant_url))
