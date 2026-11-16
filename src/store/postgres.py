"""Postgres implementations of the storage interfaces; passage vectors live in Qdrant (store/vector_db.py)."""

import uuid
from typing import Any

from fastapi import Depends
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.db import db, sessionmaker
from documents.chunking import Passage
from store.models import EmbeddingCacheRow, LibraryEntryIndex, ResultCacheRow
from store.ports import EntryMeta, Hit, StoredPassage
from store.vector_db import QdrantPassages, get_passages


def _meta(row: LibraryEntryIndex) -> EntryMeta:
    return EntryMeta(row.entry_id, row.client, row.title, row.industry, row.outcome, row.year)


class PgLibraryIndex:
    """Entry metadata in Postgres, passages and vectors in Qdrant."""

    def __init__(self, session: AsyncSession, passages: QdrantPassages) -> None:
        self._s = session
        self._passages = passages

    async def replace_entry(
        self,
        workspace_id: uuid.UUID,
        meta: EntryMeta,
        content_hash: str,
        passages: list[Passage],
        vectors: list[list[float]],
    ) -> None:
        await self.delete_entry(workspace_id, meta.entry_id)
        self._s.add(
            LibraryEntryIndex(
                workspace_id=workspace_id,
                entry_id=meta.entry_id,
                client=meta.client,
                title=meta.title,
                industry=meta.industry,
                outcome=meta.outcome,
                year=meta.year,
                content_hash=content_hash,
            )
        )
        await self._s.flush()
        await self._passages.add(workspace_id, meta.entry_id, passages, vectors)

    async def delete_entry(self, workspace_id: uuid.UUID, entry_id: str) -> bool:
        await self._passages.delete(workspace_id, entry_id)
        result = await self._s.execute(
            delete(LibraryEntryIndex).where(
                LibraryEntryIndex.workspace_id == workspace_id, LibraryEntryIndex.entry_id == entry_id
            )
        )
        return bool(getattr(result, "rowcount", 0))

    async def entries(self, workspace_id: uuid.UUID, entry_ids: list[str] | None) -> dict[str, EntryMeta]:
        q = select(LibraryEntryIndex).where(LibraryEntryIndex.workspace_id == workspace_id)
        if entry_ids is not None:
            q = q.where(LibraryEntryIndex.entry_id.in_(entry_ids))
        return {r.entry_id: _meta(r) for r in (await self._s.execute(q)).scalars()}

    async def content_hashes(self, workspace_id: uuid.UUID, entry_ids: list[str]) -> dict[str, str]:
        q = select(LibraryEntryIndex.entry_id, LibraryEntryIndex.content_hash).where(
            LibraryEntryIndex.workspace_id == workspace_id, LibraryEntryIndex.entry_id.in_(entry_ids)
        )
        return {e: h for e, h in (await self._s.execute(q)).all()}

    async def search(
        self,
        workspace_id: uuid.UUID,
        vector: list[float],
        *,
        entry_ids: list[str] | None,
        sections: list[str] | None,
        limit: int,
    ) -> list[Hit]:
        return await self._passages.search(
            workspace_id, vector, entry_ids=entry_ids, sections=sections, limit=limit
        )

    async def passage(self, workspace_id: uuid.UUID, passage_id: str) -> StoredPassage | None:
        return await self._passages.get(workspace_id, passage_id)


class PgEmbeddingCache:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        rows = await self._s.execute(select(EmbeddingCacheRow).where(EmbeddingCacheRow.key.in_(keys)))
        return {r.key: list(r.embedding) for r in rows.scalars()}

    async def put_many(self, items: dict[str, list[float]]) -> None:
        if items:
            stmt = insert(EmbeddingCacheRow).values([{"key": k, "embedding": v} for k, v in items.items()])
            await self._s.execute(stmt.on_conflict_do_nothing(index_elements=["key"]))


class PgResultCache:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, workspace_id: uuid.UUID, key: str) -> dict[str, Any] | None:
        row = await self._s.get(ResultCacheRow, (workspace_id, key))
        return row.payload if row else None

    async def put(self, workspace_id: uuid.UUID, key: str, kind: str, payload: dict[str, Any]) -> None:
        stmt = insert(ResultCacheRow).values(workspace_id=workspace_id, key=key, kind=kind, payload=payload)
        await self._s.execute(stmt.on_conflict_do_nothing(index_elements=["workspace_id", "key"]))


class PgStandaloneResultCache:
    """The result cache for work outside a request (jobs): each read or write is its own transaction, so
    no transaction stays open while a model is answering."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, workspace_id: uuid.UUID, key: str) -> dict[str, Any] | None:
        async with self._sessions() as s:
            return await PgResultCache(s).get(workspace_id, key)

    async def put(self, workspace_id: uuid.UUID, key: str, kind: str, payload: dict[str, Any]) -> None:
        async with self._sessions() as s, s.begin():
            await PgResultCache(s).put(workspace_id, key, kind, payload)


def pg_standalone_result_cache() -> PgStandaloneResultCache:
    return PgStandaloneResultCache(sessionmaker())


def library_index(
    session: AsyncSession = Depends(db), passages: QdrantPassages = Depends(get_passages)
) -> PgLibraryIndex:
    return PgLibraryIndex(session, passages)


def embedding_cache(session: AsyncSession = Depends(db)) -> PgEmbeddingCache:
    return PgEmbeddingCache(session)


def result_cache(session: AsyncSession = Depends(db)) -> PgResultCache:
    return PgResultCache(session)
