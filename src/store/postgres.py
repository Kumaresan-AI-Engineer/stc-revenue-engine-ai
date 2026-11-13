"""Postgres + pgvector implementations of the storage interfaces.

Search is an exact cosine scan inside one workspace (no approximate index), so the same query over
the same library always returns the same passages in the same order.
"""

import uuid
from typing import Any

from fastapi import Depends
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import db
from documents.chunking import Passage
from store.models import EmbeddingCacheRow, LibraryEntryIndex, LibraryPassage, ResultCacheRow
from store.ports import EntryMeta, Hit, StoredPassage

SIMILARITY_DECIMALS = 4


def _meta(row: LibraryEntryIndex) -> EntryMeta:
    return EntryMeta(row.entry_id, row.client, row.title, row.industry, row.outcome, row.year)


def _passage(row: LibraryPassage) -> StoredPassage:
    return StoredPassage(row.id, row.entry_id, row.section, row.page, row.text)


class PgLibraryIndex:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

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
        self._s.add_all(
            LibraryPassage(
                workspace_id=workspace_id,
                id=p.id,
                entry_id=meta.entry_id,
                section=p.section,
                page=p.page,
                ordinal=p.ordinal,
                text=p.text,
                embedding=v,
            )
            for p, v in zip(passages, vectors, strict=True)
        )
        await self._s.flush()

    async def delete_entry(self, workspace_id: uuid.UUID, entry_id: str) -> bool:
        await self._s.execute(
            delete(LibraryPassage).where(
                LibraryPassage.workspace_id == workspace_id, LibraryPassage.entry_id == entry_id
            )
        )
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
        distance = LibraryPassage.embedding.cosine_distance(vector)
        q = select(LibraryPassage, distance.label("distance")).where(
            LibraryPassage.workspace_id == workspace_id
        )
        if entry_ids is not None:
            q = q.where(LibraryPassage.entry_id.in_(entry_ids))
        if sections is not None:
            q = q.where(LibraryPassage.section.in_(sections))
        q = q.order_by(distance, LibraryPassage.id).limit(limit)
        return [
            Hit(_passage(row), round(1 - float(dist), SIMILARITY_DECIMALS))
            for row, dist in (await self._s.execute(q)).all()
        ]

    async def passage(self, workspace_id: uuid.UUID, passage_id: str) -> StoredPassage | None:
        row = await self._s.get(LibraryPassage, (workspace_id, passage_id))
        return _passage(row) if row else None


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


def library_index(session: AsyncSession = Depends(db)) -> PgLibraryIndex:
    return PgLibraryIndex(session)


def embedding_cache(session: AsyncSession = Depends(db)) -> PgEmbeddingCache:
    return PgEmbeddingCache(session)


def result_cache(session: AsyncSession = Depends(db)) -> PgResultCache:
    return PgResultCache(session)
