"""The pgvector store against a real database (after `alembic upgrade head`).

Skipped when Postgres is unreachable, so a green run without it has not exercised this file.
Each test works in a fresh workspace id and removes its rows; other data is never touched.
"""

import uuid
from collections.abc import AsyncIterator

import pytest
from fakes import _vector
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from core.config import get_settings
from documents.chunking import Passage
from store.models import LibraryEntryIndex, LibraryPassage, ResultCacheRow
from store.ports import EntryMeta
from store.postgres import PgEmbeddingCache, PgLibraryIndex, PgResultCache


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1 FROM library_passage LIMIT 1"))
    except Exception:
        await engine.dispose()
        pytest.skip("Postgres with the AI service schema is not reachable")
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
        await s.rollback()
    await engine.dispose()


def _passage(pid: str, text_: str, section: str = "proposal") -> Passage:
    return Passage(pid, section, 1, 0, text_)


async def _cleanup(s: AsyncSession, ws: uuid.UUID) -> None:
    for model in (LibraryPassage, LibraryEntryIndex, ResultCacheRow):
        await s.execute(delete(model).where(model.workspace_id == ws))
    await s.commit()


async def test_index_search_and_isolation(session: AsyncSession) -> None:
    ws, other = uuid.uuid4(), uuid.uuid4()
    index = PgLibraryIndex(session)
    try:
        meta = EntryMeta("e1", "Acme", "Fleet", "Logistics", "won", 2025)
        texts = ["offline driver mobile app", "employee leave portal"]
        await index.replace_entry(
            ws, meta, "h1", [_passage("p1", texts[0]), _passage("p2", texts[1])], [_vector(t) for t in texts]
        )
        await index.replace_entry(other, meta, "h1", [_passage("p1", texts[0])], [_vector(texts[0])])

        hits = await index.search(
            ws, _vector("driver mobile app"), entry_ids=None, sections=["proposal"], limit=5
        )
        assert [h.passage.id for h in hits] == ["p1", "p2"]
        assert hits[0].similarity > hits[1].similarity
        assert await index.search(ws, _vector("x"), entry_ids=["nope"], sections=None, limit=5) == []
        assert (await index.entries(ws, None)).keys() == {"e1"}
        assert await index.content_hashes(ws, ["e1"]) == {"e1": "h1"}
        assert (await index.passage(ws, "p2")) is not None
        assert (await index.passage(uuid.uuid4(), "p2")) is None

        # Re-indexing replaces the passages instead of adding to them.
        await index.replace_entry(ws, meta, "h2", [_passage("p3", "new text")], [_vector("new text")])
        assert [
            h.passage.id
            for h in await index.search(ws, _vector("new"), entry_ids=None, sections=None, limit=9)
        ] == ["p3"]
        assert await index.delete_entry(ws, "e1") is True
        assert await index.delete_entry(ws, "e1") is False
        assert (await index.entries(other, None)).keys() == {"e1"}  # the other workspace is untouched
    finally:
        await session.rollback()
        await _cleanup(session, ws)
        await _cleanup(session, other)


async def test_caches_round_trip(session: AsyncSession) -> None:
    ws = uuid.uuid4()
    embeddings, results = PgEmbeddingCache(session), PgResultCache(session)
    key = uuid.uuid4().hex
    try:
        await embeddings.put_many({key: _vector("hello")})
        await embeddings.put_many({key: _vector("ignored on conflict")})
        assert (await embeddings.get_many([key, "missing"])).keys() == {key}
        assert await embeddings.get_many([]) == {}

        await results.put(ws, "k", "analyse", {"a": 1})
        await results.put(ws, "k", "analyse", {"a": 2})
        assert await results.get(ws, "k") == {"a": 1}
        assert await results.get(uuid.uuid4(), "k") is None
    finally:
        await session.rollback()
        await _cleanup(session, ws)
