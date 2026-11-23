"""The Qdrant passage store, run against an in-memory Qdrant (same engine as the embedded mode)."""

import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from revenue_engine_ai.documents.chunking import Passage
from revenue_engine_ai.store.vector_db import QdrantPassages

from tests.fakes import _vector


@pytest.fixture
def passages() -> QdrantPassages:
    return QdrantPassages(AsyncQdrantClient(location=":memory:"))


def _p(pid: str, text: str, section: str = "proposal", page: int = 1) -> Passage:
    return Passage(pid, section, page, 0, text)


async def test_search_is_ranked_filtered_and_kept_inside_the_workspace(passages: QdrantPassages) -> None:
    ws, other = uuid.uuid4(), uuid.uuid4()
    texts = {"p1": "offline driver mobile app", "p2": "employee leave portal", "p3": "driver app for rfp"}
    await passages.add(
        ws, "e1", [_p("p1", texts["p1"]), _p("p2", texts["p2"])], [_vector(texts["p1"]), _vector(texts["p2"])]
    )
    await passages.add(ws, "e2", [_p("p3", texts["p3"], section="rfp")], [_vector(texts["p3"])])
    await passages.add(other, "e1", [_p("p1", texts["p1"])], [_vector(texts["p1"])])

    hits = await passages.search(
        ws, _vector("driver mobile app"), entry_ids=None, sections=["proposal"], limit=5
    )
    assert [h.passage.id for h in hits] == ["p1", "p2"]
    assert hits[0].similarity > hits[1].similarity
    assert hits[0].passage.entry_id == "e1" and hits[0].passage.text == texts["p1"]

    only_e2 = await passages.search(ws, _vector("driver"), entry_ids=["e2"], sections=None, limit=5)
    assert [h.passage.id for h in only_e2] == ["p3"]
    assert await passages.search(ws, _vector("driver"), entry_ids=[], sections=None, limit=5) == []
    assert (
        await passages.search(uuid.uuid4(), _vector("driver"), entry_ids=None, sections=None, limit=5) == []
    )


async def test_equal_similarity_is_ordered_by_passage_id(passages: QdrantPassages) -> None:
    ws = uuid.uuid4()
    same = _vector("identical text")
    await passages.add(ws, "e1", [_p("b", "identical text"), _p("a", "identical text")], [same, same])
    hits = await passages.search(ws, same, entry_ids=None, sections=None, limit=5)
    assert [h.passage.id for h in hits] == ["a", "b"]


async def test_get_and_delete_only_touch_their_own_workspace(passages: QdrantPassages) -> None:
    ws, other = uuid.uuid4(), uuid.uuid4()
    await passages.add(ws, "e1", [_p("p1", "text", page=4)], [_vector("text")])
    await passages.add(other, "e1", [_p("p1", "text")], [_vector("text")])

    found = await passages.get(ws, "p1")
    assert found is not None and found.page == 4
    assert await passages.get(uuid.uuid4(), "p1") is None

    await passages.delete(ws, "e1")
    assert await passages.get(ws, "p1") is None
    assert await passages.get(other, "p1") is not None
