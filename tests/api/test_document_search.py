"""Document search for the report builder: standalone documents, filters, passages, tenancy, stable order."""

import json
import uuid
from typing import Any

from conftest import Env

DOCS = {
    "d1": (
        "casestudy",
        "Fleet tracking story",
        "Logistics",
        2024,
        b"We delivered live GPS tracking for a trucking fleet.",
    ),
    "d2": ("other", "HR portal notes", "Other", 2022, b"An employee leave portal with approvals."),
}


async def _doc(env: Env, entry_id: str, ws: uuid.UUID | None = None, **meta: Any) -> Any:
    kind, title, industry, year, text = DOCS[entry_id]
    body = {"kind": kind, "client": "Acme", "title": title, "industry": industry, "year": year} | meta
    return await env.client.put(
        f"/v1/library/documents/{entry_id}",
        data={"meta": json.dumps(body)},
        files={"file": (f"{entry_id}.txt", text, "text/plain")},
        headers=env.auth(ws),
    )


async def _bid(env: Env, entry_id: str, outcome: str, text: bytes) -> None:
    meta = {
        "client": "Beta",
        "title": f"Bid {entry_id}",
        "industry": "Logistics",
        "outcome": outcome,
        "year": 2025,
    }
    r = await env.client.put(
        f"/v1/library/entries/{entry_id}",
        data={"meta": json.dumps(meta)},
        files={"proposal": ("proposal.txt", text, "text/plain")},
        headers=env.auth(),
    )
    assert r.status_code == 200, r.text


async def _search(env: Env, **body: Any) -> dict[str, Any]:
    r = await env.client.post("/v1/library/search", json={"threshold": 0.3} | body, headers=env.auth())
    assert r.status_code == 200, r.text
    return r.json()


async def test_a_document_is_indexed_once_and_found_with_its_passage(env: Env) -> None:
    first = (await _doc(env, "d1")).json()
    assert first["unchanged"] is False and first["passages"] >= 2
    embedded = len(env.embedder.embedded)
    assert (await _doc(env, "d1")).json()["unchanged"] is True and len(env.embedder.embedded) == embedded

    found = await _search(env, query="live GPS tracking for a fleet")
    [hit] = found["results"]
    assert (hit["entry_id"], hit["kind"], hit["title"], hit["year"]) == (
        "d1",
        "casestudy",
        "Fleet tracking story",
        2024,
    )
    assert (
        hit["passages"][0]["section"] in {"document", "summary"}
        and "GPS" in hit["passages"][0]["text"] + hit["title"]
    )
    assert found["candidates"] == 1


async def test_filters_narrow_the_documents_before_ranking(env: Env) -> None:
    await _doc(env, "d1")
    await _doc(env, "d2")
    await _bid(env, "b1", "lost", b"We proposed live GPS tracking for a logistics fleet.")
    await _bid(env, "b2", "won", b"We proposed live GPS tracking for another logistics fleet.")

    everything = await _search(env, query="live GPS tracking fleet")
    assert {r["entry_id"] for r in everything["results"]} >= {"d1", "b1", "b2"}
    assert everything["candidates"] == 4

    lost = await _search(env, query="live GPS tracking fleet", outcomes=["lost"])
    assert [r["entry_id"] for r in lost["results"]] == ["b1"] and lost["candidates"] == 1
    studies = await _search(env, query="live GPS tracking fleet", kinds=["casestudy"])
    assert [r["entry_id"] for r in studies["results"]] == ["d1"]
    by_year = await _search(env, query="employee leave portal", year_to=2023)
    assert [r["entry_id"] for r in by_year["results"]] == ["d2"]
    industry = await _search(
        env, query="live GPS tracking fleet", industries=["logistics"], kinds=["proposal"]
    )
    assert {r["entry_id"] for r in industry["results"]} == {"b1", "b2"}


async def test_threshold_limit_and_stable_order(env: Env) -> None:
    await _doc(env, "d1")
    await _doc(env, "d2")
    strict = await _search(env, query="live GPS tracking for a trucking fleet", threshold=0.99)
    assert strict["results"] == []
    one = await _search(env, query="leave portal tracking fleet", limit=1)
    assert len(one["results"]) == 1
    body = {"query": "leave portal tracking fleet", "threshold": 0.1}
    first = (await env.client.post("/v1/library/search", json=body, headers=env.auth())).json()
    assert first == (await env.client.post("/v1/library/search", json=body, headers=env.auth())).json()
    scores = [r["similarity"] for r in first["results"]]
    assert scores == sorted(scores, reverse=True)


async def test_search_never_crosses_workspaces(env: Env) -> None:
    await _doc(env, "d1", ws=uuid.uuid4())
    found = await _search(env, query="live GPS tracking for a trucking fleet")
    assert found["results"] == [] and found["candidates"] == 0


async def test_documents_are_not_projects_to_compare_with(env: Env) -> None:
    await _doc(env, "d1")
    await _bid(env, "b1", "won", b"We delivered live GPS tracking for a trucking fleet.")
    body = {
        "requirements": [{"key": "x", "text": "live GPS tracking"}],
        "rfp_summary": "live GPS tracking",
        "threshold": 0.3,
    }
    data = (await env.client.post("/v1/library/similar", json=body, headers=env.auth())).json()
    assert [m["entry_id"] for m in data["requirements"][0]["matches"]] == ["b1"]
    assert [p["entry_id"] for p in data["projects"]] == ["b1"]


async def test_a_document_is_removed_like_an_entry_and_its_file_must_be_readable(env: Env) -> None:
    await _doc(env, "d1")
    assert (await env.client.delete("/v1/library/entries/d1", headers=env.auth())).status_code == 204
    assert (await _search(env, query="live GPS tracking"))["results"] == []
    meta = {"kind": "other", "title": "t", "year": 2024}
    r = await env.client.put(
        "/v1/library/documents/d9",
        data={"meta": json.dumps(meta)},
        files={"file": ("photo.png", b"\x89PNG", "image/png")},
        headers=env.auth(),
    )
    assert r.status_code == 422 and r.json()["rule"] == "file_type"


async def test_search_body_is_validated(env: Env) -> None:
    for body in (
        {"query": ""},
        {"query": "x", "year_from": 2025, "year_to": 2020},
        {"query": "x", "kinds": ["nope"]},
        {"query": "x", "industries": [""]},
        {"query": "x", "industries": ["x" * 61]},
    ):
        r = await env.client.post("/v1/library/search", json=body, headers=env.auth())
        assert r.status_code == 422
    assert (await env.client.post("/v1/library/search", json={"query": "x"})).status_code == 401
