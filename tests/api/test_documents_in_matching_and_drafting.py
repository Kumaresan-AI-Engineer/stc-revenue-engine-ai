"""Case studies and reference documents next to the past bids: found per requirement, and drafted from."""

import io
import json
import uuid
from typing import Any

from conftest import Env
from pptx import Presentation
from test_api import _draft_script, _index

from revenue_engine_ai.llm.ports import AgentCall, Tool

from tests.fakes import approve_all

ISO = b"We hold ISO 27001 certification for our delivery centres. The certificate renews every year."
STORY = b"We delivered an offline driver mobile app with route sync for a trucking fleet."


async def _document(
    env: Env,
    entry_id: str,
    text: bytes,
    *,
    kind: str = "casestudy",
    client: str = "Acme",
    title: str = "Document",
    filename: str = "document.txt",
    ws: uuid.UUID | None = None,
) -> None:
    meta = {"kind": kind, "client": client, "title": title, "industry": "Logistics", "year": 2024}
    r = await env.client.put(
        f"/v1/library/documents/{entry_id}",
        data={"meta": json.dumps(meta)},
        files={"file": (filename, text, "application/octet-stream")},
        headers=env.auth(ws),
    )
    assert r.status_code == 200, r.text


async def _similar(env: Env, **body: Any) -> dict[str, Any]:
    request = {"requirements": [{"key": "x_1", "text": "offline driver mobile app"}], "threshold": 0.3}
    r = await env.client.post("/v1/library/similar", json=request | body, headers=env.auth())
    assert r.status_code == 200, r.text
    return r.json()


async def _draft(env: Env, **body: Any) -> dict[str, Any]:
    request = {"questions": [{"id": "q_sec", "text": "Describe your ISO 27001 certification."}]}
    r = await env.client.post("/v1/proposals/draft-answers", json=request | body, headers=env.auth())
    assert r.status_code == 200, r.text
    return r.json()["answers"][0]


# ---- matching --------------------------------------------------------------------------------


async def test_similar_lists_documents_only_when_asked_and_apart_from_past_bids(env: Env) -> None:
    await _index(env, "e1", "Fleet tracking", STORY)
    await _document(env, "d1", STORY, title="Driver app story", client="Acme")

    plain = (await _similar(env))["requirements"][0]
    assert [m["entry_id"] for m in plain["matches"]] == ["e1"] and plain["documents"] == []

    asked = (await _similar(env, include_documents=True))["requirements"][0]
    assert [m["entry_id"] for m in asked["matches"]] == ["e1"]  # a document is never a comparable bid
    (found,) = asked["documents"]
    assert (found["entry_id"], found["kind"], found["client"], found["title"], found["year"]) == (
        "d1",
        "casestudy",
        "Acme",
        "Driver app story",
        2024,
    )
    assert "offline driver mobile app" in found["text"] and found["page"] == 1 and found["passage_id"]


async def test_similar_documents_ignore_the_bid_scope_stay_in_the_workspace_and_keep_their_order(
    env: Env,
) -> None:
    await _index(env, "e1", "Fleet tracking", STORY)
    await _document(env, "d1", STORY, kind="other", title="Capability deck")
    await _document(env, "d2", STORY, title="Driver app story")
    await _document(env, "d3", STORY, title="Another workspace", ws=uuid.uuid4())

    result = await _similar(env, include_documents=True, entry_ids=["e1"], top_k=5)
    documents = result["requirements"][0]["documents"]
    assert [(d["entry_id"], d["kind"]) for d in documents] == [("d1", "other"), ("d2", "casestudy")]
    assert result == await _similar(env, include_documents=True, entry_ids=["e1"], top_k=5)


async def test_similar_documents_below_the_threshold_are_left_out(env: Env) -> None:
    await _document(env, "d1", b"An employee leave portal with approvals.", title="HR portal")
    assert (await _similar(env, include_documents=True))["requirements"][0]["documents"] == []


async def test_a_powerpoint_is_indexed_slide_by_slide_and_found(env: Env) -> None:
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "We deliver an offline driver mobile app with route sync"
    out = io.BytesIO()
    deck.save(out)
    await _document(env, "d1", out.getvalue(), kind="other", title="Capability deck", filename="deck.pptx")

    (found,) = (await _similar(env, include_documents=True))["requirements"][0]["documents"]
    assert found["entry_id"] == "d1" and found["page"] == 1 and "offline driver mobile app" in found["text"]


# ---- drafting --------------------------------------------------------------------------------


async def test_an_answer_is_drafted_from_a_reference_document_and_cites_it(env: Env) -> None:
    await _document(env, "d1", ISO, kind="other", client="", title="ISO certificate")
    sources: list[list[str]] = []

    async def script(call: AgentCall[Any], tools: dict[str, Tool]) -> dict[str, Any]:
        hits = json.loads(await tools["search_library"].run({"query": "ISO 27001 certification"}))
        sources.append([h["source"] for h in hits])
        return await _draft_script(call, tools)

    env.llm.on_tools("drafting", script)
    env.llm.on("review", approve_all)

    answer = await _draft(env, include_documents=True)
    assert answer["status"] == "drafted" and answer["no_confident_source"] is False
    (citation,) = answer["citations"]
    assert (citation["source_id"], citation["kind"]) == ("d1", "other")
    assert citation["title"] == "ISO certificate (2024)"  # no client to put in front of it
    assert "ISO 27001" in citation["quote"] and citation["page"] == 1
    assert sources == [["ISO certificate (2024, reference document)"]]  # the agent is told what kind it reads


async def test_documents_are_not_a_source_unless_the_request_includes_them(env: Env) -> None:
    await _index(env, "e1", "Fleet tracking", b"We run an EU data centre. We are ISO 27001 certified.")
    await _document(env, "d1", ISO, kind="other", title="ISO certificate")
    seen: list[str] = []

    async def script(call: AgentCall[Any], tools: dict[str, Tool]) -> dict[str, Any]:
        seen.extend(h["source"] for h in json.loads(await tools["search_library"].run({"query": "ISO"})))
        return await _draft_script(call, tools)

    env.llm.on_tools("drafting", script)
    env.llm.on("review", approve_all)

    answer = await _draft(env, selected_entry_ids=["e1"])
    assert seen == ["Client e1 — Fleet tracking (2025, won)"]
    assert [c["kind"] for c in answer["citations"]] == ["proposal"]


async def test_proposals_and_documents_are_searched_together(env: Env) -> None:
    await _index(env, "e1", "Fleet tracking", b"We run an EU data centre. We are ISO 27001 certified.")
    await _document(env, "d1", ISO, kind="other", client="", title="ISO certificate")
    seen: list[str] = []

    async def script(call: AgentCall[Any], tools: dict[str, Tool]) -> dict[str, Any]:
        seen.extend(h["source"] for h in json.loads(await tools["search_library"].run({"query": "ISO"})))
        return await _draft_script(call, tools)

    env.llm.on_tools("drafting", script)
    env.llm.on("review", approve_all)

    await _draft(env, selected_entry_ids=["e1"], include_documents=True)
    assert sorted(seen) == [
        "Client e1 — Fleet tracking (2025, won)",
        "ISO certificate (2024, reference document)",
    ]


async def test_another_workspaces_documents_are_never_drafted_from(env: Env) -> None:
    await _document(env, "d1", ISO, kind="other", title="ISO certificate", ws=uuid.uuid4())
    answer = await _draft(env, include_documents=True)
    assert answer["status"] == "needs_person" and answer["citations"] == []
    assert "No case studies or reference documents" in answer["reason"]
    assert env.llm.count("drafting") == 0


async def test_a_request_needs_a_proposal_or_the_documents(env: Env) -> None:
    body = {"questions": [{"id": "q", "text": "Describe your ISO 27001 certification."}]}
    r = await env.client.post("/v1/proposals/draft-answers", json=body, headers=env.auth())
    assert r.status_code == 422
