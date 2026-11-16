"""Contract tests through HTTP: status codes, body shapes and the agent pipelines, with fake providers."""

import json
import uuid
from typing import Any

from conftest import Env
from fakes import approve_all

from agents.drafting import Claim
from llm.ports import AgentCall, Tool

# As the backend sends it: {key, label, commodity, kw}.
CATALOGUE = [
    {"key": "tracking", "label": "Real-time tracking", "kw": ["gps", "tracking"], "commodity": False},
    {"key": "driverapp", "label": "Driver mobile app", "kw": ["driver app"], "commodity": False},
    {"key": "auth", "label": "Login", "kw": ["sign in"], "commodity": True},
]
RFP = (
    b"Acme Logistics invites proposals. We need real-time GPS tracking of 400 trucks.\n"
    b"Drivers must use a mobile app that works offline.\n"
    b"Please describe your ISO 27001 certification. Responses due 2026-12-01."
)


def _form(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "summary": "Acme wants truck tracking.",
        "features": [],
        "custom_requirements": [],
        "questions": [],
        "required_documents": [],
        "unclear_points": [],
        "facts": [],
    }
    return base | kw


def _problem(body: dict[str, Any], status: int) -> None:
    assert body["status"] == status
    assert {"type", "title", "detail"} <= body.keys()


# ---- Auth and contract -----------------------------------------------------------------


async def test_requests_without_a_valid_token_are_refused(env: Env) -> None:
    r = await env.client.post("/v1/library/similar", json={})
    assert r.status_code == 401 and r.headers["content-type"] == "application/problem+json"
    _problem(r.json(), 401)
    r = await env.client.post("/v1/library/similar", json={}, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


async def test_invalid_body_is_a_422_problem(env: Env) -> None:
    r = await env.client.post("/v1/proposals/draft-answers", json={"questions": []}, headers=env.auth())
    assert r.status_code == 422
    _problem(r.json(), 422)
    assert r.json()["errors"]


async def test_health_and_request_id(env: Env) -> None:
    r = await env.client.get("/healthz", headers={"X-Request-Id": "abc"})
    assert r.json() == {"status": "ok"} and r.headers["X-Request-Id"] == "abc"


# ---- Analyse ------------------------------------------------------------------------------


async def _analyse(env: Env, data: bytes = RFP, name: str = "rfp.txt") -> Any:
    return await env.client.post(
        "/v1/rfp/analyse",
        files={"file": (name, data, "text/plain")},
        data={"request": json.dumps({"catalogue": CATALOGUE})},
        headers=env.auth(),
    )


async def test_analyse_returns_grounded_requirements_and_caches(env: Env) -> None:
    env.llm.on(
        "extraction",
        lambda call: _form(
            features=[
                {
                    "key": "tracking",
                    "quote": "real-time GPS tracking of 400 trucks",
                    "page": 1,
                    "confidence": 1.4,
                },
                {"key": "auth", "quote": "Acme Logistics invites proposals", "page": 1, "confidence": 0.5},
            ],
            custom_requirements=[{"label": "Offline driver app", "quote": "works offline", "page": 1}],
            questions=[
                {
                    "text": "Describe your ISO 27001 certification.",
                    "quote": "describe your ISO 27001",
                    "page": 1,
                }
            ],
            facts=[
                {"name": "due_date", "value": "2026-12-01", "quote": "Responses due 2026-12-01", "page": 1}
            ],
        ),
    )
    env.llm.on("review", approve_all)
    r = await _analyse(env)
    assert r.status_code == 200, r.text
    body = r.json()
    # Core vs standard by the catalogue's commodity flag; confidence clamped to 0..1.
    assert body["core"] == [
        {"key": "tracking", "confidence": 1.0, "evidence": "real-time GPS tracking of 400 trucks", "page": 1}
    ]
    assert body["standard"] == ["auth"] and body["standard_evidence"][0]["key"] == "auth"
    assert body["unmatched"] == [{"label": "Offline driver app", "evidence": "works offline", "page": 1}]
    assert body["questions"] == [
        {"text": "Describe your ISO 27001 certification.", "page": 1, "evidence": "describe your ISO 27001"}
    ]
    assert (body["due_date"], body["start_date"], body["budget_hint"]) == ("2026-12-01", None, None)
    assert body["rejected"] == [] and body["ocr"] is False and body["pages"] == 1
    assert (
        body["meta"]["cached"] is False and body["meta"]["prompt_versions"]["extraction"] == "extraction-v2"
    )
    # The catalogue is in the instructions, so the long fixed prefix is cacheable.
    first = next(c for c in env.llm.calls if c.agent == "extraction")
    assert "- tracking: Real-time tracking" in first.instructions

    again = await _analyse(env)
    assert again.json()["meta"]["cached"] is True
    assert {k: v for k, v in again.json().items() if k != "meta"} == {
        k: v for k, v in body.items() if k != "meta"
    }
    assert env.llm.count("extraction") == 1


async def test_analyse_retries_once_with_notes_then_drops_what_still_fails(env: Env) -> None:
    def extract(call: AgentCall[Any]) -> dict[str, Any]:
        invented = {"key": "driverapp", "quote": "drivers fly drones", "page": 1, "confidence": 0.9}
        good = {"key": "tracking", "quote": "real-time GPS tracking", "page": 1, "confidence": 0.9}
        return _form(features=[good, invented])

    def review_call(call: AgentCall[Any]) -> dict[str, Any]:
        items = json.loads(call.input)
        return {"verdicts": [{"id": i["id"], "supported": True, "reason": "ok"} for i in items]}

    env.llm.on("extraction", extract)
    env.llm.on("review", review_call)
    body = (await _analyse(env)).json()
    assert env.llm.count("extraction") == 2
    retry = [c for c in env.llm.calls if c.agent == "extraction"][1]
    assert "feature 'driverapp': its quote is not in the document" in retry.input
    assert [f["key"] for f in body["core"]] == ["tracking"]
    assert body["rejected"] == [{"item": "feature 'driverapp'", "reason": "its quote is not in the document"}]


async def test_analyse_drops_features_the_reviewer_rejects(env: Env) -> None:
    env.llm.on(
        "extraction",
        lambda call: _form(
            features=[{"key": "tracking", "quote": "GPS tracking", "page": 1, "confidence": 0.8}]
        ),
    )
    env.llm.on(
        "review",
        lambda call: {
            "verdicts": [{"id": "feature:tracking", "supported": False, "reason": "Quote is too vague."}]
        },
    )
    body = (await _analyse(env)).json()
    assert body["core"] == []
    assert body["rejected"] == [{"item": "feature:tracking", "reason": "Quote is too vague."}]
    # A verdict is not fixed by extracting again, so it costs no second pass.
    assert env.llm.count("extraction") == 1 and env.llm.count("review") == 1


async def test_analyse_refuses_unsupported_files_and_bad_catalogue(env: Env) -> None:
    r = await _analyse(env, b"x", "deck.pptx")
    assert r.status_code == 422 and r.json()["rule"] == "file_type"
    r = await env.client.post(
        "/v1/rfp/analyse",
        files={"file": ("rfp.txt", RFP, "text/plain")},
        data={"request": json.dumps({"catalogue": []})},
        headers=env.auth(),
    )
    assert r.status_code == 422 and r.json()["errors"]


# ---- Library: index + semantic search -----------------------------------------------------


async def _index(
    env: Env, entry_id: str, title: str, text: bytes, outcome: str = "won", ws: uuid.UUID | None = None
) -> Any:
    meta = {
        "client": f"Client {entry_id}",
        "title": title,
        "industry": "Logistics",
        "outcome": outcome,
        "year": 2025,
    }
    return await env.client.put(
        f"/v1/library/entries/{entry_id}",
        data={"meta": json.dumps(meta)},
        files={"proposal": ("proposal.txt", text, "text/plain")},
        headers=env.auth(ws),
    )


async def test_index_is_idempotent_and_redacts_prices(env: Env) -> None:
    first = (
        await _index(env, "e1", "Fleet tracking", b"We built GPS tracking for 900 trucks for EUR 400000.")
    ).json()
    assert first["passages"] == 2 and first["unchanged"] is False
    embedded = len(env.embedder.embedded)
    second = (
        await _index(env, "e1", "Fleet tracking", b"We built GPS tracking for 900 trucks for EUR 400000.")
    ).json()
    assert second["unchanged"] is True and len(env.embedder.embedded) == embedded
    assert all("400000" not in p.text for p, _ in env.index.passages.values())


async def test_similar_finds_custom_requirements_per_workspace_in_fixed_order(env: Env) -> None:
    await _index(env, "e1", "Fleet tracking", b"We delivered an offline driver mobile app with route sync.")
    await _index(env, "e2", "HR portal", b"We delivered an employee leave portal with approvals.")
    other_ws = uuid.uuid4()
    await _index(env, "e3", "Driver app", b"We delivered an offline driver mobile app.", ws=other_ws)

    body = {
        "requirements": [{"key": "x_1", "text": "offline driver mobile app"}],
        "rfp_summary": "Fleet tracking",
        "threshold": 0.3,
    }
    r = await env.client.post("/v1/library/similar", json=body, headers=env.auth())
    assert r.status_code == 200, r.text
    data = r.json()
    assert [m["entry_id"] for m in data["requirements"][0]["matches"]] == ["e1"]  # e3 is another workspace
    assert data["projects"][0]["entry_id"] == "e1"
    assert data == (await env.client.post("/v1/library/similar", json=body, headers=env.auth())).json()


async def test_delete_entry(env: Env) -> None:
    await _index(env, "e1", "Fleet", b"Tracking.")
    assert (await env.client.delete("/v1/library/entries/e1", headers=env.auth())).status_code == 204
    r = await env.client.delete("/v1/library/entries/e1", headers=env.auth())
    assert r.status_code == 404
    _problem(r.json(), 404)


async def test_index_needs_content(env: Env) -> None:
    meta = {"client": "c", "title": "t", "industry": "i", "outcome": "won", "year": 2025}
    r = await env.client.put("/v1/library/entries/e1", data={"meta": json.dumps(meta)}, headers=env.auth())
    assert r.status_code == 422 and r.json()["rule"] == "content"


# ---- Draft answers ------------------------------------------------------------------------


async def _draft_script(call: AgentCall[Any], tools: dict[str, Tool]) -> dict[str, Any]:
    """Behaves like the model: search, then answer only from what came back."""
    hits = json.loads(await tools["search_library"].run({"query": "ISO 27001 certification"}))
    if not hits:
        return {"answer": "", "claims": [], "enough_material": False}
    top = hits[0]
    quote = top["text"].split(".")[0]
    claim = Claim(statement=quote, passage_id=top["passage_id"], quote=quote)
    return {"answer": f"{quote}.", "claims": [claim.model_dump()], "enough_material": True}


async def test_draft_answers_grounded_skipped_and_cached(env: Env) -> None:
    await _index(env, "e1", "Secure fleet", b"We are ISO 27001 certified since 2019. We run EU data centres.")
    env.llm.on_tools("drafting", _draft_script)
    env.llm.on("review", approve_all)
    body = {
        "questions": [
            {"id": "q_sec", "text": "Describe your ISO 27001 certification."},
            {"id": "q_price", "text": "What is your day rate for a senior engineer?"},
            {"id": "q_gov", "text": "Describe governance.", "draftable": False},
        ],
        "selected_entry_ids": ["e1"],
    }
    r = await env.client.post("/v1/proposals/draft-answers", json=body, headers=env.auth())
    assert r.status_code == 200, r.text
    answers = {a["id"]: a for a in r.json()["answers"]}
    assert answers["q_sec"]["status"] == "drafted"
    claim = answers["q_sec"]["claims"][0]
    assert claim["entry_id"] == "e1" and "ISO 27001" in claim["quote"]
    assert answers["q_price"]["status"] == "skipped_pricing"
    assert answers["q_gov"]["status"] == "skipped"
    assert env.llm.count("drafting") == 1

    again = (await env.client.post("/v1/proposals/draft-answers", json=body, headers=env.auth())).json()
    assert again["meta"]["cached"] is True and env.llm.count("drafting") == 1


async def test_draft_without_material_needs_a_person(env: Env) -> None:
    await _index(env, "e1", "HR portal", b"Leave requests.")

    async def nothing(call: AgentCall[Any], tools: dict[str, Tool]) -> dict[str, Any]:
        return {"answer": "", "claims": [], "enough_material": False}

    env.llm.on_tools("drafting", nothing)
    body = {"questions": [{"id": "q", "text": "Describe your SOC 2 report."}], "selected_entry_ids": ["e1"]}
    answer = (await env.client.post("/v1/proposals/draft-answers", json=body, headers=env.auth())).json()[
        "answers"
    ][0]
    assert answer["status"] == "needs_person" and answer["answer"] == ""


async def test_draft_that_fails_review_twice_needs_a_person(env: Env) -> None:
    await _index(env, "e1", "Secure fleet", b"We are ISO 27001 certified since 2019.")
    env.llm.on_tools("drafting", _draft_script)
    env.llm.on(
        "review", lambda call: {"verdicts": [{"id": "answer", "supported": False, "reason": "Overstated."}]}
    )
    body = {
        "questions": [{"id": "q", "text": "Describe your ISO 27001 certification."}],
        "selected_entry_ids": ["e1"],
    }
    answer = (await env.client.post("/v1/proposals/draft-answers", json=body, headers=env.auth())).json()[
        "answers"
    ][0]
    assert answer["status"] == "needs_person" and "Overstated." in answer["reason"]
    assert env.llm.count("drafting") == 2
    retry = [c for c in env.llm.calls if c.agent == "drafting"][1]
    assert "Reviewer: Overstated." in retry.input


async def test_draft_tools_cannot_read_unselected_or_other_workspace_passages(env: Env) -> None:
    await _index(env, "e1", "Selected", b"Selected proposal text.")
    await _index(env, "e2", "Not selected", b"Secret other proposal text.")
    other_id = next(pid for (_, pid), (p, _) in env.index.passages.items() if p.entry_id == "e2")
    seen: dict[str, str] = {}

    async def probe(call: AgentCall[Any], tools: dict[str, Tool]) -> dict[str, Any]:
        seen["other"] = await tools["get_passage"].run({"passage_id": other_id})
        seen["search"] = await tools["search_library"].run({"query": "secret other proposal"})
        return {"answer": "", "claims": [], "enough_material": False}

    env.llm.on_tools("drafting", probe)
    body = {"questions": [{"id": "q", "text": "Anything?"}], "selected_entry_ids": ["e1"]}
    await env.client.post("/v1/proposals/draft-answers", json=body, headers=env.auth())
    assert seen["other"] == "No such passage in the selected proposals."
    assert all(h["proposal"].startswith("Client e1") for h in json.loads(seen["search"]))


async def test_draft_refuses_unindexed_selection(env: Env) -> None:
    body = {"questions": [{"id": "q", "text": "Q?"}], "selected_entry_ids": ["missing"]}
    r = await env.client.post("/v1/proposals/draft-answers", json=body, headers=env.auth())
    assert r.status_code == 422 and r.json()["rule"] == "library_not_indexed"


# ---- Client queries -----------------------------------------------------------------------

QUERY_BODY = {
    "requirements": [{"key": "x_1", "label": "Dispatch screen", "covered": False}],
    "unclear_points": [{"id": "u1", "text": "Fleet size not stated"}],
    "engagement_model": "Fixed cost",
    "existing_questions": [],
}


def _question(about: str, text: str, assumption: str = "Up to 500 vehicles.") -> dict[str, str]:
    return {
        "category": "Scope",
        "about": about,
        "question": text,
        "why": "Changes effort.",
        "sample_answer": "About 300.",
        "assumption": assumption,
    }


async def test_client_queries_checked_reviewed_and_cached(env: Env) -> None:
    env.llm.on(
        "clarification",
        lambda call: {
            "questions": [_question("u1", "How many vehicles?"), _question("x_1", "Who uses dispatch?")]
        },
    )
    env.llm.on("review", approve_all)
    r = await env.client.post("/v1/proposals/client-queries", json=QUERY_BODY, headers=env.auth())
    assert r.status_code == 200, r.text
    assert [q["about"] for q in r.json()["questions"]] == ["u1", "x_1"]
    again = (
        await env.client.post("/v1/proposals/client-queries", json=QUERY_BODY, headers=env.auth())
    ).json()
    assert again["meta"]["cached"] is True and env.llm.count("clarification") == 1


async def test_client_queries_retry_keeps_good_questions_and_reports_dropped(env: Env) -> None:
    answers = iter(
        [
            {
                "questions": [
                    _question("u1", "How many vehicles?"),
                    _question("u1", "Budget €50k ok?", "€50k"),
                ]
            },
            {"questions": [_question("x_1", "Who uses dispatch?"), _question("zz", "Unknown?")]},
        ]
    )
    env.llm.on("clarification", lambda call: next(answers))
    env.llm.on("review", approve_all)
    body = (await env.client.post("/v1/proposals/client-queries", json=QUERY_BODY, headers=env.auth())).json()
    assert [q["question"] for q in body["questions"]] == ["How many vehicles?", "Who uses dispatch?"]
    assert body["dropped"] == ["'Unknown?' is about 'zz', which is not a given id."]
