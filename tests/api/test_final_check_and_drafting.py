"""Contract tests for the proposal builder additions: final proposal check (#4), draft answers' citations,
confidence and sources (#2), and Suggest with AI's limit and repeats (#3)."""

import json
from typing import Any

from conftest import Env
from fakes import approve_all
from test_api import QUERY_BODY, _draft_script, _index, _question

from llm.ports import AgentCall, Tool

PROPOSAL = (
    b"Final proposal for Acme Logistics.\n"
    b"We will deliver real-time GPS tracking and an offline driver app.\n"
    b"Delivery takes 26 weeks from kick-off.\n"
    b"The total price is EUR 812,000.\n"
    b"Security: we are ISO 27001 certified.\n"
    b"Enclosed: ISO 27001 certificate."
)
DRAFT = {
    "price": 820000,
    "weeks": 26,
    "core_features": [
        {"key": "tracking", "label": "Real-time tracking"},
        {"key": "driverapp", "label": "Driver mobile app"},
        {"key": "portal", "label": "Customer portal"},
    ],
    "standard_features": ["Login"],
    "responses": [
        {"id": "q_sec", "q": "Describe your ISO 27001 posture.", "response": "We are ISO 27001 certified."},
        {"id": "q_sla", "q": "State your SLA.", "response": "99.9% availability."},
    ],
    "responses_closed": 2,
    "responses_total": 2,
    "attachments": ["ISO 27001 certificate", "Company profile"],
}


def _facts(call: AgentCall[Any]) -> dict[str, Any]:
    return {
        "price": {"value": 812000, "quote": "The total price is EUR 812,000", "page": 1},
        "weeks": {"value": 26, "quote": "Delivery takes 26 weeks", "page": 1},
        "features": [
            {"key": "tracking", "quote": "real-time GPS tracking", "page": 1},
            {"key": "driverapp", "quote": "an offline driver app", "page": 1},
            {"key": "portal", "quote": "a customer portal for shippers", "page": 1},  # not in the file
            {"key": "invented", "quote": "real-time GPS tracking", "page": 1},  # not in the draft
        ],
        "answers": [{"id": "q_sec", "matches": True, "quote": "we are ISO 27001 certified", "page": 1}],
        "attachments": [
            {"name": "iso 27001 certificate", "quote": "Enclosed: ISO 27001 certificate", "page": 1}
        ],
    }


async def _check(env: Env, *, restricted: bool, draft: dict[str, Any] = DRAFT) -> Any:
    return await env.client.post(
        "/v1/proposals/final-check",
        files=[("files", ("Acme - Proposal v3.txt", PROPOSAL, "text/plain"))],
        data={"request": json.dumps({"draft": draft})},
        headers=env.auth(can_see_restricted=restricted),
    )


# ---- #4 Final proposal check --------------------------------------------------------------


async def test_final_check_reports_the_five_areas_from_verified_quotes(env: Env) -> None:
    env.llm.on("final_check", _facts)
    r = await _check(env, restricted=True)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["files"] == ["Acme - Proposal v3.txt"] and body["checked"] is True
    assert (body["price"], body["weeks"], body["draft_price"], body["draft_weeks"]) == (
        812000,
        26,
        820000,
        26,
    )
    assert body["findings"] == [
        {
            "area": "Scope",
            "ok": False,
            "text": "2 of 3 core features found in the file",
            "from": None,
            "to": None,
        },
        {
            "area": "Cost",
            "ok": False,
            "text": "Price in the file differs from the draft",
            "from": 820000,
            "to": 812000,
        },
        {"area": "Timeline", "ok": True, "text": "Same as the draft", "from": 26, "to": 26},
        {
            "area": "Responses",
            "ok": False,
            "text": "1 of 2 RFP answers match the approved responses",
            "from": None,
            "to": None,
        },
        {
            "area": "Attachments",
            "ok": False,
            "text": "1 of 2 attachments are referenced",
            "from": None,
            "to": None,
        },
    ]
    # The price was asked for, since the caller may see it.
    assert "Report the price: yes." in env.llm.calls[0].input
    again = (await _check(env, restricted=True)).json()
    assert again["meta"]["cached"] is True and again["findings"] == body["findings"]


async def test_without_restricted_access_the_price_is_neither_read_nor_returned(env: Env) -> None:
    env.llm.on("final_check", _facts)
    body = (await _check(env, restricted=False)).json()
    assert body["price"] is None and body["draft_price"] is None
    cost = body["findings"][1]
    assert cost == {
        "area": "Cost",
        "ok": True,
        "text": "Not compared: prices are restricted for your role.",
        "from": None,
        "to": None,
    }
    assert "Report the price: no." in env.llm.calls[0].input
    assert not any(ch.isdigit() for f in body["findings"] for ch in f["text"] if f["area"] == "Cost")


async def test_figures_whose_quote_is_not_in_the_file_count_as_not_found(env: Env) -> None:
    def invented(call: AgentCall[Any]) -> dict[str, Any]:
        facts = _facts(call)
        facts["price"] = {"value": 820000, "quote": "The total price is EUR 820,000", "page": 1}
        facts["weeks"] = {"value": 24, "quote": "Delivery takes 26 weeks", "page": 1}  # number not in quote
        return facts

    env.llm.on("final_check", invented)
    body = (await _check(env, restricted=True)).json()
    assert body["price"] is None and body["weeks"] is None
    cost, timeline = body["findings"][1], body["findings"][2]
    assert (cost["ok"], cost["text"], cost["from"], cost["to"]) == (
        False,
        "Price not found in the file",
        820000,
        None,
    )
    assert (timeline["ok"], timeline["text"]) == (False, "Timeline (weeks) not found in the file")


async def test_final_check_takes_one_or_two_files(env: Env) -> None:
    env.llm.on("final_check", _facts)
    files = [
        ("files", ("Technical.txt", PROPOSAL, "text/plain")),
        ("files", ("Commercial.txt", b"Total price EUR 812,000.", "text/plain")),
    ]
    r = await env.client.post(
        "/v1/proposals/final-check",
        files=files,
        data={"request": json.dumps({"draft": DRAFT})},
        headers=env.auth(can_see_restricted=True),
    )
    assert r.status_code == 200 and r.json()["files"] == ["Technical.txt", "Commercial.txt"]
    r = await env.client.post(
        "/v1/proposals/final-check",
        files=[*files, ("files", ("Third.txt", b"x", "text/plain"))],
        data={"request": json.dumps({"draft": DRAFT})},
        headers=env.auth(),
    )
    assert r.status_code == 422 and r.json()["rule"] == "files"


# ---- #2 Draft answers: citations, confidence, sources ---------------------------------------


async def test_drafts_carry_citations_and_confidence_and_never_use_lost_proposals(env: Env) -> None:
    await _index(env, "won1", "Secure fleet", b"We are ISO 27001 certified since 2019.")
    await _index(env, "lost1", "Lost bid", b"We are ISO 27001 certified since 2019.", outcome="lost")
    searched: list[list[str]] = []

    async def script(call: AgentCall[Any], tools: dict[str, Tool]) -> dict[str, Any]:
        hits = json.loads(await tools["search_library"].run({"query": "ISO 27001"}))
        searched.append([h["proposal"] for h in hits])
        return await _draft_script(call, tools)

    env.llm.on_tools("drafting", script)
    env.llm.on("review", approve_all)
    body = {
        "questions": [{"id": "q_sec", "text": "Describe your ISO 27001 certification."}],
        "selected_entry_ids": ["lost1", "won1"],
    }
    answer = (await env.client.post("/v1/proposals/draft-answers", json=body, headers=env.auth())).json()[
        "answers"
    ][0]
    assert answer["status"] == "drafted" and answer["confidence"] == 0.9
    assert answer["no_confident_source"] is False
    citation = answer["citations"][0]
    assert citation["source_id"] == "won1" and citation["title"] == "Client won1 — Secure fleet (2025)"
    assert "ISO 27001" in citation["quote"] and citation["page"] == 1 and citation["passage_id"]
    assert all(not p.startswith("Client lost1") for found in searched for p in found)


async def test_only_lost_sources_or_no_material_means_no_confident_source(env: Env) -> None:
    await _index(env, "lost1", "Lost bid", b"We are ISO 27001 certified.", outcome="lost")
    body = {"questions": [{"id": "q", "text": "Describe your ISO 27001."}], "selected_entry_ids": ["lost1"]}
    answer = (await env.client.post("/v1/proposals/draft-answers", json=body, headers=env.auth())).json()[
        "answers"
    ][0]
    assert answer["status"] == "needs_person" and answer["no_confident_source"] is True
    assert answer["confidence"] == 0 and answer["citations"] == [] and "lost" in answer["reason"]
    assert env.llm.count("drafting") == 0


# ---- #3 Suggest with AI: limit and repeats --------------------------------------------------


async def test_client_queries_respect_the_limit_and_skip_what_is_asked_or_assumed(env: Env) -> None:
    env.llm.on(
        "clarification",
        lambda call: {
            "questions": [
                _question("u1", "How many vehicles are in the fleet today?"),
                _question("u1", "How many vehicles in the fleet today?"),  # near-repeat of an existing one
                _question("x_1", "Do dispatchers work in shifts?"),  # settled by an assumption
                _question("x_1", "Who uses the dispatch screen?"),
                _question("x_1", "Which browsers must dispatch support?"),
                _question("u1", "Which depots are in scope?"),
            ]
        },
    )
    env.llm.on("review", approve_all)
    body = QUERY_BODY | {
        "existing_questions": ["How many vehicles are in the fleet?"],
        "existing_assumptions": ["Dispatchers work in shifts."],
        "limit": 3,
    }
    r = await env.client.post("/v1/proposals/client-queries", json=body, headers=env.auth())
    assert r.status_code == 200, r.text
    assert [q["question"] for q in r.json()["questions"]] == [
        "Who uses the dispatch screen?",
        "Which browsers must dispatch support?",
        "Which depots are in scope?",
    ]
    call = env.llm.calls[0]
    assert "Write at most 3 new questions." in call.input and "- Dispatchers work in shifts." in call.input
    r = await env.client.post(
        "/v1/proposals/client-queries", json=QUERY_BODY | {"limit": 13}, headers=env.auth()
    )
    assert r.status_code == 422
