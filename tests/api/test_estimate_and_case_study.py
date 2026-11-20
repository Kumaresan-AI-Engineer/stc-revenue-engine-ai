"""Contract tests for the Estimate Assistant (Step 6 'Ask AI') and the Case Study Writer: status codes, body
shapes, the code checks around each agent, the retry, the cache and the no-money rule."""

import json
from typing import Any

import pytest
from conftest import Env

from agents.estimate_assistant import UNCLEAR_MESSAGE
from core.errors import UpstreamError
from llm.ports import AgentCall

ESTIMATE = "/v1/proposals/estimate-request"
CASE_STUDY = "/v1/case-studies/write"

STATE: dict[str, Any] = {
    "engagement_model": "Fixed cost",
    "delivery_approach": "legacy_team",
    "delivery_approaches": [
        {"key": "legacy_team", "label": "Legacy development"},
        {"key": "ai_assisted", "label": "AI-assisted development", "note": "less build effort"},
    ],
    "weeks": 26,
    "team": 6,
    "person_days": 420,
    "team_min": 2,
    "team_max": 12,
    "weeks_min": 16,
    "weeks_max": 60,
    "in_scope": [
        {"key": "tracking", "label": "Live tracking", "kind": "core", "days": 60, "must": True},
        {"key": "x_driver", "label": "Driver mobile app", "kind": "custom", "days": 48},
    ],
    "out_of_scope": [{"key": "reports", "label": "Reports", "kind": "standard", "days": 0}],
    "can_change_price": True,
}


def _ask(ask: str, state: dict[str, Any] | None = None, amounts: list[str] | None = None) -> dict[str, Any]:
    return {"ask": ask, "amounts": amounts or [], "state": state or STATE}


@pytest.mark.parametrize("path", [ESTIMATE, CASE_STUDY])
async def test_both_need_a_service_token(env: Env, path: str) -> None:
    r = await env.client.post(path, json={})
    assert r.status_code == 401 and r.headers["content-type"] == "application/problem+json"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (ESTIMATE, _ask("")),
        (ESTIMATE, _ask("x" * 501)),
        (ESTIMATE, _ask("bring it to [AMOUNT_1]", amounts=["40000"])),
        (ESTIMATE, _ask("go", state={**STATE, "team_min": 9, "team_max": 4})),
        (ESTIMATE, _ask("go", state={**STATE, "out_of_scope": STATE["in_scope"][:1]})),
        (CASE_STUDY, {"entry_id": "e1", "facts": {"client": "A", "title": "B", "outcome": "pending"}}),
        (CASE_STUDY, {"entry_id": "", "facts": {"client": "A", "title": "B", "outcome": "won"}}),
    ],
)
async def test_invalid_requests_are_a_422_problem(env: Env, path: str, body: dict[str, Any]) -> None:
    r = await env.client.post(path, json=body, headers=env.auth())
    assert r.status_code == 422 and r.json()["status"] == 422


def _action(kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "key": None,
        "weeks": None,
        "team": None,
        "amount_ref": None,
        "percent": None,
    } | fields


async def test_a_dropped_action_is_asked_again_once_and_the_answer_is_cached(env: Env) -> None:
    def assistant(call: AgentCall[Any]) -> dict[str, Any]:
        if "Fix these" not in call.input:  # first answer: leaves out a must-have
            return {
                "understood": True,
                "message": "Cut the scope.",
                "actions": [{"kind": "leave_out", "key": "tracking"}, {"kind": "set_timeline", "weeks": 30}],
            }
        assert "must-have" in call.input
        return {
            "understood": True,
            "message": "I left out the driver app and extended the timeline to 30 weeks.",
            "actions": [{"kind": "leave_out", "key": "x_driver"}, {"kind": "set_timeline", "weeks": 30}],
        }

    env.llm.on("estimate_assistant", assistant)
    body = _ask("drop what we can and extend the timeline to 30 weeks")
    r = await env.client.post(ESTIMATE, json=body, headers=env.auth())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["understood"] is True and env.llm.count("estimate_assistant") == 2
    assert data["actions"] == [_action("leave_out", key="x_driver"), _action("set_timeline", weeks=30)]
    assert data["meta"]["prompt_versions"] == {"estimate_assistant": "estimate-assistant-v2"}
    sent = env.llm.calls[0].input.split("State (JSON):", 1)[1]
    assert '"price"' not in sent and '"can_change_price": true' in sent  # no price is ever sent
    again = await env.client.post(ESTIMATE, json=body, headers=env.auth())
    assert again.json()["meta"]["cached"] is True and env.llm.count("estimate_assistant") == 2


async def test_only_commercial_may_change_the_price(env: Env) -> None:
    env.llm.on(
        "estimate_assistant",
        lambda call: {
            "understood": True,
            "message": "Only Commercial can set the price.",
            "actions": [{"kind": "set_price", "amount_ref": "AMOUNT_1"}],
        },
    )
    sales = {**STATE, "can_change_price": False}
    body = _ask("bring it down to [AMOUNT_1]", state=sales, amounts=["AMOUNT_1"])
    data = (await env.client.post(ESTIMATE, json=body, headers=env.auth())).json()
    assert data["actions"] == [] and data["message"] == "Only Commercial can set the price."
    assert env.llm.count("estimate_assistant") == 2  # asked again once with the note


async def test_a_price_cut_by_a_percentage_or_an_amount_reference(env: Env) -> None:
    env.llm.on(
        "estimate_assistant",
        lambda call: {
            "understood": True,
            "message": "I cut the price by 20%.",
            "actions": [{"kind": "set_price", "percent": -20}],
        },
    )
    data = (await env.client.post(ESTIMATE, json=_ask("reduce the cost by 20%"), headers=env.auth())).json()
    assert (
        data["actions"] == [_action("set_price", percent=-20.0)]
        and data["message"] == "I cut the price by 20%."
    )


async def test_a_question_changes_nothing(env: Env) -> None:
    env.llm.on(
        "estimate_assistant",
        lambda call: {
            "understood": True,
            "message": "26 weeks: 420 person-days with a team of 6, built in two parallel tracks.",
            "actions": [],
        },
    )
    data = (await env.client.post(ESTIMATE, json=_ask("why is it 26 weeks?"), headers=env.auth())).json()
    assert data["understood"] is True and data["actions"] == []
    assert data["message"].startswith("26 weeks") and env.llm.count("estimate_assistant") == 1


async def test_a_request_that_is_not_about_the_estimate_is_not_understood(env: Env) -> None:
    env.llm.on("estimate_assistant", lambda call: {"understood": False, "message": "", "actions": []})
    data = (await env.client.post(ESTIMATE, json=_ask("write me a poem"), headers=env.auth())).json()
    assert (data["understood"], data["message"], data["actions"]) == (False, UNCLEAR_MESSAGE, [])


async def test_a_provider_failure_is_a_502_problem(env: Env) -> None:
    def fails(call: AgentCall[Any]) -> dict[str, Any]:
        raise UpstreamError("The model is not available.")

    env.llm.on("estimate_assistant", fails)
    env.llm.on("case_study_writer", fails)
    r = await env.client.post(ESTIMATE, json=_ask("shorter"), headers=env.auth())
    assert r.status_code == 502 and r.json()["status"] == 502
    body = {"entry_id": "e9", "facts": {"client": "A", "title": "B", "outcome": "won"}}
    assert (await env.client.post(CASE_STUDY, json=body, headers=env.auth())).status_code == 502


# ---- Case study ------------------------------------------------------------------------------------------

PROPOSAL = (
    b"Bluewater Logistics runs 1,200 trucks and could not see them between depots.\n\n"
    b"Our solution: a live tracking platform with alerts, built on the client's telematics.\n\n"
    b"Results: dispatch time fell by 35% within three months.\n\n"
    b"Commercials: the total fee is EUR 120,000 payable in three milestones."
)
FACTS = {
    "client": "Bluewater Logistics",
    "title": "Fleet visibility",
    "industry": "Logistics",
    "engagement_model": "Fixed cost",
    "outcome": "won",
    "reason": "Fastest go-live",
    "weeks": 26,
    "features": ["Live tracking", "Alerts"],
}
TEMPLATE = [
    {"key": "challenge", "title": "The challenge", "guidance": "Two short paragraphs, client's view."}
]


async def _index(env: Env, entry_id: str) -> None:
    meta = {
        "client": "Bluewater Logistics",
        "title": "Fleet visibility",
        "industry": "Logistics",
        "outcome": "won",
        "year": 2025,
    }
    r = await env.client.put(
        f"/v1/library/entries/{entry_id}",
        data={"meta": json.dumps(meta)},
        files={"proposal": ("proposal.txt", PROPOSAL, "text/plain")},
        headers=env.auth(),
    )
    assert r.status_code == 200, r.text


def _passage_ids(call: AgentCall[Any]) -> list[str]:
    block = call.input.split("Passages from the proposal (id, page: text):", 1)[1]
    return [
        line[2:].split(" ", 1)[0] for line in block.splitlines() if line.startswith("- ") and "(p." in line
    ]


def _story(call: AgentCall[Any], **fields: Any) -> dict[str, Any]:
    return {
        "title": "Seeing every truck: Bluewater's live fleet platform",
        "summary": "Bluewater could not see 1,200 trucks between depots; we proposed live tracking.",
        "metrics": [{"v": "35%", "l": "Faster dispatch"}, {"v": "60%", "l": "Fewer calls"}],
        "industry": "Logistics",
        "scope": "Live tracking platform",
        "engagement": "Fixed cost, 26 weeks",
        "about_client": "Bluewater Logistics runs a fleet of 1,200 trucks.",
        "challenges": [{"title": "Trucks out of sight", "text": "No view between depots."}],
        "solution_intro": "A live tracking platform on the client's telematics.",
        "solution_pillars": [{"title": "Live map", "text": "Every truck, live.", "tag": "live"}],
        "features": ["Live tracking", "Alerts"],
        "why_us": "We won on a fixed fee of EUR 120,000.",
        "sections": [
            {
                "key": "challenge",
                "paragraphs": ["Bluewater could not see its trucks.", "The fee was EUR 120,000."],
            }
        ],
        "sources": _passage_ids(call)[:2],
    } | fields


async def test_a_case_study_is_written_from_the_indexed_proposal_without_money_or_invented_metrics(
    env: Env,
) -> None:
    await _index(env, "e1")
    env.llm.on("case_study_writer", _story)
    body = {"entry_id": "e1", "facts": FACTS, "template": TEMPLATE}
    r = await env.client.post(CASE_STUDY, json=body, headers=env.auth())
    assert r.status_code == 200, r.text
    data = r.json()
    call = next(c for c in env.llm.calls if c.agent == "case_study_writer")
    assert (
        "dispatch time fell by 35%" in call.input
        and "challenge: The challenge — Two short paragraphs" in call.input
    )
    assert data["metrics"] == [{"v": "35%", "l": "Faster dispatch"}]  # 60% is in neither facts nor passages
    assert data["why_us"] == "" and any("money" in d for d in data["dropped"])
    assert data["engagement"] == "Fixed cost, 26 weeks"
    assert data["sources"] and all(s["passage_id"] in _passage_ids(call) for s in data["sources"])
    assert data["solution_pillars"][0]["tag"] == "LIVE"
    assert data["sections"] == [{"key": "challenge", "paragraphs": ["Bluewater could not see its trucks."]}]
    assert data["meta"]["prompt_versions"] == {"case_study_writer": "case-study-writer-v2"}
    again = await env.client.post(CASE_STUDY, json=body, headers=env.auth())
    assert again.json()["meta"]["cached"] is True and env.llm.count("case_study_writer") == 1


async def test_a_lost_bid_without_an_index_is_written_from_the_facts_and_retried_without_a_title(
    env: Env,
) -> None:
    def story(call: AgentCall[Any]) -> dict[str, Any]:
        assert "(none: use the facts only)" in call.input and '"outcome": "lost"' in call.input
        if "Fix these" not in call.input:
            return _story(call, title="", sources=[])
        return _story(call, title="Lessons from Fleet visibility", metrics=[], sources=["p-made-up"])

    env.llm.on("case_study_writer", story)
    body = {"entry_id": "e-new", "facts": FACTS | {"outcome": "lost", "reason": "Price too high"}}
    data = (await env.client.post(CASE_STUDY, json=body, headers=env.auth())).json()
    assert data["title"] == "Lessons from Fleet visibility" and data["sources"] == []
    assert env.llm.count("case_study_writer") == 2
