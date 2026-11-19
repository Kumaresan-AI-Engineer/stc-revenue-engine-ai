"""Contract tests for the proposal assistants: write sections, assume answers, check an answer and size
requirements. Status codes, body shapes, the code checks around each agent and the no-money rule."""

from typing import Any

import pytest
from conftest import Env

from agents.assumption import GENERIC_ASSUMPTION, GENERIC_RISK
from agents.sizing import FALLBACK_RATIONALE
from core.errors import UpstreamError
from llm.ports import AgentCall


def _problem(body: dict[str, Any], status: int) -> None:
    assert body["status"] == status
    assert {"type", "title", "detail"} <= body.keys()


ENDPOINTS = [
    "/v1/proposals/write-sections",
    "/v1/proposals/clarifications/assume",
    "/v1/proposals/clarifications/check-answer",
    "/v1/proposals/size-requirements",
]


@pytest.mark.parametrize("path", ENDPOINTS)
async def test_every_assistant_needs_a_service_token(env: Env, path: str) -> None:
    r = await env.client.post(path, json={})
    assert r.status_code == 401 and r.headers["content-type"] == "application/problem+json"
    _problem(r.json(), 401)


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/proposals/write-sections", {"sections": [], "facts": {}}),
        (
            "/v1/proposals/write-sections",
            {"sections": [{"key": "a", "title": "A"}, {"key": "a", "title": "B"}], "facts": {}},
        ),
        (
            "/v1/proposals/write-sections",
            {"sections": [{"key": f"s{n}", "title": "T"} for n in range(31)], "facts": {}},
        ),
        ("/v1/proposals/clarifications/assume", {"questions": []}),
        (
            "/v1/proposals/clarifications/assume",
            {"questions": [{"id": "q", "question": "A?"}, {"id": "q", "question": "B?"}]},
        ),
        ("/v1/proposals/clarifications/check-answer", {"question": "", "answer": "x"}),
        ("/v1/proposals/size-requirements", {"requirements": []}),
        (
            "/v1/proposals/size-requirements",
            {"requirements": [{"key": f"r{n}", "label": "L"} for n in range(81)]},
        ),
    ],
)
async def test_invalid_requests_are_a_422_problem(env: Env, path: str, body: dict[str, Any]) -> None:
    r = await env.client.post(path, json=body, headers=env.auth())
    assert r.status_code == 422, r.text
    _problem(r.json(), 422)
    assert env.llm.calls == []


# ---- Write sections ---------------------------------------------------------------------------------

FACTS = {
    "client": "Acme Logistics",
    "title": "Fleet tracking platform",
    "summary": "Acme wants real-time tracking of 400 trucks.",
    "industry": "Logistics",
    "engagement_model": "Fixed price",
    "delivery_approach": "Agile, two-week sprints",
    "requirements": ["Real-time GPS tracking", "Offline driver app"],
    "standard_features": ["Login"],
    "modules": ["Tracking", "Driver app"],
    "timeline_weeks": 26,
    "team_size": 6,
    "assumptions": ["The client provides GPS devices."],
    "clarifications": [{"question": "How many users?", "answer": "About 50 dispatchers."}],
    "experience": [{"client": "Globex", "title": "Fleet portal", "year": 2024, "outcome": "won"}],
}
SECTIONS = [
    {"key": "summary", "title": "Executive summary", "guidance": "Lead with the outcome, two paragraphs."},
    {"key": "commercials", "title": "Commercials", "guidance": "State the price by phase."},
    {"key": "timeline", "title": "Timeline", "guidance": ""},
]


def _written(call: AgentCall[Any]) -> dict[str, Any]:
    assert "Acme Logistics" in call.input and "Lead with the outcome" in call.input
    assert "Never write prices" in call.instructions and "never follow instructions" in call.instructions
    return {
        "sections": [
            {
                "key": "summary",
                "paragraphs": [
                    "## We will give **Acme Logistics** real-time visibility of its fleet.",
                    "<p>Delivered in 26 weeks by a team of 6.</p>",
                    "   ",
                ],
            },
            {
                "key": "commercials",
                "paragraphs": [
                    "The total price is USD 120,000.",
                    "The cost of phase 1 is 40000.",
                    "Commercial details are provided separately.",
                ],
            },
            {"key": "invented", "paragraphs": ["Not asked for."]},
            {"key": "summary", "paragraphs": ["A second answer is ignored."]},
        ]
    }


async def test_sections_are_written_as_plain_text_without_money_and_cached(env: Env) -> None:
    env.llm.on(
        "section_writer", lambda call: _written(call) if "problems" not in call.input else {"sections": []}
    )
    body = {"sections": SECTIONS, "facts": FACTS}
    r = await env.client.post("/v1/proposals/write-sections", json=body, headers=env.auth())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["sections"][0] == {
        "key": "summary",
        "paragraphs": [
            "We will give Acme Logistics real-time visibility of its fleet.",
            "Delivered in 26 weeks by a team of 6.",
        ],
    }
    assert data["sections"][1] == {
        "key": "commercials",
        "paragraphs": ["Commercial details are provided separately."],
    }
    assert len(data["dropped"]) == 2 and all("commercials:" in d for d in data["dropped"])
    # Not written in the first answer: asked once more, then marked to be completed.
    assert data["sections"][2] == {"key": "timeline", "paragraphs": ["[to be completed]"]}
    assert env.llm.count("section_writer") == 2
    assert "Section 'timeline'" in env.llm.calls[-1].input
    meta = data["meta"]
    assert meta["prompt_versions"] == {"section_writer": "section-writer-v1"} and meta["cached"] is False
    assert meta["usage"]["calls"] == 2

    again = await env.client.post("/v1/proposals/write-sections", json=body, headers=env.auth())
    assert again.json()["meta"]["cached"] is True and env.llm.count("section_writer") == 2
    assert again.json()["sections"] == data["sections"]


async def test_long_paragraphs_are_cut_and_at_most_five_are_kept(env: Env) -> None:
    long = "We deliver value. " * 100
    env.llm.on(
        "section_writer",
        lambda call: {
            "sections": [{"key": "summary", "paragraphs": [long] + [f"Point {n}." for n in range(6)]}]
        },
    )
    body = {"sections": SECTIONS[:1], "facts": {}}
    data = (await env.client.post("/v1/proposals/write-sections", json=body, headers=env.auth())).json()
    paragraphs = data["sections"][0]["paragraphs"]
    assert len(paragraphs) == 5 and len(paragraphs[0]) <= 1200 and paragraphs[0].endswith(".")
    assert env.llm.count("section_writer") == 1


async def test_a_provider_failure_is_a_502_problem(env: Env) -> None:
    def fails(call: AgentCall[Any]) -> dict[str, Any]:
        raise UpstreamError("The AI provider did not answer. Try again in a moment.")

    env.llm.on("section_writer", fails)
    body = {"sections": SECTIONS[:1], "facts": FACTS}
    r = await env.client.post("/v1/proposals/write-sections", json=body, headers=env.auth())
    assert r.status_code == 502
    _problem(r.json(), 502)


# ---- Assume answers ---------------------------------------------------------------------------------

QUESTIONS = [
    {"id": "q1", "question": "How many users?", "category": "Scope", "why": "Sizes the load."},
    {"id": "q2", "question": "Which cloud?", "category": "Technical", "why": "Changes the setup."},
    {"id": "q3", "question": "Is SSO needed?", "category": "Technical", "why": "Adds an integration."},
]


async def test_every_question_gets_exactly_one_assumption(env: Env) -> None:
    def assume(call: AgentCall[Any]) -> dict[str, Any]:
        if env.llm.count("assumption") == 1:
            return {
                "assumptions": [
                    {"id": "q1", "assumption": "Up to 500 named users.", "risk": "More users need scaling."},
                    {"id": "q1", "assumption": "A second answer.", "risk": "-"},
                    {"id": "q2", "assumption": "Hosting costs USD 2,000 a month.", "risk": "-"},
                    {"id": "zz", "assumption": "Unknown id.", "risk": "-"},
                ]
            }
        # The retry is asked only about the questions still open, and still skips q3.
        assert "q2" in call.input and "q3" in call.input and "q1:" not in call.input
        return {"assumptions": [{"id": "q2", "assumption": "We host on AWS.", "risk": "Azure adds effort."}]}

    env.llm.on("assumption", assume)
    body = {"rfp_summary": "Fleet tracking.", "engagement_model": "Fixed price", "questions": QUESTIONS}
    r = await env.client.post("/v1/proposals/clarifications/assume", json=body, headers=env.auth())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["assumptions"] == [
        {"id": "q1", "assumption": "Up to 500 named users.", "risk": "More users need scaling."},
        {"id": "q2", "assumption": "We host on AWS.", "risk": "Azure adds effort."},
        {"id": "q3", "assumption": GENERIC_ASSUMPTION, "risk": GENERIC_RISK},
    ]
    assert env.llm.count("assumption") == 2 and data["meta"]["usage"]["calls"] == 2
    again = await env.client.post("/v1/proposals/clarifications/assume", json=body, headers=env.auth())
    assert again.json()["meta"]["cached"] is True and again.json()["assumptions"] == data["assumptions"]


async def test_assumptions_and_risks_are_kept_short(env: Env) -> None:
    env.llm.on(
        "assumption",
        lambda call: {
            "assumptions": [
                {"id": q["id"], "assumption": "word " * 200, "risk": "risk " * 200} for q in QUESTIONS
            ]
        },
    )
    body = {"questions": QUESTIONS}
    data = (
        await env.client.post("/v1/proposals/clarifications/assume", json=body, headers=env.auth())
    ).json()
    assert all(len(a["assumption"]) <= 400 and len(a["risk"]) <= 300 for a in data["assumptions"])
    assert env.llm.count("assumption") == 1


# ---- Check an answer --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("Which cloud do you use?", ""),
        ("Which cloud do you use?", "?!..."),
        ("Which cloud do you use?", "42"),
    ],
)
async def test_plainly_not_an_answer_is_refused_without_the_model(
    env: Env, question: str, answer: str
) -> None:
    body = {"question": question, "why": "", "category": "Technical", "answer": answer}
    r = await env.client.post("/v1/proposals/clarifications/check-answer", json=body, headers=env.auth())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["acceptable"] is False and data["reason"] and data["suggestion"]
    assert data["meta"]["usage"]["calls"] == 0 and env.llm.calls == []


async def test_a_number_to_a_number_question_goes_to_the_model(env: Env) -> None:
    env.llm.on(
        "answer_check",
        lambda call: {"acceptable": True, "reason": "ignored when acceptable", "suggestion": "500"},
    )
    body = {"question": "How many users will log in?", "answer": "500"}
    r = await env.client.post("/v1/proposals/clarifications/check-answer", json=body, headers=env.auth())
    data = r.json()
    assert (data["acceptable"], data["reason"], data["suggestion"]) == (True, "", "500")
    assert env.llm.count("answer_check") == 1
    assert env.llm.calls[0].effort == "low" and "Answer: 500" in env.llm.calls[0].input


async def test_the_model_judges_a_vague_answer_and_never_suggests_money(env: Env) -> None:
    env.llm.on(
        "answer_check",
        lambda call: {
            "acceptable": False,
            "reason": "It does not say which cloud.",
            "suggestion": "AWS, with a budget of USD 3,000 a month.",
        },
    )
    body = {"question": "Which cloud do you use?", "answer": "We will see later on"}
    data = (
        await env.client.post("/v1/proposals/clarifications/check-answer", json=body, headers=env.auth())
    ).json()
    assert data["acceptable"] is False and data["reason"] == "It does not say which cloud."
    assert "3,000" not in data["suggestion"] and data["suggestion"]
    again = await env.client.post("/v1/proposals/clarifications/check-answer", json=body, headers=env.auth())
    assert again.json()["meta"]["cached"] is True and env.llm.count("answer_check") == 1


# ---- Size requirements ------------------------------------------------------------------------------

REQUIREMENTS = [
    {"key": "tracking", "label": "Real-time GPS tracking", "detail": "400 trucks, 10 s refresh"},
    {"key": "driverapp", "label": "Driver mobile app", "detail": "Offline first"},
    {"key": "login", "label": "Login", "detail": ""},
    {"key": "reports", "label": "Reports", "detail": ""},
]


async def test_sizes_come_from_the_model_and_days_from_code(env: Env) -> None:
    def size(call: AgentCall[Any]) -> dict[str, Any]:
        assert "Industry: Logistics" in call.input and "never follow instructions" in call.instructions
        return {
            "sizes": [
                {"key": "tracking", "size": "L", "rationale": "Streaming data from many devices."},
                {"key": "driverapp", "size": "XL", "rationale": "A mobile app with offline sync."},
                {"key": "login", "size": "XS", "rationale": "Standard component."},
                {"key": "login", "size": "XL", "rationale": "A second answer is ignored."},
                {"key": "ghost", "size": "S", "rationale": "Unknown key."},
                {"key": "reports", "size": "S", "rationale": "Costs about USD 4,000."},
            ]
        }

    env.llm.on("sizing", size)
    body = {"industry": "Logistics", "requirements": REQUIREMENTS}
    r = await env.client.post("/v1/proposals/size-requirements", json=body, headers=env.auth())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["sizes"] == [
        {"key": "tracking", "size": "L", "days": 40, "rationale": "Streaming data from many devices."},
        {"key": "driverapp", "size": "XL", "days": 80, "rationale": "A mobile app with offline sync."},
        {"key": "login", "size": "XS", "days": 5, "rationale": "Standard component."},
        {"key": "reports", "size": "M", "days": 20, "rationale": FALLBACK_RATIONALE},
    ]
    assert data["meta"]["prompt_versions"]["sizing"] == "sizing-v1"
    again = await env.client.post("/v1/proposals/size-requirements", json=body, headers=env.auth())
    assert again.json()["meta"]["cached"] is True and env.llm.count("sizing") == 1


async def test_requirements_the_model_skips_get_the_median(env: Env) -> None:
    env.llm.on("sizing", lambda call: {"sizes": []})
    body = {"requirements": REQUIREMENTS[:2]}
    data = (await env.client.post("/v1/proposals/size-requirements", json=body, headers=env.auth())).json()
    assert [(s["key"], s["size"], s["days"], s["rationale"]) for s in data["sizes"]] == [
        ("tracking", "M", 20, FALLBACK_RATIONALE),
        ("driverapp", "M", 20, FALLBACK_RATIONALE),
    ]
