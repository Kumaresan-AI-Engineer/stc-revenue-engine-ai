"""Contract tests for Ask AI (Workspace Assistant): status codes, body shape, the retry, the cache, the
redaction of amounts and the verified flag."""

from typing import Any

import pytest
from conftest import Env

from revenue_engine_ai.core.errors import UpstreamError
from revenue_engine_ai.llm.ports import AgentCall

ASK = "/v1/ask"

FACTS = [
    {"id": "F1", "label": "Bid pipeline: outcomes", "text": "Won: 7. Lost: 5. Win rate: 58.3%."},
    {"id": "F2", "label": "Proposal library", "text": "12 proposals: 8 won, 4 lost."},
]
DOCUMENTS = [
    {
        "id": "D1",
        "kind": "proposal",
        "title": "Fleet tracking platform",
        "client": "Nordwind",
        "passage": "Live tracking for 400 trucks, priced at $120,000.",
    }
]


def _body(question: str = "How many bids did we win?", **overrides: Any) -> dict[str, Any]:
    return {"question": question, "history": [], "facts": FACTS, "documents": DOCUMENTS, **overrides}


def _answer(**overrides: Any) -> dict[str, Any]:
    return {
        "answered": True,
        "answer": "You won 7 bids and lost 5.",
        "how": "Counted from the bid pipeline (F1).",
        "source_ids": ["F1"],
        "follow_ups": ["Which bids did we lose?"],
        **overrides,
    }


async def test_it_needs_a_service_token(env: Env) -> None:
    r = await env.client.post(ASK, json=_body())
    assert r.status_code == 401 and r.headers["content-type"] == "application/problem+json"


@pytest.mark.parametrize(
    "body",
    [
        _body(""),
        _body("x" * 1001),
        _body(facts=[{"id": "X1", "label": "a", "text": "b"}]),
        _body(facts=[FACTS[0], FACTS[0]]),
        _body(documents=[{**DOCUMENTS[0], "kind": "email"}]),
    ],
)
async def test_invalid_requests_are_a_422_problem(env: Env, body: dict[str, Any]) -> None:
    r = await env.client.post(ASK, json=body, headers=env.auth())
    assert r.status_code == 422 and r.json()["status"] == 422


async def test_an_answer_names_its_sources_and_how_and_is_cached(env: Env) -> None:
    env.llm.on("workspace_assistant", lambda call: _answer())
    r = await env.client.post(ASK, json=_body(), headers=env.auth())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"] == "You won 7 bids and lost 5." and data["verified"] is True
    assert data["source_ids"] == ["F1"] and data["how"].startswith("Counted from the bid pipeline")
    assert data["meta"]["prompt_versions"] == {"workspace_assistant": "workspace-assistant-v3"}
    sent = env.llm.calls[0].input
    assert "$120,000" not in sent and "[amount]" in sent  # amounts in passages never reach the prompt
    again = await env.client.post(ASK, json=_body(), headers=env.auth())
    assert again.json()["meta"]["cached"] is True and env.llm.count("workspace_assistant") == 1


async def test_an_untraced_figure_is_asked_again_once(env: Env) -> None:
    def assistant(call: AgentCall[Any]) -> dict[str, Any]:
        if "Fix these" not in call.input:
            return _answer(answer="You won 9 bids.")
        assert "the figures 9" in call.input
        return _answer()

    env.llm.on("workspace_assistant", assistant)
    data = (await env.client.post(ASK, json=_body(), headers=env.auth())).json()
    assert data["answer"] == "You won 7 bids and lost 5." and data["verified"] is True
    assert env.llm.count("workspace_assistant") == 2


async def test_a_figure_still_untraced_after_the_retry_is_flagged(env: Env) -> None:
    env.llm.on("workspace_assistant", lambda call: _answer(answer="You won 9 bids."))
    data = (await env.client.post(ASK, json=_body(), headers=env.auth())).json()
    assert data["verified"] is False and data["answered"] is True
    assert env.llm.count("workspace_assistant") == 2


async def test_a_question_the_data_cannot_answer(env: Env) -> None:
    env.llm.on(
        "workspace_assistant",
        lambda call: _answer(
            answered=False, answer="I can't tell the weather.", source_ids=[], follow_ups=[]
        ),
    )
    data = (await env.client.post(ASK, json=_body("What's the weather?"), headers=env.auth())).json()
    assert data["answered"] is False and data["answer"] == "I can't tell the weather."
    assert env.llm.count("workspace_assistant") == 1


async def test_history_is_sent_for_a_follow_up(env: Env) -> None:
    env.llm.on("workspace_assistant", lambda call: _answer())
    history = [{"question": "How many did we win?", "answer": "You won 7 bids."}]
    await env.client.post(ASK, json=_body("and lost?", history=history), headers=env.auth())
    assert "How many did we win?" in env.llm.calls[0].input


async def test_a_provider_failure_is_a_502_problem(env: Env) -> None:
    def fails(call: AgentCall[Any]) -> dict[str, Any]:
        raise UpstreamError("The model is not available.")

    env.llm.on("workspace_assistant", fails)
    r = await env.client.post(ASK, json=_body(), headers=env.auth())
    assert r.status_code == 502 and r.json()["status"] == 502


async def test_a_greeting_gets_a_friendly_reply_without_sources(env: Env) -> None:
    reply = "Hi! Ask me about your bids, proposals or case studies."
    env.llm.on(
        "workspace_assistant",
        lambda call: {
            "kind": "chat",
            "answered": True,
            "answer": reply,
            "how": "",
            "source_ids": [],
            "follow_ups": [],
        },
    )
    data = (await env.client.post(ASK, json=_body("hi"), headers=env.auth())).json()
    assert data["kind"] == "chat" and data["answer"] == reply and data["source_ids"] == []
    assert env.llm.count("workspace_assistant") == 1  # no retry: a chat reply needs no sources


async def test_any_question_is_answered_a_general_one_from_general_knowledge(env: Env) -> None:
    def assistant(call: AgentCall[Any]) -> dict[str, Any]:
        assert "Proposal builder has 8 steps" in call.instructions or "8 steps" in call.instructions
        return {
            "kind": "general",
            "answered": True,
            "answer": "Go to RFPs > New proposal and follow the 8 steps, from Input to Submit.",
            "how": "From the app guide, not from your workspace data.",
            "source_ids": [],
            "follow_ups": ["What happens at the Matches step?"],
        }

    env.llm.on("workspace_assistant", assistant)
    data = (await env.client.post(ASK, json=_body("How do I create a proposal?"), headers=env.auth())).json()
    assert data["kind"] == "general" and data["answered"] is True and data["source_ids"] == []
    assert (
        data["answer"].startswith("Go to RFPs > New proposal") and env.llm.count("workspace_assistant") == 1
    )
