from tests.conftest import Env

BODY = {
    "api_mode": "chat_completions",
    "base_url": "http://localhost:11434/v1",
    "model": "qwen3:8b",
    "api_key": "super-secret-key",
}


def _script(env: Env) -> None:
    env.llm.on("provider_probe", lambda call: {"ok": True, "number": 7})

    async def add(call, tools):  # type: ignore[no-untyped-def]
        assert await tools["add"].run({"a": 19, "b": 23}) == "42"
        return {"total": 42}

    env.llm.on_tools("provider_probe_tools", add)


async def test_a_model_is_tried_and_the_checks_come_back(env: Env) -> None:
    _script(env)

    res = await env.client.post("/v1/provider/probe", json=BODY, headers=env.auth())

    assert res.status_code == 200, res.text
    data = res.json()
    assert data["verdict"] == "ready"
    assert [(c["key"], c["status"]) for c in data["checks"]] == [
        ("answer", "pass"),
        ("speed", "pass"),
        ("tools", "pass"),
    ]


async def test_the_key_is_never_sent_back(env: Env) -> None:
    _script(env)

    res = await env.client.post("/v1/provider/probe", json=BODY, headers=env.auth())

    assert "super-secret-key" not in res.text


async def test_a_server_that_needs_no_key_can_be_tried(env: Env) -> None:
    _script(env)
    body = {k: v for k, v in BODY.items() if k != "api_key"}

    res = await env.client.post("/v1/provider/probe", json=body, headers=env.auth())

    assert res.status_code == 200 and res.json()["verdict"] == "ready"


async def test_a_service_token_is_required(env: Env) -> None:
    res = await env.client.post("/v1/provider/probe", json=BODY)

    assert res.status_code == 401


async def test_a_body_without_a_model_is_refused(env: Env) -> None:
    body = {k: v for k, v in BODY.items() if k != "model"}

    res = await env.client.post("/v1/provider/probe", json=body, headers=env.auth())

    assert res.status_code == 422
