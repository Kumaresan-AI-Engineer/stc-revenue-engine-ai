"""The Chat Completions adapter against a stub client: schema in the prompt, validation and correction,
the tool loop without reasoning, and error mapping."""

import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from core.config import Settings
from core.errors import UpstreamError
from llm.chat_completions_provider import ChatCompletionsLLM
from llm.ports import AgentCall, Tool
from llm.providers import PROVIDERS


class Answer(BaseModel):
    text: str


def _tool_call(name: str, args: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=args))


def _reply(
    content: str | None = None, tool_calls: list[Any] | None = None, finish: str = "stop"
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )


class StubCompletions:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _llm(replies: list[Any], **settings: Any) -> tuple[ChatCompletionsLLM, StubCompletions]:
    stub = StubCompletions(replies)
    client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return ChatCompletionsLLM(client, Settings(_env_file=None, **settings)), stub  # type: ignore[arg-type]


CALL = AgentCall("drafting", "Be precise.", "Question?", Answer, "medium")


def test_the_setting_selects_this_adapter() -> None:
    assert PROVIDERS["chat_completions"] is ChatCompletionsLLM


async def test_structured_call_puts_the_schema_in_the_prompt_and_asks_for_json() -> None:
    llm, stub = _llm([_reply('```json\n{"text": "hi"}\n```')], chat_model="openai/gpt-6-luna")
    done = await llm.structured(CALL)
    assert done.value.text == "hi" and done.usage.calls == 1 and done.usage.input_tokens == 10
    request = stub.requests[0]
    assert request["model"] == "openai/gpt-6-luna"
    assert request["response_format"] == {"type": "json_object"}
    assert request["reasoning_effort"] == "medium"
    assert request["max_completion_tokens"] == 32000
    system, user = request["messages"]
    assert system["role"] == "system" and system["content"].startswith("Be precise.")
    assert json.dumps(Answer.model_json_schema(), separators=(",", ":")) in system["content"]
    assert user == {"role": "user", "content": "Question?"}
    assert "temperature" not in request


async def test_reasoning_and_temperature_are_sent_only_when_configured() -> None:
    llm, stub = _llm([_reply('{"text": "x"}')], temperature=0)
    await llm.structured(AgentCall("review", "i", "x", Answer, None))
    assert stub.requests[0]["temperature"] == 0
    assert "reasoning_effort" not in stub.requests[0]


async def test_an_answer_that_misses_the_schema_is_sent_back_once_to_be_corrected() -> None:
    llm, stub = _llm([_reply('{"wrong": 1}'), _reply('{"text": "fixed"}')])
    done = await llm.structured(CALL)
    assert done.value.text == "fixed" and done.usage.calls == 2
    retry = stub.requests[1]["messages"]
    assert retry[2] == {"role": "assistant", "content": '{"wrong": 1}'}
    assert retry[3]["role"] == "user" and "does not match the JSON schema" in retry[3]["content"]


async def test_an_answer_still_wrong_after_correction_is_an_upstream_error() -> None:
    llm, _ = _llm([_reply("not json"), _reply('{"wrong": 1}')])
    with pytest.raises(UpstreamError):
        await llm.structured(CALL)


async def test_an_answer_cut_off_at_the_output_limit_fails_at_once_without_a_correction() -> None:
    llm, stub = _llm([_reply('{"text": "trunc', finish="length")], max_output_tokens=2048)
    with pytest.raises(UpstreamError, match="output limit"):
        await llm.structured(CALL)
    assert len(stub.requests) == 1 and stub.requests[0]["max_completion_tokens"] == 2048


async def test_tool_loop_runs_without_reasoning_and_threads_calls_and_outputs() -> None:
    seen: list[dict[str, Any]] = []

    async def search(args: dict[str, Any]) -> str:
        seen.append(args)
        return '[{"passage_id": "p1"}]'

    tool = Tool("search_library", "Search.", {"type": "object"}, search)
    llm, stub = _llm(
        [
            _reply(None, [_tool_call("search_library", '{"query": "iso"}', "c1")]),
            _reply('{"text": "done"}'),
        ]
    )
    done = await llm.with_tools(CALL, [tool], max_tool_calls=5)
    assert done.value.text == "done" and seen == [{"query": "iso"}]
    assert done.usage.calls == 2 and done.usage.tool_calls == 1
    assert all(r["reasoning_effort"] == "none" for r in stub.requests)
    spec = {"name": "search_library", "description": "Search.", "parameters": {"type": "object"}}
    assert stub.requests[0]["tools"] == [{"type": "function", "function": spec}]
    second = stub.requests[1]["messages"]
    assert second[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search_library", "arguments": '{"query": "iso"}'},
            }
        ],
    }
    assert second[3] == {"role": "tool", "tool_call_id": "c1", "content": '[{"passage_id": "p1"}]'}


async def test_tool_budget_switches_tools_off_and_bad_calls_get_an_error_text() -> None:
    async def never(args: dict[str, Any]) -> str:
        raise AssertionError("not called")

    llm, stub = _llm(
        [
            _reply(None, [_tool_call("unknown_tool", "{}", "c1")]),
            _reply(None, [_tool_call("search_library", "{not json", "c2")]),
            _reply('{"text": "final"}'),
        ]
    )
    done = await llm.with_tools(CALL, [Tool("search_library", "S", {}, never)], max_tool_calls=2)
    assert done.value.text == "final"
    assert [r["tool_choice"] for r in stub.requests] == ["auto", "auto", "none"]
    outputs = [m["content"] for m in stub.requests[2]["messages"] if m["role"] == "tool"]
    assert outputs == ["Unknown tool 'unknown_tool'.", "The arguments were not valid JSON."]


async def test_a_prose_answer_after_tools_is_corrected_with_tools_off() -> None:
    llm, stub = _llm([_reply("Here is the answer: done"), _reply('{"text": "done"}')])
    tool = Tool("t", "T", {}, lambda args: None)  # type: ignore[arg-type, return-value]
    done = await llm.with_tools(CALL, [tool], max_tool_calls=3)
    assert done.value.text == "done" and done.usage.calls == 2
    assert stub.requests[1]["tool_choice"] == "none"
    assert stub.requests[1]["messages"][-1]["role"] == "user"


async def test_model_that_never_stops_calling_tools_is_cut_off() -> None:
    replies = [_reply(None, [_tool_call("t", "{}", f"c{i}")]) for i in range(4)]

    async def run(args: dict[str, Any]) -> str:
        return "ok"

    llm, _ = _llm(replies)
    with pytest.raises(UpstreamError):
        await llm.with_tools(CALL, [Tool("t", "T", {}, run)], max_tool_calls=2)


async def test_provider_errors_become_upstream_errors() -> None:
    request = httpx.Request("POST", "https://gateway.example/v1/chat/completions")
    llm, _ = _llm([openai.APIConnectionError(request=request)])
    with pytest.raises(UpstreamError):
        await llm.structured(CALL)


async def test_images_go_as_image_url_parts_of_the_user_message() -> None:
    llm, stub = _llm([_reply('{"text": "seen"}')])
    call = AgentCall(
        "template_vision", "Read it.", "Pages follow.", Answer, "low", ("data:image/png;base64,AA",)
    )
    done = await llm.structured(call)
    assert done.value.text == "seen"
    user = stub.requests[0]["messages"][1]
    assert user == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Pages follow."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        ],
    }
