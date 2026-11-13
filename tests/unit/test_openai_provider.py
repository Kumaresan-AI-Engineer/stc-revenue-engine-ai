"""The OpenAI adapter against a stub client: request options, the tool loop and error mapping."""

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from core.config import Settings
from core.errors import UpstreamError
from llm.openai_provider import OpenAILLM
from llm.ports import AgentCall, Tool


class Answer(BaseModel):
    text: str


class Reasoning:
    type = "reasoning"

    def to_dict(self, exclude_none: bool = False) -> dict[str, Any]:
        return {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"}


def _call(name: str, args: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", name=name, arguments=args, call_id=call_id)


def _response(output: list[Any], parsed: BaseModel | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        output=output, output_parsed=parsed, usage=SimpleNamespace(input_tokens=10, output_tokens=3)
    )


class StubResponses:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.requests.append(
            {
                **kwargs,
                "input": list(kwargs["input"]) if isinstance(kwargs["input"], list) else kwargs["input"],
            }
        )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _llm(replies: list[Any], **settings: Any) -> tuple[OpenAILLM, StubResponses]:
    stub = StubResponses(replies)
    client = SimpleNamespace(responses=stub)
    return OpenAILLM(client, Settings(**settings)), stub  # type: ignore[arg-type]


CALL = AgentCall("drafting", "Be precise.", "Question?", Answer, "medium")


async def test_structured_call_sends_pinned_private_cacheable_options() -> None:
    llm, stub = _llm([_response([], Answer(text="hi"))], temperature=None)
    done = await llm.structured(CALL)
    assert done.value.text == "hi" and done.usage.calls == 1
    request = stub.requests[0]
    assert request["model"] == "gpt-5.5-2026-04-23"
    assert request["store"] is False
    assert request["prompt_cache_key"] == "revenue-engine-ai:drafting"
    assert request["reasoning"] == {"effort": "medium"}
    assert request["include"] == ["reasoning.encrypted_content"]
    assert request["text_format"] is Answer
    assert "temperature" not in request


async def test_temperature_is_sent_only_when_configured() -> None:
    llm, stub = _llm([_response([], Answer(text="x"))], temperature=0)
    await llm.structured(AgentCall("review", "i", "x", Answer, None))
    assert stub.requests[0]["temperature"] == 0
    assert "reasoning" not in stub.requests[0]


async def test_tool_loop_threads_reasoning_calls_and_outputs() -> None:
    seen: list[dict[str, Any]] = []

    async def search(args: dict[str, Any]) -> str:
        seen.append(args)
        return '[{"passage_id": "p1"}]'

    tool = Tool("search_library", "Search.", {"type": "object"}, search)
    llm, stub = _llm(
        [
            _response([Reasoning(), _call("search_library", '{"query": "iso"}', "c1")]),
            _response([], Answer(text="done")),
        ]
    )
    done = await llm.with_tools(CALL, [tool], max_tool_calls=5)
    assert done.value.text == "done" and seen == [{"query": "iso"}]
    assert done.usage.calls == 2 and done.usage.tool_calls == 1
    second = stub.requests[1]["input"]
    assert second[0] == {"role": "user", "content": "Question?"}
    assert second[1] == {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"}
    assert second[2] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "search_library",
        "arguments": '{"query": "iso"}',
    }
    assert second[3] == {"type": "function_call_output", "call_id": "c1", "output": '[{"passage_id": "p1"}]'}
    assert stub.requests[0]["tools"][0]["strict"] is True


async def test_tool_budget_switches_tools_off_and_bad_calls_get_an_error_text() -> None:
    async def never(args: dict[str, Any]) -> str:
        raise AssertionError("not called")

    llm, stub = _llm(
        [
            _response([_call("unknown_tool", "{}", "c1")]),
            _response([_call("search_library", "{not json", "c2")]),
            _response([], Answer(text="final")),
        ]
    )
    done = await llm.with_tools(CALL, [Tool("search_library", "S", {}, never)], max_tool_calls=2)
    assert done.value.text == "final"
    assert [r["tool_choice"] for r in stub.requests] == ["auto", "auto", "none"]
    outputs = [i["output"] for i in stub.requests[2]["input"] if i.get("type") == "function_call_output"]
    assert outputs == ["Unknown tool 'unknown_tool'.", "The arguments were not valid JSON."]


async def test_model_that_never_stops_calling_tools_is_cut_off() -> None:
    replies = [_response([_call("t", "{}", f"c{i}")]) for i in range(4)]

    async def run(args: dict[str, Any]) -> str:
        return "ok"

    llm, _ = _llm(replies)
    with pytest.raises(UpstreamError):
        await llm.with_tools(CALL, [Tool("t", "T", {}, run)], max_tool_calls=2)


async def test_provider_errors_and_empty_answers_become_upstream_errors() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    error = openai.APIConnectionError(request=request)
    llm, _ = _llm([error, _response([], None)])
    with pytest.raises(UpstreamError):
        await llm.structured(CALL)
    with pytest.raises(UpstreamError):
        await llm.structured(CALL)
