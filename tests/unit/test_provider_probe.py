import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from revenue_engine_ai.core.config import Settings, get_settings
from revenue_engine_ai.core.errors import ProviderCredentialError, UpstreamError
from revenue_engine_ai.llm.failures import MODEL_UNKNOWN, NO_TOOLS
from revenue_engine_ai.llm.ports import AgentCall, Completion, Tool, Usage
from revenue_engine_ai.services import provider_probe
from revenue_engine_ai.services.provider_probe import ProbeAnswer, ProbeSum, probe


class Scripted:
    """A model that answers each probe call from a function, or raises what the function raises."""

    model = "scripted-model"

    def __init__(self, answer: Any = None, tools: Any = None, delay: float = 0.0) -> None:
        self.answer = answer or (lambda: ProbeAnswer(ok=True, number=7))
        self.tools = tools or (lambda: (ProbeSum(total=42), 1))
        self.delay = delay
        self.calls: list[AgentCall[Any]] = []
        self.tool_calls = 0

    async def structured[T: BaseModel](self, call: AgentCall[T]) -> Completion[T]:
        self.calls.append(call)
        await asyncio.sleep(self.delay)
        return Completion(self.answer(), Usage(10, 5, 1, 0))  # type: ignore[arg-type]

    async def with_tools[T: BaseModel](
        self, call: AgentCall[T], tools: list[Tool], max_tool_calls: int
    ) -> Completion[T]:
        self.tool_calls += 1
        value, used = self.tools()
        assert [t.name for t in tools] == ["add"]
        return Completion(value, Usage(10, 5, 1, used))  # type: ignore[arg-type]


def _raise(error: Exception) -> Any:
    def go() -> Any:
        raise error

    return go


def _settings(**update: Any) -> Settings:
    return get_settings().model_copy(update=update)


async def test_a_model_that_answers_in_format_and_calls_tools_is_ready() -> None:
    llm = Scripted()

    result = await probe(llm, _settings())

    assert result.verdict == "ready"
    assert [(c.key, c.status) for c in result.checks] == [
        ("answer", "pass"),
        ("speed", "pass"),
        ("tools", "pass"),
    ]


async def test_the_answer_is_asked_at_the_effort_extraction_runs_at() -> None:
    llm = Scripted()

    await probe(llm, _settings(extraction_effort="high"))

    assert llm.calls[0].effort == "high"


async def test_a_model_the_provider_does_not_have_is_unsuitable_and_nothing_else_is_tried() -> None:
    llm = Scripted(answer=_raise(UpstreamError(MODEL_UNKNOWN)))

    result = await probe(llm, _settings())

    assert result.verdict == "unsuitable"
    assert [(c.key, c.status, c.message) for c in result.checks] == [("answer", "fail", MODEL_UNKNOWN)]
    assert llm.tool_calls == 0


async def test_a_refused_key_is_unsuitable_in_the_words_an_admin_can_act_on() -> None:
    llm = Scripted(answer=_raise(ProviderCredentialError()))

    result = await probe(llm, _settings())

    assert result.verdict == "unsuitable" and "Admin > AI provider" in result.checks[0].message


async def test_a_model_without_tool_support_is_unsuitable() -> None:
    llm = Scripted(tools=_raise(UpstreamError(NO_TOOLS)))

    result = await probe(llm, _settings())

    assert result.verdict == "unsuitable"
    assert [(c.key, c.status) for c in result.checks] == [
        ("answer", "pass"),
        ("speed", "pass"),
        ("tools", "fail"),
    ]
    assert result.checks[2].message == NO_TOOLS


async def test_a_model_that_does_not_use_the_tool_works_with_limits() -> None:
    llm = Scripted(tools=lambda: (ProbeSum(total=42), 0))

    result = await probe(llm, _settings())

    assert result.verdict == "limited" and result.checks[2].status == "warn"


async def test_a_wrong_tool_result_works_with_limits() -> None:
    llm = Scripted(tools=lambda: (ProbeSum(total=1), 1))

    result = await probe(llm, _settings())

    assert result.verdict == "limited" and "wrong result" in result.checks[2].message


async def test_an_answer_with_the_wrong_values_works_with_limits() -> None:
    llm = Scripted(answer=lambda: ProbeAnswer(ok=False, number=3))

    result = await probe(llm, _settings())

    assert result.verdict == "limited" and result.checks[0].status == "warn"


async def test_a_slow_model_works_with_limits_and_says_how_slow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_probe, "SLOW_SECONDS", 0.0)
    llm = Scripted(delay=0.05)

    result = await probe(llm, _settings())

    assert result.verdict == "limited"
    speed = next(c for c in result.checks if c.key == "speed")
    assert speed.status == "warn" and "Slow" in speed.message and result.seconds > 0


async def test_a_model_that_does_not_finish_in_time_is_unsuitable_and_names_the_setting() -> None:
    llm = Scripted(delay=1.0)

    result = await probe(llm, _settings(provider_probe_timeout_s=0.05))

    assert result.verdict == "unsuitable"
    assert [c.key for c in result.checks] == ["answer"]
    assert "RE_AI_PROVIDER_PROBE_TIMEOUT_S" in result.checks[0].message
