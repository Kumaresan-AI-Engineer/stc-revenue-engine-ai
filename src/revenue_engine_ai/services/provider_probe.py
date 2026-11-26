"""Use-case: try a model before it is saved (admin > AI provider).

Three tiny requests, through the same client the agents use, show whether the model can run this system: it
answers in the structured (JSON) format the agents need (which also proves the server is reached, the model
exists and the output limit is accepted), it does so in a sensible time, and it calls tools (the drafting step
needs them). A refusal is told in plain words by `llm/failures.py`. Nothing here is cached or stored.
"""

import asyncio
import logging
import time
from typing import Literal

from pydantic import BaseModel

from revenue_engine_ai.api.schemas import ProbeCheck, ProviderProbeResponse
from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.errors import DomainError
from revenue_engine_ai.llm.ports import LLM, AgentCall, Tool

log = logging.getLogger(__name__)

AGENT = "provider_probe"
# A tiny request that takes longer than this on a model says long documents will take far longer.
SLOW_SECONDS = 20.0
EXPECTED_NUMBER = 7
EXPECTED_SUM = 42


class ProbeAnswer(BaseModel):
    ok: bool
    number: int


class ProbeSum(BaseModel):
    total: int


async def _add(args: dict[str, object]) -> str:
    a, b = args.get("a"), args.get("b")
    if isinstance(a, int) and isinstance(b, int):
        return str(a + b)
    return "Both a and b must be whole numbers."


ADD_TOOL = Tool(
    name="add",
    description="Adds two whole numbers and returns the sum.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    },
    run=_add,
)


def _verdict(checks: list[ProbeCheck], seconds: float) -> ProviderProbeResponse:
    statuses = {c.status for c in checks}
    verdict: Literal["ready", "limited", "unsuitable"] = (
        "unsuitable" if "fail" in statuses else "limited" if "warn" in statuses else "ready"
    )
    return ProviderProbeResponse(verdict=verdict, checks=checks, seconds=round(seconds, 1))


async def _answer_check(llm: LLM, settings: Settings) -> tuple[ProbeCheck, float]:
    """The structured answer, at the effort extraction runs at, so a model that refuses it shows here."""
    call = AgentCall(
        agent=AGENT,
        instructions="You are a connection test. Reply with the JSON object only.",
        input=f"Set ok to true and number to {EXPECTED_NUMBER}.",
        output=ProbeAnswer,
        effort=settings.extraction_effort,
    )
    started = time.perf_counter()
    try:
        done = await llm.structured(call)
    except DomainError as exc:
        return ProbeCheck(key="answer", status="fail", message=exc.detail), time.perf_counter() - started
    seconds = time.perf_counter() - started
    if done.value.ok and done.value.number == EXPECTED_NUMBER:
        return ProbeCheck(
            key="answer", status="pass", message="Answers in the structured (JSON) format the system needs."
        ), seconds
    return ProbeCheck(
        key="answer",
        status="warn",
        message="Answered in the right format, but not with the values it was asked for.",
    ), seconds


def _speed_check(seconds: float) -> ProbeCheck:
    if seconds > SLOW_SECONDS:
        return ProbeCheck(
            key="speed",
            status="warn",
            message=f"Slow: {seconds:.1f} s for a tiny request. A long document will take much longer.",
        )
    return ProbeCheck(key="speed", status="pass", message=f"Answered in {seconds:.1f} s.")


async def _tools_check(llm: LLM, settings: Settings) -> ProbeCheck:
    call = AgentCall(
        agent=f"{AGENT}_tools",
        instructions="You are a connection test. Use the add tool, then reply with the JSON object only.",
        input="What is 19 plus 23? Set total to the sum.",
        output=ProbeSum,
        effort=None,
    )
    try:
        done = await llm.with_tools(call, [ADD_TOOL], max_tool_calls=2)
    except DomainError as exc:
        return ProbeCheck(key="tools", status="fail", message=exc.detail)
    if done.usage.tool_calls < 1:
        return ProbeCheck(
            key="tools",
            status="warn",
            message="Did not use the test tool. Drafting answers, which rely on tools, may not work with it.",
        )
    if done.value.total != EXPECTED_SUM:
        return ProbeCheck(key="tools", status="warn", message="Used the test tool, but gave a wrong result.")
    return ProbeCheck(key="tools", status="pass", message="Calls tools.")


async def _run(llm: LLM, settings: Settings) -> ProviderProbeResponse:
    answer, seconds = await _answer_check(llm, settings)
    if answer.status == "fail":
        return _verdict([answer], seconds)  # not reached, no such model, ...: nothing else can be tried
    return _verdict([answer, _speed_check(seconds), await _tools_check(llm, settings)], seconds)


async def probe(llm: LLM, settings: Settings) -> ProviderProbeResponse:
    limit = settings.provider_probe_timeout_s
    try:
        async with asyncio.timeout(limit):
            result = await _run(llm, settings)
    except TimeoutError:
        message = (
            f"The model did not finish a tiny test within {limit:.0f} s, so it is too slow for this system. "
            "Choose a smaller model, or raise RE_AI_PROVIDER_PROBE_TIMEOUT_S."
        )
        result = _verdict([ProbeCheck(key="answer", status="fail", message=message)], limit)
    log.info("provider_probe", extra={"fields": {"model": llm.model, "verdict": result.verdict}})
    return result
