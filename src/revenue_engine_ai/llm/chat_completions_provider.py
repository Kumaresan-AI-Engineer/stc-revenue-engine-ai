"""LLM port over the Chat Completions API, for OpenAI-compatible gateways without the Responses API
(e.g. Yantra).

Such gateways accept JSON mode but ignore a JSON schema, so the schema goes into the instructions, every
answer is validated against it and a wrong answer is sent back once to be corrected. They also refuse
function tools together with reasoning, so a tool-using call runs with reasoning off.
"""

import asyncio
import json
import logging
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.errors import UpstreamError
from revenue_engine_ai.llm.failures import provider_failure
from revenue_engine_ai.llm.openai_provider import run_tool
from revenue_engine_ai.llm.ports import AgentCall, Completion, Tool, Usage

log = logging.getLogger(__name__)

# One correction round: a model that cannot follow the schema twice will not on a third try.
MAX_REPAIRS = 1
NO_REASONING = "none"
LENGTH = "length"  # finish_reason when the answer hit the output-token limit


def _without_effort(options: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in options.items() if k != "reasoning_effort"}


def _refuses_effort(exc: openai.APIError, options: dict[str, Any]) -> bool:
    """Whether the server refused the reasoning effort in `options`: a 400 whose words are about reasoning."""
    if not isinstance(exc, openai.BadRequestError) or "reasoning_effort" not in options:
        return False
    return "reasoning" in str(getattr(exc, "message", "") or exc).lower()


def _usage(response: Any, tool_calls: int = 0) -> Usage:
    u = response.usage
    return Usage(u.prompt_tokens if u else 0, u.completion_tokens if u else 0, 1, tool_calls)


def _system(call: AgentCall[Any]) -> str:
    schema = json.dumps(call.output.model_json_schema(), separators=(",", ":"))
    return (
        f"{call.instructions}\n\n"
        "Answer with one JSON object only (no prose, no code fences) that matches this JSON schema "
        f"exactly, with every required field:\n{schema}"
    )


def _user_content(call: AgentCall[Any]) -> str | list[dict[str, Any]]:
    """The input as text, or as text and image parts when the call carries images."""
    if not call.images:
        return call.input
    images = [{"type": "image_url", "image_url": {"url": url}} for url in call.images]
    return [{"type": "text", "text": call.input}, *images]


def _json_text(content: str) -> str:
    """The JSON object in a reply, without code fences or text around it."""
    start, end = content.find("{"), content.rfind("}")
    return content[start : end + 1] if start != -1 and end > start else content


def _validate[T: BaseModel](content: str | None, output: type[T]) -> T:
    return output.model_validate_json(_json_text(content or ""))


class ChatCompletionsLLM:
    def __init__(self, client: AsyncOpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._limit = asyncio.Semaphore(settings.max_parallel_llm_calls)
        self.model = settings.chat_model
        # Reasoning efforts this server refused: each takes its own values (Groq: gpt-oss low, medium or high;
        # qwen none or default), so a refused one is not sent again.
        self._refused_efforts: set[str] = set()

    async def _send(self, options: dict[str, Any]) -> Any:
        async with self._limit:
            return await self._client.chat.completions.create(**options)

    async def _request(self, call: AgentCall[Any], options: dict[str, Any]) -> Any:
        """One request. The reasoning effort is a nudge: a server that answers 400 about it is asked again
        without it (and not sent that value again). Any other failure is told in plain words."""
        if options.get("reasoning_effort") in self._refused_efforts:
            options = _without_effort(options)
        try:
            return await self._send(options)
        except openai.APIError as exc:
            if not _refuses_effort(exc, options):
                raise provider_failure(exc, call.agent) from exc
        effort = options["reasoning_effort"]
        self._refused_efforts.add(effort)
        log.info("reasoning_effort_dropped", extra={"fields": {"agent": call.agent, "effort": effort}})
        try:
            return await self._send(_without_effort(options))
        except openai.APIError as exc:
            raise provider_failure(exc, call.agent) from exc

    async def _create(self, call: AgentCall[Any], messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        options: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # Gateways apply a low default (Yantra: 3,072, reasoning included) that cuts long forms off.
            "max_completion_tokens": self._settings.max_output_tokens,
            **kwargs,
        }
        if self._settings.temperature is not None:
            options["temperature"] = self._settings.temperature
        response = await self._request(call, options)
        if response.choices[0].finish_reason == LENGTH:
            # A correction would be cut off at the same point, so this fails at once.
            log.warning("llm_answer_truncated", extra={"fields": {"agent": call.agent}})
            raise UpstreamError(
                f"The AI answer for {call.agent} was longer than the output limit (RE_AI_MAX_OUTPUT_TOKENS)."
            )
        return response

    async def _answer[T: BaseModel](
        self, call: AgentCall[T], messages: list[dict[str, Any]], usage: Usage, **kwargs: Any
    ) -> T:
        """Asks for the final answer, sending a wrong one back to be corrected (MAX_REPAIRS times)."""
        for attempt in range(MAX_REPAIRS + 1):
            response = await self._create(call, messages, **kwargs)
            usage.add(_usage(response))
            content = response.choices[0].message.content
            try:
                return _validate(content, call.output)
            except ValidationError as exc:
                log.info(
                    "llm_answer_invalid",
                    extra={"fields": {"agent": call.agent, "attempt": attempt, "errors": exc.error_count()}},
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": content or ""},
                    {
                        "role": "user",
                        "content": "That answer does not match the JSON schema: "
                        f"{exc.errors(include_url=False, include_input=False)}. "
                        "Reply again with the corrected JSON object only.",
                    },
                ]
        raise UpstreamError(f"The AI provider returned no usable answer for {call.agent}.")

    async def structured[T: BaseModel](self, call: AgentCall[T]) -> Completion[T]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system(call)},
            {"role": "user", "content": _user_content(call)},
        ]
        options: dict[str, Any] = {"response_format": {"type": "json_object"}}
        if call.effort:
            options["reasoning_effort"] = call.effort
        usage = Usage()
        return Completion(await self._answer(call, messages, usage, **options), usage)

    async def with_tools[T: BaseModel](
        self, call: AgentCall[T], tools: list[Tool], max_tool_calls: int
    ) -> Completion[T]:
        by_name = {t.name: t for t in tools}
        specs = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system(call)},
            {"role": "user", "content": call.input},
        ]
        usage = Usage()
        # Hard stop even if the model keeps asking for tools after they are switched off.
        for _ in range(max_tool_calls + 2):
            allowed = usage.tool_calls < max_tool_calls
            options: dict[str, Any] = {"tools": specs, "tool_choice": "auto" if allowed else "none"}
            response = await self._create(call, messages, reasoning_effort=NO_REASONING, **options)
            message = response.choices[0].message
            calls = message.tool_calls or []
            usage.add(_usage(response, len(calls)))
            if not calls:
                try:
                    return Completion(_validate(message.content, call.output), usage)
                except ValidationError:
                    messages.append({"role": "assistant", "content": message.content or ""})
                    messages.append(
                        {"role": "user", "content": "Reply with the JSON object only, matching the schema."}
                    )
                    options["tool_choice"] = "none"
                    value = await self._answer(
                        call, messages, usage, reasoning_effort=NO_REASONING, **options
                    )
                    return Completion(value, usage)
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                        }
                        for c in calls
                    ],
                }
            )
            for c in calls:
                output = await run_tool(by_name, c.function.name, c.function.arguments)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": output})
        raise UpstreamError(f"The AI provider did not finish {call.agent} within its tool budget.")
