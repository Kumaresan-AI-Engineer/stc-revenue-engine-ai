"""OpenAI implementation of the LLM port (Responses API + structured outputs)."""

import asyncio
import json
import logging
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from core.config import Settings
from core.errors import UpstreamError
from llm.ports import AgentCall, Completion, Tool, Usage

log = logging.getLogger(__name__)


def _usage(response: Any, tool_calls: int = 0) -> Usage:
    u = response.usage
    return Usage(u.input_tokens if u else 0, u.output_tokens if u else 0, 1, tool_calls)


def _input(call: AgentCall[Any]) -> str | list[dict[str, Any]]:
    """The input as text, or as one user message of text and image parts when the call carries images."""
    if not call.images:
        return call.input
    images = [{"type": "input_image", "image_url": url} for url in call.images]
    return [{"role": "user", "content": [{"type": "input_text", "text": call.input}, *images]}]


class OpenAILLM:
    def __init__(self, client: AsyncOpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._limit = asyncio.Semaphore(settings.max_parallel_llm_calls)
        self.model = settings.chat_model

    def _options(self, call: AgentCall[Any]) -> dict[str, Any]:
        options: dict[str, Any] = {
            "model": self.model,
            "instructions": call.instructions,
            # Nothing is kept at OpenAI; reasoning is carried between tool turns encrypted instead.
            "store": False,
            # Same instructions → same cache bucket, so the long fixed prefix is billed at cache rates.
            "prompt_cache_key": f"revenue-engine-ai:{call.agent}",
        }
        if call.effort:
            options["reasoning"] = {"effort": call.effort}
            options["include"] = ["reasoning.encrypted_content"]
        if self._settings.temperature is not None:
            options["temperature"] = self._settings.temperature
        return options

    async def _parse[T: BaseModel](self, call: AgentCall[T], **kwargs: Any) -> Any:
        try:
            async with self._limit:
                return await self._client.responses.parse(
                    text_format=call.output, **self._options(call), **kwargs
                )
        except openai.APIError as exc:
            log.warning(
                "llm_call_failed", extra={"fields": {"agent": call.agent, "error": type(exc).__name__}}
            )
            raise UpstreamError("The AI provider did not answer. Try again in a moment.") from exc

    @staticmethod
    def _value[T: BaseModel](response: Any, call: AgentCall[T]) -> T:
        parsed = response.output_parsed
        if parsed is None:
            raise UpstreamError(f"The AI provider returned no usable answer for {call.agent}.")
        return parsed  # type: ignore[no-any-return]

    async def structured[T: BaseModel](self, call: AgentCall[T]) -> Completion[T]:
        response = await self._parse(call, input=_input(call))
        return Completion(self._value(response, call), _usage(response))

    async def with_tools[T: BaseModel](
        self, call: AgentCall[T], tools: list[Tool], max_tool_calls: int
    ) -> Completion[T]:
        by_name = {t.name: t for t in tools}
        specs = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": True,
            }
            for t in tools
        ]
        items: list[dict[str, Any]] = [{"role": "user", "content": call.input}]
        usage = Usage()
        # Hard stop even if the model keeps asking for tools after they are switched off.
        for _ in range(max_tool_calls + 2):
            allowed = usage.tool_calls < max_tool_calls
            response = await self._parse(
                call, input=items, tools=specs, tool_choice="auto" if allowed else "none"
            )
            calls = [o for o in response.output if o.type == "function_call"]
            usage.add(_usage(response, len(calls)))
            if not calls:
                return Completion(self._value(response, call), usage)
            for item in response.output:
                if item.type == "reasoning":
                    items.append(item.to_dict(exclude_none=True))
                elif item.type == "function_call":
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                    )
            for c in calls:
                output = await run_tool(by_name, c.name, c.arguments)
                items.append({"type": "function_call_output", "call_id": c.call_id, "output": output})
        raise UpstreamError(f"The AI provider did not finish {call.agent} within its tool budget.")


async def run_tool(tools: dict[str, Tool], name: str, arguments: str | None) -> str:
    """Runs the tool the model asked for; a bad request gets an error text the model can act on."""
    tool = tools.get(name)
    if tool is None:
        return f"Unknown tool '{name}'."
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return "The arguments were not valid JSON."
    return await tool.run(args)
