"""What the agents need from a model provider. Agents depend on these interfaces only, so the
provider (OpenAI today) can be replaced and tests can run without network calls."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from core.config import ReasoningEffort


@dataclass(frozen=True)
class Tool:
    """A function the model may call. `parameters` is a strict JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class AgentCall[T: BaseModel]:
    agent: str  # stable name: groups prompt caching and logs
    instructions: str
    input: str
    output: type[T]
    effort: ReasoningEffort | None
    # Images the model looks at with the input, as data URLs (data:image/png;base64,...). Structured calls
    # only; a vision-capable model is needed.
    images: tuple[str, ...] = ()


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    tool_calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.tool_calls += other.tool_calls


@dataclass(frozen=True)
class Completion[T: BaseModel]:
    value: T
    usage: Usage


class LLM(Protocol):
    model: str

    async def structured[T: BaseModel](self, call: AgentCall[T]) -> Completion[T]:
        """One call whose answer must match call.output exactly."""
        ...

    async def with_tools[T: BaseModel](
        self, call: AgentCall[T], tools: list[Tool], max_tool_calls: int
    ) -> Completion[T]:
        """Let the model call tools (at most max_tool_calls), then answer in call.output."""
        ...


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        ...
