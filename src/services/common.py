"""Shared pieces of the use-cases: result metadata and retry notes."""

from api.schemas import Meta, UsageOut
from llm.ports import Usage


def meta(model: str, versions: dict[str, str], usage: Usage, cached: bool) -> Meta:
    return Meta(
        model=model,
        prompt_versions=versions,
        usage=UsageOut(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            calls=usage.calls,
            tool_calls=usage.tool_calls,
        ),
        cached=cached,
    )


def notes_from(problems: dict[str, str]) -> list[str]:
    return [f"{item}: {reason}" for item, reason in problems.items()]
