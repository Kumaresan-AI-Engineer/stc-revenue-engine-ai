"""Review Agent: an independent second look at another agent's output.

It sees only a statement, its evidence and the rule to apply, never the other agent's reasoning,
and answers supported / not supported for each item. An item it does not rule on fails (deny by
default), so a skipped verdict can never let an unsupported statement through.
"""

import asyncio
import json
from dataclasses import dataclass

from pydantic import BaseModel

from core.config import ReasoningEffort
from llm.ports import LLM, AgentCall, Usage

PROMPT_VERSION = "review-v1"
BATCH_SIZE = 20

INSTRUCTIONS = """You are a strict reviewer for a bid team. For each item you get a statement, the
evidence it is based on, and a rule.

Decide `supported`:
- true only when the evidence alone fully supports the statement and the statement follows the rule;
- false when anything in the statement is missing from, stronger than, or different from the
  evidence, or when the rule is broken.
Use no outside knowledge. Do not rewrite anything. Give `reason` as one short sentence in plain words
that the author can act on. Return exactly one verdict per item id."""


class ReviewItem(BaseModel):
    id: str
    statement: str
    evidence: list[str]
    rule: str


class Verdict(BaseModel):
    id: str
    supported: bool
    reason: str


class ReviewOutput(BaseModel):
    verdicts: list[Verdict]


@dataclass(frozen=True)
class ReviewResult:
    failed: dict[str, str]  # item id -> reason
    usage: Usage


async def _batch(
    llm: LLM, items: list[ReviewItem], effort: ReasoningEffort | None
) -> tuple[dict[str, str], Usage]:
    payload = json.dumps([i.model_dump() for i in items], ensure_ascii=False)
    done = await llm.structured(AgentCall("review", INSTRUCTIONS, payload, ReviewOutput, effort))
    verdicts = {v.id: v for v in done.value.verdicts}
    failed = {}
    for item in items:
        verdict = verdicts.get(item.id)
        if verdict is None:
            failed[item.id] = "The reviewer gave no verdict."
        elif not verdict.supported:
            failed[item.id] = verdict.reason
    return failed, done.usage


async def review(llm: LLM, items: list[ReviewItem], effort: ReasoningEffort | None) -> ReviewResult:
    usage = Usage()
    failed: dict[str, str] = {}
    batches = [items[i : i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    for part_failed, part_usage in await asyncio.gather(*(_batch(llm, b, effort) for b in batches)):
        failed.update(part_failed)
        usage.add(part_usage)
    return ReviewResult(failed, usage)
