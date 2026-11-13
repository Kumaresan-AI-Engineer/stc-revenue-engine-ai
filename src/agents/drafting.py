"""Response Drafting Agent: answers one RFP question from our past proposals, with tools.

It may search only the past proposals the user selected, in its own workspace, and read their
passages. Every claim must name a passage it actually retrieved and quote it; `check` verifies that.
Pricing questions never reach this agent (see services/drafting.py).
"""

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from core.config import ReasoningEffort
from llm.ports import LLM, AgentCall, Completion, Tool
from rules.text import contains_quote, mentions_amount
from store.ports import Hit, StoredPassage

PROMPT_VERSION = "drafting-v1"
MAX_TOOL_CALLS = 5
SEARCH_RESULTS = 5

INSTRUCTIONS = """You draft one answer to an RFP question for our bid team, using only our past
proposals.

How to work:
1. Call search_library with short queries about the question. If the results are weak, try other
   wording. Call get_passage to read a passage in full.
2. Write the answer only from passages you retrieved. Never invent facts, names, numbers,
   certifications, clients or dates.
3. Put every factual statement of the answer in `claims`, each with the passage_id it comes from and
   a quote copied exactly from that passage.
4. Never write prices, rates, fees or any other amounts; a person writes those.
5. If the passages do not answer the question, set enough_material to false and leave answer empty.

Write clear, confident business English in the first person plural ("we"), 80 to 200 words."""

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "A short search phrase."}},
    "required": ["query"],
    "additionalProperties": False,
}
_PASSAGE_SCHEMA = {
    "type": "object",
    "properties": {"passage_id": {"type": "string"}},
    "required": ["passage_id"],
    "additionalProperties": False,
}


class Claim(BaseModel):
    statement: str
    passage_id: str
    quote: str


class DraftOutput(BaseModel):
    answer: str
    claims: list[Claim]
    enough_material: bool


@dataclass(frozen=True)
class QuestionContext:
    question: str
    rfp_summary: str
    requirements: list[str]


@dataclass
class Library:
    """The agent's window on the library: selected entries only, and a record of what it read."""

    workspace_id: uuid.UUID
    entry_ids: list[str]
    entry_labels: dict[str, str]
    search: Callable[[str], Awaitable[list[Hit]]]
    fetch: Callable[[str], Awaitable[StoredPassage | None]]
    retrieved: dict[str, StoredPassage] = field(default_factory=dict)

    def _view(self, p: StoredPassage, similarity: float | None = None) -> dict[str, object]:
        view: dict[str, object] = {
            "passage_id": p.id,
            "proposal": self.entry_labels.get(p.entry_id, p.entry_id),
            "page": p.page,
            "text": p.text,
        }
        if similarity is not None:
            view["similarity"] = similarity
        return view

    async def search_tool(self, args: dict[str, object]) -> str:
        hits = await self.search(str(args.get("query", "")))
        for h in hits:
            self.retrieved[h.passage.id] = h.passage
        return json.dumps([self._view(h.passage, h.similarity) for h in hits], ensure_ascii=False)

    async def passage_tool(self, args: dict[str, object]) -> str:
        passage = await self.fetch(str(args.get("passage_id", "")))
        if passage is None or passage.entry_id not in self.entry_ids:
            return "No such passage in the selected proposals."
        self.retrieved[passage.id] = passage
        return json.dumps(self._view(passage), ensure_ascii=False)

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "search_library",
                "Search the selected past proposals. Returns the most similar passages.",
                _QUERY_SCHEMA,
                self.search_tool,
            ),
            Tool(
                "get_passage",
                "Read one passage in full by its passage_id.",
                _PASSAGE_SCHEMA,
                self.passage_tool,
            ),
        ]


def _input(ctx: QuestionContext, notes: list[str]) -> str:
    parts = [
        f"RFP question: {ctx.question}",
        f"What the client wants: {ctx.rfp_summary or 'not given'}",
        "Requirements in scope: " + ("; ".join(ctx.requirements) or "not given"),
    ]
    if notes:
        parts.append("Your previous answer had problems. Fix them:\n" + "\n".join(f"- {n}" for n in notes))
    return "\n\n".join(parts)


async def draft(
    llm: LLM, ctx: QuestionContext, library: Library, notes: list[str], effort: ReasoningEffort | None
) -> Completion[DraftOutput]:
    call = AgentCall("drafting", INSTRUCTIONS, _input(ctx, notes), DraftOutput, effort)
    return await llm.with_tools(call, library.tools(), MAX_TOOL_CALLS)


def check(output: DraftOutput, retrieved: dict[str, StoredPassage]) -> tuple[list[Claim], list[str]]:
    """Grounded claims, and the problems found (empty when the draft can go to review)."""
    problems: list[str] = []
    claims: list[Claim] = []
    for claim in output.claims:
        passage = retrieved.get(claim.passage_id)
        if passage is None:
            problems.append(f"Claim '{claim.statement}' cites a passage you did not retrieve.")
        elif not contains_quote(passage.text, claim.quote):
            problems.append(f"The quote for '{claim.statement}' is not in passage {claim.passage_id}.")
        else:
            claims.append(claim)
    if mentions_amount(output.answer):
        problems.append("The answer contains an amount; remove every price, rate or fee.")
    if output.enough_material and output.answer.strip() and not claims:
        problems.append("The answer has no claim backed by a passage.")
    return claims, problems
