"""Final Proposal Check Agent: reads the user's own final proposal and reports what it states, so it can
be compared with the draft it should match.

The model only finds facts, each with a quote; code keeps a fact only when its quote is in the document
(and a figure only when the number is in its quote), then writes the five findings itself. The texts are
fixed sentences with counts, so no money figure is ever written.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from revenue_engine_ai.core.config import ReasoningEffort
from revenue_engine_ai.documents.parsing import Page, render_pages
from revenue_engine_ai.llm.ports import LLM, AgentCall, Completion
from revenue_engine_ai.rules.text import contains_quote, dedupe_key

PROMPT_VERSION = "final-check-v1"

INSTRUCTIONS = """You read a final proposal document and report what it states, so a bid team can check it
against the draft it came from.

Rules:
- Use only the document text. Never guess, compute or convert anything.
- Every item needs `quote`: words copied exactly from the document, and `page`: the number in the
  [Page N] marker above the quoted text.
- price: the proposal's total price as stated, as a whole number without currency or separators; null
  when the document states none or when you are told not to report it.
- weeks: the total delivery duration only when the document states it in weeks; otherwise null.
- features: keys from the list below whose capability the document says it will deliver.
- answers: for each approved response id below that the document answers, whether the document's answer
  says the same thing as the approved response (`matches`), quoting the document's answer.
- attachments: names from the list below that the document mentions as attached, enclosed or included.
Return empty lists for what the document does not contain."""

Area = Literal["Scope", "Cost", "Timeline", "Responses", "Attachments"]


class Quoted(BaseModel):
    quote: str
    page: int


class Figure(Quoted):
    value: int


class FoundFeature(Quoted):
    key: str


class FoundAnswer(Quoted):
    id: str
    matches: bool


class FoundAttachment(Quoted):
    name: str


class FinalFacts(BaseModel):
    price: Figure | None
    weeks: Figure | None
    features: list[FoundFeature]
    answers: list[FoundAnswer]
    attachments: list[FoundAttachment]


@dataclass(frozen=True)
class DraftSnapshot:
    price: int | None
    weeks: int | None
    core_features: dict[str, str]  # key -> label
    responses: list[tuple[str, str, str]]  # (id, question, approved response)
    attachments: list[str]


@dataclass(frozen=True)
class Finding:
    area: Area
    ok: bool
    text: str
    from_: int | None = None
    to: int | None = None


def _input(pages: list[Page], draft: DraftSnapshot, report_price: bool) -> str:
    lines = [render_pages(pages), "", "Core features (key: label):"]
    lines += [f"- {k}: {label}" for k, label in draft.core_features.items()] or ["- (none)"]
    lines += ["", "Approved responses (id: question -> response):"]
    lines += [f"- {i}: {q} -> {r}" for i, q, r in draft.responses] or ["- (none)"]
    lines += ["", "Attachments:"] + ([f"- {a}" for a in draft.attachments] or ["- (none)"])
    lines += ["", f"Report the price: {'yes' if report_price else 'no'}."]
    return "\n".join(lines)


async def read(
    llm: LLM, pages: list[Page], draft: DraftSnapshot, report_price: bool, effort: ReasoningEffort | None
) -> Completion[FinalFacts]:
    call = AgentCall("final_check", INSTRUCTIONS, _input(pages, draft, report_price), FinalFacts, effort)
    return await llm.structured(call)


_NUMBER = re.compile(r"(\d[\d,.' ]*\d|\d)\s*(k|m|thousand|million)?\b", re.IGNORECASE)
_SCALE = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000}


def numbers_in(text: str) -> set[int]:
    """Whole numbers written in a text, with thousands separators and a k / m scale understood."""
    found = set()
    for digits, scale in _NUMBER.findall(text):
        compact = re.sub(r"[,' ]", "", digits)
        parts = compact.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and not scale):
            value = float("".join(parts))  # dots as thousands separators: 820.000, 1.200.000
        else:
            value = float(compact)
        found.add(round(value * _SCALE.get(scale.lower(), 1)))
    return found


def _in_document(item: Quoted, pages: list[Page]) -> bool:
    return any(contains_quote(p.text, item.quote) for p in pages)


def check(facts: FinalFacts, pages: list[Page], draft: DraftSnapshot, report_price: bool) -> FinalFacts:
    """Keep only facts whose quote is in the document and that name what the draft has."""
    price = facts.price if report_price else None
    if price and not (_in_document(price, pages) and price.value in numbers_in(price.quote)):
        price = None
    weeks = facts.weeks
    if weeks and not (
        _in_document(weeks, pages)
        and weeks.value in numbers_in(weeks.quote)
        and "week" in weeks.quote.lower()
    ):
        weeks = None
    response_ids = {i for i, _, _ in draft.responses}
    attachments = {dedupe_key(a): a for a in draft.attachments}
    return FinalFacts(
        price=price,
        weeks=weeks,
        features=[f for f in facts.features if f.key in draft.core_features and _in_document(f, pages)],
        answers=[a for a in facts.answers if a.id in response_ids and _in_document(a, pages)],
        attachments=[
            a.model_copy(update={"name": attachments[dedupe_key(a.name)]})
            for a in facts.attachments
            if dedupe_key(a.name) in attachments and _in_document(a, pages)
        ],
    )


def merge(parts: list[FinalFacts]) -> FinalFacts:
    """Windows of one long document: the first figure found wins, lists are joined without repeats."""

    def unique(items: list[Any], key: str) -> list[Any]:
        out: dict[str, Any] = {}
        for item in items:
            out.setdefault(getattr(item, key), item)
        return list(out.values())

    return FinalFacts(
        price=next((p.price for p in parts if p.price), None),
        weeks=next((p.weeks for p in parts if p.weeks), None),
        features=unique([f for p in parts for f in p.features], "key"),
        answers=unique([a for p in parts for a in p.answers], "id"),
        attachments=unique([a for p in parts for a in p.attachments], "name"),
    )


def _count(area: Area, found: int, total: int, text: str, none: str) -> Finding:
    if not total:
        return Finding(area, True, none)
    return Finding(area, found == total, text.format(found=found, total=total))


def findings(facts: FinalFacts, draft: DraftSnapshot, report_price: bool) -> list[Finding]:
    """The five findings, always in this order: Scope, Cost, Timeline, Responses, Attachments."""
    scope = _count(
        "Scope",
        len({f.key for f in facts.features}),
        len(draft.core_features),
        "{found} of {total} core features found in the file",
        "The draft has no core features to look for",
    )
    if not report_price:
        cost = Finding("Cost", True, "Not compared: prices are restricted for your role.")
    elif facts.price is None:
        cost = Finding("Cost", False, "Price not found in the file", draft.price, None)
    elif facts.price.value == draft.price:
        cost = Finding("Cost", True, "Same as the draft", draft.price, facts.price.value)
    else:
        cost = Finding(
            "Cost", False, "Price in the file differs from the draft", draft.price, facts.price.value
        )
    if facts.weeks is None:
        timeline = Finding("Timeline", False, "Timeline (weeks) not found in the file", draft.weeks, None)
    elif facts.weeks.value == draft.weeks:
        timeline = Finding("Timeline", True, "Same as the draft", draft.weeks, facts.weeks.value)
    else:
        timeline = Finding(
            "Timeline", False, "Timeline in the file differs from the draft", draft.weeks, facts.weeks.value
        )
    responses = _count(
        "Responses",
        sum(1 for a in facts.answers if a.matches),
        len(draft.responses),
        "{found} of {total} RFP answers match the approved responses",
        "No approved responses to compare",
    )
    attachments = _count(
        "Attachments",
        len({a.name for a in facts.attachments}),
        len(draft.attachments),
        "{found} of {total} attachments are referenced",
        "No attachments to compare",
    )
    return [scope, cost, timeline, responses, attachments]
