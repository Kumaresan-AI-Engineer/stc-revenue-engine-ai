"""Requirement Extraction Agent: reads one window of RFP pages and fills a fixed form.

The feature key is an enum of the catalogue the backend sent, so the model cannot invent one.
Every item carries a quote and page, which `check` verifies against the document text.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, create_model

from core.config import ReasoningEffort
from documents.parsing import Page, render_pages
from llm.ports import LLM, AgentCall, Completion
from rules.text import contains_quote, dedupe_key

PROMPT_VERSION = "extraction-v2"

INSTRUCTIONS = """You read part of a Request for Proposal (RFP) and fill in a form for our bid team.

Rules:
- Use only the document text you are given. Never use outside knowledge and never guess.
- Every item needs `quote`: words copied exactly from the document (you may skip words with "...")
  and `page`: the number in the [Page N] marker above the quoted text.
- features: catalogue capabilities the client asks for. Use only keys from the catalogue below and
  pick one only when the document clearly asks for that capability. `confidence` (0 to 1): how plainly
  the quote asks for exactly that capability (1 = named outright, 0.5 = implied).
- custom_requirements: capabilities the client asks for that no catalogue entry covers, each with a
  short plain-words label.
- questions: questions or response items the RFP asks the bidder to answer, in the RFP's own words.
- required_documents: documents or certificates the bidder must attach.
- unclear_points: what a bid team cannot estimate without asking the client (missing volumes,
  vague scope, conflicting dates). Write `question` as a short, polite question to the client.
- facts: only facts the document states: client, due_date (YYYY-MM-DD), start_date, budget.
- summary: two or three plain sentences on what the client wants.
Return an empty list for anything this part does not contain.

Catalogue (key: label — words that suggest it):
"""

FactName = Literal["client", "due_date", "start_date", "budget"]


class CatalogueFeature(BaseModel):
    """One catalogue entry, as the backend sends it ({key, label, commodity, kw})."""

    model_config = ConfigDict(populate_by_name=True)

    key: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=200)
    keywords: list[str] = Field(default_factory=list, validation_alias=AliasChoices("kw", "keywords"))
    # Commodity feature (login, notifications, ...): kept in scope, not used for matching.
    standard: bool = Field(False, validation_alias=AliasChoices("commodity", "standard"))


class Quoted(BaseModel):
    quote: str
    page: int


class CustomRequirement(Quoted):
    label: str


class RfpQuestion(Quoted):
    text: str


class RequiredDocument(Quoted):
    name: str


class UnclearPoint(Quoted):
    topic: str
    question: str


class Fact(Quoted):
    name: FactName
    value: str


class FeatureHit(Quoted):
    key: str
    confidence: float = 0.5


class Extraction(BaseModel):
    """The checked, merged result (keys are plain strings once validated)."""

    summary: str
    features: list[FeatureHit]
    custom_requirements: list[CustomRequirement]
    questions: list[RfpQuestion]
    required_documents: list[RequiredDocument]
    unclear_points: list[UnclearPoint]
    facts: list[Fact]


@lru_cache(maxsize=32)
def output_model(keys: tuple[str, ...]) -> type[BaseModel]:
    """The form the model fills, with `features[].key` restricted to this catalogue."""
    key_type: Any = Literal[keys]  # type: ignore[valid-type]  # built from the catalogue at run time
    hit = create_model("CatalogueFeatureHit", __base__=Quoted, key=(key_type, ...), confidence=(float, ...))
    fields: dict[str, Any] = {
        name: (list[hit] if name == "features" else info.annotation, ...)  # type: ignore[valid-type]
        for name, info in Extraction.model_fields.items()
    }
    return create_model("ExtractionForm", **fields)


def instructions(catalogue: list[CatalogueFeature]) -> str:
    lines = [f"- {f.key}: {f.label} — {', '.join(f.keywords)}" for f in catalogue]
    return INSTRUCTIONS + "\n".join(lines)


async def extract(
    llm: LLM,
    catalogue: list[CatalogueFeature],
    window: list[Page],
    notes: list[str],
    effort: ReasoningEffort | None,
) -> Completion[Extraction]:
    text = render_pages(window)
    if notes:
        text += "\n\nYour previous answer had problems. Fix or drop these items:\n" + "\n".join(
            f"- {n}" for n in notes
        )
    form = output_model(tuple(f.key for f in catalogue))
    done = await llm.structured(AgentCall("extraction", instructions(catalogue), text, form, effort))
    return Completion(Extraction.model_validate(done.value.model_dump()), done.usage)


@dataclass(frozen=True)
class Rejection:
    item: str
    reason: str


def _locate(item: Quoted, pages: list[Page]) -> int | None:
    """The page that holds the quote: the cited one first, else any page (models misread markers)."""
    by_number = {p.number: p for p in pages}
    cited = by_number.get(item.page)
    if cited and contains_quote(cited.text, item.quote):
        return item.page
    return next((p.number for p in pages if contains_quote(p.text, item.quote)), None)


def check(
    result: Extraction, pages: list[Page], catalogue_keys: set[str]
) -> tuple[Extraction, list[Rejection]]:
    """Keep items whose quote is really in the document (fixing the page); reject the rest."""
    rejected: list[Rejection] = []

    def keep(kind: str, label: str, items: list[Any]) -> list[Any]:
        out = []
        for item in items:
            page = _locate(item, pages)
            if page is None:
                rejected.append(
                    Rejection(f"{kind} '{getattr(item, label)}'", "its quote is not in the document")
                )
            else:
                out.append(item.model_copy(update={"page": page}))
        return out

    features = [
        f.model_copy(update={"confidence": round(min(max(f.confidence, 0.0), 1.0), 2)})
        for f in result.features
        if f.key in catalogue_keys
    ]
    checked = Extraction(
        summary=result.summary.strip(),
        features=keep("feature", "key", features),
        custom_requirements=keep("custom requirement", "label", result.custom_requirements),
        questions=keep("question", "text", result.questions),
        required_documents=keep("required document", "name", result.required_documents),
        unclear_points=keep("unclear point", "topic", result.unclear_points),
        facts=keep("fact", "name", result.facts),
    )
    return checked, rejected


def merge(parts: list[Extraction]) -> Extraction:
    """Combine windows in page order; the first mention of a thing wins."""

    def unique(items: list[Any], key: str) -> list[Any]:
        seen: set[str] = set()
        out = []
        for item in items:
            k = dedupe_key(str(getattr(item, key)))
            if k not in seen:
                seen.add(k)
                out.append(item)
        return out

    def all_of(field: str) -> list[Any]:
        return [item for part in parts for item in getattr(part, field)]

    summaries = [p.summary for p in parts if p.summary]
    return Extraction(
        # The overview of an RFP is at its start, so the first window's summary describes the whole.
        summary=summaries[0] if summaries else "",
        features=unique(all_of("features"), "key"),
        custom_requirements=unique(all_of("custom_requirements"), "label"),
        questions=unique(all_of("questions"), "text"),
        required_documents=unique(all_of("required_documents"), "name"),
        unclear_points=unique(all_of("unclear_points"), "question"),
        facts=unique(all_of("facts"), "name"),
    )
