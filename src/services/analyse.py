"""Use-case: analyse an RFP (Step 1 → 2).

parse, with OCR for scanned pages (code) → per window: extract (agent) → check quotes (code) → review
features (agent) → a retry with notes only for misquotes → drop what still fails → merge windows (code)
→ map to the contract (core / standard / unmatched / facts) → cache.
"""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agents import extraction, review
from agents.extraction import CatalogueFeature, Extraction, Rejection
from agents.review import ReviewItem
from api.schemas import (
    AnalyseResponse,
    AnalyseResult,
    CoreFeature,
    Problem,
    RequiredDocumentOut,
    RfpQuestionOut,
    StandardEvidence,
    UnclearPointOut,
    Unmatched,
)
from core.config import Settings
from core.security import ServiceContext
from documents.parsing import Page, ReadDocument, page_windows, read_document
from llm.ports import LLM, Usage
from rules.text import dedupe_key, stable_hash
from services.common import meta, notes_from
from store.ports import ResultCache

# ~15k tokens of document per call: long RFPs are read in several windows, in parallel.
WINDOW_CHARS = 60_000
VERSIONS = {"extraction": extraction.PROMPT_VERSION, "review": review.PROMPT_VERSION}
# Steps reported to a job's progress callback: reading, finding requirements, checking them.
STEPS = 3

# (message, done, total) -> reported to whoever waits (a job's progress callback); may be a no-op.
Progress = Callable[[str, int, int], Awaitable[None]]


async def _no_progress(message: str, done: int, total: int) -> None:
    return None


SCOPE_RULE = "The evidence must show that the client asks for this capability."


@dataclass
class _WindowResult:
    extraction: Extraction
    rejected: list[Rejection]
    usage: Usage


def _review_items(result: Extraction, labels: dict[str, str]) -> list[ReviewItem]:
    """Scope drives matching and the estimate, so features and custom requirements get a second look."""
    items = [
        ReviewItem(
            id=f"feature:{f.key}",
            statement=f"The client asks for: {labels[f.key]}",
            evidence=[f.quote],
            rule=SCOPE_RULE,
        )
        for f in result.features
    ]
    items += [
        ReviewItem(
            id=f"custom:{dedupe_key(c.label)}",
            statement=f"The client asks for: {c.label}",
            evidence=[c.quote],
            rule=SCOPE_RULE,
        )
        for c in result.custom_requirements
    ]
    return items


def _drop(result: Extraction, failed: dict[str, str]) -> Extraction:
    return result.model_copy(
        update={
            "features": [f for f in result.features if f"feature:{f.key}" not in failed],
            "custom_requirements": [
                c for c in result.custom_requirements if f"custom:{dedupe_key(c.label)}" not in failed
            ],
        }
    )


async def _attempt(
    llm: LLM,
    catalogue: list[CatalogueFeature],
    window: list[Page],
    notes: list[str],
    settings: Settings,
    usage: Usage,
) -> tuple[Extraction, dict[str, str], dict[str, str]]:
    """One extraction, checked and reviewed: (result, misquoted items, items the reviewer rejected)."""
    done = await extraction.extract(llm, catalogue, window, notes, settings.extraction_effort)
    usage.add(done.usage)
    checked, rejected = extraction.check(done.value, window, {f.key for f in catalogue})
    labels = {f.key: f.label for f in catalogue}
    reviewed = await review.review(llm, _review_items(checked, labels), settings.review_effort)
    usage.add(reviewed.usage)
    return _drop(checked, reviewed.failed), {r.item: r.reason for r in rejected}, reviewed.failed


async def _window(
    llm: LLM, catalogue: list[CatalogueFeature], window: list[Page], settings: Settings
) -> _WindowResult:
    usage = Usage()
    result, misquoted, refused = await _attempt(llm, catalogue, window, [], settings, usage)
    # Only a misquote can be fixed by asking again (the model finds the right words). The reviewer's
    # verdicts are judgments: those items are already dropped, and a whole second extraction to repeat
    # them doubled the latency without improving the result.
    if misquoted:
        result, misquoted, refused = await _attempt(
            llm, catalogue, window, notes_from(misquoted), settings, usage
        )
    problems = misquoted | refused
    return _WindowResult(result, [Rejection(item, reason) for item, reason in problems.items()], usage)


def to_result(
    merged: Extraction, rejected: list[Problem], catalogue: list[CatalogueFeature], doc: ReadDocument
) -> AnalyseResult:
    """The contract's shape: core vs standard by the catalogue's commodity flag, facts by name."""
    standard_keys = {f.key for f in catalogue if f.standard}
    facts = {f.name: f.value.strip() or None for f in merged.facts}
    return AnalyseResult(
        summary=merged.summary,
        core=[
            CoreFeature(key=f.key, confidence=f.confidence, evidence=f.quote, page=f.page)
            for f in merged.features
            if f.key not in standard_keys
        ],
        standard=[f.key for f in merged.features if f.key in standard_keys],
        standard_evidence=[
            StandardEvidence(key=f.key, evidence=f.quote, page=f.page)
            for f in merged.features
            if f.key in standard_keys
        ],
        unmatched=[
            Unmatched(label=c.label, evidence=c.quote, page=c.page) for c in merged.custom_requirements
        ],
        due_date=facts.get("due_date"),
        start_date=facts.get("start_date"),
        budget_hint=facts.get("budget"),
        client=facts.get("client"),
        questions=[RfpQuestionOut(text=q.text, page=q.page, evidence=q.quote) for q in merged.questions],
        required_documents=[d.name for d in merged.required_documents],
        documents=[
            RequiredDocumentOut(name=d.name, evidence=d.quote, page=d.page) for d in merged.required_documents
        ],
        unclear_points=[
            UnclearPointOut(topic=u.topic, question=u.question, evidence=u.quote, page=u.page)
            for u in merged.unclear_points
        ],
        rejected=rejected,
        ocr=doc.ocr,
        pages=len(doc.pages),
    )


async def analyse(
    ctx: ServiceContext,
    filename: str,
    data: bytes,
    catalogue: list[CatalogueFeature],
    llm: LLM,
    cache: ResultCache,
    settings: Settings,
    progress: Progress = _no_progress,
) -> AnalyseResponse:
    key = stable_hash(
        "analyse", hashlib.sha256(data).hexdigest(), [c.model_dump() for c in catalogue], llm.model, VERSIONS
    )
    if cached := await cache.get(ctx.workspace_id, key):
        return AnalyseResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    await progress("Reading the document", 0, STEPS)
    # Parsing and OCR are CPU work: off the event loop.
    doc = await asyncio.to_thread(read_document, filename, data, ocr_max_pages=settings.ocr_max_pages)
    windows = page_windows(doc.pages, WINDOW_CHARS)
    await progress(f"Finding requirements in {len(doc.pages)} pages", 1, STEPS)
    parts = await asyncio.gather(*(_window(llm, catalogue, w, settings) for w in windows))
    await progress("Putting the results together", 2, STEPS)
    usage = Usage()
    for part in parts:
        usage.add(part.usage)
    merged = extraction.merge([p.extraction for p in parts])
    rejected = [Problem(item=r.item, reason=r.reason) for p in parts for r in p.rejected]
    result = to_result(merged, rejected, catalogue, doc)
    await cache.put(ctx.workspace_id, key, "analyse", result.model_dump())
    return AnalyseResponse(**result.model_dump(), meta=meta(llm.model, VERSIONS, usage, cached=False))
