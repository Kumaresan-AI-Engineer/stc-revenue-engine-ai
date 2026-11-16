"""Use-case: analyse an RFP (Step 1 → 2).

parse (code) → per window: extract (agent) → check quotes (code) → review features (agent)
→ one retry with the problems as notes → drop what still fails → merge windows (code) → cache.
"""

import asyncio
import hashlib
from dataclasses import dataclass

from agents import extraction, review
from agents.extraction import CatalogueFeature, Extraction, Rejection
from agents.review import ReviewItem
from api.schemas import AnalyseResponse, AnalyseResult, Problem
from core.config import Settings
from core.security import ServiceContext
from documents.parsing import Page, extract_pages, page_windows
from llm.ports import LLM, Usage
from rules.text import dedupe_key, stable_hash
from services.common import meta, notes_from
from store.ports import ResultCache

# ~15k tokens of document per call: long RFPs are read in several windows, in parallel.
WINDOW_CHARS = 60_000
VERSIONS = {"extraction": extraction.PROMPT_VERSION, "review": review.PROMPT_VERSION}
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


async def analyse(
    ctx: ServiceContext,
    filename: str,
    data: bytes,
    catalogue: list[CatalogueFeature],
    llm: LLM,
    cache: ResultCache,
    settings: Settings,
) -> AnalyseResponse:
    key = stable_hash(
        "analyse", hashlib.sha256(data).hexdigest(), [c.model_dump() for c in catalogue], llm.model, VERSIONS
    )
    if cached := await cache.get(ctx.workspace_id, key):
        return AnalyseResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    windows = page_windows(extract_pages(filename, data), WINDOW_CHARS)
    parts = await asyncio.gather(*(_window(llm, catalogue, w, settings) for w in windows))
    usage = Usage()
    for part in parts:
        usage.add(part.usage)
    merged = extraction.merge([p.extraction for p in parts])
    result = AnalyseResult(
        **merged.model_dump(),
        rejected=[Problem(item=r.item, reason=r.reason) for p in parts for r in p.rejected],
    )
    await cache.put(ctx.workspace_id, key, "analyse", result.model_dump())
    return AnalyseResponse(**result.model_dump(), meta=meta(llm.model, VERSIONS, usage, cached=False))
