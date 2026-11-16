"""Use-case: check the user's own final proposal against the draft (Step 8 Submit).

parse the file(s) (code, OCR for scans) → per window: read facts with quotes (agent) → keep only facts
whose quote is in the document (code) → merge → five findings (code) → cache. A price is read and compared
only when the caller may see restricted values (the JWT's can_see_restricted).
"""

import asyncio
import hashlib

from agents import final_check as agent
from agents.final_check import DraftSnapshot, FinalFacts
from api.schemas import DraftSnapshotIn, FinalCheckResponse, FindingOut
from core.config import Settings
from core.security import ServiceContext
from documents.parsing import Page, page_windows, read_document
from llm.ports import LLM, Usage
from rules.text import stable_hash
from services.analyse import WINDOW_CHARS
from services.common import meta
from store.ports import ResultCache

VERSIONS = {"final_check": agent.PROMPT_VERSION}


def _snapshot(draft: DraftSnapshotIn) -> DraftSnapshot:
    return DraftSnapshot(
        price=draft.price,
        weeks=draft.weeks,
        core_features={f.key: f.label for f in draft.core_features},
        responses=[(r.id, r.q, r.response) for r in draft.responses if r.response.strip()],
        attachments=list(dict.fromkeys(draft.attachments)),
    )


def _pages(files: list[tuple[str, bytes]], settings: Settings) -> list[Page]:
    """All files as one document, pages numbered on (a technical then a commercial file read as one)."""
    pages: list[Page] = []
    for name, data in files:
        for page in read_document(name, data, ocr_max_pages=settings.ocr_max_pages).pages:
            pages.append(Page(len(pages) + 1, page.text))
    return pages


async def _read(
    llm: LLM, window: list[Page], draft: DraftSnapshot, report_price: bool, settings: Settings
) -> tuple[FinalFacts, Usage]:
    done = await agent.read(llm, window, draft, report_price, settings.final_check_effort)
    return agent.check(done.value, window, draft, report_price), done.usage


async def final_check(
    ctx: ServiceContext,
    files: list[tuple[str, bytes]],
    draft_in: DraftSnapshotIn,
    llm: LLM,
    cache: ResultCache,
    settings: Settings,
) -> FinalCheckResponse:
    report_price = ctx.can_see_restricted
    draft = _snapshot(draft_in)
    key = stable_hash(
        "final_check",
        [(n, hashlib.sha256(d).hexdigest()) for n, d in files],
        draft_in.model_dump(),
        report_price,
        llm.model,
        VERSIONS,
    )
    if cached := await cache.get(ctx.workspace_id, key):
        return FinalCheckResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    pages = await asyncio.to_thread(_pages, files, settings)
    windows = page_windows(pages, WINDOW_CHARS)
    parts = await asyncio.gather(*(_read(llm, w, draft, report_price, settings) for w in windows))
    usage = Usage()
    for _, part_usage in parts:
        usage.add(part_usage)
    facts = agent.merge([f for f, _ in parts])
    response = FinalCheckResponse(
        files=[n for n, _ in files],
        checked=True,
        price=facts.price.value if facts.price and report_price else None,
        weeks=facts.weeks.value if facts.weeks else None,
        draft_price=draft.price if report_price else None,
        draft_weeks=draft.weeks,
        findings=[
            FindingOut.model_validate(
                {"area": f.area, "ok": f.ok, "text": f.text, "from": f.from_, "to": f.to}
            )
            for f in agent.findings(facts, draft, report_price)
        ],
        meta=meta(llm.model, VERSIONS, usage, cached=False),
    )
    await cache.put(
        ctx.workspace_id, key, "final_check", response.model_dump(exclude={"meta"}, by_alias=True)
    )
    return response
