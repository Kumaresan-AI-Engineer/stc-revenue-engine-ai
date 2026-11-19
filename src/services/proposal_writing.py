"""Use-case: write the proposal's sections (Step 7 Proposal).

write (agent) → plain text, no money (code) → one retry for the sections left without a paragraph → any
still empty says MISSING → cache. The facts are the only source; nothing is retrieved.
"""

from agents import section_writer as agent
from agents.section_writer import SectionBrief
from api.schemas import WriteSectionsRequest, WriteSectionsResponse, WrittenSectionOut
from core.config import Settings
from core.security import ServiceContext
from llm.ports import LLM, Usage
from rules.text import stable_hash
from services.common import meta
from store.ports import ResultCache

VERSIONS = {"section_writer": agent.PROMPT_VERSION}
NOTHING_WRITTEN = "no usable paragraph was written for it; write it again without any money amounts."


async def _write(
    llm: LLM,
    briefs: list[SectionBrief],
    req: WriteSectionsRequest,
    notes: list[str],
    settings: Settings,
    usage: Usage,
) -> tuple[dict[str, list[str]], list[str]]:
    done = await agent.ask(llm, briefs, req.facts.model_dump(), notes, settings.section_writer_effort)
    usage.add(done.usage)
    return agent.check(done.value, [b.key for b in briefs])


async def write_sections(
    ctx: ServiceContext, req: WriteSectionsRequest, llm: LLM, cache: ResultCache, settings: Settings
) -> WriteSectionsResponse:
    key = stable_hash("write_sections", req.model_dump(), llm.model, VERSIONS)
    if cached := await cache.get(ctx.workspace_id, key):
        return WriteSectionsResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    briefs = [SectionBrief(s.key, s.title, s.guidance) for s in req.sections]
    usage = Usage()
    written, dropped = await _write(llm, briefs, req, [], settings, usage)
    missing = [b for b in briefs if not written.get(b.key)]
    if missing:
        notes = [f"Section '{b.key}': {NOTHING_WRITTEN}" for b in missing]
        retry, retry_dropped = await _write(llm, missing, req, notes, settings, usage)
        written |= {k: v for k, v in retry.items() if v}
        dropped += retry_dropped
    response = WriteSectionsResponse(
        sections=[
            WrittenSectionOut(key=b.key, paragraphs=written.get(b.key) or [agent.MISSING]) for b in briefs
        ],
        dropped=dropped,
        meta=meta(llm.model, VERSIONS, usage, cached=False),
    )
    await cache.put(ctx.workspace_id, key, "write_sections", response.model_dump(exclude={"meta"}))
    return response
