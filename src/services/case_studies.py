"""Use-case: write a case study from a decided past proposal.

passages of the proposal (vector search within the entry, when it is indexed) → story (agent) → plain text, no
money, no invented metric (code) → one retry when the title or summary is empty or a template section is not
written → cache.
"""

from dataclasses import asdict
from typing import Any

from agents import case_study_writer as agent
from agents.case_study_writer import CaseStudyDraft, SourcePassage, TemplateSection
from api.schemas import WriteCaseStudyRequest, WriteCaseStudyResponse
from core.config import Settings
from core.security import ServiceContext
from llm.ports import LLM, Usage
from rules.text import stable_hash
from services.common import meta
from services.library import SECTIONS
from services.vectors import Vectorizer
from store.ports import LibraryIndex, ResultCache

VERSIONS = {"case_study_writer": agent.PROMPT_VERSION}
# What a case study needs from the proposal: the client's challenge, what we proposed, and what it achieved.
QUERIES = (
    "{client}: the client's business, its problem, challenges and needs",
    "our proposed solution, approach, architecture and delivery",
    "results, outcomes, benefits, impact and measurable improvements",
)
PER_QUERY, MAX_PASSAGES, PASSAGE_MAX = 4, 10, 1500


async def _passages(
    ctx: ServiceContext, req: WriteCaseStudyRequest, index: LibraryIndex, vectors: Vectorizer
) -> list[SourcePassage]:
    """The proposal's own passages for each question, best first, each once; none when it is not indexed."""
    if not await index.entries(ctx.workspace_id, [req.entry_id]):
        return []
    embedded = await vectors.embed([q.format(client=req.facts.client) for q in QUERIES])
    found: dict[str, SourcePassage] = {}
    for vector in embedded:
        hits = await index.search(
            ctx.workspace_id,
            vector,
            entry_ids=[req.entry_id],
            sections=[*SECTIONS, "summary"],
            limit=PER_QUERY,
        )
        for h in hits:
            p = h.passage
            found.setdefault(p.id, SourcePassage(p.id, p.page, p.text[:PASSAGE_MAX]))
    return list(found.values())[:MAX_PASSAGES]


def _body(draft: CaseStudyDraft, passages: list[SourcePassage], dropped: list[str]) -> dict[str, Any]:
    """The response without its meta, as cached."""
    pages = {p.id: p.page for p in passages}
    body = asdict(draft)
    body["sources"] = [{"passage_id": s, "page": pages.get(s)} for s in draft.sources]
    return body | {"dropped": dropped}


async def write_case_study(
    ctx: ServiceContext,
    req: WriteCaseStudyRequest,
    llm: LLM,
    cache: ResultCache,
    index: LibraryIndex,
    vectors: Vectorizer,
    settings: Settings,
) -> WriteCaseStudyResponse:
    passages = await _passages(ctx, req, index, vectors)
    # The passages are part of the key: a proposal indexed again is written again.
    key = stable_hash("write_case_study", req.model_dump(), [p.id for p in passages], llm.model, VERSIONS)
    if cached := await cache.get(ctx.workspace_id, key):
        return WriteCaseStudyResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    facts = req.facts.model_dump()
    template = [TemplateSection(t.key, t.title, t.guidance) for t in req.template]
    usage = Usage()
    done = await agent.ask(llm, facts, template, passages, [], settings.case_study_writer_effort)
    usage.add(done.usage)
    draft, dropped = agent.check(done.value, facts, passages, template)
    if not draft.title or not draft.summary or len(draft.sections) < len(template):
        done = await agent.ask(llm, facts, template, passages, dropped, settings.case_study_writer_effort)
        usage.add(done.usage)
        draft, dropped = agent.check(done.value, facts, passages, template)
    body = _body(draft, passages, dropped)
    await cache.put(ctx.workspace_id, key, "write_case_study", body)
    return WriteCaseStudyResponse(**body, meta=meta(llm.model, VERSIONS, usage, cached=False))
