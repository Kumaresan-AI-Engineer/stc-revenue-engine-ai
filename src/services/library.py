"""Use-cases: index a past proposal (Library indexer) and semantic search (Step 3 Matches)."""

import hashlib

from api.schemas import (
    EntryMetaIn,
    IndexResponse,
    PassageMatch,
    ProjectSimilarity,
    RequirementMatches,
    SimilarRequest,
    SimilarResponse,
)
from core.errors import NotFound, RuleViolation
from core.security import ServiceContext
from documents.chunking import CHUNKING_VERSION, Passage, passages, summary_passage
from documents.parsing import extract_pages
from rules.matching import SEARCH_POOL, best_per_entry
from rules.text import stable_hash
from services.vectors import Vectorizer
from store.ports import EntryMeta, LibraryIndex

SECTIONS = ("rfp", "proposal")


def _summary_text(meta: EntryMetaIn) -> str:
    head = f"{meta.title} for {meta.client} ({meta.industry}, {meta.year}, {meta.outcome})."
    return f"{head} {meta.summary}".strip()


async def index_entry(
    ctx: ServiceContext,
    entry_id: str,
    meta: EntryMetaIn,
    files: dict[str, tuple[str, bytes]],  # section -> (filename, content)
    index: LibraryIndex,
    vectors: Vectorizer,
) -> IndexResponse:
    if not files and not meta.summary.strip():
        raise RuleViolation("content", "Send the RFP, the proposal or a summary to index.")
    content_hash = stable_hash(
        meta.model_dump(),
        {s: (name, hashlib.sha256(data).hexdigest()) for s, (name, data) in files.items()},
        CHUNKING_VERSION,
    )
    if (await index.content_hashes(ctx.workspace_id, [entry_id])).get(entry_id) == content_hash:
        return IndexResponse(entry_id=entry_id, passages=0, content_hash=content_hash, unchanged=True)

    found: list[Passage] = [summary_passage(entry_id, _summary_text(meta))]
    for section in SECTIONS:
        if section in files:
            name, data = files[section]
            found += passages(entry_id, section, extract_pages(name, data))
    embedded = await vectors.embed([p.text for p in found])
    entry = EntryMeta(entry_id, meta.client, meta.title, meta.industry, meta.outcome, meta.year)
    await index.replace_entry(ctx.workspace_id, entry, content_hash, found, embedded)
    return IndexResponse(entry_id=entry_id, passages=len(found), content_hash=content_hash, unchanged=False)


async def delete_entry(ctx: ServiceContext, entry_id: str, index: LibraryIndex) -> None:
    if not await index.delete_entry(ctx.workspace_id, entry_id):
        raise NotFound("This proposal is not in the index.")


async def similar(
    ctx: ServiceContext, req: SimilarRequest, index: LibraryIndex, vectors: Vectorizer, embedding_model: str
) -> SimilarResponse:
    texts = [r.text for r in req.requirements] + ([req.rfp_summary] if req.rfp_summary else [])
    embedded = await vectors.embed(texts)

    requirements = []
    for r, vector in zip(req.requirements, embedded, strict=False):
        hits = await index.search(
            ctx.workspace_id, vector, entry_ids=req.entry_ids, sections=list(SECTIONS), limit=SEARCH_POOL
        )
        matches = best_per_entry(hits, req.threshold)[: req.top_k]
        requirements.append(
            RequirementMatches(
                key=r.key,
                matches=[
                    PassageMatch(
                        entry_id=m.entry_id,
                        similarity=m.similarity,
                        passage_id=m.passage_id,
                        page=m.page,
                        text=m.text,
                    )
                    for m in matches
                ],
            )
        )

    projects: list[ProjectSimilarity] = []
    if req.rfp_summary:
        hits = await index.search(
            ctx.workspace_id, embedded[-1], entry_ids=req.entry_ids, sections=["summary"], limit=SEARCH_POOL
        )
        projects = [
            ProjectSimilarity(entry_id=m.entry_id, similarity=m.similarity) for m in best_per_entry(hits, 0.0)
        ]
    return SimilarResponse(requirements=requirements, projects=projects, embedding_model=embedding_model)
