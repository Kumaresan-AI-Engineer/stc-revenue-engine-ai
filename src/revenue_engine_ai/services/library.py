"""Use-cases: index a past proposal (Library indexer) and semantic search (Step 3 Matches)."""

import hashlib

from revenue_engine_ai.api.schemas import (
    DocumentMatch,
    DocumentMetaIn,
    EntryMetaIn,
    IndexResponse,
    PassageMatch,
    ProjectSimilarity,
    RequirementMatches,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarRequest,
    SimilarResponse,
)
from revenue_engine_ai.core.errors import NotFound, RuleViolation
from revenue_engine_ai.core.security import ServiceContext
from revenue_engine_ai.documents.chunking import CHUNKING_VERSION, Passage, passages, summary_passage
from revenue_engine_ai.documents.parsing import extract_pages
from revenue_engine_ai.rules.matching import (
    DOCUMENT_KINDS,
    DOCUMENT_SECTION,
    SEARCH_POOL,
    best_per_entry,
    group_hits,
)
from revenue_engine_ai.rules.text import stable_hash
from revenue_engine_ai.services.vectors import Vectorizer
from revenue_engine_ai.store.ports import EntryFilter, EntryMeta, LibraryIndex

SECTIONS = ("rfp", "proposal")
# A document search reads every section: a past bid's two documents and its summary, or a standalone document.
SEARCH_SECTIONS = (*SECTIONS, "document", "summary")
PROPOSAL_KIND = "proposal"


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


async def index_document(
    ctx: ServiceContext,
    entry_id: str,
    meta: DocumentMetaIn,
    file: tuple[str, bytes],  # (filename, content)
    index: LibraryIndex,
    vectors: Vectorizer,
) -> IndexResponse:
    """Index one standalone document (a case study or another file) so a report can search it."""
    name, data = file
    content_hash = stable_hash(meta.model_dump(), name, hashlib.sha256(data).hexdigest(), CHUNKING_VERSION)
    if (await index.content_hashes(ctx.workspace_id, [entry_id])).get(entry_id) == content_hash:
        return IndexResponse(entry_id=entry_id, passages=0, content_hash=content_hash, unchanged=True)
    head = f"{meta.title} ({meta.client or 'no client'}, {meta.year}, {meta.kind})."
    found = [
        summary_passage(entry_id, head),
        *passages(entry_id, DOCUMENT_SECTION, extract_pages(name, data)),
    ]
    embedded = await vectors.embed([p.text for p in found])
    entry = EntryMeta(entry_id, meta.client, meta.title, meta.industry, "pending", meta.year, meta.kind)
    await index.replace_entry(ctx.workspace_id, entry, content_hash, found, embedded)
    return IndexResponse(entry_id=entry_id, passages=len(found), content_hash=content_hash, unchanged=False)


async def delete_entry(ctx: ServiceContext, entry_id: str, index: LibraryIndex) -> None:
    if not await index.delete_entry(ctx.workspace_id, entry_id):
        raise NotFound("This proposal is not in the index.")


async def _supporting_documents(
    ctx: ServiceContext,
    vector: list[float],
    documents: dict[str, EntryMeta],
    req: SimilarRequest,
    index: LibraryIndex,
) -> list[DocumentMatch]:
    """The case studies and reference documents that cover one requirement, best first."""
    if not documents:
        return []
    hits = await index.search(
        ctx.workspace_id, vector, entry_ids=sorted(documents), sections=[DOCUMENT_SECTION], limit=SEARCH_POOL
    )
    return [
        DocumentMatch(
            entry_id=m.entry_id,
            kind=documents[m.entry_id].kind,
            client=documents[m.entry_id].client,
            title=documents[m.entry_id].title,
            year=documents[m.entry_id].year,
            similarity=m.similarity,
            passage_id=m.passage_id,
            page=m.page,
            text=m.text,
        )
        for m in best_per_entry(hits, req.threshold)[: req.top_k]
    ]


async def similar(
    ctx: ServiceContext, req: SimilarRequest, index: LibraryIndex, vectors: Vectorizer, embedding_model: str
) -> SimilarResponse:
    texts = [r.text for r in req.requirements] + ([req.rfp_summary] if req.rfp_summary else [])
    embedded = await vectors.embed(texts)
    # The whole library means the past bids: a standalone document is not a project to compare with.
    scope = req.entry_ids
    if scope is None:
        known = await index.entries(ctx.workspace_id, None)
        scope = [e for e, m in known.items() if m.kind == PROPOSAL_KIND]
    documents = (
        await index.find_entries(ctx.workspace_id, EntryFilter(kinds=DOCUMENT_KINDS))
        if req.include_documents
        else {}
    )

    requirements = []
    for r, vector in zip(req.requirements, embedded, strict=False):
        hits = await index.search(
            ctx.workspace_id, vector, entry_ids=scope, sections=list(SECTIONS), limit=SEARCH_POOL
        )
        matches = best_per_entry(hits, req.threshold)[: req.top_k]
        requirements.append(
            RequirementMatches(
                key=r.key,
                documents=await _supporting_documents(ctx, vector, documents, req, index),
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
            ctx.workspace_id, embedded[-1], entry_ids=scope, sections=["summary"], limit=SEARCH_POOL
        )
        projects = [
            ProjectSimilarity(entry_id=m.entry_id, similarity=m.similarity) for m in best_per_entry(hits, 0.0)
        ]
    return SimilarResponse(requirements=requirements, projects=projects, embedding_model=embedding_model)


async def search(
    ctx: ServiceContext, req: SearchRequest, index: LibraryIndex, vectors: Vectorizer, embedding_model: str
) -> SearchResponse:
    """Documents whose text is about the query, best first, each with the passages that matched. No model but
    the embedder: the filters narrow the entries first, then the passages are ranked by cosine."""
    scope = EntryFilter(
        tuple(req.kinds), tuple(req.outcomes), tuple(req.industries), req.year_from, req.year_to
    )
    candidates = await index.find_entries(ctx.workspace_id, scope)
    found: list[SearchResult] = []
    if candidates:
        (vector,) = await vectors.embed([req.query])
        pool = min(1000, req.limit * req.passages_per_document * 4)
        hits = await index.search(
            ctx.workspace_id, vector, entry_ids=list(candidates), sections=list(SEARCH_SECTIONS), limit=pool
        )
        for grouped in group_hits(hits, req.threshold, req.passages_per_document, req.limit):
            meta = candidates[grouped.entry_id]
            found.append(
                SearchResult.model_validate(
                    {
                        "entry_id": meta.entry_id,
                        "kind": meta.kind,
                        "client": meta.client,
                        "title": meta.title,
                        "industry": meta.industry,
                        "outcome": meta.outcome,
                        "year": meta.year,
                        "similarity": grouped.similarity,
                        "passages": [
                            {
                                "passage_id": h.passage.id,
                                "section": h.passage.section,
                                "page": h.passage.page,
                                "similarity": h.similarity,
                                "text": h.passage.text,
                            }
                            for h in grouped.passages
                        ],
                    }
                )
            )
    return SearchResponse(
        results=found, candidates=len(candidates), threshold=req.threshold, embedding_model=embedding_model
    )
