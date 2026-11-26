"""Use-case: draft answers to the RFP's questions (Step 4 Responses).

Per question, in parallel: skip pricing (code) → draft with tools (agent) → check claims against the
passages it read (code) → review (agent) → one retry with the problems as notes → else "needs a person".

Sources: the selected past proposals except lost ones (never an answer source), won ones first, and, when the
request asks for them, every case study and reference document the workspace has indexed (they have no
outcome to lose). A draft's confidence says how much checking it needed: accepted at the first try, or only
after a correction.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from revenue_engine_ai.agents import drafting, review
from revenue_engine_ai.agents.drafting import Claim, DraftOutput, Library, QuestionContext
from revenue_engine_ai.agents.review import ReviewItem
from revenue_engine_ai.api.schemas import (
    AnswerOut,
    CitationOut,
    ClaimOut,
    DraftAnswersRequest,
    DraftAnswersResponse,
    DraftStatus,
    QuestionIn,
)
from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.errors import RuleViolation
from revenue_engine_ai.core.security import ServiceContext
from revenue_engine_ai.llm.ports import LLM, Usage
from revenue_engine_ai.rules.matching import DOCUMENT_KINDS, DOCUMENT_SECTION, KIND_LABELS
from revenue_engine_ai.rules.text import is_pricing_question, stable_hash
from revenue_engine_ai.services.common import meta
from revenue_engine_ai.services.vectors import Vectorizer
from revenue_engine_ai.store.ports import (
    EntryFilter,
    EntryMeta,
    Hit,
    LibraryIndex,
    ResultCache,
    StoredPassage,
)

# "answer" versions the shape of a cached answer (confidence and citations were added in v2).
VERSIONS = {"drafting": drafting.PROMPT_VERSION, "review": review.PROMPT_VERSION, "answer": "answer-v2"}
FIRST_TRY_CONFIDENCE = 0.9
CORRECTED_CONFIDENCE = 0.7
ONLY_LOST = "Only lost proposals were selected, and they are never used as sources."
NO_DOCUMENTS = "No case studies or reference documents are indexed yet."
ANSWER_RULE = (
    "Every fact in the statement must be supported by the evidence. The statement must contain no "
    "prices, rates or fees."
)
NO_MATERIAL = "Our past proposals and documents do not answer this question."

R = TypeVar("R")


class _Db:
    """One request session is shared by parallel questions; it must never run two queries at once."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run(self, fn: Callable[[], Awaitable[R]]) -> R:
        async with self._lock:
            return await fn()


def _answer(
    q: QuestionIn,
    status: DraftStatus,
    reason: str,
    answer: str = "",
    claims: list[ClaimOut] | None = None,
    confidence: float = 0.0,
    titles: dict[str, str] | None = None,
    kinds: dict[str, str] | None = None,
) -> AnswerOut:
    claims = claims or []
    citations = [
        CitationOut(
            source_id=c.entry_id,
            kind=(kinds or {}).get(c.entry_id, "proposal"),
            title=(titles or {}).get(c.entry_id, c.entry_id),
            quote=c.quote,
            page=c.page,
            passage_id=c.passage_id,
        )
        for c in claims
    ]
    return AnswerOut(
        id=q.id,
        status=status,
        answer=answer,
        claims=claims,
        reason=reason,
        confidence=confidence,
        no_confident_source=status == "needs_person",
        citations=citations,
    )


def _sources(entries: dict[str, EntryMeta]) -> list[str]:
    """Usable sources: never a lost proposal; won ones first, then by id (a stable order)."""
    usable = [e for e, m in entries.items() if m.outcome != "lost"]
    return sorted(usable, key=lambda e: (entries[e].outcome != "won", e))


def _name(m: EntryMeta) -> str:
    """Client — title; a document has no client unless a person tagged it."""
    return " — ".join(part for part in (m.client, m.title) if part)


def _claims_out(claims: list[Claim], retrieved: dict[str, StoredPassage]) -> list[ClaimOut]:
    return [
        ClaimOut(
            statement=c.statement,
            passage_id=c.passage_id,
            entry_id=retrieved[c.passage_id].entry_id,
            page=retrieved[c.passage_id].page,
            quote=c.quote,
        )
        for c in claims
    ]


async def _draft_one(
    q: QuestionIn,
    ctx: QuestionContext,
    library: Library,
    llm: LLM,
    settings: Settings,
    usage: Usage,
    titles: dict[str, str],
    kinds: dict[str, str],
) -> AnswerOut:
    notes: list[str] = []
    for attempt in range(2):
        done = await drafting.draft(llm, ctx, library, notes, settings.drafting_effort)
        usage.add(done.usage)
        out: DraftOutput = done.value
        if not out.enough_material or not out.answer.strip():
            return _answer(q, "needs_person", NO_MATERIAL)
        claims, problems = drafting.check(out, library.retrieved)
        if not problems:
            item = ReviewItem(
                id="answer", statement=out.answer, evidence=[c.quote for c in claims], rule=ANSWER_RULE
            )
            reviewed = await review.review(llm, [item], settings.review_effort)
            usage.add(reviewed.usage)
            if not reviewed.failed:
                confidence = FIRST_TRY_CONFIDENCE if attempt == 0 else CORRECTED_CONFIDENCE
                claims_out = _claims_out(claims, library.retrieved)
                return _answer(q, "drafted", "", out.answer, claims_out, confidence, titles, kinds)
            problems = [f"Reviewer: {reason}" for reason in reviewed.failed.values()]
        notes = problems
    return _answer(
        q, "needs_person", "The draft could not be backed by our past proposals: " + " ".join(notes)
    )


async def draft_answers(
    ctx: ServiceContext,
    req: DraftAnswersRequest,
    llm: LLM,
    index: LibraryIndex,
    vectors: Vectorizer,
    cache: ResultCache,
    settings: Settings,
) -> DraftAnswersResponse:
    ws = ctx.workspace_id
    selected = sorted(set(req.selected_entry_ids))
    entries = await index.entries(ws, selected)
    missing = [e for e in selected if e not in entries]
    if missing:
        raise RuleViolation(
            "library_not_indexed", f"These proposals are not indexed yet: {', '.join(missing)}."
        )
    documents = (
        await index.find_entries(ws, EntryFilter(kinds=DOCUMENT_KINDS)) if req.include_documents else {}
    )
    proposals = _sources(entries)
    sources = [*proposals, *documents]  # documents come in id order: the order never reshuffles
    hashes = await index.content_hashes(ws, sources)
    labels = {e: f"{m.client} — {m.title} ({m.year}, {m.outcome})" for e, m in entries.items()}
    labels |= {e: f"{_name(m)} ({m.year}, {KIND_LABELS[m.kind]})" for e, m in documents.items()}
    titles = {e: f"{m.client} — {m.title} ({m.year})" for e, m in entries.items()}
    titles |= {e: f"{_name(m)} ({m.year})" for e, m in documents.items()}
    kinds = {e: m.kind for e, m in documents.items()}
    empty = ONLY_LOST if selected else NO_DOCUMENTS
    question_ctx = (req.rfp_summary, req.requirements)
    db = _Db()
    usage = Usage()

    async def search(query: str) -> list[Hit]:
        async def run() -> list[Hit]:
            vector = (await vectors.embed([query]))[0]
            return await index.search(
                ws,
                vector,
                entry_ids=sources,
                sections=["rfp", "proposal", DOCUMENT_SECTION],
                limit=drafting.SEARCH_RESULTS,
            )

        return await db.run(run)

    async def fetch(passage_id: str) -> StoredPassage | None:
        return await db.run(lambda: index.passage(ws, passage_id))

    async def one(q: QuestionIn) -> AnswerOut:
        if not q.draftable:
            return _answer(q, "skipped", "A person writes this one.")
        if is_pricing_question(q.text):
            return _answer(q, "skipped_pricing", "Pricing is never drafted by AI. A person writes this one.")
        if not sources:
            return _answer(q, "needs_person", empty)
        key = stable_hash("draft", q.text, question_ctx, sorted(hashes.items()), llm.model, VERSIONS)
        if cached := await db.run(lambda: cache.get(ws, key)):
            return AnswerOut(**{**cached, "id": q.id})
        library = Library(ws, sources, labels, search, fetch)
        context = QuestionContext(q.text, req.rfp_summary, req.requirements)
        answer = await _draft_one(q, context, library, llm, settings, usage, titles, kinds)
        if answer.status == "drafted":
            await db.run(lambda: cache.put(ws, key, "draft", answer.model_dump()))
        return answer

    answers = await asyncio.gather(*(one(q) for q in req.questions))
    cached_all = usage.calls == 0 and any(a.status == "drafted" for a in answers)
    return DraftAnswersResponse(
        answers=list(answers), meta=meta(llm.model, VERSIONS, usage, cached=cached_all)
    )
