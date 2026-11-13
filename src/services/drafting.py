"""Use-case: draft answers to the RFP's questions (Step 4 Responses).

Per question, in parallel: skip pricing (code) → draft with tools (agent) → check claims against the
passages it read (code) → review (agent) → one retry with the problems as notes → else "needs a person".
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from agents import drafting, review
from agents.drafting import Claim, DraftOutput, Library, QuestionContext
from agents.review import ReviewItem
from api.schemas import (
    AnswerOut,
    ClaimOut,
    DraftAnswersRequest,
    DraftAnswersResponse,
    DraftStatus,
    QuestionIn,
)
from core.config import Settings
from core.errors import RuleViolation
from core.security import ServiceContext
from llm.ports import LLM, Usage
from rules.text import is_pricing_question, stable_hash
from services.common import meta
from services.vectors import Vectorizer
from store.ports import Hit, LibraryIndex, ResultCache, StoredPassage

VERSIONS = {"drafting": drafting.PROMPT_VERSION, "review": review.PROMPT_VERSION}
ANSWER_RULE = (
    "Every fact in the statement must be supported by the evidence. The statement must contain no "
    "prices, rates or fees."
)
NO_MATERIAL = "Our past proposals do not answer this question."

R = TypeVar("R")


class _Db:
    """One request session is shared by parallel questions; it must never run two queries at once."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run(self, fn: Callable[[], Awaitable[R]]) -> R:
        async with self._lock:
            return await fn()


def _answer(
    q: QuestionIn, status: DraftStatus, reason: str, answer: str = "", claims: list[ClaimOut] | None = None
) -> AnswerOut:
    return AnswerOut(id=q.id, status=status, answer=answer, claims=claims or [], reason=reason)


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
    q: QuestionIn, ctx: QuestionContext, library: Library, llm: LLM, settings: Settings, usage: Usage
) -> AnswerOut:
    notes: list[str] = []
    for _ in range(2):
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
                return _answer(q, "drafted", "", out.answer, _claims_out(claims, library.retrieved))
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
    hashes = await index.content_hashes(ws, selected)
    labels = {e: f"{m.client} — {m.title} ({m.year}, {m.outcome})" for e, m in entries.items()}
    question_ctx = (req.rfp_summary, req.requirements)
    db = _Db()
    usage = Usage()

    async def search(query: str) -> list[Hit]:
        async def run() -> list[Hit]:
            vector = (await vectors.embed([query]))[0]
            return await index.search(
                ws, vector, entry_ids=selected, sections=["rfp", "proposal"], limit=drafting.SEARCH_RESULTS
            )

        return await db.run(run)

    async def fetch(passage_id: str) -> StoredPassage | None:
        return await db.run(lambda: index.passage(ws, passage_id))

    async def one(q: QuestionIn) -> AnswerOut:
        if not q.draftable:
            return _answer(q, "skipped", "A person writes this one.")
        if is_pricing_question(q.text):
            return _answer(q, "skipped_pricing", "Pricing is never drafted by AI. A person writes this one.")
        key = stable_hash("draft", q.text, question_ctx, sorted(hashes.items()), llm.model, VERSIONS)
        if cached := await db.run(lambda: cache.get(ws, key)):
            return AnswerOut(**{**cached, "id": q.id})
        library = Library(ws, selected, labels, search, fetch)
        context = QuestionContext(q.text, req.rfp_summary, req.requirements)
        answer = await _draft_one(q, context, library, llm, settings, usage)
        if answer.status == "drafted":
            await db.run(lambda: cache.put(ws, key, "draft", answer.model_dump()))
        return answer

    answers = await asyncio.gather(*(one(q) for q in req.questions))
    cached_all = usage.calls == 0 and any(a.status == "drafted" for a in answers)
    return DraftAnswersResponse(
        answers=list(answers), meta=meta(llm.model, VERSIONS, usage, cached=cached_all)
    )
