"""Use-case: questions for the client (Step 5 Queries).

ask (agent) → rules (code) → review (agent) → one retry with the problems as notes → drop what still fails.
"""

from agents import clarification, review
from agents.clarification import ClarificationInput, ClientQuestion, Topic
from agents.review import ReviewItem
from api.schemas import ClientQueriesRequest, ClientQueriesResponse
from core.config import Settings
from core.security import ServiceContext
from llm.ports import LLM, Usage
from rules.text import stable_hash
from services.common import meta
from store.ports import ResultCache

VERSIONS = {"clarification": clarification.PROMPT_VERSION, "review": review.PROMPT_VERSION}
QUESTION_RULE = (
    "The question is clear, asks about the topic in the evidence, and the assumption is a safe, "
    "conservative basis for an estimate."
)


def _input(req: ClientQueriesRequest) -> ClarificationInput:
    return ClarificationInput(
        requirements=[(Topic(r.key, r.label), r.covered) for r in req.requirements],
        unclear_points=[Topic(u.id, u.text) for u in req.unclear_points],
        engagement_model=req.engagement_model,
        existing_questions=req.existing_questions,
        existing_assumptions=req.existing_assumptions,
        limit=req.limit,
    )


async def _attempt(
    llm: LLM, data: ClarificationInput, notes: list[str], settings: Settings, usage: Usage
) -> tuple[list[ClientQuestion], list[str]]:
    done = await clarification.ask(llm, data, notes, settings.clarification_effort)
    usage.add(done.usage)
    kept, problems = clarification.check(done.value, data)
    topics = data.topics()
    items = [
        ReviewItem(
            id=str(i),
            statement=f"Question: {q.question} Assumption if no reply: {q.assumption}",
            evidence=[topics[q.about]],
            rule=QUESTION_RULE,
        )
        for i, q in enumerate(kept)
    ]
    reviewed = await review.review(llm, items, settings.review_effort)
    usage.add(reviewed.usage)
    failed = {kept[int(i)].question: reason for i, reason in reviewed.failed.items()}
    passed = [q for q in kept if q.question not in failed]
    return passed, problems + [f"'{q}': {reason}" for q, reason in failed.items()]


async def client_queries(
    ctx: ServiceContext, req: ClientQueriesRequest, llm: LLM, cache: ResultCache, settings: Settings
) -> ClientQueriesResponse:
    key = stable_hash("clarify", req.model_dump(), llm.model, VERSIONS)
    if cached := await cache.get(ctx.workspace_id, key):
        return ClientQueriesResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    data = _input(req)
    usage = Usage()
    questions, problems = await _attempt(llm, data, [], settings, usage)
    # A second round only to make up a shortfall: with `limit` good questions in hand, asking and reviewing
    # again cannot improve what is returned, it only doubles the wait ('Suggest with AI' targets 10 s).
    if problems and len(questions) < req.limit:
        retry, problems = await _attempt(llm, data, problems, settings, usage)
        # Keep good questions from the first attempt too; the rules drop any repeat.
        merged = clarification.ClarificationOutput(questions=questions + retry)
        questions, _ = clarification.check(merged, data)
    questions = questions[: req.limit]
    dropped = sorted(problems)
    response = ClientQueriesResponse(
        questions=questions, dropped=dropped, meta=meta(llm.model, VERSIONS, usage, cached=False)
    )
    await cache.put(ctx.workspace_id, key, "clarify", response.model_dump(exclude={"meta"}))
    return response
