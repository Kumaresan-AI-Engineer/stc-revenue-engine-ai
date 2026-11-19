"""Use-case: assume answers for open clarification questions (Step 5 Queries).

assume (agent) → one per known id, no money (code) → one retry for the ids still missing → a generic,
conservative assumption for any id still missing → cache. Every question gets exactly one assumption.
"""

from agents import assumption as agent
from agents.assumption import Assumed, AssumptionInput, OpenQuestion
from api.schemas import AssumeAnswersRequest, AssumeAnswersResponse, AssumptionOut
from core.config import Settings
from core.security import ServiceContext
from llm.ports import LLM, Usage
from rules.text import stable_hash
from services.common import meta
from store.ports import ResultCache

VERSIONS = {"assumption": agent.PROMPT_VERSION}


async def _assume(llm: LLM, data: AssumptionInput, settings: Settings, usage: Usage) -> dict[str, Assumed]:
    done = await agent.ask(llm, data, settings.assumption_effort)
    usage.add(done.usage)
    return agent.check(done.value, [q.id for q in data.questions])


async def assume_answers(
    ctx: ServiceContext, req: AssumeAnswersRequest, llm: LLM, cache: ResultCache, settings: Settings
) -> AssumeAnswersResponse:
    key = stable_hash("assume_answers", req.model_dump(), llm.model, VERSIONS)
    if cached := await cache.get(ctx.workspace_id, key):
        return AssumeAnswersResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    questions = [OpenQuestion(q.id, q.question, q.category, q.why) for q in req.questions]
    data = AssumptionInput(req.rfp_summary, req.engagement_model, req.requirements, questions)
    usage = Usage()
    assumed = await _assume(llm, data, settings, usage)
    missing = [q for q in questions if q.id not in assumed]
    if missing:
        retry = AssumptionInput(req.rfp_summary, req.engagement_model, req.requirements, missing)
        assumed |= await _assume(llm, retry, settings, usage)
    response = AssumeAnswersResponse(
        assumptions=[
            AssumptionOut(**(assumed.get(q.id) or agent.generic(q.id)).model_dump()) for q in questions
        ],
        meta=meta(llm.model, VERSIONS, usage, cached=False),
    )
    await cache.put(ctx.workspace_id, key, "assume_answers", response.model_dump(exclude={"meta"}))
    return response
