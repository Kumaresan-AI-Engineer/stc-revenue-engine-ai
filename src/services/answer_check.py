"""Use-case: check one typed clarification answer while the user waits.

pre-check (code: plainly not an answer, no model call) → judge (agent, low effort) → short texts, no money
(code) → cache.
"""

from agents import answer_check as agent
from agents.answer_check import AnswerToCheck
from api.schemas import CheckAnswerRequest, CheckAnswerResponse
from core.config import Settings
from core.security import ServiceContext
from llm.ports import LLM, Usage
from rules.text import stable_hash
from services.common import meta
from store.ports import ResultCache

VERSIONS = {"answer_check": agent.PROMPT_VERSION}


async def check_answer(
    ctx: ServiceContext, req: CheckAnswerRequest, llm: LLM, cache: ResultCache, settings: Settings
) -> CheckAnswerResponse:
    data = AnswerToCheck(req.question, req.why, req.category, req.answer)
    if (verdict := agent.precheck(data)) is not None:
        return CheckAnswerResponse(
            **verdict.model_dump(), meta=meta(llm.model, VERSIONS, Usage(), cached=False)
        )
    key = stable_hash("check_answer", req.model_dump(), llm.model, VERSIONS)
    if cached := await cache.get(ctx.workspace_id, key):
        return CheckAnswerResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    done = await agent.ask(llm, data, settings.answer_check_effort)
    response = CheckAnswerResponse(
        **agent.check(done.value).model_dump(), meta=meta(llm.model, VERSIONS, done.usage, cached=False)
    )
    await cache.put(ctx.workspace_id, key, "check_answer", response.model_dump(exclude={"meta"}))
    return response
