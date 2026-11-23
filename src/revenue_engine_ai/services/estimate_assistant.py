"""Use-case: Step 6 'Ask AI' about a proposal's estimate.

actions (agent) → only what the estimate allows (code) → one retry with the notes when an action was dropped →
cache. The backend applies the actions and computes every figure; no price is ever sent here.
"""

from revenue_engine_ai.agents import estimate_assistant as agent
from revenue_engine_ai.agents.estimate_assistant import Action, Limits
from revenue_engine_ai.api.schemas import EstimateActionOut, EstimateRequestRequest, EstimateRequestResponse
from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.security import ServiceContext
from revenue_engine_ai.llm.ports import LLM, Usage
from revenue_engine_ai.rules.text import stable_hash
from revenue_engine_ai.services.common import meta
from revenue_engine_ai.store.ports import ResultCache

VERSIONS = {"estimate_assistant": agent.PROMPT_VERSION}


def _limits(req: EstimateRequestRequest) -> Limits:
    s = req.state
    return Limits(
        in_scope={i.key: i.must for i in s.in_scope},
        out_of_scope=frozenset(i.key for i in s.out_of_scope),
        approaches=frozenset(a.key for a in s.delivery_approaches),
        approach=s.delivery_approach,
        team_min=s.team_min,
        team_max=s.team_max,
        can_change_price=s.can_change_price,
        amounts=frozenset(req.amounts),
    )


def _out(a: Action) -> EstimateActionOut:
    return EstimateActionOut(
        kind=a.kind, key=a.key, weeks=a.weeks, team=a.team, amount_ref=a.amount_ref, percent=a.percent
    )


async def _ask(
    llm: LLM, req: EstimateRequestRequest, limits: Limits, notes: list[str], settings: Settings, usage: Usage
) -> tuple[bool, str, list[Action], list[str]]:
    done = await agent.ask(
        llm, req.ask, req.amounts, req.state.model_dump(), notes, settings.estimate_assistant_effort
    )
    usage.add(done.usage)
    return agent.check(done.value, limits)


async def estimate_request(
    ctx: ServiceContext, req: EstimateRequestRequest, llm: LLM, cache: ResultCache, settings: Settings
) -> EstimateRequestResponse:
    key = stable_hash("estimate_request", req.model_dump(), llm.model, VERSIONS)
    if cached := await cache.get(ctx.workspace_id, key):
        return EstimateRequestResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    limits = _limits(req)
    usage = Usage()
    understood, message, actions, notes = await _ask(llm, req, limits, [], settings, usage)
    if notes:
        understood, message, actions, _ = await _ask(llm, req, limits, notes, settings, usage)
    response = EstimateRequestResponse(
        understood=understood,
        message=message,
        actions=[_out(a) for a in actions],
        meta=meta(llm.model, VERSIONS, usage, cached=False),
    )
    await cache.put(ctx.workspace_id, key, "estimate_request", response.model_dump(exclude={"meta"}))
    return response
