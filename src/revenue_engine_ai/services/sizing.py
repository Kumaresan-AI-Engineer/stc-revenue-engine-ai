"""Use-case: size requirements by industry benchmarks (the estimate's starting effort).

size classes (agent) → one per key, person-days by class (code) → the median for any key not sized → cache.
"""

from revenue_engine_ai.agents import sizing as agent
from revenue_engine_ai.agents.sizing import Requirement
from revenue_engine_ai.api.schemas import SizeOut, SizeRequirementsRequest, SizeRequirementsResponse
from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.security import ServiceContext
from revenue_engine_ai.llm.ports import LLM, Usage
from revenue_engine_ai.rules.text import stable_hash
from revenue_engine_ai.services.common import meta
from revenue_engine_ai.store.ports import ResultCache

# "size_days" versions the code's class → person-days table (agents/sizing.SIZE_DAYS).
VERSIONS = {"sizing": agent.PROMPT_VERSION, "size_days": "size-days-v1"}


async def size_requirements(
    ctx: ServiceContext, req: SizeRequirementsRequest, llm: LLM, cache: ResultCache, settings: Settings
) -> SizeRequirementsResponse:
    key = stable_hash("size_requirements", req.model_dump(), llm.model, VERSIONS)
    if cached := await cache.get(ctx.workspace_id, key):
        return SizeRequirementsResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    requirements = [Requirement(r.key, r.label, r.detail) for r in req.requirements]
    done = await agent.ask(llm, req.industry, requirements, settings.sizing_effort)
    response = SizeRequirementsResponse(
        sizes=[
            SizeOut(key=s.key, size=s.size, days=s.days, rationale=s.rationale)
            for s in agent.sizes(done.value, requirements)
        ],
        meta=meta(llm.model, VERSIONS, done.usage, cached=False),
    )
    await cache.put(ctx.workspace_id, key, "size_requirements", response.model_dump(exclude={"meta"}))
    return response
