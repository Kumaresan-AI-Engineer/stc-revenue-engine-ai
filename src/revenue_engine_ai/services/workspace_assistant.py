"""Use-case: Ask AI, the chatbot (any question; workspace questions answered from the workspace).

answer (agent) → check (code: cited ids, traceable figures, no money) → one retry with the notes → cache. The
backend counts every figure and chooses the passages; amounts are redacted again here before the prompt.
"""

from revenue_engine_ai.agents import workspace_assistant as agent
from revenue_engine_ai.agents.workspace_assistant import Answer, Context
from revenue_engine_ai.api.schemas import WorkspaceAskRequest, WorkspaceAskResponse
from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.security import ServiceContext
from revenue_engine_ai.llm.ports import LLM, Usage
from revenue_engine_ai.rules.text import redact_amounts, stable_hash
from revenue_engine_ai.services.common import meta
from revenue_engine_ai.store.ports import ResultCache

VERSIONS = {"workspace_assistant": agent.PROMPT_VERSION}


def _prompt_data(
    req: WorkspaceAskRequest,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    history = [{"question": t.question, "answer": t.answer} for t in req.history]
    facts = [{"id": f.id, "label": f.label, "text": redact_amounts(f.text)} for f in req.facts]
    documents = [
        {
            "id": d.id,
            "kind": d.kind,
            "title": d.title,
            "client": d.client,
            "passage": redact_amounts(d.passage),
        }
        for d in req.documents
    ]
    return history, facts, documents


def _context(
    req: WorkspaceAskRequest,
    history: list[dict[str, str]],
    facts: list[dict[str, str]],
    documents: list[dict[str, str]],
) -> Context:
    parts = [req.question]
    parts += [f"{t['question']} {t['answer']}" for t in history]
    parts += [f"{f['label']} {f['text']}" for f in facts]
    parts += [f"{d['title']} {d['client']} {d['passage']}" for d in documents]
    return Context(
        ids=frozenset(f["id"] for f in facts) | frozenset(d["id"] for d in documents), text="\n".join(parts)
    )


async def ask(
    ctx: ServiceContext, req: WorkspaceAskRequest, llm: LLM, cache: ResultCache, settings: Settings
) -> WorkspaceAskResponse:
    key = stable_hash("workspace_ask", req.model_dump(), llm.model, VERSIONS)
    if cached := await cache.get(ctx.workspace_id, key):
        return WorkspaceAskResponse(**cached, meta=meta(llm.model, VERSIONS, Usage(), cached=True))

    history, facts, documents = _prompt_data(req)
    context = _context(req, history, facts, documents)
    usage = Usage()

    async def attempt(notes: list[str]) -> tuple[Answer, list[str]]:
        done = await agent.ask(
            llm, req.question, history, facts, documents, notes, settings.workspace_assistant_effort
        )
        usage.add(done.usage)
        return agent.check(done.value, context)

    answer, notes = await attempt([])
    if notes:
        answer, _ = await attempt(notes)
    response = WorkspaceAskResponse(
        kind=answer.kind,
        answered=answer.answered,
        answer=answer.answer,
        how=answer.how,
        source_ids=answer.source_ids,
        follow_ups=answer.follow_ups,
        verified=answer.verified,
        meta=meta(llm.model, VERSIONS, usage, cached=False),
    )
    await cache.put(ctx.workspace_id, key, "workspace_ask", response.model_dump(exclude={"meta"}))
    return response
