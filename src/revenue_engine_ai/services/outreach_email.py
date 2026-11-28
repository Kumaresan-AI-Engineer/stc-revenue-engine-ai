"""Use-case for grounded, interactive outreach email drafting."""

import json

from revenue_engine_ai.agents import outreach_email as agent
from revenue_engine_ai.agents import review as review_agent
from revenue_engine_ai.agents.review import ReviewItem
from revenue_engine_ai.api.schemas import (
    OutreachEmailDraftRequest,
    OutreachEmailDraftResponse,
)
from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.errors import RuleViolation
from revenue_engine_ai.core.security import ServiceContext
from revenue_engine_ai.llm.ports import LLM, Usage
from revenue_engine_ai.rules.text import stable_hash
from revenue_engine_ai.services import common
from revenue_engine_ai.store.ports import ResultCache

VERSIONS = {
    "outreach_email": agent.PROMPT_VERSION,
    "review": review_agent.PROMPT_VERSION,
    "outreach_review_rule": "outreach-review-v2",
}
_FAILED = "The email could not be checked against its context. Write or review it yourself."
_DETAIL_MAX = 300


def _detail(problems: list[str], reviewed: review_agent.ReviewResult | None = None) -> str:
    """The user-facing failure, plus the last actionable reasons so a retry is not guesswork."""
    reasons = [*problems]
    if reviewed is not None:
        reasons.extend(reviewed.failed.values())
    text = "; ".join(reason.strip().rstrip(".") for reason in reasons[:3] if reason.strip())
    return f"{_FAILED} ({text[:_DETAIL_MAX]})" if text else _FAILED


def _review_items(email: agent.CheckedEmail, req: OutreachEmailDraftRequest) -> list[ReviewItem]:
    context = req.model_dump(exclude={"references"})
    evidence = [json.dumps(context, ensure_ascii=False)]
    reference_evidence = [
        json.dumps(reference.model_dump(), ensure_ascii=False) for reference in req.references
    ]
    items = [
        ReviewItem(
            id="subject",
            statement=email.subject,
            evidence=[*evidence, *reference_evidence],
            rule=(
                "The subject must fit the requested goal and contain no unsupported factual claim or "
                "money amount."
            ),
        )
    ]
    for index, paragraph in enumerate(email.paragraphs):
        paragraph_evidence = [*evidence]
        paragraph_evidence.extend(
            json.dumps(req.references[ref - 1].model_dump(), ensure_ascii=False)
            for ref in paragraph.references
        )
        items.append(
            ReviewItem(
                id=f"paragraph-{index + 1}",
                statement=paragraph.text,
                evidence=paragraph_evidence,
                rule=(
                    "Every factual claim, number, prior relationship, capability or result must be "
                    "supported by the supplied context or a cited reference. Generic courtesy, "
                    "questions and an offer to discuss the requested topic are allowed. Do not add "
                    "prices or unsupported results."
                ),
            )
        )
    return items


def _fallback(req: OutreachEmailDraftRequest) -> agent.CheckedEmail | None:
    """A grounded draft for when the model keeps making claims the evidence cannot support."""
    first = (req.recipient_name.split() or ["there"])[0]
    topic = (req.priorities[0].strip() if req.priorities else "") or req.industry.strip() or "your work"
    paragraphs = [
        agent.EmailParagraph(
            text=f"Hi {first}, I wanted to reach out about {topic.lower()} at {req.company}.",
            references=[],
        )
    ]
    if req.references:
        reference = req.references[0]
        detail = f"We worked with {reference.client} on {reference.title}."
        if reference.outcome:
            detail += f" {reference.outcome.rstrip('.')}."
        if reference.shared:
            detail += f" The shared focus was {', '.join(reference.shared)}."
        paragraphs.append(agent.EmailParagraph(text=detail[: agent.MAX_PARAGRAPH_CHARS], references=[1]))
    paragraphs.append(
        agent.EmailParagraph(
            text=(
                "Would it help if I shared a short summary?"
                if req.goal == "share"
                else "Would you be open to a short conversation?"
            ),
            references=[],
        )
    )
    checked, _ = agent.check(
        agent.OutreachEmailOutput(subject=f"A note for {req.company}"[:180], paragraphs=paragraphs),
        req,
    )
    if checked is not None:
        return checked
    checked, _ = agent.check(
        agent.OutreachEmailOutput(
            subject="A note",
            paragraphs=[
                agent.EmailParagraph(text=f"Hi {first}, I wanted to reach out.", references=[]),
                agent.EmailParagraph(text="Would you be open to a short conversation?", references=[]),
            ],
        ),
        req,
    )
    return checked


async def _write(
    req: OutreachEmailDraftRequest,
    llm: LLM,
    settings: Settings,
    usage: Usage,
    notes: list[str],
) -> tuple[agent.CheckedEmail | None, list[str]]:
    done = await agent.ask(llm, req, notes, settings.drafting_effort)
    usage.add(done.usage)
    return agent.check(done.value, req)


async def draft_email(
    ctx: ServiceContext,
    req: OutreachEmailDraftRequest,
    llm: LLM,
    cache: ResultCache,
    settings: Settings,
) -> OutreachEmailDraftResponse:
    key = stable_hash("outreach_email", req.model_dump(), llm.model, VERSIONS)
    if cached := await cache.get(ctx.workspace_id, key):
        return OutreachEmailDraftResponse(
            **cached, meta=common.meta(llm.model, VERSIONS, Usage(), cached=True)
        )

    usage = Usage()
    email, problems = await _write(req, llm, settings, usage, [])
    if email is None:
        email, problems = await _write(req, llm, settings, usage, problems)
        if email is None:
            raise RuleViolation("outreach_email", _detail(problems))

    reviewed = await review_agent.review(llm, _review_items(email, req), settings.review_effort)
    usage.add(reviewed.usage)
    if reviewed.failed:
        email, problems = await _write(req, llm, settings, usage, common.notes_from(reviewed.failed))
        if email is None:
            email = _fallback(req)
            if email is None:
                raise RuleViolation("outreach_email", _detail(problems, reviewed))
        else:
            reviewed = await review_agent.review(llm, _review_items(email, req), settings.review_effort)
            usage.add(reviewed.usage)
            if reviewed.failed:
                email = _fallback(req)
                if email is None:
                    raise RuleViolation("outreach_email", _detail([], reviewed))

    body, cited_references = agent.render(email)
    response = OutreachEmailDraftResponse(
        subject=email.subject,
        body=body,
        cited_references=cited_references,
        meta=common.meta(llm.model, VERSIONS, usage, cached=False),
    )
    await cache.put(ctx.workspace_id, key, "outreach_email", response.model_dump(exclude={"meta"}))
    return response
