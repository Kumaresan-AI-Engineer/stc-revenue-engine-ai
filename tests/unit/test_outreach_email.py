"""The outreach writer's budget, retry and failure detail, without a model."""

import uuid

import pytest

from revenue_engine_ai.api.schemas import OutreachEmailDraftRequest, OutreachReference
from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.errors import RuleViolation
from revenue_engine_ai.core.security import ServiceContext
from revenue_engine_ai.services import outreach_email

from tests.fakes import FakeLLM, InMemoryResultCache, approve_all

REQUEST = OutreachEmailDraftRequest(
    recipient_name="Katrin Vogel",
    recipient_title="Head of Operations",
    company="Baltic Cold Chain",
    industry="Logistics",
    notes="",
    known=[],
    priorities=["Operational efficiency"],
    signals=["New depot"],
    goal="share",
    tone="Consultative",
    length="medium",
    variant="A",
    sender_name="Sri",
    references=[
        OutreachReference(
            client="Harbourline Ports",
            title="Port operations platform",
            year=2025,
            outcome="",
            shared=["Analytics"],
        )
    ],
)


def _ctx() -> ServiceContext:
    return ServiceContext(
        workspace_id=uuid.uuid4(),
        actor_id=uuid.uuid4(),
        can_see_restricted=False,
        request_id=None,
    )


def _email(words: int) -> dict[str, object]:
    text = " ".join(["word"] * words)
    return {
        "subject": "An analytics idea for Baltic Cold Chain",
        "paragraphs": [
            {"text": text, "references": [1]},
            {"text": text, "references": [1]},
            {"text": text, "references": [1]},
        ],
    }


async def test_the_brief_carries_the_word_budget_and_a_grounded_email_passes() -> None:
    llm = FakeLLM()
    llm.on("outreach_email", lambda call: _email(30))
    llm.on("review", approve_all)

    result = await outreach_email.draft_email(_ctx(), REQUEST, llm, InMemoryResultCache(), Settings())

    assert "Word budget: aim for about 110 words and never exceed 130 words" in llm.calls[0].input
    assert result.subject == "An analytics idea for Baltic Cold Chain"
    assert result.cited_references == [1]
    assert "{{c1}}" in result.body


async def test_an_over_budget_draft_is_retried_with_the_reason() -> None:
    llm = FakeLLM()
    calls = 0

    def write(call: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _email(50 if calls == 1 else 30)

    llm.on("outreach_email", write)
    llm.on("review", approve_all)

    result = await outreach_email.draft_email(_ctx(), REQUEST, llm, InMemoryResultCache(), Settings())

    assert llm.count("outreach_email") == 2
    assert "Fix these issues in the next draft" in llm.calls[1].input
    assert "at most 130 words" in llm.calls[1].input
    assert "{{c1}}" in result.body


async def test_a_draft_that_stays_over_budget_reports_the_reason() -> None:
    llm = FakeLLM()
    llm.on("outreach_email", lambda call: _email(50))
    llm.on("review", approve_all)

    with pytest.raises(RuleViolation) as caught:
        await outreach_email.draft_email(_ctx(), REQUEST, llm, InMemoryResultCache(), Settings())

    assert caught.value.extra["rule"] == "outreach_email"
    assert "130 words" in caught.value.detail
    assert llm.count("review") == 0
