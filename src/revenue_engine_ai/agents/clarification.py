"""Clarification Agent: writes the questions to settle with the client before pricing.

Each question must point at a requirement or unclear point it was given (`about`), say why we ask,
and state the assumption we will price on if the client does not reply.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from revenue_engine_ai.core.config import ReasoningEffort
from revenue_engine_ai.llm.ports import LLM, AgentCall, Completion
from revenue_engine_ai.rules.text import mentions_amount, near_duplicate

PROMPT_VERSION = "clarification-v2"
MAX_QUESTIONS = 12

Category = Literal["Scope", "Technical", "Commercial", "Timeline"]

INSTRUCTIONS = """You prepare the questions a bid team sends a client before estimating a proposal.

You get the requirements in scope (with whether a past project covers them), the points the RFP
leaves unclear, the engagement model, the questions already on the list and the assumptions already
made.

Write at most the number of new questions asked for below, most important first. Fewer is fine, and none
at all when nothing worth asking is left:
- Ask only what changes the effort, risk or timeline; prefer requirements no past project covers.
- `about` must be the id of the requirement or unclear point the question is about.
- One topic per question, short and polite, no jargon, not leading.
- `why`: one sentence on what the answer changes in our estimate.
- `sample_answer`: a realistic example of what the client might reply.
- `assumption`: the safe, conservative basis we estimate on if the client does not reply.
- Never write prices, rates or amounts. Do not repeat or rephrase a question already on the list, and
  do not ask about something an existing assumption already settles."""


class ClientQuestion(BaseModel):
    category: Category
    about: str
    question: str
    why: str
    sample_answer: str
    assumption: str


class ClarificationOutput(BaseModel):
    questions: list[ClientQuestion]


@dataclass(frozen=True)
class Topic:
    id: str
    text: str


@dataclass(frozen=True)
class ClarificationInput:
    requirements: list[tuple[Topic, bool]]  # (requirement, covered by a past project)
    unclear_points: list[Topic]
    engagement_model: str
    existing_questions: list[str]
    existing_assumptions: list[str] = field(default_factory=list)
    limit: int = MAX_QUESTIONS

    def topics(self) -> dict[str, str]:
        return {t.id: t.text for t, _ in self.requirements} | {t.id: t.text for t in self.unclear_points}


def _input(data: ClarificationInput, notes: list[str]) -> str:
    lines = ["Requirements:"]
    lines += [
        f"- {t.id}: {t.text} ({'covered' if covered else 'NOT covered'})" for t, covered in data.requirements
    ]
    lines += ["", "Unclear points:"] + [f"- {t.id}: {t.text}" for t in data.unclear_points]
    lines += ["", f"Engagement model: {data.engagement_model}", "", "Already on the list:"]
    lines += [f"- {q}" for q in data.existing_questions] or ["- (none)"]
    lines += ["", "Assumptions already made:"]
    lines += [f"- {a}" for a in data.existing_assumptions] or ["- (none)"]
    lines += ["", f"Write at most {data.limit} new questions."]
    if notes:
        lines += ["", "Your previous answer had problems. Fix or drop these:"] + [f"- {n}" for n in notes]
    return "\n".join(lines)


async def ask(
    llm: LLM, data: ClarificationInput, notes: list[str], effort: ReasoningEffort | None
) -> Completion[ClarificationOutput]:
    call = AgentCall("clarification", INSTRUCTIONS, _input(data, notes), ClarificationOutput, effort)
    return await llm.structured(call)


def check(output: ClarificationOutput, data: ClarificationInput) -> tuple[list[ClientQuestion], list[str]]:
    """Questions that pass the rules, and a note for each one dropped."""
    topics = data.topics()
    seen = list(data.existing_questions)
    kept: list[ClientQuestion] = []
    problems: list[str] = []
    for q in output.questions:
        if q.about not in topics:
            problems.append(f"'{q.question}' is about '{q.about}', which is not a given id.")
        elif any(mentions_amount(t) for t in (q.question, q.sample_answer, q.assumption)):
            problems.append(f"'{q.question}' contains an amount.")
        elif any(near_duplicate(q.question, s) for s in seen):
            problems.append(f"'{q.question}' repeats a question already on the list.")
        elif any(near_duplicate(q.question, a) for a in data.existing_assumptions):
            problems.append(f"'{q.question}' is already settled by an assumption.")
        else:
            seen.append(q.question)
            kept.append(q)
    return kept[: min(data.limit, MAX_QUESTIONS)], problems
