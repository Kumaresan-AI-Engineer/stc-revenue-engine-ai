"""The code checks that run on every agent answer, without a model."""

from revenue_engine_ai.agents import clarification, drafting, extraction
from revenue_engine_ai.agents.clarification import (
    ClarificationInput,
    ClarificationOutput,
    ClientQuestion,
    Topic,
)
from revenue_engine_ai.agents.drafting import Claim, DraftOutput
from revenue_engine_ai.agents.extraction import CustomRequirement, Extraction, FeatureHit, RfpQuestion
from revenue_engine_ai.documents.parsing import Page
from revenue_engine_ai.store.ports import StoredPassage

PAGES = [
    Page(1, "Overview. The client needs real-time vehicle tracking."),
    Page(2, "Drivers need an offline app."),
]


def _extraction(**kw: object) -> Extraction:
    base: dict[str, object] = {
        "summary": " Fleet platform. ",
        "features": [],
        "custom_requirements": [],
        "questions": [],
        "required_documents": [],
        "unclear_points": [],
        "facts": [],
    }
    return Extraction.model_validate(base | kw)


def test_extraction_check_fixes_pages_and_rejects_invented_quotes() -> None:
    result = _extraction(
        features=[
            FeatureHit(key="tracking", quote="real-time vehicle tracking", page=2),  # wrong page
            FeatureHit(key="driverapp", quote="drivers need a flying app", page=2),  # not in the text
            FeatureHit(key="unknown", quote="Overview", page=1),  # not in the catalogue
        ],
        custom_requirements=[CustomRequirement(label="Offline mode", quote="offline app", page=2)],
    )
    checked, rejected = extraction.check(result, PAGES, {"tracking", "driverapp"})
    assert [(f.key, f.page) for f in checked.features] == [("tracking", 1)]
    assert [c.label for c in checked.custom_requirements] == ["Offline mode"]
    assert [r.item for r in rejected] == ["feature 'driverapp'"]
    assert checked.summary == "Fleet platform."


def test_merge_keeps_first_mention_and_first_summary() -> None:
    first = _extraction(summary="First.", questions=[RfpQuestion(text="Describe SLA.", quote="q", page=1)])
    second = _extraction(
        summary="Second.",
        questions=[
            RfpQuestion(text="describe sla", quote="q", page=9),
            RfpQuestion(text="Describe security.", quote="q", page=9),
        ],
    )
    merged = extraction.merge([first, second])
    assert merged.summary == "First."
    assert [(q.text, q.page) for q in merged.questions] == [("Describe SLA.", 1), ("Describe security.", 9)]


def test_extraction_form_restricts_keys_to_the_catalogue() -> None:
    form = extraction.output_model(("tracking", "iot"))
    schema = form.model_json_schema()
    assert schema["$defs"]["CatalogueFeatureHit"]["properties"]["key"]["enum"] == ["tracking", "iot"]


RETRIEVED = {"p1": StoredPassage("p1", "e1", "proposal", 4, "We are ISO 27001 certified since 2019.")}


def test_draft_check_accepts_grounded_claims() -> None:
    out = DraftOutput(
        answer="We hold ISO 27001.",
        claims=[Claim(statement="ISO 27001", passage_id="p1", quote="ISO 27001 certified")],
        enough_material=True,
    )
    claims, problems = drafting.check(out, RETRIEVED)
    assert len(claims) == 1 and problems == []


def test_draft_check_flags_unread_passages_wrong_quotes_and_amounts() -> None:
    out = DraftOutput(
        answer="We hold ISO 27001 and charge €900 a day.",
        claims=[
            Claim(statement="a", passage_id="p9", quote="x"),
            Claim(statement="b", passage_id="p1", quote="SOC 2 certified"),
        ],
        enough_material=True,
    )
    claims, problems = drafting.check(out, RETRIEVED)
    assert claims == []
    assert len(problems) == 4  # unread passage, missing quote, amount, no grounded claim


def _clarify_input() -> ClarificationInput:
    return ClarificationInput(
        requirements=[
            (Topic("tracking", "Real-time tracking"), True),
            (Topic("x_1", "Dispatch screen"), False),
        ],
        unclear_points=[Topic("u1", "Number of vehicles not stated")],
        engagement_model="Fixed cost",
        existing_questions=["How many vehicles are in the fleet?"],
    )


def _q(about: str, question: str, assumption: str = "Up to 500 vehicles.") -> ClientQuestion:
    return ClientQuestion(
        category="Scope", about=about, question=question, why="w", sample_answer="s", assumption=assumption
    )


def test_clarification_check_applies_the_rules() -> None:
    out = ClarificationOutput(
        questions=[
            _q("x_1", "Who uses the dispatch screen?"),
            _q("nope", "Unknown topic?"),
            _q("u1", "How many vehicles are in the fleet?"),  # already on the list
            _q("tracking", "Is a budget of €50k fine?", "Budget €50k."),
            _q("x_1", "Who uses the dispatch screen ?"),  # repeat
        ]
    )
    kept, problems = clarification.check(out, _clarify_input())
    assert [q.question for q in kept] == ["Who uses the dispatch screen?"]
    assert len(problems) == 4
