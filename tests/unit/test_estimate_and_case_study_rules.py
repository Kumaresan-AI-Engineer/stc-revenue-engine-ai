"""The code checks around the Estimate Assistant and the Case Study Writer, without a model."""

from agents import case_study_writer, estimate_assistant
from agents.case_study_writer import CaseStudyWriterOutput, SourcePassage, TemplateSection
from agents.estimate_assistant import Action, ActionOut, EstimateAssistantOutput, Limits

LIMITS = Limits(
    in_scope={"tracking": True, "driver_app": False, "x_yard": False},
    out_of_scope=frozenset({"reports"}),
    approaches=frozenset({"legacy_team", "ai_assisted"}),
    approach="legacy_team",
    team_min=2,
    team_max=12,
    can_change_price=True,
    amounts=frozenset({"AMOUNT_1"}),
)


def _answer(*actions: dict, message: str = "Done.", understood: bool = True) -> EstimateAssistantOutput:
    return EstimateAssistantOutput(
        understood=understood, message=message, actions=[ActionOut(**a) for a in actions]
    )


def test_the_actions_the_estimate_allows_are_kept_with_only_their_own_fields() -> None:
    out = _answer(
        {"kind": "leave_out", "key": "driver_app", "weeks": 3},
        {"kind": "add_back", "key": "reports"},
        {"kind": "delivery_approach", "key": "ai_assisted"},
        {"kind": "set_timeline", "weeks": 30},
        {"kind": "set_price", "percent": -20},
    )
    understood, message, actions, notes = estimate_assistant.check(out, LIMITS)
    assert understood and message == "Done." and notes == []
    assert actions == [
        Action("leave_out", key="driver_app"),
        Action("add_back", key="reports"),
        Action("delivery_approach", key="ai_assisted"),
        Action("set_timeline", weeks=30),
        Action("set_price", percent=-20),
    ]


def test_actions_the_estimate_does_not_allow_are_dropped_with_a_note() -> None:
    out = _answer(
        {"kind": "leave_out", "key": "tracking"},  # a must-have
        {"kind": "leave_out", "key": "reports"},  # not in scope
        {"kind": "add_back", "key": "driver_app"},  # not out of scope
        {"kind": "delivery_approach", "key": "legacy_team"},  # already the approach
        {"kind": "delivery_approach", "key": "offshore"},  # not listed
        {"kind": "set_team", "team": 40},
        {"kind": "set_timeline", "weeks": 0},
        {"kind": "set_price", "amount_ref": "AMOUNT_7"},
        {"kind": "set_price", "amount_ref": "AMOUNT_1", "percent": -10},
        {"kind": "set_price", "percent": -95},
        {"kind": "dance"},
    )
    _, _, actions, notes = estimate_assistant.check(out, LIMITS)
    assert actions == [] and len(notes) == 11
    assert "must-have" in notes[0] and "not an action kind" in notes[-1]


def test_one_change_per_group_and_per_item() -> None:
    out = _answer(
        {"kind": "set_timeline", "weeks": 30},
        {"kind": "set_team", "team": 4},
        {"kind": "set_price", "amount_ref": "AMOUNT_1"},
        {"kind": "reset_price"},
        {"kind": "leave_out", "key": "x_yard"},
        {"kind": "leave_out", "key": "x_yard"},
    )
    _, _, actions, notes = estimate_assistant.check(out, LIMITS)
    assert [a.kind for a in actions] == ["set_timeline", "set_price", "leave_out"]
    assert len(notes) == 3


def test_a_person_who_may_not_change_the_price_gets_no_price_action() -> None:
    sales = Limits(**{**LIMITS.__dict__, "can_change_price": False})
    out = _answer({"kind": "set_price", "percent": -20}, {"kind": "reset_price"})
    _, _, actions, notes = estimate_assistant.check(out, sales)
    assert actions == [] and all("may not change the price" in n for n in notes)


def test_a_question_has_no_actions_and_money_never_reaches_the_person() -> None:
    q = estimate_assistant.check(_answer(message="It takes 26 weeks with a team of 6."), LIMITS)
    assert q == (True, "It takes 26 weeks with a team of 6.", [], [])
    _, message, _, notes = estimate_assistant.check(_answer(message="The price is now $40,000."), LIMITS)
    assert message == estimate_assistant.UNCLEAR_MESSAGE and "money" in notes[0]
    cut = _answer({"kind": "set_price", "percent": -20}, message="I cut the cost by 20%.")
    assert estimate_assistant.check(cut, LIMITS)[1] == "I cut the cost by 20%."  # a percentage is not money
    unclear = _answer({"kind": "set_team", "team": 4}, message="", understood=False)
    assert estimate_assistant.check(unclear, LIMITS) == (False, estimate_assistant.UNCLEAR_MESSAGE, [], [])


FACTS = {
    "client": "Bluewater Logistics",
    "title": "Fleet visibility platform",
    "industry": "Logistics",
    "engagement_model": "Fixed cost",
    "outcome": "won",
    "reason": "Fastest go-live",
    "weeks": 26,
    "features": ["Live tracking"],
}
PASSAGES = [SourcePassage("p1", 4, "Dispatch time fell by 35% across 1,200 trucks. Total fee EUR 120,000.")]


def _story(**fields: object) -> CaseStudyWriterOutput:
    base: dict[str, object] = {
        "title": "Live fleet visibility for Bluewater",
        "summary": "Bluewater needed to see every truck.",
        "metrics": [],
        "industry": "Logistics",
        "scope": "Tracking platform",
        "engagement": "Fixed cost, 26 weeks",
        "about_client": "A logistics company.",
        "challenges": [{"title": "No visibility", "text": "Trucks went dark."}],
        "solution_intro": "We proposed a tracking platform.",
        "solution_pillars": [{"title": "Live map", "text": "Every truck.", "tag": "live"}],
        "features": ["Live tracking", ""],
        "why_us": "Fastest go-live.",
        "sections": [],
        "sources": ["p1", "p-unknown", "p1"],
    }
    return CaseStudyWriterOutput.model_validate(base | fields)


def test_metrics_only_from_the_facts_or_the_passages_and_never_money() -> None:
    metrics = [
        {"v": "35%", "l": "Faster dispatch"},  # in the passage
        {"v": "26 wk", "l": "Delivery timeline"},  # in the facts
        {"v": "1,200", "l": "Trucks tracked"},  # in the passage, thousands separator
        {"v": "50%", "l": "Fewer calls"},  # invented
        {"v": "2", "l": "Weeks early"},  # "2" only inside 120,000 and 1,200: not written as a number
        {"v": "EUR 120,000", "l": "Contract value"},  # money
    ]
    draft, notes = case_study_writer.check(_story(metrics=metrics), FACTS, PASSAGES, [])
    assert draft.metrics == [
        {"v": "35%", "l": "Faster dispatch"},
        {"v": "26 wk", "l": "Delivery timeline"},
        {"v": "1,200", "l": "Trucks tracked"},
    ]
    assert len(notes) == 3 and "money" in notes[-1]


def test_the_story_is_plain_text_within_the_limits_and_cites_known_passages() -> None:
    story = _story(
        summary="**Bold** <b>claim</b> " + "word " * 400,
        why_us="We won because the price was 40,000.",
        solution_pillars=[{"title": "Live map", "text": "x", "tag": "live"}] * 8,
    )
    draft, notes = case_study_writer.check(story, FACTS, PASSAGES, [])
    assert draft.summary.startswith("Bold claim word") and len(draft.summary) <= case_study_writer.SUMMARY_MAX
    assert draft.why_us == "" and "why_us" in notes[0]
    assert len(draft.solution_pillars) == 6 and draft.solution_pillars[0]["tag"] == "LIVE"
    assert draft.features == ["Live tracking"] and draft.sources == ["p1"]
    assert draft.engagement == "Fixed cost, 26 weeks"  # the engagement model name is not money
    _, notes = case_study_writer.check(_story(title="", summary=" "), FACTS, PASSAGES, [])
    assert "title: write a title." in notes and "summary: write a summary." in notes


def test_every_template_section_is_written_in_its_order_as_plain_paragraphs_without_money() -> None:
    template = [
        TemplateSection("s1", "Summary", "One paragraph."),
        TemplateSection("s2", "The challenge", "The client's view."),
        TemplateSection("s3", "Results", "Numbers from the proposal only."),
    ]
    story = _story(
        sections=[
            {"key": "s2", "paragraphs": ["**Trucks** went dark between depots.", "The fee was EUR 120,000."]},
            {"key": "s1", "paragraphs": ["Bluewater sees every truck."] * 7},
            {"key": "s9", "paragraphs": ["Not a template section."]},
        ]
    )
    draft, notes = case_study_writer.check(story, FACTS, PASSAGES, template)
    assert [s["key"] for s in draft.sections] == ["s1", "s2"]  # the template's order; an unknown key dropped
    assert len(draft.sections[0]["paragraphs"]) == 5  # at most MAX_PARAGRAPHS
    assert draft.sections[1]["paragraphs"] == ["Trucks went dark between depots."]  # plain, money dropped
    assert any("money" in n for n in notes) and "section 'Results' (s3): write it." in notes
