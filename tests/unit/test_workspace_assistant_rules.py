"""The Workspace Assistant's code checks: cited ids, traceable figures and the no-money rule (no LLM)."""

from revenue_engine_ai.agents.workspace_assistant import (
    NO_MONEY_ANSWER,
    NOT_ANSWERED,
    AssistantOutput,
    Context,
    check,
    untraced_figures,
)

CONTEXT = Context(
    ids=frozenset({"F1", "F2", "D1"}),
    text="Bid pipeline outcomes: won 7, lost 5, win rate 58.3%. Library: 1,240 person-days on Nordwind.",
)


def _out(**overrides: object) -> AssistantOutput:
    base: dict[str, object] = {
        "answered": True,
        "answer": "You won 7 bids and lost 5 (win rate 58.3%).",
        "how": "Counted from the bid pipeline (F1).",
        "source_ids": ["F1"],
        "follow_ups": ["Which bids did we lose?"],
    }
    return AssistantOutput.model_validate({**base, **overrides})


def test_a_traced_answer_with_its_sources_passes() -> None:
    answer, notes = check(_out(), CONTEXT)
    assert notes == [] and answer.verified and answer.answered
    assert answer.source_ids == ["F1"] and answer.follow_ups == ["Which bids did we lose?"]


def test_unknown_source_ids_are_dropped_and_noted() -> None:
    answer, notes = check(_out(source_ids=["F9", "F1", "F1"]), CONTEXT)
    assert answer.source_ids == ["F1"] and any("F9" in n for n in notes)


def test_an_answer_without_sources_is_noted() -> None:
    _, notes = check(_out(source_ids=[]), CONTEXT)
    assert any("source_ids" in n for n in notes)


def test_a_figure_not_in_the_facts_is_flagged() -> None:
    answer, notes = check(_out(answer="You won 9 bids."), CONTEXT)
    assert answer.verified is False and any("9" in n for n in notes)


def test_figures_are_matched_without_thousands_separators() -> None:
    assert untraced_figures("We spent 1240 person-days.", CONTEXT.text) == []
    assert untraced_figures("58.3% and 12", CONTEXT.text) == ["12"]


def test_a_money_amount_replaces_the_answer() -> None:
    answer, notes = check(_out(answer="The price was $40,000.", source_ids=["D1"]), CONTEXT)
    assert answer.answer == NO_MONEY_ANSWER and answer.answered is False and answer.source_ids == []
    assert any("money" in n for n in notes)


def test_an_empty_answer_gets_the_default_message() -> None:
    answer, notes = check(_out(answered=False, answer="", source_ids=[]), CONTEXT)
    assert answer.answer == NOT_ANSWERED and notes == []


def test_lines_are_kept_and_follow_ups_capped() -> None:
    output = _out(answer="Two bids:\n-  Nordwind   won 7\n\n- Northbay", follow_ups=["a", "b", "c", "d", " "])
    answer, _ = check(output, CONTEXT)
    assert answer.answer == "Two bids:\n- Nordwind won 7\n- Northbay" and answer.follow_ups == ["a", "b", "c"]


def test_a_chat_reply_needs_no_sources() -> None:
    reply = "Hi! I can tell you about your bids, proposals and case studies."
    answer, notes = check(_out(kind="chat", answer=reply, source_ids=[], how=""), CONTEXT)
    assert notes == [] and answer.kind == "chat" and answer.answer == reply and answer.source_ids == []


def test_a_general_answer_needs_no_sources_and_its_figures_are_not_traced() -> None:
    reply = "An RFP is a request for proposal. Most teams answer one in 2 to 4 weeks."
    answer, notes = check(
        _out(kind="general", answer=reply, source_ids=[], how="General best practice."), CONTEXT
    )
    assert notes == [] and answer.kind == "general" and answer.verified and answer.answered


def test_an_answer_that_cites_workspace_sources_is_a_data_answer_and_is_checked() -> None:
    answer, notes = check(_out(kind="general", answer="You won 99 bids.", source_ids=["F1"]), CONTEXT)
    assert answer.kind == "data" and not answer.verified and any("99" in n for n in notes)


def test_a_general_answer_still_may_not_write_money() -> None:
    answer, notes = check(
        _out(kind="general", answer="A day rate of $900 is common.", source_ids=[]), CONTEXT
    )
    assert answer.answer == NO_MONEY_ANSWER and any("money" in n for n in notes)
