from revenue_engine_ai.rules.text import (
    AMOUNT_PLACEHOLDER,
    contains_quote,
    dedupe_key,
    is_pricing_question,
    mentions_amount,
    redact_amounts,
    stable_hash,
)

SOURCE = (
    "The platform shall provide “real-time” tracking of every vehicle.\nDrivers  use a mobile app — offline."
)


def test_quote_found_despite_smart_quotes_case_and_spacing() -> None:
    assert contains_quote(SOURCE, 'the platform shall provide "real-time" TRACKING')
    assert contains_quote(SOURCE, "drivers use a mobile app - offline")


def test_quote_with_ellipsis_needs_parts_in_order() -> None:
    assert contains_quote(SOURCE, "The platform shall provide ... every vehicle")
    assert not contains_quote(SOURCE, "every vehicle ... The platform shall provide")


def test_invented_or_empty_quote_is_not_found() -> None:
    assert not contains_quote(SOURCE, "The platform shall provide drone delivery")
    assert not contains_quote(SOURCE, " ... ")


def test_amounts_are_detected_and_redacted() -> None:
    for text in ("€820k", "$ 1,200", "EUR 45000", "12,500 euros", "INR 3.2m"):
        assert mentions_amount(text), text
        assert AMOUNT_PLACEHOLDER in redact_amounts(f"Total {text} fixed.")
    assert not mentions_amount("We delivered 12 releases in 2024 for 3 clients.")


def test_pricing_questions_are_recognised() -> None:
    assert is_pricing_question("What is your proposed day rate for a senior engineer?")
    assert is_pricing_question("Provide your pricing for year two.")
    assert not is_pricing_question("Describe your escalation and governance model.")


def test_stable_hash_ignores_dict_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash("x") != stable_hash("y")


def test_dedupe_key_ignores_case_and_punctuation() -> None:
    assert dedupe_key("Real-time Tracking!") == dedupe_key("real time tracking")
