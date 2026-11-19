"""The code checks around the proposal assistants and the template review, without a model."""

import io

import docx
import pytest
from PIL import Image

from agents import answer_check, assumption, section_writer, sizing, template_analysis
from agents.answer_check import AnswerToCheck, AnswerVerdict
from agents.assumption import AssumptionOutput
from agents.section_writer import SectionWriterOutput
from agents.sizing import Requirement, SizingOutput
from agents.template_analysis import ImageSectionsOutput, TemplateAnalysisOutput
from documents import template_reader
from documents.template_reader import TemplateKind
from rules.text import mentions_money


@pytest.mark.parametrize(
    "text",
    [
        "The price is 40,000 for phase 1.",
        "A 40k fee applies.",
        "Rs. 500 per user.",
        "The total cost will be 25000.",
        "USD 120,000",
        "€ 5 million",
    ],
)
def test_money_is_found(text: str) -> None:
    assert mentions_money(text)


@pytest.mark.parametrize(
    "text",
    [
        "We deliver in 12 weeks with a team of 6.",
        "99.9% availability.",
        "Pricing is provided separately.",
        "A success rate of 95%.",
    ],
)
def test_text_without_money_passes(text: str) -> None:
    assert not mentions_money(text)


def test_section_writer_output_is_plain_text_and_never_money() -> None:
    output = SectionWriterOutput.model_validate(
        {
            "sections": [
                {"key": "a", "paragraphs": ["- **Bold** &amp; <b>tagged</b>", "Fee: EUR 300 per day.", ""]},
                {"key": "b", "paragraphs": [f"Part {n}." for n in range(8)]},
                {"key": "x", "paragraphs": ["Not asked for."]},
            ]
        }
    )
    written, dropped = section_writer.check(output, ["a", "b"])
    assert written == {"a": ["Bold & tagged"], "b": [f"Part {n}." for n in range(5)]}
    assert dropped == ["a: 'Fee: EUR 300 per day.' contains a money amount."]


def test_a_long_paragraph_without_a_sentence_end_is_cut_at_a_word() -> None:
    output = SectionWriterOutput.model_validate({"sections": [{"key": "a", "paragraphs": ["word " * 400]}]})
    paragraph = section_writer.check(output, ["a"])[0]["a"][0]
    assert len(paragraph) <= section_writer.PARAGRAPH_MAX and paragraph.endswith("word…")


def test_assumptions_keep_the_first_known_answer_without_money() -> None:
    output = AssumptionOutput.model_validate(
        {
            "assumptions": [
                {"id": "q1", "assumption": "  We host on AWS. ", "risk": ""},
                {"id": "q1", "assumption": "Second.", "risk": "x"},
                {"id": "q2", "assumption": "Licences cost INR 50,000.", "risk": "x"},
                {"id": "q9", "assumption": "Unknown.", "risk": "x"},
                {"id": "q3", "assumption": "   ", "risk": "x"},
            ]
        }
    )
    kept = assumption.check(output, ["q1", "q2", "q3"])
    assert list(kept) == ["q1"]
    assert (kept["q1"].assumption, kept["q1"].risk) == ("We host on AWS.", assumption.GENERIC_RISK)


@pytest.mark.parametrize(
    ("question", "answer", "reason"),
    [
        ("Which cloud?", "   ", answer_check.NO_WORDS),
        ("Which cloud?", "--", answer_check.NO_WORDS),
        ("Which cloud?", "2024", answer_check.DIGITS_ONLY),
    ],
)
def test_precheck_turns_away_what_is_plainly_not_an_answer(question: str, answer: str, reason: str) -> None:
    verdict = answer_check.precheck(AnswerToCheck(question, "", "", answer))
    assert verdict is not None and not verdict.acceptable and verdict.reason == reason


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("How many trucks?", "400"),
        ("How many trucks?", "about 400"),
        ("Which cloud?", "AWS, in Frankfurt"),
        ("Which cloud?", "AWS"),
        ("Is there SSO?", "Yes"),
    ],
)
def test_precheck_leaves_real_answers_to_the_model(question: str, answer: str) -> None:
    assert answer_check.precheck(AnswerToCheck(question, "", "", answer)) is None


def test_answer_check_texts_are_short_and_never_money() -> None:
    checked = answer_check.check(AnswerVerdict(acceptable=False, reason="", suggestion="Budget USD 5,000."))
    assert checked.reason == answer_check.NOT_AN_ANSWER
    assert checked.suggestion == answer_check.GENERIC_SUGGESTION
    ok = answer_check.check(AnswerVerdict(acceptable=True, reason="Fine.", suggestion="x" * 500))
    assert ok.reason == "" and len(ok.suggestion) == answer_check.SUGGESTION_MAX


def test_sizes_are_one_per_requirement_with_days_from_code() -> None:
    requirements = [Requirement("a", "A", ""), Requirement("b", "B", ""), Requirement("c", "C", "")]
    output = SizingOutput.model_validate(
        {
            "sizes": [
                {"key": "a", "size": "S", "rationale": "One screen."},
                {"key": "b", "size": "L", "rationale": "  "},
                {"key": "z", "size": "XL", "rationale": "Unknown."},
            ]
        }
    )
    assert [(s.key, s.size, s.days) for s in sizing.sizes(output, requirements)] == [
        ("a", "S", 10),
        ("b", "M", 20),
        ("c", "M", 20),
    ]
    assert sizing.SIZE_DAYS == {"XS": 5, "S": 10, "M": 20, "L": 40, "XL": 80}


def _spec(*sections: dict[str, object]) -> dict[str, object]:
    return {"sections": list(sections), "confidence": 0.5, "warnings": ["read"]}


def test_merge_keeps_unadvised_keys_and_never_repeats_a_known_key() -> None:
    spec = _spec(
        {"key": "summary", "title": "Summary", "order": 1, "kind": "generated", "guidance": "g"},
        {"key": "custom", "title": "Intro", "order": 2, "kind": "generated", "guidance": ""},
        {"key": "custom", "title": "Plan", "order": 3, "kind": "generated", "guidance": ""},
    )
    output = TemplateAnalysisOutput.model_validate(
        {
            "sections": [
                {"order": 2, "key": "summary", "kind": "generated", "guidance": "Open warmly."},
                {"order": 3, "key": "Timeline", "kind": "boilerplate", "guidance": ""},  # nothing to fix
            ]
        }
    )
    merged = template_analysis.merge(spec, output, TemplateKind.PROPOSAL)
    sections = merged["sections"]
    assert [(s["key"], s["kind"]) for s in sections] == [
        ("summary", "generated"),
        ("custom", "generated"),
        ("timeline", "generated"),
    ]
    assert sections[1]["guidance"] == "Open warmly." and "fixed_text" not in sections[2]
    assert merged["confidence"] == round(0.5 + 0.3 * 2 / 3, 2)
    assert merged["warnings"] == ["read", template_analysis.REVIEWED_WARNING]


def test_an_unknown_key_from_the_model_is_custom() -> None:
    spec = _spec({"key": "custom", "title": "X", "order": 1, "kind": "generated", "guidance": ""})
    output = TemplateAnalysisOutput.model_validate(
        {"sections": [{"order": 1, "key": "pricing_table", "kind": "generated", "guidance": "g"}]}
    )
    assert template_analysis.merge(spec, output, TemplateKind.PROPOSAL)["sections"][0]["key"] == "custom"


def test_image_sections_are_capped_and_titled_by_code() -> None:
    output = ImageSectionsOutput.model_validate(
        {"sections": [{"title": f"Part {n}", "guidance": ""} for n in range(70)]}
    )
    sections = template_analysis.image_sections(output, TemplateKind.CASE_STUDY)
    assert len(sections) == template_reader.MAX_SECTIONS and {s["key"] for s in sections} == {"custom"}
    assert [s["order"] for s in sections] == list(range(1, 61))


def _docx_with(pictures: list[bytes], headings: list[str]) -> bytes:
    document = docx.Document()
    for picture in pictures:
        document.add_picture(io.BytesIO(picture))
    for heading in headings:
        document.add_heading(heading, level=1)
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def _image(color: tuple[int, int, int], fmt: str) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (10, 10), color).save(out, format=fmt)
    return out.getvalue()


def test_the_reader_counts_headings_and_finds_embedded_pictures() -> None:
    data = _docx_with([_image((1, 2, 3), "PNG"), _image((4, 5, 6), "JPEG"), _image((7, 8, 9), "PNG")], ["A"])
    assert template_reader.heading_count(data) == 1
    images = template_reader.embedded_images(data, limit=2)
    assert [mime for mime, _ in images] == ["image/png", "image/jpeg"]
    assert template_reader.embedded_images(_docx_with([], ["A", "B"]), limit=4) == []
    with pytest.raises(template_reader.UnreadableDocument):
        template_reader.embedded_images(b"not a zip", limit=4)
    with pytest.raises(template_reader.UnreadableDocument):
        template_reader.heading_count(b"not a zip")
