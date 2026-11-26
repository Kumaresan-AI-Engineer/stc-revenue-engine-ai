import io

import pytest
from docx import Document
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches

from revenue_engine_ai.core.errors import RuleViolation
from revenue_engine_ai.documents.chunking import MAX_PASSAGE_CHARS, passages, summary_passage
from revenue_engine_ai.documents.parsing import Page, extract_pages, page_windows, render_pages


def _docx(paragraphs: list[str]) -> bytes:
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def _xlsx() -> bytes:
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "Requirements"
    sheet.append(["ID", "Requirement"])
    sheet.append(["R1", "Real-time tracking"])
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def _pptx() -> bytes:
    deck = Presentation()
    first = deck.slides.add_slide(deck.slide_layouts[5])  # title only
    first.shapes.title.text = "Fleet tracking platform"
    first.notes_slide.notes_text_frame.text = "Delivered in 14 weeks"
    deck.slides.add_slide(deck.slide_layouts[6])  # blank: nothing to read
    table_slide = deck.slides.add_slide(deck.slide_layouts[6])
    grid = table_slide.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(1)).table
    for (row, col), value in {(0, 0): "KPI", (0, 1): "Result", (1, 0): "Uptime", (1, 1): "99.9%"}.items():
        grid.cell(row, col).text = value
    group = table_slide.shapes.add_group_shape()
    group.shapes.add_textbox(
        Inches(1), Inches(3), Inches(3), Inches(1)
    ).text_frame.text = "ISO 27001 certified"
    out = io.BytesIO()
    deck.save(out)
    return out.getvalue()


def test_text_file_is_one_page() -> None:
    assert extract_pages("rfp.txt", b"Hello\nworld") == [Page(1, "Hello\nworld")]


def test_word_paragraphs_are_grouped_into_pages() -> None:
    pages = extract_pages("rfp.docx", _docx([f"Paragraph {i}" for i in range(45)]))
    assert [p.number for p in pages] == [1, 2]
    assert "Paragraph 44" in pages[1].text


def test_excel_sheet_is_a_page() -> None:
    (page,) = extract_pages("rfp.xlsx", _xlsx())
    assert page.text.startswith("Sheet: Requirements")
    assert "R1 | Real-time tracking" in page.text


def test_powerpoint_slide_is_a_page_with_its_table_group_and_notes() -> None:
    first, second = extract_pages("capability.pptx", _pptx())
    assert (first.number, second.number) == (1, 3)  # the blank slide is skipped, numbers stay the slides'
    assert "Fleet tracking platform" in first.text and "Notes: Delivered in 14 weeks" in first.text
    assert "KPI | Result" in second.text and "Uptime | 99.9%" in second.text
    assert "ISO 27001 certified" in second.text


def test_unsupported_unreadable_and_empty_files_are_refused() -> None:
    with pytest.raises(RuleViolation) as kind:
        extract_pages("rfp.ppt", b"x")
    assert kind.value.extra["rule"] == "file_type"
    with pytest.raises(RuleViolation) as broken:
        extract_pages("rfp.docx", b"not a zip")
    assert broken.value.extra["rule"] == "file_unreadable"
    with pytest.raises(RuleViolation) as broken_deck:
        extract_pages("deck.pptx", b"not a zip")
    assert broken_deck.value.extra["rule"] == "file_unreadable"
    with pytest.raises(RuleViolation) as empty:
        extract_pages("rfp.txt", b"   ")
    assert empty.value.extra["rule"] == "file_empty"


def test_windows_keep_page_order_and_size() -> None:
    pages = [Page(i, "x" * 40) for i in range(1, 6)]
    windows = page_windows(pages, max_chars=100)
    assert [[p.number for p in w] for w in windows] == [[1, 2], [3, 4], [5]]
    assert render_pages(windows[0]).startswith("[Page 1]\n")


def test_passages_are_deterministic_redacted_and_bounded() -> None:
    text = " ".join(f"Sentence {i} about fleet tracking costs €12k." for i in range(200))
    first = passages("entry-1", "proposal", [Page(3, text)])
    again = passages("entry-1", "proposal", [Page(3, text)])
    assert first == again
    assert len({p.id for p in first}) == len(first)
    assert all(len(p.text) <= MAX_PASSAGE_CHARS and p.page == 3 for p in first)
    assert not any("€" in p.text for p in first)


def test_list_items_are_passages_of_their_own() -> None:
    page = Page(
        2,
        "Scope\n• Plagiarism checking and rubric-based grading\n- LTI 1.3 support for third-party tools\n"
        "3. Migration of 6 years of course content\n17,000 students use it daily\nShort line",
    )
    found = passages("e", "rfp", [page])
    assert found[0].text.startswith("Scope")  # the packed passage first, for context
    assert [p.text for p in found[1:]] == [
        "Plagiarism checking and rubric-based grading",
        "LTI 1.3 support for third-party tools",
        "Migration of 6 years of course content",
        "17,000 students use it daily",  # a number that is not an item marker is kept
    ]
    assert all(p.page == 2 for p in found) and len({p.id for p in found}) == len(found)


def test_a_single_line_is_not_indexed_twice() -> None:
    assert len(passages("e", "rfp", [Page(1, "We delivered an offline driver mobile app.")])) == 1


def test_very_long_sentence_is_split() -> None:
    (page_passages) = passages("e", "rfp", [Page(1, "a" * (MAX_PASSAGE_CHARS * 3))])
    assert len(page_passages) >= 3


def test_summary_passage_is_redacted() -> None:
    p = summary_passage("e", "Fleet platform, value $500k.")
    assert p.section == "summary" and "$" not in p.text
