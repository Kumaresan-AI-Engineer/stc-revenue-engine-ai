import io

import pytest
from docx import Document
from openpyxl import Workbook

from core.errors import RuleViolation
from documents.chunking import MAX_PASSAGE_CHARS, passages, summary_passage
from documents.parsing import Page, extract_pages, page_windows, render_pages


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


def test_unsupported_unreadable_and_empty_files_are_refused() -> None:
    with pytest.raises(RuleViolation) as kind:
        extract_pages("rfp.pptx", b"x")
    assert kind.value.extra["rule"] == "file_type"
    with pytest.raises(RuleViolation) as broken:
        extract_pages("rfp.docx", b"not a zip")
    assert broken.value.extra["rule"] == "file_unreadable"
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


def test_very_long_sentence_is_split() -> None:
    (page_passages) = passages("e", "rfp", [Page(1, "a" * (MAX_PASSAGE_CHARS * 3))])
    assert len(page_passages) >= 3


def test_summary_passage_is_redacted() -> None:
    p = summary_passage("e", "Fleet platform, value $500k.")
    assert p.section == "summary" and "$" not in p.text
