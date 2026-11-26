"""Turn an uploaded document into numbered pages of plain text. Deterministic; no model involved.

PDF pages are real pages. Word and Excel have no fixed pages, so a Word "page" is a block of
paragraphs, an Excel "page" is one sheet and a PowerPoint "page" is one slide; the number is what a citation
points at. A PDF page with no
text layer (a scan) is read with OCR when the caller allows it (`ocr_max_pages`).
"""

import io
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from revenue_engine_ai.core.errors import RuleViolation
from revenue_engine_ai.documents.ocr import ocr_pdf_pages

log = logging.getLogger(__name__)

WORD_PARAGRAPHS_PER_PAGE = 40
TEXT_EXTENSIONS = (".txt", ".md", ".eml")


@dataclass(frozen=True)
class Page:
    number: int
    text: str


def _pdf(data: bytes) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]


def _docx(data: bytes) -> list[str]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        lines.extend(" | ".join(c.text.strip() for c in row.cells) for row in table.rows)
    step = WORD_PARAGRAPHS_PER_PAGE
    return ["\n".join(lines[i : i + step]) for i in range(0, len(lines), step)]


def _xlsx(data: bytes) -> list[str]:
    from openpyxl import load_workbook

    book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    pages = []
    for sheet in book.worksheets:
        rows = (" | ".join(str(v) for v in row if v is not None) for row in sheet.iter_rows(values_only=True))
        pages.append(f"Sheet: {sheet.title}\n" + "\n".join(r for r in rows if r))
    return pages


def _slide_lines(shapes: Iterable[Any]) -> list[str]:
    """The text of a slide's shapes in reading order: text frames, table rows, and what a group holds."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    lines: list[str] = []
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            lines.extend(_slide_lines(shape.shapes))
        elif shape.has_table:
            lines.extend(" | ".join(c.text.strip() for c in row.cells) for row in shape.table.rows)
        elif shape.has_text_frame and shape.text_frame.text.strip():
            lines.append(shape.text_frame.text.strip())
    return lines


def _pptx(data: bytes) -> list[str]:
    """One page per slide, so a citation's page is the slide number; the speaker notes follow the slide."""
    from pptx import Presentation

    pages = []
    for slide in Presentation(io.BytesIO(data)).slides:
        lines = _slide_lines(slide.shapes)
        notes = slide.notes_slide.notes_text_frame if slide.has_notes_slide else None
        if notes is not None and notes.text.strip():
            lines.append(f"Notes: {notes.text.strip()}")
        pages.append("\n".join(lines))
    return pages


@dataclass(frozen=True)
class ReadDocument:
    pages: list[Page]
    ocr: bool  # some pages were read with OCR


def _with_ocr(data: bytes, texts: list[str], max_pages: int) -> tuple[list[str], bool]:
    """Fill the pages without a text layer by OCR, the first `max_pages` of them."""
    blank = [i for i, t in enumerate(texts) if not t.strip()][:max_pages]
    if not blank:
        return texts, False
    try:
        read = ocr_pdf_pages(data, blank)
    except Exception:
        # OCR is a best effort: without it the scan is refused below with a clear reason.
        log.exception("ocr_failed")
        return texts, False
    return [read.get(i, t) for i, t in enumerate(texts)], any(v.strip() for v in read.values())


def read_document(filename: str, data: bytes, *, ocr_max_pages: int = 0) -> ReadDocument:
    name = filename.lower()
    ocr = False
    try:
        if name.endswith(".pdf"):
            texts = _pdf(data)
            if ocr_max_pages:
                texts, ocr = _with_ocr(data, texts, ocr_max_pages)
        elif name.endswith(".docx"):
            texts = _docx(data)
        elif name.endswith((".xlsx", ".xlsm")):
            texts = _xlsx(data)
        elif name.endswith(".pptx"):
            texts = _pptx(data)
        elif name.endswith(TEXT_EXTENSIONS):
            texts = [data.decode("utf-8", errors="replace")]
        else:
            raise RuleViolation(
                "file_type", "Upload a PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx) or text file."
            )
    except RuleViolation:
        raise
    except Exception as exc:
        raise RuleViolation("file_unreadable", f"We could not read '{filename}'.") from exc
    pages = [Page(i + 1, t.strip()) for i, t in enumerate(texts) if t.strip()]
    if not pages:
        raise RuleViolation("file_empty", f"'{filename}' has no readable text.")
    return ReadDocument(pages, ocr)


def extract_pages(filename: str, data: bytes, *, ocr_max_pages: int = 0) -> list[Page]:
    return read_document(filename, data, ocr_max_pages=ocr_max_pages).pages


def page_windows(pages: list[Page], max_chars: int) -> list[list[Page]]:
    """Consecutive groups of pages, each within max_chars, for documents too long for one call."""
    windows: list[list[Page]] = []
    current: list[Page] = []
    size = 0
    for page in pages:
        if current and size + len(page.text) > max_chars:
            windows.append(current)
            current, size = [], 0
        current.append(page)
        size += len(page.text)
    if current:
        windows.append(current)
    return windows


def render_pages(pages: list[Page]) -> str:
    return "\n\n".join(f"[Page {p.number}]\n{p.text}" for p in pages)
