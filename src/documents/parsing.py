"""Turn an uploaded document into numbered pages of plain text. Deterministic; no model involved.

PDF pages are real pages. Word and Excel have no fixed pages, so a Word "page" is a block of
paragraphs and an Excel "page" is one sheet; the number is what a citation points at.
"""

import io
from dataclasses import dataclass

from core.errors import RuleViolation

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


def extract_pages(filename: str, data: bytes) -> list[Page]:
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            texts = _pdf(data)
        elif name.endswith(".docx"):
            texts = _docx(data)
        elif name.endswith((".xlsx", ".xlsm")):
            texts = _xlsx(data)
        elif name.endswith(TEXT_EXTENSIONS):
            texts = [data.decode("utf-8", errors="replace")]
        else:
            raise RuleViolation("file_type", "Upload a PDF, Word (.docx), Excel (.xlsx) or text file.")
    except RuleViolation:
        raise
    except Exception as exc:
        raise RuleViolation("file_unreadable", f"We could not read '{filename}'.") from exc
    pages = [Page(i + 1, t.strip()) for i, t in enumerate(texts) if t.strip()]
    if not pages:
        # Scanned PDFs have no text layer; OCR is a later step.
        raise RuleViolation("file_empty", f"'{filename}' has no readable text (is it a scan?).")
    return pages


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
