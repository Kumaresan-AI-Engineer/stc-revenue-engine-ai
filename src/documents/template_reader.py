"""template_extract: read a template (DOCX) into the backend's TemplateSpec shape, without the model.

The document's real headings become sections (a proposal template's mapped to the backend's section keys
where the titles match; a case study template's all `custom`, written for each case study when the heading
names a part of one and fixed text otherwise), its styles give fonts and colours, its first section the page,
header and footer. Deterministic and pure (no I/O). The backend validates and cleans the result (docgen
apply_extraction) and an admin reviews it before it is used. The document's text is copied in as data and
never interpreted.

Kept in step with the backend's local reader (stc-revenue-engine-api modules/docgen/mock_ai.py).
"""

import html
import io
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import docx
from docx.document import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.section import Section, _Footer, _Header
from docx.shared import Length
from docx.styles.style import ParagraphStyle
from docx.text.paragraph import Paragraph


class TemplateKind(StrEnum):
    PROPOSAL = "proposal"
    CASE_STUDY = "case_study"


class SectionKey(StrEnum):
    """The backend's proposal sections (docgen spec.SectionKey), or `custom`: one the template adds."""

    SUMMARY = "summary"
    REQUIREMENTS = "requirements"
    SOLUTION = "solution"
    RESPONSES = "responses"
    CLARIFICATIONS = "clarifications"
    COMMERCIAL_CLARIFICATIONS = "commercial_clarifications"
    ASSUMPTIONS = "assumptions"
    TIMELINE = "timeline"
    EFFORT = "effort"
    COMMERCIALS = "commercials"
    EXPERIENCE = "experience"
    ATTACHMENTS = "attachments"
    CUSTOM = "custom"


MAX_SECTIONS = 60
STANDARD_SECTIONS: tuple[tuple[SectionKey, str], ...] = (
    (SectionKey.SUMMARY, "Executive summary"),
    (SectionKey.REQUIREMENTS, "Our understanding of your requirements"),
    (SectionKey.SOLUTION, "Proposed solution and scope"),
    (SectionKey.RESPONSES, "Responses to the RFP"),
    (SectionKey.CLARIFICATIONS, "Clarifications from you"),
    (SectionKey.COMMERCIAL_CLARIFICATIONS, "Commercial clarifications"),
    (SectionKey.ASSUMPTIONS, "Assumptions"),
    (SectionKey.TIMELINE, "Timeline"),
    (SectionKey.EFFORT, "Delivery effort"),
    (SectionKey.COMMERCIALS, "Commercials"),
    (SectionKey.EXPERIENCE, "Relevant experience"),
    (SectionKey.ATTACHMENTS, "Attachments"),
)

REVIEW_WARNING = "Read from the document's headings and styles. Check every section before approving."
_GUIDANCE_MAX = 2000
_FIXED_TEXT_MAX = 18000
_LABEL_MAX = 100
_LINE_MAX = 500

# Phrases in a heading that name one of our sections, tried longest first ("commercial clarifications"
# before "commercial"). A heading that matches none, or a section already taken, is custom.
_SECTION_PHRASES: dict[str, SectionKey] = {
    "commercial clarifications": SectionKey.COMMERCIAL_CLARIFICATIONS,
    "executive summary": SectionKey.SUMMARY,
    "summary": SectionKey.SUMMARY,
    "overview": SectionKey.SUMMARY,
    "requirements": SectionKey.REQUIREMENTS,
    "understanding": SectionKey.REQUIREMENTS,
    "solution": SectionKey.SOLUTION,
    "approach": SectionKey.SOLUTION,
    "methodology": SectionKey.SOLUTION,
    "scope": SectionKey.SOLUTION,
    "responses": SectionKey.RESPONSES,
    "response": SectionKey.RESPONSES,
    "compliance": SectionKey.RESPONSES,
    "clarifications": SectionKey.CLARIFICATIONS,
    "assumptions": SectionKey.ASSUMPTIONS,
    "timeline": SectionKey.TIMELINE,
    "schedule": SectionKey.TIMELINE,
    "milestones": SectionKey.TIMELINE,
    "project plan": SectionKey.TIMELINE,
    "effort": SectionKey.EFFORT,
    "estimate": SectionKey.EFFORT,
    "commercials": SectionKey.COMMERCIALS,
    "commercial": SectionKey.COMMERCIALS,
    "pricing": SectionKey.COMMERCIALS,
    "price": SectionKey.COMMERCIALS,
    "fees": SectionKey.COMMERCIALS,
    "experience": SectionKey.EXPERIENCE,
    "case studies": SectionKey.EXPERIENCE,
    "references": SectionKey.EXPERIENCE,
    "credentials": SectionKey.EXPERIENCE,
    "attachments": SectionKey.ATTACHMENTS,
    "appendix": SectionKey.ATTACHMENTS,
    "appendices": SectionKey.ATTACHMENTS,
    "annex": SectionKey.ATTACHMENTS,
}
_PHRASES = sorted(_SECTION_PHRASES, key=len, reverse=True)

# Words in a case study template's heading that name a part written for each case study. Any other heading
# with text under it is the same in every case study (e.g. "About us").
_CASE_STUDY_PARTS = frozenset(
    {
        "summary",
        "overview",
        "glance",
        "client",
        "customer",
        "background",
        "challenge",
        "challenges",
        "problem",
        "solution",
        "approach",
        "results",
        "result",
        "outcome",
        "outcomes",
        "impact",
        "benefits",
        "technology",
        "technologies",
        "stack",
        "testimonial",
        "quote",
        "timeline",
        "team",
        "services",
    }
)
CASE_STUDY_SECTIONS = ("Summary", "The client", "The challenge", "Our solution", "Results")

# Cover lines by what they ask for, e.g. "Prepared for: [Client]".
_COVER_LABELS: tuple[tuple[str, str], ...] = (
    ("prepared for", "client"),
    ("prepared by", "prepared_by"),
    ("engagement model", "engagement_model"),
    ("client", "client"),
    ("customer", "client"),
    ("date", "date"),
    ("reference", "reference"),
    ("ref", "reference"),
)
_HEADING = re.compile(r"^heading (\d)$")
_NUMBERING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[ivxlc]+\.)\s+", re.IGNORECASE)
_NOT_WORD = re.compile(r"[^a-z0-9]+")
_ALIGN = {WD_ALIGN_PARAGRAPH.CENTER: "center", WD_ALIGN_PARAGRAPH.RIGHT: "right"}
# (name, short side, long side) in millimetres
_PAPER = (("A4", 210.0, 297.0), ("Letter", 215.9, 279.4), ("Legal", 215.9, 355.6))
_PAPER_TOLERANCE_MM = 2.0
_DRAWING = "./w:r//w:drawing | ./w:r//w:pict"


class UnreadableDocument(Exception):
    """python-docx could not read the file as a Word document."""


@dataclass
class _Heading:
    title: str
    body: list[str] = field(default_factory=list)


def read_template(data: bytes, kind: TemplateKind = TemplateKind.PROPOSAL) -> dict[str, Any]:
    """The template_extract result for this DOCX, as raw JSON (validated by the caller like a callback's).
    Raises UnreadableDocument."""
    try:
        document = docx.Document(io.BytesIO(data))
        return _extract(document, kind)
    except Exception as exc:  # any failure to parse an uploaded file means the same: it cannot be read
        raise UnreadableDocument from exc


def _extract(document: Document, kind: TemplateKind) -> dict[str, Any]:
    paragraphs = list(document.paragraphs)
    level = _section_level(paragraphs)
    cover_part, headings = _split(paragraphs, level)
    warnings = [REVIEW_WARNING]
    reader = _READERS[kind]
    sections = reader.sections(headings)
    if sections:
        recognised = sum(1 for s in sections if reader.recognised(s))
        confidence = round(min(0.9, 0.4 + 0.5 * recognised / len(sections)), 2)
    else:
        warnings.append(reader.fallback_warning)
        confidence = 0.3
        sections = [
            {"key": key.value, "title": title, "order": n, "kind": "generated", "guidance": ""}
            for n, (key, title) in enumerate(reader.standard, start=1)
        ]
    first = document.sections[0]
    return {
        "branding": {
            "colors": _colors(document),
            "fonts": {"heading": _font(document, "Heading 1"), "body": _font(document, "Normal")},
            "logo": _logo(first, cover_part),
        },
        "page": _page(first),
        "header": _header_footer(first.header),
        "footer": _header_footer(first.footer),
        "cover": _cover(cover_part),
        "sections": sections,
        "confidence": confidence,
        "warnings": warnings,
    }


# ---- Sections -------------------------------------------------------------------------------------


def _style_name(paragraph: Paragraph) -> str:
    style = paragraph.style
    return (style.name or "").strip().lower() if style is not None else ""


def _heading_level(paragraph: Paragraph) -> int | None:
    match = _HEADING.match(_style_name(paragraph))
    return int(match.group(1)) if match else None


def _section_level(paragraphs: list[Paragraph]) -> int | None:
    """The top heading level the document uses: its headings at that level are the sections."""
    levels = [lvl for p in paragraphs if (lvl := _heading_level(p)) is not None and p.text.strip()]
    return min(levels, default=None)


def _split(paragraphs: list[Paragraph], level: int | None) -> tuple[list[Paragraph], list[_Heading]]:
    """The paragraphs before the first section (the cover) and each section's heading with its text."""
    cover: list[Paragraph] = []
    headings: list[_Heading] = []
    for p in paragraphs:
        text = p.text.strip()
        if level is not None and _heading_level(p) == level and text:
            headings.append(_Heading(text))
        elif headings:
            if text:
                headings[-1].body.append(text)
        else:
            cover.append(p)
    return cover, headings


def _words(title: str) -> str:
    """The heading's words, lower case, without numbering, padded with spaces for whole-word matching."""
    return " " + _NOT_WORD.sub(" ", _NUMBERING.sub("", title).lower()).strip() + " "


def section_key(title: str) -> SectionKey:
    """Our section a heading names, by its words; CUSTOM when it names none."""
    words = _words(title)
    return next((_SECTION_PHRASES[p] for p in _PHRASES if f" {p} " in words), SectionKey.CUSTOM)


def _sections(headings: list[_Heading]) -> list[dict[str, Any]]:
    taken: set[SectionKey] = set()
    out = []
    for order, heading in enumerate(headings[:MAX_SECTIONS], start=1):
        key = section_key(heading.title)
        if key in taken:
            key = SectionKey.CUSTOM
        taken.add(key)
        body = "\n".join(heading.body)
        section: dict[str, Any] = {"key": key.value, "title": heading.title[:200], "order": order}
        if key == SectionKey.CUSTOM and body:
            # A section of its own with text already written: the same in every proposal.
            section |= {"kind": "boilerplate", "fixed_text": _paragraphs_html(heading.body), "guidance": ""}
        else:
            section |= {"kind": "generated", "guidance": body[:_GUIDANCE_MAX]}
        out.append(section)
    return out


def _case_study_sections(headings: list[_Heading]) -> list[dict[str, Any]]:
    """A case study template's sections, all custom: written for each case study when the heading names a part
    of one (its text is the guidance), fixed text when it does not and has text."""
    out = []
    for order, heading in enumerate(headings[:MAX_SECTIONS], start=1):
        section: dict[str, Any] = {
            "key": SectionKey.CUSTOM.value,
            "title": heading.title[:200],
            "order": order,
        }
        if heading.body and not _names_case_study_part(heading.title):
            section |= {"kind": "boilerplate", "fixed_text": _paragraphs_html(heading.body), "guidance": ""}
        else:
            section |= {"kind": "generated", "guidance": "\n".join(heading.body)[:_GUIDANCE_MAX]}
        out.append(section)
    return out


def _names_case_study_part(title: str) -> bool:
    return bool(set(_words(title).split()) & _CASE_STUDY_PARTS)


@dataclass(frozen=True)
class _Reader:
    """How one kind of template is read: its sections, which of them count as recognised (for the
    confidence), and the sections that stand in when the document has no headings."""

    sections: Callable[[list[_Heading]], list[dict[str, Any]]]
    recognised: Callable[[dict[str, Any]], bool]
    standard: tuple[tuple[SectionKey, str], ...]
    fallback_warning: str


_READERS: dict[TemplateKind, _Reader] = {
    TemplateKind.PROPOSAL: _Reader(
        sections=_sections,
        recognised=lambda section: section["key"] != SectionKey.CUSTOM,
        standard=STANDARD_SECTIONS,
        fallback_warning="No headings were found, so the standard proposal sections are listed.",
    ),
    TemplateKind.CASE_STUDY: _Reader(
        sections=_case_study_sections,
        recognised=lambda section: _names_case_study_part(section["title"]),
        standard=tuple((SectionKey.CUSTOM, title) for title in CASE_STUDY_SECTIONS),
        fallback_warning="No headings were found, so the standard case study sections are listed.",
    ),
}


def _paragraphs_html(lines: Iterable[str]) -> str:
    out, size = [], 0
    for line in lines:
        block = f"<p>{html.escape(line)}</p>"
        size += len(block)
        if size > _FIXED_TEXT_MAX:
            break
        out.append(block)
    return "".join(out)


# ---- Branding, page, header and footer, cover -------------------------------------------------------


def _style(document: Document, name: str) -> ParagraphStyle | None:
    try:
        style = document.styles[name]
    except KeyError:
        return None
    return style if isinstance(style, ParagraphStyle) else None


def _font(document: Document, style_name: str) -> str | None:
    style = _style(document, style_name)
    name = style.font.name if style is not None else None
    return name[:_LABEL_MAX] if name else None


def _color(document: Document, style_name: str) -> str | None:
    style = _style(document, style_name)
    rgb = style.font.color.rgb if style is not None else None
    return f"#{rgb}" if rgb is not None else None


def _colors(document: Document) -> dict[str, str | None]:
    return {
        "primary": _color(document, "Heading 1"),
        "secondary": _color(document, "Heading 2"),
        "accent": _color(document, "Title"),
        "text": _color(document, "Normal"),
    }


def _has_drawing(paragraph: Paragraph) -> bool:
    return bool(paragraph._p.xpath(_DRAWING))


def _logo(first: Section, cover: list[Paragraph]) -> dict[str, str] | None:
    places = (
        ("header", [] if first.header.is_linked_to_previous else first.header.paragraphs),
        ("cover", cover),
        ("footer", [] if first.footer.is_linked_to_previous else first.footer.paragraphs),
    )
    for placement, paragraphs in places:
        holder = next((p for p in paragraphs if _has_drawing(p)), None)
        if holder is not None:
            align = _ALIGN.get(holder.alignment, "left") if holder.alignment is not None else "left"
            return {"placement": placement, "align": align}
    return None


def _mm(length: Length | None, default: float) -> float:
    return round(length.mm, 1) if length is not None else default


def _page(first: Section) -> dict[str, Any]:
    width, height = _mm(first.page_width, 210.0), _mm(first.page_height, 297.0)
    short, long_ = sorted((width, height))
    size = next(
        (
            name
            for name, a, b in _PAPER
            if abs(short - a) <= _PAPER_TOLERANCE_MM and abs(long_ - b) <= _PAPER_TOLERANCE_MM
        ),
        "custom",
    )
    landscape = first.orientation == WD_ORIENT.LANDSCAPE or width > height
    return {
        "size": size,
        "orientation": "landscape" if landscape else "portrait",
        "width_mm": width,
        "height_mm": height,
        "margins_mm": {
            "top": _mm(first.top_margin, 25.4),
            "right": _mm(first.right_margin, 25.4),
            "bottom": _mm(first.bottom_margin, 25.4),
            "left": _mm(first.left_margin, 25.4),
        },
    }


def _header_footer(part: _Header | _Footer) -> dict[str, Any]:
    if part.is_linked_to_previous:  # the first section has none of its own
        return {"text": "", "page_numbers": False}
    text = " · ".join(p.text.strip() for p in part.paragraphs if p.text.strip())
    element = part._element
    fields = [
        *element.xpath(".//w:fldSimple/@w:instr"),
        *(t.text or "" for t in element.xpath(".//w:instrText")),
    ]
    return {"text": text[:_LINE_MAX], "page_numbers": any("PAGE" in f.upper().split() for f in fields)}


def _cover_field(paragraph: Paragraph) -> dict[str, str] | None:
    text = paragraph.text.strip()
    style = _style_name(paragraph)
    if not text:
        return None
    if style == "title":
        return {"key": "title", "label": "Title"}
    if style == "subtitle":
        return {"key": "subtitle", "label": "Subtitle"}
    label = text.split(":", 1)[0].strip()
    lowered = label.lower()
    key = next((k for phrase, k in _COVER_LABELS if lowered.startswith(phrase)), None)
    return {"key": key, "label": label[:_LABEL_MAX]} if key else None


def _cover(paragraphs: list[Paragraph]) -> dict[str, Any]:
    fields = [f for p in paragraphs if (f := _cover_field(p)) is not None]
    return {"enabled": bool(fields), "fields": fields[:20]}
