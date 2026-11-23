"""template_extract: read a template (DOCX) into the backend's TemplateSpec shape, without the model.

The document's real headings become sections: its Word heading styles, or, in a document that has none, the
lines set apart by hand as headings (bold, at the size most such lines share; a larger one is the title, a
smaller one a label), even one run on at the end of a paragraph. A proposal template's are mapped to the
backend's section keys where the titles match; a case study template's are all `custom`, written for each case
study when the heading names a part of one and fixed text otherwise. Its styles give fonts and colours (the
colour of headings set by hand when it has no heading styles), its first section the page,
header and footer. Deterministic and pure (no I/O). The backend validates and cleans the result (docgen
apply_extraction) and an admin reviews it before it is used. The document's text is copied in as data and
never interpreted.

Kept in step with the backend's local reader (stc-revenue-engine-api modules/docgen/mock_ai.py).
The template_extract job then has the model review this reading (services/templates.py).
"""

import html
import io
import posixpath
import re
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import docx
from docx.document import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.section import Section, _Footer, _Header
from docx.styles.style import ParagraphStyle
from docx.text.paragraph import Paragraph
from docx.text.run import Run


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
        # A sample case study laid out like the project's own brief: what was asked and how it was judged.
        "scope",
        "work",
        "requirements",
        "requirement",
        "deliverables",
        "features",
        "objectives",
        "goals",
        "context",
        "situation",
        "criteria",
        "evaluation",
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
_MEDIA = "word/media/"
_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
# A picture larger than this is not sent to the model (a page scan is far smaller).
_IMAGE_MAX_BYTES = 5 * 1024 * 1024


class UnreadableDocument(Exception):
    """python-docx could not read the file as a Word document."""


@dataclass
class _Heading:
    title: str
    body: list[str] = field(default_factory=list)
    colour: str | None = None  # the heading's own text colour, when set by hand


@dataclass(frozen=True)
class _Part:
    """A paragraph, or the part of one before or after a heading run on at its end, and whether it is a
    heading (at what size, in what colour)."""

    paragraph: Paragraph
    text: str
    heading: bool = False
    size: float | None = None
    colour: str | None = None


def read_template(data: bytes, kind: TemplateKind = TemplateKind.PROPOSAL) -> dict[str, Any]:
    """The template_extract result for this DOCX, as raw JSON (validated by the caller like a callback's).
    Raises UnreadableDocument."""
    try:
        document = docx.Document(io.BytesIO(data))
        return _extract(document, kind)
    except Exception as exc:  # any failure to parse an uploaded file means the same: it cannot be read
        raise UnreadableDocument from exc


def heading_count(data: bytes) -> int:
    """How many sections the document's headings give (0 when it has none). Raises UnreadableDocument."""
    try:
        paragraphs = list(docx.Document(io.BytesIO(data)).paragraphs)
    except Exception as exc:  # the same as read_template: a file that cannot be parsed cannot be read
        raise UnreadableDocument from exc
    return len(_split(paragraphs)[1])


def embedded_images(data: bytes, limit: int) -> list[tuple[str, bytes]]:
    """Up to `limit` pictures stored in the document (word/media), in file order, as (MIME type, bytes).
    Only types a vision model reads; vector formats such as EMF are skipped."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = sorted(n for n in archive.namelist() if n.startswith(_MEDIA))
            found = []
            for name in names:
                mime = _IMAGE_TYPES.get(posixpath.splitext(name)[1].lower())
                info = archive.getinfo(name)
                if mime and 0 < info.file_size <= _IMAGE_MAX_BYTES:
                    found.append((mime, archive.read(name)))
                if len(found) == limit:
                    break
            return found
    except zipfile.BadZipFile as exc:
        raise UnreadableDocument from exc


def _extract(document: Document, kind: TemplateKind) -> dict[str, Any]:
    paragraphs = list(document.paragraphs)
    cover_part, headings = _split(paragraphs)
    warnings = [REVIEW_WARNING]
    reader = READERS[kind]
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
            "colors": _colors(document, headings),
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


def _split(paragraphs: list[Paragraph]) -> tuple[list[Paragraph], list[_Heading]]:
    """The paragraphs before the first section (the cover) and each section's heading with its text."""
    level = _section_level(paragraphs)
    if level is not None:
        blocks = [_Part(p, p.text.strip(), _heading_level(p) == level) for p in paragraphs]
    else:
        blocks = _styled_blocks(paragraphs)
    cover: list[Paragraph] = []
    headings: list[_Heading] = []
    for b in blocks:
        if b.heading and b.text:
            headings.append(_Heading(b.text, colour=b.colour))
        elif headings:
            if b.text:
                headings[-1].body.append(b.text)
        elif b.paragraph not in cover:
            cover.append(b.paragraph)
    return cover, headings


# ---- Headings set by hand (no heading styles) ---------------------------------------------------------

_HEADING_MAX_CHARS, _HEADING_MAX_WORDS = 80, 10
_NOT_A_HEADING_END = (".", ",", ";", ":")
_LIST_ITEM = re.compile(r"^[\u2022\u25aa\u25cf\u25e6\u2013\u2014*-]")
_MIN_STYLED_HEADINGS = 2


def _size(run: Run) -> float | None:
    return run.font.size.pt if run.font.size is not None else None


def _bold(run: Run, paragraph: Paragraph) -> bool:
    if run.bold is not None:
        return bool(run.bold)
    style = paragraph.style
    return bool(style is not None and style.font.bold)


def _run_colour(run: Run) -> str | None:
    colour = run.font.color
    if colour is None or colour.type is None or colour.rgb is None:
        return None
    return f"#{colour.rgb}"


def _body_size(paragraphs: list[Paragraph]) -> float | None:
    """The size most of the plain text is set in."""
    sizes: Counter[float] = Counter()
    for p in paragraphs:
        for r in p.runs:
            if (size := _size(r)) is not None and r.text.strip() and not _bold(r, p):
                sizes[size] += len(r.text)
    return sizes.most_common(1)[0][0] if sizes else None


def _text(runs: list[Run]) -> str:
    return "".join(r.text for r in runs).strip()


def _headline(paragraph: Paragraph, runs: list[Run], body: float | None) -> tuple[str, float | None] | None:
    """The runs' text and size when they read as a heading: short, bold, not smaller than the body text."""
    text = _text(runs)
    visible = [r for r in runs if r.text.strip()]
    if (
        not visible
        or len(text) > _HEADING_MAX_CHARS
        or len(text.split()) > _HEADING_MAX_WORDS
        or text.endswith(_NOT_A_HEADING_END)
        or _LIST_ITEM.match(text)
        or not all(_bold(r, paragraph) for r in visible)
    ):
        return None
    sizes = [size for r in visible if (size := _size(r)) is not None]
    size = max(sizes, default=None)
    if size is not None and body is not None and size < body:
        return None  # a label above the title ("EDUCATION"), not a heading
    return text, size


def _part(paragraph: Paragraph, runs: list[Run], body: float | None) -> _Part:
    found = _headline(paragraph, runs, body)
    if found is None:
        return _Part(paragraph, _text(runs))
    colour = next((c for r in runs if r.text.strip() and (c := _run_colour(r))), None)
    return _Part(paragraph, found[0], heading=True, size=found[1], colour=colour)


def _parts(paragraph: Paragraph, body: float | None) -> list[_Part]:
    """The paragraph whole, or split before a heading run on at its end ("... Budget [amount] Background")."""
    runs = list(paragraph.runs)
    whole = _part(paragraph, runs, body)
    if whole.heading:
        return [whole]
    for start in range(1, len(runs)):
        # The heading starts at the first of its runs, after some plain text.
        if not _text(runs[:start]) or _headline(paragraph, runs[start - 1 :], body):
            continue
        tail = _part(paragraph, runs[start:], body)
        if tail.heading:
            return [_Part(paragraph, _text(runs[:start])), tail]
    return [whole]


def _styled_blocks(paragraphs: list[Paragraph]) -> list[_Part]:
    """The document's blocks with its headings found by their formatting: the lines that read as headings at
    the size most of them share are the sections; a larger one (the title) and the rest are text."""
    body = _body_size(paragraphs)
    parts = [part for p in paragraphs for part in _parts(p, body)]
    sizes = Counter(part.size for part in parts if part.heading)
    # The section size: the most common; on a tie the smaller, since a title is set larger than its sections.
    level = min(sizes, key=lambda size: (-sizes[size], size or 0.0), default=None)
    found = level is not None and sizes[level] >= _MIN_STYLED_HEADINGS
    return [replace(part, heading=found and part.heading and part.size == level) for part in parts]


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
            section |= {"kind": "boilerplate", "fixed_text": paragraphs_html(heading.body), "guidance": ""}
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
            section |= {"kind": "boilerplate", "fixed_text": paragraphs_html(heading.body), "guidance": ""}
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


READERS: dict[TemplateKind, _Reader] = {
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


def paragraphs_html(lines: Iterable[str]) -> str:
    """Lines of the document's text as escaped HTML paragraphs, within the fixed text's size."""
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


def _colors(document: Document, headings: list[_Heading]) -> dict[str, str | None]:
    by_hand = next((h.colour for h in headings if h.colour), None)
    return {
        "primary": by_hand or _color(document, "Heading 1"),  # the headings it really uses
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


_TWIPS_PER_MM = 1440 / 25.4


def _mm(section: Section, part: str, side: str, default: float) -> float:
    """A page size or margin in mm, read from the raw twips: Google Docs and LibreOffice write decimals
    ("1133.857421875"), which python-docx refuses."""
    value = section._sectPr.xpath(f"./w:{part}/@w:{side}")
    try:
        return round(float(value[0]) / _TWIPS_PER_MM, 1) if value else default
    except ValueError:
        return default


def _page(first: Section) -> dict[str, Any]:
    width, height = _mm(first, "pgSz", "w", 210.0), _mm(first, "pgSz", "h", 297.0)
    short, long_ = sorted((width, height))
    size = next(
        (
            name
            for name, a, b in _PAPER
            if abs(short - a) <= _PAPER_TOLERANCE_MM and abs(long_ - b) <= _PAPER_TOLERANCE_MM
        ),
        "custom",
    )
    landscape = first._sectPr.xpath("./w:pgSz/@w:orient") == ["landscape"] or width > height
    return {
        "size": size,
        "orientation": "landscape" if landscape else "portrait",
        "width_mm": width,
        "height_mm": height,
        "margins_mm": {
            "top": _mm(first, "pgMar", "top", 25.4),
            "right": _mm(first, "pgMar", "right", 25.4),
            "bottom": _mm(first, "pgMar", "bottom", 25.4),
            "left": _mm(first, "pgMar", "left", 25.4),
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
