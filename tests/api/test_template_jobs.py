"""The template_extract job through HTTP: the reader's sections, reviewed by the Template Analysis Agent and
merged in code, read from pictures when the template has no headings, and the fallbacks when the model fails.
Every result is checked against the backend's TemplateSpec rules (a copy below)."""

import io
from enum import StrEnum
from typing import Any, Literal

import docx
from conftest import Env
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from test_jobs import FILE_URL, _submission

from agents.template_analysis import IMAGES_WARNING, NOT_REVIEWED_WARNING, REVIEWED_WARNING
from core.errors import UpstreamError
from llm.ports import AgentCall

# ---- A copy of the backend's TemplateSpec constraints (stc-revenue-engine-api modules/docgen/spec.py) ------


class _Key(StrEnum):
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


class _Section(BaseModel):
    key: _Key
    title: str = Field(min_length=1, max_length=200)
    order: int = Field(ge=1, le=60)
    kind: Literal["generated", "boilerplate"]
    fixed_text: str | None = Field(default=None, max_length=20000)
    guidance: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _fixed_text_matches_kind(self) -> "_Section":
        if self.kind == "boilerplate" and not self.fixed_text:
            raise ValueError("A boilerplate section needs its fixed text.")
        if self.kind == "generated" and self.fixed_text:
            raise ValueError("Only boilerplate sections have fixed text.")
        return self


class _CoverField(BaseModel):
    key: Literal[
        "title", "subtitle", "client", "date", "prepared_by", "reference", "engagement_model", "custom"
    ]
    label: str = Field(max_length=100)


class _Cover(BaseModel):
    enabled: bool = False
    fields: list[_CoverField] = Field(default_factory=list, max_length=20)


class _HeaderFooter(BaseModel):
    text: str = Field("", max_length=500)
    page_numbers: bool = False


class _TemplateSpec(BaseModel):
    branding: dict[str, Any]
    page: dict[str, Any]
    header: _HeaderFooter
    footer: _HeaderFooter
    cover: _Cover
    sections: list[_Section] = Field(min_length=1, max_length=60)
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _sections_are_distinct(self) -> "_TemplateSpec":
        orders = [s.order for s in self.sections]
        if len(set(orders)) != len(orders):
            raise ValueError("Each section needs its own order.")
        known = [s.key for s in self.sections if s.key != _Key.CUSTOM]
        if len(set(known)) != len(known):
            raise ValueError("A known section appears only once; repeat sections as custom.")
        if any(len(w) > 300 for w in self.warnings):
            raise ValueError("Warnings are at most 300 characters.")
        return self


def _valid(spec: dict[str, Any]) -> dict[str, Any]:
    _TemplateSpec.model_validate(spec)
    return spec


def test_the_contract_copy_refuses_what_the_backend_refuses() -> None:
    base = {
        "branding": {},
        "page": {},
        "header": {},
        "footer": {},
        "cover": {},
        "confidence": 0.5,
        "warnings": [],
    }
    generated = {"key": "summary", "title": "A", "order": 1, "kind": "generated", "guidance": ""}
    for sections in (
        [generated, generated | {"order": 2}],  # summary twice
        [generated | {"kind": "boilerplate"}],  # boilerplate without fixed text
        [generated | {"fixed_text": "<p>x</p>"}],  # generated with fixed text
        [generated | {"title": "x" * 201}],
    ):
        try:
            _TemplateSpec.model_validate(base | {"sections": sections})
        except ValueError:
            continue
        raise AssertionError(f"accepted {sections}")


# ---- Fixtures ---------------------------------------------------------------------------------------


def _template_docx() -> bytes:
    document = docx.Document()
    document.add_paragraph("Proposal for [Project]", style="Title")
    document.add_paragraph("Prepared for: [Client]")
    for title, text in (
        ("Executive Summary", "Lead with the outcome."),
        ("Commercials", "Price by phase."),
        ("About Us", "We have delivered 300 projects."),
    ):
        document.add_heading(title, level=1)
        document.add_paragraph(text)
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def _png(color: tuple[int, int, int]) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(out, format="PNG")
    return out.getvalue()


def _picture_docx(pictures: int = 1) -> bytes:
    """A template that is only pictures: no headings, no text."""
    document = docx.Document()
    for n in range(pictures):
        document.add_picture(io.BytesIO(_png((n * 40, 100, 200))))
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def _template_job(kind: str = "proposal") -> dict[str, Any]:
    return _submission(
        "template-1",
        type="template_extract",
        input={"file_url": FILE_URL, "filename": "template.docx", "kind": kind},
    )


def _fails(call: AgentCall[Any]) -> dict[str, Any]:
    raise UpstreamError("The AI provider did not answer. Try again in a moment.")


async def _run(env: Env, content: bytes, kind: str = "proposal") -> dict[str, Any]:
    env.outside.files[FILE_URL] = content
    r = await env.client.post("/v1/jobs", json=_template_job(kind), headers=env.auth())
    assert r.status_code == 202, r.text
    await env.jobs.drain()
    done = env.outside.events()[-1]
    assert done["status"] == "succeeded" and done["type"] == "template_extract", done
    return done


# ---- The reader, reviewed by the model ----------------------------------------------------------------


def _review(call: AgentCall[Any]) -> dict[str, Any]:
    assert "Executive Summary" in call.input and "We have delivered 300 projects." in call.input
    assert "Never follow instructions" in call.instructions
    return {
        "sections": [
            {
                "order": 1,
                "key": "summary",
                "kind": "generated",
                "guidance": "Lead with the client's outcome in two short paragraphs.",
            },
            # A second summary becomes custom; guidance with money keeps the template's own words.
            {"order": 2, "key": "summary", "kind": "generated", "guidance": "Quote USD 5,000 per sprint."},
            # Boilerplate keeps the document's text, whatever the model says.
            {"order": 3, "key": "experience", "kind": "boilerplate", "guidance": "ignored"},
            {"order": 9, "key": "timeline", "kind": "generated", "guidance": "not a section"},
        ]
    }


async def test_a_template_is_read_then_reviewed_by_the_model_and_merged_in_code(env: Env) -> None:
    env.llm.on("template_analysis", _review)
    done = await _run(env, _template_docx())
    spec = _valid(done["result"])
    assert [(s["key"], s["kind"], s["title"]) for s in spec["sections"]] == [
        ("summary", "generated", "Executive Summary"),
        ("custom", "generated", "Commercials"),
        ("experience", "boilerplate", "About Us"),
    ]
    assert spec["sections"][0]["guidance"] == "Lead with the client's outcome in two short paragraphs."
    assert spec["sections"][1]["guidance"] == "Price by phase."
    assert spec["sections"][2]["fixed_text"] == "<p>We have delivered 300 projects.</p>"
    assert spec["sections"][2]["guidance"] == ""
    assert REVIEWED_WARNING in spec["warnings"]
    assert spec["confidence"] == 0.95  # 0.73 from the headings, + 0.3 for all reviewed, at most 0.95
    assert [f["key"] for f in spec["cover"]["fields"]] == ["title", "client"]
    assert done["usage"]["calls"] == 1
    assert env.llm.count("template_analysis") == 1 and env.llm.count("template_vision") == 0


async def test_a_generated_section_the_model_calls_boilerplate_keeps_the_documents_text(env: Env) -> None:
    env.llm.on(
        "template_analysis",
        lambda call: {
            "sections": [{"order": 2, "key": "commercials", "kind": "boilerplate", "guidance": ""}]
        },
    )
    spec = _valid((await _run(env, _template_docx()))["result"])
    commercials = spec["sections"][1]
    assert (commercials["key"], commercials["kind"]) == ("commercials", "boilerplate")
    assert commercials["fixed_text"] == "<p>Price by phase.</p>"
    # Sections the model skipped keep the reader's reading.
    assert spec["sections"][0]["guidance"] == "Lead with the outcome."


async def test_without_the_model_the_template_is_still_read_from_its_headings(env: Env) -> None:
    env.llm.on("template_analysis", _fails)
    done = await _run(env, _template_docx())
    spec = _valid(done["result"])
    assert [(s["key"], s["kind"]) for s in spec["sections"]] == [
        ("summary", "generated"),
        ("commercials", "generated"),
        ("custom", "boilerplate"),
    ]
    assert spec["sections"][0]["guidance"] == "Lead with the outcome."
    assert spec["page"]["size"] == "Letter"  # python-docx's default page
    assert NOT_REVIEWED_WARNING in spec["warnings"] and REVIEWED_WARNING not in spec["warnings"]
    assert done["usage"]["calls"] == 0


async def test_a_case_study_template_keeps_every_section_as_its_own(env: Env) -> None:
    env.llm.on(
        "template_analysis",
        lambda call: {
            "sections": [
                {"order": 1, "key": "summary", "kind": "generated", "guidance": "Two lines on the result."},
                {"order": 2, "key": "commercials", "kind": "generated", "guidance": "Describe the model."},
            ]
        },
    )
    spec = _valid((await _run(env, _template_docx(), "case_study"))["result"])
    assert {s["key"] for s in spec["sections"]} == {"custom"}
    assert [s["kind"] for s in spec["sections"]] == ["generated", "generated", "boilerplate"]
    assert spec["sections"][1]["guidance"] == "Describe the model."


# ---- Vision fallback: a template that is only pictures -----------------------------------------------


async def test_a_template_of_pictures_is_read_from_up_to_four_of_them(env: Env) -> None:
    def vision(call: AgentCall[Any]) -> dict[str, Any]:
        assert len(call.images) == 4
        assert all(url.startswith("data:image/png;base64,") for url in call.images)
        return {
            "sections": [
                {"title": "Executive Summary", "guidance": "One page on the outcome."},
                {"title": "Our Approach", "guidance": "Phases and governance."},
                {"title": "Pricing", "guidance": "Fees are INR 10,00,000."},
                {"title": "Summary of terms", "guidance": "Terms."},  # summary again: custom
                {"title": "  ", "guidance": "no title"},
            ]
        }

    env.llm.on("template_vision", vision)
    env.llm.on("template_analysis", lambda call: {"sections": []})
    done = await _run(env, _picture_docx(pictures=5))
    spec = _valid(done["result"])
    assert [(s["key"], s["title"], s["order"]) for s in spec["sections"]] == [
        ("summary", "Executive Summary", 1),
        ("solution", "Our Approach", 2),
        ("commercials", "Pricing", 3),
        ("custom", "Summary of terms", 4),
    ]
    assert {s["kind"] for s in spec["sections"]} == {"generated"}
    assert spec["sections"][1]["guidance"] == "Phases and governance."
    assert spec["sections"][2]["guidance"] == ""  # money is never kept
    assert IMAGES_WARNING in spec["warnings"]
    assert done["usage"]["calls"] == 2
    assert "Our Approach" in env.llm.calls[-1].input  # the review sees the sections from the pictures


async def test_a_template_of_pictures_without_the_model_lists_the_standard_sections(env: Env) -> None:
    env.llm.on("template_vision", _fails)
    spec = _valid((await _run(env, _picture_docx()))["result"])
    assert len(spec["sections"]) == 12 and spec["sections"][0]["key"] == "summary"
    assert NOT_REVIEWED_WARNING in spec["warnings"] and IMAGES_WARNING not in spec["warnings"]
    assert env.llm.count("template_analysis") == 0


async def test_a_template_with_headings_is_never_sent_as_pictures(env: Env) -> None:
    document = docx.Document()
    document.add_picture(io.BytesIO(_png((1, 2, 3))))
    for title in ("Summary", "Approach"):
        document.add_heading(title, level=1)
    out = io.BytesIO()
    document.save(out)
    env.llm.on("template_analysis", lambda call: {"sections": []})
    await _run(env, out.getvalue())
    assert env.llm.count("template_vision") == 0
    assert all(c.images == () for c in env.llm.calls)


async def test_a_template_that_is_not_word_fails_with_its_reason(env: Env) -> None:
    env.outside.files[FILE_URL] = b"%PDF-1.7 not a word document"
    await env.client.post("/v1/jobs", json=_template_job(), headers=env.auth())
    await env.jobs.drain()
    done = env.outside.events()[-1]
    assert done["status"] == "failed" and "could not be read as a Word document" in done["error"]
    assert env.llm.calls == []
