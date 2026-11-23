"""Use-case: read a template (the template_extract job).

read the DOCX (code, deterministic) → no headings but pictures: read the sections from up to 4 pictures
(agent, vision) → review every section's key, kind and guidance (agent) → merge in code. The model only
improves the reader's result: when it fails, the reader's result is returned with a warning, and the job
still succeeds.
"""

import base64
import logging
from collections.abc import Callable
from typing import Any

from revenue_engine_ai.agents import template_analysis as agent
from revenue_engine_ai.agents.template_analysis import TemplateAnalysisInput
from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.errors import DomainError
from revenue_engine_ai.documents import template_reader
from revenue_engine_ai.documents.template_reader import TemplateKind
from revenue_engine_ai.llm.ports import LLM, Usage

log = logging.getLogger(__name__)

VERSIONS = {"template_analysis": agent.PROMPT_VERSION, "template_vision": agent.VISION_PROMPT_VERSION}
# Fewer headings than this and the structure is probably in pictures (a scanned or designed template).
MIN_HEADINGS = 2
MAX_IMAGES = 4


def _data_urls(images: list[tuple[str, bytes]]) -> list[str]:
    return [f"data:{mime};base64,{base64.b64encode(data).decode()}" for mime, data in images]


async def _from_images(
    content: bytes, spec: dict[str, Any], kind: TemplateKind, llm: LLM, settings: Settings, usage: Usage
) -> dict[str, Any]:
    """The spec with the sections seen in the template's pictures, when it has no headings to read."""
    if template_reader.heading_count(content) >= MIN_HEADINGS:
        return spec
    images = template_reader.embedded_images(content, MAX_IMAGES)
    if not images:
        return spec
    done = await agent.read_images(llm, _data_urls(images), kind, settings.template_analysis_effort)
    usage.add(done.usage)
    sections = agent.image_sections(done.value, kind)
    if len(sections) < MIN_HEADINGS:
        return spec
    # The pictures gave the sections: the reader's "no headings, standard sections listed" no longer applies.
    standard = {r.fallback_warning for r in template_reader.READERS.values()}
    warnings = [w for w in spec["warnings"] if w not in standard]
    return spec | {"sections": sections, "warnings": [*warnings, agent.IMAGES_WARNING]}


async def extract(
    content: bytes, kind: TemplateKind, llm: Callable[[], LLM], settings: Settings
) -> tuple[dict[str, Any], Usage]:
    """The template_extract result (the backend's TemplateSpec shape) and the model usage.
    Raises template_reader.UnreadableDocument."""
    # A template is read in milliseconds, so on the event loop.
    spec = template_reader.read_template(content, kind)
    usage = Usage()
    try:
        model = llm()
        spec = await _from_images(content, spec, kind, model, settings, usage)
        data = TemplateAnalysisInput(kind, spec["sections"], [f["label"] for f in spec["cover"]["fields"]])
        done = await agent.ask(model, data, settings.template_analysis_effort)
        usage.add(done.usage)
    except DomainError as exc:
        log.warning("template_review_skipped", extra={"fields": {"error": type(exc).__name__}})
        return spec | {"warnings": [*spec["warnings"], agent.NOT_REVIEWED_WARNING]}, usage
    return agent.merge(spec, done.value, kind), usage
