"""Pure rules of the final check, near-duplicate questions and callback signatures; and OCR of a scan."""

import hashlib
import hmac
import io

import pytest

from revenue_engine_ai.agents.final_check import DraftSnapshot, FinalFacts, findings, numbers_in
from revenue_engine_ai.core.errors import RuleViolation
from revenue_engine_ai.core.signing import sign
from revenue_engine_ai.documents import parsing
from revenue_engine_ai.documents.parsing import read_document
from revenue_engine_ai.rules.text import near_duplicate


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("EUR 812,000", 812000),
        ("€ 812.000", 812000),
        ("1.200.000 euros", 1200000),
        ("812k", 812000),
        ("1.5 million", 1500000),
        ("26 weeks", 26),
    ],
)
def test_numbers_in_reads_separators_and_scales(text: str, number: int) -> None:
    assert number in numbers_in(text)


def test_near_duplicates() -> None:
    assert near_duplicate("How many vehicles are in the fleet?", "how many vehicles are in the fleet")
    assert near_duplicate("How many vehicles in the fleet today?", "How many vehicles are in the fleet?")
    assert not near_duplicate("Which depots are in scope?", "Who uses the dispatch screen?")


def test_signature_matches_the_backend_format() -> None:
    expected = hmac.new(b"s", b"1700000000.{}", hashlib.sha256).hexdigest()
    assert sign("s", b"{}", timestamp=1700000000) == f"t=1700000000,v1={expected}"


def test_empty_draft_lists_are_ok_and_missing_figures_are_not() -> None:
    draft = DraftSnapshot(price=None, weeks=None, core_features={}, responses=[], attachments=[])
    facts = FinalFacts(price=None, weeks=None, features=[], answers=[], attachments=[])
    out = findings(facts, draft, report_price=True)
    assert [f.area for f in out] == ["Scope", "Cost", "Timeline", "Responses", "Attachments"]
    assert [f.ok for f in out] == [True, False, False, True, True]


def _scan() -> bytes:
    """A PDF page that is only an image of text: what a scanner produces."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1240, 300), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default(size=40)
    ImageDraw.Draw(image).text((40, 100), "We need GPS tracking of 400 trucks.", fill="black", font=font)
    out = io.BytesIO()
    image.save(out, "PDF")
    return out.getvalue()


def test_a_scanned_pdf_is_read_with_ocr_only_when_allowed() -> None:
    data = _scan()
    with pytest.raises(RuleViolation) as caught:
        read_document("scan.pdf", data)
    assert caught.value.extra["rule"] == "file_empty"
    doc = read_document("scan.pdf", data, ocr_max_pages=5)
    assert doc.ocr and "gps tracking" in doc.pages[0].text.lower()


def test_ocr_failure_falls_back_to_the_clear_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(data: bytes, pages: list[int]) -> dict[int, str]:
        raise RuntimeError("no models")

    monkeypatch.setattr(parsing, "ocr_pdf_pages", broken)
    with pytest.raises(RuleViolation):
        read_document("scan.pdf", _scan(), ocr_max_pages=5)
