"""OCR for scanned PDF pages (no text layer): render with pdfium, read with RapidOCR (ONNX on CPU, local).

Like the embeddings, it runs here: a scanned RFP never leaves the service. The engine loads its models once.
"""

import io
from functools import lru_cache
from typing import Any

# Pixels per PDF point: 2 renders a page at about 144 dpi, enough for body text.
RENDER_SCALE = 2


@lru_cache(maxsize=1)
def _engine() -> Any:
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def ocr_pdf_pages(data: bytes, page_indexes: list[int]) -> dict[int, str]:
    """Text read from the given (0-based) pages, top to bottom, one line per detected text box."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(io.BytesIO(data))
    try:
        out = {}
        for index in page_indexes:
            image = document[index].render(scale=RENDER_SCALE).to_pil()
            found, _ = _engine()(image)
            # Each box: (corners, text, score); the engine returns them in reading order.
            out[index] = "\n".join(str(box[1]) for box in found or [])
        return out
    finally:
        document.close()
