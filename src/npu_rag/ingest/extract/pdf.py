"""PDF extractor: per-page text-layer routing (task 4.3).

pypdfium2 probes each page's extractable text. At or above
``IngestConfig.min_page_chars`` the page becomes a ``ProseSegment`` in
content-stream order with a 1-based ``PageLocator``. Below the threshold
the page is rendered to PNG and emitted as an ``ImageRef``. This module
never imports vision, state, or httpx.

``SourceFile.path`` is opened as given, including the Windows ``\\\\?\\`` form.
"""

from __future__ import annotations

from io import BytesIO

import pypdfium2 as pdfium  # type: ignore[import-untyped]

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.extract.base import normalise
from npu_rag.ingest.types import (
    Extracted,
    ImageRef,
    PageLocator,
    ProseSegment,
    Segment,
    SourceFile,
)

__all__ = ["PdfExtractor"]


class PdfExtractor:
    """Route each PDF page to prose or a rendered image. Conforms to ``Extractor``."""

    def extract(self, source: SourceFile, config: IngestConfig) -> Extracted:
        try:
            payload = source.path.read_bytes()
        except OSError as exc:
            raise ExtractionError(
                f"could not read {source.relative_path.as_posix()} as a PDF",
                stage="extraction",
                path=source.path,
            ) from exc
        try:
            document = pdfium.PdfDocument(payload)
        except pdfium.PdfiumError as exc:
            raise ExtractionError(
                f"could not read {source.relative_path.as_posix()} as a PDF",
                stage="extraction",
                path=source.path,
            ) from exc
        try:
            segments = _segments(document, config.min_page_chars)
        except pdfium.PdfiumError as exc:
            raise ExtractionError(
                f"could not read {source.relative_path.as_posix()} as a PDF",
                stage="extraction",
                path=source.path,
            ) from exc
        finally:
            document.close()
        return Extracted(title=None, segments=tuple(segments), omissions=())


def _segments(document: pdfium.PdfDocument, threshold: int) -> list[Segment]:
    segments: list[Segment] = []
    for index in range(len(document)):
        page_number = index + 1
        page = document[index]
        try:
            segments.append(_segment_for_page(page, page_number, threshold))
        finally:
            page.close()
    return segments


def _segment_for_page(
    page: pdfium.PdfPage, page_number: int, threshold: int
) -> Segment:
    locator = PageLocator(page=page_number)
    extractable = _extractable_text(page)
    if len(extractable.strip()) >= threshold:
        return ProseSegment(text=normalise(extractable), locator=locator)
    data, width, height = _render_png(page)
    return ImageRef(
        data=data,
        mime="image/png",
        width=width,
        height=height,
        locator=locator,
    )


def _extractable_text(page: pdfium.PdfPage) -> str:
    textpage = page.get_textpage()
    try:
        text = textpage.get_text_range()
    finally:
        textpage.close()
    if not isinstance(text, str):
        raise TypeError("pypdfium2 get_text_range must return str")
    return text


def _render_png(page: pdfium.PdfPage) -> tuple[bytes, int, int]:
    bitmap = page.render(scale=1)
    try:
        image = bitmap.to_pil()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        width, height = image.size
        return buffer.getvalue(), int(width), int(height)
    finally:
        bitmap.close()
