"""Standalone raster extractor: one file to one ``ImageRef`` (task 4.7).

Read the file as given (the Discoverer's ``\\\\?\\`` path), take width,
height and MIME from the image header, and emit a single ``ImageRef`` with
an ``ImageFileLocator``. The size threshold is recorded, not applied —
requirement 5.3's gate lives at vision. This module never imports vision,
state, or httpx.

Decode failures become ``ExtractionError``.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.types import Extracted, ImageFileLocator, ImageRef, SourceFile

__all__ = ["ImageExtractor"]


class ImageExtractor:
    """Read a raster file, emit one image reference. Conforms to ``Extractor``."""

    def extract(self, source: SourceFile, config: IngestConfig) -> Extracted:
        del config
        try:
            data = source.path.read_bytes()
        except OSError as exc:
            raise ExtractionError(
                f"could not read {source.relative_path.as_posix()} as an image",
                stage="extraction",
                path=source.path,
            ) from exc
        try:
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                fmt = image.format
        except (UnidentifiedImageError, OSError) as exc:
            raise ExtractionError(
                f"could not decode {source.relative_path.as_posix()} as an image",
                stage="extraction",
                path=source.path,
            ) from exc
        if not fmt:
            raise ExtractionError(
                f"could not decode {source.relative_path.as_posix()} as an image",
                stage="extraction",
                path=source.path,
            )
        mime = Image.MIME.get(fmt, "application/octet-stream")
        return Extracted(
            title=None,
            segments=(
                ImageRef(
                    data=data,
                    mime=mime,
                    width=int(width),
                    height=int(height),
                    locator=ImageFileLocator(),
                ),
            ),
            omissions=(),
        )
