"""Plain-text extractor: the smallest ``Extractor`` (task 4.1).

Read the file as UTF-8 (the Discoverer's ``\\\\?\\`` path, as given), normalise,
and emit one ``ProseSegment`` covering the file's line range. Title is not
declared, so it stays ``None``. Decode failures become ``ExtractionError``.
"""

from __future__ import annotations

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.extract.base import normalise
from npu_rag.ingest.types import Extracted, MarkdownLocator, ProseSegment, SourceFile

__all__ = ["TextExtractor"]


class TextExtractor:
    """Read a text file, normalise, emit prose. Conforms to ``Extractor``."""

    def extract(self, source: SourceFile, config: IngestConfig) -> Extracted:
        del config
        try:
            raw = source.path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ExtractionError(
                f"could not read {source.relative_path.as_posix()} as UTF-8 text",
                stage="extraction",
                path=source.path,
            ) from exc
        text = normalise(raw)
        if not text:
            return Extracted(title=None, segments=(), omissions=())
        line_count = len(raw.splitlines())
        locator = MarkdownLocator(line_range=(1, line_count), ordinal=0)
        return Extracted(
            title=None,
            segments=(ProseSegment(text=text, locator=locator),),
            omissions=(),
        )
