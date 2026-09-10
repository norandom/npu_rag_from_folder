"""Extraction adapters: one file to ordered segments, offline.

``extract/*`` inherit the extract rank. They never import vision, state, or
httpx; ImageRefs are emitted, never resolved.
"""

from __future__ import annotations

from npu_rag.ingest.extract.base import Extracted, Extractor, normalise
from npu_rag.ingest.extract.markdown import MarkdownExtractor
from npu_rag.ingest.extract.text import TextExtractor

__all__ = [
    "Extracted",
    "Extractor",
    "MarkdownExtractor",
    "TextExtractor",
    "normalise",
]
