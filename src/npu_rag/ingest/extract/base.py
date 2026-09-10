"""Extractor protocol, Extracted re-export, and shared normalisation (task 4.1).

design.md's Extraction Service Interface: adapters implement ``Extractor`` and
return the ``Extracted`` already defined in ``npu_rag.ingest.types``. Requirement
3.3's Unicode and whitespace folding is a pure function used by every
text-bearing adapter.

This module sits in the extract rank. It imports only leftward (types, config)
and never vision, state, or httpx.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.types import Extracted, SourceFile

__all__ = ["Extracted", "Extractor", "normalise"]

_WHITESPACE_RUN = re.compile(r"[ \t\n\r]+")


class Extractor(Protocol):
    """Turn one ``SourceFile`` into ordered segments without touching the network.

    Preconditions: ``source.path`` exists and the router selected this extractor.
    On failure: raise ``ExtractionError`` with a stage and the path.
    """

    def extract(self, source: SourceFile, config: IngestConfig) -> Extracted: ...


def normalise(text: str) -> str:
    """NFC-compose and collapse runs of space, tab, and newlines.

    Words are not stemmed, joined, or stripped of punctuation; combining marks
    that belong to a letter are composed, not dropped.
    """
    composed = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RUN.sub(" ", composed).strip()
