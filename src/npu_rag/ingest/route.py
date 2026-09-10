"""Extraction-path routing (task 3.2).

Select a path token from the file extension together with a content sniff —
PDF magic bytes, the zip signature for workbooks, an image header via pillow
— and return that token only. The token-to-adapter table belongs to the
pipeline. Anything unmatched is an unsupported omission, never a raised error.

This module sits to the right of discover and to the left of extract. It
does not import extract adapters, vision, or httpx. On Windows,
``SourceFile.path`` is ``\\\\?\\``-prefixed; that form is opened as given.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from npu_rag.ingest.types import Omission, OmissionCategory, SourceFile

__all__ = ["ExtractionPath", "Router"]

_PDF_MAGIC = b"%PDF"
_ZIP_SIGNATURE = b"PK\x03\x04"
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_TEXT_SUFFIXES = frozenset({".txt"})
_PDF_SUFFIXES = frozenset({".pdf"})
_EXCEL_SUFFIXES = frozenset({".xlsx"})
_IMAGE_SUFFIXES = frozenset({".png", ".jpeg", ".jpg", ".gif", ".webp"})


class ExtractionPath(StrEnum):
    """Path token naming one of the five extract adapters, not the adapter."""

    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    EXCEL = "excel"
    IMAGE = "image"


class Router:
    """``SourceFile`` → ``ExtractionPath`` by extension plus a content sniff.

    Requirements 2.1 and 2.4. Unmatched types become
    ``Omission(UNSUPPORTED)`` and never raise.
    """

    def route(self, source: SourceFile) -> ExtractionPath | Omission:
        sniffed = _sniff(source.path)
        if sniffed is not None:
            return sniffed
        suffix = source.path.suffix.lower()
        if suffix in _MARKDOWN_SUFFIXES:
            return ExtractionPath.MARKDOWN
        if suffix in _TEXT_SUFFIXES:
            return ExtractionPath.TEXT
        if suffix in _PDF_SUFFIXES:
            return ExtractionPath.PDF
        if suffix in _EXCEL_SUFFIXES:
            return ExtractionPath.EXCEL
        if suffix in _IMAGE_SUFFIXES:
            return ExtractionPath.IMAGE
        return Omission(
            category=OmissionCategory.UNSUPPORTED,
            path=source.path,
            reason=f"unsupported file type: {source.relative_path.as_posix()}",
        )


def _sniff(path: Path) -> ExtractionPath | None:
    """Inspect leading bytes; content wins over a misleading extension."""
    with path.open("rb") as handle:
        header = handle.read(max(len(_PDF_MAGIC), len(_ZIP_SIGNATURE)))
        if header.startswith(_PDF_MAGIC):
            return ExtractionPath.PDF
        if header.startswith(_ZIP_SIGNATURE):
            return ExtractionPath.EXCEL
        handle.seek(0)
        return _image_from_header(handle)


def _image_from_header(handle: BinaryIO) -> ExtractionPath | None:
    """Identify an image from its header; do not decode the raster."""
    try:
        with Image.open(handle) as image:
            if image.format:
                return ExtractionPath.IMAGE
    except (UnidentifiedImageError, OSError):
        return None
    return None
