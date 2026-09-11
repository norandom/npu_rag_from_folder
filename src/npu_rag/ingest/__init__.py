"""Document ingest: archive to embedding-ready chunk records.

Public surface for this package exports the ingest entry point, configuration,
and output contracts.
"""

from __future__ import annotations

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.pipeline import run_ingest
from npu_rag.ingest.report import RunReport
from npu_rag.ingest.types import (
    BlockSegment,
    ChartRanges,
    ChunkKind,
    ChunkRecord,
    Extracted,
    FigureSegment,
    FormulaSegment,
    ImageFileLocator,
    ImageRef,
    Locator,
    MarkdownLocator,
    Omission,
    OmissionCategory,
    PageLocator,
    ProseSegment,
    Provenance,
    Resolved,
    Segment,
    SheetLocator,
    SourceFile,
    TableSegment,
)

__all__ = [
    "BlockSegment",
    "ChartRanges",
    "ChunkKind",
    "ChunkRecord",
    "Extracted",
    "FigureSegment",
    "FormulaSegment",
    "ImageFileLocator",
    "ImageRef",
    "IngestConfig",
    "Locator",
    "MarkdownLocator",
    "Omission",
    "OmissionCategory",
    "PageLocator",
    "ProseSegment",
    "Provenance",
    "Resolved",
    "RunReport",
    "Segment",
    "SheetLocator",
    "SourceFile",
    "TableSegment",
    "run_ingest",
]
