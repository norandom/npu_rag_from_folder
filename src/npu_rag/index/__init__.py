"""Vector index package for npu-rag.

Provides embedded storage, hybrid retrieval (dense + lexical BM25),
diversity control, neighbouring-chunk context expansion, and query orchestration.
"""

from __future__ import annotations

from npu_rag.index.errors import (
    ConfigurationError,
    QueryError,
    SchemaError,
    StoreError,
    SyncError,
    VectorIndexError,
)
from npu_rag.index.rerank import PassThroughReranker, Reranker
from npu_rag.index.service import VectorIndex
from npu_rag.index.types import (
    DistanceMetric,
    FusionMethod,
    IndexStats,
    SearchMode,
    SearchOptions,
    SearchResponse,
    SearchResult,
    SyncSummary,
)

__all__ = [
    "ConfigurationError",
    "DistanceMetric",
    "FusionMethod",
    "IndexStats",
    "PassThroughReranker",
    "QueryError",
    "Reranker",
    "SchemaError",
    "SearchMode",
    "SearchOptions",
    "SearchResponse",
    "SearchResult",
    "StoreError",
    "SyncError",
    "SyncSummary",
    "VectorIndex",
    "VectorIndexError",
]
