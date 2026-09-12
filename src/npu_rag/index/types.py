"""Domain types, enums, and data classes for the vector-index package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SearchMode(StrEnum):
    """Retrieval mode determining which engines execute."""

    DENSE_ONLY = "dense_only"
    LEXICAL_ONLY = "lexical_only"
    HYBRID = "hybrid"


class DistanceMetric(StrEnum):
    """Distance metric used for dense vector comparisons."""

    COSINE = "cosine"
    DOT = "dot"
    L2 = "l2"


class FusionMethod(StrEnum):
    """Fusion algorithm used to combine dense and lexical rankings."""

    RRF = "rrf"
    LINEAR = "linear"


@dataclass(frozen=True)
class SearchOptions:
    """Configurable options for index queries."""

    limit: int = 10
    mode: SearchMode = SearchMode.HYBRID
    metric: DistanceMetric = DistanceMetric.COSINE
    filter_sql: str | None = None
    fusion_method: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    linear_alpha: float = 0.5
    mmr_diversity: float | None = None
    max_chunks_per_document: int | None = None
    document_rollup: bool = False
    context_window: int = 0


@dataclass(frozen=True)
class SearchResult:
    """A single ranked result from the vector index."""

    chunk_id: str
    file_path: str
    title: str | None
    chunk_kind: str
    chunk_index: int
    headings: tuple[str, ...]
    locator_type: str
    locator_json: str
    text: str
    score: float
    dense_distance: float | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    expanded_context: str | None = None
    token_count: int = 0
    vector: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SearchResponse:
    """Full response returned by an index query."""

    query: str
    mode: SearchMode
    results: tuple[SearchResult, ...]
    total_candidates: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SyncSummary:
    """Summary of an ingest-to-index synchronization run."""

    chunks_added: int
    chunks_updated: int
    chunks_deleted: int
    total_chunks: int
    files_indexed: int
    elapsed_seconds: float


@dataclass(frozen=True)
class IndexStats:
    """High-level statistics about the vector index."""

    total_chunks: int
    total_documents: int
    vector_dimension: int
    has_fts_index: bool
    index_size_bytes: int
