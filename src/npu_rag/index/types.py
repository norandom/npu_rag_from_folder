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
    #: Candidate depth to retrieve before fusion, MMR, rollup and truncation.
    #: ``None`` keeps the derived default (``max(limit * 4, 30)``), which is what
    #: every caller got before this field existed, so adding it changes no
    #: existing behaviour. Set it to widen the pool when recall matters more
    #: than latency, or to narrow it on a large index. design.md already wrote
    #: ``fetch_k`` into the retrieval pseudocode (section 3, lines 118 and 122);
    #: this makes the value a caller's choice rather than a hidden constant.
    fetch_k: int | None = None


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
    #: Distinct embedding-model identities actually recorded on the rows,
    #: sorted, with unstamped rows excluded (they are counted separately).
    #: This is what the index **was built with**, which is a different fact
    #: from what configuration currently says and the only one a reporter may
    #: attribute to the index. Empty for an index whose rows are all unstamped,
    #: and for an empty index.
    model_identities: tuple[str, ...] = ()
    #: Rows carrying no embedding-model identity at all - written either before
    #: the column existed or by a caller that supplied none. Reported rather
    #: than hidden, because an unstamped row's vector space is unknowable and
    #: a reporter must be able to say so.
    unstamped_chunks: int = 0

    @property
    def is_mixed_model(self) -> bool:
        """Whether this table holds vectors from more than one vector space.

        True for two or more recorded identities, and also for one recorded
        identity sitting alongside unstamped rows - those rows came from some
        other write and nothing says it used the same model. An index that is
        *entirely* unstamped is not mixed: it is unknown, which
        :attr:`unstamped_chunks` already states.
        """
        if len(self.model_identities) > 1:
            return True
        return bool(self.model_identities) and self.unstamped_chunks > 0
