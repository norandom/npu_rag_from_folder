"""Tests for vector-index domain types, options, and error taxonomy."""

from __future__ import annotations

import pytest

from npu_rag.index.errors import (
    ConfigurationError,
    QueryError,
    SchemaError,
    StoreError,
    SyncError,
    VectorIndexError,
)
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


def test_search_mode_enum_values() -> None:
    assert SearchMode.DENSE_ONLY.value == "dense_only"
    assert SearchMode.LEXICAL_ONLY.value == "lexical_only"
    assert SearchMode.HYBRID.value == "hybrid"


def test_distance_metric_enum_values() -> None:
    assert DistanceMetric.COSINE.value == "cosine"
    assert DistanceMetric.DOT.value == "dot"
    assert DistanceMetric.L2.value == "l2"


def test_fusion_method_enum_values() -> None:
    assert FusionMethod.RRF.value == "rrf"
    assert FusionMethod.LINEAR.value == "linear"


def test_search_options_defaults() -> None:
    opts = SearchOptions()
    assert opts.limit == 10
    assert opts.mode == SearchMode.HYBRID
    assert opts.metric == DistanceMetric.COSINE
    assert opts.filter_sql is None
    assert opts.fusion_method == FusionMethod.RRF
    assert opts.rrf_k == 60
    assert opts.dense_weight == 1.0
    assert opts.lexical_weight == 1.0
    assert opts.linear_alpha == 0.5
    assert opts.mmr_diversity is None
    assert opts.max_chunks_per_document is None
    assert not opts.document_rollup
    assert opts.context_window == 0


def test_search_options_frozen() -> None:
    opts = SearchOptions()
    with pytest.raises(AttributeError):
        opts.limit = 20  # type: ignore[misc]


def test_search_result_and_response() -> None:
    res = SearchResult(
        chunk_id="chunk_1",
        file_path="docs/sample.md",
        title="Sample Document",
        chunk_kind="prose",
        chunk_index=0,
        headings=("Introduction", "Background"),
        locator_type="markdown",
        locator_json='{"line_from": 1, "line_to": 10}',
        text="Sample text content.",
        score=0.95,
        dense_distance=0.05,
        dense_rank=0,
        lexical_score=4.2,
        lexical_rank=1,
        expanded_context="Full expanded text",
        token_count=15,
    )
    assert res.chunk_id == "chunk_1"
    assert res.headings == ("Introduction", "Background")
    assert res.score == 0.95

    resp = SearchResponse(
        query="sample query",
        mode=SearchMode.HYBRID,
        results=(res,),
        total_candidates=1,
        elapsed_seconds=0.012,
    )
    assert resp.query == "sample query"
    assert len(resp.results) == 1
    assert resp.results[0].chunk_id == "chunk_1"


def test_sync_summary() -> None:
    summary = SyncSummary(
        chunks_added=50,
        chunks_updated=10,
        chunks_deleted=5,
        total_chunks=145,
        files_indexed=20,
        elapsed_seconds=1.45,
    )
    assert summary.chunks_added == 50
    assert summary.chunks_deleted == 5
    assert summary.total_chunks == 145


def test_index_stats() -> None:
    stats = IndexStats(
        total_chunks=1000,
        total_documents=120,
        vector_dimension=768,
        has_fts_index=True,
        index_size_bytes=4096000,
    )
    assert stats.total_chunks == 1000
    assert stats.vector_dimension == 768
    assert stats.has_fts_index is True


def test_error_taxonomy() -> None:
    # All inherit from VectorIndexError
    for err_cls in (StoreError, QueryError, SyncError, SchemaError, ConfigurationError):
        err = err_cls("Test failure")
        assert isinstance(err, VectorIndexError)
        assert isinstance(err, Exception)

    # Distinguishable by type
    with pytest.raises(StoreError):
        raise StoreError("Store connection failed")

    with pytest.raises(QueryError):
        raise QueryError("Invalid filter")

    with pytest.raises(SyncError):
        raise SyncError("Sync aborted")
