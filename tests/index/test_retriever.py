"""Tests for DenseRetriever and retrieval primitives."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pytest

from npu_rag.index.errors import QueryError
from npu_rag.index.retriever import DenseRetriever
from npu_rag.index.schema import VECTOR_DIMENSION, ChunkSchema
from npu_rag.index.store import LanceIndexStore
from npu_rag.index.types import DistanceMetric


def _make_unit_vector(active_idx: int, dim: int = VECTOR_DIMENSION) -> list[float]:
    """Create a unit vector with 1.0 at active_idx and 0.0 elsewhere."""
    vec = [0.0] * dim
    vec[active_idx % dim] = 1.0
    return vec


def _make_chunk(
    chunk_id: str,
    file_path: str = "doc.md",
    chunk_index: int = 0,
    chunk_kind: str = "prose",
    text: str = "sample text",
    vector: Sequence[float] | None = None,
) -> ChunkSchema:
    if vector is None:
        vector = _make_unit_vector(0)
    return ChunkSchema(
        chunk_id=chunk_id,
        file_path=file_path,
        file_hash="hash123",
        chunk_index=chunk_index,
        chunk_kind=chunk_kind,
        title="Document Title",
        headings=["Section 1"],
        locator_type="markdown",
        locator_json='{"type": "markdown", "line_range": [1, 5], "ordinal": 0}',
        text=text,
        vector=list(vector),
        token_count=10,
        created_at=100.0,
    )


@pytest.fixture
def empty_store(tmp_path: Path) -> LanceIndexStore:
    return LanceIndexStore(tmp_path / "lancedb_empty")


@pytest.fixture
def populated_store(tmp_path: Path) -> LanceIndexStore:
    store = LanceIndexStore(tmp_path / "lancedb_populated")
    v0 = _make_unit_vector(0)
    v1 = _make_unit_vector(1)
    v2 = _make_unit_vector(2)

    chunks = [
        _make_chunk("c1", file_path="alpha.md", chunk_index=0, chunk_kind="prose", vector=v0),
        _make_chunk("c2", file_path="alpha.md", chunk_index=1, chunk_kind="table", vector=v1),
        _make_chunk("c3", file_path="beta.md", chunk_index=0, chunk_kind="prose", vector=v2),
    ]
    store.add(chunks)
    return store


def test_dense_retriever_empty_store(empty_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(empty_store)
    query_vec = _make_unit_vector(0)
    results = retriever.retrieve(query_vec, limit=5)
    assert results == []


def test_dense_retriever_dimension_validation(populated_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(populated_store)
    with pytest.raises(QueryError, match=f"dimension must be {VECTOR_DIMENSION}"):
        retriever.retrieve([0.1, 0.2, 0.3])


def test_dense_retriever_limit_validation(populated_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(populated_store)
    query_vec = _make_unit_vector(0)
    with pytest.raises(QueryError, match="Limit must be greater than zero"):
        retriever.retrieve(query_vec, limit=0)


def test_dense_retriever_invalid_metric(populated_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(populated_store)
    query_vec = _make_unit_vector(0)
    with pytest.raises(QueryError, match="Unsupported distance metric"):
        retriever.retrieve(query_vec, metric="manhattan")


def test_dense_retriever_exact_knn_ranking(populated_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(populated_store)
    # Query matching v0 exactly: c1 should be rank 0 with distance near 0.0
    query_vec = _make_unit_vector(0)
    results = retriever.retrieve(query_vec, limit=3, metric=DistanceMetric.COSINE)

    assert len(results) == 3
    # Top result should be c1
    assert results[0].chunk_id == "c1"
    assert results[0].dense_rank == 0
    assert results[0].dense_distance is not None
    assert math.isclose(results[0].dense_distance, 0.0, abs_tol=1e-4)
    assert math.isclose(results[0].score, 1.0, abs_tol=1e-4)
    assert results[0].headings == ("Section 1",)
    assert results[0].vector is not None
    assert len(results[0].vector) == VECTOR_DIMENSION

    # Orthogonal vectors c2 and c3 have distance near 1.0
    assert results[1].dense_distance is not None
    assert math.isclose(results[1].dense_distance, 1.0, abs_tol=1e-4)


def test_dense_retriever_dot_metric(populated_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(populated_store)
    query_vec = _make_unit_vector(1)
    results = retriever.retrieve(query_vec, limit=2, metric=DistanceMetric.DOT)

    assert len(results) == 2
    assert results[0].chunk_id == "c2"
    assert results[0].dense_rank == 0
    assert results[0].dense_distance is not None
    assert math.isclose(results[0].dense_distance, 0.0, abs_tol=1e-4)


def test_dense_retriever_l2_metric(populated_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(populated_store)
    query_vec = _make_unit_vector(2)
    results = retriever.retrieve(query_vec, limit=2, metric=DistanceMetric.L2)

    assert len(results) == 2
    assert results[0].chunk_id == "c3"
    assert results[0].dense_rank == 0
    assert results[0].dense_distance is not None
    assert math.isclose(results[0].dense_distance, 0.0, abs_tol=1e-4)
    assert math.isclose(results[0].score, 1.0, abs_tol=1e-4)


def test_dense_retriever_metadata_filtering(populated_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(populated_store)
    query_vec = _make_unit_vector(0)

    # Filter to only beta.md chunks (excludes c1 and c2)
    results = retriever.retrieve(
        query_vec,
        limit=5,
        filter_sql="file_path = 'beta.md'",
        prefilter=True,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "c3"
    assert results[0].file_path == "beta.md"

    # Filter with prefilter=False (post-filtering)
    results_post = retriever.retrieve(
        query_vec,
        limit=5,
        filter_sql="chunk_kind = 'table'",
        prefilter=False,
    )
    assert len(results_post) == 1
    assert results_post[0].chunk_id == "c2"


def test_dense_retriever_ann_parameters_and_exact_flag(populated_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(populated_store)
    query_vec = _make_unit_vector(0)

    # Passing nprobes, refine_factor, and exact=True runs cleanly without error
    results = retriever.retrieve(
        query_vec,
        limit=2,
        nprobes=10,
        refine_factor=2,
        exact=True,
    )
    assert len(results) == 2
    assert results[0].chunk_id == "c1"


def test_dense_retriever_invalid_sql_raises_query_error(populated_store: LanceIndexStore) -> None:
    retriever = DenseRetriever(populated_store)
    query_vec = _make_unit_vector(0)
    with pytest.raises(QueryError, match="Dense vector retrieval failed"):
        retriever.retrieve(query_vec, filter_sql="INVALID SQL SYNTAX !!!")
