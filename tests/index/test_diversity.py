"""Tests for diversity control: Maximal Marginal Relevance (MMR) and document rollup."""

from __future__ import annotations

import pytest

from npu_rag.index.diversity import (
    MaximalMarginalRelevance,
    _cosine_similarity,
    maximal_marginal_relevance,
)
from npu_rag.index.errors import ConfigurationError
from npu_rag.index.types import SearchResult


def _make_candidate(
    chunk_id: str,
    vector: tuple[float, ...],
    score: float = 1.0,
    file_path: str = "doc.md",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        file_path=file_path,
        title=f"Doc {chunk_id}",
        chunk_kind="prose",
        chunk_index=0,
        headings=("Heading",),
        locator_type="markdown",
        locator_json="{}",
        text=f"Text for {chunk_id}",
        score=score,
        token_count=10,
        vector=vector,
    )


def test_cosine_similarity_edge_cases() -> None:
    # Identical
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    # Orthogonal
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # Opposite
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    # Zero vector
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_mmr_promotes_dissimilar_candidate() -> None:
    query_vec = (1.0, 0.0, 0.0)

    # c1: perfectly matches query
    c1 = _make_candidate("c1", vector=(1.0, 0.0, 0.0), score=1.0)
    # c2: near duplicate of c1 (very close angle: sim to query ≈ 0.999, sim to c1 ≈ 0.999)
    c2 = _make_candidate("c2", vector=(0.999, 0.01, 0.0), score=0.99)
    # c3: distinct angle, moderate similarity to query (sim to query ≈ 0.707, sim to c1 ≈ 0.707)
    c3 = _make_candidate("c3", vector=(0.707, 0.707, 0.0), score=0.7)

    # Standard MMR with diversity lambda = 0.5
    mmr = MaximalMarginalRelevance(lambda_mult=0.5)
    reranked = mmr.rerank(query_vec, [c1, c2, c3], limit=2)

    # c1 is selected first (sim to query = 1.0)
    assert reranked[0].chunk_id == "c1"
    # For second position:
    # c2: 0.5 * 0.999 - 0.5 * 0.999 ≈ 0.0
    # c3: 0.5 * 0.707 - 0.5 * 0.707 ≈ 0.0 (slightly adjusted)
    # Let's verify with lambda = 0.4 to strongly promote diversity
    mmr_diverse = MaximalMarginalRelevance(lambda_mult=0.4)
    reranked_diverse = mmr_diverse.rerank(query_vec, [c1, c2, c3], limit=2)
    assert reranked_diverse[0].chunk_id == "c1"
    assert reranked_diverse[1].chunk_id == "c3"


def test_mmr_pure_relevance_lambda_one() -> None:
    query_vec = (1.0, 0.0, 0.0)

    c1 = _make_candidate("c1", vector=(1.0, 0.0, 0.0))
    c2 = _make_candidate("c2", vector=(0.9, 0.1, 0.0))
    c3 = _make_candidate("c3", vector=(0.5, 0.5, 0.0))

    # lambda = 1.0 -> purely based on query similarity
    reranked = maximal_marginal_relevance(query_vec, [c3, c2, c1], lambda_mult=1.0, limit=3)
    order = [r.chunk_id for r in reranked]
    assert order == ["c1", "c2", "c3"]


def test_mmr_limit_truncation() -> None:
    query_vec = (1.0, 0.0)
    candidates = [_make_candidate(f"c{i}", vector=(float(i), 1.0)) for i in range(10)]

    reranked = maximal_marginal_relevance(query_vec, candidates, limit=4)
    assert len(reranked) == 4


def test_mmr_empty_and_single_candidate() -> None:
    query_vec = (1.0, 0.0)
    mmr = MaximalMarginalRelevance()

    assert mmr.rerank(query_vec, []) == []

    c1 = _make_candidate("c1", vector=(1.0, 0.0))
    assert mmr.rerank(query_vec, [c1]) == [c1]
    assert mmr.rerank(query_vec, [c1], limit=0) == []


def test_mmr_validation_errors() -> None:
    with pytest.raises(ConfigurationError, match="must be in"):
        MaximalMarginalRelevance(lambda_mult=-0.1)

    with pytest.raises(ConfigurationError, match="must be in"):
        MaximalMarginalRelevance(lambda_mult=1.5)
