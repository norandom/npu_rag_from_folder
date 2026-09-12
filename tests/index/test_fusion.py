"""Tests for score fusion algorithms (RRF and Linear Combination)."""

from __future__ import annotations

import math

import pytest

from npu_rag.index.errors import ConfigurationError
from npu_rag.index.fusion import ReciprocalRankFusion, reciprocal_rank_fusion
from npu_rag.index.types import SearchResult


def _make_result(
    chunk_id: str,
    *,
    score: float = 1.0,
    dense_distance: float | None = None,
    dense_rank: int | None = None,
    lexical_score: float | None = None,
    lexical_rank: int | None = None,
    vector: tuple[float, ...] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        file_path=f"docs/{chunk_id}.md",
        title=f"Doc {chunk_id}",
        chunk_kind="prose",
        chunk_index=0,
        headings=("Heading 1",),
        locator_type="markdown",
        locator_json="{}",
        text=f"Content for {chunk_id}",
        score=score,
        dense_distance=dense_distance,
        dense_rank=dense_rank,
        lexical_score=lexical_score,
        lexical_rank=lexical_rank,
        token_count=10,
        vector=vector,
    )


def test_rrf_agreement_promotion() -> None:
    # d1 appears in both lists (dense rank 0, lexical rank 1)
    # d2 appears only in dense (dense rank 1)
    # d3 appears only in lexical (lexical rank 0)
    dense = [
        _make_result("d1", dense_distance=0.05, dense_rank=0),
        _make_result("d2", dense_distance=0.10, dense_rank=1),
    ]
    lexical = [
        _make_result("d3", lexical_score=15.0, lexical_rank=0),
        _make_result("d1", lexical_score=12.0, lexical_rank=1),
    ]

    rrf = ReciprocalRankFusion(k=60)
    fused = rrf.fuse(dense, lexical)

    # d1 score: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62 ≈ 0.016393 + 0.016129 = 0.032522
    # d3 score: 1/(60+0+1) = 1/61 ≈ 0.016393
    # d2 score: 1/(60+1+1) = 1/62 ≈ 0.016129
    assert len(fused) == 3
    assert fused[0].chunk_id == "d1"
    assert fused[1].chunk_id == "d3"
    assert fused[2].chunk_id == "d2"

    expected_d1_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert math.isclose(fused[0].score, expected_d1_score, rel_tol=1e-5)
    assert fused[0].dense_rank == 0
    assert fused[0].lexical_rank == 1
    assert fused[0].dense_distance == 0.05
    assert fused[0].lexical_score == 12.0


def test_rrf_asymmetric_results() -> None:
    dense = [_make_result("d_only", dense_distance=0.01, dense_rank=0)]
    lexical = [_make_result("l_only", lexical_score=20.0, lexical_rank=0)]

    fused = reciprocal_rank_fusion(dense, lexical, k=60)
    assert len(fused) == 2
    ids = {r.chunk_id for r in fused}
    assert ids == {"d_only", "l_only"}

    d_res = next(r for r in fused if r.chunk_id == "d_only")
    assert d_res.dense_distance == 0.01
    assert d_res.dense_rank == 0
    assert d_res.lexical_score is None
    assert d_res.lexical_rank is None

    l_res = next(r for r in fused if r.chunk_id == "l_only")
    assert l_res.dense_distance is None
    assert l_res.lexical_score == 20.0
    assert l_res.lexical_rank == 0


def test_rrf_configurable_k_and_weights() -> None:
    dense = [_make_result("dense_item", dense_rank=0)]
    lexical = [_make_result("lex_item", lexical_rank=0)]

    # High dense weight pushes dense_item ahead
    rrf = ReciprocalRankFusion(k=20, dense_weight=5.0, lexical_weight=1.0)
    fused = rrf.fuse(dense, lexical)

    assert fused[0].chunk_id == "dense_item"
    assert math.isclose(fused[0].score, 5.0 / 21.0, rel_tol=1e-5)
    assert math.isclose(fused[1].score, 1.0 / 21.0, rel_tol=1e-5)


def test_rrf_limit_truncation() -> None:
    dense = [_make_result(f"d{i}", dense_rank=i) for i in range(10)]
    lexical = [_make_result(f"l{i}", lexical_rank=i) for i in range(10)]

    fused = reciprocal_rank_fusion(dense, lexical, limit=5)
    assert len(fused) == 5


def test_rrf_empty_candidates() -> None:
    rrf = ReciprocalRankFusion()
    assert rrf.fuse([], []) == []

    dense = [_make_result("d1", dense_rank=0)]
    fused_dense_only = rrf.fuse(dense, [])
    assert len(fused_dense_only) == 1
    assert fused_dense_only[0].chunk_id == "d1"

    lexical = [_make_result("l1", lexical_rank=0)]
    fused_lex_only = rrf.fuse([], lexical)
    assert len(fused_lex_only) == 1
    assert fused_lex_only[0].chunk_id == "l1"


def test_rrf_configuration_validation() -> None:
    with pytest.raises(ConfigurationError, match="RRF constant k must be greater than zero"):
        ReciprocalRankFusion(k=0)

    with pytest.raises(ConfigurationError, match="RRF weights must be non-negative"):
        ReciprocalRankFusion(dense_weight=-1.0)
