"""Tests for score fusion algorithms (RRF and Linear Combination)."""

from __future__ import annotations

import math

import pytest

from npu_rag.index.errors import ConfigurationError
from npu_rag.index.fusion import (
    LinearCombination,
    ReciprocalRankFusion,
    linear_combination_fusion,
    reciprocal_rank_fusion,
)
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


def test_linear_combination_normalized_range_and_blending() -> None:
    # d1 is strong in both (dense score 0.9, lexical 20.0)
    # d2 is strong in dense only (dense score 0.85, not in lexical)
    # d3 is strong in lexical only (not in dense, lexical 25.0)
    dense = [
        _make_result("d1", score=0.9, dense_distance=0.1, dense_rank=0),
        _make_result("d2", score=0.85, dense_distance=0.15, dense_rank=1),
    ]
    lexical = [
        _make_result("d3", score=25.0, lexical_score=25.0, lexical_rank=0),
        _make_result("d1", score=20.0, lexical_score=20.0, lexical_rank=1),
    ]

    lc = LinearCombination(alpha=0.5)
    fused = lc.fuse(dense, lexical)

    # In dense: d1 has score 0.9 (norm 1.0), d2 has score 0.85 (norm 0.0)
    # In lexical: d3 has score 25.0 (norm 1.0), d1 has score 20.0 (norm 0.0)
    # Combined:
    # d1: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    # d3: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    # d2: 0.5 * 0.0 + 0.5 * 0.0 = 0.0
    for res in fused:
        assert 0.0 <= res.score <= 1.0

    scores = {r.chunk_id: r.score for r in fused}
    assert math.isclose(scores["d1"], 0.5, abs_tol=1e-5)
    assert math.isclose(scores["d3"], 0.5, abs_tol=1e-5)
    assert math.isclose(scores["d2"], 0.0, abs_tol=1e-5)


def test_linear_combination_dynamic_alpha_shift() -> None:
    dense = [
        _make_result("dense_top", score=1.0, dense_rank=0),
        _make_result("dense_bot", score=0.2, dense_rank=1),
    ]
    lexical = [
        _make_result("lex_top", score=30.0, lexical_score=30.0, lexical_rank=0),
        _make_result("dense_top", score=5.0, lexical_score=5.0, lexical_rank=1),
    ]

    # When alpha = 1.0: dense ranking dominates completely
    fused_dense = linear_combination_fusion(dense, lexical, alpha=1.0)
    assert fused_dense[0].chunk_id == "dense_top"

    # When alpha = 0.0: lexical ranking dominates completely
    fused_lex = linear_combination_fusion(dense, lexical, alpha=0.0)
    assert fused_lex[0].chunk_id == "lex_top"


def test_linear_combination_single_candidate_normalization() -> None:
    dense = [_make_result("d1", score=0.8, dense_rank=0)]
    lexical = [_make_result("l1", score=10.0, lexical_score=10.0, lexical_rank=0)]

    fused = linear_combination_fusion(dense, lexical, alpha=0.7)
    assert len(fused) == 2
    # Single candidate in dense normalized to 1.0; 0.7 * 1.0 = 0.7
    d1 = next(r for r in fused if r.chunk_id == "d1")
    assert math.isclose(d1.score, 0.7, abs_tol=1e-5)

    # Single candidate in lexical normalized to 1.0; 0.3 * 1.0 = 0.3
    l1 = next(r for r in fused if r.chunk_id == "l1")
    assert math.isclose(l1.score, 0.3, abs_tol=1e-5)


def test_linear_combination_limit_and_empty() -> None:
    lc = LinearCombination(alpha=0.5)
    assert lc.fuse([], []) == []

    dense = [_make_result(f"d{i}", score=float(i), dense_rank=i) for i in range(10)]
    lexical = [_make_result(f"l{i}", score=float(i), lexical_score=float(i), lexical_rank=i) for i in range(10)]
    fused = lc.fuse(dense, lexical, limit=3)
    assert len(fused) == 3


def test_linear_combination_alpha_validation() -> None:
    with pytest.raises(ConfigurationError, match="must be in"):
        LinearCombination(alpha=-0.1)

    with pytest.raises(ConfigurationError, match="must be in"):
        LinearCombination(alpha=1.1)

