"""Score fusion algorithms for combining dense and lexical search rankings."""

from __future__ import annotations

import math
from typing import Sequence

from npu_rag.index.errors import ConfigurationError
from npu_rag.index.types import SearchResult


def _merge_search_result(
    primary: SearchResult,
    alternate: SearchResult | None,
    fused_score: float,
) -> SearchResult:
    """Combine metadata and score diagnostics from primary and alternate candidates."""
    dense_dist = primary.dense_distance
    dense_rnk = primary.dense_rank
    lex_scr = primary.lexical_score
    lex_rnk = primary.lexical_rank
    vec = primary.vector

    if alternate is not None:
        if dense_dist is None:
            dense_dist = alternate.dense_distance
        if dense_rnk is None:
            dense_rnk = alternate.dense_rank
        if lex_scr is None:
            lex_scr = alternate.lexical_score
        if lex_rnk is None:
            lex_rnk = alternate.lexical_rank
        if vec is None:
            vec = alternate.vector

    return SearchResult(
        chunk_id=primary.chunk_id,
        file_path=primary.file_path,
        title=primary.title,
        chunk_kind=primary.chunk_kind,
        chunk_index=primary.chunk_index,
        headings=primary.headings,
        locator_type=primary.locator_type,
        locator_json=primary.locator_json,
        text=primary.text,
        score=fused_score,
        dense_distance=dense_dist,
        dense_rank=dense_rnk,
        lexical_score=lex_scr,
        lexical_rank=lex_rnk,
        expanded_context=primary.expanded_context,
        token_count=primary.token_count,
        vector=vec,
    )


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) for merging dense and lexical rankings."""

    def __init__(
        self,
        k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        if k <= 0:
            raise ConfigurationError("RRF constant k must be greater than zero")
        if dense_weight < 0 or lexical_weight < 0:
            raise ConfigurationError("RRF weights must be non-negative")
        self.k = k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def fuse(
        self,
        dense_results: Sequence[SearchResult],
        lexical_results: Sequence[SearchResult],
        *,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Fuse dense and lexical rankings using RRF.

        RRF score for chunk d:
          score(d) = w_dense / (k + rank_dense + 1) + w_lex / (k + rank_lex + 1)
        where rank is 0-indexed position in the candidate list.
        """
        dense_by_id = {res.chunk_id: res for res in dense_results}
        lexical_by_id = {res.chunk_id: res for res in lexical_results}

        all_chunk_ids: list[str] = []
        for res in dense_results:
            if res.chunk_id not in all_chunk_ids:
                all_chunk_ids.append(res.chunk_id)
        for res in lexical_results:
            if res.chunk_id not in all_chunk_ids:
                all_chunk_ids.append(res.chunk_id)

        if not all_chunk_ids:
            return []

        scored_results: list[tuple[float, SearchResult]] = []

        for cid in all_chunk_ids:
            dense_res = dense_by_id.get(cid)
            lexical_res = lexical_by_id.get(cid)

            dense_component = 0.0
            if dense_res is not None:
                rank = dense_res.dense_rank if dense_res.dense_rank is not None else 0
                dense_component = self.dense_weight / (self.k + rank + 1)

            lexical_component = 0.0
            if lexical_res is not None:
                rank = lexical_res.lexical_rank if lexical_res.lexical_rank is not None else 0
                lexical_component = self.lexical_weight / (self.k + rank + 1)

            total_score = dense_component + lexical_component

            primary = dense_res if dense_res is not None else lexical_res
            assert primary is not None
            alternate = lexical_res if dense_res is not None else None

            merged = _merge_search_result(primary, alternate, fused_score=total_score)
            scored_results.append((total_score, merged))

        # Sort descending by fused score; tie-break deterministically by chunk_id
        scored_results.sort(key=lambda item: (-item[0], item[1].chunk_id))

        fused = [item[1] for item in scored_results]
        if limit is not None and limit > 0:
            return fused[:limit]
        return fused


def reciprocal_rank_fusion(
    dense_results: Sequence[SearchResult],
    lexical_results: Sequence[SearchResult],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    limit: int | None = None,
) -> list[SearchResult]:
    """Convenience functional interface for ReciprocalRankFusion."""
    rrf = ReciprocalRankFusion(k=k, dense_weight=dense_weight, lexical_weight=lexical_weight)
    return rrf.fuse(dense_results, lexical_results, limit=limit)


def _normalize_scores(values: Sequence[float]) -> list[float]:
    """Normalize a sequence of floats to [0.0, 1.0] using min-max scaling."""
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if math.isclose(max_v, min_v, abs_tol=1e-9):
        return [1.0 for _ in values]
    spread = max_v - min_v
    return [(v - min_v) / spread for v in values]


class LinearCombination:
    """Linear combination score fusion with min-max normalization."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ConfigurationError(
                f"Linear combination alpha must be in [0.0, 1.0], got {alpha}"
            )
        self.alpha = alpha

    def fuse(
        self,
        dense_results: Sequence[SearchResult],
        lexical_results: Sequence[SearchResult],
        *,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Fuse dense and lexical rankings using normalized linear combination:

        score(d) = alpha * s_norm_dense(d) + (1 - alpha) * s_norm_lex(d)
        """
        dense_by_id = {res.chunk_id: res for res in dense_results}
        lexical_by_id = {res.chunk_id: res for res in lexical_results}

        all_chunk_ids: list[str] = []
        for res in dense_results:
            if res.chunk_id not in all_chunk_ids:
                all_chunk_ids.append(res.chunk_id)
        for res in lexical_results:
            if res.chunk_id not in all_chunk_ids:
                all_chunk_ids.append(res.chunk_id)

        if not all_chunk_ids:
            return []

        # 1. Normalize dense candidate scores
        dense_scores = [r.score for r in dense_results]
        norm_dense_list = _normalize_scores(dense_scores)
        dense_norm_by_id = {
            r.chunk_id: norm for r, norm in zip(dense_results, norm_dense_list)
        }

        # 2. Normalize lexical candidate scores
        lex_scores = [
            r.lexical_score if r.lexical_score is not None else r.score
            for r in lexical_results
        ]
        norm_lex_list = _normalize_scores(lex_scores)
        lex_norm_by_id = {
            r.chunk_id: norm for r, norm in zip(lexical_results, norm_lex_list)
        }

        # 3. Combine scores
        scored_results: list[tuple[float, SearchResult]] = []
        for cid in all_chunk_ids:
            dense_res = dense_by_id.get(cid)
            lexical_res = lexical_by_id.get(cid)

            s_dense = dense_norm_by_id.get(cid, 0.0)
            s_lex = lex_norm_by_id.get(cid, 0.0)
            combined_score = (self.alpha * s_dense) + ((1.0 - self.alpha) * s_lex)

            primary = dense_res if dense_res is not None else lexical_res
            assert primary is not None
            alternate = lexical_res if dense_res is not None else None

            merged = _merge_search_result(primary, alternate, fused_score=combined_score)
            scored_results.append((combined_score, merged))

        # Sort descending by fused score; tie-break deterministically by chunk_id
        scored_results.sort(key=lambda item: (-item[0], item[1].chunk_id))

        fused = [item[1] for item in scored_results]
        if limit is not None and limit > 0:
            return fused[:limit]
        return fused


def linear_combination_fusion(
    dense_results: Sequence[SearchResult],
    lexical_results: Sequence[SearchResult],
    *,
    alpha: float = 0.5,
    limit: int | None = None,
) -> list[SearchResult]:
    """Convenience functional interface for LinearCombination."""
    lc = LinearCombination(alpha=alpha)
    return lc.fuse(dense_results, lexical_results, limit=limit)

