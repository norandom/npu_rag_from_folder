"""Score fusion algorithms for combining dense and lexical search rankings."""

from __future__ import annotations

from typing import Sequence

from npu_rag.index.errors import ConfigurationError
from npu_rag.index.types import SearchResult


def _merge_search_result(
    primary: SearchResult,
    secondary: SearchResult | None,
    fused_score: float,
) -> SearchResult:
    """Combine metadata and score diagnostics from primary and secondary candidates."""
    dense_dist = primary.dense_distance
    dense_rnk = primary.dense_rank
    lex_scr = primary.lexical_score
    lex_rnk = primary.lexical_rank
    vec = primary.vector

    if secondary is not None:
        if dense_dist is None:
            dense_dist = secondary.dense_distance
        if dense_rnk is None:
            dense_rnk = secondary.dense_rank
        if lex_scr is None:
            lex_scr = secondary.lexical_score
        if lex_rnk is None:
            lex_rnk = secondary.lexical_rank
        if vec is None:
            vec = secondary.vector

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
            secondary = lexical_res if dense_res is not None else None

            merged = _merge_search_result(primary, secondary, fused_score=total_score)
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
