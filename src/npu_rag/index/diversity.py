"""Diversity control: Maximal Marginal Relevance (MMR) and document rollup."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from npu_rag.index.errors import ConfigurationError
from npu_rag.index.types import SearchResult


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class MaximalMarginalRelevance:
    """Maximal Marginal Relevance (MMR) ranking for diverse search results."""

    def __init__(self, lambda_mult: float = 0.7) -> None:
        if not (0.0 <= lambda_mult <= 1.0):
            raise ConfigurationError(
                f"MMR lambda must be in [0.0, 1.0], got {lambda_mult}"
            )
        self.lambda_mult = lambda_mult

    def rerank(
        self,
        query_vector: Sequence[float],
        candidates: Sequence[SearchResult],
        *,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Rerank candidates using MMR to balance query relevance with result diversity.

        MMR formula:
          d* = argmax_{d_i in R \\ S} [ lambda * Sim(d_i, q) - (1 - lambda) * max_{d_j in S} Sim(d_i, d_j) ]
        """
        if limit <= 0:
            return []

        candidate_list = list(candidates)
        if not candidate_list:
            return []

        if len(candidate_list) <= 1:
            return candidate_list[:limit]

        # Precompute query similarities for all candidates with vectors
        query_sims: list[float] = []
        for cand in candidate_list:
            if cand.vector is not None:
                sim = _cosine_similarity(cand.vector, query_vector)
            else:
                sim = cand.score
            query_sims.append(sim)

        selected_indices: list[int] = []
        remaining_indices = list(range(len(candidate_list)))

        # Select first candidate with highest query similarity
        best_first = max(remaining_indices, key=lambda idx: (query_sims[idx], -idx))
        selected_indices.append(best_first)
        remaining_indices.remove(best_first)

        target_count = min(limit, len(candidate_list))

        while len(selected_indices) < target_count and remaining_indices:
            best_score = float("-inf")
            best_idx = remaining_indices[0]

            for idx in remaining_indices:
                cand_vec = candidate_list[idx].vector
                sim_to_query = query_sims[idx]

                # Maximum similarity to any already selected candidate
                max_sim_to_selected = 0.0
                if cand_vec is not None:
                    sims_to_selected: list[float] = []
                    for s_idx in selected_indices:
                        s_vec = candidate_list[s_idx].vector
                        if s_vec is not None:
                            sims_to_selected.append(_cosine_similarity(cand_vec, s_vec))
                    if sims_to_selected:
                        max_sim_to_selected = max(sims_to_selected)

                mmr_score = (
                    self.lambda_mult * sim_to_query
                    - (1.0 - self.lambda_mult) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [candidate_list[i] for i in selected_indices]


def maximal_marginal_relevance(
    query_vector: Sequence[float],
    candidates: Sequence[SearchResult],
    *,
    lambda_mult: float = 0.7,
    limit: int = 10,
) -> list[SearchResult]:
    """Convenience functional interface for MaximalMarginalRelevance."""
    mmr = MaximalMarginalRelevance(lambda_mult=lambda_mult)
    return mmr.rerank(query_vector, candidates, limit=limit)
