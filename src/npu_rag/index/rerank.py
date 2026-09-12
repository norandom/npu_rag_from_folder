"""Cross-encoder reranker protocol and pass-through implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from npu_rag.index.types import SearchResult


@runtime_checkable
class Reranker(Protocol):
    """Protocol for optional cross-encoder reranking models."""

    def rerank(self, query: str, results: Sequence[SearchResult]) -> list[SearchResult]:
        """Rerank candidates based on full-text query relevance."""
        ...


class PassThroughReranker:
    """Default reranker that preserves the input candidate ordering and scores."""

    def rerank(self, query: str, results: Sequence[SearchResult]) -> list[SearchResult]:
        """Return candidate results unmodified."""
        return list(results)
