"""Neighbouring-chunk context expansion for vector-index search results."""

from __future__ import annotations

from typing import Sequence

from npu_rag.index.errors import ConfigurationError
from npu_rag.index.store import LanceIndexStore
from npu_rag.index.types import SearchResult


class ContextExpander:
    """Expands retrieved chunks by retrieving neighbouring chunks within window [-w, +w]."""

    def __init__(self, store: LanceIndexStore) -> None:
        self._store = store

    @property
    def store(self) -> LanceIndexStore:
        return self._store

    def expand(
        self,
        results: Sequence[SearchResult],
        *,
        window: int = 1,
        separator: str = "\n\n",
    ) -> list[SearchResult]:
        """Expand results with adjacent chunks from the same document.

        Args:
            results: Input list of SearchResults.
            window: Number of chunks before and after the matched chunk (w >= 1).
            separator: Text delimiter between expanded chunks (default newline).

        Returns:
            List of new SearchResult objects with `expanded_context` populated.
        """
        if window < 0:
            raise ConfigurationError(f"Context window must be non-negative, got {window}")

        if window == 0 or not results:
            return list(results)

        expanded_results: list[SearchResult] = []

        for res in results:
            min_idx = max(0, res.chunk_index - window)
            max_idx = res.chunk_index + window
            escaped_path = res.file_path.replace("'", "''")

            filter_sql = (
                f"file_path = '{escaped_path}' "
                f"AND chunk_index >= {min_idx} "
                f"AND chunk_index <= {max_idx}"
            )

            try:
                rows = (
                    self._store.table.search()
                    .where(filter_sql)
                    .limit((2 * window) + 1)
                    .to_list()
                )
            except Exception:
                rows = []

            if rows:
                sorted_rows = sorted(rows, key=lambda r: int(r.get("chunk_index", 0)))
                joined_text = separator.join(
                    str(r.get("text", "")) for r in sorted_rows
                )
            else:
                joined_text = res.text

            updated_res = SearchResult(
                chunk_id=res.chunk_id,
                file_path=res.file_path,
                title=res.title,
                chunk_kind=res.chunk_kind,
                chunk_index=res.chunk_index,
                headings=res.headings,
                locator_type=res.locator_type,
                locator_json=res.locator_json,
                text=res.text,
                score=res.score,
                dense_distance=res.dense_distance,
                dense_rank=res.dense_rank,
                lexical_score=res.lexical_score,
                lexical_rank=res.lexical_rank,
                expanded_context=joined_text,
                token_count=res.token_count,
                vector=res.vector,
            )
            expanded_results.append(updated_res)

        return expanded_results


def expand_context(
    store: LanceIndexStore,
    results: Sequence[SearchResult],
    *,
    window: int = 1,
    separator: str = "\n\n",
) -> list[SearchResult]:
    """Convenience functional interface for ContextExpander."""
    expander = ContextExpander(store)
    return expander.expand(results, window=window, separator=separator)
