"""Retrieval primitives for vector search and lexical search."""

from __future__ import annotations

from typing import Any, Sequence, cast

from lancedb.query import LanceFtsQueryBuilder, LanceVectorQueryBuilder

from npu_rag.index.errors import QueryError
from npu_rag.index.schema import VECTOR_DIMENSION
from npu_rag.index.store import LanceIndexStore
from npu_rag.index.types import DistanceMetric, SearchResult


def _row_to_search_result(
    row: dict[str, Any],
    *,
    score: float,
    dense_distance: float | None = None,
    dense_rank: int | None = None,
    lexical_score: float | None = None,
    lexical_rank: int | None = None,
    include_vector: bool = True,
) -> SearchResult:
    """Convert a raw LanceDB table row dict into a typed SearchResult."""
    headings_raw = row.get("headings") or ()
    headings = tuple(str(h) for h in headings_raw)

    vec: tuple[float, ...] | None = None
    if include_vector and "vector" in row and row["vector"] is not None:
        vec = tuple(float(x) for x in row["vector"])

    return SearchResult(
        chunk_id=str(row.get("chunk_id", "")),
        file_path=str(row.get("file_path", "")),
        title=row.get("title"),
        chunk_kind=str(row.get("chunk_kind", "")),
        chunk_index=int(row.get("chunk_index", 0)),
        headings=headings,
        locator_type=str(row.get("locator_type", "")),
        locator_json=str(row.get("locator_json", "{}")),
        text=str(row.get("text", "")),
        score=score,
        dense_distance=dense_distance,
        dense_rank=dense_rank,
        lexical_score=lexical_score,
        lexical_rank=lexical_rank,
        token_count=int(row.get("token_count", 0)),
        vector=vec,
    )


class DenseRetriever:
    """Dense semantic vector retriever backed by LanceDB."""

    def __init__(self, store: LanceIndexStore) -> None:
        self._store = store

    @property
    def store(self) -> LanceIndexStore:
        return self._store

    def retrieve(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 10,
        metric: DistanceMetric | str = DistanceMetric.COSINE,
        filter_sql: str | None = None,
        prefilter: bool | None = None,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        exact: bool = False,
        include_vector: bool = True,
    ) -> list[SearchResult]:
        """Execute dense kNN vector retrieval.

        Args:
            query_vector: Embedding vector of length VECTOR_DIMENSION (768).
            limit: Maximum number of nearest neighbors to return.
            metric: Distance metric to use (cosine, dot, l2).
            filter_sql: Optional SQL WHERE predicate for metadata filtering.
            prefilter: True to filter before ANN index search, False for post-filtering.
            nprobes: Number of partitions to check when an ANN index is present.
            refine_factor: Refine factor multiplier when an ANN index is present.
            exact: True to bypass vector index and force brute-force exact scan.
            include_vector: Whether to attach candidate vector to each SearchResult.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        if len(query_vector) != VECTOR_DIMENSION:
            raise QueryError(
                f"Query vector dimension must be {VECTOR_DIMENSION}, got {len(query_vector)}"
            )
        if limit <= 0:
            raise QueryError("Limit must be greater than zero")

        if isinstance(metric, DistanceMetric):
            resolved_metric = metric
        elif isinstance(metric, str):
            try:
                resolved_metric = DistanceMetric(metric.lower())
            except ValueError:
                raise QueryError(f"Unsupported distance metric: {metric}")
        else:
            raise QueryError(f"Unsupported distance metric: {metric}")

        if self._store.count() == 0:
            return []

        try:
            vec_list = [float(x) for x in query_vector]
            builder = cast(
                LanceVectorQueryBuilder,
                self._store.table.search(vec_list),
            ).metric(resolved_metric.value).limit(limit)

            if exact:
                builder = builder.bypass_vector_index()

            if filter_sql:
                if prefilter is not None:
                    builder = builder.where(filter_sql, prefilter=prefilter)
                else:
                    builder = builder.where(filter_sql)

            if nprobes is not None:
                builder = builder.nprobes(nprobes)

            if refine_factor is not None:
                builder = builder.refine_factor(refine_factor)

            rows = builder.to_list()
        except Exception as exc:
            raise QueryError(f"Dense vector retrieval failed: {exc}") from exc

        results: list[SearchResult] = []
        for rank, row in enumerate(rows):
            dist = float(row.get("_distance", 0.0))
            if resolved_metric == DistanceMetric.L2:
                score = 1.0 / (1.0 + dist)
            else:
                score = 1.0 - dist

            res = _row_to_search_result(
                row,
                score=score,
                dense_distance=dist,
                dense_rank=rank,
                include_vector=include_vector,
            )
            results.append(res)

        return results


class LexicalRetriever:
    """Native lexical BM25 retriever backed by LanceDB FTS index."""

    def __init__(self, store: LanceIndexStore) -> None:
        self._store = store

    @property
    def store(self) -> LanceIndexStore:
        return self._store

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        filter_sql: str | None = None,
        prefilter: bool = True,
        include_vector: bool = True,
    ) -> list[SearchResult]:
        """Execute lexical BM25 search over the full-text search index.

        Args:
            query: Keyword or phrase query string.
            limit: Maximum number of ranked records to return.
            filter_sql: Optional SQL WHERE predicate for metadata filtering.
            prefilter: True to filter before FTS scoring, False for post-filtering.
            include_vector: Whether to attach candidate vector to each SearchResult.

        Returns:
            List of SearchResult objects sorted by BM25 relevance score.
        """
        if limit <= 0:
            raise QueryError("Limit must be greater than zero")

        if not query or not query.strip():
            return []

        if self._store.count() == 0:
            return []

        clean_query = query.strip()

        try:
            builder = cast(
                LanceFtsQueryBuilder,
                self._store.table.search(clean_query, query_type="fts"),
            ).limit(limit)

            if filter_sql:
                builder = builder.where(filter_sql, prefilter=prefilter)

            rows = builder.to_list()
        except ValueError as exc:
            msg = str(exc)
            if "Cannot perform full text search" in msg:
                # If FTS index has not been created yet, attempt to build it once
                try:
                    self._store.create_fts_index(field="text", replace=True)
                    builder = cast(
                        LanceFtsQueryBuilder,
                        self._store.table.search(clean_query, query_type="fts"),
                    ).limit(limit)
                    if filter_sql:
                        builder = builder.where(filter_sql, prefilter=prefilter)
                    rows = builder.to_list()
                except Exception as inner_exc:
                    raise QueryError(f"Lexical retrieval failed: {inner_exc}") from inner_exc
            else:
                raise QueryError(f"Lexical retrieval failed: {exc}") from exc
        except Exception as exc:
            raise QueryError(f"Lexical retrieval failed: {exc}") from exc

        results: list[SearchResult] = []
        for rank, row in enumerate(rows):
            score = float(row.get("_score", 0.0))
            res = _row_to_search_result(
                row,
                score=score,
                lexical_score=score,
                lexical_rank=rank,
                include_vector=include_vector,
            )
            results.append(res)

        return results

