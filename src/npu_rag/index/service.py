"""Unified VectorIndex query orchestration facade and public service API."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from npu_rag.embedding.types import ProviderChoice
from npu_rag.index.diversity import DocumentRollup, maximal_marginal_relevance
from npu_rag.index.errors import QueryError, SyncError
from npu_rag.index.expansion import ContextExpander
from npu_rag.index.fusion import linear_combination_fusion, reciprocal_rank_fusion
from npu_rag.index.rerank import PassThroughReranker, Reranker
from npu_rag.index.retriever import DenseRetriever, LexicalRetriever
from npu_rag.index.store import LanceIndexStore
from npu_rag.index.sync import IndexSynchronizer
from npu_rag.index.types import (
    DistanceMetric,
    FusionMethod,
    IndexStats,
    SearchMode,
    SearchOptions,
    SearchResponse,
    SearchResult,
    SyncSummary,
)
from npu_rag.ingest.types import RunReport


def _get_query_vector(embed_service: Any, query: str) -> Sequence[float]:
    """Obtain query vector from embed_service with asymmetric query prefix."""
    if hasattr(embed_service, "embed_query"):
        res = embed_service.embed_query(query)
        if hasattr(res, "vectors"):
            return list(res.vectors[0])
        return list(res)
    elif hasattr(embed_service, "embed_queries"):
        res = embed_service.embed_queries([query], provider=ProviderChoice.AUTO)
        if hasattr(res, "vectors"):
            return list(res.vectors[0])
        return list(res[0])
    raise QueryError("Embedding service does not provide embed_query or embed_queries")


class VectorIndex:
    """Unified facade coordinating storage, search, sync, and diversity ranking."""

    def __init__(
        self,
        store: LanceIndexStore | Path | str,
        embed_service: Any | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if isinstance(store, (Path, str)):
            self._store = LanceIndexStore(store)
        else:
            self._store = store

        self._embed_service = embed_service
        self._reranker: Reranker = reranker or PassThroughReranker()

        self._dense_retriever = DenseRetriever(self._store)
        self._lexical_retriever = LexicalRetriever(self._store)
        self._synchronizer = IndexSynchronizer(self._store)
        self._context_expander = ContextExpander(self._store)

    @property
    def store(self) -> LanceIndexStore:
        return self._store

    @property
    def embed_service(self) -> Any | None:
        return self._embed_service

    @property
    def reranker(self) -> Reranker:
        return self._reranker

    def stats(self) -> IndexStats:
        """Return high-level index storage statistics."""
        return self._store.stats()

    def sync(
        self,
        report: RunReport,
        embed_service: Any | None = None,
        *,
        provider: ProviderChoice | str = ProviderChoice.AUTO,
        batch_size: int = 64,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SyncSummary:
        """Synchronize ingest report chunks with vector store and FTS index."""
        service = embed_service or self._embed_service
        if service is None:
            raise SyncError("Cannot synchronize: no embed_service was provided")
        return self._synchronizer.sync(
            report,
            service,
            provider=provider,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

    def search(
        self,
        query: str,
        options: SearchOptions | None = None,
        embed_service: Any | None = None,
    ) -> SearchResponse:
        """Execute full query retrieval pipeline end-to-end.

        Pipeline stages:
          1. Candidate Retrieval (Dense, Lexical, or Hybrid)
          2. Score Fusion (RRF or Linear Combination)
          3. Reranking (optional cross-encoder)
          4. Diversity Control (MMR)
          5. Document Rollup / Capping
          6. Truncation to limit
          7. Context Expansion (adjacent chunks)
        """
        t0 = time.perf_counter()
        opts = options or SearchOptions()
        if opts.limit <= 0:
            raise QueryError("Search limit must be greater than zero")

        if not query or not query.strip():
            return SearchResponse(
                query=query,
                mode=opts.mode,
                results=(),
                total_candidates=0,
                elapsed_seconds=time.perf_counter() - t0,
            )

        clean_query = query.strip()
        service = embed_service or self._embed_service

        # Over-fetch candidates to ensure downstream stages (fusion, MMR, rollup)
        # have adequate candidate depth before final truncation.
        #
        # An explicit `fetch_k` wins; `None` keeps the derived default, so a
        # caller that never sets it sees exactly the behaviour this line had
        # before the field existed. The floor of 30 is deliberately NOT applied
        # to an explicit value: a caller asking for a narrow pool on a large
        # index means it, and silently widening the request would make the
        # parameter a suggestion rather than a setting.
        fetch_k = opts.fetch_k if opts.fetch_k is not None else max(opts.limit * 4, 30)

        # 1. Retrieval
        candidates: list[SearchResult] = []
        q_vec: Sequence[float] | None = None

        if opts.mode == SearchMode.DENSE_ONLY:
            if service is None:
                raise QueryError("Dense search requested but no embed_service was provided")
            q_vec = _get_query_vector(service, clean_query)
            candidates = self._dense_retriever.retrieve(
                q_vec,
                limit=fetch_k,
                metric=opts.metric,
                filter_sql=opts.filter_sql,
            )

        elif opts.mode == SearchMode.LEXICAL_ONLY:
            candidates = self._lexical_retriever.retrieve(
                clean_query,
                limit=fetch_k,
                filter_sql=opts.filter_sql,
            )

        elif opts.mode == SearchMode.HYBRID:
            if service is None:
                raise QueryError("Hybrid search requested but no embed_service was provided")
            q_vec = _get_query_vector(service, clean_query)
            dense_cands = self._dense_retriever.retrieve(
                q_vec,
                limit=fetch_k,
                metric=opts.metric,
                filter_sql=opts.filter_sql,
            )
            lex_cands = self._lexical_retriever.retrieve(
                clean_query,
                limit=fetch_k,
                filter_sql=opts.filter_sql,
            )

            # 2. Score Fusion
            if opts.fusion_method == FusionMethod.RRF:
                candidates = reciprocal_rank_fusion(
                    dense_cands,
                    lex_cands,
                    k=opts.rrf_k,
                    dense_weight=opts.dense_weight,
                    lexical_weight=opts.lexical_weight,
                    limit=fetch_k,
                )
            elif opts.fusion_method == FusionMethod.LINEAR:
                candidates = linear_combination_fusion(
                    dense_cands,
                    lex_cands,
                    alpha=opts.linear_alpha,
                    limit=fetch_k,
                )
            else:
                raise QueryError(f"Unsupported fusion method: {opts.fusion_method}")
        else:
            raise QueryError(f"Unsupported search mode: {opts.mode}")

        total_candidates = len(candidates)

        # 3. Reranking Hook
        if candidates:
            candidates = self._reranker.rerank(clean_query, candidates)

        # 4. Diversity Control (MMR)
        if opts.mmr_diversity is not None and opts.mmr_diversity > 0.0 and candidates:
            if q_vec is None:
                if service is not None:
                    q_vec = _get_query_vector(service, clean_query)
            if q_vec is not None:
                # Map diversity factor d in [0, 1] to lambda = (1.0 - d)
                lambda_val = max(0.0, min(1.0, 1.0 - opts.mmr_diversity))
                candidates = maximal_marginal_relevance(
                    q_vec,
                    candidates,
                    lambda_mult=lambda_val,
                    limit=fetch_k,
                )

        # 5. Document Rollup / Capping
        if (opts.document_rollup or opts.max_chunks_per_document is not None) and candidates:
            rollup = DocumentRollup(
                max_chunks_per_document=opts.max_chunks_per_document,
                rollup=opts.document_rollup,
            )
            candidates = rollup.apply(candidates)

        # 6. Truncate to limit
        final_results = candidates[: opts.limit]

        # 7. Neighbouring-chunk Context Expansion
        if opts.context_window > 0 and final_results:
            final_results = self._context_expander.expand(
                final_results,
                window=opts.context_window,
            )

        elapsed = time.perf_counter() - t0
        return SearchResponse(
            query=clean_query,
            mode=opts.mode,
            results=tuple(final_results),
            total_candidates=total_candidates,
            elapsed_seconds=elapsed,
        )
