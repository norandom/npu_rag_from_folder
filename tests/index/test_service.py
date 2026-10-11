"""Tests for VectorIndex facade and public service API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from npu_rag.embedding.types import DocumentText, ExecutionMode, ProviderChoice
from npu_rag.index import (
    DistanceMetric,
    FusionMethod,
    IndexStats,
    PassThroughReranker,
    QueryError,
    Reranker,
    SearchMode,
    SearchOptions,
    SearchResponse,
    SearchResult,
    SyncError,
    SyncSummary,
    VectorIndex,
)
from npu_rag.index.schema import VECTOR_DIMENSION
from npu_rag.index.store import LanceIndexStore
from npu_rag.ingest.types import ChunkKind, ChunkRecord, MarkdownLocator, RunReport


@dataclass
class MockEmbedResult:
    vectors: npt.NDArray[np.float32]
    provider_served: ProviderChoice = ProviderChoice.CPU
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS
    partition_verified: bool | None = None
    fallback_reason: str | None = None
    truncated_indices: tuple[int, ...] = ()
    elapsed_seconds: float = 0.01
    input_count: int = 1


class MockEmbeddingService:
    """Mock service recording calls to embed_query and embed_documents."""

    def __init__(self) -> None:
        self.query_calls: list[str] = []
        self.doc_calls: list[list[DocumentText]] = []

    def embed_query(self, query: str) -> MockEmbedResult:
        self.query_calls.append(query)
        vec = np.zeros((1, VECTOR_DIMENSION), dtype=np.float32)
        # Give query vector active coordinate based on content
        if "neural" in query.lower():
            vec[0, 0] = 1.0
        elif "quantum" in query.lower():
            vec[0, 1] = 1.0
        else:
            vec[0, 2] = 1.0
        return MockEmbedResult(vectors=vec, input_count=1)

    def embed_documents(
        self,
        texts: Sequence[DocumentText],
        provider: ProviderChoice = ProviderChoice.AUTO,
        progress: Any = None,
    ) -> MockEmbedResult:
        self.doc_calls.append(list(texts))
        count = len(texts)
        vecs = np.zeros((count, VECTOR_DIMENSION), dtype=np.float32)
        for i, dt in enumerate(texts):
            if "neural" in dt.content.lower():
                vecs[i, 0] = 1.0
            elif "quantum" in dt.content.lower():
                vecs[i, 1] = 1.0
            else:
                vecs[i, 2] = 1.0
        return MockEmbedResult(vectors=vecs, input_count=count)


def _make_chunk(cid: str, path: str, text: str, ordinal: int = 0) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid,
        source_path=Path(path),
        root_id="root",
        author="author",
        title=f"Title {cid}",
        kind=ChunkKind.PROSE,
        text=text,
        locator=MarkdownLocator(line_range=(1, 5), ordinal=ordinal),
        ordinal=ordinal,
        token_count=12,
        truncated=False,
        provenance=None,
    )


@pytest.fixture
def populated_index(tmp_path: Path) -> tuple[VectorIndex, MockEmbeddingService]:
    embed_svc = MockEmbeddingService()
    index = VectorIndex(tmp_path / "lancedb_facade", embed_service=embed_svc)

    # Sync sample corpus
    chunks = [
        _make_chunk("c1", "hardware.md", "Neural processing units deliver high throughput matrix math.", 0),
        _make_chunk("c2", "hardware.md", "Quantum annealing explores combinatorial optimization spaces.", 1),
        _make_chunk("c3", "software.md", "Compiler passes optimize dataflow graphs for target accelerators.", 0),
    ]
    report = RunReport(
        files_processed=2,
        files_unchanged=0,
        files_skipped=0,
        files_failed=0,
        omissions=(),
        removed_chunk_ids=(),
        no_work_required=False,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=tuple(chunks),
    )
    index.sync(report)
    return index, embed_svc


def test_public_exports() -> None:
    import npu_rag.index as idx

    assert hasattr(idx, "VectorIndex")
    assert hasattr(idx, "SearchOptions")
    assert hasattr(idx, "SearchResponse")
    assert hasattr(idx, "SearchResult")
    assert hasattr(idx, "SearchMode")
    assert hasattr(idx, "DistanceMetric")
    assert hasattr(idx, "FusionMethod")
    assert hasattr(idx, "SyncSummary")
    assert hasattr(idx, "IndexStats")
    assert hasattr(idx, "Reranker")
    assert hasattr(idx, "PassThroughReranker")


def test_search_dense_only_uses_embed_query(
    populated_index: tuple[VectorIndex, MockEmbeddingService],
) -> None:
    index, embed_svc = populated_index
    embed_svc.query_calls.clear()

    resp = index.search(
        "neural network accelerators",
        options=SearchOptions(mode=SearchMode.DENSE_ONLY, limit=2),
    )

    assert isinstance(resp, SearchResponse)
    assert resp.mode == SearchMode.DENSE_ONLY
    assert len(embed_svc.query_calls) == 1
    assert embed_svc.query_calls[0] == "neural network accelerators"
    assert len(resp.results) > 0
    assert resp.results[0].chunk_id == "c1"


def test_search_lexical_only_skips_embed_query(
    populated_index: tuple[VectorIndex, MockEmbeddingService],
) -> None:
    index, embed_svc = populated_index
    embed_svc.query_calls.clear()

    resp = index.search(
        "quantum annealing",
        options=SearchOptions(mode=SearchMode.LEXICAL_ONLY, limit=2),
    )

    assert resp.mode == SearchMode.LEXICAL_ONLY
    assert len(embed_svc.query_calls) == 0  # No embedding called
    assert len(resp.results) > 0
    assert resp.results[0].chunk_id == "c2"


def test_search_hybrid_rrf_and_linear(
    populated_index: tuple[VectorIndex, MockEmbeddingService],
) -> None:
    index, _embed_svc = populated_index

    # 1. RRF mode
    resp_rrf = index.search(
        "neural processing",
        options=SearchOptions(mode=SearchMode.HYBRID, fusion_method=FusionMethod.RRF, limit=3),
    )
    assert len(resp_rrf.results) > 0
    assert resp_rrf.results[0].chunk_id == "c1"

    # 2. Linear combination mode
    resp_linear = index.search(
        "neural processing",
        options=SearchOptions(
            mode=SearchMode.HYBRID,
            fusion_method=FusionMethod.LINEAR,
            linear_alpha=0.6,
            limit=3,
        ),
    )
    assert len(resp_linear.results) > 0
    assert resp_linear.results[0].chunk_id == "c1"


def test_pluggable_reranker(populated_index: tuple[VectorIndex, MockEmbeddingService]) -> None:
    index, _embed_svc = populated_index

    class ReverseReranker:
        def rerank(self, query: str, results: Sequence[SearchResult]) -> list[SearchResult]:
            return list(reversed(results))

    index._reranker = ReverseReranker()
    resp = index.search(
        "matrix math",
        options=SearchOptions(mode=SearchMode.LEXICAL_ONLY, limit=10),
    )
    assert len(resp.results) > 0


def test_search_with_mmr_and_context_expansion(
    populated_index: tuple[VectorIndex, MockEmbeddingService],
) -> None:
    index, _embed_svc = populated_index

    resp = index.search(
        "matrix",
        options=SearchOptions(
            mode=SearchMode.HYBRID,
            mmr_diversity=0.3,
            context_window=1,
            limit=2,
        ),
    )
    assert len(resp.results) > 0
    top = resp.results[0]
    # Context expansion should have populated surrounding text
    assert top.expanded_context is not None
    assert top.text in top.expanded_context


def test_search_validation_errors(populated_index: tuple[VectorIndex, MockEmbeddingService]) -> None:
    index, _ = populated_index

    # Empty query
    empty_resp = index.search("")
    assert empty_resp.results == ()
    assert empty_resp.total_candidates == 0

    # Limit <= 0
    with pytest.raises(QueryError, match="greater than zero"):
        index.search("query", options=SearchOptions(limit=0))

    # Dense search with no embedding service
    index_no_embed = VectorIndex(index.store)
    with pytest.raises(QueryError, match="no embed_service was provided"):
        index_no_embed.search("query", options=SearchOptions(mode=SearchMode.DENSE_ONLY))


def test_index_stats(populated_index: tuple[VectorIndex, MockEmbeddingService]) -> None:
    index, _ = populated_index
    stats = index.stats()
    assert isinstance(stats, IndexStats)
    assert stats.total_chunks == 3
    assert stats.total_documents == 2


# --------------------------------------------------------------------------
# Candidate depth (`SearchOptions.fetch_k`)
#
# design.md section 3 already wrote `fetch_k` into the retrieval pseudocode
# (lines 118 and 122), but the implementation hardcoded `max(limit * 4, 30)`
# with no seam, so `search-cli`'s `--fetch-k` flag had nowhere to go and was
# omitted. The field was added on 2026-10-11 to close that gap.
# --------------------------------------------------------------------------


class _DepthRecorder:
    """Stands in for a retriever and records the candidate depth asked for."""

    def __init__(self) -> None:
        self.limits: list[int] = []

    def retrieve(self, *args: Any, limit: int, **kwargs: Any) -> list[SearchResult]:
        self.limits.append(limit)
        return []


def _recorded_depths(
    index: VectorIndex, options: SearchOptions
) -> tuple[list[int], list[int]]:
    dense = _DepthRecorder()
    lexical = _DepthRecorder()
    index._dense_retriever = dense  # type: ignore[assignment]
    index._lexical_retriever = lexical  # type: ignore[assignment]
    index.search("query", options=options)
    return dense.limits, lexical.limits


def test_an_unset_fetch_k_keeps_the_derived_depth(
    populated_index: tuple[VectorIndex, MockEmbeddingService],
) -> None:
    """`None` must reproduce exactly what every caller got before the field existed.

    Both arms of `max(limit * 4, 30)` are checked, because a mutant that drops
    either one still satisfies the other: limit 10 exercises the floor (40 > 30
    would hide it) and limit 2 exercises the product.
    """
    index, _ = populated_index

    dense, _lex = _recorded_depths(index, SearchOptions(limit=10, mode=SearchMode.DENSE_ONLY))
    assert dense == [40], "limit * 4 must win when it exceeds the floor"

    dense, _lex = _recorded_depths(index, SearchOptions(limit=2, mode=SearchMode.DENSE_ONLY))
    assert dense == [30], "the floor of 30 must win when limit * 4 is smaller"


def test_an_explicit_fetch_k_is_honoured_verbatim(
    populated_index: tuple[VectorIndex, MockEmbeddingService],
) -> None:
    """The named depth reaches the retriever unchanged, in every mode."""
    index, _ = populated_index

    dense, _lex = _recorded_depths(index, SearchOptions(limit=10, fetch_k=7, mode=SearchMode.DENSE_ONLY))
    assert dense == [7]

    _dense, lexical = _recorded_depths(index, SearchOptions(limit=10, fetch_k=7, mode=SearchMode.LEXICAL_ONLY))
    assert lexical == [7]

    dense, lexical = _recorded_depths(index, SearchOptions(limit=10, fetch_k=7, mode=SearchMode.HYBRID))
    assert dense == [7] and lexical == [7], "hybrid must use one depth for both arms"


def test_an_explicit_fetch_k_is_not_widened_by_the_floor(
    populated_index: tuple[VectorIndex, MockEmbeddingService],
) -> None:
    """A deliberately narrow pool stays narrow.

    `max(30, fetch_k)` would look correct against every value above 30 and would
    quietly turn the parameter into a suggestion. A caller asking for 3 on a
    large index means 3.
    """
    index, _ = populated_index
    dense, _lex = _recorded_depths(index, SearchOptions(limit=10, fetch_k=3, mode=SearchMode.DENSE_ONLY))
    assert dense == [3]


def test_fetch_k_is_independent_of_limit(
    populated_index: tuple[VectorIndex, MockEmbeddingService],
) -> None:
    """A control: the depth must not be derived from `limit` once it is named."""
    index, _ = populated_index
    first, _lex = _recorded_depths(index, SearchOptions(limit=1, fetch_k=25, mode=SearchMode.DENSE_ONLY))
    second, _lex2 = _recorded_depths(index, SearchOptions(limit=50, fetch_k=25, mode=SearchMode.DENSE_ONLY))
    assert first == second == [25]
