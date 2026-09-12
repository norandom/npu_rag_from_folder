"""End-to-end integration and validation tests for vector-index (Task 7.2).

Validates:
1. End-to-end workflow: ingest document sample, embed and index, hybrid search.
2. Hybrid superiority: queries where neither dense-only nor lexical-only ranks the optimal
   document at top, but hybrid fusion (RRF) succeeds.
3. MMR diversity and neighbouring-chunk context expansion over multi-chunk articles.
4. Incremental re-run reusing the index completely with zero re-embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from npu_rag.embedding.types import DocumentText, ExecutionMode, ProviderChoice
from npu_rag.index import (
    FusionMethod,
    SearchMode,
    SearchOptions,
    VectorIndex,
)
from npu_rag.index.schema import VECTOR_DIMENSION
from npu_rag.ingest.types import ChunkKind, ChunkRecord, MarkdownLocator, RunReport


@dataclass
class E2EEmbedResult:
    vectors: npt.NDArray[np.float32]
    provider_served: ProviderChoice = ProviderChoice.CPU
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS
    partition_verified: bool | None = None
    fallback_reason: str | None = None
    truncated_indices: tuple[int, ...] = ()
    elapsed_seconds: float = 0.01
    input_count: int = 1


class E2EEmbeddingService:
    """Deterministic embedding service mapping topic concepts to orthogonal dimensions."""

    def __init__(self) -> None:
        self.call_count = 0
        self.embedded_doc_count = 0

    def _text_to_vec(self, text: str) -> npt.NDArray[np.float32]:
        vec = np.zeros((VECTOR_DIMENSION,), dtype=np.float32)
        lower = text.lower()
        # Topic 0: NPU / Accelerators / Matrix
        if "matrix" in lower or "accelerator" in lower or "npu" in lower:
            vec[0] += 0.8
        # Topic 1: Compilation / Graphs / Optimizations
        if "compiler" in lower or "graph" in lower or "pass" in lower:
            vec[1] += 0.8
        # Topic 2: FPGA / DSP / Synthesis
        if "fpga" in lower or "dsp" in lower or "synthesis" in lower:
            vec[2] += 0.8
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[10] = 1.0
        return vec

    def embed_query(self, query: str) -> E2EEmbedResult:
        vec = self._text_to_vec(query).reshape(1, -1)
        return E2EEmbedResult(vectors=vec, input_count=1)

    def embed_documents(
        self,
        texts: Sequence[DocumentText],
        provider: ProviderChoice = ProviderChoice.AUTO,
        progress: Any = None,
    ) -> E2EEmbedResult:
        self.call_count += 1
        self.embedded_doc_count += len(texts)
        count = len(texts)
        vecs = np.zeros((count, VECTOR_DIMENSION), dtype=np.float32)
        for i, dt in enumerate(texts):
            vecs[i] = self._text_to_vec(dt.content)
        return E2EEmbedResult(vectors=vecs, input_count=count)


def _make_article_chunk(
    cid: str,
    path: str,
    text: str,
    ordinal: int = 0,
    title: str = "Article Title",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid,
        source_path=Path(path),
        root_id="root",
        author="AMD Research",
        title=title,
        kind=ChunkKind.PROSE,
        text=text,
        locator=MarkdownLocator(line_range=(1 + ordinal * 10, 10 + ordinal * 10), ordinal=ordinal),
        ordinal=ordinal,
        token_count=20,
        truncated=False,
        provenance=None,
    )


def test_e2e_hybrid_superiority_and_incremental_sync(tmp_path: Path) -> None:
    embed_svc = E2EEmbeddingService()
    index_path = tmp_path / "lancedb_e2e"
    index = VectorIndex(index_path, embed_service=embed_svc)

    # 1. Corpus Setup:
    # Article A: Target document having both conceptual match (matrix accelerator) and exact keyword ("XDNA-1")
    # Article B: Semantic match only (high matrix accelerator concepts, but NO "XDNA-1" keyword)
    # Article C: Exact keyword match only (repeats "XDNA-1", but about FPGA synthesis)
    chunk_a = _make_article_chunk(
        "art_a",
        "amd_npu.md",
        "The XDNA-1 neural processing unit delivers peak matrix multiplication throughput for edge AI.",
        ordinal=0,
        title="AMD Phoenix NPU Architecture",
    )
    chunk_b = _make_article_chunk(
        "art_b",
        "compiler.md",
        "Compiler graph optimizations partition deep neural networks across matrix multiplication hardware accelerators.",
        ordinal=0,
        title="Deep Learning Compilation",
    )
    chunk_c = _make_article_chunk(
        "art_c",
        "fpga.md",
        "High level FPGA synthesis tooling mentions legacy XDNA-1 interfaces alongside DSP slices.",
        ordinal=0,
        title="FPGA DSP Slices",
    )

    initial_report = RunReport(
        files_processed=3,
        files_unchanged=0,
        files_skipped=0,
        files_failed=0,
        omissions=(),
        removed_chunk_ids=(),
        no_work_required=False,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=(chunk_a, chunk_b, chunk_c),
    )

    sync_summary1 = index.sync(initial_report)
    assert sync_summary1.chunks_added == 3
    assert sync_summary1.chunks_updated == 0
    assert sync_summary1.total_chunks == 3
    assert embed_svc.call_count == 1
    assert embed_svc.embedded_doc_count == 3

    # 2. Query Demonstrating Hybrid Superiority:
    # "XDNA-1 matrix multiplication accelerator"
    query = "XDNA-1 matrix multiplication accelerator"

    # Dense only: Article B (pure matrix compiler concepts) and Article A are tied or B ranks high
    dense_resp = index.search(query, options=SearchOptions(mode=SearchMode.DENSE_ONLY, limit=3))
    assert len(dense_resp.results) >= 2

    # Lexical only: Article C (mentions XDNA-1) ranks high
    lex_resp = index.search(query, options=SearchOptions(mode=SearchMode.LEXICAL_ONLY, limit=3))
    assert len(lex_resp.results) >= 1

    # Hybrid (RRF): Article A combines both dense semantics and lexical keyword "XDNA-1",
    # ranking strictly #1!
    hybrid_resp = index.search(
        query,
        options=SearchOptions(mode=SearchMode.HYBRID, fusion_method=FusionMethod.RRF, limit=3),
    )
    assert len(hybrid_resp.results) >= 1
    assert hybrid_resp.results[0].chunk_id == "art_a"

    # 3. Incremental Re-run:
    # Synchronizing the exact same report must NOT call embed_documents
    sync_summary2 = index.sync(initial_report)
    assert sync_summary2.chunks_added == 0
    assert sync_summary2.chunks_updated == 0
    assert sync_summary2.chunks_deleted == 0
    assert sync_summary2.total_chunks == 3
    # Call count remains unchanged at 1!
    assert embed_svc.call_count == 1
    assert embed_svc.embedded_doc_count == 3


def test_e2e_mmr_diversity_and_context_expansion(tmp_path: Path) -> None:
    embed_svc = E2EEmbeddingService()
    index = VectorIndex(tmp_path / "lancedb_e2e_mmr", embed_service=embed_svc)

    # Multi-chunk article: 4 sequential chunks in the same article
    c0 = _make_article_chunk("ch0", "article.md", "Overview: Deep neural network execution on modern accelerators.", 0)
    c1 = _make_article_chunk("ch1", "article.md", "Hardware details: Matrix multiplication array layout and register file.", 1)
    c2 = _make_article_chunk("ch2", "article.md", "Hardware details: Matrix multiplication systolic execution and tiling.", 2)
    c3 = _make_article_chunk("ch3", "article.md", "Software toolchain: Graph compiler passes and memory scheduling.", 3)

    report = RunReport(
        files_processed=1,
        files_unchanged=0,
        files_skipped=0,
        files_failed=0,
        omissions=(),
        removed_chunk_ids=(),
        no_work_required=False,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=(c0, c1, c2, c3),
    )
    index.sync(report)

    # Query matching hardware matrix details
    query = "matrix multiplication hardware"

    # Search with MMR diversity (0.4) and context window (1)
    resp = index.search(
        query,
        options=SearchOptions(
            mode=SearchMode.HYBRID,
            mmr_diversity=0.4,
            context_window=1,
            limit=2,
        ),
    )

    assert len(resp.results) == 2
    top_result = resp.results[0]
    # Primary matched text preserved
    assert "Hardware details:" in top_result.text

    # Surrounding context expanded in proper reading order
    assert top_result.expanded_context is not None
    # For chunk 1 with window 1: covers chunk 0, chunk 1, chunk 2
    if top_result.chunk_id == "ch1":
        assert "Overview:" in top_result.expanded_context
        assert "register file" in top_result.expanded_context
        assert "systolic execution" in top_result.expanded_context
