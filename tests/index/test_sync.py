"""Tests for IndexSynchronizer bridge between ingest and index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from npu_rag.embedding.types import DocumentText, ExecutionMode, ProviderChoice
from npu_rag.index.errors import SyncError
from npu_rag.index.schema import VECTOR_DIMENSION
from npu_rag.index.store import LanceIndexStore
from npu_rag.index.sync import IndexSynchronizer
from npu_rag.ingest.types import (
    ChunkKind,
    ChunkRecord,
    MarkdownLocator,
    RunReport,
)


@dataclass
class FakeEmbedResult:
    vectors: npt.NDArray[np.float32]
    provider_served: ProviderChoice = ProviderChoice.CPU
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS
    partition_verified: bool | None = None
    fallback_reason: str | None = None
    truncated_indices: tuple[int, ...] = ()
    elapsed_seconds: float = 0.01
    input_count: int = 1


class FakeEmbeddingService:
    """Mock embedding service for unit testing without ONNX/NPU hardware."""

    def __init__(self) -> None:
        self.call_count = 0
        self.embedded_texts: list[DocumentText] = []

    def embed_documents(
        self,
        texts: Sequence[DocumentText],
        provider: ProviderChoice = ProviderChoice.AUTO,
        progress: Any = None,
    ) -> FakeEmbedResult:
        self.call_count += 1
        self.embedded_texts.extend(texts)
        count = len(texts)
        # Create deterministic pseudo-vectors
        vectors = np.zeros((count, VECTOR_DIMENSION), dtype=np.float32)
        for i in range(count):
            vectors[i, 0] = 1.0
            vectors[i, 1] = float(i % 10)
        return FakeEmbedResult(vectors=vectors, input_count=count)


def _make_ingest_chunk(
    chunk_id: str,
    source_path: str = "doc1.md",
    text: str = "sample text",
    ordinal: int = 0,
    title: str | None = "Document Title",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source_path=Path(source_path),
        root_id="root_1",
        author="author",
        title=title,
        kind=ChunkKind.PROSE,
        text=text,
        locator=MarkdownLocator(line_range=(1, 10), ordinal=ordinal),
        ordinal=ordinal,
        token_count=15,
        truncated=False,
        provenance=None,
    )


def _make_report(
    records: Sequence[ChunkRecord] = (),
    removed_chunk_ids: Sequence[str] = (),
    files_processed: int = 1,
    files_unchanged: int = 0,
) -> RunReport:
    return RunReport(
        files_processed=files_processed,
        files_unchanged=files_unchanged,
        files_skipped=0,
        files_failed=0,
        omissions=(),
        removed_chunk_ids=tuple(removed_chunk_ids),
        no_work_required=False,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=tuple(records),
    )


def test_initial_sync(tmp_path: Path) -> None:
    store = LanceIndexStore(tmp_path / "lancedb_sync")
    sync = IndexSynchronizer(store)
    embed_service = FakeEmbeddingService()

    c1 = _make_ingest_chunk("c1", source_path="doc1.md", text="First chunk content")
    c2 = _make_ingest_chunk("c2", source_path="doc1.md", text="Second chunk content", ordinal=1)
    c3 = _make_ingest_chunk("c3", source_path="doc2.md", text="Other document chunk")

    report = _make_report(records=[c1, c2, c3])
    summary = sync.sync(report, embed_service)

    assert summary.chunks_added == 3
    assert summary.chunks_updated == 0
    assert summary.chunks_deleted == 0
    assert summary.total_chunks == 3
    assert summary.files_indexed == 2
    assert embed_service.call_count == 1
    assert store.stats().has_fts_index is True


def test_unchanged_sync_skips_embedding(tmp_path: Path) -> None:
    store = LanceIndexStore(tmp_path / "lancedb_sync")
    sync = IndexSynchronizer(store)
    embed_service = FakeEmbeddingService()

    c1 = _make_ingest_chunk("c1", source_path="doc1.md", text="First chunk content")
    c2 = _make_ingest_chunk("c2", source_path="doc1.md", text="Second chunk content", ordinal=1)

    report = _make_report(records=[c1, c2])
    # 1. First sync
    summary1 = sync.sync(report, embed_service)
    assert summary1.chunks_added == 2
    assert embed_service.call_count == 1

    # 2. Second sync with identical report
    summary2 = sync.sync(report, embed_service)
    assert summary2.chunks_added == 0
    assert summary2.chunks_updated == 0
    assert summary2.chunks_deleted == 0
    assert summary2.total_chunks == 2
    # embed_service should NOT have been called again!
    assert embed_service.call_count == 1


def test_modified_chunk_sync(tmp_path: Path) -> None:
    store = LanceIndexStore(tmp_path / "lancedb_sync")
    sync = IndexSynchronizer(store)
    embed_service = FakeEmbeddingService()

    c1 = _make_ingest_chunk("c1", source_path="doc1.md", text="Initial text")
    sync.sync(_make_report(records=[c1]), embed_service)
    assert embed_service.call_count == 1

    # Chunk c1 has modified text
    c1_modified = _make_ingest_chunk("c1", source_path="doc1.md", text="Updated text after edit")
    summary = sync.sync(_make_report(records=[c1_modified]), embed_service)

    assert summary.chunks_updated == 1
    assert summary.chunks_added == 0
    assert summary.total_chunks == 1
    assert embed_service.call_count == 2

    # Check updated content is stored
    stored_row = store.table.search().where("chunk_id = 'c1'").to_list()[0]
    assert stored_row["text"] == "Updated text after edit"


def test_deleted_chunks_purge(tmp_path: Path) -> None:
    store = LanceIndexStore(tmp_path / "lancedb_sync")
    sync = IndexSynchronizer(store)
    embed_service = FakeEmbeddingService()

    c1 = _make_ingest_chunk("c1", source_path="doc1.md", text="Chunk 1")
    c2 = _make_ingest_chunk("c2", source_path="doc2.md", text="Chunk 2")
    sync.sync(_make_report(records=[c1, c2]), embed_service)
    assert store.count() == 2

    # Report specifies c1 was removed
    del_report = _make_report(records=[c2], removed_chunk_ids=["c1"])
    summary = sync.sync(del_report, embed_service)

    assert summary.chunks_deleted == 1
    assert summary.total_chunks == 1
    assert store.count() == 1
    assert store.table.search().where("chunk_id = 'c1'").to_list() == []


def test_sync_batching_and_progress(tmp_path: Path) -> None:
    store = LanceIndexStore(tmp_path / "lancedb_sync")
    sync = IndexSynchronizer(store)
    embed_service = FakeEmbeddingService()

    chunks = [
        _make_ingest_chunk(f"c{i}", source_path="doc.md", text=f"Chunk {i}", ordinal=i)
        for i in range(5)
    ]

    progress_calls: list[tuple[int, int]] = []

    def on_progress(done: int, total: int) -> None:
        progress_calls.append((done, total))

    summary = sync.sync(
        _make_report(records=chunks),
        embed_service,
        batch_size=2,
        progress_callback=on_progress,
    )

    assert summary.chunks_added == 5
    # 5 chunks with batch_size 2 => 3 batches (2, 2, 1)
    assert embed_service.call_count == 3
    assert len(progress_calls) == 3
    assert progress_calls[-1] == (5, 5)


def test_sync_validation_and_errors(tmp_path: Path) -> None:
    store = LanceIndexStore(tmp_path / "lancedb_sync")
    sync = IndexSynchronizer(store)
    embed_service = FakeEmbeddingService()

    with pytest.raises(SyncError, match="must not be None"):
        sync.sync(None, embed_service)  # type: ignore[arg-type]

    with pytest.raises(SyncError, match="Batch size must be greater than zero"):
        sync.sync(_make_report(), embed_service, batch_size=0)
