"""Tests for LanceIndexStore table operations, upsert, delete, and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from npu_rag.index.schema import VECTOR_DIMENSION, ChunkSchema
from npu_rag.index.store import LanceIndexStore


def _make_schema_item(
    chunk_id: str,
    file_path: str = "doc.md",
    chunk_index: int = 0,
    text: str = "sample text",
) -> ChunkSchema:
    vec = [0.01 * (i % 50) for i in range(VECTOR_DIMENSION)]
    return ChunkSchema(
        chunk_id=chunk_id,
        file_path=file_path,
        file_hash="hash123",
        chunk_index=chunk_index,
        chunk_kind="prose",
        title="Sample Title",
        headings=["Introduction"],
        locator_type="markdown",
        locator_json='{"type": "markdown", "line_range": [1, 5], "ordinal": 0}',
        text=text,
        vector=vec,
        token_count=10,
        created_at=100.0,
    )


def test_store_creation_and_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "lancedb"
    store = LanceIndexStore(db_path)
    assert store.count() == 0

    item1 = _make_schema_item("c1", "file1.md", 0, "text one")
    item2 = _make_schema_item("c2", "file1.md", 1, "text two")
    added = store.add([item1, item2])
    assert added == 2
    assert store.count() == 2

    # Re-open store from same path
    reopened = LanceIndexStore(db_path)
    assert reopened.count() == 2


def test_store_upsert(tmp_path: Path) -> None:
    db_path = tmp_path / "lancedb"
    store = LanceIndexStore(db_path)

    item1 = _make_schema_item("c1", "file1.md", 0, "initial text")
    store.add([item1])
    assert store.count() == 1

    # Upsert c1 with updated text, plus new c2
    item1_updated = _make_schema_item("c1", "file1.md", 0, "updated text")
    item2 = _make_schema_item("c2", "file2.md", 0, "new chunk")
    store.upsert([item1_updated, item2])

    assert store.count() == 2

    # Verify c1 text is updated
    results = store.table.search().where("chunk_id = 'c1'").limit(1).to_list()
    assert len(results) == 1
    assert results[0]["text"] == "updated text"


def test_store_delete_by_chunk_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "lancedb"
    store = LanceIndexStore(db_path)

    store.add([
        _make_schema_item("c1", "file1.md", 0),
        _make_schema_item("c2", "file1.md", 1),
        _make_schema_item("c3", "file2.md", 0),
    ])
    assert store.count() == 3

    deleted = store.delete_by_chunk_ids(["c1", "c3"])
    assert deleted == 2
    assert store.count() == 1

    remaining = store.table.search().limit(10).to_list()
    assert len(remaining) == 1
    assert remaining[0]["chunk_id"] == "c2"


def test_store_delete_by_file_path(tmp_path: Path) -> None:
    db_path = tmp_path / "lancedb"
    store = LanceIndexStore(db_path)

    store.add([
        _make_schema_item("c1", "docs/report.pdf", 0),
        _make_schema_item("c2", "docs/report.pdf", 1),
        _make_schema_item("c3", "docs/other.md", 0),
    ])
    assert store.count() == 3

    store.delete_by_file_path("docs/report.pdf")
    assert store.count() == 1

    remaining = store.table.search().limit(10).to_list()
    assert len(remaining) == 1
    assert remaining[0]["chunk_id"] == "c3"


def test_store_delete_by_predicate(tmp_path: Path) -> None:
    db_path = tmp_path / "lancedb"
    store = LanceIndexStore(db_path)

    store.add([
        _make_schema_item("c1", "file1.md", 0),
        _make_schema_item("c2", "file1.md", 1),
        _make_schema_item("c3", "file2.md", 0),
    ])

    store.delete_by_predicate("chunk_index = 0")
    assert store.count() == 1

    remaining = store.table.search().limit(10).to_list()
    assert len(remaining) == 1
    assert remaining[0]["chunk_id"] == "c2"


def test_store_compact_and_stats(tmp_path: Path) -> None:
    db_path = tmp_path / "lancedb"
    store = LanceIndexStore(db_path)

    store.add([
        _make_schema_item("c1", "file1.md", 0),
        _make_schema_item("c2", "file2.md", 0),
    ])

    store.compact()
    stats = store.stats()
    assert stats.total_chunks == 2
    assert stats.total_documents == 2
    assert stats.vector_dimension == VECTOR_DIMENSION
    assert stats.index_size_bytes > 0


def test_store_fts_index_lifecycle(tmp_path: Path) -> None:
    db_path = tmp_path / "lancedb"
    store = LanceIndexStore(db_path)

    store.add([
        _make_schema_item("c1", "file1.md", 0, text="Neural processing units accelerate matrix multiplication"),
        _make_schema_item("c2", "file2.md", 0, text="Traditional database indexing uses b-trees and hash tables"),
    ])

    # 1. Create FTS index
    store.create_fts_index(field="text", replace=True)
    stats = store.stats()
    assert stats.has_fts_index is True

    # 2. Query FTS
    results = store.table.search("neural processing", query_type="fts").limit(5).to_list()
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "c1"

    # 3. Refresh/rebuild index after new records
    store.add([
        _make_schema_item("c3", "file3.md", 0, text="Deep learning architectures rely heavily on neural networks"),
    ])
    store.create_fts_index(field="text", replace=True)

    results_updated = store.table.search("neural", query_type="fts").limit(5).to_list()
    assert len(results_updated) == 2
    found_ids = {r["chunk_id"] for r in results_updated}
    assert found_ids == {"c1", "c3"}
