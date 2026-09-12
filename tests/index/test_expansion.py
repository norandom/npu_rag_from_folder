"""Tests for neighbouring-chunk context expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from npu_rag.index.errors import ConfigurationError
from npu_rag.index.expansion import ContextExpander, expand_context
from npu_rag.index.schema import VECTOR_DIMENSION, ChunkSchema
from npu_rag.index.store import LanceIndexStore
from npu_rag.index.types import SearchResult


def _make_schema_chunk(
    chunk_id: str,
    file_path: str,
    chunk_index: int,
    text: str,
) -> ChunkSchema:
    vec = [0.0] * VECTOR_DIMENSION
    return ChunkSchema(
        chunk_id=chunk_id,
        file_path=file_path,
        file_hash="hash123",
        chunk_index=chunk_index,
        chunk_kind="prose",
        title="Document Title",
        headings=["Chapter 1"],
        locator_type="markdown",
        locator_json="{}",
        text=text,
        vector=vec,
        token_count=10,
        created_at=100.0,
    )


def _make_result_from_schema(item: ChunkSchema, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk_id=item.chunk_id,
        file_path=item.file_path,
        title=item.title,
        chunk_kind=item.chunk_kind,
        chunk_index=item.chunk_index,
        headings=tuple(item.headings),
        locator_type=item.locator_type,
        locator_json=item.locator_json,
        text=item.text,
        score=score,
        token_count=item.token_count,
        vector=tuple(item.vector),
    )


@pytest.fixture
def multi_chunk_store(tmp_path: Path) -> LanceIndexStore:
    store = LanceIndexStore(tmp_path / "lancedb_expansion")
    # File A has 4 sequential chunks: 0, 1, 2, 3
    chunks = [
        _make_schema_chunk("a0", "docA.md", 0, "Paragraph 0: Introduction to NPU architecture."),
        _make_schema_chunk("a1", "docA.md", 1, "Paragraph 1: Matrix multiplication hardware details."),
        _make_schema_chunk("a2", "docA.md", 2, "Paragraph 2: Compilation and graph optimization steps."),
        _make_schema_chunk("a3", "docA.md", 3, "Paragraph 3: Benchmarking and final conclusions."),
        # File B has chunks that should not bleed into File A expansion
        _make_schema_chunk("b0", "docB.md", 0, "Doc B introductory text."),
        _make_schema_chunk("b1", "docB.md", 1, "Doc B middle content."),
    ]
    store.add(chunks)
    return store


def test_context_expansion_window_one(multi_chunk_store: LanceIndexStore) -> None:
    expander = ContextExpander(multi_chunk_store)
    c1 = _make_result_from_schema(_make_schema_chunk("a1", "docA.md", 1, "Paragraph 1: Matrix multiplication hardware details."))

    expanded = expander.expand([c1], window=1)
    assert len(expanded) == 1
    res = expanded[0]

    # Primary text remains unchanged
    assert res.text == "Paragraph 1: Matrix multiplication hardware details."
    # Expanded context contains chunk 0, chunk 1, and chunk 2
    assert res.expanded_context is not None
    assert "Paragraph 0:" in res.expanded_context
    assert "Paragraph 1:" in res.expanded_context
    assert "Paragraph 2:" in res.expanded_context
    assert "Paragraph 3:" not in res.expanded_context
    # File B text is not included
    assert "Doc B" not in res.expanded_context


def test_context_expansion_lower_boundary(multi_chunk_store: LanceIndexStore) -> None:
    # First chunk in document (chunk_index = 0)
    c0 = _make_result_from_schema(_make_schema_chunk("a0", "docA.md", 0, "Paragraph 0: Introduction to NPU architecture."))
    expanded = expand_context(multi_chunk_store, [c0], window=1)

    assert len(expanded) == 1
    res = expanded[0]
    assert res.expanded_context is not None
    # Contains chunk 0 and chunk 1
    assert "Paragraph 0:" in res.expanded_context
    assert "Paragraph 1:" in res.expanded_context
    assert "Paragraph 2:" not in res.expanded_context


def test_context_expansion_upper_boundary(multi_chunk_store: LanceIndexStore) -> None:
    # Final chunk in document (chunk_index = 3)
    c3 = _make_result_from_schema(_make_schema_chunk("a3", "docA.md", 3, "Paragraph 3: Benchmarking and final conclusions."))
    expanded = expand_context(multi_chunk_store, [c3], window=1)

    assert len(expanded) == 1
    res = expanded[0]
    assert res.expanded_context is not None
    # Contains chunk 2 and chunk 3
    assert "Paragraph 2:" in res.expanded_context
    assert "Paragraph 3:" in res.expanded_context
    assert "Paragraph 0:" not in res.expanded_context


def test_context_expansion_window_two(multi_chunk_store: LanceIndexStore) -> None:
    c1 = _make_result_from_schema(_make_schema_chunk("a1", "docA.md", 1, "Paragraph 1: Matrix multiplication hardware details."))
    expanded = expand_context(multi_chunk_store, [c1], window=2)

    assert len(expanded) == 1
    res = expanded[0]
    assert res.expanded_context is not None
    # Window 2 around index 1 covers 0, 1, 2, 3 (entire document)
    for i in range(4):
        assert f"Paragraph {i}:" in res.expanded_context


def test_context_expansion_window_zero_and_empty(multi_chunk_store: LanceIndexStore) -> None:
    expander = ContextExpander(multi_chunk_store)
    assert expander.expand([], window=1) == []

    c1 = _make_result_from_schema(_make_schema_chunk("a1", "docA.md", 1, "text"))
    res = expander.expand([c1], window=0)
    assert len(res) == 1
    assert res[0].expanded_context is None


def test_context_expansion_negative_window(multi_chunk_store: LanceIndexStore) -> None:
    expander = ContextExpander(multi_chunk_store)
    with pytest.raises(ConfigurationError, match="must be non-negative"):
        expander.expand([], window=-1)
