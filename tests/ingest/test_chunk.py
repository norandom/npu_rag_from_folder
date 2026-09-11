"""Unit tests for chunking against the token budget (tasks 6.1 - 6.3).

Requirements:
- 6.1: Chunk token budget is an input parameter; mismatch with tokenizer limit is refused.
- 6.2: Measured using the runtime's tokenizer with document title attached.
- 6.3: No emitted chunk exceeds the budget.
- 6.4: Prose chunks have configurable overlap.
- 6.5: Tables, blocks, formulas, figures have no overlap (OVERLAP_POLICY).
- 6.6: Prefer split at heading boundary before budget is reached.
- 7.1: Assemble ChunkRecord with source_path, kind, locator, ordinal, token_count, etc.

Boundary: npu_rag.ingest.chunk
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Final

import pytest

from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.ingest.chunk import OVERLAP_POLICY, Chunker
from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.identity import ParamsFingerprint, chunk_id, params_fingerprint
from npu_rag.ingest.types import (
    BlockSegment,
    ChunkKind,
    ChunkRecord,
    Extracted,
    FigureSegment,
    FormulaSegment,
    MarkdownLocator,
    Omission,
    PageLocator,
    ProseSegment,
    Provenance,
    SheetLocator,
    SourceFile,
    TableSegment,
)

SOURCE = SourceFile(
    path=Path("root/author1/doc.md"),
    root=Path("root"),
    relative_path=Path("author1/doc.md"),
    author="author1",
)
PARAMS = ParamsFingerprint(digest="test_params_digest_123")


# --------------------------------------------------------------------------
# Task 6.1: Measurement, policy, and record assembly
# --------------------------------------------------------------------------


def test_chunker_refuses_budget_mismatch_with_tokenizer(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Requirement 6.1: Mismatch between budget and tokenizer limit is refused."""
    limit = runtime_tokenizer.max_input_tokens
    with pytest.raises(ValueError, match="[Bb]udget"):
        Chunker(runtime_tokenizer, budget=limit + 10, prose_overlap_tokens=32)

    with pytest.raises(ValueError, match="[Bb]udget"):
        Chunker(runtime_tokenizer, budget=limit - 10, prose_overlap_tokens=32)

    # Matching limit is accepted
    chunker = Chunker(runtime_tokenizer, budget=limit, prose_overlap_tokens=32)
    assert chunker.budget == limit


def test_overlap_policy_covers_every_chunk_kind() -> None:
    """Requirement 6.5 / design.md: OVERLAP_POLICY covers every ChunkKind, only PROSE is True."""
    assert set(OVERLAP_POLICY.keys()) == set(ChunkKind)
    for kind, has_overlap in OVERLAP_POLICY.items():
        if kind is ChunkKind.PROSE:
            assert has_overlap is True
        else:
            assert has_overlap is False


def test_measurement_with_title_reports_over_budget_when_content_alone_fits(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Observable 6.1: Content alone fits budget, but content + title exceeds it."""
    # Under a model profile with a document title template (such as embeddinggemma-300m)
    profile_with_title = dataclasses.replace(
        runtime_tokenizer.profile,
        document_template="title: {title} | text: {content}",
    )
    titled_tokenizer = ModelTokenizer(
        profile=profile_with_title,
        tokenizer=runtime_tokenizer.tokenizer,
        tokenizer_id="test-titled-tokenizer",
    )
    limit = titled_tokenizer.max_input_tokens  # 512
    chunker = Chunker(titled_tokenizer, budget=limit, prose_overlap_tokens=32)

    title = "A Comprehensive and Extremely Detailed Analysis of Document Ingest Architecture"

    # Pre-seed content close to limit - 10 (each "word " is ~1 token)
    content = "word " * (limit - 25)
    while chunker.measure(content, title=None) < limit - 10:
        content += " word"

    content_alone_tokens = chunker.measure(content, title=None)
    assert content_alone_tokens <= limit, (
        f"content alone should fit budget {limit}, got {content_alone_tokens}"
    )

    with_title_tokens = chunker.measure(content, title=title)
    assert with_title_tokens > limit, (
        f"content + title should exceed budget {limit}, got {with_title_tokens}"
    )
    assert chunker.exceeds_budget(content, title=title) is True
    assert chunker.exceeds_budget(content, title=None) is False


def test_assemble_record_produces_valid_chunk_record(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Requirement 7.1: Record assembly with chunk_id, tokens, locator, and provenance."""
    limit = runtime_tokenizer.max_input_tokens
    chunker = Chunker(runtime_tokenizer, budget=limit, prose_overlap_tokens=32)

    record = chunker.assemble_record(
        source=SOURCE,
        title="My Title",
        kind=ChunkKind.PROSE,
        text="Sample prose text content",
        locator=MarkdownLocator(line_range=(1, 5), ordinal=0),
        ordinal=0,
        params=PARAMS,
    )

    assert isinstance(record, ChunkRecord)
    assert record.kind is ChunkKind.PROSE
    assert record.text == "Sample prose text content"
    assert record.title == "My Title"
    assert record.source_path == SOURCE.path
    assert record.root_id == SOURCE.root.as_posix()
    assert record.author == "author1"
    assert record.ordinal == 0
    assert record.token_count > 0
    assert record.truncated is False
    assert record.provenance is None
    # chunk_id matches deterministic derivation
    expected_id = chunk_id(
        root_id=SOURCE.root.as_posix(),
        relative_path=SOURCE.relative_path,
        kind=ChunkKind.PROSE,
        locator=MarkdownLocator(line_range=(1, 5), ordinal=0),
        ordinal=0,
        params=PARAMS,
    )
    assert record.chunk_id == expected_id
