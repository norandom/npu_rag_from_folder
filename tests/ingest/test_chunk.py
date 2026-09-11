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


# --------------------------------------------------------------------------
# Task 6.2: Prose accumulation, heading-first splitting, and overlap
# --------------------------------------------------------------------------


def test_splits_at_heading_before_budget_is_reached(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Requirement 6.6 / Observable 6.2: Fixture with two headings splits before budget."""
    limit = runtime_tokenizer.max_input_tokens
    chunker = Chunker(runtime_tokenizer, budget=limit, prose_overlap_tokens=32)

    p1 = ProseSegment(
        text="This is the introduction section under the first heading.",
        locator=MarkdownLocator(line_range=(1, 5), ordinal=0),
        heading_path=("Introduction",),
    )
    p2 = ProseSegment(
        text="This is the architecture section under the second heading.",
        locator=MarkdownLocator(line_range=(7, 12), ordinal=0),
        heading_path=("Architecture",),
    )

    extracted = Extracted(title="System Design", segments=(p1, p2), omissions=())
    records = chunker.chunk(SOURCE, extracted, params=PARAMS)

    assert len(records) == 2
    assert records[0].kind is ChunkKind.PROSE
    assert records[1].kind is ChunkKind.PROSE
    assert records[0].text == p1.text
    assert records[1].text == p2.text
    assert records[0].ordinal == 0
    assert records[1].ordinal == 1
    assert records[0].locator == p1.locator
    assert records[1].locator == p2.locator
    assert records[0].title == "System Design"
    assert records[1].title == "System Design"


def test_consecutive_prose_chunks_share_exactly_the_configured_overlap(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Requirement 6.4 / Observable 6.2: Consecutive prose chunks share configured overlap."""
    limit = runtime_tokenizer.max_input_tokens  # 512
    overlap = 32
    chunker = Chunker(runtime_tokenizer, budget=limit, prose_overlap_tokens=overlap)

    # Paragraphs of ~220 tokens each; 3 paragraphs will exceed 512 limit and split
    words_p1 = "First paragraph discussing neural network architectures and local inference in detail. " * 20
    words_p2 = "Second paragraph describing vector indexing algorithms and offline storage layouts. " * 20
    words_p3 = "Third paragraph detailing retrieval scoring and rank reciprocal fusion techniques. " * 20

    p1 = ProseSegment(
        text=words_p1.strip(),
        locator=MarkdownLocator(line_range=(1, 10), ordinal=0),
        heading_path=("Details",),
    )
    p2 = ProseSegment(
        text=words_p2.strip(),
        locator=MarkdownLocator(line_range=(12, 22), ordinal=0),
        heading_path=("Details",),
    )
    p3 = ProseSegment(
        text=words_p3.strip(),
        locator=MarkdownLocator(line_range=(24, 34), ordinal=0),
        heading_path=("Details",),
    )

    extracted = Extracted(title="Tech Overview", segments=(p1, p2, p3), omissions=())
    records = chunker.chunk(SOURCE, extracted, params=PARAMS)

    assert len(records) >= 2
    # Verify consecutive prose chunks share exactly `overlap` tokens
    for i in range(len(records) - 1):
        r1 = records[i]
        r2 = records[i + 1]
        assert r1.kind is ChunkKind.PROSE
        assert r2.kind is ChunkKind.PROSE

        ids1 = runtime_tokenizer.tokenizer.encode(r1.text, add_special_tokens=False)
        ids2 = runtime_tokenizer.tokenizer.encode(r2.text, add_special_tokens=False)

        assert len(ids1) >= overlap
        assert len(ids2) >= overlap
        assert ids1[-overlap:] == ids2[:overlap], (
            f"Expected chunks {i} and {i+1} to share exactly {overlap} tokens"
        )


def test_table_chunk_shares_nothing_with_neighbours(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Requirement 6.5 / Observable 6.2: Table chunk shares nothing with its neighbours."""
    limit = runtime_tokenizer.max_input_tokens
    overlap = 32
    chunker = Chunker(runtime_tokenizer, budget=limit, prose_overlap_tokens=overlap)

    p1 = ProseSegment(
        text="Prose section before the table discussing financial results.",
        locator=MarkdownLocator(line_range=(1, 5), ordinal=0),
    )
    t1 = TableSegment(
        text="Quarter | Revenue | Margin\nQ1 | $10M | 45%\nQ2 | $12M | 48%",
        locator=MarkdownLocator(line_range=(7, 11), ordinal=0),
    )
    p2 = ProseSegment(
        text="Prose section after the table discussing future projections and outlook.",
        locator=MarkdownLocator(line_range=(13, 17), ordinal=0),
    )

    extracted = Extracted(title="Financials", segments=(p1, t1, p2), omissions=())
    records = chunker.chunk(SOURCE, extracted, params=PARAMS)

    assert len(records) == 3
    assert records[0].kind is ChunkKind.PROSE
    assert records[1].kind is ChunkKind.TABLE
    assert records[2].kind is ChunkKind.PROSE

    assert records[1].text == t1.text
    # Table shares nothing with prose before it
    ids_p1 = runtime_tokenizer.tokenizer.encode(records[0].text, add_special_tokens=False)
    ids_t = runtime_tokenizer.tokenizer.encode(records[1].text, add_special_tokens=False)
    ids_p2 = runtime_tokenizer.tokenizer.encode(records[2].text, add_special_tokens=False)

    # Longest common suffix-prefix between p1 and table is 0
    assert not any(ids_p1[-k:] == ids_t[:k] for k in range(1, min(len(ids_p1), len(ids_t)) + 1))
    # Longest common suffix-prefix between table and p2 is 0
    assert not any(ids_t[-k:] == ids_p2[:k] for k in range(1, min(len(ids_t), len(ids_p2)) + 1))
    # p2 did not receive overlap from p1 across the table
    assert not any(ids_p1[-k:] == ids_p2[:k] for k in range(1, min(len(ids_p1), len(ids_p2)) + 1))


def test_figure_segment_passes_through_with_provenance_and_no_overlap(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Requirement 5.8 / 6.5: Figure passes through with provenance and no overlap."""
    limit = runtime_tokenizer.max_input_tokens
    chunker = Chunker(runtime_tokenizer, budget=limit, prose_overlap_tokens=32)

    prov = Provenance(vision_model="det+rec", prompt_version="pipeline_v1")
    fig = FigureSegment(
        text="Bar chart showing quarterly growth from 2024 to 2025.",
        locator=MarkdownLocator(line_range=(10, 15), ordinal=0),
        provenance=prov,
    )

    extracted = Extracted(title="Report", segments=(fig,), omissions=())
    records = chunker.chunk(SOURCE, extracted, params=PARAMS)

    assert len(records) == 1
    assert records[0].kind is ChunkKind.FIGURE
    assert records[0].provenance == prov
    assert records[0].text == fig.text
    assert records[0].locator == fig.locator


def test_no_emitted_chunk_exceeds_tokenizer_limit(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Requirement 6.3: No emitted chunk exceeds the token budget."""
    limit = runtime_tokenizer.max_input_tokens
    chunker = Chunker(runtime_tokenizer, budget=limit, prose_overlap_tokens=32)

    title = "A Substantial Title For The Document"
    segments = [
        ProseSegment(
            text=f"Paragraph {i} with substantial contents explaining detailed mechanisms. " * 15,
            locator=MarkdownLocator(line_range=(i * 10, i * 10 + 8), ordinal=0),
            heading_path=(f"Section_{i // 2}",),
        )
        for i in range(6)
    ]
    extracted = Extracted(title=title, segments=tuple(segments), omissions=())
    records = chunker.chunk(SOURCE, extracted, params=PARAMS)

    assert len(records) > 0
    for record in records:
        assert chunker.measure(record.text, record.title) <= limit
        assert chunker.exceeds_budget(record.text, record.title) is False

