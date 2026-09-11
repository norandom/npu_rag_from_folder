"""Tests for LanceDB schema and ChunkRecord conversions."""

from __future__ import annotations

from pathlib import Path

import pytest

from npu_rag.index.errors import SchemaError
from npu_rag.index.schema import (
    VECTOR_DIMENSION,
    ChunkSchema,
    chunk_record_to_schema,
    schema_to_chunk_record,
)
from npu_rag.ingest.types import (
    ChunkKind,
    ChunkRecord,
    ImageFileLocator,
    MarkdownLocator,
    PageLocator,
    Provenance,
    SheetLocator,
)


@pytest.fixture
def dummy_vector() -> list[float]:
    return [0.01 * (i % 50) for i in range(VECTOR_DIMENSION)]


def test_chunk_schema_creation(dummy_vector: list[float]) -> None:
    schema = ChunkSchema(
        chunk_id="c_test",
        file_path="sample.md",
        file_hash="abc123hash",
        chunk_index=0,
        chunk_kind="prose",
        title="Title",
        headings=["Heading 1"],
        locator_type="markdown",
        locator_json='{"type": "markdown", "line_range": [1, 5], "ordinal": 0}',
        text="Hello world text",
        vector=dummy_vector,
        token_count=12,
        created_at=1000.0,
    )
    assert schema.chunk_id == "c_test"
    assert len(schema.vector) == VECTOR_DIMENSION
    assert schema.headings == ["Heading 1"]


def test_vector_dimension_mismatch_raises(dummy_vector: list[float]) -> None:
    record = ChunkRecord(
        chunk_id="c1",
        source_path=Path("docs/doc.md"),
        root_id="root",
        author="Author",
        title="Doc",
        kind=ChunkKind.PROSE,
        text="Sample",
        locator=MarkdownLocator(line_range=(1, 5), ordinal=0),
        ordinal=0,
        token_count=5,
        truncated=False,
        provenance=None,
    )
    with pytest.raises(SchemaError, match="dimension mismatch"):
        chunk_record_to_schema(record, dummy_vector[:100])


def test_round_trip_markdown_locator(dummy_vector: list[float]) -> None:
    record = ChunkRecord(
        chunk_id="c_md",
        source_path=Path("docs/doc.md"),
        root_id="root_1",
        author="Alice",
        title="My Document",
        kind=ChunkKind.PROSE,
        text="Paragraph of text",
        locator=MarkdownLocator(line_range=(10, 20), ordinal=1),
        ordinal=1,
        token_count=25,
        truncated=False,
        provenance=None,
    )
    schema = chunk_record_to_schema(
        record,
        dummy_vector,
        file_hash="hash_md_123",
        headings=["Overview", "Details"],
        created_at=123456.78,
    )
    assert schema.locator_type == "markdown"
    assert schema.file_hash == "hash_md_123"
    assert schema.headings == ["Overview", "Details"]

    reconstructed = schema_to_chunk_record(schema)
    assert reconstructed.chunk_id == record.chunk_id
    assert reconstructed.source_path == record.source_path
    assert reconstructed.root_id == record.root_id
    assert reconstructed.author == record.author
    assert reconstructed.title == record.title
    assert reconstructed.kind == record.kind
    assert reconstructed.text == record.text
    assert reconstructed.locator == record.locator
    assert reconstructed.ordinal == record.ordinal
    assert reconstructed.token_count == record.token_count
    assert reconstructed.truncated == record.truncated
    assert reconstructed.provenance == record.provenance


def test_round_trip_page_locator(dummy_vector: list[float]) -> None:
    record = ChunkRecord(
        chunk_id="c_pdf",
        source_path=Path("docs/manual.pdf"),
        root_id="root_2",
        author="Bob",
        title="User Manual",
        kind=ChunkKind.PROSE,
        text="Page two instructions",
        locator=PageLocator(page=2),
        ordinal=0,
        token_count=40,
        truncated=True,
        provenance=None,
    )
    schema = chunk_record_to_schema(record, dummy_vector)
    assert schema.locator_type == "page"

    reconstructed = schema_to_chunk_record(schema)
    assert reconstructed.locator == PageLocator(page=2)
    assert reconstructed.truncated is True


def test_round_trip_sheet_locator(dummy_vector: list[float]) -> None:
    record = ChunkRecord(
        chunk_id="c_excel",
        source_path=Path("models/finances.xlsx"),
        root_id="root_3",
        author="Charlie",
        title="Financial Model",
        kind=ChunkKind.TABLE,
        text="Revenue: 100\nCosts: 40",
        locator=SheetLocator(sheet="Income Statement", cell_range="A1:D10"),
        ordinal=2,
        token_count=35,
        truncated=False,
        provenance=None,
    )
    schema = chunk_record_to_schema(record, dummy_vector)
    assert schema.locator_type == "sheet"

    reconstructed = schema_to_chunk_record(schema)
    assert reconstructed.locator == SheetLocator(sheet="Income Statement", cell_range="A1:D10")
    assert reconstructed.kind == ChunkKind.TABLE


def test_round_trip_figure_with_provenance(dummy_vector: list[float]) -> None:
    record = ChunkRecord(
        chunk_id="c_fig",
        source_path=Path("images/chart.png"),
        root_id="root_4",
        author="",
        title=None,
        kind=ChunkKind.FIGURE,
        text="Bar chart showing quarterly growth from 10% to 25%",
        locator=ImageFileLocator(),
        ordinal=0,
        token_count=18,
        truncated=False,
        provenance=Provenance(vision_model="infiniflow-ocr", prompt_version="v1"),
    )
    schema = chunk_record_to_schema(record, dummy_vector)
    assert schema.locator_type == "image_file"
    assert schema.provenance_json is not None

    reconstructed = schema_to_chunk_record(schema)
    assert reconstructed.kind == ChunkKind.FIGURE
    assert reconstructed.provenance is not None
    assert reconstructed.provenance.vision_model == "infiniflow-ocr"
    assert reconstructed.provenance.prompt_version == "v1"
    assert reconstructed.locator == ImageFileLocator()


def test_schema_to_chunk_record_from_dict(dummy_vector: list[float]) -> None:
    record = ChunkRecord(
        chunk_id="c_dict",
        source_path=Path("docs/test.txt"),
        root_id="r",
        author="Me",
        title="Test",
        kind=ChunkKind.PROSE,
        text="Dict conversion text",
        locator=PageLocator(page=1),
        ordinal=0,
        token_count=10,
        truncated=False,
        provenance=None,
    )
    schema = chunk_record_to_schema(record, dummy_vector)
    raw_dict = schema.model_dump()
    reconstructed = schema_to_chunk_record(raw_dict)
    assert reconstructed.chunk_id == "c_dict"
    assert reconstructed.locator == PageLocator(page=1)
