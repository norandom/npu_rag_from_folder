"""Unit tests for the ingest domain vocabulary (task 1.2).

``types.py`` is the leftmost layer of design.md's dependency direction
(``types, errors -> config -> ... -> pipeline``), so two things are asserted
here that no later test can assert for it: that it imports nothing from this
project at all, and that each Domain Model invariant is unconstructible when
negated.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from npu_rag.ingest.types import (
    BlockSegment,
    ChartRanges,
    ChunkKind,
    ChunkRecord,
    Extracted,
    FigureSegment,
    FormulaSegment,
    ImageFileLocator,
    ImageRef,
    Locator,
    MarkdownLocator,
    Omission,
    OmissionCategory,
    PageLocator,
    ProseSegment,
    Provenance,
    Resolved,
    RunReport,
    Segment,
    SheetLocator,
    SourceFile,
    TableSegment,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "ingest" / "types.py"
)

SOURCE_PATH = Path("archive/alice/notes.md")
PROVENANCE = Provenance(vision_model="mistralai/mistral-small-3.2-24b-instruct", prompt_version="1")
MARKDOWN_LOCATOR = MarkdownLocator(line_range=(1, 4), ordinal=0)
PAGE_LOCATOR = PageLocator(page=3)
SHEET_LOCATOR = SheetLocator(sheet="Q1", cell_range="A1:C10")
IMAGE_FILE_LOCATOR = ImageFileLocator()


def make_record(**overrides: object) -> ChunkRecord:
    values: dict[str, object] = {
        "chunk_id": "c1",
        "source_path": SOURCE_PATH,
        "root_id": "root-a",
        "author": "alice",
        "title": "Notes",
        "kind": ChunkKind.PROSE,
        "text": "hello",
        "locator": MARKDOWN_LOCATOR,
        "ordinal": 0,
        "token_count": 8,
        "truncated": False,
        "provenance": None,
    }
    values.update(overrides)
    return ChunkRecord(**values)  # type: ignore[arg-type]


def make_figure_segment() -> FigureSegment:
    return FigureSegment(
        text="a bar chart of quarterly revenue",
        locator=PAGE_LOCATOR,
        provenance=PROVENANCE,
    )


# --------------------------------------------------------------------------
# Declared vocabularies
# --------------------------------------------------------------------------


def test_chunk_kind_offers_exactly_the_five_kinds() -> None:
    """design.md, Domain Model: PROSE, TABLE, BLOCK, FORMULA, FIGURE."""
    assert {kind.value for kind in ChunkKind} == {
        "prose",
        "table",
        "block",
        "formula",
        "figure",
    }


def test_omission_category_is_exactly_the_set_named_in_the_design() -> None:
    """Every category the design raises or records, and no extras."""
    assert {category.value for category in OmissionCategory} == {
        "root_unavailable",
        "unsupported",
        "value_unavailable",
        "hidden_sheet",
        "vision_unavailable",
        "below_threshold",
        "vision_failed",
        "failed",
    }


@pytest.mark.parametrize(
    "member, text",
    [
        (ChunkKind.PROSE, "prose"),
        (ChunkKind.TABLE, "table"),
        (ChunkKind.BLOCK, "block"),
        (ChunkKind.FORMULA, "formula"),
        (ChunkKind.FIGURE, "figure"),
        (OmissionCategory.ROOT_UNAVAILABLE, "root_unavailable"),
        (OmissionCategory.UNSUPPORTED, "unsupported"),
        (OmissionCategory.VALUE_UNAVAILABLE, "value_unavailable"),
        (OmissionCategory.HIDDEN_SHEET, "hidden_sheet"),
        (OmissionCategory.VISION_UNAVAILABLE, "vision_unavailable"),
        (OmissionCategory.BELOW_THRESHOLD, "below_threshold"),
        (OmissionCategory.VISION_FAILED, "vision_failed"),
        (OmissionCategory.FAILED, "failed"),
    ],
)
def test_every_vocabulary_member_is_its_own_wire_text(member: str, text: str) -> None:
    assert member == text
    assert f"{member}" == text


# --------------------------------------------------------------------------
# Locator union
# --------------------------------------------------------------------------


def test_locator_union_is_exactly_the_four_kinds() -> None:
    """design.md, Domain Model: MarkdownLocator, PageLocator, SheetLocator,
    ImageFileLocator."""
    assert set(Locator.__args__) == {
        MarkdownLocator,
        PageLocator,
        SheetLocator,
        ImageFileLocator,
    }


def test_markdown_locator_carries_line_range_and_ordinal() -> None:
    """design.md, Domain Model / MarkdownExtractor: line_range and ordinal."""
    locator = MarkdownLocator(line_range=(10, 18), ordinal=2)
    assert locator.line_range == (10, 18)
    assert locator.ordinal == 2


def test_page_locator_carries_the_page() -> None:
    assert PageLocator(page=4).page == 4


def test_sheet_locator_carries_sheet_and_cell_range() -> None:
    locator = SheetLocator(sheet="Income", cell_range="B2:F20")
    assert (locator.sheet, locator.cell_range) == ("Income", "B2:F20")


def test_image_file_locator_is_a_value_with_no_coordinates() -> None:
    assert ImageFileLocator() == ImageFileLocator()


def test_locators_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PAGE_LOCATOR.page = 9  # type: ignore[misc]


# --------------------------------------------------------------------------
# SourceFile, segments, Extracted
# --------------------------------------------------------------------------


def test_source_file_carries_path_root_relative_path_and_author() -> None:
    """design.md, Discoverer summary: path, root, relative path, author."""
    source = SourceFile(
        path=Path("D:/archive/alice/notes.md"),
        root=Path("D:/archive"),
        relative_path=Path("alice/notes.md"),
        author="alice",
    )
    assert source.path == Path("D:/archive/alice/notes.md")
    assert source.root == Path("D:/archive")
    assert source.relative_path == Path("alice/notes.md")
    assert source.author == "alice"


def test_segment_union_is_the_extractor_vocabulary() -> None:
    """design.md, Extraction Service Interface: Prose, Table, Block, Formula,
    ImageRef. FigureSegment is the vision seam's replacement, not an extractor
    output."""
    assert set(Segment.__args__) == {
        ProseSegment,
        TableSegment,
        BlockSegment,
        FormulaSegment,
        ImageRef,
    }


def test_prose_segment_carries_text_locator_and_heading_path() -> None:
    """design.md, MarkdownExtractor: headings maintain a heading_path."""
    segment = ProseSegment(
        text="body",
        locator=MARKDOWN_LOCATOR,
        heading_path=("Intro", "Setup"),
    )
    assert segment.text == "body"
    assert segment.locator == MARKDOWN_LOCATOR
    assert segment.heading_path == ("Intro", "Setup")


def test_table_segment_carries_text_and_locator() -> None:
    segment = TableSegment(text="| a | b |", locator=MARKDOWN_LOCATOR)
    assert segment.text == "| a | b |"
    assert segment.locator == MARKDOWN_LOCATOR


def test_block_segment_carries_label_header_rows_and_locator() -> None:
    """design.md, Extraction Implementation Notes: label, header_row, rows with
    the row label as column 0."""
    segment = BlockSegment(
        label="Revenue",
        header_row=("item", "FY24", "FY25"),
        rows=(("North", "1", "2"),),
        locator=SHEET_LOCATOR,
    )
    assert segment.label == "Revenue"
    assert segment.header_row == ("item", "FY24", "FY25")
    assert segment.rows == (("North", "1", "2"),)
    assert segment.locator == SHEET_LOCATOR


def test_formula_segment_carries_formula_text_and_locator() -> None:
    segment = FormulaSegment(text="=SUM(B2:B4)", locator=SHEET_LOCATOR)
    assert segment.text == "=SUM(B2:B4)"
    assert segment.locator == SHEET_LOCATOR


def test_image_ref_carries_bytes_mime_dimensions_locator_and_chart_ranges() -> None:
    """design.md, Extraction invariants: raw bytes, MIME, width, height so
    resolution needs nothing from the extractor afterwards; ChartRanges where
    applicable (4.7)."""
    ranges = ChartRanges(
        value_ranges=("Sheet1!$B$2:$B$5",),
        category_ranges=("Sheet1!$A$2:$A$5",),
    )
    ref = ImageRef(
        data=b"\x89PNG",
        mime="image/png",
        width=800,
        height=600,
        locator=PAGE_LOCATOR,
        chart_ranges=ranges,
    )
    assert ref.data == b"\x89PNG"
    assert ref.mime == "image/png"
    assert (ref.width, ref.height) == (800, 600)
    assert ref.locator == PAGE_LOCATOR
    assert ref.chart_ranges == ranges


def test_image_ref_chart_ranges_are_optional() -> None:
    ref = ImageRef(
        data=b"\xff\xd8",
        mime="image/jpeg",
        width=64,
        height=64,
        locator=IMAGE_FILE_LOCATOR,
    )
    assert ref.chart_ranges is None


def test_figure_segment_carries_provenance() -> None:
    """design.md, Vision seam: FigureSegment.provenance names the model id and
    prompt version (5.8)."""
    figure = make_figure_segment()
    assert figure.provenance == PROVENANCE
    assert figure.locator == PAGE_LOCATOR


def test_extracted_carries_title_segments_and_omissions() -> None:
    extracted = Extracted(
        title="Notes",
        segments=(ProseSegment(text="hi", locator=MARKDOWN_LOCATOR),),
        omissions=(),
    )
    assert extracted.title == "Notes"
    assert len(extracted.segments) == 1
    assert extracted.omissions == ()


# --------------------------------------------------------------------------
# Domain Model invariants — each planted as its negation
# --------------------------------------------------------------------------


def test_chunk_record_refuses_a_missing_locator() -> None:
    """design.md, Domain Model: a ChunkRecord has exactly one Locator."""
    with pytest.raises((TypeError, ValueError), match="[Ll]ocator"):
        make_record(locator=None)


def test_chunk_record_refuses_a_non_locator() -> None:
    with pytest.raises((TypeError, ValueError), match="[Ll]ocator"):
        make_record(locator="archive/alice/notes.md")


def test_chunk_record_refuses_multiple_locators() -> None:
    with pytest.raises((TypeError, ValueError), match="[Ll]ocator"):
        make_record(locator=(PAGE_LOCATOR, SHEET_LOCATOR))


@pytest.mark.parametrize(
    "locator",
    [MARKDOWN_LOCATOR, PAGE_LOCATOR, SHEET_LOCATOR, IMAGE_FILE_LOCATOR],
    ids=["markdown", "page", "sheet", "image_file"],
)
def test_chunk_record_accepts_each_locator_kind(locator: Locator) -> None:
    record = make_record(locator=locator)
    assert record.locator == locator


def test_a_figure_record_without_provenance_is_refused() -> None:
    """design.md, Domain Model: a FIGURE record has a non-None Provenance.
    Requirement 5.8: vision-derived chunks are marked with the model id."""
    with pytest.raises(ValueError, match="provenance"):
        make_record(kind=ChunkKind.FIGURE, provenance=None)


@pytest.mark.parametrize(
    "kind",
    [ChunkKind.PROSE, ChunkKind.TABLE, ChunkKind.BLOCK, ChunkKind.FORMULA],
)
def test_a_non_figure_record_with_provenance_is_refused(kind: ChunkKind) -> None:
    """design.md, Domain Model: no kind other than FIGURE carries Provenance."""
    with pytest.raises(ValueError, match="provenance"):
        make_record(kind=kind, provenance=PROVENANCE)


def test_a_figure_record_with_provenance_is_accepted() -> None:
    record = make_record(
        kind=ChunkKind.FIGURE,
        text="a described chart",
        locator=PAGE_LOCATOR,
        provenance=PROVENANCE,
    )
    assert record.kind is ChunkKind.FIGURE
    assert record.provenance == PROVENANCE


def test_vision_unavailable_omission_without_missing_capability_is_refused() -> None:
    """design.md, Domain Model / requirement 9.4: missing_capability is set
    iff category is VISION_UNAVAILABLE."""
    with pytest.raises(ValueError, match="missing_capability"):
        Omission(
            category=OmissionCategory.VISION_UNAVAILABLE,
            path=Path("charts/q1.png"),
            reason="no credential",
        )


def test_vision_unavailable_omission_with_a_blank_capability_is_refused() -> None:
    with pytest.raises(ValueError, match="missing_capability"):
        Omission(
            category=OmissionCategory.VISION_UNAVAILABLE,
            path=Path("charts/q1.png"),
            reason="no credential",
            missing_capability="  ",
        )


@pytest.mark.parametrize(
    "category",
    [
        OmissionCategory.ROOT_UNAVAILABLE,
        OmissionCategory.UNSUPPORTED,
        OmissionCategory.VALUE_UNAVAILABLE,
        OmissionCategory.HIDDEN_SHEET,
        OmissionCategory.BELOW_THRESHOLD,
        OmissionCategory.VISION_FAILED,
        OmissionCategory.FAILED,
    ],
)
def test_non_vision_unavailable_omission_with_a_capability_is_refused(
    category: OmissionCategory,
) -> None:
    with pytest.raises(ValueError, match="missing_capability"):
        Omission(
            category=category,
            path=Path("file.bin"),
            reason="skipped",
            missing_capability="vision",
        )


def test_vision_unavailable_omission_names_the_missing_capability() -> None:
    omission = Omission(
        category=OmissionCategory.VISION_UNAVAILABLE,
        path=Path("charts/q1.png"),
        reason="no credential configured",
        missing_capability="vision",
    )
    assert omission.missing_capability == "vision"


def test_resolved_refuses_neither_figure_nor_omission() -> None:
    """design.md, Domain Model: Resolved has exactly one of figure/omission."""
    with pytest.raises(ValueError, match="figure|omission"):
        Resolved(figure=None, omission=None)


def test_resolved_refuses_both_figure_and_omission() -> None:
    with pytest.raises(ValueError, match="figure|omission"):
        Resolved(
            figure=make_figure_segment(),
            omission=Omission(
                category=OmissionCategory.BELOW_THRESHOLD,
                path=Path("icon.png"),
                reason="shorter side below threshold",
            ),
        )


def test_resolved_accepts_a_figure_alone() -> None:
    resolved = Resolved(figure=make_figure_segment(), omission=None)
    assert resolved.figure is not None
    assert resolved.omission is None


def test_resolved_accepts_an_omission_alone() -> None:
    omission = Omission(
        category=OmissionCategory.BELOW_THRESHOLD,
        path=Path("icon.png"),
        reason="shorter side below threshold",
    )
    resolved = Resolved(figure=None, omission=omission)
    assert resolved.figure is None
    assert resolved.omission is omission


# --------------------------------------------------------------------------
# ChunkRecord logical model and JSON round-trip
# --------------------------------------------------------------------------


def test_chunk_record_carries_the_logical_data_model_fields() -> None:
    """design.md, Logical Data Model; requirements 7.1 and 7.3."""
    record = make_record()
    assert record.chunk_id == "c1"
    assert record.source_path == SOURCE_PATH
    assert record.root_id == "root-a"
    assert record.author == "alice"
    assert record.title == "Notes"
    assert record.kind is ChunkKind.PROSE
    assert record.text == "hello"
    assert record.locator == MARKDOWN_LOCATOR
    assert record.ordinal == 0
    assert record.token_count == 8
    assert record.truncated is False
    assert record.provenance is None


def test_chunk_record_is_frozen() -> None:
    record = make_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.text = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    "locator",
    [MARKDOWN_LOCATOR, PAGE_LOCATOR, SHEET_LOCATOR, IMAGE_FILE_LOCATOR],
    ids=["markdown", "page", "sheet", "image_file"],
)
def test_chunk_record_survives_json_round_trip(locator: Locator) -> None:
    """Physical Data Model: the full ChunkRecord is persisted as JSON in
    chunk_registry.record_json. Paths, enums, and the locator union must
    round-trip."""
    record = make_record(locator=locator, title=None)
    restored = ChunkRecord.from_json(record.to_json())
    assert restored == record


def test_figure_chunk_survives_json_round_trip() -> None:
    record = make_record(
        kind=ChunkKind.FIGURE,
        text="described",
        locator=PAGE_LOCATOR,
        provenance=PROVENANCE,
        truncated=True,
    )
    restored = ChunkRecord.from_json(record.to_json())
    assert restored == record
    assert restored.provenance == PROVENANCE
    assert restored.kind is ChunkKind.FIGURE


def test_json_round_trip_preserves_a_non_ascii_path() -> None:
    record = make_record(source_path=Path("archive/åse/rapport.md"))
    restored = ChunkRecord.from_json(record.to_json())
    assert restored.source_path == record.source_path
    assert restored == record


def test_json_payload_uses_enum_values_and_a_locator_discriminator() -> None:
    """The persisted form is JSON, so the union needs a discriminator and the
    kind must be its wire text rather than a repr that cannot be parsed back."""
    payload = json.loads(make_record(locator=PAGE_LOCATOR).to_json())
    assert payload["kind"] == "prose"
    assert isinstance(payload["source_path"], str)
    assert payload["locator"]["type"] == "page"
    assert payload["locator"]["page"] == 3
    assert payload["provenance"] is None


# --------------------------------------------------------------------------
# RunReport data shape (rendering is task 7.1)
# --------------------------------------------------------------------------


def test_run_report_carries_counts_omissions_removed_ids_vision_and_records() -> None:
    """design.md, Monitoring / Batch contract: counts, omissions, removed
    chunk ids, no_work_required, vision request vs cache counts, records."""
    record = make_record()
    omission = Omission(
        category=OmissionCategory.UNSUPPORTED,
        path=Path("file.bin"),
        reason="unknown type",
    )
    report = RunReport(
        files_processed=1,
        files_unchanged=2,
        files_skipped=1,
        files_failed=0,
        omissions=(omission,),
        removed_chunk_ids=("old-chunk",),
        no_work_required=False,
        vision_requests_issued=3,
        vision_cache_hits=4,
        records=(record,),
    )
    assert report.files_processed == 1
    assert report.files_unchanged == 2
    assert report.files_skipped == 1
    assert report.files_failed == 0
    assert report.omissions == (omission,)
    assert report.removed_chunk_ids == ("old-chunk",)
    assert report.no_work_required is False
    assert report.vision_requests_issued == 3
    assert report.vision_cache_hits == 4
    assert report.records == (record,)


def test_run_report_is_frozen() -> None:
    report = RunReport(
        files_processed=0,
        files_unchanged=0,
        files_skipped=0,
        files_failed=0,
        omissions=(),
        removed_chunk_ids=(),
        no_work_required=True,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.no_work_required = False  # type: ignore[misc]


# --------------------------------------------------------------------------
# Boundary guard
# --------------------------------------------------------------------------


def test_types_imports_nothing_from_this_project() -> None:
    """design.md, Architecture: types is the leftmost layer. Anything it
    imported from ``npu_rag`` would be a cycle by construction."""
    imported: list[str] = []
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            assert node.level == 0, "a relative import is still a project import"

    assert [name for name in imported if name.startswith("npu_rag")] == []
