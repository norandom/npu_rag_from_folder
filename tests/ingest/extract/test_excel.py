"""Unit tests for the Excel extractor (tasks 4.4, 4.5, and 4.6).

Requirement 4.1: a run of entirely empty rows is the block boundary; a row is
empty only if every cell is empty and no valued merged range covers it.
Requirement 4.2: each block carries its label, period header row, and row
labels. Requirement 4.4: a formula cell's cached value is emitted inside the
block and its formula text as a FormulaSegment. Requirement 4.5: a formula
cell with no cached value is an unavailable-value omission naming the cell;
a value is never substituted. Requirement 4.6: each anchored image and each
chart is an ImageRef. Requirement 4.7: a chart's series source ranges are
read from the object and attached to its reference, never obtained by
image-to-text. Requirement 4.8: a hidden sheet is skipped and recorded by
name, never raised. Requirement 7.3: every block carries a SheetLocator.
The committed workbook is built by generate.py and is read only.
"""

from __future__ import annotations

import ast
import importlib.util
import zipfile
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from typing import Protocol, cast

import pytest
from openpyxl import load_workbook  # type: ignore[import-untyped]
from openpyxl.cell.cell import MergedCell  # type: ignore[import-untyped]

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.extract.base import Extractor
from npu_rag.ingest.types import (
    BlockSegment,
    ChartRanges,
    Extracted,
    FormulaSegment,
    ImageRef,
    Omission,
    OmissionCategory,
    SheetLocator,
    SourceFile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "src" / "npu_rag" / "ingest"
EXCEL_PATH = PACKAGE_ROOT / "extract" / "excel.py"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "excel"
FIXTURE_PATH = FIXTURE_DIR / "models.xlsx"
GENERATE_PATH = FIXTURE_DIR / "generate.py"

LONG_PREFIX = "\\\\?\\"
TRADITIONAL_LIMIT = 260
BUDGET = 256
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

FORBIDDEN_IMPORTS = (
    "npu_rag.ingest.vision",
    "npu_rag.ingest.state",
    "httpx",
)


class _ExcelFixtureGenerate(Protocol):
    INCOME_SHEET: str
    HIDDEN_SHEET: str
    MERGED_LABEL: str
    MERGED_LABEL_RANGE: str
    FIRST_BLOCK_LABEL: str
    FIRST_BLOCK_RANGE: str
    FIRST_BLOCK_HEADER: tuple[str, ...]
    FIRST_BLOCK_ROWS: tuple[tuple[str, ...], ...]
    SECOND_BLOCK_LABEL: str
    SECOND_BLOCK_RANGE: str
    SECOND_BLOCK_HEADER: tuple[str, ...]
    SECOND_BLOCK_ROWS: tuple[tuple[str, ...], ...]
    CACHED_FORMULA_CELL: str
    CACHED_FORMULA: str
    CACHED_VALUE: int
    UNCACHED_FORMULA_CELL: str
    UNCACHED_FORMULA: str
    CHART_ANCHOR: str
    IMAGE_ANCHOR: str

    def committed_fixture_bytes(self) -> bytes: ...


def load_generate() -> _ExcelFixtureGenerate:
    spec = importlib.util.spec_from_file_location(
        "ingest_excel_fixture_generate", GENERATE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_ExcelFixtureGenerate, module)


GENERATE = load_generate()

# Requirement 4.4/4.5: blocks carry cached values, never formula text and
# never a guessed number for an uncached formula cell (C6, D6, B12).
EXTRACTED_FIRST_BLOCK_ROWS = (
    ("Product", "100", "110", "121"),
    ("Services", "50", "55", "61"),
    ("Total", str(GENERATE.CACHED_VALUE), "", ""),
)
EXTRACTED_SECOND_BLOCK_ROWS = (
    ("COGS", "30", "32", "34"),
    ("OpEx", "20", "21", "22"),
    ("Uncached", "", "", ""),
)
SUBSTITUTED_C6 = "165"  # C4+C5 = 110+55, must not appear
SUBSTITUTED_D6 = "182"  # D4+D5 = 121+61, must not appear
SUBSTITUTED_B12 = "50"  # B10+B11 = 30+20, must not appear in the uncached cell


def as_windows_long_path(path: Path) -> Path:
    """Discoverer always prefixes ``SourceFile.path`` with ``\\\\?\\`` on Windows."""
    text = str(path if path.is_absolute() else path.resolve())
    if text.startswith("\\\\?\\UNC\\") or text.startswith("\\\\?\\"):
        return Path(text)
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path(LONG_PREFIX + text)


def _without_long_prefix(path: Path | str) -> str:
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    if text.startswith("\\\\?\\"):
        return text[4:]
    return text


def traditional_length(path: Path) -> int:
    return len(_without_long_prefix(path))


def make_config(**overrides: object) -> IngestConfig:
    values: dict[str, object] = {
        "roots": (Path("inputs"),),
        "token_budget": BUDGET,
    }
    values.update(overrides)
    return IngestConfig(**values)  # type: ignore[arg-type]


def plant(root: Path, relative: str, data: bytes) -> SourceFile:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    relative_path = Path(relative)
    author = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
    return SourceFile(
        path=as_windows_long_path(path),
        root=as_windows_long_path(root),
        relative_path=relative_path,
        author=author,
    )


def source_for_fixture() -> SourceFile:
    path = FIXTURE_PATH.resolve()
    return SourceFile(
        path=as_windows_long_path(path),
        root=as_windows_long_path(path.parent),
        relative_path=Path("alice") / "models.xlsx",
        author="alice",
    )


def extract(source: SourceFile, **overrides: object) -> Extracted:
    from npu_rag.ingest.extract.excel import ExcelExtractor

    extractor: Extractor = ExcelExtractor()
    return extractor.extract(source, make_config(**overrides))


def imported_names(path: Path) -> list[str]:
    containing = "npu_rag.ingest.extract"
    parts = containing.split(".")
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                root = node.module or ""
            else:
                base = ".".join(parts[: len(parts) - node.level + 1])
                root = f"{base}.{node.module}" if node.module else base
            names.append(root)
            names.extend(f"{root}.{alias.name}" for alias in node.names)
    return names


def blocks_of(result: Extracted) -> list[BlockSegment]:
    return [segment for segment in result.segments if isinstance(segment, BlockSegment)]


def formulas_of(result: Extracted) -> list[FormulaSegment]:
    return [
        segment for segment in result.segments if isinstance(segment, FormulaSegment)
    ]


def formula_by_cell(result: Extracted, cell: str) -> FormulaSegment:
    matches = [
        segment
        for segment in formulas_of(result)
        if isinstance(segment.locator, SheetLocator)
        and segment.locator.cell_range == cell
    ]
    assert len(matches) == 1, (cell, formulas_of(result))
    return matches[0]


def image_refs_of(result: Extracted) -> list[ImageRef]:
    return [segment for segment in result.segments if isinstance(segment, ImageRef)]


def fixture_media_png() -> bytes:
    names = zipfile.ZipFile(FIXTURE_PATH).namelist()
    media = [name for name in names if name.startswith("xl/media/")]
    assert len(media) == 1
    payload = zipfile.ZipFile(FIXTURE_PATH).read(media[0])
    assert payload.startswith(PNG_MAGIC)
    return payload


def functions_touching_private_drawing_attrs(path: Path) -> set[str]:
    """Function names in *path* that read ``_images`` / ``_charts``."""
    found: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            previous = self.function
            self.function = node.name
            self.generic_visit(node)
            self.function = previous

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr in {"_images", "_charts"}:
                found.add(self.function or "<module>")
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> None:
            if node.value in {"_images", "_charts"}:
                found.add(self.function or "<module>")
            self.generic_visit(node)

    Visitor().visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


def _naive_row_is_empty(
    row_index: int,
    values: Sequence[object],
    valued_merged_ranges: Sequence[tuple[int, int]],
) -> bool:
    """All-empty-cells rule: ignores merged coverage. Rows 2-3 of the fixture
    are all-None at the cell level, so this splits the three-row merged label.
    """
    del row_index, valued_merged_ranges
    return all(value is None or value == "" for value in values)


# --------------------------------------------------------------------------
# Fixture non-vacuity
# --------------------------------------------------------------------------


def test_the_generator_reproduces_the_committed_workbook_byte_for_byte() -> None:
    assert GENERATE.committed_fixture_bytes() == FIXTURE_PATH.read_bytes()


def test_the_committed_fixture_still_has_two_blocks_a_three_row_merge_formulas_hidden_sheet_chart_and_image() -> None:
    """Non-vacuity: the on-disk workbook still has every ingredient later tasks consume."""
    raw = FIXTURE_PATH.read_bytes()
    assert raw == GENERATE.committed_fixture_bytes()
    assert raw[:4] == b"PK\x03\x04"

    names = zipfile.ZipFile(BytesIO(raw)).namelist()
    assert any(name.startswith("xl/charts/") for name in names)
    media = [name for name in names if name.startswith("xl/media/")]
    assert len(media) == 1
    image_bytes = zipfile.ZipFile(BytesIO(raw)).read(media[0])
    assert image_bytes.startswith(PNG_MAGIC)

    workbook = load_workbook(BytesIO(raw), data_only=False)
    try:
        assert GENERATE.INCOME_SHEET in workbook.sheetnames
        assert GENERATE.HIDDEN_SHEET in workbook.sheetnames
        income = workbook[GENERATE.INCOME_SHEET]
        notes = workbook[GENERATE.HIDDEN_SHEET]
        assert income.sheet_state == "visible"
        assert notes.sheet_state == "hidden"
        merges = {str(item) for item in income.merged_cells.ranges}
        assert GENERATE.MERGED_LABEL_RANGE in merges
        assert income["A1"].value == GENERATE.MERGED_LABEL
        assert income["A2"].value is None
        assert income["A3"].value is None
        assert isinstance(income["A2"], MergedCell)
        assert isinstance(income["A3"], MergedCell)
        for row in income.iter_rows(min_row=2, max_row=3, max_col=4):
            assert all(cell.value is None for cell in row)
        assert income[GENERATE.CACHED_FORMULA_CELL].value == GENERATE.CACHED_FORMULA
        assert income[GENERATE.UNCACHED_FORMULA_CELL].value == GENERATE.UNCACHED_FORMULA
        assert len(income._charts) == 1
        assert len(income._images) == 1
    finally:
        workbook.close()

    values_only = load_workbook(BytesIO(raw), data_only=True)
    try:
        income = values_only[GENERATE.INCOME_SHEET]
        assert income[GENERATE.CACHED_FORMULA_CELL].value == GENERATE.CACHED_VALUE
        assert income[GENERATE.UNCACHED_FORMULA_CELL].value is None
        assert income[GENERATE.CACHED_FORMULA_CELL].value != income[
            GENERATE.UNCACHED_FORMULA_CELL
        ].value
    finally:
        values_only.close()


def test_a_naive_all_empty_cells_scan_of_the_fixture_splits_the_merged_label() -> None:
    """The three-row merge is stored only in the anchor; all-None rows 2-3
    would be a boundary if merged coverage were ignored.
    """
    workbook = load_workbook(FIXTURE_PATH, data_only=False)
    try:
        sheet = workbook[GENERATE.INCOME_SHEET]
        naive_blocks = _naive_row_runs(sheet)
    finally:
        workbook.close()
    assert naive_blocks != [(1, 6), (9, 12)]
    assert naive_blocks[0] == (1, 1)
    assert naive_blocks[0][1] < 3
    assert len(naive_blocks) == 3


def _naive_row_runs(sheet: object) -> list[tuple[int, int]]:
    max_row = int(getattr(sheet, "max_row"))
    max_col = int(getattr(sheet, "max_column"))
    iter_rows = getattr(sheet, "iter_rows")
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for row_index, row in enumerate(
        iter_rows(min_row=1, max_row=max_row, max_col=max_col), start=1
    ):
        empty = all(cell.value is None or cell.value == "" for cell in row)
        if empty:
            if start is not None:
                runs.append((start, row_index - 1))
                start = None
        elif start is None:
            start = row_index
    if start is not None:
        runs.append((start, max_row))
    return runs


# --------------------------------------------------------------------------
# Requirements 4.1, 4.2, 4.8, 7.3: the fixture observable
# --------------------------------------------------------------------------


def test_the_fixture_yields_exactly_two_blocks_with_the_expected_ranges() -> None:
    result = extract(source_for_fixture())
    blocks = blocks_of(result)
    assert len(blocks) == 2

    first, second = blocks
    assert first.label == GENERATE.FIRST_BLOCK_LABEL
    assert first.header_row == GENERATE.FIRST_BLOCK_HEADER
    assert first.rows == EXTRACTED_FIRST_BLOCK_ROWS
    assert first.locator == SheetLocator(
        sheet=GENERATE.INCOME_SHEET, cell_range=GENERATE.FIRST_BLOCK_RANGE
    )
    assert first.rows[0][0] == "Product"
    assert first.rows[1][0] == "Services"
    assert first.rows[2][0] == "Total"

    assert second.label == GENERATE.SECOND_BLOCK_LABEL
    assert second.header_row == GENERATE.SECOND_BLOCK_HEADER
    assert second.rows == EXTRACTED_SECOND_BLOCK_ROWS
    assert second.locator == SheetLocator(
        sheet=GENERATE.INCOME_SHEET, cell_range=GENERATE.SECOND_BLOCK_RANGE
    )
    assert second.rows[0][0] == "COGS"
    assert second.rows[1][0] == "OpEx"
    assert second.rows[2][0] == "Uncached"


def test_every_block_from_the_fixture_carries_a_sheet_locator() -> None:
    result = extract(source_for_fixture())
    blocks = blocks_of(result)
    assert blocks
    for block in blocks:
        assert isinstance(block.locator, SheetLocator)
        assert block.locator.sheet == GENERATE.INCOME_SHEET
        assert ":" in block.locator.cell_range


def test_the_hidden_sheet_is_recorded_as_a_named_omission_and_is_not_raised() -> None:
    source = source_for_fixture()
    result = extract(source)
    assert result.omissions
    hidden = [
        item
        for item in result.omissions
        if item.category is OmissionCategory.HIDDEN_SHEET
    ]
    assert len(hidden) == 1
    omission = hidden[0]
    assert isinstance(omission, Omission)
    assert omission.path == source.path
    assert GENERATE.HIDDEN_SHEET in omission.reason
    assert not any(
        isinstance(segment, BlockSegment)
        and isinstance(segment.locator, SheetLocator)
        and segment.locator.sheet == GENERATE.HIDDEN_SHEET
        for segment in result.segments
    )


def test_a_naive_all_empty_cells_rule_planted_in_place_splits_the_merged_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observable: planting the all-None rule in place of the merged-aware
    rule splits the three-row merged label and no longer yields two blocks.
    """
    from npu_rag.ingest.extract import excel as excel_module

    monkeypatch.setattr(excel_module, "_row_is_empty", _naive_row_is_empty)
    result = extract(source_for_fixture())
    blocks = blocks_of(result)
    assert len(blocks) != 2
    assert len(blocks) == 3
    assert blocks[0].locator == SheetLocator(
        sheet=GENERATE.INCOME_SHEET, cell_range="A1:D1"
    )
    assert blocks[0].locator.cell_range != GENERATE.FIRST_BLOCK_RANGE


def test_excel_extractor_opens_the_windows_long_path_form() -> None:
    source = source_for_fixture()
    assert str(source.path).startswith(LONG_PREFIX)
    result = extract(source)
    assert len(blocks_of(result)) == 2


def test_a_workbook_behind_a_path_longer_than_the_traditional_limit_is_extracted(
    tmp_path: Path,
) -> None:
    root = as_windows_long_path(tmp_path / "archive")
    root.mkdir(parents=True, exist_ok=True)
    directory = root
    name = "models.xlsx"
    while traditional_length(directory / name) <= TRADITIONAL_LIMIT:
        directory = directory / ("subdir" * 4)
        directory.mkdir(parents=True, exist_ok=True)
        if traditional_length(directory) > 400:
            pytest.fail(
                f"cannot construct a path longer than {TRADITIONAL_LIMIT} under {root}"
            )
    target = directory / name
    target.write_bytes(FIXTURE_PATH.read_bytes())
    assert traditional_length(target) > TRADITIONAL_LIMIT
    relative = Path(_without_long_prefix(target)).relative_to(
        Path(_without_long_prefix(root))
    )
    source = SourceFile(
        path=as_windows_long_path(target),
        root=root,
        relative_path=relative,
        author="archive",
    )
    result = extract(source)
    assert len(blocks_of(result)) == 2
    assert blocks_of(result)[0].locator == SheetLocator(
        sheet=GENERATE.INCOME_SHEET, cell_range=GENERATE.FIRST_BLOCK_RANGE
    )


def test_a_corrupt_workbook_raises_extraction_error_with_stage_and_path(
    tmp_path: Path,
) -> None:
    source = plant(tmp_path, "alice/broken.xlsx", b"this is not a workbook")
    with pytest.raises(ExtractionError) as caught:
        extract(source)
    error = caught.value
    assert error.stage == "extraction"
    assert error.path == source.path
    assert str(source.path).startswith(LONG_PREFIX)


def test_an_unreadable_path_raises_extraction_error_with_stage_and_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "alice" / "gone.xlsx"
    source = SourceFile(
        path=as_windows_long_path(missing),
        root=as_windows_long_path(tmp_path),
        relative_path=Path("alice") / "gone.xlsx",
        author="alice",
    )
    with pytest.raises(ExtractionError) as caught:
        extract(source)
    error = caught.value
    assert error.stage == "extraction"
    assert error.path == source.path


def test_excel_extractor_does_not_import_vision_state_or_httpx() -> None:
    names = imported_names(EXCEL_PATH)
    for forbidden in FORBIDDEN_IMPORTS:
        assert all(
            name != forbidden and not name.startswith(f"{forbidden}.")
            for name in names
        ), names


def test_row_is_empty_is_the_hook_the_naive_plant_replaces() -> None:
    """The merged-aware rule is a module-level function so the plant is in place."""
    from npu_rag.ingest.extract.excel import _row_is_empty

    hook: Callable[..., bool] = _row_is_empty
    assert hook(2, (None, None, None, None), ((1, 3),)) is False
    assert hook(7, (None, None, None, None), ((1, 3),)) is True
    assert hook(4, (None, "100", None, None), ()) is False


# --------------------------------------------------------------------------
# Requirements 4.4, 4.5: cached values, formula segments, unavailable values
# --------------------------------------------------------------------------


def test_the_extractor_loads_once_for_cached_values_and_once_for_formula_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """design.md Extraction: ExcelExtractor loads twice (values, then formulas)."""
    from npu_rag.ingest.extract import excel as excel_module

    calls: list[bool] = []
    real = getattr(excel_module, "load_workbook")

    def spy(
        source: object,
        *args: object,
        data_only: bool = False,
        **kwargs: object,
    ) -> object:
        calls.append(data_only)
        return real(source, *args, data_only=data_only, **kwargs)

    monkeypatch.setattr(excel_module, "load_workbook", spy)
    extract(source_for_fixture())
    assert calls == [True, False]


def test_the_cached_formula_cells_value_is_in_the_block_not_the_formula_text() -> None:
    """Requirement 4.4: the block carries the cached number, not ``=B4+B5``."""
    result = extract(source_for_fixture())
    first = blocks_of(result)[0]
    total_row = first.rows[2]
    assert total_row[0] == "Total"
    assert total_row[1] == str(GENERATE.CACHED_VALUE)
    assert total_row[1] != GENERATE.CACHED_FORMULA
    assert GENERATE.CACHED_FORMULA not in total_row


def test_each_formula_cell_emits_a_formula_segment_with_the_formula_text() -> None:
    """Requirement 4.4: formula text is a distinct FormulaSegment with a cell locator."""
    result = extract(source_for_fixture())
    expected = {
        GENERATE.CACHED_FORMULA_CELL: GENERATE.CACHED_FORMULA,
        "C6": "=C4+C5",
        "D6": "=D4+D5",
        GENERATE.UNCACHED_FORMULA_CELL: GENERATE.UNCACHED_FORMULA,
    }
    formulas = formulas_of(result)
    assert len(formulas) == len(expected)
    for cell, text in expected.items():
        segment = formula_by_cell(result, cell)
        assert segment.text == text
        assert segment.locator == SheetLocator(
            sheet=GENERATE.INCOME_SHEET, cell_range=cell
        )


def test_an_uncached_formula_cell_is_an_unavailable_value_omission_naming_the_cell() -> None:
    """Requirement 4.5: B12 has no cached value; record it, never guess."""
    source = source_for_fixture()
    result = extract(source)
    unavailable = [
        item
        for item in result.omissions
        if item.category is OmissionCategory.VALUE_UNAVAILABLE
    ]
    named = {item.reason for item in unavailable}
    assert any(GENERATE.UNCACHED_FORMULA_CELL in reason for reason in named)
    uncached = [
        item
        for item in unavailable
        if GENERATE.UNCACHED_FORMULA_CELL in item.reason
    ]
    assert len(uncached) == 1
    omission = uncached[0]
    assert isinstance(omission, Omission)
    assert omission.path == source.path
    assert omission.missing_capability is None
    assert "no cached value" in omission.reason
    assert GENERATE.HIDDEN_SHEET not in omission.reason


def test_uncached_formula_cells_are_empty_in_the_block_never_a_substituted_value() -> None:
    """Requirement 4.5: do not put a guessed number or the formula text in the block."""
    result = extract(source_for_fixture())
    first, second = blocks_of(result)
    total_row = first.rows[2]
    uncached_row = second.rows[2]

    assert total_row[1] == str(GENERATE.CACHED_VALUE)
    assert total_row[2] == ""
    assert total_row[3] == ""
    assert total_row[2] != SUBSTITUTED_C6
    assert total_row[3] != SUBSTITUTED_D6
    assert "=C4+C5" not in total_row
    assert "=D4+D5" not in total_row

    assert uncached_row[0] == "Uncached"
    assert uncached_row[1] == ""
    assert uncached_row[1] != SUBSTITUTED_B12
    assert uncached_row[1] != GENERATE.UNCACHED_FORMULA
    assert GENERATE.UNCACHED_FORMULA not in uncached_row

    block_cells = [cell for block in (first, second) for row in block.rows for cell in row]
    assert SUBSTITUTED_C6 not in block_cells
    assert SUBSTITUTED_D6 not in block_cells


def test_every_uncached_formula_cell_is_named_and_the_cached_cell_is_not() -> None:
    """C6 and D6 have no cached ``<v>`` either; B6 does, so it is not omitted."""
    result = extract(source_for_fixture())
    unavailable = [
        item
        for item in result.omissions
        if item.category is OmissionCategory.VALUE_UNAVAILABLE
    ]
    reasons = " ".join(item.reason for item in unavailable)
    assert GENERATE.CACHED_FORMULA_CELL not in reasons
    assert "C6" in reasons
    assert "D6" in reasons
    assert GENERATE.UNCACHED_FORMULA_CELL in reasons
    assert len(unavailable) == 3


def test_an_uncached_formula_still_emits_its_formula_segment() -> None:
    """design.md: formula text still goes to its FORMULA chunk when the value is None."""
    result = extract(source_for_fixture())
    segment = formula_by_cell(result, GENERATE.UNCACHED_FORMULA_CELL)
    assert segment.text == GENERATE.UNCACHED_FORMULA
    assert blocks_of(result)[1].rows[2][1] == ""


# --------------------------------------------------------------------------
# Requirements 4.6, 4.7: anchored images, charts, and series source ranges
# --------------------------------------------------------------------------

PINNED_OPENPYXL = "3.1.5"
CHART_VALUE_RANGE = "'Income'!$B$4:$B$5"
CHART_CATEGORY_RANGE = "'Income'!$A$4:$A$5"
DEFAULT_MIN_IMAGE_PIXELS = 200


def test_openpyxl_is_pinned_at_the_designed_version() -> None:
    """design.md Technology Stack 3.1.5: private drawing attrs are version-pinned."""
    import openpyxl

    assert openpyxl.__version__ == PINNED_OPENPYXL


def test_only_anchored_objects_touches_the_private_drawing_attributes() -> None:
    """design.md Extraction: all access to ws._images / ws._charts lives in one function."""
    assert functions_touching_private_drawing_attrs(EXCEL_PATH) == {"anchored_objects"}


def test_anchored_objects_fails_loudly_if_the_private_attributes_disappear() -> None:
    """Smoke: missing _images / _charts is AttributeError, never a silent empty list."""
    from npu_rag.ingest.extract.excel import anchored_objects

    workbook = load_workbook(FIXTURE_PATH, data_only=False)
    try:
        sheet = workbook[GENERATE.INCOME_SHEET]
        assert hasattr(sheet, "_images"), (
            "openpyxl worksheet no longer exposes _images; the adapter cannot find "
            "anchored images"
        )
        assert hasattr(sheet, "_charts"), (
            "openpyxl worksheet no longer exposes _charts; the adapter cannot find "
            "embedded charts"
        )
        refs = anchored_objects(sheet)
        assert len(refs) == 2
    finally:
        workbook.close()

    class BareWorksheet:
        title = GENERATE.INCOME_SHEET

    try:
        result = anchored_objects(BareWorksheet())
    except AttributeError:
        return
    pytest.fail(
        "anchored_objects returned an empty-or-silent result when _images/_charts "
        f"were absent: {result!r}"
    )


def test_planting_empty_private_lists_yields_no_image_refs() -> None:
    """The adapter reads ws._images and ws._charts in place, not a copy taken earlier."""
    from npu_rag.ingest.extract.excel import anchored_objects

    workbook = load_workbook(FIXTURE_PATH, data_only=False)
    try:
        sheet = workbook[GENERATE.INCOME_SHEET]
        sheet._images = []
        sheet._charts = []
        assert anchored_objects(sheet) == []
    finally:
        workbook.close()


def test_anchored_objects_emits_the_fixture_image_and_chart_with_series_ranges() -> None:
    """Requirement 4.6/4.7: one ImageRef per anchored image and chart; ranges from the series."""
    from npu_rag.ingest.extract.excel import anchored_objects

    workbook = load_workbook(FIXTURE_PATH, data_only=False)
    try:
        refs = anchored_objects(workbook[GENERATE.INCOME_SHEET])
    finally:
        workbook.close()

    assert len(refs) == 2
    assert all(isinstance(ref, ImageRef) for ref in refs)

    chart_refs = [ref for ref in refs if ref.chart_ranges is not None]
    image_refs = [ref for ref in refs if ref.chart_ranges is None]
    assert len(chart_refs) == 1
    assert len(image_refs) == 1

    chart = chart_refs[0]
    assert chart.chart_ranges == ChartRanges(
        value_ranges=(CHART_VALUE_RANGE,),
        category_ranges=(CHART_CATEGORY_RANGE,),
    )
    assert chart.locator == SheetLocator(
        sheet=GENERATE.INCOME_SHEET, cell_range=GENERATE.CHART_ANCHOR
    )
    assert chart.mime == "image/png"
    assert chart.data.startswith(PNG_MAGIC)
    assert min(chart.width, chart.height) >= DEFAULT_MIN_IMAGE_PIXELS

    image = image_refs[0]
    assert image.data == fixture_media_png()
    assert image.mime == "image/png"
    assert (image.width, image.height) == (8, 8)
    assert image.locator == SheetLocator(
        sheet=GENERATE.INCOME_SHEET, cell_range=GENERATE.IMAGE_ANCHOR
    )
    assert image.chart_ranges is None
    assert chart.data != image.data


def test_the_extractor_emits_image_refs_for_the_fixture_chart_and_image() -> None:
    """Requirement 4.6: ExcelExtractor routes anchored objects as ImageRefs."""
    result = extract(source_for_fixture())
    refs = image_refs_of(result)
    assert len(refs) == 2
    assert len(blocks_of(result)) == 2

    chart = next(ref for ref in refs if ref.chart_ranges is not None)
    image = next(ref for ref in refs if ref.chart_ranges is None)

    assert chart.chart_ranges == ChartRanges(
        value_ranges=(CHART_VALUE_RANGE,),
        category_ranges=(CHART_CATEGORY_RANGE,),
    )
    assert chart.locator == SheetLocator(
        sheet=GENERATE.INCOME_SHEET, cell_range=GENERATE.CHART_ANCHOR
    )
    assert image.data == fixture_media_png()
    assert image.locator == SheetLocator(
        sheet=GENERATE.INCOME_SHEET, cell_range=GENERATE.IMAGE_ANCHOR
    )


def test_chart_source_ranges_are_present_without_an_image_to_text_call() -> None:
    """Requirement 4.7: ranges come from the chart object; extract never imports vision."""
    result = extract(source_for_fixture())
    chart = next(ref for ref in image_refs_of(result) if ref.chart_ranges is not None)
    ranges = chart.chart_ranges
    assert ranges is not None
    assert CHART_VALUE_RANGE in ranges.value_ranges
    assert CHART_CATEGORY_RANGE in ranges.category_ranges
    names = imported_names(EXCEL_PATH)
    assert all(
        name != "npu_rag.ingest.vision"
        and not name.startswith("npu_rag.ingest.vision.")
        and name != "httpx"
        and not name.startswith("httpx.")
        for name in names
    ), names
