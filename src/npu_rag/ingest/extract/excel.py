"""Excel extractor: blank-row blocks, values, formulas, charts, and images.

A row is empty only if every cell is empty and no valued merged range covers
it. The workbook is loaded twice: cached values first, then formula text.
Hidden sheets become ``Omission(HIDDEN_SHEET)`` and are never raised. A
formula cell with no cached value becomes ``Omission(VALUE_UNAVAILABLE)``
naming the cell; a value is never substituted. Every block carries a
``SheetLocator``. Anchored images and charts are read only through
``anchored_objects``, which is the sole access to the library's private
``_images`` / ``_charts`` attributes. Chart series source ranges are copied
onto the image reference; this module never imports vision, state, or httpx.

``SourceFile.path`` is opened as given, including the Windows ``\\\\?\\`` form,
by reading bytes and handing them to openpyxl.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

from openpyxl import load_workbook  # type: ignore[import-untyped]
from openpyxl.cell.cell import MergedCell  # type: ignore[import-untyped]
from openpyxl.utils import get_column_letter  # type: ignore[import-untyped]
from openpyxl.utils.exceptions import InvalidFileException  # type: ignore[import-untyped]
from openpyxl.worksheet.worksheet import Worksheet  # type: ignore[import-untyped]
from PIL import Image

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.extract.base import normalise
from npu_rag.ingest.types import (
    BlockSegment,
    ChartRanges,
    Extracted,
    FormulaSegment,
    ImageRef,
    Omission,
    OmissionCategory,
    Segment,
    SheetLocator,
    SourceFile,
)

__all__ = ["ExcelExtractor", "anchored_objects"]

_EMU_PER_PIXEL = 9525  # 96 dpi
_CM_TO_PIXELS = 96 / 2.54
_DEFAULT_CHART_WIDTH_CM = 15.0
_DEFAULT_CHART_HEIGHT_CM = 7.5


_LOAD_ERRORS = (
    InvalidFileException,
    zipfile.BadZipFile,
    OSError,
    KeyError,
    ValueError,
)


class ExcelExtractor:
    """Segment visible worksheets into labelled blocks. Conforms to ``Extractor``."""

    def extract(self, source: SourceFile, config: IngestConfig) -> Extracted:
        del config
        try:
            payload = source.path.read_bytes()
        except OSError as exc:
            raise ExtractionError(
                f"could not read {source.relative_path.as_posix()} as a workbook",
                stage="extraction",
                path=source.path,
            ) from exc
        try:
            values_workbook = load_workbook(BytesIO(payload), data_only=True)
        except _LOAD_ERRORS as exc:
            raise ExtractionError(
                f"could not read {source.relative_path.as_posix()} as a workbook",
                stage="extraction",
                path=source.path,
            ) from exc
        try:
            try:
                formulas_workbook = load_workbook(BytesIO(payload), data_only=False)
            except _LOAD_ERRORS as exc:
                raise ExtractionError(
                    f"could not read {source.relative_path.as_posix()} as a workbook",
                    stage="extraction",
                    path=source.path,
                ) from exc
            try:
                segments: list[Segment] = []
                omissions: list[Omission] = []
                for sheet in formulas_workbook.worksheets:
                    if sheet.sheet_state != "visible":
                        omissions.append(
                            Omission(
                                category=OmissionCategory.HIDDEN_SHEET,
                                path=source.path,
                                reason=f"hidden sheet {sheet.title!r} skipped",
                            )
                        )
                        continue
                    values_sheet = values_workbook[sheet.title]
                    segments.extend(_blocks_for_sheet(sheet, values_sheet))
                    formula_segments, value_omissions = _formulas_for_sheet(
                        sheet, values_sheet, source.path
                    )
                    segments.extend(formula_segments)
                    omissions.extend(value_omissions)
                    segments.extend(anchored_objects(sheet))
            finally:
                formulas_workbook.close()
        finally:
            values_workbook.close()
        return Extracted(title=None, segments=tuple(segments), omissions=tuple(omissions))


def anchored_objects(ws: Worksheet) -> list[ImageRef]:
    """ImageRefs for every anchored image and chart on *ws*.

    Direct attribute access: if ``_images`` or ``_charts`` disappear, this
    raises ``AttributeError`` instead of returning an empty list.
    """
    images: Sequence[object] = ws._images
    charts: Sequence[object] = ws._charts
    located: list[tuple[tuple[int, int], ImageRef]] = []
    for image in images:
        located.append((_anchor_order(image), _image_ref(ws, image)))
    for chart in charts:
        located.append((_anchor_order(chart), _chart_ref(ws, chart)))
    located.sort(key=lambda item: item[0])
    return [ref for _, ref in located]


def _image_ref(sheet: Worksheet, image: object) -> ImageRef:
    loader = getattr(image, "_data")
    payload = loader() if callable(loader) else loader
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("anchored image _data did not return bytes")
    return _raster_ref(
        bytes(payload),
        SheetLocator(sheet=sheet.title, cell_range=_anchor_cell(image)),
        ranges=None,
    )


def _chart_ref(sheet: Worksheet, chart: object) -> ImageRef:
    width, height = _chart_pixel_size(chart)
    return _raster_ref(
        _png_bytes(width, height),
        SheetLocator(sheet=sheet.title, cell_range=_anchor_cell(chart)),
        ranges=_chart_ranges(chart),
    )


def _raster_ref(
    data: bytes, locator: SheetLocator, ranges: ChartRanges | None
) -> ImageRef:
    with Image.open(BytesIO(data)) as parsed:
        width, height = parsed.size
        fmt = parsed.format
    if not fmt:
        raise TypeError("anchored object produced an image with no format")
    mime = Image.MIME.get(fmt, "application/octet-stream")
    return ImageRef(
        data=data,
        mime=mime,
        width=int(width),
        height=int(height),
        locator=locator,
        chart_ranges=ranges,
    )


def _chart_ranges(chart: object) -> ChartRanges:
    values: list[str] = []
    categories: list[str] = []
    series = getattr(chart, "series", ()) or ()
    for item in series:
        value = _series_formula(getattr(item, "val", None))
        if value is not None:
            values.append(value)
        category = _series_formula(getattr(item, "cat", None))
        if category is not None:
            categories.append(category)
    return ChartRanges(
        value_ranges=tuple(values), category_ranges=tuple(categories)
    )


def _series_formula(source: object) -> str | None:
    if source is None:
        return None
    for name in ("numRef", "strRef", "multiLvlStrRef"):
        ref = getattr(source, name, None)
        formula = getattr(ref, "f", None) if ref is not None else None
        if isinstance(formula, str) and formula:
            return formula
    return None


def _chart_pixel_size(chart: object) -> tuple[int, int]:
    anchor = getattr(chart, "anchor", None)
    ext = getattr(anchor, "ext", None)
    cx = getattr(ext, "cx", None)
    cy = getattr(ext, "cy", None)
    if isinstance(cx, int) and isinstance(cy, int) and cx > 0 and cy > 0:
        return (
            max(1, round(cx / _EMU_PER_PIXEL)),
            max(1, round(cy / _EMU_PER_PIXEL)),
        )
    width_cm = getattr(chart, "width", None)
    height_cm = getattr(chart, "height", None)
    width = (
        float(width_cm)
        if isinstance(width_cm, (int, float))
        else _DEFAULT_CHART_WIDTH_CM
    )
    height = (
        float(height_cm)
        if isinstance(height_cm, (int, float))
        else _DEFAULT_CHART_HEIGHT_CM
    )
    return (
        max(1, round(width * _CM_TO_PIXELS)),
        max(1, round(height * _CM_TO_PIXELS)),
    )


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _anchor_cell(obj: object) -> str:
    anchor = getattr(obj, "anchor", None)
    if isinstance(anchor, str) and anchor:
        return anchor
    marker = getattr(anchor, "_from", None)
    if marker is None:
        return "A1"
    column = int(getattr(marker, "col")) + 1
    row = int(getattr(marker, "row")) + 1
    return f"{get_column_letter(column)}{row}"


def _anchor_order(obj: object) -> tuple[int, int]:
    anchor = getattr(obj, "anchor", None)
    marker = getattr(anchor, "_from", None)
    if marker is None:
        return (0, 0)
    return (int(getattr(marker, "row")), int(getattr(marker, "col")))


def _row_is_empty(
    row_index: int,
    values: Sequence[object],
    valued_merged_ranges: Sequence[tuple[int, int]],
) -> bool:
    """A row is empty iff every cell is empty and no valued merged range covers it."""
    if any(not _is_empty_cell(value) for value in values):
        return False
    return not any(start <= row_index <= end for start, end in valued_merged_ranges)


def _is_empty_cell(value: object) -> bool:
    return value is None or value == ""


def _blocks_for_sheet(
    sheet: Worksheet, values_sheet: Worksheet
) -> list[BlockSegment]:
    max_row = int(sheet.max_row or 0)
    max_col = int(sheet.max_column or 0)
    if max_row == 0 or max_col == 0:
        return []
    spans = _valued_merged_row_spans(sheet)
    rows = list(sheet.iter_rows(min_row=1, max_row=max_row, max_col=max_col))
    return [
        _block_from_run(sheet, values_sheet, start, end, max_col)
        for start, end in _nonempty_runs(rows, spans)
    ]


def _formulas_for_sheet(
    sheet: Worksheet, values_sheet: Worksheet, path: Path
) -> tuple[list[FormulaSegment], list[Omission]]:
    max_row = int(sheet.max_row or 0)
    max_col = int(sheet.max_column or 0)
    if max_row == 0 or max_col == 0:
        return [], []
    segments: list[FormulaSegment] = []
    omissions: list[Omission] = []
    for row in sheet.iter_rows(min_row=1, max_row=max_row, max_col=max_col):
        for cell in row:
            text = _formula_text(cell)
            if text is None:
                continue
            coordinate = str(cell.coordinate)
            segments.append(
                FormulaSegment(
                    text=text,
                    locator=SheetLocator(sheet=sheet.title, cell_range=coordinate),
                )
            )
            cached = values_sheet[coordinate].value
            if _is_empty_cell(cached):
                omissions.append(
                    Omission(
                        category=OmissionCategory.VALUE_UNAVAILABLE,
                        path=path,
                        reason=(
                            f"cell {coordinate} on sheet {sheet.title!r} has no "
                            "cached value; the workbook was not saved by an "
                            "application that computes formulas"
                        ),
                    )
                )
    return segments, omissions


def _formula_text(cell: object) -> str | None:
    if isinstance(cell, MergedCell):
        return None
    if getattr(cell, "data_type", None) != "f":
        return None
    value = getattr(cell, "value", None)
    if isinstance(value, str):
        return value if value.startswith("=") else f"={value}"
    text = getattr(value, "text", None)
    if isinstance(text, str) and text:
        return text if text.startswith("=") else f"={text}"
    return None


def _valued_merged_row_spans(sheet: Worksheet) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for merged in sheet.merged_cells.ranges:
        anchor = sheet.cell(merged.min_row, merged.min_col)
        if not _is_empty_cell(anchor.value):
            spans.append((int(merged.min_row), int(merged.max_row)))
    return tuple(spans)


def _nonempty_runs(
    rows: list[tuple[object, ...]],
    spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, row in enumerate(rows, start=1):
        values = tuple(getattr(cell, "value", None) for cell in row)
        if _row_is_empty(index, values, spans):
            if start is not None:
                runs.append((start, index - 1))
                start = None
        elif start is None:
            start = index
    if start is not None:
        runs.append((start, len(rows)))
    return runs


def _block_from_run(
    sheet: Worksheet,
    values_sheet: Worksheet,
    start: int,
    end: int,
    max_col: int,
) -> BlockSegment:
    min_col, used_col = _used_columns(sheet, start, end, max_col)
    label = ""
    header_row: tuple[str, ...] = ()
    header_seen = False
    data_rows: list[tuple[str, ...]] = []
    for row_index in range(start, end + 1):
        texts = tuple(
            _cell_text(values_sheet, row_index, column)
            for column in range(min_col, used_col + 1)
        )
        if row_index == start:
            label = texts[0] if texts else ""
        if not header_seen:
            if any(texts[1:]):
                header_row = texts
                header_seen = True
            continue
        if _row_has_own_value(sheet, row_index, min_col, used_col):
            data_rows.append(texts)
    cell_range = (
        f"{get_column_letter(min_col)}{start}:{get_column_letter(used_col)}{end}"
    )
    return BlockSegment(
        label=label,
        header_row=header_row,
        rows=tuple(data_rows),
        locator=SheetLocator(sheet=sheet.title, cell_range=cell_range),
    )


def _used_columns(
    sheet: Worksheet, start: int, end: int, max_col: int
) -> tuple[int, int]:
    min_used: int | None = None
    max_used: int | None = None

    def _span(column: int) -> None:
        nonlocal min_used, max_used
        min_used = column if min_used is None else min(min_used, column)
        max_used = column if max_used is None else max(max_used, column)

    for row_index in range(start, end + 1):
        for column in range(1, max_col + 1):
            cell = sheet.cell(row_index, column)
            if isinstance(cell, MergedCell):
                continue
            if not _is_empty_cell(cell.value):
                _span(column)
    for merged in sheet.merged_cells.ranges:
        if int(merged.max_row) < start or int(merged.min_row) > end:
            continue
        anchor = sheet.cell(merged.min_row, merged.min_col)
        if _is_empty_cell(anchor.value):
            continue
        _span(int(merged.min_col))
        _span(int(merged.max_col))
    if min_used is None or max_used is None:
        return 1, 1
    return min_used, max_used


def _row_has_own_value(
    sheet: Worksheet, row_index: int, min_col: int, max_col: int
) -> bool:
    for column in range(min_col, max_col + 1):
        cell = sheet.cell(row_index, column)
        if isinstance(cell, MergedCell):
            continue
        if not _is_empty_cell(cell.value):
            return True
    return False


def _cell_text(sheet: Worksheet, row_index: int, column: int) -> str:
    cell = sheet.cell(row_index, column)
    value = cell.value
    if _is_empty_cell(value):
        return ""
    if isinstance(value, bool):
        return normalise(str(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        return normalise(value)
    return normalise(str(value))
