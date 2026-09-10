"""The vocabulary every other ingest module speaks (task 1.2).

This is the leftmost layer of design.md's dependency direction — ``types,
errors -> config -> credential -> identity -> state -> discover -> route ->
extract -> vision -> chunk, report -> pipeline`` — so it imports nothing from
this project at all. Everything here is a value: an enumeration or a frozen
dataclass, with no behaviour beyond the invariants that make an illegal value
unconstructible and the JSON round-trip the chunk registry persists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

__all__ = [
    "BlockSegment",
    "ChartRanges",
    "ChunkKind",
    "ChunkRecord",
    "Extracted",
    "FigureSegment",
    "FormulaSegment",
    "ImageFileLocator",
    "ImageRef",
    "Locator",
    "MarkdownLocator",
    "Omission",
    "OmissionCategory",
    "PageLocator",
    "ProseSegment",
    "Provenance",
    "Resolved",
    "RunReport",
    "Segment",
    "SheetLocator",
    "SourceFile",
    "TableSegment",
]


class ChunkKind(StrEnum):
    """The kind of a persisted chunk record (design.md, Domain Model)."""

    PROSE = "prose"
    TABLE = "table"
    BLOCK = "block"
    FORMULA = "formula"
    FIGURE = "figure"


class OmissionCategory(StrEnum):
    """Why an input was skipped or failed (design.md, Requirements Traceability)."""

    ROOT_UNAVAILABLE = "root_unavailable"
    UNSUPPORTED = "unsupported"
    VALUE_UNAVAILABLE = "value_unavailable"
    HIDDEN_SHEET = "hidden_sheet"
    VISION_UNAVAILABLE = "vision_unavailable"
    BELOW_THRESHOLD = "below_threshold"
    VISION_FAILED = "vision_failed"
    FAILED = "failed"


@dataclass(frozen=True)
class MarkdownLocator:
    """A position in a Markdown file: enclosing block line range and image ordinal."""

    line_range: tuple[int, int]
    ordinal: int


@dataclass(frozen=True)
class PageLocator:
    """A page in a paginated document."""

    page: int


@dataclass(frozen=True)
class SheetLocator:
    """A sheet name and cell range in a workbook."""

    sheet: str
    cell_range: str


@dataclass(frozen=True)
class ImageFileLocator:
    """A standalone raster file; the file itself is the location."""


Locator: TypeAlias = MarkdownLocator | PageLocator | SheetLocator | ImageFileLocator
_LOCATOR_TYPES = (MarkdownLocator, PageLocator, SheetLocator, ImageFileLocator)


def _require_locator(value: object, owner: str) -> None:
    if not isinstance(value, _LOCATOR_TYPES):
        raise TypeError(
            f"{owner} requires exactly one Locator, got {type(value).__name__}"
        )


@dataclass(frozen=True)
class Provenance:
    """Vision origin of a figure chunk (requirement 5.8)."""

    vision_model: str
    prompt_version: str


@dataclass(frozen=True)
class SourceFile:
    """One discovered file: path, its root, the relative path, and derived author."""

    path: Path
    root: Path
    relative_path: Path
    author: str


@dataclass(frozen=True)
class ChartRanges:
    """Series source ranges attached to a chart ImageRef (requirement 4.7)."""

    value_ranges: tuple[str, ...]
    category_ranges: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProseSegment:
    """Extractor prose: text, locator, and the heading path in reading order."""

    text: str
    locator: Locator
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class TableSegment:
    """A table as text, located in its source."""

    text: str
    locator: Locator


@dataclass(frozen=True)
class BlockSegment:
    """A labelled spreadsheet block the chunker can split without re-parsing."""

    label: str
    header_row: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    locator: Locator


@dataclass(frozen=True)
class FormulaSegment:
    """Formula text for one cell, located by sheet and range."""

    text: str
    locator: Locator


@dataclass(frozen=True)
class ImageRef:
    """Raw image bytes plus everything resolution needs; never a network call."""

    data: bytes
    mime: str
    width: int
    height: int
    locator: Locator
    chart_ranges: ChartRanges | None = None


Segment: TypeAlias = (
    ProseSegment | TableSegment | BlockSegment | FormulaSegment | ImageRef
)


@dataclass(frozen=True)
class FigureSegment:
    """A described image: text, source locator, and vision provenance."""

    text: str
    locator: Locator
    provenance: Provenance


@dataclass(frozen=True)
class Omission:
    """One skipped or failed input, by path and reason.

    Invariant: ``missing_capability`` is set if and only if ``category`` is
    ``VISION_UNAVAILABLE`` (requirement 9.4).
    """

    category: OmissionCategory
    path: Path
    reason: str
    missing_capability: str | None = None

    def __post_init__(self) -> None:
        named = self.missing_capability
        if named is not None and not named.strip():
            raise ValueError(
                "missing_capability must name a capability, not a blank string"
            )
        is_unavailable = self.category is OmissionCategory.VISION_UNAVAILABLE
        has_name = named is not None
        if is_unavailable != has_name:
            raise ValueError(
                "missing_capability is set if and only if category is "
                "VISION_UNAVAILABLE"
            )


@dataclass(frozen=True)
class Extracted:
    """One file's extraction result: title, reading-order segments, omissions."""

    title: str | None
    segments: tuple[Segment, ...]
    omissions: tuple[Omission, ...]


@dataclass(frozen=True)
class Resolved:
    """The vision seam's answer for one ImageRef.

    Invariant: exactly one of ``figure`` / ``omission`` is set.
    """

    figure: FigureSegment | None
    omission: Omission | None

    def __post_init__(self) -> None:
        if (self.figure is None) == (self.omission is None):
            raise ValueError(
                "Resolved requires exactly one of figure or omission"
            )


@dataclass(frozen=True)
class ChunkRecord:
    """The contract vector-index builds against (design.md, Logical Data Model).

    Invariants: exactly one Locator; a FIGURE record has a non-None Provenance
    and no other kind does (requirements 5.8, 7.1, 7.3).
    """

    chunk_id: str
    source_path: Path
    root_id: str
    author: str
    title: str | None
    kind: ChunkKind
    text: str
    locator: Locator
    ordinal: int
    token_count: int
    truncated: bool
    provenance: Provenance | None

    def __post_init__(self) -> None:
        _require_locator(self.locator, "ChunkRecord")
        if self.kind is ChunkKind.FIGURE:
            if self.provenance is None:
                raise ValueError(
                    "a FIGURE record requires provenance naming the vision model"
                )
        elif self.provenance is not None:
            raise ValueError("only a FIGURE record carries provenance")

    def to_json(self) -> str:
        """Serialize for ``chunk_registry.record_json``."""
        return json.dumps(self._to_dict())

    @classmethod
    def from_json(cls, payload: str) -> ChunkRecord:
        """Parse a record previously produced by :meth:`to_json`."""
        loaded: object = json.loads(payload)
        if not isinstance(loaded, dict):
            raise TypeError("ChunkRecord JSON must be an object")
        return cls._from_dict(loaded)

    def _to_dict(self) -> dict[str, object]:
        provenance: dict[str, str] | None = None
        if self.provenance is not None:
            provenance = {
                "vision_model": self.provenance.vision_model,
                "prompt_version": self.provenance.prompt_version,
            }
        return {
            "chunk_id": self.chunk_id,
            "source_path": str(self.source_path),
            "root_id": self.root_id,
            "author": self.author,
            "title": self.title,
            "kind": self.kind.value,
            "text": self.text,
            "locator": _locator_to_dict(self.locator),
            "ordinal": self.ordinal,
            "token_count": self.token_count,
            "truncated": self.truncated,
            "provenance": provenance,
        }

    @classmethod
    def _from_dict(cls, payload: dict[str, object]) -> ChunkRecord:
        title = payload["title"]
        if title is not None and not isinstance(title, str):
            raise TypeError("title must be a string or null")
        locator_payload = payload["locator"]
        if not isinstance(locator_payload, dict):
            raise TypeError("locator must be an object")
        provenance_payload = payload["provenance"]
        provenance: Provenance | None
        if provenance_payload is None:
            provenance = None
        elif isinstance(provenance_payload, dict):
            vision_model = provenance_payload["vision_model"]
            prompt_version = provenance_payload["prompt_version"]
            if not isinstance(vision_model, str) or not isinstance(prompt_version, str):
                raise TypeError("provenance fields must be strings")
            provenance = Provenance(
                vision_model=vision_model, prompt_version=prompt_version
            )
        else:
            raise TypeError("provenance must be an object or null")
        kind_value = payload["kind"]
        if not isinstance(kind_value, str):
            raise TypeError("kind must be a string")
        source_path = payload["source_path"]
        if not isinstance(source_path, str):
            raise TypeError("source_path must be a string")
        chunk_id = payload["chunk_id"]
        root_id = payload["root_id"]
        author = payload["author"]
        text = payload["text"]
        if not isinstance(chunk_id, str):
            raise TypeError("chunk_id must be a string")
        if not isinstance(root_id, str):
            raise TypeError("root_id must be a string")
        if not isinstance(author, str):
            raise TypeError("author must be a string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ordinal = payload["ordinal"]
        token_count = payload["token_count"]
        truncated = payload["truncated"]
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise TypeError("ordinal must be an int")
        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise TypeError("token_count must be an int")
        if not isinstance(truncated, bool):
            raise TypeError("truncated must be a bool")
        return cls(
            chunk_id=chunk_id,
            source_path=Path(source_path),
            root_id=root_id,
            author=author,
            title=title,
            kind=ChunkKind(kind_value),
            text=text,
            locator=_locator_from_dict(locator_payload),
            ordinal=ordinal,
            token_count=token_count,
            truncated=truncated,
            provenance=provenance,
        )


@dataclass(frozen=True)
class RunReport:
    """The data record search-cli renders; rendering itself is task 7.1."""

    files_processed: int
    files_unchanged: int
    files_skipped: int
    files_failed: int
    omissions: tuple[Omission, ...]
    removed_chunk_ids: tuple[str, ...]
    no_work_required: bool
    vision_requests_issued: int
    vision_cache_hits: int
    records: tuple[ChunkRecord, ...]


def _locator_to_dict(locator: Locator) -> dict[str, object]:
    if isinstance(locator, MarkdownLocator):
        return {
            "type": "markdown",
            "line_range": [locator.line_range[0], locator.line_range[1]],
            "ordinal": locator.ordinal,
        }
    if isinstance(locator, PageLocator):
        return {"type": "page", "page": locator.page}
    if isinstance(locator, SheetLocator):
        return {
            "type": "sheet",
            "sheet": locator.sheet,
            "cell_range": locator.cell_range,
        }
    if isinstance(locator, ImageFileLocator):
        return {"type": "image_file"}
    raise TypeError(f"unsupported locator type {type(locator).__name__}")


def _locator_from_dict(payload: dict[str, object]) -> Locator:
    kind = payload.get("type")
    if kind == "markdown":
        raw_range = payload["line_range"]
        ordinal = payload["ordinal"]
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise TypeError("markdown locator line_range must be a pair")
        start, end = raw_range
        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError("line_range start must be an int")
        if not isinstance(end, int) or isinstance(end, bool):
            raise TypeError("line_range end must be an int")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise TypeError("ordinal must be an int")
        return MarkdownLocator(line_range=(start, end), ordinal=ordinal)
    if kind == "page":
        page = payload["page"]
        if not isinstance(page, int) or isinstance(page, bool):
            raise TypeError("page must be an int")
        return PageLocator(page=page)
    if kind == "sheet":
        sheet = payload["sheet"]
        cell_range = payload["cell_range"]
        if not isinstance(sheet, str) or not isinstance(cell_range, str):
            raise TypeError("sheet locator fields must be strings")
        return SheetLocator(sheet=sheet, cell_range=cell_range)
    if kind == "image_file":
        return ImageFileLocator()
    raise ValueError(f"unknown locator type {kind!r}")
