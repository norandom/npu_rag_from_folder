"""Unit tests for the PDF extractor (task 4.3).

Requirement 2.2: a page at or above min_page_chars is extracted locally as
prose and is not routed to image-to-text. Requirement 2.3: a page below the
threshold is routed as an ImageRef. Requirement 2.5: the threshold is
configuration. Requirement 3.4: local extraction preserves content-stream
order within the page. Requirement 7.3: every segment carries a page locator.
Page numbers are 1-based. Corrupt PDFs raise ExtractionError with stage and
path. SourceFile.path is opened in its Windows ``\\\\?\\`` form as given.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Protocol, cast

import pytest
from PIL import Image

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.extract.base import Extractor, normalise
from npu_rag.ingest.extract.pdf import PdfExtractor
from npu_rag.ingest.types import (
    Extracted,
    ImageRef,
    PageLocator,
    ProseSegment,
    SourceFile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "src" / "npu_rag" / "ingest"
PDF_PATH = PACKAGE_ROOT / "extract" / "pdf.py"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "pdf"
FIXTURE_PATH = FIXTURE_DIR / "text_and_textless.pdf"
GENERATE_PATH = FIXTURE_DIR / "generate.py"

LONG_PREFIX = "\\\\?\\"
TRADITIONAL_LIMIT = 260
BUDGET = 256
DEFAULT_MIN_PAGE_CHARS = 50
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

FORBIDDEN_IMPORTS = (
    "npu_rag.ingest.vision",
    "npu_rag.ingest.state",
    "httpx",
)

FIRST_SENTENCE = "First sentence of extractable prose on the text page."
SECOND_SENTENCE = "Second sentence follows in content-stream order."
EXPECTED_PROSE = (
    "First sentence of extractable prose on the text page. "
    "Second sentence follows in content-stream order."
)
PAGE_WIDTH = 300
PAGE_HEIGHT = 200


class _PdfFixtureGenerate(Protocol):
    def build_pdf(self, pages: Sequence[object]) -> bytes: ...
    def committed_fixture_bytes(self) -> bytes: ...


def load_generate() -> _PdfFixtureGenerate:
    spec = importlib.util.spec_from_file_location(
        "ingest_pdf_fixture_generate", GENERATE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_PdfFixtureGenerate, module)


GENERATE = load_generate()


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
        relative_path=Path("alice") / "text_and_textless.pdf",
        author="alice",
    )


def extract(source: SourceFile, **overrides: object) -> Extracted:
    extractor: Extractor = PdfExtractor()
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


# --------------------------------------------------------------------------
# Fixture non-vacuity
# --------------------------------------------------------------------------


def test_the_committed_fixture_still_has_a_text_page_and_a_textless_page() -> None:
    """Non-vacuity: the file on disk still has one text layer and one empty layer."""
    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    raw = FIXTURE_PATH.read_bytes()
    assert raw == GENERATE.committed_fixture_bytes()
    assert len(raw) < 4096
    assert raw.startswith(b"%PDF")
    first_at = raw.find(FIRST_SENTENCE.encode("ascii"))
    second_at = raw.find(SECOND_SENTENCE.encode("ascii"))
    assert first_at != -1
    assert second_at != -1
    assert first_at < second_at
    document = pdfium.PdfDocument(raw)
    try:
        assert len(document) == 2
        first = document[0]
        try:
            textpage = first.get_textpage()
            try:
                text = textpage.get_text_range()
            finally:
                textpage.close()
        finally:
            first.close()
        second = document[1]
        try:
            textpage = second.get_textpage()
            try:
                empty = textpage.get_text_range()
            finally:
                textpage.close()
        finally:
            second.close()
    finally:
        document.close()
    assert FIRST_SENTENCE in text
    assert SECOND_SENTENCE in text
    assert text.find(FIRST_SENTENCE) < text.find(SECOND_SENTENCE)
    assert len(text.strip()) > DEFAULT_MIN_PAGE_CHARS
    assert empty.strip() == ""
    assert len(empty.strip()) < DEFAULT_MIN_PAGE_CHARS


def test_the_generator_reproduces_the_committed_fixture_byte_for_byte() -> None:
    assert GENERATE.committed_fixture_bytes() == FIXTURE_PATH.read_bytes()


# --------------------------------------------------------------------------
# Requirements 2.2, 2.3, 3.4, 7.3: the fixture observable
# --------------------------------------------------------------------------


def test_the_text_page_yields_prose_and_the_textless_page_yields_an_image_ref() -> None:
    result = extract(source_for_fixture())

    assert result.title is None
    assert result.omissions == ()
    assert len(result.segments) == 2
    prose, image = result.segments

    assert isinstance(prose, ProseSegment)
    assert not isinstance(prose, ImageRef)
    assert prose.text == EXPECTED_PROSE
    assert prose.text == normalise(
        FIRST_SENTENCE + "\r\n" + SECOND_SENTENCE
    )
    assert FIRST_SENTENCE in prose.text
    assert SECOND_SENTENCE in prose.text
    assert prose.text.find(FIRST_SENTENCE) < prose.text.find(SECOND_SENTENCE)
    assert "\r" not in prose.text
    assert "\n" not in prose.text
    assert prose.locator == PageLocator(page=1)
    assert prose.locator.page != 0
    assert prose.heading_path == ()

    assert isinstance(image, ImageRef)
    assert not isinstance(image, ProseSegment)
    assert image.mime == "image/png"
    assert image.data.startswith(PNG_MAGIC)
    assert image.locator == PageLocator(page=2)
    assert image.locator.page != 0
    assert image.chart_ranges is None
    with Image.open(BytesIO(image.data)) as rendered:
        assert rendered.format == "PNG"
        assert rendered.size == (PAGE_WIDTH, PAGE_HEIGHT)
    assert (image.width, image.height) == (PAGE_WIDTH, PAGE_HEIGHT)

    assert not any(
        isinstance(segment, ImageRef) and segment.locator == PageLocator(page=1)
        for segment in result.segments
    )
    assert not any(
        isinstance(segment, ProseSegment) and segment.locator == PageLocator(page=2)
        for segment in result.segments
    )


def test_every_segment_from_the_fixture_carries_its_one_based_page() -> None:
    result = extract(source_for_fixture())
    pages = []
    for segment in result.segments:
        assert isinstance(segment.locator, PageLocator)
        assert segment.locator.page >= 1
        pages.append(segment.locator.page)
    assert pages == [1, 2]
    assert 0 not in pages


def test_pdf_extractor_opens_the_windows_long_path_form() -> None:
    source = source_for_fixture()
    assert str(source.path).startswith(LONG_PREFIX)
    result = extract(source)
    assert isinstance(result.segments[0], ProseSegment)
    assert result.segments[0].text == EXPECTED_PROSE


def test_a_pdf_behind_a_path_longer_than_the_traditional_limit_is_extracted(
    tmp_path: Path,
) -> None:
    root = as_windows_long_path(tmp_path / "archive")
    root.mkdir(parents=True, exist_ok=True)
    directory = root
    name = "pages.pdf"
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
    assert len(result.segments) == 2
    assert isinstance(result.segments[0], ProseSegment)
    assert result.segments[0].text == EXPECTED_PROSE
    assert isinstance(result.segments[1], ImageRef)


# --------------------------------------------------------------------------
# Requirement 2.5: min_page_chars is configuration and is applied per page
# --------------------------------------------------------------------------


def test_a_page_at_the_threshold_is_extracted_as_prose(tmp_path: Path) -> None:
    """Requirement 2.2: at or above the minimum, extract locally."""
    text = "a" * DEFAULT_MIN_PAGE_CHARS
    assert len(text) == DEFAULT_MIN_PAGE_CHARS
    source = plant(tmp_path, "alice/exact.pdf", GENERATE.build_pdf((text,)))
    result = extract(source)
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert isinstance(segment, ProseSegment)
    assert segment.text == text
    assert segment.locator == PageLocator(page=1)
    assert not any(isinstance(item, ImageRef) for item in result.segments)


def test_a_page_just_below_the_threshold_is_routed_as_an_image_ref(
    tmp_path: Path,
) -> None:
    """Requirement 2.3: below the minimum, route to image-to-text."""
    text = "b" * (DEFAULT_MIN_PAGE_CHARS - 1)
    assert len(text) == 49
    source = plant(tmp_path, "alice/short.pdf", GENERATE.build_pdf((text,)))
    result = extract(source)
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert isinstance(segment, ImageRef)
    assert segment.locator == PageLocator(page=1)
    assert segment.mime == "image/png"
    assert not any(isinstance(item, ProseSegment) for item in result.segments)


def test_raising_the_configured_threshold_reroutes_the_text_page_to_an_image_ref() -> None:
    """Requirement 2.5: the extractor reads min_page_chars from configuration."""
    source = source_for_fixture()
    default = extract(source)
    assert isinstance(default.segments[0], ProseSegment)
    raised = extract(source, min_page_chars=10_000)
    assert len(raised.segments) == 2
    assert all(isinstance(segment, ImageRef) for segment in raised.segments)
    assert [segment.locator for segment in raised.segments] == [
        PageLocator(page=1),
        PageLocator(page=2),
    ]
    assert not any(isinstance(segment, ProseSegment) for segment in raised.segments)


def test_lowering_the_configured_threshold_does_not_turn_a_textless_page_into_prose() -> None:
    result = extract(source_for_fixture(), min_page_chars=1)
    assert len(result.segments) == 2
    assert isinstance(result.segments[0], ProseSegment)
    assert isinstance(result.segments[1], ImageRef)
    assert result.segments[1].locator == PageLocator(page=2)


# --------------------------------------------------------------------------
# Corrupt PDF: ExtractionError with stage and path (requirement 8.4 later)
# --------------------------------------------------------------------------


def test_a_corrupt_pdf_raises_extraction_error_with_stage_and_path(
    tmp_path: Path,
) -> None:
    source = plant(tmp_path, "alice/broken.pdf", b"this is not a pdf document")
    with pytest.raises(ExtractionError) as caught:
        extract(source)
    error = caught.value
    assert error.stage == "extraction"
    assert error.path == source.path
    assert str(source.path).startswith(LONG_PREFIX)


def test_an_unreadable_path_raises_extraction_error_with_stage_and_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "alice" / "gone.pdf"
    source = SourceFile(
        path=as_windows_long_path(missing),
        root=as_windows_long_path(tmp_path),
        relative_path=Path("alice") / "gone.pdf",
        author="alice",
    )
    with pytest.raises(ExtractionError) as caught:
        extract(source)
    error = caught.value
    assert error.stage == "extraction"
    assert error.path == source.path


# --------------------------------------------------------------------------
# Layer guard: extract never imports vision, state, or httpx
# --------------------------------------------------------------------------


def test_pdf_extractor_does_not_import_vision_state_or_httpx() -> None:
    names = imported_names(PDF_PATH)
    for forbidden in FORBIDDEN_IMPORTS:
        assert all(
            name != forbidden and not name.startswith(f"{forbidden}.")
            for name in names
        ), names
