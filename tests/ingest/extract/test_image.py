"""Unit tests for the standalone image extractor (task 4.7).

Requirement 5.3: dimensions are recorded so the size threshold is measurable;
the extractor still emits an ImageRef when the shorter side is below
``IngestConfig.min_image_pixels`` (default 200). The gate itself is vision's.
Requirement 7.3: the file itself is the location — ``ImageFileLocator``.
Corrupt or undecodable images raise ``ExtractionError`` with stage and path
(requirement 8.4). ``SourceFile.path`` is opened in its Windows ``\\\\?\\``
form as given.
"""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.extract.base import Extractor
from npu_rag.ingest.extract.image import ImageExtractor
from npu_rag.ingest.types import (
    Extracted,
    ImageFileLocator,
    ImageRef,
    SourceFile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "src" / "npu_rag" / "ingest"
IMAGE_PATH = PACKAGE_ROOT / "extract" / "image.py"
EXTRACT_INIT_PATH = PACKAGE_ROOT / "extract" / "__init__.py"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "images"
CHART_PATH = FIXTURE_DIR / "chart.png"
ICON_PATH = FIXTURE_DIR / "icon.jpg"

LONG_PREFIX = "\\\\?\\"
TRADITIONAL_LIMIT = 260
BUDGET = 256
DEFAULT_MIN_IMAGE_PIXELS = 200
CHART_SIZE = (256, 201)
ICON_SIZE = (16, 16)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

FORBIDDEN_IMPORTS = (
    "npu_rag.ingest.vision",
    "npu_rag.ingest.state",
    "httpx",
)


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


def source_for(path: Path) -> SourceFile:
    resolved = path.resolve()
    return SourceFile(
        path=as_windows_long_path(resolved),
        root=as_windows_long_path(resolved.parent),
        relative_path=Path("alice") / resolved.name,
        author="alice",
    )


def extract(source: SourceFile, **overrides: object) -> Extracted:
    extractor: Extractor = ImageExtractor()
    return extractor.extract(source, make_config(**overrides))


def header_info(path: Path) -> tuple[bytes, str, int, int]:
    data = path.read_bytes()
    with Image.open(BytesIO(data)) as image:
        width, height = image.size
        fmt = image.format
    assert fmt is not None
    return data, Image.MIME[fmt], int(width), int(height)


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


def the_only_image_ref(result: Extracted) -> ImageRef:
    assert result.title is None
    assert result.omissions == ()
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert isinstance(segment, ImageRef)
    return segment


# --------------------------------------------------------------------------
# Fixture non-vacuity
# --------------------------------------------------------------------------


def test_the_committed_chart_is_above_the_default_size_threshold() -> None:
    assert CHART_PATH.is_file()
    assert CHART_PATH.read_bytes().startswith(PNG_MAGIC)
    data, mime, width, height = header_info(CHART_PATH)
    assert (width, height) == CHART_SIZE
    assert min(width, height) > DEFAULT_MIN_IMAGE_PIXELS
    assert mime == "image/png"
    assert data == CHART_PATH.read_bytes()
    assert make_config().min_image_pixels == DEFAULT_MIN_IMAGE_PIXELS


def test_the_committed_icon_is_below_the_default_size_threshold() -> None:
    assert ICON_PATH.is_file()
    assert ICON_PATH.read_bytes().startswith(JPEG_MAGIC)
    data, mime, width, height = header_info(ICON_PATH)
    assert (width, height) == ICON_SIZE
    assert min(width, height) < DEFAULT_MIN_IMAGE_PIXELS
    assert mime == "image/jpeg"
    assert data == ICON_PATH.read_bytes()
    assert min(*CHART_SIZE) != min(*ICON_SIZE)


# --------------------------------------------------------------------------
# Requirements 5.3, 7.3: the fixture observable
# --------------------------------------------------------------------------


def test_the_chart_fixture_yields_a_reference_with_width_height_and_mime() -> None:
    data, mime, width, height = header_info(CHART_PATH)
    result = extract(source_for(CHART_PATH))
    ref = the_only_image_ref(result)

    assert ref.data == data
    assert ref.data == CHART_PATH.read_bytes()
    assert ref.mime == mime
    assert ref.mime == "image/png"
    assert (ref.width, ref.height) == (width, height) == CHART_SIZE
    assert min(ref.width, ref.height) > DEFAULT_MIN_IMAGE_PIXELS
    assert isinstance(ref.locator, ImageFileLocator)
    assert ref.locator == ImageFileLocator()
    assert ref.chart_ranges is None


def test_the_icon_fixture_yields_a_reference_whose_shorter_side_is_below_threshold() -> None:
    data, mime, width, height = header_info(ICON_PATH)
    result = extract(source_for(ICON_PATH))
    ref = the_only_image_ref(result)

    assert ref.data == data
    assert ref.mime == mime
    assert ref.mime == "image/jpeg"
    assert (ref.width, ref.height) == (width, height) == ICON_SIZE
    assert min(ref.width, ref.height) < DEFAULT_MIN_IMAGE_PIXELS
    assert isinstance(ref.locator, ImageFileLocator)
    assert ref.locator == ImageFileLocator()
    assert ref.chart_ranges is None


def test_the_extractor_still_emits_an_image_ref_when_the_shorter_side_is_below_threshold() -> None:
    """Requirement 5.3's gate is applied later at vision, not here."""
    result = extract(source_for(ICON_PATH), min_image_pixels=10_000)
    ref = the_only_image_ref(result)
    assert min(ref.width, ref.height) < DEFAULT_MIN_IMAGE_PIXELS
    assert min(ref.width, ref.height) < 10_000
    assert result.omissions == ()


# --------------------------------------------------------------------------
# Windows long paths (requirement 1.5 / design 3.1)
# --------------------------------------------------------------------------


def test_image_extractor_opens_the_windows_long_path_form() -> None:
    source = source_for(CHART_PATH)
    assert str(source.path).startswith(LONG_PREFIX)
    result = extract(source)
    ref = the_only_image_ref(result)
    assert ref.data == CHART_PATH.read_bytes()
    assert (ref.width, ref.height) == CHART_SIZE


def test_an_image_behind_a_path_longer_than_the_traditional_limit_is_extracted(
    tmp_path: Path,
) -> None:
    root = as_windows_long_path(tmp_path / "archive")
    root.mkdir(parents=True, exist_ok=True)
    directory = root
    name = "chart.png"
    while traditional_length(directory / name) <= TRADITIONAL_LIMIT:
        directory = directory / ("subdir" * 4)
        directory.mkdir(parents=True, exist_ok=True)
        if traditional_length(directory) > 400:
            pytest.fail(
                f"cannot construct a path longer than {TRADITIONAL_LIMIT} under {root}"
            )
    target = directory / name
    payload = CHART_PATH.read_bytes()
    target.write_bytes(payload)
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
    ref = the_only_image_ref(result)
    assert ref.data == payload
    assert (ref.width, ref.height) == CHART_SIZE
    assert ref.mime == "image/png"


# --------------------------------------------------------------------------
# Corrupt / undecodable image: ExtractionError with stage and path (8.4)
# --------------------------------------------------------------------------


def test_a_corrupt_image_raises_extraction_error_with_stage_and_path(
    tmp_path: Path,
) -> None:
    source = plant(tmp_path, "alice/broken.png", b"this is not an image")
    with pytest.raises(ExtractionError) as caught:
        extract(source)
    error = caught.value
    assert error.stage == "extraction"
    assert error.path == source.path
    assert str(source.path).startswith(LONG_PREFIX)


def test_an_unreadable_path_raises_extraction_error_with_stage_and_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "alice" / "gone.png"
    source = SourceFile(
        path=as_windows_long_path(missing),
        root=as_windows_long_path(tmp_path),
        relative_path=Path("alice") / "gone.png",
        author="alice",
    )
    with pytest.raises(ExtractionError) as caught:
        extract(source)
    error = caught.value
    assert error.stage == "extraction"
    assert error.path == source.path


# --------------------------------------------------------------------------
# Package export
# --------------------------------------------------------------------------


def test_image_extractor_is_exported_from_the_extract_package() -> None:
    from npu_rag.ingest.extract import ImageExtractor as Exported
    from npu_rag.ingest.extract import __all__ as extract_all

    assert "ImageExtractor" in extract_all
    assert Exported is ImageExtractor


# --------------------------------------------------------------------------
# Layer guard: extract never imports vision, state, or httpx
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [IMAGE_PATH, EXTRACT_INIT_PATH],
    ids=["image.py", "__init__.py"],
)
def test_extract_image_modules_do_not_import_vision_state_or_httpx(path: Path) -> None:
    names = imported_names(path)
    for forbidden in FORBIDDEN_IMPORTS:
        assert all(
            name != forbidden and not name.startswith(f"{forbidden}.")
            for name in names
        ), names
