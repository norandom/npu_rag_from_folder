"""Unit tests for extraction-path routing (task 3.2).

Requirements 2.1 and 2.4: the path is chosen from the file type together
with inspected content, not the extension alone; an unsupported type is
recorded with its path and does not fail the run.

``route.py`` sits to the right of discover and to the left of extract. These
tests never import extract adapters, vision, or httpx. The token-to-adapter
table belongs to the pipeline.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from npu_rag.ingest.errors import IngestError
from npu_rag.ingest.route import ExtractionPath, Router
from npu_rag.ingest.types import Omission, OmissionCategory, SourceFile

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "ingest" / "route.py"
)

LONG_PREFIX = "\\\\?\\"
TRADITIONAL_LIMIT = 260

FORBIDDEN_IMPORTS = (
    "npu_rag.ingest.extract",
    "npu_rag.ingest.vision",
    "npu_rag.ingest.pipeline",
    "httpx",
)

# 1x1 RGBA PNG (67 bytes). Pillow identifies this from the header without a
# full raster decode; the bytes are the fixture, not a mock of pillow.
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)

PDF_HEADER = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
ZIP_HEADER = b"PK\x03\x04" + b"\x00" * 16
MARKDOWN_BODY = b"# Title\n\nA paragraph.\n"
TEXT_BODY = b"plain text notes\n"
UNKNOWN_BODY = b"\x00\x01\x02not-a-supported-type"


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


def route(source: SourceFile) -> ExtractionPath | Omission:
    return Router().route(source)


# --------------------------------------------------------------------------
# ExtractionPath tokens name the five adapters; they are not the adapters
# --------------------------------------------------------------------------


def test_extraction_path_tokens_match_the_five_adapter_names() -> None:
    """design.md File Structure Plan: markdown, text, pdf, excel, image."""
    assert {member.value for member in ExtractionPath} == {
        "markdown",
        "text",
        "pdf",
        "excel",
        "image",
    }


# --------------------------------------------------------------------------
# Requirement 2.1: extension without a stronger sniff
# --------------------------------------------------------------------------


def test_markdown_extension_without_a_stronger_sniff_routes_to_markdown(
    tmp_path: Path,
) -> None:
    source = plant(tmp_path, "alice/notes.md", MARKDOWN_BODY)
    assert route(source) is ExtractionPath.MARKDOWN


def test_markdown_long_extension_without_a_stronger_sniff_routes_to_markdown(
    tmp_path: Path,
) -> None:
    source = plant(tmp_path, "alice/notes.markdown", MARKDOWN_BODY)
    assert route(source) is ExtractionPath.MARKDOWN


def test_markdown_extension_is_matched_case_insensitively(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/notes.MD", MARKDOWN_BODY)
    assert route(source) is ExtractionPath.MARKDOWN


def test_txt_extension_without_a_stronger_sniff_routes_to_text(
    tmp_path: Path,
) -> None:
    source = plant(tmp_path, "alice/notes.txt", TEXT_BODY)
    assert route(source) is ExtractionPath.TEXT


def test_pdf_magic_with_a_pdf_extension_routes_to_pdf(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/paper.pdf", PDF_HEADER)
    assert route(source) is ExtractionPath.PDF


def test_zip_signature_with_an_xlsx_extension_routes_to_excel(
    tmp_path: Path,
) -> None:
    source = plant(tmp_path, "alice/model.xlsx", ZIP_HEADER)
    assert route(source) is ExtractionPath.EXCEL


def test_png_header_with_a_png_extension_routes_to_image(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/chart.png", PNG_1X1)
    assert route(source) is ExtractionPath.IMAGE


def test_unknown_bytes_with_a_pdf_extension_route_to_pdf(tmp_path: Path) -> None:
    """Inconclusive sniff still follows a known extension so extract can fail."""
    source = plant(tmp_path, "alice/paper.pdf", UNKNOWN_BODY)
    assert route(source) is ExtractionPath.PDF


def test_unknown_bytes_with_an_xlsx_extension_route_to_excel(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/model.xlsx", UNKNOWN_BODY)
    assert route(source) is ExtractionPath.EXCEL


def test_unknown_bytes_with_a_png_extension_route_to_image(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/chart.png", UNKNOWN_BODY)
    assert route(source) is ExtractionPath.IMAGE


@pytest.mark.parametrize(
    "relative",
    (
        "alice/photo.jpeg",
        "alice/photo.jpg",
        "alice/anim.gif",
        "alice/shot.webp",
    ),
)
def test_unknown_bytes_with_a_raster_extension_route_to_image(
    tmp_path: Path, relative: str
) -> None:
    source = plant(tmp_path, relative, UNKNOWN_BODY)
    assert route(source) is ExtractionPath.IMAGE


# --------------------------------------------------------------------------
# Requirement 2.1: content sniff wins over a misleading extension
# --------------------------------------------------------------------------


def test_pdf_magic_in_a_markdown_file_routes_to_pdf(tmp_path: Path) -> None:
    """Observable: a file with a misleading extension is routed by its content."""
    source = plant(tmp_path, "alice/notes.md", PDF_HEADER)
    assert route(source) is ExtractionPath.PDF


def test_pdf_magic_in_a_text_file_routes_to_pdf(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/notes.txt", PDF_HEADER)
    assert route(source) is ExtractionPath.PDF


def test_zip_signature_in_a_pdf_named_file_routes_to_excel(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/model.pdf", ZIP_HEADER)
    assert route(source) is ExtractionPath.EXCEL


def test_image_header_in_a_text_file_routes_to_image(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/chart.txt", PNG_1X1)
    assert route(source) is ExtractionPath.IMAGE


def test_pdf_magic_is_preferred_to_an_image_extension(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/scan.png", PDF_HEADER)
    assert route(source) is ExtractionPath.PDF


def test_zip_signature_is_preferred_to_an_image_extension(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/chart.png", ZIP_HEADER)
    assert route(source) is ExtractionPath.EXCEL


# --------------------------------------------------------------------------
# Requirement 2.4: unsupported type is an omission with its path; never raised
# --------------------------------------------------------------------------


def test_unknown_type_is_recorded_as_unsupported_with_its_path(
    tmp_path: Path,
) -> None:
    """Observable: an unknown type appears with its path, and the run does not fail."""
    source = plant(tmp_path, "alice/weird.xyz", UNKNOWN_BODY)
    try:
        result = route(source)
    except IngestError as exc:  # pragma: no cover - the assertion is the point
        pytest.fail(f"unsupported type was raised rather than omitted: {exc}")

    assert isinstance(result, Omission)
    assert result.category is OmissionCategory.UNSUPPORTED
    assert result.path == source.path
    assert result.reason
    assert result.missing_capability is None


def test_a_file_with_no_extension_and_unknown_bytes_is_unsupported(
    tmp_path: Path,
) -> None:
    source = plant(tmp_path, "alice/README", UNKNOWN_BODY)
    result = route(source)
    assert isinstance(result, Omission)
    assert result.category is OmissionCategory.UNSUPPORTED
    assert result.path == source.path


# --------------------------------------------------------------------------
# 3.1: SourceFile.path is always \\\\?\\-prefixed; the router opens that form
# --------------------------------------------------------------------------


def test_router_opens_the_windows_long_path_form(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/notes.txt", TEXT_BODY)
    assert str(source.path).startswith(LONG_PREFIX)
    assert route(source) is ExtractionPath.TEXT


def test_a_file_behind_a_path_longer_than_the_traditional_limit_is_routed(
    tmp_path: Path,
) -> None:
    root = as_windows_long_path(tmp_path / "archive")
    root.mkdir(parents=True, exist_ok=True)
    directory = root
    name = "notes.txt"
    while traditional_length(directory / name) <= TRADITIONAL_LIMIT:
        directory = directory / ("subdir" * 4)
        directory.mkdir(parents=True, exist_ok=True)
        if traditional_length(directory) > 400:
            pytest.fail(
                f"cannot construct a path longer than {TRADITIONAL_LIMIT} under {root}"
            )
    target = directory / name
    target.write_bytes(TEXT_BODY)
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
    assert route(source) is ExtractionPath.TEXT


# --------------------------------------------------------------------------
# Boundary: tokens only; no extract adapters, vision, or httpx
# --------------------------------------------------------------------------


def test_route_does_not_import_extract_vision_or_httpx() -> None:
    imported: list[str] = []
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            assert node.level == 0, "a relative import still reaches a rightward module"

    for forbidden in FORBIDDEN_IMPORTS:
        assert all(
            name != forbidden and not name.startswith(f"{forbidden}.")
            for name in imported
        ), imported


def test_route_module_does_not_name_extractor_adapters() -> None:
    """The token-to-adapter table belongs to the pipeline, not the router."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "Extractor" not in source
    assert "npu_rag.ingest.extract" not in source
