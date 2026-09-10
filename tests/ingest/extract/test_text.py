"""Unit tests for extract/base normalisation and the plain-text adapter (task 4.1).

Requirement 3.3: Unicode representation is normalised and redundant whitespace
is collapsed without altering the words themselves. design.md's Extraction
Service Interface: one Extractor protocol, Extracted re-exported from types,
plain text as the smallest conforming adapter.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.extract.base import Extracted as BaseExtracted
from npu_rag.ingest.extract.base import Extractor, normalise
from npu_rag.ingest.extract.text import TextExtractor
from npu_rag.ingest.types import (
    Extracted,
    ImageFileLocator,
    MarkdownLocator,
    ProseSegment,
    SourceFile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "src" / "npu_rag" / "ingest"
BASE_PATH = PACKAGE_ROOT / "extract" / "base.py"
TEXT_PATH = PACKAGE_ROOT / "extract" / "text.py"
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "text" / "mixed_unicode.txt"
)

LONG_PREFIX = "\\\\?\\"
TRADITIONAL_LIMIT = 260
BUDGET = 256

FORBIDDEN_IMPORTS = (
    "npu_rag.ingest.vision",
    "npu_rag.ingest.state",
    "httpx",
)

# Requirement 3.3 observable: mixed NFC/NFD plus redundant space/tab/newlines
# folds to this literal, with every word intact (café, don't, foo, bar, café).
EXPECTED_NORMALISED = "caf\u00e9 don't foo bar caf\u00e9"

# The committed fixture must keep both Unicode forms and the redundant
# whitespace; an editor that NFC-folds the file would make the observable
# vacuous.
FIXTURE_NFC_CAFE = "caf\u00e9".encode("utf-8")
FIXTURE_NFD_CAFE = "cafe\u0301".encode("utf-8")
FIXTURE_BYTES = FIXTURE_NFC_CAFE + b"   don't\t\tfoo  bar\n\n" + FIXTURE_NFD_CAFE + b"\n"


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


def make_config() -> IngestConfig:
    return IngestConfig(roots=(Path("inputs"),), token_budget=BUDGET)


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
        relative_path=Path("alice") / "mixed_unicode.txt",
        author="alice",
    )


def extract(source: SourceFile) -> Extracted:
    extractor: Extractor = TextExtractor()
    return extractor.extract(source, make_config())


def prose_text(result: Extracted) -> str:
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert isinstance(segment, ProseSegment)
    return segment.text


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
# Requirement 3.3: the fixture observable
# --------------------------------------------------------------------------


def test_the_committed_fixture_still_mixes_unicode_forms_and_whitespace() -> None:
    """Non-vacuity: the file on disk has not been NFC-folded or whitespace-collapsed."""
    raw = FIXTURE_PATH.read_bytes()
    assert raw == FIXTURE_BYTES
    assert FIXTURE_NFC_CAFE in raw
    assert FIXTURE_NFD_CAFE in raw
    assert FIXTURE_NFC_CAFE != FIXTURE_NFD_CAFE
    assert b"   " in raw
    assert b"\t\t" in raw
    assert b"foo  bar" in raw
    assert b"don't" in raw
    assert b"\n\n" in raw


def test_the_mixed_unicode_fixture_normalises_to_the_literal_expected_string() -> None:
    """Observable: mixed Unicode forms and redundant whitespace, every word intact."""
    raw = FIXTURE_PATH.read_text(encoding="utf-8")
    assert normalise(raw) == EXPECTED_NORMALISED
    assert EXPECTED_NORMALISED == "caf\u00e9 don't foo bar caf\u00e9"
    for word in ("caf\u00e9", "don't", "foo", "bar"):
        assert word in EXPECTED_NORMALISED
    assert "foobar" not in EXPECTED_NORMALISED
    assert "dont" not in EXPECTED_NORMALISED.split()
    assert "cafe\u0301" not in EXPECTED_NORMALISED


def test_normalise_composes_nfd_marks_and_collapses_whitespace_without_joining_words() -> None:
    assert normalise("cafe\u0301") == "caf\u00e9"
    assert normalise("foo  \t\n  bar") == "foo bar"
    assert normalise("don't") == "don't"
    assert normalise("running") == "running"


# --------------------------------------------------------------------------
# Plain-text adapter: smallest Extractor
# --------------------------------------------------------------------------


def test_text_extractor_emits_one_normalised_prose_segment_from_the_fixture() -> None:
    result = extract(source_for_fixture())

    assert result.title is None
    assert result.omissions == ()
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert isinstance(segment, ProseSegment)
    assert segment.text == EXPECTED_NORMALISED
    assert isinstance(segment.locator, MarkdownLocator)
    assert not isinstance(segment.locator, ImageFileLocator)
    assert segment.locator.line_range == (1, 3)
    assert segment.locator.ordinal == 0
    assert segment.heading_path == ()


def test_text_extractor_opens_the_windows_long_path_form() -> None:
    source = source_for_fixture()
    assert str(source.path).startswith(LONG_PREFIX)
    result = extract(source)
    assert prose_text(result) == EXPECTED_NORMALISED


def test_a_file_behind_a_path_longer_than_the_traditional_limit_is_extracted(
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
    target.write_bytes(FIXTURE_BYTES)
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
    assert prose_text(result) == EXPECTED_NORMALISED


def test_whitespace_only_text_emits_no_segments(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/blank.txt", b"  \n\t\n  ")
    result = extract(source)
    assert result.title is None
    assert result.segments == ()
    assert result.omissions == ()


def test_undecodable_bytes_raise_extraction_error_with_stage_and_path(
    tmp_path: Path,
) -> None:
    """Requirement 8.4's undecodable path: ExtractionError, stage and path set."""
    source = plant(tmp_path, "alice/notes.txt", b"\xff\xfe not utf-8")
    with pytest.raises(ExtractionError) as caught:
        extract(source)
    error = caught.value
    assert error.stage == "extraction"
    assert error.path == source.path
    assert str(source.path).startswith(LONG_PREFIX)


# --------------------------------------------------------------------------
# Extracted is the types.Extracted; protocol lives in extract.base
# --------------------------------------------------------------------------


def test_extracted_is_reexported_from_types_and_not_redefined() -> None:
    assert BaseExtracted is Extracted
    tree = ast.parse(BASE_PATH.read_text(encoding="utf-8"))
    defined = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]
    assert "Extracted" not in defined
    assert "Extractor" in defined


# --------------------------------------------------------------------------
# Layer guard: extract never imports vision, state, or httpx
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", [BASE_PATH, TEXT_PATH], ids=["base.py", "text.py"])
def test_extract_modules_do_not_import_vision_state_or_httpx(path: Path) -> None:
    names = imported_names(path)
    for forbidden in FORBIDDEN_IMPORTS:
        assert all(
            name != forbidden and not name.startswith(f"{forbidden}.")
            for name in names
        ), names
