"""Unit tests for source-root discovery (task 3.1).

Requirements 1.1–1.6: configurable roots, recursive walk, include/exclude,
unreadable roots reported as omissions while the run continues, long and
non-ASCII paths, overlapping roots de-duplicated. Requirement 7.2: author
is the first directory component beneath the file's root.

``discover.py`` sits to the right of state and to the left of route. These
tests never import vision, extract, or state.
"""

from __future__ import annotations

import ast
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.discover import Discoverer, Discovery
from npu_rag.ingest.errors import DiscoveryError
from npu_rag.ingest.types import OmissionCategory, SourceFile

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "ingest" / "discover.py"
)

BUDGET = 256
TRADITIONAL_LIMIT = 260
LONG_PREFIX = "\\\\?\\"

FORBIDDEN_IMPORTS = (
    "npu_rag.ingest.vision",
    "npu_rag.ingest.extract",
    "npu_rag.ingest.state",
    "npu_rag.ingest.route",
    "npu_rag.ingest.pipeline",
    "httpx",
)


def _config(
    *roots: Path,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> IngestConfig:
    return IngestConfig(
        roots=roots,
        token_budget=BUDGET,
        include=include,
        exclude=exclude,
    )


def discover(
    *roots: Path,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> Discovery:
    return Discoverer().discover(_config(*roots, include=include, exclude=exclude))


def write_file(root: Path, relative: str, content: str = "x") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def as_windows_long_path(path: Path) -> Path:
    text = _without_long_prefix(path if path.is_absolute() else path.resolve())
    if text.startswith("\\\\"):
        return Path(LONG_PREFIX + "UNC\\" + text[2:])
    return Path(LONG_PREFIX + text)


def _without_long_prefix(path: Path | str) -> str:
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    if text.startswith("\\\\?\\"):
        return text[4:]
    return text


def path_key(path: Path) -> str:
    return os.path.normcase(_without_long_prefix(path.resolve()))


def find_source(result: Discovery, planted: Path) -> SourceFile | None:
    key = path_key(planted)
    for source in result.files:
        if path_key(source.path) == key:
            return source
    return None


def traditional_length(path: Path) -> int:
    return len(_without_long_prefix(path))


@pytest.fixture
def archive(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "archive"
    root.mkdir()
    try:
        yield root
    finally:
        long_root = as_windows_long_path(root)
        if long_root.exists():
            shutil.rmtree(long_root, ignore_errors=True)


# --------------------------------------------------------------------------
# Requirement 1.2: recursive walk, files only
# --------------------------------------------------------------------------


def test_discover_walks_every_configured_root_recursively(archive: Path) -> None:
    nested = write_file(archive, "alice/deep/nested.md")
    sibling_root = archive.parent / "models"
    other = write_file(sibling_root, "bob/sheet.xlsx")

    result = discover(archive, sibling_root)

    assert find_source(result, nested) is not None
    assert find_source(result, other) is not None
    assert result.omissions == ()


def test_directories_are_not_emitted_as_source_files(archive: Path) -> None:
    directory = archive / "alice"
    directory.mkdir()
    notes = write_file(archive, "alice/notes.md")

    result = discover(archive)

    assert find_source(result, notes) is not None
    assert all(path_key(source.path) != path_key(directory) for source in result.files)
    assert all(source.path.is_file() for source in result.files)


# --------------------------------------------------------------------------
# Requirement 1.6: overlapping roots de-duplicated by resolved path
# --------------------------------------------------------------------------


def test_two_overlapping_roots_yield_one_source_file_per_physical_file(
    archive: Path,
) -> None:
    """The surviving SourceFile keeps the first root in config order."""
    outer_only = write_file(archive, "alice/outer.md")
    shared = write_file(archive, "shared/both.md")
    inner = archive / "shared"
    inner_only = write_file(inner, "inner.md")

    result = discover(archive, inner)

    assert find_source(result, outer_only) is not None
    assert find_source(result, inner_only) is not None
    source = find_source(result, shared)
    assert source is not None
    assert path_key(source.root) == path_key(archive)
    assert source.relative_path.as_posix() == "shared/both.md"
    assert sum(1 for item in result.files if path_key(item.path) == path_key(shared)) == 1
    assert len(result.files) == 3


def test_the_later_overlapping_root_does_not_replace_the_first(archive: Path) -> None:
    shared = write_file(archive, "shared/both.md")
    inner = archive / "shared"

    first_inner = discover(inner, archive)
    source = find_source(first_inner, shared)
    assert source is not None
    assert path_key(source.root) == path_key(inner)
    assert source.relative_path.as_posix() == "both.md"


# --------------------------------------------------------------------------
# Requirement 1.5: long and non-ASCII paths
# --------------------------------------------------------------------------


def _plant_long_non_ascii_file(root: Path) -> Path:
    long_root = as_windows_long_path(root)
    long_root.mkdir(parents=True, exist_ok=True)
    directory = long_root / ("文档" * 80)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "笔记"
    name = stem + ".md"
    target = directory / name
    while traditional_length(target) <= TRADITIONAL_LIMIT:
        name = stem + name
        if len(name) > 255:
            raise AssertionError(
                f"cannot construct a path longer than {TRADITIONAL_LIMIT} under {root}"
            )
        target = directory / name
    target.write_text("long-non-ascii", encoding="utf-8")
    return target


def test_a_non_ascii_path_longer_than_the_traditional_limit_is_enumerated(
    archive: Path,
) -> None:
    planted = _plant_long_non_ascii_file(archive)
    assert traditional_length(planted) > TRADITIONAL_LIMIT
    assert any(ord(character) > 127 for character in str(planted))

    result = discover(archive)

    source = find_source(result, planted)
    assert source is not None
    assert traditional_length(source.path) > TRADITIONAL_LIMIT
    assert any(ord(character) > 127 for character in str(source.path))
    assert source.path.read_text(encoding="utf-8") == "long-non-ascii"
    assert "文档" in source.relative_path.as_posix() or "文档" in str(source.path)


# --------------------------------------------------------------------------
# Requirement 1.4: missing / unreadable root is an omission; others continue
# --------------------------------------------------------------------------


def test_a_missing_root_produces_one_root_unavailable_omission_and_others_are_walked(
    archive: Path,
) -> None:
    kept = write_file(archive, "alice/notes.md")
    missing = archive.parent / "does-not-exist"

    result = discover(missing, archive)

    assert find_source(result, kept) is not None
    assert len(result.omissions) == 1
    omission = result.omissions[0]
    assert omission.category is OmissionCategory.ROOT_UNAVAILABLE
    assert path_key(omission.path) == path_key(missing)
    assert omission.reason
    assert omission.missing_capability is None


def test_an_unreadable_file_root_is_an_omission_not_a_raised_error(
    archive: Path,
) -> None:
    kept = write_file(archive, "alice/notes.md")
    not_a_directory = archive.parent / "not-a-root.txt"
    not_a_directory.write_text("nope", encoding="utf-8")

    try:
        result = discover(not_a_directory, archive)
    except DiscoveryError as exc:  # pragma: no cover - the assertion is the point
        pytest.fail(f"unreadable root was raised rather than omitted: {exc}")

    assert find_source(result, kept) is not None
    assert len(result.omissions) == 1
    assert result.omissions[0].category is OmissionCategory.ROOT_UNAVAILABLE
    assert path_key(result.omissions[0].path) == path_key(not_a_directory)


# --------------------------------------------------------------------------
# Requirement 1.3: include / exclude applied before any extraction
# --------------------------------------------------------------------------


def test_include_rules_filter_enumerated_files(archive: Path) -> None:
    kept = write_file(archive, "alice/keep.md")
    skipped = write_file(archive, "alice/skip.txt")

    without_rule = discover(archive)
    assert find_source(without_rule, skipped) is not None
    assert find_source(without_rule, kept) is not None

    filtered = discover(archive, include=("**/*.md",))
    assert find_source(filtered, kept) is not None
    assert find_source(filtered, skipped) is None


def test_exclude_rules_filter_enumerated_files(archive: Path) -> None:
    kept = write_file(archive, "alice/keep.md")
    drafted = write_file(archive, "alice/drafts/secret.md")

    without_rule = discover(archive)
    assert find_source(without_rule, drafted) is not None

    filtered = discover(archive, exclude=("**/drafts/**",))
    assert find_source(filtered, kept) is not None
    assert find_source(filtered, drafted) is None


def test_exclude_wins_when_a_file_matches_include_and_exclude(archive: Path) -> None:
    kept = write_file(archive, "alice/keep.md")
    drafted = write_file(archive, "alice/drafts/secret.md")

    filtered = discover(archive, include=("**/*.md",), exclude=("**/drafts/**",))
    assert find_source(filtered, kept) is not None
    assert find_source(filtered, drafted) is None


def test_empty_include_means_include_all(archive: Path) -> None:
    markdown = write_file(archive, "alice/keep.md")
    text = write_file(archive, "alice/notes.txt")

    result = discover(archive, include=())
    assert find_source(result, markdown) is not None
    assert find_source(result, text) is not None


# --------------------------------------------------------------------------
# Requirement 7.2: author is the first directory beneath the root
# --------------------------------------------------------------------------


def test_author_is_the_first_directory_under_the_root(archive: Path) -> None:
    notes = write_file(archive, "alice/deep/notes.md")
    models = write_file(archive, "bob/sheet.xlsx")

    result = discover(archive)

    alice = find_source(result, notes)
    bob = find_source(result, models)
    assert alice is not None
    assert bob is not None
    assert alice.author == "alice"
    assert bob.author == "bob"
    assert alice.relative_path.as_posix() == "alice/deep/notes.md"


def test_a_file_directly_in_the_root_has_an_empty_author(archive: Path) -> None:
    stray = write_file(archive, "readme.md")

    result = discover(archive)

    source = find_source(result, stray)
    assert source is not None
    assert source.author == ""
    assert source.relative_path.as_posix() == "readme.md"


# --------------------------------------------------------------------------
# Requirement 1.1: no hardcoded archive path
# --------------------------------------------------------------------------


def test_discover_module_contains_no_hardcoded_source_root() -> None:
    """Requirement 1.1: no source root path as a fixed value."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "substack" not in lowered
    assert "d:\\source" not in lowered
    assert "d:/source" not in lowered


# --------------------------------------------------------------------------
# Boundary guard
# --------------------------------------------------------------------------


def test_discover_does_not_import_vision_extract_or_state() -> None:
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
