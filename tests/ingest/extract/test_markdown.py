"""Unit tests for the Markdown extractor (task 4.2).

Requirement 3.1: front matter, HTML and image syntax are dropped; heading
structure and reading order are kept. Requirement 3.2: each image reference
is an ImageRef anchored at the enclosing block's line range and ordinal.
Requirement 3.5: a declared title is recorded (YAML ``title:``, else the
first heading). Locators are 1-based inclusive; markdown-it maps are not
passed through raw.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.extract.base import Extractor
from npu_rag.ingest.extract.markdown import MarkdownExtractor
from npu_rag.ingest.types import (
    Extracted,
    ImageRef,
    MarkdownLocator,
    ProseSegment,
    SourceFile,
    TableSegment,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "src" / "npu_rag" / "ingest"
MARKDOWN_PATH = PACKAGE_ROOT / "extract" / "markdown.py"
FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "markdown"
)
DOCUMENT_PATH = FIXTURE_DIR / "document.md"
PLAIN_PNG = FIXTURE_DIR / "images" / "plain.png"
ALT_PNG = FIXTURE_DIR / "images" / "with-alt.png"

LONG_PREFIX = "\\\\?\\"
TRADITIONAL_LIMIT = 260
BUDGET = 256

FORBIDDEN_IMPORTS = (
    "npu_rag.ingest.vision",
    "npu_rag.ingest.state",
    "httpx",
)

MARKUP_FRAGMENTS = (
    "![",
    "](",
    "<div",
    "</div>",
    "<img",
    "---",
    "|",
    "# Overview",
    "## Foo",
    "### Bar",
)

HTML_INNER = "This HTML should vanish"
FRONT_MATTER_TITLE = "Fixture Title"
ALT_TEXT = "revenue chart"

PLAIN_SIZE = (4, 3)
ALT_SIZE = (8, 6)


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
    path = DOCUMENT_PATH.resolve()
    return SourceFile(
        path=as_windows_long_path(path),
        root=as_windows_long_path(path.parent),
        relative_path=Path("alice") / "document.md",
        author="alice",
    )


def extract(source: SourceFile) -> Extracted:
    extractor: Extractor = MarkdownExtractor()
    return extractor.extract(source, make_config())


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


def texts_of(result: Extracted) -> list[str]:
    return [segment.text for segment in result.segments if hasattr(segment, "text")]


# --------------------------------------------------------------------------
# Fixture non-vacuity
# --------------------------------------------------------------------------


def test_the_committed_fixture_still_has_front_matter_html_headings_table_and_images() -> None:
    raw = DOCUMENT_PATH.read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    assert "title: Fixture Title" in raw
    assert "<div class=\"note\">" in raw
    assert HTML_INNER in raw
    assert "# Overview" in raw
    assert "## Foo" in raw
    assert "### Bar" in raw
    assert "| Name | Value |" in raw
    assert "![](images/plain.png)" in raw
    assert f"![{ALT_TEXT}](images/with-alt.png)" in raw
    assert PLAIN_PNG.is_file()
    assert ALT_PNG.is_file()


# --------------------------------------------------------------------------
# Requirement 3.1, 3.2, 3.5: the fixture observable
# --------------------------------------------------------------------------


def test_the_markdown_fixture_yields_heading_paths_one_table_and_anchored_images() -> None:
    result = extract(source_for_fixture())

    assert result.title == FRONT_MATTER_TITLE
    assert result.title != "Overview"
    assert result.omissions == ()
    assert len(result.segments) == 9

    intro, plain_ref, intro_tail, foo, bar, alt_ref, bar_tail, table, closing = (
        result.segments
    )

    assert isinstance(intro, ProseSegment)
    assert intro.text == "Intro paragraph with an image"
    assert intro.heading_path == ("Overview",)
    assert intro.locator == MarkdownLocator(line_range=(8, 8), ordinal=0)

    assert isinstance(plain_ref, ImageRef)
    assert plain_ref.data == PLAIN_PNG.read_bytes()
    assert plain_ref.mime == "image/png"
    assert (plain_ref.width, plain_ref.height) == PLAIN_SIZE
    assert plain_ref.locator == MarkdownLocator(line_range=(8, 8), ordinal=0)
    assert plain_ref.chart_ranges is None

    assert isinstance(intro_tail, ProseSegment)
    assert intro_tail.text == "and more text."
    assert intro_tail.heading_path == ("Overview",)
    assert intro_tail.locator == MarkdownLocator(line_range=(8, 8), ordinal=0)

    assert isinstance(foo, ProseSegment)
    assert foo.text == "Prose under Foo."
    assert foo.heading_path == ("Overview", "Foo")
    assert foo.locator == MarkdownLocator(line_range=(16, 16), ordinal=0)

    assert isinstance(bar, ProseSegment)
    assert bar.text == "Prose under Bar with"
    assert bar.heading_path == ("Overview", "Foo", "Bar")
    assert bar.locator == MarkdownLocator(line_range=(20, 20), ordinal=0)

    assert isinstance(alt_ref, ImageRef)
    assert alt_ref.data == ALT_PNG.read_bytes()
    assert alt_ref.mime == "image/png"
    assert (alt_ref.width, alt_ref.height) == ALT_SIZE
    assert alt_ref.locator == MarkdownLocator(line_range=(20, 20), ordinal=0)
    assert alt_ref.chart_ranges is None

    assert isinstance(bar_tail, ProseSegment)
    assert bar_tail.text == "nearby."
    assert bar_tail.heading_path == ("Overview", "Foo", "Bar")
    assert bar_tail.locator == MarkdownLocator(line_range=(20, 20), ordinal=0)

    assert isinstance(table, TableSegment)
    assert table.text == "Name Value alpha 1 beta 2"
    assert table.locator == MarkdownLocator(line_range=(22, 25), ordinal=0)

    assert isinstance(closing, ProseSegment)
    assert closing.text == "Closing paragraph."
    assert closing.heading_path == ("Overview", "Foo", "Bar")
    assert closing.locator == MarkdownLocator(line_range=(27, 27), ordinal=0)

    tables = [segment for segment in result.segments if isinstance(segment, TableSegment)]
    assert len(tables) == 1
    images = [segment for segment in result.segments if isinstance(segment, ImageRef)]
    assert len(images) == 2


def test_emitted_text_contains_no_markup_front_matter_html_or_image_syntax() -> None:
    result = extract(source_for_fixture())
    emitted = texts_of(result)
    assert emitted
    for text in emitted:
        for fragment in MARKUP_FRAGMENTS:
            assert fragment not in text, (fragment, text)
        assert HTML_INNER not in text
        assert FRONT_MATTER_TITLE not in text
        assert ALT_TEXT not in text
        assert "author: alice" not in text
        assert "plain.png" not in text
        assert "with-alt.png" not in text


def test_image_locators_are_one_based_inclusive_not_raw_token_maps() -> None:
    """markdown-it token.map is 0-based exclusive; locators must not pass it through."""
    result = extract(source_for_fixture())
    images = [segment for segment in result.segments if isinstance(segment, ImageRef)]
    assert isinstance(images[0].locator, MarkdownLocator)
    assert isinstance(images[1].locator, MarkdownLocator)
    assert images[0].locator.line_range == (8, 8)
    assert images[0].locator.line_range != (7, 8)
    assert images[1].locator.line_range == (20, 20)
    assert images[1].locator.line_range != (19, 20)
    table = next(
        segment for segment in result.segments if isinstance(segment, TableSegment)
    )
    assert isinstance(table.locator, MarkdownLocator)
    assert table.locator.line_range == (22, 25)
    assert table.locator.line_range != (21, 25)


def test_markdown_extractor_opens_the_windows_long_path_form() -> None:
    source = source_for_fixture()
    assert str(source.path).startswith(LONG_PREFIX)
    result = extract(source)
    assert result.title == FRONT_MATTER_TITLE
    assert any(isinstance(segment, TableSegment) for segment in result.segments)


def test_a_markdown_file_behind_a_path_longer_than_the_traditional_limit_is_extracted(
    tmp_path: Path,
) -> None:
    root = as_windows_long_path(tmp_path / "archive")
    root.mkdir(parents=True, exist_ok=True)
    directory = root
    name = "notes.md"
    while traditional_length(directory / name) <= TRADITIONAL_LIMIT:
        directory = directory / ("subdir" * 4)
        directory.mkdir(parents=True, exist_ok=True)
        if traditional_length(directory) > 400:
            pytest.fail(
                f"cannot construct a path longer than {TRADITIONAL_LIMIT} under {root}"
            )
    target = directory / name
    target.write_text("# Deep\n\nHello from far away.\n", encoding="utf-8")
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
    assert result.title == "Deep"
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert isinstance(segment, ProseSegment)
    assert segment.text == "Hello from far away."
    assert segment.heading_path == ("Deep",)


# --------------------------------------------------------------------------
# Requirement 3.5: title fallback
# --------------------------------------------------------------------------


def test_title_falls_back_to_the_first_heading_when_front_matter_has_none(
    tmp_path: Path,
) -> None:
    source = plant(
        tmp_path,
        "alice/notes.md",
        b"---\nauthor: bob\n---\n\n# First Heading\n\nBody.\n",
    )
    result = extract(source)
    assert result.title == "First Heading"


def test_title_is_none_when_the_document_declares_none(tmp_path: Path) -> None:
    source = plant(tmp_path, "alice/notes.md", b"Just prose, no heading.\n")
    result = extract(source)
    assert result.title is None
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert isinstance(segment, ProseSegment)
    assert segment.text == "Just prose, no heading."
    assert segment.heading_path == ()


# --------------------------------------------------------------------------
# Missing image: pinned as ExtractionError, never a silent skip
# --------------------------------------------------------------------------


def test_a_missing_image_file_raises_extraction_error_with_stage_and_path(
    tmp_path: Path,
) -> None:
    source = plant(
        tmp_path,
        "alice/notes.md",
        b"# Doc\n\n![gone](images/missing.png)\n",
    )
    with pytest.raises(ExtractionError) as caught:
        extract(source)
    error = caught.value
    assert error.stage == "extraction"
    assert error.path == source.path
    assert "missing.png" in error.message
    assert str(source.path).startswith(LONG_PREFIX)


def test_two_images_in_one_block_receive_distinct_ordinals(tmp_path: Path) -> None:
    images = tmp_path / "alice" / "images"
    images.mkdir(parents=True)
    shutil.copy(PLAIN_PNG, images / "plain.png")
    shutil.copy(ALT_PNG, images / "with-alt.png")
    source = plant(
        tmp_path,
        "alice/notes.md",
        b"![one](images/plain.png) ![two](images/with-alt.png)\n",
    )
    result = extract(source)
    refs = [segment for segment in result.segments if isinstance(segment, ImageRef)]
    assert len(refs) == 2
    assert isinstance(refs[0].locator, MarkdownLocator)
    assert isinstance(refs[1].locator, MarkdownLocator)
    assert refs[0].locator.line_range == (1, 1)
    assert refs[1].locator.line_range == (1, 1)
    assert refs[0].locator.ordinal == 0
    assert refs[1].locator.ordinal == 1
    assert refs[0].data == PLAIN_PNG.read_bytes()
    assert refs[1].data == ALT_PNG.read_bytes()


# --------------------------------------------------------------------------
# Requirement 3.2: image tokens in table cells and headings, not only paragraphs
# --------------------------------------------------------------------------


def test_an_image_in_a_table_cell_is_emitted_as_image_ref(tmp_path: Path) -> None:
    """``_visible_text`` skips ``image`` tokens; a cell image must still be an ImageRef."""
    images = tmp_path / "alice" / "images"
    images.mkdir(parents=True)
    shutil.copy(PLAIN_PNG, images / "plain.png")
    source = plant(
        tmp_path,
        "alice/notes.md",
        b"| H |\n|---|\n| ![x](images/plain.png) |\n",
    )
    result = extract(source)
    refs = [segment for segment in result.segments if isinstance(segment, ImageRef)]
    assert len(refs) == 1
    ref = refs[0]
    assert ref.data == PLAIN_PNG.read_bytes()
    assert ref.mime == "image/png"
    assert (ref.width, ref.height) == PLAIN_SIZE
    assert isinstance(ref.locator, MarkdownLocator)
    assert ref.locator.line_range == (1, 3)
    assert ref.locator.line_range != (0, 3)
    assert ref.locator.ordinal == 0
    tables = [segment for segment in result.segments if isinstance(segment, TableSegment)]
    assert len(tables) == 1
    assert tables[0].text == "H"
    assert "![" not in tables[0].text
    assert "plain.png" not in tables[0].text
    assert "x" not in tables[0].text


def test_an_image_in_a_heading_is_emitted_as_image_ref(tmp_path: Path) -> None:
    """Heading inlines used ``_visible_text`` only; a heading image must still be an ImageRef."""
    images = tmp_path / "alice" / "images"
    images.mkdir(parents=True)
    shutil.copy(PLAIN_PNG, images / "plain.png")
    source = plant(
        tmp_path,
        "alice/notes.md",
        b"# Title ![x](images/plain.png)\n\nBody.\n",
    )
    result = extract(source)
    refs = [segment for segment in result.segments if isinstance(segment, ImageRef)]
    assert len(refs) == 1
    ref = refs[0]
    assert ref.data == PLAIN_PNG.read_bytes()
    assert ref.mime == "image/png"
    assert (ref.width, ref.height) == PLAIN_SIZE
    assert isinstance(ref.locator, MarkdownLocator)
    assert ref.locator.line_range == (1, 1)
    assert ref.locator.line_range != (0, 1)
    assert ref.locator.ordinal == 0
    assert result.title == "Title"
    assert result.segments[0] is ref
    prose = [segment for segment in result.segments if isinstance(segment, ProseSegment)]
    assert len(prose) == 1
    assert prose[0].text == "Body."
    assert prose[0].heading_path == ("Title",)
    emitted = texts_of(result)
    for text in emitted:
        assert "![" not in text
        assert "plain.png" not in text
        assert "x" not in text


# --------------------------------------------------------------------------
# Layer guard: extract never imports vision, state, or httpx
# --------------------------------------------------------------------------


def test_markdown_extractor_does_not_import_vision_state_or_httpx() -> None:
    names = imported_names(MARKDOWN_PATH)
    for forbidden in FORBIDDEN_IMPORTS:
        assert all(
            name != forbidden and not name.startswith(f"{forbidden}.")
            for name in names
        ), names
