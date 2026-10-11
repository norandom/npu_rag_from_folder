"""Tests for `npu_rag.cli.render` (task 3.1).

Requirements under test: 8.1 (styled UTF-8 terminal output), 8.2 (automatic
ANSI stripping for a non-TTY or `NO_COLOR`), 8.3 (`--json`), 8.4 (UTF-8 stdout
on Windows), 4.4 (hierarchical breadcrumb attribution) and 6.3 (provider
attribution in *both* outputs). Design anchor: §3.4, including its worked
sample output block.

Three standing lessons from this spec's implementation notes shape the file.

1. **Every table is pinned against a separate hand-written literal** and given
   a *structural* completeness assertion, so that adding a locator kind, an
   SGR name or a JSON key without a test fails loudly rather than silently
   (notes 1.1, 2.2).
2. **No vacuous fixtures.** The ANSI-stripping assertions run on output that
   genuinely contains escape bytes - the renderer's own styling plus escapes
   planted inside the result *data* - never on a string that never had any.
3. **Non-ASCII means real non-ASCII**: Devanagari (what design §3.4's sample
   is), CJK and German umlauts. Requirement 8.4 exists because of a charmap
   crash on this platform, and an ASCII placeholder cannot see it.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import npu_rag.cli.render as render
from npu_rag.cli.render import (
    BREADCRUMB_SEPARATOR,
    CPU_PROVIDER,
    HIT_MARKER,
    JSON_INDENT,
    LOCATOR_FORMATTERS,
    NO_COLOR_VARIABLE,
    NO_RESULTS_LINE,
    NPU_PROVIDER,
    RULE_WIDTH,
    SGR,
    SNIPPET_LIMIT,
    UNKNOWN_DOCUMENT,
    UNKNOWN_LOCATOR,
    emit,
    ensure_utf8_stdout,
    format_breadcrumb,
    format_locator,
    format_provider,
    render_search,
    render_search_json,
    render_search_text,
    should_colorize,
    strip_ansi,
    style,
)
from npu_rag.index.types import SearchMode, SearchResponse, SearchResult

# --------------------------------------------------------------------------- #
# Fixture vocabulary - real archive shapes, real scripts
# --------------------------------------------------------------------------- #

#: What `ingest.discover._windows_long_path` stamps on every absolute path, and
#: what `index/schema.py:76` therefore stores verbatim in `file_path`. Measured
#: 7/7 on real discovered paths (implementation note 2.2), so this - not a bare
#: `D:\...` - is the realistic fixture.
ARCHIVE_ROOT = r"\\?\D:\Source\substack_dl\archive"
UNC_ARCHIVE_ROOT = r"\\?\UNC\nas\share\archive"

FIRST_DAY = r"\\?\D:\Source\substack_dl\archive\Bhagwati Ugra Tara Sadhana\Savarni Nath - First Day.md"
SECOND_DAY = r"\\?\D:\Source\substack_dl\archive\Bhagwati Ugra Tara Sadhana\Savarni Nath - Second Day.md"
GARUDA = r"\\?\D:\Source\substack_dl\archive\Bhagwan Garuda Sadhana\Day 1.md"
SUBSTACK = r"\\?\D:\Source\substack_dl\archive\businessengineer.ai\20240402_084002_the-three-layers-of-ai.md"

#: Design §3.4's own sample text, verbatim. Requirement 8.4 names Devanagari
#: first for a reason: this is what the archive is full of.
DEVANAGARI = "पवित्रीकरण जल छिड़क लीजिए... ॐ अपवित्रः पवित्रो वा..."
DEVANAGARI_2 = "अब आचमनी में जल लीजिए और मेरे साथ कहिए ओम आम आम ह्रीम"
CJK = "全部的文件都在这里：第三章，水的净化"
GERMAN = "Grüße aus München - Übung macht den Meister, größer als Äpfel"

MARKDOWN_LOCATOR = json.dumps({"type": "markdown", "line_range": [45, 80], "ordinal": 0})
PAGE_LOCATOR = json.dumps({"type": "page", "page": 3})
SHEET_LOCATOR = json.dumps({"type": "sheet", "sheet": "Summary", "cell_range": "A1:C9"})
IMAGE_LOCATOR = json.dumps({"type": "image_file"})


def make_result(
    file_path: str,
    *,
    chunk_index: int = 12,
    score: float = 0.892,
    text: str = DEVANAGARI,
    locator_type: str = "markdown",
    locator_json: str = MARKDOWN_LOCATOR,
    title: str | None = None,
    chunk_kind: str = "prose",
    headings: tuple[str, ...] = (),
    expanded_context: str | None = None,
    token_count: int = 128,
    dense_distance: float | None = None,
    dense_rank: int | None = None,
    lexical_score: float | None = None,
    lexical_rank: int | None = None,
) -> SearchResult:
    """One `SearchResult` with the real column shapes task 3.1 must render."""
    return SearchResult(
        chunk_id=f"{file_path}#{chunk_index}",
        file_path=file_path,
        title=title,
        chunk_kind=chunk_kind,
        chunk_index=chunk_index,
        headings=headings,
        locator_type=locator_type,
        locator_json=locator_json,
        text=text,
        score=score,
        dense_distance=dense_distance,
        dense_rank=dense_rank,
        lexical_score=lexical_score,
        lexical_rank=lexical_rank,
        expanded_context=expanded_context,
        token_count=token_count,
    )


def make_response(
    results: tuple[SearchResult, ...],
    *,
    query: str = "ritual purification",
    mode: SearchMode = SearchMode.HYBRID,
    total_candidates: int = 57,
    elapsed_seconds: float = 0.1423,
) -> SearchResponse:
    return SearchResponse(
        query=query,
        mode=mode,
        results=results,
        total_candidates=total_candidates,
        elapsed_seconds=elapsed_seconds,
    )


# --------------------------------------------------------------------------- #
# Requirement 8.2 - ANSI stripping, pinned against hand-written literals
# --------------------------------------------------------------------------- #

#: Hand-written, one entry per escape family `strip_ansi` claims to remove.
#: Written out rather than generated: a table derived from the module's own
#: regex would restate the implementation instead of checking it.
ANSI_CASES: tuple[tuple[str, str], ...] = (
    ("\x1b[31mred\x1b[0m", "red"),
    ("\x1b[1;36;40mbold cyan on black\x1b[0m", "bold cyan on black"),
    ("\x1b[2J\x1b[Hcleared", "cleared"),
    ("\x1b[38;5;208m256 colour\x1b[39m", "256 colour"),
    ("\x1b]0;window title\x07kept", "kept"),
    ("\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\", "link"),
    ("\x1bMreverse index", "reverse index"),
    ("\x1b7saved\x1b8", "saved"),
    # Non-escapes that must SURVIVE: the bracket forms only look like CSI.
    ("[31mnot an escape[0m", "[31mnot an escape[0m"),
    ("a -> b [Chunk #12 | Lines 45-80]", "a -> b [Chunk #12 | Lines 45-80]"),
    # Requirement 8.4's scripts must pass through byte-for-byte.
    (DEVANAGARI, DEVANAGARI),
    (CJK, CJK),
    (GERMAN, GERMAN),
    (f"\x1b[33m{DEVANAGARI}\x1b[0m", DEVANAGARI),
)


@pytest.mark.parametrize(("raw", "expected"), ANSI_CASES)
def test_strip_ansi_matches_its_hand_written_table(raw: str, expected: str) -> None:
    assert strip_ansi(raw) == expected


def test_strip_ansi_removes_every_escape_byte_from_a_styled_composition() -> None:
    """Non-vacuous by construction: the input is built by `style()` itself."""
    styled = " ".join(style(name, name) for name in sorted(SGR) if name != "reset")
    assert "\x1b" in styled, "fixture must genuinely contain styling"
    stripped = strip_ansi(styled)
    assert "\x1b" not in stripped
    assert stripped == " ".join(sorted(name for name in SGR if name != "reset"))


def test_strip_ansi_is_idempotent() -> None:
    for raw, expected in ANSI_CASES:
        assert strip_ansi(strip_ansi(raw)) == expected


# --------------------------------------------------------------------------- #
# Requirement 8.1 - the SGR table, pinned and complete
# --------------------------------------------------------------------------- #

#: A separate hand-written copy. Note 1.1's lesson: a misspelled or transposed
#: code cannot be caught by resolving it against the thing being tested.
EXPECTED_SGR: dict[str, str] = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
}


def test_sgr_table_matches_the_literal_table() -> None:
    assert SGR == EXPECTED_SGR


def test_every_sgr_code_is_a_well_formed_single_csi_sequence() -> None:
    """The structural half, so a new entry cannot be a free-form string."""
    for name, code in SGR.items():
        assert re.fullmatch(r"\x1b\[\d+m", code), f"{name} -> {code!r}"
        assert strip_ansi(code) == "", f"{name} must be strippable"


def test_style_wraps_with_every_named_code_then_resets() -> None:
    assert style("x", "bold", "cyan") == "\x1b[1m\x1b[36mx\x1b[0m"


def test_style_disabled_returns_the_text_untouched() -> None:
    assert style("x", "bold", "cyan", enabled=False) == "x"


def test_style_without_codes_returns_the_text_untouched() -> None:
    assert style("x") == "x"


def test_style_rejects_an_unknown_code_name() -> None:
    with pytest.raises(ValueError, match="unknown style 'chartreuse'"):
        style("x", "chartreuse")


# --------------------------------------------------------------------------- #
# Requirement 8.2 - when colour is allowed
# --------------------------------------------------------------------------- #


class FakeStream(io.StringIO):
    """A stream whose TTY-ness is dictated, not inherited from the test runner."""

    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class BrokenStream(io.StringIO):
    """A detached stream: `isatty()` raises, as it does on a closed handle."""

    def isatty(self) -> bool:
        raise ValueError("I/O operation on closed file")


class StreamWithoutIsatty:
    """Not every object a caller passes as a stream implements `isatty`."""

    def write(self, text: str) -> int:  # pragma: no cover - never written to
        return len(text)


class TruthyNonBoolStream(io.StringIO):
    """A stream whose `isatty()` is truthy but is not a `bool`.

    `should_colorize` reaches `isatty` through `getattr`, which yields `Any`,
    so mypy cannot see a non-bool leaking out of a `-> bool` function. Only an
    identity assertion can.
    """

    def isatty(self) -> bool:
        return 1  # type: ignore[return-value]


def test_colour_on_a_tty_with_no_environment_objection() -> None:
    assert should_colorize(stream=FakeStream(tty=True), env={}) is True


def test_no_colour_when_stdout_is_not_a_tty() -> None:
    """Requirement 8.2's redirection case."""
    assert should_colorize(stream=FakeStream(tty=False), env={}) is False


def test_no_colour_when_the_no_color_variable_is_set() -> None:
    assert should_colorize(stream=FakeStream(tty=True), env={"NO_COLOR": "1"}) is False


def test_no_colour_variable_name_is_the_published_spelling() -> None:
    assert NO_COLOR_VARIABLE == "NO_COLOR"


@pytest.mark.parametrize("value", ["1", "0", "false", "anything at all", " x "])
def test_no_color_disables_regardless_of_its_value(value: str) -> None:
    assert should_colorize(stream=FakeStream(tty=True), env={"NO_COLOR": value}) is False


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_a_blank_no_color_is_absence_as_config_py_already_rules(value: str) -> None:
    """`config.py` treats an empty/whitespace env var as absence; so does this."""
    assert should_colorize(stream=FakeStream(tty=True), env={"NO_COLOR": value}) is True


def test_prefer_color_false_wins_over_a_tty() -> None:
    """The `--no-color` flag and `ansi_colors = false` path."""
    assert should_colorize(prefer_color=False, stream=FakeStream(tty=True), env={}) is False


def test_a_stream_without_isatty_is_not_a_tty() -> None:
    assert should_colorize(stream=StreamWithoutIsatty(), env={}) is False  # type: ignore[arg-type]


def test_a_stream_whose_isatty_raises_is_not_a_tty() -> None:
    assert should_colorize(stream=BrokenStream(), env={}) is False


def test_a_truthy_non_bool_isatty_still_yields_a_real_bool() -> None:
    """Identity, not truthiness: the declared return type must hold."""
    result = should_colorize(stream=TruthyNonBoolStream(), env={})
    assert result is True
    assert type(result) is bool


def test_should_colorize_reads_the_real_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert should_colorize(stream=FakeStream(tty=True)) is False
    monkeypatch.delenv("NO_COLOR")
    assert should_colorize(stream=FakeStream(tty=True)) is True


def test_should_colorize_defaults_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stdout", FakeStream(tty=False))
    assert should_colorize() is False
    monkeypatch.setattr(sys, "stdout", FakeStream(tty=True))
    assert should_colorize() is True


# --------------------------------------------------------------------------- #
# Requirement 8.4 - UTF-8 stdout on Windows, and the guard that makes it safe
# --------------------------------------------------------------------------- #


def cp1252_stream() -> io.TextIOWrapper:
    """A stdout as Windows actually hands it over: a cp1252 `TextIOWrapper`."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")


def test_a_cp1252_stdout_is_reconfigured_to_utf8() -> None:
    stream = cp1252_stream()
    assert stream.encoding == "cp1252"
    assert ensure_utf8_stdout(stream) == "utf-8"
    assert stream.encoding == "utf-8"


def test_the_reconfigure_is_what_prevents_the_charmap_crash() -> None:
    """The failure requirement 8.4 names, demonstrated and then prevented."""
    doomed = cp1252_stream()
    with pytest.raises(UnicodeEncodeError):
        doomed.write(DEVANAGARI)
        doomed.flush()

    saved = cp1252_stream()
    ensure_utf8_stdout(saved)
    saved.write(f"{DEVANAGARI} {CJK} {GERMAN}")
    saved.flush()
    buffer = saved.buffer
    assert isinstance(buffer, io.BytesIO)
    assert buffer.getvalue().decode("utf-8") == f"{DEVANAGARI} {CJK} {GERMAN}"


def test_the_guard_skips_a_stream_that_cannot_be_reconfigured() -> None:
    """Trap 1: `reconfigure` is absent on any non-`TextIOWrapper` stdout.

    Under pytest's capture, and for anything a caller substitutes, a bare
    `sys.stdout.reconfigure(...)` raises `AttributeError`. The guard must
    therefore be observable, not incidental - hence the return value.
    """
    plain = io.StringIO()
    assert not hasattr(plain, "reconfigure")
    assert ensure_utf8_stdout(plain) is None


def test_the_guard_is_structurally_present_in_the_source() -> None:
    source = Path(render.__file__).read_text(encoding="utf-8")
    assert "isinstance(target, io.TextIOWrapper)" in source


def test_ensure_utf8_stdout_is_idempotent() -> None:
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    assert ensure_utf8_stdout(stream) == "utf-8"
    assert ensure_utf8_stdout(stream) == "utf-8"


def test_ensure_utf8_stdout_defaults_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    assert ensure_utf8_stdout() == "utf-8"
    assert stream.encoding == "utf-8"


def test_emit_enforces_utf8_before_writing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The enforcement and the write are one operation, so 4.1 cannot split them."""
    stream = cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    emit(DEVANAGARI)
    buffer = stream.buffer
    assert isinstance(buffer, io.BytesIO)
    assert buffer.getvalue().decode("utf-8") == f"{DEVANAGARI}\n"


def test_emit_does_not_double_the_trailing_newline() -> None:
    stream = io.StringIO()
    emit("one\n", stream=stream)
    assert stream.getvalue() == "one\n"


def test_emit_appends_a_single_trailing_newline() -> None:
    stream = io.StringIO()
    emit("two", stream=stream)
    assert stream.getvalue() == "two\n"


# --------------------------------------------------------------------------- #
# Requirement 4.4 - locator rendering, all four kinds plus the unknown
# --------------------------------------------------------------------------- #

#: Hand-written, independent of the module's own formatter table. Keys are the
#: `locator_type` strings `ingest.types._locator_to_dict` emits.
EXPECTED_LOCATOR_LABELS: dict[str, tuple[str, str]] = {
    "markdown": (MARKDOWN_LOCATOR, "Lines 45-80"),
    "page": (PAGE_LOCATOR, "Page 3"),
    "sheet": (SHEET_LOCATOR, "Sheet 'Summary'!A1:C9"),
    "image_file": (IMAGE_LOCATOR, "Image file"),
}


@pytest.mark.parametrize(
    ("locator_type", "locator_json", "expected"),
    [(name, payload, label) for name, (payload, label) in EXPECTED_LOCATOR_LABELS.items()],
)
def test_each_locator_kind_renders_its_pinned_label(
    locator_type: str, locator_json: str, expected: str
) -> None:
    assert format_locator(locator_type, locator_json) == expected


def test_the_formatter_table_covers_exactly_the_four_locator_dataclasses() -> None:
    """Structural completeness: a fifth locator kind must fail loudly here.

    Built from `ingest.types` itself - the authority on what `locator_type`
    values can reach the index - and compared against BOTH the module's table
    and this file's hand-written one, so neither can drift alone.
    """
    from npu_rag.ingest.types import (
        ImageFileLocator,
        Locator,
        MarkdownLocator,
        PageLocator,
        SheetLocator,
        _LOCATOR_TYPES,
        _locator_to_dict,
    )

    instances: dict[type[Locator], Locator] = {
        MarkdownLocator: MarkdownLocator(line_range=(45, 80), ordinal=0),
        PageLocator: PageLocator(page=3),
        SheetLocator: SheetLocator(sheet="Summary", cell_range="A1:C9"),
        ImageFileLocator: ImageFileLocator(),
    }
    assert set(instances) == set(_LOCATOR_TYPES), "a Locator dataclass has no instance here"
    emitted = {str(_locator_to_dict(value)["type"]) for value in instances.values()}
    assert emitted == set(LOCATOR_FORMATTERS)
    assert emitted == set(EXPECTED_LOCATOR_LABELS)


def test_a_locator_round_tripped_through_the_real_serializer_still_renders() -> None:
    """The join, not the halves: `schema.py` is what produces `locator_json`."""
    from npu_rag.ingest.types import MarkdownLocator, _locator_to_dict

    payload = json.dumps(_locator_to_dict(MarkdownLocator(line_range=(45, 80), ordinal=2)))
    assert format_locator("markdown", payload) == "Lines 45-80"


def test_a_single_line_range_renders_in_the_singular() -> None:
    payload = json.dumps({"type": "markdown", "line_range": [7, 7], "ordinal": 0})
    assert format_locator("markdown", payload) == "Line 7"


def test_a_sheet_without_a_cell_range_renders_the_sheet_alone() -> None:
    payload = json.dumps({"type": "sheet", "sheet": "Bilanz Übersicht", "cell_range": ""})
    assert format_locator("sheet", payload) == "Sheet 'Bilanz Übersicht'"


@pytest.mark.parametrize(
    ("locator_type", "locator_json", "expected"),
    [
        ("video", json.dumps({"type": "video", "frame": 10}), "video"),
        ("", "{}", UNKNOWN_LOCATOR),
        ("   ", "{}", UNKNOWN_LOCATOR),
        ("markdown", "not json at all", "markdown"),
        ("markdown", "[1, 2]", "markdown"),
        ("markdown", json.dumps({"type": "markdown"}), "markdown"),
        ("markdown", json.dumps({"type": "markdown", "line_range": [1], "ordinal": 0}), "markdown"),
        ("markdown", json.dumps({"type": "markdown", "line_range": "45-80", "ordinal": 0}), "markdown"),
        ("page", json.dumps({"type": "page", "page": "three"}), "page"),
        ("page", json.dumps({"type": "page", "page": True}), "page"),
        ("sheet", json.dumps({"type": "sheet", "sheet": 1, "cell_range": "A1"}), "sheet"),
    ],
)
def test_an_unrenderable_locator_degrades_instead_of_raising(
    locator_type: str, locator_json: str, expected: str
) -> None:
    """A breadcrumb must never be the reason a search fails to print."""
    assert format_locator(locator_type, locator_json) == expected


# --------------------------------------------------------------------------- #
# Requirement 6.3 - provider attribution
# --------------------------------------------------------------------------- #


def test_provider_constants_agree_with_the_runtime_that_reports_them() -> None:
    """Pinned against literals AND cross-checked against the owning modules."""
    from npu_rag.embedding.environment.capability import VITISAI_PROVIDER
    from npu_rag.embedding.providers.cpu import CPU_PROVIDER as RUNTIME_CPU_PROVIDER

    assert NPU_PROVIDER == "VitisAIExecutionProvider" == VITISAI_PROVIDER
    assert CPU_PROVIDER == "CPUExecutionProvider" == RUNTIME_CPU_PROVIDER


def test_the_npu_provider_is_labelled_as_the_npu() -> None:
    """Requirement 6.3's first worked example, verbatim."""
    assert format_provider(NPU_PROVIDER) == "[Provider: VitisAIExecutionProvider (NPU)]"


def test_the_cpu_provider_is_labelled_without_embellishment() -> None:
    """Requirement 6.3's second worked example, verbatim."""
    assert format_provider(CPU_PROVIDER) == "[Provider: CPUExecutionProvider]"


def test_a_provider_name_carrying_escapes_is_sanitised() -> None:
    """A provider string reaches here from a runtime, not from a constant."""
    assert (
        format_provider("\x1b[31mCPUExecutionProvider\x1b[0m")
        == "[Provider: CPUExecutionProvider]"
    )


def test_an_unrecognised_provider_is_reported_verbatim() -> None:
    """Render reports what it was given; it never upgrades a guess to a fact."""
    assert format_provider("DmlExecutionProvider") == "[Provider: DmlExecutionProvider]"
    assert format_provider("npu") == "[Provider: npu]"


@pytest.mark.parametrize("provider", ["", "   ", "\t\n"])
def test_a_blank_provider_is_refused_rather_than_silently_erased(provider: str) -> None:
    """Requirement 6.2's zero-silent-fallback discipline, at the render seam."""
    with pytest.raises(ValueError, match="provider"):
        format_provider(provider)


def test_a_blank_provider_is_refused_by_both_renderers() -> None:
    response = make_response((make_result(FIRST_DAY),))
    with pytest.raises(ValueError, match="provider"):
        render_search_text(response, provider="")
    with pytest.raises(ValueError, match="provider"):
        render_search_json(response, provider="")


# --------------------------------------------------------------------------- #
# Requirement 4.4 - the breadcrumb itself
# --------------------------------------------------------------------------- #

BREADCRUMB_KWARGS: dict[str, Any] = {
    "folder": "Bhagwati Ugra Tara Sadhana",
    "document": "Savarni Nath - First Day.md",
    "document_index": 1,
    "document_total": 13,
    "chunk_index": 12,
    "locator": "Lines 45-80",
}


def test_the_breadcrumb_matches_requirement_4_4_verbatim() -> None:
    assert format_breadcrumb(**BREADCRUMB_KWARGS) == (
        "[Folder: Bhagwati Ugra Tara Sadhana] -> "
        "[Doc 1/13: Savarni Nath - First Day.md] -> "
        "[Chunk #12 | Lines 45-80]"
    )


def test_the_breadcrumb_has_exactly_three_segments_in_hierarchy_order() -> None:
    segments = format_breadcrumb(**BREADCRUMB_KWARGS).split(BREADCRUMB_SEPARATOR)
    assert len(segments) == 3
    assert segments[0].startswith("[Folder: ")
    assert segments[1].startswith("[Doc ")
    assert segments[2].startswith("[Chunk #")


def test_the_breadcrumb_separator_is_the_designed_arrow() -> None:
    assert BREADCRUMB_SEPARATOR == " -> "


def test_a_coloured_breadcrumb_strips_back_to_the_plain_one() -> None:
    coloured = format_breadcrumb(**BREADCRUMB_KWARGS, color=True)
    assert "\x1b" in coloured
    assert strip_ansi(coloured) == format_breadcrumb(**BREADCRUMB_KWARGS)


def test_the_three_breadcrumb_segments_carry_three_distinct_colours() -> None:
    """Colour *semantics*, without freezing the palette.

    Which code decorates which segment is cosmetic and deliberately left free
    to change (see the module's deviation 5 reasoning). What is not cosmetic is
    that the three levels of the hierarchy remain visually separable: a change
    collapsing two segments onto one colour defeats the point of colouring a
    breadcrumb at all, and that is what this catches.
    """
    coloured = format_breadcrumb(**BREADCRUMB_KWARGS, color=True)
    matches = [
        re.search(rf"(\x1b\[\d+m)\[{prefix}", coloured)
        for prefix in ("Folder: ", "Doc ", "Chunk #")
    ]
    assert all(matches), coloured
    codes = [match.group(1) for match in matches if match is not None]
    assert len(codes) == 3
    assert len(set(codes)) == 3, codes
    assert set(codes) <= set(SGR.values())


def test_the_breadcrumb_carries_non_ascii_document_names_intact() -> None:
    crumb = format_breadcrumb(
        folder="Bhagwati Ugra Tara Sadhana",
        document="उग्र तारा - पहला दिन.md",
        document_index=2,
        document_total=13,
        chunk_index=3,
        locator="Page 3",
    )
    assert "उग्र तारा - पहला दिन.md" in crumb


def test_the_breadcrumb_sanitises_escapes_planted_in_its_data() -> None:
    crumb = format_breadcrumb(
        folder="\x1b[31mFolder\x1b[0m",
        document="\x1b]0;hijack\x07doc.md",
        document_index=1,
        document_total=1,
        chunk_index=0,
        locator="Line 1",
    )
    assert "\x1b" not in crumb
    assert "hijack" not in crumb
    assert crumb == "[Folder: Folder] -> [Doc 1/1: doc.md] -> [Chunk #0 | Line 1]"


# --------------------------------------------------------------------------- #
# Requirement 8.1 - the terminal rendering, against design §3.4's sample
# --------------------------------------------------------------------------- #

RULE = "=" * RULE_WIDTH


def test_grouped_output_matches_the_designed_sample_block() -> None:
    """Design §3.4's worked sample, reproduced line for line."""
    response = make_response(
        (
            make_result(FIRST_DAY, chunk_index=12, score=0.892, text=DEVANAGARI),
            make_result(
                FIRST_DAY,
                chunk_index=13,
                score=0.774,
                text=DEVANAGARI_2,
                locator_json=json.dumps(
                    {"type": "markdown", "line_range": [81, 96], "ordinal": 0}
                ),
            ),
        )
    )
    rendered = render_search_text(
        response,
        provider=NPU_PROVIDER,
        archive_root=ARCHIVE_ROOT,
        group_by_folder=True,
        folder_tags={"Bhagwati Ugra Tara Sadhana": ("siddha",)},
    )
    assert rendered == "\n".join(
        [
            "[Query: ritual purification] [Mode: hybrid] "
            "[Provider: VitisAIExecutionProvider (NPU)]",
            "[Results: 2] [Candidates: 57] [Elapsed: 0.142s]",
            "",
            RULE,
            "[Folder: Bhagwati Ugra Tara Sadhana] (Tag: siddha)",
            RULE,
            f"  {HIT_MARKER} [Folder: Bhagwati Ugra Tara Sadhana] -> "
            "[Doc 1/1: Savarni Nath - First Day.md] -> "
            "[Chunk #12 | Lines 45-80] [Score: 0.892]",
            f'      "{DEVANAGARI}"',
            "",
            f"  {HIT_MARKER} [Folder: Bhagwati Ugra Tara Sadhana] -> "
            "[Doc 1/1: Savarni Nath - First Day.md] -> "
            "[Chunk #13 | Lines 81-96] [Score: 0.774]",
            f'      "{DEVANAGARI_2}"',
        ]
    )


def test_the_hit_marker_is_the_designed_glyph() -> None:
    assert HIT_MARKER == "\u25b6"


def test_the_banner_rule_is_eighty_columns() -> None:
    assert RULE_WIDTH == 80


def test_the_long_path_prefix_never_reaches_the_output() -> None:
    """Implementation note 2.2: unstripped, every breadcrumb reads `?/D:/...`."""
    response = make_response((make_result(FIRST_DAY),))
    rendered = render_search_text(response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT)
    assert "\\\\?\\" not in rendered
    assert "?" not in rendered
    assert "[Doc 1/1: Savarni Nath - First Day.md]" in rendered


def test_a_unc_long_path_prefix_is_stripped_before_the_leaf_is_taken() -> None:
    """Tested FIRST per note 2.2: `\\\\?\\UNC\\` begins with `\\\\?\\`."""
    path = UNC_ARCHIVE_ROOT + r"\Bhagwan Garuda Sadhana\Day 1.md"
    response = make_response((make_result(path),))
    rendered = render_search_text(
        response, provider=CPU_PROVIDER, archive_root=UNC_ARCHIVE_ROOT, group_by_folder=True
    )
    assert "UNC" not in rendered
    assert "[Folder: Bhagwan Garuda Sadhana]" in rendered
    assert "[Doc 1/1: Day 1.md]" in rendered


def test_flat_output_carries_the_full_breadcrumb_for_every_hit() -> None:
    """Requirement 4.4 says *every hit*, and a redirected line must stand alone."""
    response = make_response(
        (
            make_result(FIRST_DAY, chunk_index=12, score=0.892),
            make_result(GARUDA, chunk_index=4, score=0.511, text=DEVANAGARI_2),
        )
    )
    rendered = render_search_text(response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT)
    assert RULE not in rendered, "a flat render has no folder banner"
    hit_lines = [line for line in rendered.splitlines() if HIT_MARKER in line]
    assert len(hit_lines) == 2
    for line in hit_lines:
        segments = line.split(BREADCRUMB_SEPARATOR)
        assert len(segments) == 3
        assert "[Folder: " in segments[0]
        assert "[Doc " in segments[1]
        assert "[Chunk #" in segments[2]


def test_the_score_is_rendered_to_three_decimals() -> None:
    response = make_response((make_result(FIRST_DAY, score=0.8924999),))
    rendered = render_search_text(response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT)
    assert "[Score: 0.892]" in rendered
    assert "[Score: 0.89]" not in rendered
    assert "[Score: 0.8925]" not in rendered


def test_the_elapsed_time_is_rendered_to_three_decimals() -> None:
    response = make_response((make_result(FIRST_DAY),), elapsed_seconds=1.23456)
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    assert "[Elapsed: 1.235s]" in rendered


def test_the_header_reports_the_provider_in_the_terminal_output() -> None:
    """Requirement 6.3's terminal half."""
    response = make_response((make_result(FIRST_DAY),))
    rendered = render_search_text(response, provider=NPU_PROVIDER)
    assert "[Provider: VitisAIExecutionProvider (NPU)]" in rendered


def test_the_header_reports_the_query_the_mode_and_the_counts() -> None:
    response = make_response((make_result(FIRST_DAY),), query="जल शुद्धि", total_candidates=91)
    rendered = render_search_text(response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT)
    first = rendered.splitlines()[0]
    assert "[Query: जल शुद्धि]" in first
    assert "[Mode: hybrid]" in first
    assert "[Results: 1] [Candidates: 91]" in rendered.splitlines()[1]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (SearchMode.HYBRID, "hybrid"),
        (SearchMode.DENSE_ONLY, "dense_only"),
        (SearchMode.LEXICAL_ONLY, "lexical_only"),
    ],
)
def test_every_search_mode_renders_as_its_value(mode: SearchMode, expected: str) -> None:
    response = make_response((make_result(FIRST_DAY),), mode=mode)
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    assert f"[Mode: {expected}]" in rendered
    payload: Any = json.loads(render_search_json(response, provider=CPU_PROVIDER))
    assert payload["mode"] == expected


def test_an_empty_result_set_says_so_and_still_reports_the_provider() -> None:
    response = make_response((), total_candidates=0, elapsed_seconds=0.0012)
    rendered = render_search_text(response, provider=CPU_PROVIDER, group_by_folder=True)
    assert rendered.splitlines() == [
        "[Query: ritual purification] [Mode: hybrid] [Provider: CPUExecutionProvider]",
        "[Results: 0] [Candidates: 0] [Elapsed: 0.001s]",
        "",
        NO_RESULTS_LINE,
    ]


# --------------------------------------------------------------------------- #
# The document index/total denominator - ruling 1
# --------------------------------------------------------------------------- #


def test_the_document_total_counts_documents_not_hits() -> None:
    """Three hits, two documents: the denominator is 2, honestly computed."""
    response = make_response(
        (
            make_result(FIRST_DAY, chunk_index=12, score=0.9),
            make_result(FIRST_DAY, chunk_index=13, score=0.8),
            make_result(SECOND_DAY, chunk_index=1, score=0.7),
        )
    )
    rendered = render_search_text(
        response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT, group_by_folder=True
    )
    assert "[Doc 1/2: Savarni Nath - First Day.md]" in rendered
    assert "[Doc 2/2: Savarni Nath - Second Day.md]" in rendered
    assert "/3:" not in rendered

    payload: Any = json.loads(
        render_search_json(
            response,
            provider=CPU_PROVIDER,
            archive_root=ARCHIVE_ROOT,
            group_by_folder=True,
        )
    )
    row = payload["folders"][0]
    assert row["result_count"] == 3
    assert row["document_count"] == 2


def test_the_denominator_is_scoped_to_the_folder_in_both_modes() -> None:
    """One document from each of two folders: each is `1/1`, never `1/2`."""
    response = make_response(
        (
            make_result(FIRST_DAY, score=0.9),
            make_result(GARUDA, score=0.8),
        )
    )
    for group_by_folder in (False, True):
        rendered = render_search_text(
            response,
            provider=CPU_PROVIDER,
            archive_root=ARCHIVE_ROOT,
            group_by_folder=group_by_folder,
        )
        assert rendered.count("[Doc 1/1: ") == 2, rendered


def test_same_named_documents_in_different_folders_are_counted_apart() -> None:
    """The document key is the path, not the leaf name."""
    left = r"\\?\D:\Source\substack_dl\archive\author-a\images\plate.md"
    right = r"\\?\D:\Source\substack_dl\archive\author-b\images\plate.md"
    response = make_response((make_result(left, score=0.9), make_result(right, score=0.8)))
    rendered = render_search_text(
        response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT, group_by_folder=True
    )
    assert rendered.count("[Doc 1/1: plate.md]") == 2
    assert "[Folder: author-a]" in rendered
    assert "[Folder: author-b]" in rendered


def test_same_named_documents_in_one_folder_are_counted_apart() -> None:
    """The key is the whole path: two `plate.md` under one folder are two docs."""
    left = r"\\?\D:\Source\substack_dl\archive\author-a\images\plate.md"
    right = r"\\?\D:\Source\substack_dl\archive\author-a\notes\plate.md"
    response = make_response((make_result(left, score=0.9), make_result(right, score=0.8)))
    rendered = render_search_text(
        response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT, group_by_folder=True
    )
    assert "[Doc 1/2: plate.md]" in rendered
    assert "[Doc 2/2: plate.md]" in rendered


def test_grouping_makes_a_folder_contiguous_against_the_score_order() -> None:
    """The case where grouped order genuinely differs from the ranking."""
    response = make_response(
        (
            make_result(FIRST_DAY, chunk_index=12, score=0.9),
            make_result(GARUDA, chunk_index=4, score=0.8),
            make_result(SECOND_DAY, chunk_index=2, score=0.7),
        )
    )
    flat: Any = json.loads(
        render_search_json(response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT)
    )
    grouped: Any = json.loads(
        render_search_json(
            response,
            provider=CPU_PROVIDER,
            archive_root=ARCHIVE_ROOT,
            group_by_folder=True,
        )
    )
    assert [entry["chunk_index"] for entry in flat["results"]] == [12, 4, 2]
    assert [entry["chunk_index"] for entry in grouped["results"]] == [12, 2, 4]
    assert [entry["folder"] for entry in grouped["folders"]] == [
        "Bhagwati Ugra Tara Sadhana",
        "Bhagwan Garuda Sadhana",
    ]


def test_a_document_sitting_in_the_archive_root_is_grouped_under_the_root() -> None:
    path = ARCHIVE_ROOT + r"\loose-note.md"
    response = make_response((make_result(path),))
    rendered = render_search_text(
        response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT, group_by_folder=True
    )
    assert "[Folder: .]" in rendered


def test_a_document_with_no_usable_path_still_renders() -> None:
    response = make_response((make_result(""),))
    rendered = render_search_text(response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT)
    assert UNKNOWN_DOCUMENT in rendered


# --------------------------------------------------------------------------- #
# Tag annotation - ruling 2
# --------------------------------------------------------------------------- #


def test_a_single_tag_renders_in_the_singular_as_the_design_shows() -> None:
    response = make_response((make_result(FIRST_DAY),))
    rendered = render_search_text(
        response,
        provider=CPU_PROVIDER,
        archive_root=ARCHIVE_ROOT,
        group_by_folder=True,
        folder_tags={"Bhagwati Ugra Tara Sadhana": ["siddha"]},
    )
    assert "[Folder: Bhagwati Ugra Tara Sadhana] (Tag: siddha)" in rendered


def test_several_tags_render_in_the_plural_in_the_order_given() -> None:
    response = make_response((make_result(FIRST_DAY),))
    rendered = render_search_text(
        response,
        provider=CPU_PROVIDER,
        archive_root=ARCHIVE_ROOT,
        group_by_folder=True,
        folder_tags={"Bhagwati Ugra Tara Sadhana": ["tara", "siddha", "शक्ति"]},
    )
    # The caller's order, not an alphabetised one: `[tags]` in `npu-rag.toml`
    # is written by a person and its order may be meaningful.
    assert "(Tags: tara, siddha, शक्ति)" in rendered


def test_an_untagged_folder_gets_no_annotation_and_no_dangling_space() -> None:
    response = make_response((make_result(FIRST_DAY),))
    rendered = render_search_text(
        response,
        provider=CPU_PROVIDER,
        archive_root=ARCHIVE_ROOT,
        group_by_folder=True,
        folder_tags={"Some Other Folder": ["siddha"]},
    )
    assert "[Folder: Bhagwati Ugra Tara Sadhana]\n" in rendered
    assert "(Tag" not in rendered


def test_render_is_ignorant_of_config_shape_and_accepts_any_mapping() -> None:
    """Ruling 2: the caller owns the folder-to-tags mapping, not `TagRegistry`."""
    response = make_response((make_result(FIRST_DAY),))
    rendered = render_search_text(
        response,
        provider=CPU_PROVIDER,
        archive_root=ARCHIVE_ROOT,
        group_by_folder=True,
        folder_tags={"Bhagwati Ugra Tara Sadhana": ("siddha",)},
    )
    assert "(Tag: siddha)" in rendered


# --------------------------------------------------------------------------- #
# Snippets, context expansion and non-ASCII
# --------------------------------------------------------------------------- #


def test_the_snippet_collapses_whitespace_onto_one_line() -> None:
    response = make_response((make_result(FIRST_DAY, text="पहला\n\n  दूसरा\tतीसरा"),))
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    assert '      "पहला दूसरा तीसरा"' in rendered


def test_a_long_snippet_is_truncated_with_an_ellipsis() -> None:
    long_text = " ".join(["शुद्धि"] * 200)
    response = make_response((make_result(FIRST_DAY, text=long_text),))
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    snippet = next(line for line in rendered.splitlines() if line.startswith('      "'))
    assert snippet.endswith('..."')
    assert len(snippet) <= SNIPPET_LIMIT + len('      "..."')


def test_a_truncated_snippet_ends_on_a_whole_word() -> None:
    """A preview cut through the middle of a word is a worse preview."""
    response = make_response((make_result(FIRST_DAY, text=" ".join(["शुद्धि"] * 200)),))
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    snippet = next(line for line in rendered.splitlines() if line.startswith('      "'))
    body = snippet.strip().strip('"').removesuffix("...")
    assert body
    assert body == " ".join(["शुद्धि"] * len(body.split()))


def test_a_long_unbroken_run_is_cut_at_the_budget_not_at_an_early_space() -> None:
    """The word-boundary trim applies only when it keeps most of the budget.

    Without that guard a text whose only early space sits at position 1 - a
    mantra syllable followed by an unbroken Devanagari run, which this archive
    does contain - would be previewed as a single character plus an ellipsis.
    """
    response = make_response((make_result(FIRST_DAY, text="ॐ " + "अ" * 400),))
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    snippet = next(line for line in rendered.splitlines() if line.startswith('      "'))
    body = snippet.strip().strip('"')
    assert body.endswith("...")
    assert len(body) == SNIPPET_LIMIT + len("...")


def test_a_snippet_of_exactly_the_budget_is_returned_whole() -> None:
    """The boundary value: `<= limit`, not `< limit`.

    Off by one here needlessly truncates a snippet that fits exactly and
    discards its last word, which is the same class of guard defect as the
    word-boundary rule above. Length is asserted so the fixture cannot be
    quietly the wrong size.
    """
    text = " ".join(["शुद्धि"] * 28) + " अअअअ"
    assert len(text) == SNIPPET_LIMIT, len(text)
    response = make_response((make_result(FIRST_DAY, text=text),))
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    assert f'      "{text}"' in rendered
    assert "..." not in rendered


def test_a_snippet_one_character_over_the_budget_is_truncated() -> None:
    """The other side of the same boundary, so the guard is bracketed."""
    text = " ".join(["शुद्धि"] * 28) + " अअअअअ"
    assert len(text) == SNIPPET_LIMIT + 1, len(text)
    response = make_response((make_result(FIRST_DAY, text=text),))
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    assert f'      "{text}"' not in rendered
    assert "..." in rendered


def test_a_short_snippet_is_not_truncated() -> None:
    response = make_response((make_result(FIRST_DAY, text=DEVANAGARI),))
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    assert f'      "{DEVANAGARI}"' in rendered


def test_expanded_context_is_rendered_as_a_labelled_continuation() -> None:
    """`ContextExpander` fills one field; it does not invent a second hit."""
    response = make_response(
        (
            make_result(
                FIRST_DAY,
                text=DEVANAGARI,
                expanded_context=f"{DEVANAGARI}\n\n{DEVANAGARI_2}",
            ),
        )
    )
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    assert f'      [Context] "{DEVANAGARI} {DEVANAGARI_2}"' in rendered


def test_expanded_context_equal_to_the_text_is_not_repeated() -> None:
    response = make_response((make_result(FIRST_DAY, text=DEVANAGARI, expanded_context=DEVANAGARI),))
    rendered = render_search_text(response, provider=CPU_PROVIDER)
    assert "[Context]" not in rendered


@pytest.mark.parametrize("script", [DEVANAGARI, CJK, GERMAN])
def test_every_required_script_survives_the_whole_render(script: str) -> None:
    """Requirement 8.4's three named scripts, end to end and byte-identical."""
    response = make_response((make_result(FIRST_DAY, text=script),), query=script)
    rendered = render_search_text(response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT)
    assert script in rendered
    assert rendered.encode("utf-8").decode("utf-8") == rendered


def test_non_ascii_folder_and_document_names_render_intact() -> None:
    path = ARCHIVE_ROOT + "\\उग्र तारा साधना\\पहला दिन - Übung 第三章.md"
    response = make_response((make_result(path),))
    rendered = render_search_text(
        response, provider=CPU_PROVIDER, archive_root=ARCHIVE_ROOT, group_by_folder=True
    )
    assert "[Folder: उग्र तारा साधना]" in rendered
    assert "[Doc 1/1: पहला दिन - Übung 第三章.md]" in rendered


def test_the_whole_render_is_writable_to_a_cp1252_stdout_once_enforced() -> None:
    """Requirement 8.4's actual failure mode, exercised through `emit`."""
    response = make_response(
        (make_result(FIRST_DAY, text=f"{DEVANAGARI} {CJK} {GERMAN}"),),
        query=CJK,
    )
    rendered = render_search_text(response, provider=NPU_PROVIDER, archive_root=ARCHIVE_ROOT)
    stream = cp1252_stream()
    emit(rendered, stream=stream)
    buffer = stream.buffer
    assert isinstance(buffer, io.BytesIO)
    assert buffer.getvalue().decode("utf-8") == rendered + "\n"


# --------------------------------------------------------------------------- #
# Requirement 8.2 end to end - colour on, colour off
# --------------------------------------------------------------------------- #


def hostile_response() -> SearchResponse:
    """A response whose *data* carries escape sequences, as a document may."""
    return make_response(
        (
            make_result(
                FIRST_DAY,
                text=f"\x1b[31m{DEVANAGARI}\x1b[0m",
                expanded_context="\x1b]0;hijack\x07continuation",
            ),
        ),
        query="\x1b[5mblinking\x1b[0m",
    )


def test_colour_off_leaves_not_one_escape_byte() -> None:
    """Non-vacuous: the composition genuinely contains styling before stripping."""
    rendered = render_search_text(
        hostile_response(),
        provider=NPU_PROVIDER,
        archive_root=ARCHIVE_ROOT,
        group_by_folder=True,
        folder_tags={"Bhagwati Ugra Tara Sadhana": ("siddha",)},
        color=False,
    )
    assert "\x1b" not in rendered
    assert "hijack" not in rendered
    assert DEVANAGARI in rendered


def test_colour_on_emits_styling_and_still_sanitises_the_data() -> None:
    rendered = render_search_text(
        hostile_response(),
        provider=NPU_PROVIDER,
        archive_root=ARCHIVE_ROOT,
        group_by_folder=True,
        folder_tags={"Bhagwati Ugra Tara Sadhana": ("siddha",)},
        color=True,
    )
    assert "\x1b" in rendered
    assert SGR["bold"] in rendered
    assert SGR["cyan"] in rendered
    assert SGR["green"] in rendered
    # The data's own escapes are gone: neither is in the module's SGR table.
    assert "\x1b[31m" not in rendered
    assert "\x1b[5m" not in rendered
    assert "hijack" not in rendered


def test_the_coloured_render_strips_back_to_the_uncoloured_one() -> None:
    """The two paths differ only by styling, which is what 8.2 promises."""
    kwargs: dict[str, Any] = {
        "provider": NPU_PROVIDER,
        "archive_root": ARCHIVE_ROOT,
        "group_by_folder": True,
        "folder_tags": {"Bhagwati Ugra Tara Sadhana": ("siddha",)},
    }
    response = make_response(
        (
            make_result(FIRST_DAY, chunk_index=12, score=0.892),
            make_result(SECOND_DAY, chunk_index=2, score=0.5, text=DEVANAGARI_2),
        )
    )
    coloured = render_search_text(response, color=True, **kwargs)
    plain = render_search_text(response, color=False, **kwargs)
    assert "\x1b" in coloured
    assert strip_ansi(coloured) == plain


def test_every_sgr_code_used_by_the_renderer_is_a_named_table_entry() -> None:
    coloured = render_search_text(
        make_response((make_result(FIRST_DAY),)),
        provider=NPU_PROVIDER,
        archive_root=ARCHIVE_ROOT,
        group_by_folder=True,
        folder_tags={"Bhagwati Ugra Tara Sadhana": ("siddha",)},
        color=True,
    )
    used = set(re.findall(r"\x1b\[\d+m", coloured))
    assert used <= set(SGR.values()), used - set(SGR.values())
    assert used, "the coloured render emitted no styling at all"


# --------------------------------------------------------------------------- #
# Requirement 8.3 / 6.3 - the JSON payload
# --------------------------------------------------------------------------- #

#: Hand-written envelope schema. snake_case and stable; a key added or renamed
#: without a decision fails here.
EXPECTED_ENVELOPE_KEYS: tuple[str, ...] = (
    "query",
    "mode",
    "provider",
    "grouped",
    "total_candidates",
    "elapsed_seconds",
    "folders",
    "results",
)

EXPECTED_FOLDER_KEYS: tuple[str, ...] = (
    "folder",
    "tags",
    "result_count",
    "document_count",
)

EXPECTED_RESULT_KEYS: tuple[str, ...] = (
    "folder",
    "tags",
    "document",
    "document_index",
    "document_total",
    "file_path",
    "source_path",
    "title",
    "chunk_id",
    "chunk_index",
    "chunk_kind",
    "headings",
    "locator_type",
    "locator",
    "locator_label",
    "score",
    "text",
    "expanded_context",
    "token_count",
    "dense_distance",
    "dense_rank",
    "lexical_score",
    "lexical_rank",
    "breadcrumb",
)


def full_response() -> SearchResponse:
    return make_response(
        (
            make_result(
                FIRST_DAY,
                chunk_index=12,
                score=0.892,
                title="उग्र तारा साधना",
                headings=("Day One", "Purification"),
                dense_distance=0.108,
                dense_rank=1,
                lexical_score=4.5,
                lexical_rank=2,
                expanded_context=f"{DEVANAGARI} {DEVANAGARI_2}",
            ),
            make_result(SECOND_DAY, chunk_index=2, score=0.774, text=DEVANAGARI_2),
            make_result(
                SUBSTACK,
                chunk_index=0,
                score=0.400,
                text=GERMAN,
                locator_type="page",
                locator_json=PAGE_LOCATOR,
            ),
        )
    )


def parsed_json(**kwargs: Any) -> dict[str, Any]:
    text = render_search_json(
        full_response(),
        provider=NPU_PROVIDER,
        archive_root=ARCHIVE_ROOT,
        folder_tags={"Bhagwati Ugra Tara Sadhana": ("siddha", "tara")},
        **kwargs,
    )
    loaded: Any = json.loads(text)
    assert isinstance(loaded, dict)
    return loaded


def test_the_json_payload_parses() -> None:
    """Requirement 8.3: valid, structured JSON for `jq`."""
    assert parsed_json()["query"] == "ritual purification"


def test_the_envelope_keys_match_the_literal_schema() -> None:
    assert tuple(parsed_json()) == EXPECTED_ENVELOPE_KEYS


def test_every_result_object_has_exactly_the_literal_schema() -> None:
    for entry in parsed_json()["results"]:
        assert tuple(entry) == EXPECTED_RESULT_KEYS


def test_every_folder_object_has_exactly_the_literal_schema() -> None:
    for entry in parsed_json()["folders"]:
        assert tuple(entry) == EXPECTED_FOLDER_KEYS


def test_the_json_reports_the_provider() -> None:
    """Requirement 6.3's machine half - the provider is a field, not a banner."""
    assert parsed_json()["provider"] == NPU_PROVIDER


def test_the_json_reports_the_mode_the_counts_and_the_timing() -> None:
    payload = parsed_json()
    assert payload["mode"] == "hybrid"
    assert payload["total_candidates"] == 57
    assert payload["elapsed_seconds"] == pytest.approx(0.1423)


def test_the_json_schema_is_identical_whether_grouping_is_on() -> None:
    """A consumer must not need two parsers (the stability constraint)."""
    flat = parsed_json(group_by_folder=False)
    grouped = parsed_json(group_by_folder=True)
    assert tuple(flat) == tuple(grouped)
    assert [tuple(r) for r in flat["results"]] == [tuple(r) for r in grouped["results"]]
    assert flat["grouped"] is False
    assert grouped["grouped"] is True
    # Same hits either way, only the order and the folder metadata differ.
    assert {r["chunk_id"] for r in flat["results"]} == {
        r["chunk_id"] for r in grouped["results"]
    }


def test_grouping_is_visible_in_the_json_through_folders_and_ordering() -> None:
    grouped = parsed_json(group_by_folder=True)
    folders = [entry["folder"] for entry in grouped["folders"]]
    assert folders == ["Bhagwati Ugra Tara Sadhana", "businessengineer.ai"]
    sequence = [entry["folder"] for entry in grouped["results"]]
    assert sequence == [
        "Bhagwati Ugra Tara Sadhana",
        "Bhagwati Ugra Tara Sadhana",
        "businessengineer.ai",
    ]


def test_every_result_names_its_folder_so_a_jq_filter_needs_no_nesting() -> None:
    for entry in parsed_json()["results"]:
        assert entry["folder"]
    selected = [
        entry["chunk_index"]
        for entry in parsed_json()["results"]
        if entry["folder"] == "Bhagwati Ugra Tara Sadhana"
    ]
    assert selected == [12, 2]


def test_the_folder_rows_carry_their_tags_and_their_honest_counts() -> None:
    rows = {entry["folder"]: entry for entry in parsed_json()["folders"]}
    assert rows["Bhagwati Ugra Tara Sadhana"]["tags"] == ["siddha", "tara"]
    assert rows["Bhagwati Ugra Tara Sadhana"]["result_count"] == 2
    assert rows["Bhagwati Ugra Tara Sadhana"]["document_count"] == 2
    assert rows["businessengineer.ai"]["tags"] == []


def test_the_json_carries_every_locator_kind_parsed_and_labelled() -> None:
    results = {entry["chunk_index"]: entry for entry in parsed_json()["results"]}
    markdown = results[12]
    assert markdown["locator_type"] == "markdown"
    assert markdown["locator"] == {"type": "markdown", "line_range": [45, 80], "ordinal": 0}
    assert markdown["locator_label"] == "Lines 45-80"
    page = results[0]
    assert page["locator"] == {"type": "page", "page": 3}
    assert page["locator_label"] == "Page 3"


def test_an_unparseable_locator_becomes_null_without_losing_its_label() -> None:
    response = make_response((make_result(FIRST_DAY, locator_json="{nope"),))
    payload: Any = json.loads(render_search_json(response, provider=CPU_PROVIDER))
    entry = payload["results"][0]
    assert entry["locator"] is None
    assert entry["locator_label"] == "markdown"


def test_the_json_file_path_is_stripped_and_separator_normalised() -> None:
    entry = parsed_json()["results"][0]
    assert entry["file_path"] == (
        "D:/Source/substack_dl/archive/Bhagwati Ugra Tara Sadhana/"
        "Savarni Nath - First Day.md"
    )
    assert "\\" not in entry["file_path"]


def test_the_json_carries_the_source_path_verbatim() -> None:
    """The openable form, byte-identical to the `file_path` column."""
    entry = parsed_json()["results"][0]
    assert entry["source_path"] == FIRST_DAY
    assert entry["source_path"].startswith("\\\\?\\")


def test_a_unc_source_path_round_trips_while_the_display_path_stays_readable() -> None:
    """The contract gap this field exists to close.

    `order._components` discards empty components (`order.py:272-274`), so the
    display form of a UNC source loses its leading `\\\\` and becomes
    unopenable - at any path length. `ingest/discover.py:148-155` stamps
    `\\\\?\\UNC\\` on every file under a share, so this is realised the first
    time `--archive-path` names one.
    """
    path = UNC_ARCHIVE_ROOT + r"\Bhagwan Garuda Sadhana\Day 1.md"
    response = make_response((make_result(path),))
    payload: Any = json.loads(
        render_search_json(response, provider=CPU_PROVIDER, archive_root=UNC_ARCHIVE_ROOT)
    )
    entry = payload["results"][0]
    assert entry["source_path"] == path
    assert entry["file_path"] == "nas/share/archive/Bhagwan Garuda Sadhana/Day 1.md"
    assert entry["source_path"] != entry["file_path"]


def test_the_json_carries_the_document_position_and_the_breadcrumb() -> None:
    entry = parsed_json()["results"][0]
    assert entry["document"] == "Savarni Nath - First Day.md"
    assert entry["document_index"] == 1
    assert entry["document_total"] == 2
    assert entry["breadcrumb"] == (
        "[Folder: Bhagwati Ugra Tara Sadhana] -> "
        "[Doc 1/2: Savarni Nath - First Day.md] -> "
        "[Chunk #12 | Lines 45-80]"
    )
    assert "\x1b" not in entry["breadcrumb"]


def test_the_json_carries_the_retrieval_diagnostics_as_nullable_fields() -> None:
    results = {entry["chunk_index"]: entry for entry in parsed_json()["results"]}
    assert results[12]["dense_distance"] == pytest.approx(0.108)
    assert results[12]["dense_rank"] == 1
    assert results[12]["lexical_score"] == pytest.approx(4.5)
    assert results[12]["lexical_rank"] == 2
    assert results[2]["dense_distance"] is None
    assert results[2]["lexical_rank"] is None


def test_the_json_keeps_the_full_text_untruncated_and_unsanitised() -> None:
    """Machine data, not a terminal: truncating or rewriting it would corrupt it."""
    long_text = " ".join(["शुद्धि"] * 200)
    response = make_response(
        (make_result(FIRST_DAY, text=f"\x1b[31m{long_text}\x1b[0m"),)
    )
    payload: Any = json.loads(render_search_json(response, provider=CPU_PROVIDER))
    assert payload["results"][0]["text"] == f"\x1b[31m{long_text}\x1b[0m"


def test_the_json_keeps_non_ascii_literal_rather_than_escaped() -> None:
    """`ensure_ascii=False`, so `jq` and a human both see Devanagari."""
    text = render_search_json(full_response(), provider=NPU_PROVIDER, archive_root=ARCHIVE_ROOT)
    assert DEVANAGARI in text
    assert "\\u0915" not in text


def test_the_json_title_headings_and_token_count_survive() -> None:
    entry = parsed_json()["results"][0]
    assert entry["title"] == "उग्र तारा साधना"
    assert entry["headings"] == ["Day One", "Purification"]
    assert entry["token_count"] == 128
    assert entry["expanded_context"] == f"{DEVANAGARI} {DEVANAGARI_2}"


def test_a_missing_title_and_context_are_null_not_absent() -> None:
    entry = parsed_json()["results"][1]
    assert entry["title"] is None
    assert entry["expanded_context"] is None
    assert entry["headings"] == []


def test_an_empty_result_set_is_still_a_valid_envelope() -> None:
    payload: Any = json.loads(
        render_search_json(make_response(()), provider=CPU_PROVIDER)
    )
    assert tuple(payload) == EXPECTED_ENVELOPE_KEYS
    assert payload["results"] == []
    assert payload["folders"] == []
    assert payload["provider"] == CPU_PROVIDER


def test_the_json_indent_is_configurable_and_defaults_to_two() -> None:
    assert JSON_INDENT == 2
    response = make_response((make_result(FIRST_DAY),))
    indented = render_search_json(response, provider=CPU_PROVIDER)
    compact = render_search_json(response, provider=CPU_PROVIDER, indent=None)
    assert "\n" in indented
    assert "\n" not in compact
    assert json.loads(indented) == json.loads(compact)


def test_the_json_never_carries_the_raw_vector() -> None:
    """768 floats per hit would make `--json` unusable; it is deliberately out."""
    response = make_response((make_result(FIRST_DAY),))
    assert "vector" not in render_search_json(response, provider=CPU_PROVIDER)


# --------------------------------------------------------------------------- #
# The dispatcher task 4.1 calls
# --------------------------------------------------------------------------- #


def test_render_search_dispatches_to_json() -> None:
    response = make_response((make_result(FIRST_DAY),))
    assert render_search(response, provider=CPU_PROVIDER, as_json=True) == render_search_json(
        response, provider=CPU_PROVIDER
    )


def test_render_search_dispatches_to_text() -> None:
    response = make_response((make_result(FIRST_DAY),))
    assert render_search(
        response, provider=CPU_PROVIDER, as_json=False, color=True
    ) == render_search_text(response, provider=CPU_PROVIDER, color=True)


def test_render_search_passes_every_knob_through() -> None:
    response = make_response((make_result(FIRST_DAY),))
    kwargs: dict[str, Any] = {
        "archive_root": ARCHIVE_ROOT,
        "group_by_folder": True,
        "folder_tags": {"Bhagwati Ugra Tara Sadhana": ("siddha",)},
    }
    assert render_search(
        response, provider=NPU_PROVIDER, as_json=False, color=False, **kwargs
    ) == render_search_text(response, provider=NPU_PROVIDER, color=False, **kwargs)
    assert render_search(
        response, provider=NPU_PROVIDER, as_json=True, **kwargs
    ) == render_search_json(response, provider=NPU_PROVIDER, **kwargs)


def test_an_archive_root_as_a_path_object_is_accepted() -> None:
    response = make_response((make_result(FIRST_DAY),))
    rendered = render_search_text(
        response,
        provider=CPU_PROVIDER,
        archive_root=Path(r"\\?\D:\Source\substack_dl\archive"),
        group_by_folder=True,
    )
    assert "[Folder: Bhagwati Ugra Tara Sadhana]" in rendered


def test_without_an_archive_root_the_group_degrades_visibly_not_silently() -> None:
    """Note 2.2's ruling: the fallback label states what happened."""
    response = make_response((make_result(FIRST_DAY),))
    rendered = render_search_text(response, provider=CPU_PROVIDER, group_by_folder=True)
    assert (
        "[Folder: D:/Source/substack_dl/archive/Bhagwati Ugra Tara Sadhana]" in rendered
    )


def test_the_public_surface_is_exactly_what_is_exported() -> None:
    """Tasks 4.1 and 4.2 build against `__all__`; it is a contract."""
    assert tuple(render.__all__) == (
        "BREADCRUMB_SEPARATOR",
        "CPU_PROVIDER",
        "HIT_MARKER",
        "JSON_INDENT",
        "LOCATOR_FORMATTERS",
        "NO_COLOR_VARIABLE",
        "NO_RESULTS_LINE",
        "NPU_PROVIDER",
        "RULE_WIDTH",
        "SGR",
        "SNIPPET_LIMIT",
        "UNKNOWN_DOCUMENT",
        "UNKNOWN_LOCATOR",
        "emit",
        "ensure_utf8_stdout",
        "format_breadcrumb",
        "format_locator",
        "format_provider",
        "render_search",
        "render_search_json",
        "render_search_text",
        "should_colorize",
        "strip_ansi",
        "style",
    )
    for name in render.__all__:
        assert hasattr(render, name), name


def test_no_unfinished_work_markers_in_the_module() -> None:
    source = Path(render.__file__).read_text(encoding="utf-8")
    for marker in ("TODO", "FIXME", "XXX", "HACK", "TBD"):
        assert marker not in source, marker


# --------------------------------------------------------------------------- #
# Requirement 1.2 - nothing heavy on the search path
# --------------------------------------------------------------------------- #

HEAVY_MODULES: tuple[str, ...] = ("lancedb", "onnxruntime", "torch", "transformers")

_PROBE = '''\
import json
import sys
from pathlib import Path

import npu_rag.cli.render  # noqa: F401

HEAVY = {heavy!r}
Path({result!r}).write_text(
    json.dumps(
        {{
            "heavy": sorted(name for name in HEAVY if name in sys.modules),
            "count": len(sys.modules),
        }}
    ),
    encoding="utf-8",
)
'''


def test_importing_render_imports_no_heavy_module(tmp_path: Path) -> None:
    """`npu_rag.index.types` drags in lancedb, torch and transformers (3266
    modules, measured in task 2.2), so `SearchResponse`/`SearchResult` are
    referenced only under `TYPE_CHECKING`.
    """
    result_path = tmp_path / "render_probe.json"
    script_path = tmp_path / "render_probe.py"
    script_path.write_text(
        _PROBE.format(heavy=HEAVY_MODULES, result=str(result_path)),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload: Any = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["heavy"] == [], f"npu_rag.cli.render pulled in heavy modules: {payload}"
    assert payload["count"] < 300, payload["count"]


def test_render_module_has_no_runtime_import_of_the_index_package() -> None:
    """The structural half: the import must sit inside `if TYPE_CHECKING:`."""
    source = Path(render.__file__)
    text = source.read_text(encoding="utf-8")
    body, _, guarded = text.partition("if TYPE_CHECKING:")
    assert guarded, "render.py must guard its SearchResponse import"
    assert not re.search(r"^from npu_rag\.index", body, flags=re.MULTILINE)
    assert "from npu_rag.index.types import SearchResponse, SearchResult" in guarded


def test_render_reuses_orders_long_path_stripping_rather_than_re_deriving_it() -> None:
    """Implementation note 2.2 instructs reuse; this is the structural pin."""
    text = Path(render.__file__).read_text(encoding="utf-8")
    assert re.search(r"^from npu_rag\.cli\.order import .*\b_components\b", text, re.MULTILINE)
    # `order._components` applies `_strip_long_prefix` (order.py:272-274), so
    # reusing the splitter reuses the strip. A separator character class in
    # render.py would mean the prefix handling had been re-derived here.
    assert "[\\\\/]" not in text, "render.py must not re-derive path splitting"
