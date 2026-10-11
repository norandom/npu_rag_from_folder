"""Terminal and JSON rendering for `npu-rag search` (task 3.1).

Design anchor: `.kiro/specs/search-cli/design.md` §3.4 (`render.py`), including
its worked sample output block; requirements 8.1 (styled UTF-8 output), 8.2
(automatic ANSI stripping when redirected or `NO_COLOR` is set), 8.3
(structured `--json`), 8.4 (UTF-8 stdout on Windows), 4.4 (hierarchical
breadcrumb attribution) and 6.3 (visible provider attribution in *both*
outputs).

This module is pure formatting plus one stream side effect. It computes
nothing about the archive: the folder label comes from `order.folder_of`, the
narrative order from `order.group_and_order_results`, and the tag annotation
from a mapping the caller hands in. Printing is `emit()`, which exists only so
that requirement 8.4's encoding enforcement cannot be separated from the
write.

----------------------------------------------------------------------------
Ruling 1 - `[Doc 1/13: ...]` counts the documents IN THESE RESULTS, per folder
----------------------------------------------------------------------------

Requirement 4.4 asks for `[Document (Index/Total)]` and no requirement defines
the denominator. The two candidates are the documents present in this result
set and the documents that folder holds *in the index*. The second is not
knowable here - `SearchResponse` carries no such count, and inventing one by
querying would make a formatter issue database calls - so the denominator is
the number of **distinct documents from that folder among the rendered
results**, and the index is that document's position in the rendered order of
first appearance.

The scope is the folder in *both* modes, grouped or not, so the field has one
meaning rather than two. A number that looks authoritative and is not is worse
than an honest one: `[Doc 1/2]` here means "the first of the two documents
this search surfaced in this folder", never "the first of two documents in the
folder". The header line states the same counts, so the scale is on screen.

----------------------------------------------------------------------------
Ruling 2 - `(Tag: siddha)` is the caller's mapping, not a reverse lookup here
----------------------------------------------------------------------------

Design §3.4 annotates the folder banner with its tag. `TagRegistry` has no
reverse lookup - its surface is `tags`, `tag_names()`, `resolve_tag()`,
`resolve_tags()` and `build_filter_sql()`, none of which maps a folder back to
the tags containing it - and `tags.py` is outside this task's boundary.

So `folder_tags` is a plain `Mapping[str, Sequence[str]]` from folder label to
tag names. Render stays ignorant of `CliConfig`, of glob semantics, and of how
a folder came to carry a tag; it renders what it is told, in the order it is
told, or nothing at all for a folder that is absent from the mapping.

**The obligation this places on task 4.1**: build the mapping keyed by exactly
the labels `order.folder_of(file_path, archive_root=...)` produces - that is,
the first path component beneath the resolved archive root - by matching each
`[tags]` entry's folder patterns against the folders actually present in the
results. A key that does not match a produced label annotates nothing and
fails silently, which is why the keying rule is stated here.

----------------------------------------------------------------------------
Ruling 3 - the JSON stays FLAT, with the folder on every result
----------------------------------------------------------------------------

Requirement 8.3 wants JSON a `jq` user can work with, and the schema must be
stable whether or not `--group-by-folder` is on - a consumer must not need two
parsers. Nesting results under folder keys would force exactly that: a script
would read `.results[]` in one mode and `.folders["Some Folder"][]` in the
other, and the second form also makes a folder name - arbitrary user text -
into an object key, so `jq` would need `.folders["..."]` quoting for every
real archive name.

The payload is therefore one flat `results` array in which every entry names
its `folder`, plus a `folders` array of per-folder rows (tags and counts) in
render order. One filter works in both modes:

    npu-rag search ... --json | jq '.results[] | select(.folder=="X") | .text'

Grouping remains visible without changing the shape: `grouped` is a boolean,
the `results` array is ordered (groups contiguous, in group order, narrative
order within a group), and `folders` lists the groups in that same order.

----------------------------------------------------------------------------
Ruling 4 - what `--json` carries, and the provider's shape
----------------------------------------------------------------------------

Requirements 6.3 and 8.3 together mean the machine payload must carry
everything the pretty output reports: `provider`, `mode`, `total_candidates`,
`elapsed_seconds`, and a usable locator. Keys are snake_case and stable;
`locator` is the parsed locator object (or `null` when `locator_json` cannot be
parsed) alongside the human `locator_label`, so a consumer can read
`.locator.page` without re-parsing a string. `vector` is deliberately absent:
768 floats per hit would make `--json` unusable for its stated purpose.

Paths appear twice, deliberately: `file_path` is normalised for reading and
`source_path` is the index column verbatim, which is the only form that can be
opened. See :func:`_display_path` for the UNC case that makes the distinction
load-bearing rather than cosmetic.

**`SearchResponse` carries no provider field**, so the provider arrives as a
separate, required, keyword-only `str`. It is a `str` and not an enum because
the value requirement 6.3 wants reported is the *active* ONNX Runtime provider
name - `VitisAIExecutionProvider` / `CPUExecutionProvider`, as
`session.get_providers()` returns it - which is runtime data, not a closed set
this package owns. It is required, and a blank value is refused, so that an
execution can never be rendered without attribution; and it is reported
verbatim, with `(NPU)` appended only for the exact Vitis AI name, so render
never upgrades a *requested* provider (`--provider npu`) into a claim about
what actually ran. Requirement 6.2 forbids silent fallback; silently
flattering a request would be the same defect one layer up.

Task 4.1 must therefore pass the provider the session reports, not the
`--provider` alias.

----------------------------------------------------------------------------
Startup cost (requirement 1.2)
----------------------------------------------------------------------------

`render.py` is on the `search` path. Importing `npu_rag.index.types` for
`SearchResponse`/`SearchResult` pulls in lancedb, torch and transformers
(3266 modules, measured in task 2.2), so those names are referenced only under
`TYPE_CHECKING`; nothing outside the standard library and `npu_rag.cli.order`
is imported at module scope. A probe test pins this.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from typing import TYPE_CHECKING, Final

from npu_rag.cli.order import _components, folder_of, group_and_order_results

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path
    from typing import IO

    from npu_rag.index.types import SearchResponse, SearchResult

__all__ = [
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
]

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

#: The active ONNX Runtime provider names requirement 6.3 cites verbatim.
#: Spelled here rather than imported: `npu_rag.embedding.providers.*` and
#: `environment.capability` own these constants, and importing either would put
#: numpy and the onnxruntime probe on the `search` path against requirement
#: 1.2. A test cross-checks both spellings against the owning modules, so the
#: duplication cannot drift unnoticed.
NPU_PROVIDER: Final = "VitisAIExecutionProvider"
CPU_PROVIDER: Final = "CPUExecutionProvider"

#: The environment variable of the `NO_COLOR` convention (https://no-color.org).
NO_COLOR_VARIABLE: Final = "NO_COLOR"

#: Requirement 4.4's arrow between breadcrumb segments.
BREADCRUMB_SEPARATOR: Final = " -> "

#: Design §3.4's banner rule and hit marker.
RULE_WIDTH: Final = 80
HIT_MARKER: Final = "\u25b6"  # BLACK RIGHT-POINTING TRIANGLE

#: Snippet budget in characters. Counted in characters and never in bytes:
#: Devanagari and CJK cost 3 bytes a character here, and a byte budget would
#: silently shrink the preview to a third for exactly the content this archive
#: is made of.
SNIPPET_LIMIT: Final = 200
SNIPPET_ELLIPSIS: Final = "..."

#: Fixed decimal places. Pinned as constants because a score's precision is
#: how a user compares two hits, and drifting it changes every line.
SCORE_DIGITS: Final = 3
ELAPSED_DIGITS: Final = 3

UNKNOWN_LOCATOR: Final = "Unknown location"
UNKNOWN_DOCUMENT: Final = "(unknown document)"
NO_RESULTS_LINE: Final = "No results."

#: Default `json.dumps` indent. `--json` output is read by `jq` and by people.
JSON_INDENT: Final = 2

#: The SGR sequences this module emits, by name. A closed, named table rather
#: than inline escapes: `strip_ansi` has to be able to remove exactly what
#: `style` adds, and a test pins every entry against a hand-written literal
#: (implementation note 1.1 - a code resolved against itself proves nothing).
SGR: Final[dict[str, str]] = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
}

# An ANSI escape in any of the three forms that reach a terminal: a CSI
# sequence (`\x1b[` parameters intermediates final), an OSC string (`\x1b]`
# ... BEL or ST - window titles and hyperlinks), and a two-character Fe/Fp
# escape. The OSC alternative must precede the two-character one, because `]`
# falls inside the latter's character range and would otherwise swallow the
# introducer and leave the payload text behind.
_ANSI_ESCAPE: Final = re.compile(
    "\x1b(?:"
    r"\[[0-?]*[ -/]*[@-~]"
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|[0-?@-Z\\-_]"
    ")"
)


# --------------------------------------------------------------------------- #
# Requirement 8.2 - styling, stripping, and when colour is allowed
# --------------------------------------------------------------------------- #


def strip_ansi(text: str) -> str:
    """Remove every ANSI escape sequence from `text`.

    Requirement 8.2's literal verb. Used twice, for two different reasons:

    * on the composed output, when colour is off - which is what makes the
      no-colour path a *strip* rather than a second set of format strings, so
      the two paths cannot drift apart; and
    * on every string that came from the archive, always - a document, a folder
      name or a query containing escape bytes would otherwise repaint or retitle
      the user's terminal, and in `--json` consumers it is data, not control.
    """
    return _ANSI_ESCAPE.sub("", text)


def style(text: str, *names: str, enabled: bool = True) -> str:
    """Wrap `text` in the named SGR codes, then reset.

    :param names: Keys of :data:`SGR`, applied in the order given.
    :param enabled: When false, returns `text` unchanged - so a caller can
        thread a colour decision through without branching at each site.
    :raises ValueError: for a name absent from :data:`SGR`. A style name is
        programmer input, not user input, so this is a defect and not a
        diagnostic to be rendered.
    """
    if not enabled or not names:
        return text
    try:
        prefix = "".join(SGR[name] for name in names)
    except KeyError as exc:
        raise ValueError(
            f"unknown style {exc.args[0]!r}; available: {sorted(SGR)}"
        ) from exc
    return f"{prefix}{text}{SGR['reset']}"


def should_colorize(
    *,
    prefer_color: bool = True,
    stream: IO[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Whether ANSI styling may be emitted (requirement 8.2).

    Three independent vetoes, any one of which disables colour:

    1. `prefer_color` is false - the `--no-color` flag, or
       `CliConfig.ansi_colors = false`. The user's stated preference wins over
       everything, which is why it is checked first.
    2. `NO_COLOR` is set to a non-blank value. Per https://no-color.org the
       value itself is irrelevant; "blank means absent" is `config.py`'s
       existing rule for `NPU_RAG_*` variables and is kept identical here.
    3. The stream is not a TTY - requirement 8.2's redirection case. A stream
       with no `isatty`, or one whose `isatty()` raises because the handle is
       closed or detached, counts as not a TTY: a diagnostic about colour must
       never be the thing that fails a search.

    This function is the only part of the module that inspects the process
    environment; the renderers take a plain `color` boolean, which is what
    makes them pure and exactly testable.
    """
    if not prefer_color:
        return False
    variables = os.environ if env is None else env
    if (variables.get(NO_COLOR_VARIABLE) or "").strip():
        return False
    target = sys.stdout if stream is None else stream
    isatty = getattr(target, "isatty", None)
    if not callable(isatty):
        return False
    try:
        # `bool()` is not decoration. `getattr` yields `Any`, so a stream whose
        # `isatty()` returns a truthy non-bool - `1`, or any object - would make
        # this `-> bool` function return that object, and mypy cannot see it.
        # A test asserts identity with `True`, not truthiness.
        return bool(isatty())
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Requirement 8.4 - UTF-8 stdout on Windows
# --------------------------------------------------------------------------- #


def ensure_utf8_stdout(stream: IO[str] | None = None) -> str | None:
    """Force `stream` (default `sys.stdout`) to UTF-8; report what it became.

    Requirement 8.4. On Windows a console stdout is a cp1252 `TextIOWrapper`,
    and writing Devanagari, CJK or even `ü` through it raises
    `UnicodeEncodeError` - the charmap crash the requirement names.

    **The `isinstance` narrowing is load-bearing, not a type-checker
    appeasement.** Design §3.4 prescribes an unconditional
    `sys.stdout.reconfigure(encoding="utf-8")`, but `reconfigure` exists only
    on `io.TextIOWrapper`: `sys.stdout` is typed `TextIO`, which has no such
    method, and under pytest's capture or any redirection to a non-wrapper the
    bare call raises `AttributeError`. The same bug was fixed in
    `tools/transcribe_audio.py` in commit `e828554`.

    The return value exists so the guard is *observable*: `None` means the
    stream could not be reconfigured and the caller is writing through whatever
    encoding it has. A guard that can only be verified by not crashing is a
    guard nobody notices breaking.

    :returns: The stream's encoding after the call, or `None` when the stream
        is not a reconfigurable text wrapper.
    """
    target = sys.stdout if stream is None else stream
    if not isinstance(target, io.TextIOWrapper):
        return None
    target.reconfigure(encoding="utf-8")
    return target.encoding


def emit(text: str, *, stream: IO[str] | None = None) -> None:
    """Write `text` as one line, having first enforced UTF-8 on the stream.

    The encoding enforcement and the write are one operation deliberately:
    requirement 8.4 is a property of *printing*, and a caller that remembered
    `render_search_text()` but forgot `ensure_utf8_stdout()` would crash on the
    first Devanagari result. Tasks 4.1 and 4.2 should print through this.
    """
    target = sys.stdout if stream is None else stream
    ensure_utf8_stdout(target)
    target.write(text if text.endswith("\n") else f"{text}\n")
    target.flush()


# --------------------------------------------------------------------------- #
# Requirement 4.4 - locators
# --------------------------------------------------------------------------- #


def _as_int(value: object) -> int:
    """Narrow a parsed JSON value to `int`, rejecting `bool`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an int, got {type(value).__name__}")
    return value


def _as_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a str, got {type(value).__name__}")
    return value


def _markdown_locator(payload: Mapping[str, object]) -> str:
    """`Lines 45-80`, or `Line 7` for a single-line block.

    `MarkdownLocator.ordinal` - the image ordinal inside the block - is not in
    the label: it disambiguates figures within one line range and would be
    noise on every prose hit. It survives in `--json` under `locator`.
    """
    raw = payload["line_range"]
    if not isinstance(raw, list) or len(raw) != 2:
        raise TypeError("markdown locator line_range must be a pair")
    start = _as_int(raw[0])
    end = _as_int(raw[1])
    if start == end:
        return f"Line {start}"
    return f"Lines {start}-{end}"


def _page_locator(payload: Mapping[str, object]) -> str:
    """`Page 3`."""
    return f"Page {_as_int(payload['page'])}"


def _sheet_locator(payload: Mapping[str, object]) -> str:
    """`Sheet 'Summary'!A1:C9` - the workbook convention, quoted.

    Quoted because sheet names contain spaces ("Bilanz Übersicht"), and
    prefixed with the noun so the segment reads the same way as `Page 3` and
    `Lines 45-80` do.
    """
    sheet = _as_text(payload["sheet"])
    cell_range = _as_text(payload["cell_range"])
    if not cell_range:
        return f"Sheet '{sheet}'"
    return f"Sheet '{sheet}'!{cell_range}"


def _image_file_locator(payload: Mapping[str, object]) -> str:
    """`Image file` - `ImageFileLocator` carries no position; the file is one."""
    return "Image file"


#: Every `locator_type` string `ingest.types._locator_to_dict` can emit, mapped
#: to its label formatter. The four keys are the four `Locator` dataclasses;
#: `index/schema.py:61-63` is what flattens them into `locator_type` +
#: `locator_json`, so this table is the whole of the reverse direction. A test
#: derives the expected key set from `ingest.types._LOCATOR_TYPES` itself, so a
#: fifth locator kind added upstream fails loudly here instead of quietly
#: rendering as a bare type name.
LOCATOR_FORMATTERS: Final[dict[str, Callable[[Mapping[str, object]], str]]] = {
    "markdown": _markdown_locator,
    "page": _page_locator,
    "sheet": _sheet_locator,
    "image_file": _image_file_locator,
}


def _parse_locator(locator_json: str) -> dict[str, object] | None:
    """The locator object, or `None` for anything that is not one."""
    try:
        loaded: object = json.loads(locator_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def format_locator(locator_type: str, locator_json: str) -> str:
    """Render `SearchResult`'s flattened locator as a breadcrumb segment.

    Requirement 4.4's "Page/Line Locator". Degrades rather than raising, at
    every step: an unknown kind, unparseable JSON, a missing field or a field
    of the wrong type all fall back to the `locator_type` itself, and a blank
    type to :data:`UNKNOWN_LOCATOR`. A malformed locator is a defect somewhere
    upstream, but it is not a reason to refuse to show the user the text that
    actually matched their query.
    """
    payload = _parse_locator(locator_json)
    formatter = LOCATOR_FORMATTERS.get(locator_type)
    if formatter is not None and payload is not None:
        try:
            return formatter(payload)
        except (KeyError, TypeError, ValueError):
            pass
    return locator_type.strip() or UNKNOWN_LOCATOR


# --------------------------------------------------------------------------- #
# Paths, documents, snippets
# --------------------------------------------------------------------------- #


def _path_parts(file_path: str) -> tuple[str, ...]:
    """The path's components, Windows long-path prefix already removed.

    `order._components` is reused rather than re-derived, per implementation
    note 2.2: it applies `order._strip_long_prefix` (`order.py:272-274`) and
    accepts both separator spellings. Every `file_path` in a real index carries
    `\\\\?\\` or `\\\\?\\UNC\\` (`ingest/discover.py:148-155` stamps it,
    `index/schema.py:76` stores it verbatim, measured 7/7 on real paths);
    unstripped, every breadcrumb would read `?/D:/.../<Folder>`.
    """
    return _components(file_path)


def _display_path(file_path: str) -> str:
    """The path as a human and a script should see it: stripped, forward-slashed.

    Separators are unified because this value goes into `--json`, where a
    Windows path would otherwise arrive as `"D:\\\\Source\\\\..."` - correct
    JSON and miserable to read or to match in `jq`. It doubles as the document
    identity key, so two files sharing a leaf name in different folders are
    never counted as one document.

    **This value is for reading, never for opening.** `order._components`
    discards empty components (`order.py:272-274`), so a UNC source loses its
    leading `\\\\`: `\\\\?\\UNC\\nas\\share\\archive\\f.md` displays as
    `nas/share/archive/f.md`, which no `open()` can resolve. That is not a
    length problem and no path is long enough to avoid it -
    `ingest/discover.py:148-155` stamps `\\\\?\\UNC\\` on every file under a
    share, so it is realised the first time `--archive-path` points at one.
    `--json` therefore also carries `source_path`, the `file_path` column
    verbatim, which is the openable form.
    """
    return "/".join(_path_parts(file_path))


def _document_name(file_path: str) -> str:
    """The document's file name - what the user can open and what §3.4 shows.

    `SearchResult.title` is deliberately not used here: it is the extracted H1,
    which may be absent, may be shared by every file in a series, and is not
    what a `--folder` argument or a file manager refers to. The title travels
    in `--json` as its own field.
    """
    parts = _path_parts(file_path)
    return parts[-1] if parts else UNKNOWN_DOCUMENT


def _snippet(text: str, *, limit: int = SNIPPET_LIMIT) -> str:
    """One line of preview: escapes removed, whitespace collapsed, truncated.

    Sanitised *before* truncation, so a cut can never land inside an escape
    sequence and leave its tail as visible junk.
    """
    collapsed = " ".join(strip_ansi(text).split())
    if len(collapsed) <= limit:
        return collapsed
    head = collapsed[:limit]
    boundary = head.rfind(" ")
    if boundary >= limit // 2:
        head = head[:boundary]
    return f"{head.rstrip()}{SNIPPET_ELLIPSIS}"


# --------------------------------------------------------------------------- #
# Requirement 6.3 - provider attribution
# --------------------------------------------------------------------------- #


def _require_provider(provider: str) -> str:
    """The provider name, or a `ValueError` - never a blank attribution.

    Requirement 6.3 says *every* execution reports its provider. A blank string
    would render `[Provider: ]`, which is attribution that has silently gone
    missing - the exact shape requirement 6.2 forbids one layer down.
    """
    if not provider.strip():
        raise ValueError(
            "provider must name the execution provider that ran this query "
            f"(e.g. {NPU_PROVIDER!r} or {CPU_PROVIDER!r}); got {provider!r}"
        )
    return provider


def format_provider(provider: str, *, color: bool = False) -> str:
    """`[Provider: VitisAIExecutionProvider (NPU)]` (requirement 6.3).

    Reported verbatim with one exception: the exact Vitis AI provider name also
    gets `(NPU)`, because that is the fact the user is looking for and the name
    alone does not say it. Nothing else is translated or embellished - see
    ruling 4.
    """
    name = strip_ansi(_require_provider(provider))
    suffix = " (NPU)" if name == NPU_PROVIDER else ""
    return style(f"[Provider: {name}{suffix}]", "bold", "magenta", enabled=color)


# --------------------------------------------------------------------------- #
# Requirement 4.4 - the breadcrumb
# --------------------------------------------------------------------------- #


def format_breadcrumb(
    *,
    folder: str,
    document: str,
    document_index: int,
    document_total: int,
    chunk_index: int,
    locator: str,
    color: bool = False,
) -> str:
    """`[Folder: F] -> [Doc 1/13: D.md] -> [Chunk #12 | Lines 45-80]`.

    Requirement 4.4's three segments, in its order, with its arrow. Every
    data-derived part is passed through :func:`strip_ansi` so a document name
    carrying escape bytes cannot restyle the line.

    :param locator: An already-formatted label - see :func:`format_locator`.
    """
    segments = (
        style(f"[Folder: {strip_ansi(folder)}]", "cyan", enabled=color),
        style(
            f"[Doc {document_index}/{document_total}: {strip_ansi(document)}]",
            "bold",
            enabled=color,
        ),
        style(
            f"[Chunk #{chunk_index} | {strip_ansi(locator)}]",
            "yellow",
            enabled=color,
        ),
    )
    return style(BREADCRUMB_SEPARATOR, "dim", enabled=color).join(segments)


# --------------------------------------------------------------------------- #
# The rendered sequence
# --------------------------------------------------------------------------- #


class _Hit:
    """One result in render order, with the context a breadcrumb needs.

    A hand-written class and not a dataclass: `dataclasses` is not otherwise on
    this import path and requirement 1.2 is measured in modules loaded.
    """

    __slots__ = (
        "document",
        "document_index",
        "document_total",
        "folder",
        "result",
        "tags",
    )

    def __init__(
        self,
        *,
        folder: str,
        tags: tuple[str, ...],
        document: str,
        document_index: int,
        document_total: int,
        result: SearchResult,
    ) -> None:
        self.folder = folder
        self.tags = tags
        self.document = document
        self.document_index = document_index
        self.document_total = document_total
        self.result = result


def _folder_tags(
    folder: str, folder_tags: Mapping[str, Sequence[str]] | None
) -> tuple[str, ...]:
    """The tags the caller attributes to `folder`, sanitised, order preserved."""
    if folder_tags is None:
        return ()
    return tuple(strip_ansi(str(tag)) for tag in folder_tags.get(folder, ()))


def _sequence(
    response: SearchResponse,
    *,
    archive_root: str | Path | None,
    group_by_folder: bool,
) -> list[tuple[str, SearchResult]]:
    """Every result paired with its folder, in the order it will be rendered.

    Grouped, this is `order.group_and_order_results` flattened - whose mapping
    order is contract (its ruling 3: best-scoring folder first) and whose
    per-folder order is requirement 4.2's narrative sequence. Ungrouped, it is
    the response's own ranking, untouched; the folder label is still computed
    for every hit, because requirement 4.4 wants the folder segment on every
    breadcrumb whether or not grouping was asked for.
    """
    if group_by_folder:
        grouped = group_and_order_results(response.results, archive_root=archive_root)
        return [(folder, result) for folder, items in grouped.items() for result in items]
    return [
        (folder_of(result.file_path, archive_root=archive_root), result)
        for result in response.results
    ]


def _hits(
    response: SearchResponse,
    *,
    archive_root: str | Path | None,
    folder_tags: Mapping[str, Sequence[str]] | None,
    group_by_folder: bool,
) -> list[_Hit]:
    """The render sequence, with each hit's document position filled in.

    Two passes, because the denominator is not known until the folder's results
    have all been seen. Document identity is the full display path and not the
    file name, so `author-a/images/plate.md` and `author-b/images/plate.md` are
    two documents (ruling 1).
    """
    sequence = _sequence(
        response, archive_root=archive_root, group_by_folder=group_by_folder
    )

    positions: dict[str, dict[str, int]] = {}
    for folder, result in sequence:
        documents = positions.setdefault(folder, {})
        key = _display_path(result.file_path)
        if key not in documents:
            documents[key] = len(documents) + 1

    hits: list[_Hit] = []
    for folder, result in sequence:
        documents = positions[folder]
        hits.append(
            _Hit(
                folder=folder,
                tags=_folder_tags(folder, folder_tags),
                document=_document_name(result.file_path),
                document_index=documents[_display_path(result.file_path)],
                document_total=len(documents),
                result=result,
            )
        )
    return hits


def _folder_rows(hits: Sequence[_Hit]) -> list[dict[str, object]]:
    """Per-folder metadata in render order: tags, hit count, document count.

    `document_count` is ruling 1's denominator, stated once per folder so a
    `jq` consumer need not derive it from `document_total` on every result.
    """
    counts: dict[str, int] = {}
    tags: dict[str, tuple[str, ...]] = {}
    totals: dict[str, int] = {}
    for hit in hits:
        counts[hit.folder] = counts.get(hit.folder, 0) + 1
        tags[hit.folder] = hit.tags
        totals[hit.folder] = hit.document_total
    return [
        {
            "folder": folder,
            "tags": list(tags[folder]),
            "result_count": count,
            "document_count": totals[folder],
        }
        for folder, count in counts.items()
    ]


def _tags_label(tags: Sequence[str]) -> str:
    """`(Tag: siddha)` for one, `(Tags: a, b)` for several - design §3.4's form."""
    if len(tags) == 1:
        return f"(Tag: {tags[0]})"
    return f"(Tags: {', '.join(tags)})"


# --------------------------------------------------------------------------- #
# Requirement 8.1 - the terminal rendering
# --------------------------------------------------------------------------- #


def _header_lines(response: SearchResponse, provider: str) -> list[str]:
    """The two lines that carry the query, the mode, the provider and the scale.

    The counts are here and not only in the folder banners because they are
    what makes ruling 1's denominator honest: the user can see that `Doc 1/2`
    is drawn from this many results out of this many candidates.
    """
    query = strip_ansi(response.query)
    mode = strip_ansi(str(response.mode))
    elapsed = f"{response.elapsed_seconds:.{ELAPSED_DIGITS}f}"
    return [
        " ".join(
            [
                style(f"[Query: {query}]", "dim"),
                style(f"[Mode: {mode}]", "dim"),
                format_provider(provider, color=True),
            ]
        ),
        style(
            f"[Results: {len(response.results)}] "
            f"[Candidates: {response.total_candidates}] "
            f"[Elapsed: {elapsed}s]",
            "dim",
        ),
    ]


def _banner_lines(hit: _Hit) -> list[str]:
    """Design §3.4's folder banner: a rule, the folder with its tag, a rule."""
    rule = style("=" * RULE_WIDTH, "dim")
    label = style(f"[Folder: {strip_ansi(hit.folder)}]", "bold", "cyan")
    if hit.tags:
        label = f"{label} {style(_tags_label(hit.tags), 'magenta')}"
    return [rule, label, rule]


def _hit_lines(hit: _Hit) -> list[str]:
    """One hit: its full breadcrumb and score, its snippet, its context.

    Design §3.4's sample omits the folder from the per-hit line because the
    banner above carries it. The folder is kept here anyway - see the
    `render_search_text` docstring for why - and the sample's second entry,
    `[Chunk #13 | Context Continuation]`, is rendered as a labelled `[Context]`
    block rather than as a second hit, because `ContextExpander.expand`
    (`index/expansion.py:76-95`) returns one result per hit with
    `expanded_context` filled in. Synthesising a second `▶` line would mean
    inventing a chunk ordinal and a locator that no result carries.
    """
    result = hit.result
    breadcrumb = format_breadcrumb(
        folder=hit.folder,
        document=hit.document,
        document_index=hit.document_index,
        document_total=hit.document_total,
        chunk_index=result.chunk_index,
        locator=format_locator(result.locator_type, result.locator_json),
        color=True,
    )
    score = style(f"[Score: {result.score:.{SCORE_DIGITS}f}]", "green")
    lines = [
        f"  {style(HIT_MARKER, 'cyan')} {breadcrumb} {score}",
        f'      "{_snippet(result.text)}"',
    ]
    context = result.expanded_context
    if context is not None and context != result.text:
        lines.append(f'      {style("[Context]", "dim")} "{_snippet(context)}"')
    return lines


def render_search_text(
    response: SearchResponse,
    *,
    provider: str,
    archive_root: str | Path | None = None,
    group_by_folder: bool = False,
    folder_tags: Mapping[str, Sequence[str]] | None = None,
    color: bool = False,
) -> str:
    """Render a search response for a terminal (requirements 8.1, 8.2, 4.4, 6.3).

    The composition is **always** styled and then stripped when `color` is
    false, rather than styled conditionally. Two consequences, both wanted: the
    coloured and plain outputs cannot drift apart, since one is literally
    `strip_ansi` of the other; and requirement 8.2's "strip all ANSI styling"
    is implemented as the strip it says it is.

    Every hit line carries the **full** three-segment breadcrumb, including the
    folder, even in grouped mode where the banner names the folder too.
    Requirement 4.4 asks for full attribution "for every hit", and requirement
    8.2 contemplates redirected output: a line fished out of a pipe by `grep`
    has to say which folder it came from, and a banner scrolled off the top of
    a screen says nothing. This is a deliberate departure from design §3.4's
    sample, which drops the folder from the hit line.

    :param provider: The *active* execution provider's name - see ruling 4.
        Required, and refused if blank.
    :param archive_root: Resolved and absolute, please: `order.folder_of`
        degrades to a full parent path for a root that does not match, which is
        visible but not what the user wants (implementation note 2.2).
    :param folder_tags: Folder label to tag names - see ruling 2.
    :param color: From :func:`should_colorize`.
    """
    _require_provider(provider)
    hits = _hits(
        response,
        archive_root=archive_root,
        folder_tags=folder_tags,
        group_by_folder=group_by_folder,
    )

    lines = [*_header_lines(response, provider), ""]
    if not hits:
        lines.append(NO_RESULTS_LINE)
    else:
        current: str | None = None
        for hit in hits:
            if group_by_folder and hit.folder != current:
                current = hit.folder
                lines.extend(_banner_lines(hit))
            lines.extend(_hit_lines(hit))
            lines.append("")

    text = "\n".join(lines).rstrip()
    return text if color else strip_ansi(text)


# --------------------------------------------------------------------------- #
# Requirement 8.3 - the JSON rendering
# --------------------------------------------------------------------------- #


def _result_payload(hit: _Hit) -> dict[str, object]:
    """One result as JSON. Key order is the schema and a test pins it.

    The text fields are carried **verbatim**: untruncated, with their own
    escape bytes intact. `--json` is data for `jq`, and a renderer that
    silently shortened or rewrote a document's text would be corrupting the
    payload rather than presenting it. The pretty output is where truncation
    and sanitisation belong.

    Two path fields, for two different jobs: `file_path` is the readable
    display form and `source_path` is the `file_path` column exactly as the
    index stores it, which is the only one that can be opened (see
    :func:`_display_path`). A consumer that merely prints paths can ignore
    `source_path`; one that opens them must use it and nothing else.
    """
    result = hit.result
    locator_label = format_locator(result.locator_type, result.locator_json)
    return {
        "folder": hit.folder,
        "tags": list(hit.tags),
        "document": hit.document,
        "document_index": hit.document_index,
        "document_total": hit.document_total,
        "file_path": _display_path(result.file_path),
        "source_path": result.file_path,
        "title": result.title,
        "chunk_id": result.chunk_id,
        "chunk_index": result.chunk_index,
        "chunk_kind": result.chunk_kind,
        "headings": list(result.headings),
        "locator_type": result.locator_type,
        "locator": _parse_locator(result.locator_json),
        "locator_label": locator_label,
        "score": result.score,
        "text": result.text,
        "expanded_context": result.expanded_context,
        "token_count": result.token_count,
        "dense_distance": result.dense_distance,
        "dense_rank": result.dense_rank,
        "lexical_score": result.lexical_score,
        "lexical_rank": result.lexical_rank,
        "breadcrumb": format_breadcrumb(
            folder=hit.folder,
            document=hit.document,
            document_index=hit.document_index,
            document_total=hit.document_total,
            chunk_index=result.chunk_index,
            locator=locator_label,
        ),
    }


def render_search_json(
    response: SearchResponse,
    *,
    provider: str,
    archive_root: str | Path | None = None,
    group_by_folder: bool = False,
    folder_tags: Mapping[str, Sequence[str]] | None = None,
    indent: int | None = JSON_INDENT,
) -> str:
    """Render a search response as structured JSON (requirements 8.3, 6.3).

    Flat `results`, every entry naming its `folder`; `folders` carries the
    per-group metadata in render order. The schema does not change with
    `group_by_folder` - only the ordering of `results` and the `grouped` flag
    do - so one `jq` filter works either way. See rulings 3 and 4.

    `ensure_ascii=False`: requirement 8.4's scripts stay literal, so a human
    reading the payload and `jq` filtering on it both see Devanagari rather
    than `\\u0915`. Pair it with :func:`emit` on Windows.
    """
    _require_provider(provider)
    hits = _hits(
        response,
        archive_root=archive_root,
        folder_tags=folder_tags,
        group_by_folder=group_by_folder,
    )
    payload: dict[str, object] = {
        "query": response.query,
        "mode": str(response.mode),
        "provider": provider,
        "grouped": group_by_folder,
        "total_candidates": response.total_candidates,
        "elapsed_seconds": response.elapsed_seconds,
        "folders": _folder_rows(hits),
        "results": [_result_payload(hit) for hit in hits],
    }
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def render_search(
    response: SearchResponse,
    *,
    provider: str,
    as_json: bool,
    archive_root: str | Path | None = None,
    group_by_folder: bool = False,
    folder_tags: Mapping[str, Sequence[str]] | None = None,
    color: bool = False,
    indent: int | None = JSON_INDENT,
) -> str:
    """Dispatch to the JSON or the terminal renderer.

    One call site for task 4.1, so that `--json` cannot be wired up in a way
    that forgets the provider: both branches require it.
    """
    if as_json:
        return render_search_json(
            response,
            provider=provider,
            archive_root=archive_root,
            group_by_folder=group_by_folder,
            folder_tags=folder_tags,
            indent=indent,
        )
    return render_search_text(
        response,
        provider=provider,
        archive_root=archive_root,
        group_by_folder=group_by_folder,
        folder_tags=folder_tags,
        color=color,
    )
