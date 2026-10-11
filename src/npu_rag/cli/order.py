"""Folder grouping and narrative sequence ordering (task 2.2).

Design anchor: `.kiro/specs/search-cli/design.md` §3.3 (`order.py`);
requirements 4.1 and 4.2.

Requirement 4.2 asks for one thing: when `--group-by-folder` is on, hits are
"grouped under their parent folder and ordered by natural document sequence
(chronological dates or session numbers like `Day 1`, `Part 1`) and
intra-document `chunk_index`". A lecture series read in score order is a pile
of shards; read in session order it is the course again.

Design §3.3 is a sketch rather than a specification - it calls an undefined
`_extract_folder`, annotates the key as `tuple[Any, ...]`, and is silent on
folder identity, between-folder order and ties. The four rulings this module
makes are set out below, each with the reason and each pinned by a test in
`tests/cli/test_cli_order.py`.

----------------------------------------------------------------------------
Ruling 1 - the folder is the first path component beneath the archive root
----------------------------------------------------------------------------

`npu_rag.ingest.discover._author` derives the indexed `author` column as "the
first directory component beneath that file's root", and task 2.1's `--folder`
filter matches exactly that column. Grouping on the same notion is what makes
`npu-rag search --folder X --group-by-folder` return a single coherent group
instead of a scatter.

The alternative - the file's *immediate* parent directory - fails requirement
4.2 on its own terms. In a nested archive `Author/Course/Day 1/notes.md`, the
immediate parent is `Day 1`, so every session becomes its own group and there
is no "document sequence" left inside any group to order. The sequence tokens
requirement 4.2 names (`Day 1`, `Part 1`) have to live *inside* the group, not
be the group.

`SearchResult` carries no `author` column (see `npu_rag.index.types`), only
`file_path`, so the root cannot be recovered from a result alone.
`group_and_order_results` therefore takes an optional `archive_root`; task 4.1
holds one as `CliConfig.archive_path` and should pass it resolved and
absolute. Without it - or for a path that does not lie beneath it - the
fallback is the file's **full parent directory path**, which over-splits a
nested archive but can never merge two unrelated folders that happen to share
a leaf name (`.../author-a/images` and `.../author-b/images`). Over-splitting
is visible to the user; a false merge is not.

----------------------------------------------------------------------------
Ruling 2 - the sort key is TYPE-STABLE, by construction
----------------------------------------------------------------------------

**Invariant: every position of a key compares against the same Python type at
that position, for every possible input.** A key that mixes `int` and `str` at
one position raises `TypeError: '<' not supported between instances of 'int'
and 'str'` the first time a folder holds both `Day 2.md` and `Intro.md` -
which is the ordinary case, not an edge case, so `sorted()` would crash on
real archive data. Design §3.3's `tuple[Any, ...]` invites precisely that.

The key is therefore built from fixed-shape atoms. Each path component becomes
a tuple of `(kind, number, text)` triples where a digit run is
`(_NUMBER, value, "")` and everything else is `(_TEXT, 0, casefolded)`: the
number slot is always an `int` and the text slot always a `str`, whichever
kind an atom is. Numbers compare as numbers, so `Day 2` precedes `Day 10`.

----------------------------------------------------------------------------
Ruling 3 - folders come back best-score-first, and that order is contract
----------------------------------------------------------------------------

Requirement 4.2 constrains the order *within* a folder and says nothing about
the order *between* folders, but a caller rendering the grouping has to emit
them in some order and "whatever `dict` insertion happened to do" is not an
answer a user can predict. These are search results, so the folder holding the
best hit leads: groups are ordered by their highest member score, descending,
ties broken by folder name ascending. The returned mapping's iteration order
is part of this function's contract - `render.py` (task 3.1) may iterate it
directly.

----------------------------------------------------------------------------
Ruling 4 - ties and unparseable names are total, never incidental
----------------------------------------------------------------------------

The key ends with the normalized path spelled exactly as given, so two
*distinct* paths can never compare equal however unparseable their names are
(`_download-meta.txt`, or the UUID-named PDFs this archive is full of) and
however they collide after case folding or leading-zero normalization
(`Day 01.md` and `Day 1.md` tokenize identically up to that point). Ordering
is thus a function of the input set alone and never of the order results
happened to arrive in. Two results for the same chunk of the same file are the
one remaining tie, and `list.sort` is stable, so they keep their input order.

----------------------------------------------------------------------------
Startup cost (requirement 1.2)
----------------------------------------------------------------------------

`order.py` is on the `search` path. Importing `npu_rag.index.types` for
`SearchResult` pulls in `lancedb`, `torch` and `transformers` (probed
2026-09-21: 3266 modules), so the import is confined to `TYPE_CHECKING` and
the runtime code touches only `file_path`, `chunk_index` and `score`. A probe
test pins this.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from npu_rag.index.types import SearchResult

__all__ = [
    "ROOT_FOLDER",
    "SESSION_KEYWORDS",
    "folder_of",
    "group_and_order_results",
    "natural_sort_key",
]

#: The group key for a document sitting directly in the archive root, and the
#: fallback for a path with no parent at all. `discover._author` spells the
#: same case as `""`, which is a poor label for a caller rendering
#: `[Folder: ...]`; `"."` names the root the way a shell does.
ROOT_FOLDER: Final = "."

#: Words that mark a token as a position in a series rather than a quantity.
#: Chosen from what requirement 4.2 names (`Day`, `Part`) plus what the real
#: archive uses; see the module test for the survey. A number word is only read
#: as a sequence number when it sits directly beside one of these, so
#: `the-three-layers-of-ai` keeps its `three` as prose while `ai-moats-part-one`
#: becomes `part 1`.
SESSION_KEYWORDS: Final[tuple[str, ...]] = (
    "day",
    "part",
    "volume",
    "vol",
    "session",
    "episode",
    "chapter",
    "lesson",
    "week",
    "module",
)

# Kind discriminators. A digit run sorts before text at the same position,
# which puts a dated or numbered document ahead of an untitled sibling.
_NUMBER: Final = 0
_TEXT: Final = 1

_Atom = tuple[int, int, str]
_Component = tuple[_Atom, ...]

#: What `natural_sort_key` returns: the per-component atoms, then the
#: normalized path as the total-order tiebreak (ruling 4).
SortKey = tuple[tuple[_Component, ...], str]

# Windows long-path prefixes. `npu_rag.ingest.discover` stamps every discovered
# file with `\\?\`, and `file_path` is `str(Path)` of that, so the prefix can
# reach this module. Splitting it naively would leave a bogus `?` component.
_LONG_PREFIX: Final = "\\\\?\\"
_UNC_LONG_PREFIX: Final = "\\\\?\\UNC\\"

#: Both separator spellings, always. Implementation note 2.1: `file_path` is
#: backslash-separated on Windows, and no test on this platform can naturally
#: exercise the `/` branch, so the tests construct POSIX values by hand.
_SEPARATORS: Final = re.compile(r"[\\/]+")

_DIGIT_RUN: Final = re.compile(r"\d+")

_NUMBER_WORDS: Final[dict[str, int]] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}

# Longest alternative first. This is belt-and-braces and is *not* what makes
# `nineteen` read as 19 rather than 9 - the trailing `(?![a-z0-9])` below is,
# by refusing the short alternative and forcing the alternation to backtrack.
# Verified by mutation: reversing both sort orders changes no observable
# behaviour. The ordering is kept because it saves the backtracking and because
# a future edit that relaxed the lookahead would otherwise silently regress.
_KEYWORD_ALT: Final = "|".join(sorted(SESSION_KEYWORDS, key=len, reverse=True))
_WORD_ALT: Final = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

# What separates two words inside one path component: a space, and the three
# characters this archive's slugs actually use.
_WORD_GAP: Final = r"[\s._\-]+"

# `Part One`, `Day-two`, `volumes_three`.
_KEYWORD_THEN_WORD: Final = re.compile(
    rf"(?<![a-z0-9])((?:{_KEYWORD_ALT})s?{_WORD_GAP})({_WORD_ALT})(?![a-z0-9])"
)
# `First Day`, `second-part` - design §3.4's own sample file is
# `Savarni Nath - First Day.md`, so this order has to work too.
_WORD_THEN_KEYWORD: Final = re.compile(
    rf"(?<![a-z0-9])({_WORD_ALT})({_WORD_GAP}(?:{_KEYWORD_ALT})s?)(?![a-z0-9])"
)


# --------------------------------------------------------------------------- #
# Path decomposition
# --------------------------------------------------------------------------- #


def _strip_long_prefix(text: str) -> str:
    """Drop a Windows `\\\\?\\` or `\\\\?\\UNC\\` prefix.

    Mirrors `npu_rag.ingest.discover._without_long_prefix` rather than importing
    it: that module is private to `ingest` and pulls `IngestConfig` in with it,
    which has no business on the search path.

    Not an edge case on Windows but the only case: `discover._windows_long_path`
    stamps `\\\\?\\` onto every absolute path it walks, and that value reaches
    `file_path` unchanged (`discover.py:110` -> `chunk.py:160` ->
    `schema.py:76`). Left in place it becomes a bogus `?` component and every
    group in the archive would be labelled `?/D:/.../<Author>`.

    The UNC form is tested **first**, deliberately: `_UNC_LONG_PREFIX` begins
    with `_LONG_PREFIX`, so matching the shorter one first would strip four
    characters and leave a bogus `UNC` component behind.

    One honest caveat: the `\\\\` put back in front of a UNC path is *not*
    observable through this module, because `_components` discards empty
    components and so reaches the same tuple either way - a mutation dropping
    it survives the suite. It is kept so that this function is correct as a
    function, which matters because task 3.1's breadcrumb renderer meets the
    same prefix and should be able to reuse it rather than re-derive it.
    """
    if text.startswith(_UNC_LONG_PREFIX):
        return "\\\\" + text[len(_UNC_LONG_PREFIX) :]
    if text.startswith(_LONG_PREFIX):
        return text[len(_LONG_PREFIX) :]
    return text


def _components(file_path: str) -> tuple[str, ...]:
    """Split a path into non-empty components, either separator spelling."""
    return tuple(part for part in _SEPARATORS.split(_strip_long_prefix(file_path)) if part)


def _normalized(file_path: str) -> str:
    """The path with separators unified, case preserved - the tiebreak value."""
    return "/".join(_components(file_path))


# --------------------------------------------------------------------------- #
# Sequence tokens
# --------------------------------------------------------------------------- #


def _keyword_then_word_digits(match: re.Match[str]) -> str:
    """`part one` -> `part 1`; the keyword and its gap are kept verbatim."""
    return f"{match.group(1)}{_NUMBER_WORDS[match.group(2)]}"


def _word_then_keyword_digits(match: re.Match[str]) -> str:
    """`first day` -> `1 day`; the keyword and its gap are kept verbatim."""
    return f"{_NUMBER_WORDS[match.group(1)]}{match.group(2)}"


def _digits_for_session_words(text: str) -> str:
    """Rewrite `part one` as `part 1`, but leave `three layers` alone.

    The anchor is deliberate. Every archive surveyed for this task is dominated
    by prose slugs in which cardinals are quantities - `the-three-layers-of-ai`,
    `the-five-level-hierarchy` - and reading those as sequence numbers would
    scatter a folder rather than order it. Requiring a `SESSION_KEYWORDS`
    neighbour keeps the two genuine series in that corpus (`ai-moats-part-one`
    and `-part-two`) and the spec's `Part 1` / `First Day` forms, and touches
    nothing else.
    """
    text = _KEYWORD_THEN_WORD.sub(_keyword_then_word_digits, text)
    return _WORD_THEN_KEYWORD.sub(_word_then_keyword_digits, text)


def _atoms(component: str) -> _Component:
    """Tokenize one path component into same-shaped `(kind, number, text)` atoms.

    Digit runs become numbers, so `Day 2` precedes `Day 10`; everything between
    them stays text. No separate date parser is needed: `20240310_182951_x.md`
    yields `20240310`, `182951` and the slug, and `2026-03-29` yields year,
    month and day, and both already sort chronologically as integers.
    """
    text = _digits_for_session_words(component.casefold())
    atoms: list[_Atom] = []
    position = 0
    for match in _DIGIT_RUN.finditer(text):
        if match.start() > position:
            atoms.append((_TEXT, 0, text[position : match.start()]))
        atoms.append((_NUMBER, int(match[0]), ""))
        position = match.end()
    if position < len(text):
        atoms.append((_TEXT, 0, text[position:]))
    return tuple(atoms)


def natural_sort_key(file_path: str) -> SortKey:
    """A total, type-stable ordering key for a document path.

    Parses chronological and session tokens the way design §3.3 asks - `Day 1`,
    `Part 2`, `2026-03-29`, and the `YYYYMMDD_HHMMSS_slug` form that makes up
    90% of the real archive - by splitting digit runs out of each path
    component and comparing them as integers.

    **Type stability is an invariant, not an accident.** Position 0 of the
    result is always a tuple of components, position 1 always a `str`; inside a
    component every atom is `(int, int, str)`. `sorted()` over any mixture of
    names - numbered, dated, prose, UUID, empty - therefore cannot raise
    `TypeError`. Ruling 2 in the module docstring explains why this is the
    defect worth guarding against.

    Components are compared one at a time rather than over a flattened string,
    so a directory boundary is never crossed by a token: `Course/Day 2` and
    `Course2/Day 1` cannot interleave.

    :param file_path: `SearchResult.file_path` - `str(Path)`, so
        backslash-separated on Windows and possibly `\\\\?\\`-prefixed. Both
        separator spellings are accepted.
    :returns: A key usable directly with `sorted()`, `min()` and `max()`.
    """
    components = _components(file_path)
    return tuple(_atoms(component) for component in components), _normalized(file_path)


# --------------------------------------------------------------------------- #
# Folder identity
# --------------------------------------------------------------------------- #


def _is_beneath(root: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    """Whether `parts` starts with `root`, under the platform's casing rule.

    `os.path.normcase` and not `str.casefold`: on Windows two spellings of one
    directory are one directory, on POSIX they are two, and grouping must not
    invent a merge the filesystem does not have. Same call
    `discover._dedup_key` uses.
    """
    if len(parts) <= len(root):
        return False
    return all(os.path.normcase(a) == os.path.normcase(b) for a, b in zip(root, parts))


def _fallback_folder(parts: tuple[str, ...]) -> str:
    """The full parent directory path - see ruling 1 for why not the leaf name."""
    if len(parts) <= 1:
        return ROOT_FOLDER
    return "/".join(parts[:-1])


def folder_of(file_path: str, *, archive_root: str | Path | None = None) -> str:
    """The folder a document is grouped under (requirement 4.2).

    With `archive_root`, this is the first path component beneath it - the same
    value `npu_rag.ingest.discover` stores in the `author` column and the same
    one task 2.1's `--folder` filter matches, so a `--folder X` search grouped
    by folder yields exactly the group `X`. A document lying directly in the
    root has no such component and is grouped under `ROOT_FOLDER`.

    Without a root, or for a path that does not lie beneath the one given, the
    fallback is the document's full parent directory path.

    :param file_path: `SearchResult.file_path`.
    :param archive_root: The archive root, ideally absolute and resolved -
        `CliConfig.archive_path` is relative by default and will simply not
        match an absolute `file_path`, which degrades to the fallback rather
        than mis-grouping.
    """
    parts = _components(file_path)
    if archive_root is not None:
        root = _components(str(archive_root))
        if root and _is_beneath(root, parts):
            below = parts[len(root) :]
            return below[0] if len(below) > 1 else ROOT_FOLDER
    return _fallback_folder(parts)


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #


def group_and_order_results(
    results: Sequence[SearchResult],
    *,
    archive_root: str | Path | None = None,
) -> dict[str, list[SearchResult]]:
    """Group hits by folder and restore each folder's narrative order.

    Requirement 4.2. Within a folder, documents follow their natural sequence -
    session numbers and dates - and chunks of one document follow their
    `chunk_index`, so a series reads as the series again instead of as score
    fragments. Between folders, the best hit leads (ruling 3); the returned
    mapping's iteration order is part of the contract.

    :param results: The ranked hits, in any order. Not mutated.
    :param archive_root: Passed to `folder_of`; see ruling 1.
    :returns: Folder name to its hits. Every input result appears exactly once.
    """
    grouped: dict[str, list[SearchResult]] = {}
    for result in results:
        folder = folder_of(result.file_path, archive_root=archive_root)
        grouped.setdefault(folder, []).append(result)

    for items in grouped.values():
        items.sort(key=lambda r: (natural_sort_key(r.file_path), r.chunk_index))

    return {
        folder: items
        for folder, items in sorted(
            grouped.items(),
            key=lambda entry: (-max(item.score for item in entry[1]), entry[0]),
        )
    }
