"""Tests for `npu_rag.cli.order` (task 2.2).

Requirements 4.1 and 4.2; design §3.3.

Two habits govern every ordering test here.

**No fixture may already be in the answer's order.** An input whose given
order equals its expected order is passed by the identity function, so it
proves nothing about the sort. Every sequence below is handed over scrambled.

**No membership assertions.** `x in result` cannot see position, and position
is the entire subject of this task, so expectations are full sequences.

Where a naive implementation would coincidentally agree with the correct one -
`part one` before `part two` is the same answer alphabetically as numerically -
the fixture is chosen so the two disagree (`two` before `three` numerically,
`three` before `two` alphabetically).
"""

from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import npu_rag.cli.order as order
from npu_rag.cli.order import (
    ROOT_FOLDER,
    SESSION_KEYWORDS,
    folder_of,
    group_and_order_results,
    natural_sort_key,
)
from npu_rag.index.types import SearchResult

ARCHIVE = r"D:\Archive"


def _hit(
    file_path: str,
    *,
    chunk_index: int = 0,
    score: float = 0.5,
) -> SearchResult:
    """A `SearchResult` carrying only the three fields `order.py` reads."""
    return SearchResult(
        chunk_id=f"{file_path}#{chunk_index}",
        file_path=file_path,
        title=None,
        chunk_kind="prose",
        chunk_index=chunk_index,
        headings=(),
        locator_type="line",
        locator_json="{}",
        text="",
        score=score,
    )


def _sorted_paths(paths: list[str]) -> list[str]:
    return sorted(paths, key=natural_sort_key)


def _paths(results: list[SearchResult]) -> list[tuple[str, int]]:
    return [(r.file_path, r.chunk_index) for r in results]


# --------------------------------------------------------------------------- #
# The design's two attribute references, and the third this module adds
# --------------------------------------------------------------------------- #


def test_search_result_exposes_the_fields_this_module_reads() -> None:
    """Design §3.3 dereferences `file_path` and `chunk_index`; ruling 3 adds `score`."""
    hit = _hit(r"D:\Archive\A\x.md", chunk_index=4, score=0.75)
    assert hit.file_path == r"D:\Archive\A\x.md"
    assert hit.chunk_index == 4
    assert hit.score == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# Ruling 2 - type stability
# --------------------------------------------------------------------------- #

#: Deliberately mixed: numbered, dated, prose, UUID, dot-leading, non-ASCII and
#: a bare digit name. A folder holding both `Day 2.md` and `Intro.md` is the
#: ordinary case, and it is exactly what a key mixing `int` and `str` at one
#: position dies on.
MIXED_BATCH: list[str] = [
    r"D:\Archive\A\Day 2.md",
    r"D:\Archive\A\Intro.md",
    r"D:\Archive\A\10.md",
    r"D:\Archive\A\2026-03-29.md",
    r"D:\Archive\A\20240310_182951_the-innovation-flywheel.md",
    r"D:\Archive\A\5e1aaf12-2413-43ae-97b4-9fdc939c2350.pdf",
    r"D:\Archive\A\_download-meta.txt",
    r"D:\Archive\A\Part Three.md",
    r"D:\Archive\A\बहुत.md",
    r"D:\Archive\A\1",
]


def test_sorting_a_mixed_batch_does_not_raise_and_keeps_every_element() -> None:
    result = _sorted_paths(list(MIXED_BATCH))
    assert sorted(result) == sorted(MIXED_BATCH)
    assert len(result) == len(MIXED_BATCH)


def test_every_key_position_holds_a_single_type() -> None:
    """The invariant stated in `natural_sort_key`'s docstring, checked structurally."""
    for path in MIXED_BATCH:
        components, tiebreak = natural_sort_key(path)
        assert isinstance(components, tuple)
        assert isinstance(tiebreak, str)
        for component in components:
            assert isinstance(component, tuple)
            for atom in component:
                kind, number, text = atom
                assert isinstance(kind, int)
                assert isinstance(number, int)
                assert isinstance(text, str)


def test_the_type_stability_guard_can_see_the_defect_it_guards_against() -> None:
    """A naive `int`-or-`str` key dies on `MIXED_BATCH`; ours does not.

    Without this, `test_sorting_a_mixed_batch_does_not_raise` would pass for any
    batch at all, including one that could never have triggered the bug.
    """

    def naive_key(path: str) -> tuple[object, ...]:
        stem = Path(path).stem
        head = stem.split(" ")[-1]
        return (int(head),) if head.isdigit() else (head,)

    with pytest.raises(TypeError):
        sorted(MIXED_BATCH, key=naive_key)

    assert len(_sorted_paths(list(MIXED_BATCH))) == len(MIXED_BATCH)


# --------------------------------------------------------------------------- #
# natural_sort_key - numbers are numbers
# --------------------------------------------------------------------------- #


def test_session_numbers_compare_numerically_not_lexically() -> None:
    given = [
        r"D:\Archive\A\Day 10.md",
        r"D:\Archive\A\Day 2.md",
        r"D:\Archive\A\Day 1.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\Day 1.md",
        r"D:\Archive\A\Day 2.md",
        r"D:\Archive\A\Day 10.md",
    ]


def test_session_directories_compare_numerically_too() -> None:
    """The sequence token may name a directory, not only a file."""
    given = [
        r"D:\Archive\A\Course\Day 10\notes.md",
        r"D:\Archive\A\Course\Day 2\notes.md",
        r"D:\Archive\A\Course\Day 1\notes.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\Course\Day 1\notes.md",
        r"D:\Archive\A\Course\Day 2\notes.md",
        r"D:\Archive\A\Course\Day 10\notes.md",
    ]


def test_part_numbers_order_the_way_the_requirement_names_them() -> None:
    given = [
        r"D:\Archive\A\Part 12 - Closing.md",
        r"D:\Archive\A\Part 2 - Opening.md",
        r"D:\Archive\A\Part 3 - Middle.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\Part 2 - Opening.md",
        r"D:\Archive\A\Part 3 - Middle.md",
        r"D:\Archive\A\Part 12 - Closing.md",
    ]


# --------------------------------------------------------------------------- #
# natural_sort_key - dates
# --------------------------------------------------------------------------- #


def test_compact_timestamp_prefixes_order_chronologically() -> None:
    """`YYYYMMDD_HHMMSS_slug` is ~90% of the surveyed archive (1205 of 1343 files)."""
    given = [
        r"D:\Archive\A\20240311_124244_business-survival-almanac.md",
        r"D:\Archive\A\20240310_182951_the-innovation-flywheel.md",
        r"D:\Archive\A\20240310_082951_an-earlier-post-the-same-day.md",
        r"D:\Archive\A\20180522_200101_it-will-make-you-think-clearly.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\20180522_200101_it-will-make-you-think-clearly.md",
        r"D:\Archive\A\20240310_082951_an-earlier-post-the-same-day.md",
        r"D:\Archive\A\20240310_182951_the-innovation-flywheel.md",
        r"D:\Archive\A\20240311_124244_business-survival-almanac.md",
    ]


def test_iso_dates_order_chronologically() -> None:
    """Design §3.3 names `2026-03-29` explicitly."""
    given = [
        "/srv/archive/A/2026-04-01-notes.md",
        "/srv/archive/A/2025-12-31-notes.md",
        "/srv/archive/A/2026-03-29-notes.md",
    ]
    assert _sorted_paths(given) == [
        "/srv/archive/A/2025-12-31-notes.md",
        "/srv/archive/A/2026-03-29-notes.md",
        "/srv/archive/A/2026-04-01-notes.md",
    ]


def test_iso_dates_beat_a_lexical_day_of_month_comparison() -> None:
    """`2026-03-09` before `2026-03-10` is the same lexically; `9` vs `10` is not."""
    given = [
        "/srv/archive/A/2026-3-10-notes.md",
        "/srv/archive/A/2026-3-9-notes.md",
    ]
    assert _sorted_paths(given) == [
        "/srv/archive/A/2026-3-9-notes.md",
        "/srv/archive/A/2026-3-10-notes.md",
    ]


# --------------------------------------------------------------------------- #
# natural_sort_key - spelled-out session numbers
# --------------------------------------------------------------------------- #


def test_spelled_session_numbers_after_a_keyword_become_numbers() -> None:
    """`ai-moats-part-one` / `-part-two` is a real series in the archive.

    Alphabetically `one < three < two`; the expectation below is the numeric
    order, so a build that dropped the word table would fail rather than agree
    by coincidence.
    """
    given = [
        r"D:\Archive\A\ai-moats-part-three.md",
        r"D:\Archive\A\ai-moats-part-one.md",
        r"D:\Archive\A\ai-moats-part-two.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\ai-moats-part-one.md",
        r"D:\Archive\A\ai-moats-part-two.md",
        r"D:\Archive\A\ai-moats-part-three.md",
    ]


def test_spelled_session_numbers_before_a_keyword_become_numbers() -> None:
    """Design §3.4's own sample document is `Savarni Nath - First Day.md`.

    Alphabetically `ninth < second < tenth`; numerically `second < ninth <
    tenth`.
    """
    given = [
        r"D:\Archive\A\Savarni Nath - Tenth Day.md",
        r"D:\Archive\A\Savarni Nath - Ninth Day.md",
        r"D:\Archive\A\Savarni Nath - Second Day.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\Savarni Nath - Second Day.md",
        r"D:\Archive\A\Savarni Nath - Ninth Day.md",
        r"D:\Archive\A\Savarni Nath - Tenth Day.md",
    ]


def test_a_number_word_without_a_session_keyword_stays_prose() -> None:
    """`the-three-layers-of-ai` is a quantity, not session three.

    Numerically `two` would precede `three`; as prose `three` precedes `two`.
    The expectation is the prose order, which is the only one that keeps the
    archive's quantity-slugs from scattering their folder.
    """
    given = [
        r"D:\Archive\A\the-two-sided-market.md",
        r"D:\Archive\A\the-three-layers-of-ai.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\the-three-layers-of-ai.md",
        r"D:\Archive\A\the-two-sided-market.md",
    ]


def test_a_number_word_inside_a_longer_word_is_not_a_session_number() -> None:
    given = [
        r"D:\Archive\A\partone-anomaly.md",
        r"D:\Archive\A\part-one.md",
    ]
    # `part-one` -> `part 1`; `partone` is untouched, so `part ` < `partone`.
    assert _sorted_paths(given) == [
        r"D:\Archive\A\part-one.md",
        r"D:\Archive\A\partone-anomaly.md",
    ]


def test_longer_number_words_are_not_truncated_to_a_shorter_one() -> None:
    """`sixteen` must not be read as `six`, nor `nineteen` as `nine`.

    `Sixteen` against `Seven` is the pair that separates the two readings: 7
    precedes 16, but a truncating parser produces `6teen` and puts `Sixteen`
    first. `Nine` against `Nineteen` would agree either way and prove nothing,
    so it is here only as company.
    """
    given = [
        r"D:\Archive\A\Volume Twenty.md",
        r"D:\Archive\A\Volume Sixteen.md",
        r"D:\Archive\A\Volume Nineteen.md",
        r"D:\Archive\A\Volume Nine.md",
        r"D:\Archive\A\Volume Seven.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\Volume Seven.md",
        r"D:\Archive\A\Volume Nine.md",
        r"D:\Archive\A\Volume Sixteen.md",
        r"D:\Archive\A\Volume Nineteen.md",
        r"D:\Archive\A\Volume Twenty.md",
    ]


def test_a_session_keyword_that_is_only_a_word_ending_is_not_a_keyword() -> None:
    """`counterpart two` is not part two.

    As prose `three < two`; as sequence numbers `2 < 3`. The expectation is the
    prose order, so a parser that matched `part` inside `counterpart` fails.
    """
    given = [
        r"D:\Archive\A\counterpart two.md",
        r"D:\Archive\A\counterpart three.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\counterpart three.md",
        r"D:\Archive\A\counterpart two.md",
    ]


def test_a_number_word_that_is_only_a_word_ending_is_not_a_session_number() -> None:
    """`Stone Day` is not day one - English is full of words ending in `one`.

    Alphabetically `stoic < stone`. A parser that rewrote `Stone Day` as
    `st1 day` would compare the truncated `st` first and put `Stone` ahead.
    """
    given = [
        r"D:\Archive\A\Stone Day 2.md",
        r"D:\Archive\A\Stoic Day 1.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\Stoic Day 1.md",
        r"D:\Archive\A\Stone Day 2.md",
    ]


def test_a_keyword_that_is_only_a_word_opening_is_not_a_keyword() -> None:
    """`two-parting-ways` is not part two - the mirror of the two tests above.

    This direction has no second guard behind it. Longest-first alternation
    covers the keyword-then-word form (`volume` is tried before `vol`), but
    there is no longer keyword that `parting`, `weekly` or `sessional` could
    match instead, so the trailing `(?![a-z0-9])` on the word-then-keyword
    pattern is the only thing standing here.

    As prose `three- < two-`; read as sequence numbers it reverses to `2 < 3`.
    """
    given = [
        r"D:\Archive\A\two-parting-ways.md",
        r"D:\Archive\A\three-parting-ways.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\three-parting-ways.md",
        r"D:\Archive\A\two-parting-ways.md",
    ]


def test_a_dated_document_leads_an_untitled_sibling() -> None:
    """Which kind of atom wins at the same position, pinned.

    A name that opens with a number carries a position in the series; a prose
    name does not, so the numbered one leads. Given in the opposite order.
    """
    given = [
        r"D:\Archive\A\Intro.md",
        r"D:\Archive\A\2026-03-29 Session.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\2026-03-29 Session.md",
        r"D:\Archive\A\Intro.md",
    ]


# Implementation note 1.1's standing lesson, in the shape `HEAVY_MODULES` uses
# below: a guard list has to be pinned against a **literal**. The
# parameterization underneath this is derived from `SESSION_KEYWORDS` itself, so
# it cannot see a keyword that was deleted (the suite simply runs one case
# fewer) nor one that was misspelled (`chaptre` activates the word table just as
# obediently as `chapter`, and the case count does not even change). Measured:
# dropping any of `vol`, `episode`, `chapter`, `lesson`, `week`, `module` left
# the suite green, and misspelling `chapter` left it green at an unchanged
# count. Only this literal comparison kills either mutant. It looks redundant
# and it is the opposite: it does not prevent the set from growing, it forces
# growth to be deliberate.
EXPECTED_SESSION_KEYWORDS: tuple[str, ...] = (
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

# The 20 cardinals and the 20 ordinals `_NUMBER_WORDS` must spell, in order.
# Written out rather than generated, for the same reason.
CARDINAL_WORDS: tuple[str, ...] = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
)
ORDINAL_WORDS: tuple[str, ...] = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
)


def test_session_keywords_are_pinned_against_a_literal() -> None:
    assert SESSION_KEYWORDS == EXPECTED_SESSION_KEYWORDS


def test_the_whole_number_word_table_is_correct_not_merely_spot_checked() -> None:
    """Every one of the 40 entries, in three assertions.

    Fixtures name only 14 of the words, so a transposed value on any of the
    other 26 is invisible to them - and a transposition is *observable*, not
    equivalent: two adjacent words then tie on the number atom and the order
    collapses to the alphabetical path tiebreak, which would put `Chapter
    Fourteen` ahead of `Chapter Thirteen`. Planted on `thirteen`, `fifth`,
    `eighteen` and `four`, all four survived the fixtures and all four die
    here. Checking the mapping wholesale also grows with the table instead of
    needing a new fixture per word.
    """
    table = order._NUMBER_WORDS
    assert [table[word] for word in CARDINAL_WORDS] == list(range(1, 21))
    assert [table[word] for word in ORDINAL_WORDS] == list(range(1, 21))
    assert set(table) == set(CARDINAL_WORDS) | set(ORDINAL_WORDS)


@pytest.mark.parametrize("keyword", SESSION_KEYWORDS)
def test_every_declared_session_keyword_activates_the_word_table(keyword: str) -> None:
    given = [
        rf"D:\Archive\A\{keyword} three.md",
        rf"D:\Archive\A\{keyword} two.md",
    ]
    assert _sorted_paths(given) == [
        rf"D:\Archive\A\{keyword} two.md",
        rf"D:\Archive\A\{keyword} three.md",
    ]


# --------------------------------------------------------------------------- #
# natural_sort_key - separators, prefixes, and the total order
# --------------------------------------------------------------------------- #


def test_both_separator_spellings_produce_the_same_key() -> None:
    """Implementation note 2.1: `file_path` is backslash-separated here.

    The `/` branch cannot be reached naturally on Windows, so the POSIX value is
    constructed by hand.
    """
    assert natural_sort_key(r"D:\Archive\A\Day 1.md") == natural_sort_key(
        "D:/Archive/A/Day 1.md"
    )


def test_a_windows_long_path_prefix_is_ignored_by_the_key() -> None:
    assert natural_sort_key("\\\\?\\D:\\Archive\\A\\Day 1.md") == natural_sort_key(
        r"D:\Archive\A\Day 1.md"
    )


def test_tokens_do_not_cross_a_directory_boundary() -> None:
    """Components are compared one at a time, so `Set 2` beats `Set 10` outright."""
    given = [
        r"D:\Archive\A\Set 10\Day 1.md",
        r"D:\Archive\A\Set 2\Day 99.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\Set 2\Day 99.md",
        r"D:\Archive\A\Set 10\Day 1.md",
    ]


def test_a_shorter_directory_name_sorts_before_a_longer_one_that_extends_it() -> None:
    """Distinguishes component-wise comparison from a flattened path string.

    Component-wise, `A` precedes `A B` because the shorter component is a
    prefix of the longer. Flattened, the separator byte after `A` (`\\`, 0x5C)
    is compared against the space in `A B` (0x20) and the order reverses.
    """
    given = [
        r"D:\Archive\A B\Day 1.md",
        r"D:\Archive\A\Day 2.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\Day 2.md",
        r"D:\Archive\A B\Day 1.md",
    ]


def test_case_differences_do_not_reorder_documents() -> None:
    given = [
        r"D:\Archive\A\beta.md",
        r"D:\Archive\A\Alpha.md",
    ]
    assert _sorted_paths(given) == [
        r"D:\Archive\A\Alpha.md",
        r"D:\Archive\A\beta.md",
    ]


def test_names_that_fold_together_still_get_a_total_order() -> None:
    """Ruling 4: distinct paths never tie, however they collide after folding."""
    given = [
        r"D:\Archive\A\day 1.md",
        r"D:\Archive\A\Day 01.md",
        r"D:\Archive\A\Day 1.md",
    ]
    first = _sorted_paths(given)
    for permutation in itertools.permutations(given):
        assert _sorted_paths(list(permutation)) == first
    assert len(set(natural_sort_key(path) for path in given)) == len(given)


def test_unparseable_names_are_ordered_deterministically() -> None:
    """UUID-named PDFs are the archive's second-largest naming convention."""
    given = [
        r"D:\Archive\A\files\d77fe7c0-26d0-41e4-b143-5bc149f327a1.pdf",
        r"D:\Archive\A\files\10ba29bd-0459-407a-b0ee-7a7823c2a01e.pdf",
        r"D:\Archive\A\files\5e1aaf12-2413-43ae-97b4-9fdc939c2350.pdf",
    ]
    first = _sorted_paths(given)
    for permutation in itertools.permutations(given):
        assert _sorted_paths(list(permutation)) == first


# --------------------------------------------------------------------------- #
# Ruling 1 - folder identity
# --------------------------------------------------------------------------- #


def test_folder_is_the_first_component_beneath_the_archive_root() -> None:
    assert (
        folder_of(rf"{ARCHIVE}\Author\Course\Day 1\notes.md", archive_root=ARCHIVE)
        == "Author"
    )


def test_a_nested_archive_keeps_its_sessions_inside_one_group() -> None:
    """The case that separates the two candidate definitions of "folder".

    The immediate parent would make `Day 1` and `Day 2` two groups, leaving no
    document sequence for requirement 4.2 to order.
    """
    folders = {
        folder_of(path, archive_root=ARCHIVE)
        for path in (
            rf"{ARCHIVE}\Author\Course\Day 1\notes.md",
            rf"{ARCHIVE}\Author\Course\Day 2\notes.md",
        )
    }
    assert folders == {"Author"}


def test_a_document_directly_in_the_archive_root_groups_under_root_folder() -> None:
    assert folder_of(rf"{ARCHIVE}\readme.md", archive_root=ARCHIVE) == ROOT_FOLDER


def test_without_a_root_the_folder_is_the_full_parent_directory_path() -> None:
    assert (
        folder_of(rf"{ARCHIVE}\Author\Course\Day 1\notes.md")
        == "D:/Archive/Author/Course/Day 1"
    )


def test_without_a_root_two_like_named_leaf_folders_stay_separate() -> None:
    """Over-splitting is visible; a false merge of two `images` folders is not."""
    assert folder_of(rf"{ARCHIVE}\author-a\images\x.md") != folder_of(
        rf"{ARCHIVE}\author-b\images\x.md"
    )


def test_a_path_outside_the_archive_root_falls_back_instead_of_mis_grouping() -> None:
    assert folder_of(r"D:\Elsewhere\Author\x.md", archive_root=ARCHIVE) == (
        "D:/Elsewhere/Author"
    )


def test_a_relative_root_that_cannot_match_falls_back() -> None:
    """`CliConfig.archive_path` defaults to `Path(".")`; task 4.1 must resolve it."""
    assert folder_of(rf"{ARCHIVE}\Author\x.md", archive_root=".") == "D:/Archive/Author"


def test_folder_extraction_accepts_either_separator_spelling() -> None:
    assert folder_of(rf"{ARCHIVE}\Author\x.md", archive_root="D:/Archive") == "Author"
    assert folder_of("/srv/archive/Author/x.md", archive_root="/srv/archive") == "Author"


def test_folder_extraction_sees_through_a_long_path_prefix() -> None:
    assert (
        folder_of("\\\\?\\D:\\Archive\\Author\\x.md", archive_root=ARCHIVE) == "Author"
    )


def test_folder_extraction_sees_through_a_unc_long_path_prefix() -> None:
    assert (
        folder_of(
            "\\\\?\\UNC\\server\\share\\Archive\\Author\\x.md",
            archive_root=r"\\server\share\Archive",
        )
        == "Author"
    )


def test_folder_extraction_accepts_a_path_object_as_the_root() -> None:
    assert folder_of(rf"{ARCHIVE}\Author\x.md", archive_root=Path(ARCHIVE)) == "Author"


@pytest.mark.skipif(os.name != "nt", reason="path casing is significant off Windows")
def test_root_matching_follows_the_platform_casing_rule() -> None:
    assert folder_of(rf"{ARCHIVE}\Author\x.md", archive_root=r"d:\archive") == "Author"


def test_the_folder_filter_and_the_grouping_agree_on_one_group() -> None:
    """Requirement 3.1 + 4.2: `--folder X --group-by-folder` yields exactly `X`.

    `tags._folder_clause` scopes on the `author` column, which
    `discover._author` fills with the first component beneath the root - the
    same value `folder_of` returns.
    """
    scoped = [
        _hit(rf"{ARCHIVE}\Bhagwati Ugra Tara Sadhana\Course\Day 2\a.md"),
        _hit(rf"{ARCHIVE}\Bhagwati Ugra Tara Sadhana\intro.md"),
    ]
    grouped = group_and_order_results(scoped, archive_root=ARCHIVE)
    assert list(grouped) == ["Bhagwati Ugra Tara Sadhana"]


# --------------------------------------------------------------------------- #
# group_and_order_results - narrative flow (requirement 4.2)
# --------------------------------------------------------------------------- #

SADHANA = rf"{ARCHIVE}\Bhagwati Ugra Tara Sadhana"
FIRST_DAY = rf"{SADHANA}\Savarni Nath - First Day.md"
SECOND_DAY = rf"{SADHANA}\Savarni Nath - Second Day.md"
TENTH_DAY = rf"{SADHANA}\Savarni Nath - Tenth Day.md"


def test_ordered_presentation_restores_narrative_flow_across_session_parts() -> None:
    """The task's acceptance sentence, as one assertion.

    Handed a scrambled score-ordered mixture of three sessions, the folder must
    come back as the course reads: session 1 then 2 then 10, and within each
    session the chunks in `chunk_index` order.
    """
    given = [
        _hit(TENTH_DAY, chunk_index=1, score=0.99),
        _hit(FIRST_DAY, chunk_index=13, score=0.61),
        _hit(SECOND_DAY, chunk_index=3, score=0.80),
        _hit(FIRST_DAY, chunk_index=12, score=0.55),
        _hit(SECOND_DAY, chunk_index=1, score=0.72),
    ]
    grouped = group_and_order_results(given, archive_root=ARCHIVE)
    assert _paths(grouped["Bhagwati Ugra Tara Sadhana"]) == [
        (FIRST_DAY, 12),
        (FIRST_DAY, 13),
        (SECOND_DAY, 1),
        (SECOND_DAY, 3),
        (TENTH_DAY, 1),
    ]


def test_document_sequence_outranks_chunk_index() -> None:
    """The two sort components, in the order requirement 4.2 states them.

    Swapping them would put `Tenth Day #1` ahead of `First Day #12`.
    """
    given = [
        _hit(TENTH_DAY, chunk_index=1),
        _hit(FIRST_DAY, chunk_index=12),
    ]
    grouped = group_and_order_results(given, archive_root=ARCHIVE)
    assert _paths(grouped["Bhagwati Ugra Tara Sadhana"]) == [
        (FIRST_DAY, 12),
        (TENTH_DAY, 1),
    ]


def test_chunks_of_one_document_follow_chunk_index_numerically() -> None:
    given = [
        _hit(FIRST_DAY, chunk_index=10),
        _hit(FIRST_DAY, chunk_index=2),
        _hit(FIRST_DAY, chunk_index=7),
        _hit(FIRST_DAY, chunk_index=1),
    ]
    grouped = group_and_order_results(given, archive_root=ARCHIVE)
    assert [r.chunk_index for r in grouped["Bhagwati Ugra Tara Sadhana"]] == [
        1,
        2,
        7,
        10,
    ]


def test_the_result_order_does_not_depend_on_the_input_order() -> None:
    given = [
        _hit(SECOND_DAY, chunk_index=3, score=0.80),
        _hit(FIRST_DAY, chunk_index=12, score=0.55),
        _hit(TENTH_DAY, chunk_index=1, score=0.99),
    ]
    expected = _paths(
        group_and_order_results(given, archive_root=ARCHIVE)[
            "Bhagwati Ugra Tara Sadhana"
        ]
    )
    assert expected == [(FIRST_DAY, 12), (SECOND_DAY, 3), (TENTH_DAY, 1)]
    for permutation in itertools.permutations(given):
        grouped = group_and_order_results(list(permutation), archive_root=ARCHIVE)
        assert _paths(grouped["Bhagwati Ugra Tara Sadhana"]) == expected


# --------------------------------------------------------------------------- #
# Ruling 3 - between-folder order
# --------------------------------------------------------------------------- #

ALPHA = rf"{ARCHIVE}\Alpha Folder\Day 1.md"
BETA = rf"{ARCHIVE}\Beta Folder\Day 1.md"
GAMMA = rf"{ARCHIVE}\Gamma Folder\Day 1.md"


def test_folders_are_returned_best_score_first() -> None:
    """Given in name order; expected in score order, so the two disagree."""
    given = [
        _hit(ALPHA, score=0.10),
        _hit(BETA, score=0.90),
        _hit(GAMMA, score=0.50),
    ]
    grouped = group_and_order_results(given, archive_root=ARCHIVE)
    assert list(grouped) == ["Beta Folder", "Gamma Folder", "Alpha Folder"]


def test_a_folders_rank_uses_its_best_member_not_its_first_or_its_worst() -> None:
    """`Alpha` arrives first and holds the worst hit, yet owns the best one."""
    given = [
        _hit(ALPHA, chunk_index=0, score=0.01),
        _hit(ALPHA, chunk_index=1, score=0.95),
        _hit(BETA, chunk_index=0, score=0.90),
    ]
    grouped = group_and_order_results(given, archive_root=ARCHIVE)
    assert list(grouped) == ["Alpha Folder", "Beta Folder"]


def test_folders_with_equal_best_scores_fall_back_to_name_order() -> None:
    given = [
        _hit(GAMMA, score=0.42),
        _hit(BETA, score=0.42),
        _hit(ALPHA, score=0.42),
    ]
    grouped = group_and_order_results(given, archive_root=ARCHIVE)
    assert list(grouped) == ["Alpha Folder", "Beta Folder", "Gamma Folder"]


def test_between_folder_order_does_not_depend_on_the_input_order() -> None:
    given = [
        _hit(ALPHA, score=0.10),
        _hit(BETA, score=0.90),
        _hit(GAMMA, score=0.50),
    ]
    for permutation in itertools.permutations(given):
        grouped = group_and_order_results(list(permutation), archive_root=ARCHIVE)
        assert list(grouped) == ["Beta Folder", "Gamma Folder", "Alpha Folder"]


# --------------------------------------------------------------------------- #
# group_and_order_results - conservation and edges
# --------------------------------------------------------------------------- #


def test_every_input_result_appears_exactly_once() -> None:
    given = [
        _hit(ALPHA, chunk_index=1, score=0.3),
        _hit(BETA, chunk_index=2, score=0.4),
        _hit(FIRST_DAY, chunk_index=3, score=0.5),
        _hit(SECOND_DAY, chunk_index=4, score=0.6),
    ]
    grouped = group_and_order_results(given, archive_root=ARCHIVE)
    flattened = [hit for items in grouped.values() for hit in items]
    assert sorted(hit.chunk_id for hit in flattened) == sorted(
        hit.chunk_id for hit in given
    )


def test_an_empty_result_set_yields_an_empty_mapping() -> None:
    assert group_and_order_results([]) == {}


def test_the_caller_sequence_is_not_mutated() -> None:
    given = [
        _hit(TENTH_DAY, chunk_index=1, score=0.99),
        _hit(FIRST_DAY, chunk_index=12, score=0.55),
    ]
    snapshot = list(given)
    group_and_order_results(given, archive_root=ARCHIVE)
    assert given == snapshot


def test_duplicate_results_for_one_chunk_keep_their_arrival_order() -> None:
    """The one remaining tie; `list.sort` is stable, so arrival order survives."""
    first = _hit(FIRST_DAY, chunk_index=5, score=0.9)
    second = _hit(FIRST_DAY, chunk_index=5, score=0.1)
    grouped = group_and_order_results([first, second], archive_root=ARCHIVE)
    assert grouped["Bhagwati Ugra Tara Sadhana"] == [first, second]


def test_grouping_without_a_root_still_orders_within_each_group() -> None:
    """Ruling 1's fallback must still satisfy requirement 4.2 inside a group."""
    given = [
        _hit(rf"{SADHANA}\Savarni Nath - Tenth Day.md", chunk_index=1),
        _hit(rf"{SADHANA}\Savarni Nath - First Day.md", chunk_index=12),
        _hit(rf"{SADHANA}\Savarni Nath - Second Day.md", chunk_index=3),
    ]
    grouped = group_and_order_results(given)
    (items,) = grouped.values()
    assert _paths(items) == [(FIRST_DAY, 12), (SECOND_DAY, 3), (TENTH_DAY, 1)]


# --------------------------------------------------------------------------- #
# Requirement 1.2 - nothing heavy on the search path
# --------------------------------------------------------------------------- #

HEAVY_MODULES: tuple[str, ...] = ("lancedb", "onnxruntime", "torch", "transformers")

_PROBE = '''\
import json
import sys
from pathlib import Path

import npu_rag.cli.order  # noqa: F401

HEAVY = {heavy!r}
Path({result!r}).write_text(
    json.dumps(sorted(name for name in HEAVY if name in sys.modules)),
    encoding="utf-8",
)
'''


def test_importing_order_imports_no_heavy_module(tmp_path: Path) -> None:
    """`npu_rag.index.types` drags in lancedb, torch and transformers.

    Probed 2026-09-21: importing it alone loads 3266 modules. `SearchResult` is
    therefore referenced only under `TYPE_CHECKING`; this test is what stops a
    later edit from turning that into a runtime import.
    """
    result_path = tmp_path / "order_probe.json"
    script_path = tmp_path / "order_probe.py"
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
    assert payload == [], f"npu_rag.cli.order pulled in heavy modules: {payload}"


def test_order_module_has_no_runtime_import_of_the_index_package() -> None:
    """The structural half: the import must sit inside `if TYPE_CHECKING:`."""
    source = Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "cli" / "order.py"
    text = source.read_text(encoding="utf-8")
    body, _, guarded = text.partition("if TYPE_CHECKING:")
    assert guarded, "order.py must guard its SearchResult import"
    assert not re.search(r"^from npu_rag\.index", body, flags=re.MULTILINE)
    assert "from npu_rag.index.types import SearchResult" in guarded
