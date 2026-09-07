"""The throwaway generator behind the committed benchmark fixture (task 6.1).

``tools/build_benchmark_corpus.py`` is run once and its output is committed, so
it would be easy to argue it needs no tests. It does, for two reasons.

First, the extraction rules decide what third-party text ends up in a public
repository, and task 6.1's constraint is specific: markdown scaffolding is
stripped, but "do not paraphrase or alter the words themselves". That is a
property of these functions and of nothing else.

Second, and generalising Implementation Note 5.3: a function whose only claim to
correctness is that it reads correctly is untested. The loader
(``npu_rag.embedding.bench.corpus``) re-checks the *policy* on every load, so a
generator bug that broke the cap could not reach the benchmark - but a generator
bug that mangled words, or that sampled one corner of the archive, would sail
straight through, because the loader has nothing to compare against.

These tests import both the generator and the package loader, which the package
itself may not do. This module is allowed to: it is not the package, the same
licence ``tests/tools/test_provision_npu.py`` already relies on for the same
kind of forced duplication (Implementation Note 1.5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npu_rag.embedding.bench import corpus as loader
from tools import build_benchmark_corpus as generator


# --------------------------------------------------------------------------
# The policy constants are duplicated by force; pin them to each other
# --------------------------------------------------------------------------


def test_the_generator_and_the_loader_agree_on_the_excerpt_policy() -> None:
    """The package must not import from ``tools/`` (design.md, Out of Boundary),
    so the caps exist twice. Implementation Note 1.5 records what happens when
    such a pair is kept in step by convention alone: the capability check and
    the provisioner silently diverged 3-vs-6 for the whole life of the project.
    The relation is a test, not a comment."""
    assert generator.MAX_CHUNK_CHARACTERS == loader.MAX_CHUNK_CHARACTERS
    assert generator.MAX_CHUNKS_PER_ARTICLE == loader.MAX_CHUNKS_PER_ARTICLE


def test_the_generator_never_emits_more_chunks_than_the_loader_accepts() -> None:
    assert generator.MINIMUM_CHUNK_CHARACTERS < generator.MAX_CHUNK_CHARACTERS


# --------------------------------------------------------------------------
# Markdown to prose: strip scaffolding, never touch the words
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("# A heading", "A heading"),
        ("#### Introduction", "Introduction"),
        ("> quoted line", "quoted line"),
        ("- a bullet", "a bullet"),
        ("3. a numbered item", "a numbered item"),
        ("**bold** and _italic_ and `code`", "bold and italic and code"),
        ("a [linked phrase](https://example.com/x) inside", "a linked phrase inside"),
        ("[![alt](img/a.png)](img/a.png)", ""),
        ("![alt](img/a.png)", ""),
        ("see https://example.com/deep/path now", "see now"),
        ("<em>tagged</em> text", "tagged text"),
        (r"he said \"stop\"", 'he said "stop"'),
        ("snake_case_name survives", "snake_case_name survives"),
    ],
)
def test_scaffolding_is_removed_and_the_words_are_not(
    markdown: str, expected: str
) -> None:
    assert generator.strip_markup(markdown) == expected


def test_link_text_survives_because_it_is_the_author_s_prose() -> None:
    """The distinction the whole cleaner turns on: the *target* of a link is
    markup and goes; the words a reader sees are the author's and stay."""
    cleaned = generator.strip_markup(
        "Back then, [the first modern monopolies](https://x.example/a) were born."
    )

    assert cleaned == "Back then, the first modern monopolies were born."


def test_paragraphs_are_split_on_blank_lines_and_rules() -> None:
    document = "# Title\n\nFirst para\nsecond line.\n\n* * *\n\nSecond para.\n"

    assert list(generator.paragraphs(document)) == [
        "Title",
        "First para second line.",
        "Second para.",
    ]


def test_zero_width_characters_are_folded_away() -> None:
    """This archive's exports carry U+200B inside link text. A chunk that kept
    them would embed differently from the same words typed by hand."""
    assert generator.normalise("a​b﻿c\xa0d") == "abc d"


def test_boilerplate_paragraphs_are_recognised() -> None:
    assert generator.is_boilerplate(
        "The Business Engineer is a reader-supported publication."
    )
    assert not generator.is_boilerplate(
        "Bell Labs followed a hybrid model of innovation."
    )


def test_fragments_that_are_not_prose_are_rejected() -> None:
    assert not generator.is_prose("Fig. 1")
    assert not generator.is_prose("a" * 400)  # one long token, no sentences
    assert not generator.is_prose("lowercase start " * 30)
    assert generator.is_prose(
        "The interesting point about this laboratory is that it followed a "
        "hybrid model of innovation, which ran through loops that went along "
        "roughly these lines and repeated over many years of work."
    )


# --------------------------------------------------------------------------
# The excerpt cap
# --------------------------------------------------------------------------


def test_a_short_paragraph_is_returned_untouched() -> None:
    short = "A paragraph well under the cap."

    assert generator.cap(short) == short


def test_a_long_paragraph_is_cut_at_a_sentence_boundary_within_the_cap() -> None:
    sentence = "This is a sentence of a useful length that repeats. "
    paragraph = sentence * 20

    capped = generator.cap(paragraph)

    assert len(capped) <= generator.MAX_CHUNK_CHARACTERS
    assert capped.endswith("repeats.")
    assert paragraph.startswith(capped)


def test_a_long_paragraph_without_a_sentence_end_is_cut_at_a_word_boundary() -> None:
    paragraph = " ".join(["word"] * 300)

    capped = generator.cap(paragraph)

    assert len(capped) <= generator.MAX_CHUNK_CHARACTERS
    assert capped.endswith("word")
    assert "  " not in capped


# --------------------------------------------------------------------------
# What "representative" means: spread, allocation, and no cross-article repeats
# --------------------------------------------------------------------------


def test_allocation_is_proportional_but_respects_the_floor() -> None:
    allocation = generator.allocate(
        {"big": 849, "small": 187, "smaller": 169}, total=120, floor=24
    )

    assert sum(allocation.values()) == 120
    assert min(allocation.values()) >= 24
    assert allocation["big"] == max(allocation.values())


def test_allocation_never_asks_for_more_articles_than_exist() -> None:
    allocation = generator.allocate({"big": 100, "tiny": 5}, total=120, floor=24)

    assert allocation["tiny"] <= 5
    assert allocation["big"] <= 100


def test_selection_spreads_over_the_whole_ordered_list() -> None:
    """The property that makes commitment 3 real. Taking a contiguous run would
    satisfy every count in the fixture while sampling one moment of one
    publication, so the stride is checked directly: the picks must reach both
    ends of the list, not cluster at its start."""
    items = list(range(100))

    picked = generator.ordered_selection(items, 10)

    assert len(picked) == 10
    assert picked == sorted(picked)
    assert min(picked) < 10 and max(picked) > 89
    assert generator.ordered_selection(items, 10) == picked  # deterministic


def test_selection_returns_everything_when_more_is_wanted_than_exists() -> None:
    assert generator.ordered_selection([1, 2, 3], 10) == [1, 2, 3]


def test_positions_take_from_spread_places_inside_an_article() -> None:
    """Commitment 4: not only introductions. With ten candidate paragraphs the
    two picks must straddle the middle."""
    picked = generator.positions(10, 2)

    assert len(picked) == 2
    assert picked[0] < 5 <= picked[1]


def test_positions_cannot_exceed_the_per_article_ceiling() -> None:
    assert len(generator.positions(50, generator.MAX_CHUNKS_PER_ARTICLE)) == (
        generator.MAX_CHUNKS_PER_ARTICLE
    )
    assert generator.positions(1, 2) == [0]


def test_a_paragraph_repeated_across_articles_is_dropped_from_both(
    tmp_path: Path,
) -> None:
    """Measured on the real archive: this publication repeats a "get some help
    via ..." paragraph verbatim across many posts. Two chunks carrying identical
    text would be one document counted twice in the retrieval pool, and the
    per-article ceiling cannot see it because the copies live in different
    articles."""
    shared = "Shared furniture " * 5
    unique = "Only here "
    articles = {
        "pub": [
            generator.Article(
                publication="pub",
                path=tmp_path / "a.md",
                article_id="20200101-a",
                title="A",
                candidates=(shared, unique + "one"),
            ),
            generator.Article(
                publication="pub",
                path=tmp_path / "b.md",
                article_id="20200102-b",
                title="B",
                candidates=(shared, unique + "two"),
            ),
        ]
    }

    kept = generator.drop_repeated_across_articles(articles)

    assert [article.candidates for article in kept["pub"]] == [
        (unique + "one",),
        (unique + "two",),
    ]


def test_the_duplicate_key_ignores_punctuation_and_case() -> None:
    assert generator.dedup_key("Get help, now!") == generator.dedup_key("get help now")
    assert generator.dedup_key("one") != generator.dedup_key("two")


# --------------------------------------------------------------------------
# End to end over a miniature archive
# --------------------------------------------------------------------------


def _write_article(directory: Path, name: str, title: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def test_the_generator_produces_a_fixture_the_loader_accepts(tmp_path: Path) -> None:
    """The end-to-end property Implementation Note 5.4 asks for in the other
    direction: the two halves must be assembled together at least once, or each
    is only proven against its own idea of the other."""
    archive = tmp_path / "archive"
    paragraph = (
        "This paragraph is long enough to count as prose because it carries "
        "well over twenty spaces and reads as ordinary sentences rather than "
        "as a caption or a fragment of code, which is the whole test. It also "
        "clears the generator's minimum length, so the extractor keeps it "
        "instead of discarding it as a stray fragment of a caption. "
    )
    for publication in ("alpha", "beta"):
        for day in range(1, 6):
            _write_article(
                archive / publication,
                f"2024010{day}_120000_post-{publication}-{day}.md",
                f"Post {publication} {day}",
                f"{paragraph} Number {publication}{day} one.\n\n"
                f"{paragraph} Number {publication}{day} two.\n\n"
                f"{paragraph} Number {publication}{day} three.\n",
            )

    out = tmp_path / "fixture"
    out.mkdir()
    (out / "relevance.jsonl").write_text(
        '{"query_id": "q1", "text": "a question", "relevant_chunk_ids": '
        '["alpha/20240101-post-alpha-1#0"]}\n',
        encoding="utf-8",
    )
    assert generator.main(["--archive", str(archive), "--out", str(out)]) == 0

    loaded = loader.load_corpus(out)

    assert {chunk.publication for chunk in loaded.chunks} == {"alpha", "beta"}
    assert loaded.composition.article_count == 10
    assert loaded.composition.observed_max_chunks_per_article <= (
        loader.MAX_CHUNKS_PER_ARTICLE
    )
    assert [query.query_id for query in loaded.queries] == ["q1"]


def test_the_generator_is_deterministic(tmp_path: Path) -> None:
    """Regenerating from the same archive must reproduce the same fixture, so a
    diff in the committed file means the archive changed and nothing else."""
    archive = tmp_path / "archive"
    paragraph = (
        "A paragraph with enough words in it to be treated as prose by the "
        "extractor, carrying more than twenty spaces and ending properly. It "
        "runs past the minimum length the generator insists on, so that this "
        "test exercises selection rather than the length filter. "
    )
    for day in range(1, 9):
        _write_article(
            archive / "solo",
            f"2024010{day}_090000_post-{day}.md",
            f"Post {day}",
            f"{paragraph} Body {day} first.\n\n{paragraph} Body {day} second.\n",
        )

    first = generator.build(archive)
    second = generator.build(archive)

    assert first == second
    assert len(first) > 0
