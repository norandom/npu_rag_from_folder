"""The committed benchmark fixture and its loader (task 6.1, requirement 6.5).

Requirement 6.5 asks for "a representative sample of the target corpus" whose
"size and composition" the benchmark states, so that benchmarking "does not
depend on a completed corpus index". design.md's Out of Boundary section spells
out the mechanism: the benchmark "does not read ``.md`` or ``.pdf`` files. It
consumes a small pre-extracted fixture committed at
``tests/fixtures/benchmark-corpus/``".

These tests are written against the *committed* files, not against a synthetic
in-memory stand-in, because Implementation Note 5.4 records that a fixture can
be non-vacuous against a system that was never assembled that way. The real
loader is pointed at the real bytes that ship in the repository.

The standing lesson in tasks.md is that vacuous fixtures are this project's
recurring defect, and this task *is* a fixture. So every property below is
paired with the wrong implementation it would catch:

===========================================  ==================================
Property                                     Wrong data it rejects
===========================================  ==================================
all three publications present               a sample drawn from whichever
                                             directory sorts first
per-publication article floor                a sample where a publication is
                                             present by one token article
excerpt cap enforced on load                 a chunk that reproduces a whole
                                             article
per-article chunk ceiling enforced           an article reproduced across many
                                             chunks
relevance ids resolve                        a query set pointing at chunks
                                             that were never committed
composition record cross-checked             a composition record that drifted
                                             away from the data it describes
lexical baseline scores poorly               a query set satisfiable by keyword
                                             overlap, which would prove nothing
                                             about embeddings in task 6.6
===========================================  ==================================
"""

from __future__ import annotations

import ast
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path

import pytest

from npu_rag.embedding.bench.corpus import (
    MAX_CHUNKS_PER_ARTICLE,
    MAX_CHUNK_CHARACTERS,
    BenchmarkCorpus,
    CorpusChunk,
    CorpusFixtureError,
    RelevanceQuery,
    SampleComposition,
    default_corpus_directory,
    load_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "benchmark-corpus"

#: The three publications in the source archive. Requirement 6.5's
#: "representative" is meaningless if the sample silently collapses onto one of
#: them, so the set is named here rather than derived from the data it checks.
EXPECTED_PUBLICATIONS = frozenset(
    {"businessengineer.ai", "nurijanian", "quantbeckman.com"}
)

#: Floors, not descriptions. They are deliberately far below the fixture's
#: actual figures so that ordinary regeneration does not churn the tests, while
#: a sample that collapsed to a token size would still fail.
MINIMUM_CHUNKS = 100
MINIMUM_ARTICLES = 50
MINIMUM_ARTICLES_PER_PUBLICATION = 10
MINIMUM_QUERIES = 12


@pytest.fixture(scope="module")
def corpus() -> BenchmarkCorpus:
    return load_corpus(FIXTURE_DIR)


# --------------------------------------------------------------------------
# The fixture exists and is reachable without any source document
# --------------------------------------------------------------------------


def test_the_fixture_directory_ships_the_three_committed_files() -> None:
    """design.md's File Structure Plan lists ``chunks.jsonl`` and
    ``relevance.jsonl``; ``composition.json`` is this task's addition, because
    requirement 6.5 asks for the size and composition to be *stated* and task
    6.7 must render that statement into the benchmark document. Prose in a
    README is not loadable."""
    assert sorted(p.name for p in FIXTURE_DIR.iterdir()) == [
        "README.md",
        "chunks.jsonl",
        "composition.json",
        "relevance.jsonl",
    ]


def test_the_corpus_loads_from_a_copy_of_the_directory_alone(
    tmp_path: Path,
) -> None:
    """Task 6.1's Observable: "the fixture loads without touching any source
    document". Copying only the fixture directory to a scratch location and
    loading from there proves it structurally - anything the loader still
    needed from the archive, or from the repository, would be absent."""
    destination = tmp_path / "elsewhere" / "benchmark-corpus"
    shutil.copytree(FIXTURE_DIR, destination)

    loaded = load_corpus(destination)

    assert len(loaded.chunks) >= MINIMUM_CHUNKS
    assert len(loaded.queries) >= MINIMUM_QUERIES


def test_the_loader_module_names_no_archive_path_and_imports_no_tooling() -> None:
    """The generator is throwaway tooling under ``tools/``; the loader is the
    part the benchmark keeps. If the loader could reach either the archive or
    the generator, the Observable above would hold only by luck."""
    source = (
        REPO_ROOT / "src" / "npu_rag" / "embedding" / "bench" / "corpus.py"
    ).read_text(encoding="utf-8")

    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert [name for name in imported if name.split(".")[0] == "tools"] == []

    # Prose may discuss the archive; no *value* may point at it. Every string
    # the module can act on is checked, docstrings excluded - a path literal is
    # the only way this module could reach outside the directory it is handed.
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        )
    }
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    ]
    assert [
        text
        for text in literals
        # A rooted path needs something after the root; a bare "/" is the chunk
        # id's own separator, not a filesystem reference.
        if re.match(r"^([A-Za-z]:[\\/]|[\\/][^\\/ ]|\.\.[\\/])", text)
        or "substack" in text.lower()
    ] == []


def test_the_default_directory_points_at_the_committed_fixture() -> None:
    assert default_corpus_directory() == FIXTURE_DIR


def test_loading_a_directory_without_the_fixture_is_an_explicit_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(CorpusFixtureError, match="chunks.jsonl"):
        load_corpus(tmp_path)


# --------------------------------------------------------------------------
# Size and composition (requirement 6.5)
# --------------------------------------------------------------------------


def test_the_sample_spans_every_publication_in_the_archive(
    corpus: BenchmarkCorpus,
) -> None:
    """"Representative" is given one concrete meaning here: every publication
    in the source archive contributes, and none contributes by a token article.
    A sample drawn from whichever directory sorts first passes no part of it."""
    articles_per_publication = Counter(
        (chunk.publication, chunk.article_id) for chunk in corpus.chunks
    )
    counted: Counter[str] = Counter()
    for publication, _article in articles_per_publication:
        counted[publication] += 1

    assert set(counted) == EXPECTED_PUBLICATIONS
    assert min(counted.values()) >= MINIMUM_ARTICLES_PER_PUBLICATION


def test_the_sample_is_large_enough_to_rank_against(
    corpus: BenchmarkCorpus,
) -> None:
    assert len(corpus.chunks) >= MINIMUM_CHUNKS
    assert len({chunk.article_id for chunk in corpus.chunks}) >= MINIMUM_ARTICLES


def test_the_sample_spreads_across_each_publication_s_date_range(
    corpus: BenchmarkCorpus,
) -> None:
    """Articles are identified by their archive filename, which begins with a
    ``YYYYMMDD`` stamp. Drawing thirty consecutive posts from one month would
    satisfy every count above while sampling one moment of one publication, so
    the spread is checked directly: each publication's selected articles must
    cover at least half the calendar years present *in the sample* for that
    publication.

    Note what this does NOT say. It cannot compare against the years the
    publication spans in the archive, because reading the archive is precisely
    what this task's Observable forbids - so a publication the sample happens to
    draw from a single year passes trivially. The archive-side property is
    covered on the generator's side instead, by
    ``test_selection_spreads_over_the_whole_ordered_list``."""
    years_by_publication: dict[str, set[str]] = {}
    for chunk in corpus.chunks:
        years_by_publication.setdefault(chunk.publication, set()).add(
            chunk.article_id[:4]
        )

    for publication, years in years_by_publication.items():
        span = max(years), min(years)
        expected = int(span[0]) - int(span[1]) + 1
        assert len(years) >= math.ceil(expected / 2), (
            f"{publication} draws from {sorted(years)} across a "
            f"{expected}-year span"
        )


def test_the_composition_record_agrees_with_the_data_it_describes(
    corpus: BenchmarkCorpus,
) -> None:
    """The record is committed rather than derived, so that it can state the
    *policy* (the caps, the archive it came from, the generator that made it)
    and not only the counts. A committed record can drift from its data; the
    loader cross-checks it, so this asserts the cross-check has teeth by
    recomputing every field independently of the loader."""
    composition = corpus.composition
    chunks = corpus.chunks

    assert composition.chunk_count == len(chunks)
    assert composition.article_count == len(
        {(c.publication, c.article_id) for c in chunks}
    )
    assert composition.chunks_per_publication == dict(
        Counter(c.publication for c in chunks)
    )
    assert composition.articles_per_publication == {
        publication: len(
            {c.article_id for c in chunks if c.publication == publication}
        )
        for publication in sorted({c.publication for c in chunks})
    }
    assert composition.observed_max_chunk_characters == max(
        len(c.text) for c in chunks
    )
    assert composition.observed_min_chunk_characters == min(
        len(c.text) for c in chunks
    )
    assert composition.observed_max_chunks_per_article == max(
        Counter((c.publication, c.article_id) for c in chunks).values()
    )
    assert composition.query_count == len(corpus.queries)
    assert composition.max_chunk_characters == MAX_CHUNK_CHARACTERS
    assert composition.max_chunks_per_article == MAX_CHUNKS_PER_ARTICLE


def test_a_composition_record_that_drifts_from_the_data_is_rejected(
    tmp_path: Path,
) -> None:
    """The mutant: a record claiming one more chunk than ships. Without the
    cross-check the benchmark would state a size the sample does not have,
    which is precisely the claim requirement 6.5 makes."""
    destination = tmp_path / "benchmark-corpus"
    shutil.copytree(FIXTURE_DIR, destination)
    record = json.loads((destination / "composition.json").read_text("utf-8"))
    record["chunk_count"] += 1
    (destination / "composition.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    with pytest.raises(CorpusFixtureError, match="chunk_count"):
        load_corpus(destination)


# --------------------------------------------------------------------------
# The excerpt cap and the per-article ceiling (task 6.1, 2026-09-07)
# --------------------------------------------------------------------------


def test_every_committed_chunk_is_within_the_excerpt_cap(
    corpus: BenchmarkCorpus,
) -> None:
    """The cap exists because this repository pushes to public GitHub and the
    archive is third-party writing. It costs the benchmark nothing: the NPU
    graph is compiled at a static ``(batch_size, 512)``, so every input is
    padded to the compiled length and consumes identical compute regardless of
    its real text length."""
    over_cap = [c.chunk_id for c in corpus.chunks if len(c.text) > MAX_CHUNK_CHARACTERS]

    assert over_cap == []


def test_a_chunk_over_the_cap_is_refused_at_load_time(tmp_path: Path) -> None:
    """Mutant: one chunk lengthened past the cap. The cap must be enforced by
    the loader and not merely by the generator's good behaviour, because the
    generator is throwaway and the data outlives it."""
    destination = _fixture_copy_with_mutated_chunks(
        tmp_path,
        lambda records: _replace_first(
            records, "text", "x" * (MAX_CHUNK_CHARACTERS + 1)
        ),
    )

    with pytest.raises(CorpusFixtureError, match="exceeds"):
        load_corpus(destination)


def test_no_article_contributes_more_than_the_permitted_chunks(
    corpus: BenchmarkCorpus,
) -> None:
    per_article = Counter((c.publication, c.article_id) for c in corpus.chunks)

    assert max(per_article.values()) <= MAX_CHUNKS_PER_ARTICLE


def test_an_article_contributing_too_many_chunks_is_refused(
    tmp_path: Path,
) -> None:
    """Mutant: one article reproduced across ``MAX_CHUNKS_PER_ARTICLE + 1``
    chunks. Caps on individual chunks do not bound how much of one article the
    fixture carries; only this does."""

    def overload(records: list[dict[str, object]]) -> list[dict[str, object]]:
        seed = records[0]
        extra = [
            {**seed, "chunk_id": f"{seed['chunk_id']}-extra-{n}"}
            for n in range(MAX_CHUNKS_PER_ARTICLE)
        ]
        return records + extra

    destination = _fixture_copy_with_mutated_chunks(tmp_path, overload)

    with pytest.raises(CorpusFixtureError, match="chunks from one article"):
        load_corpus(destination)


def test_the_committed_bytes_stay_small_enough_to_be_excerpts(
    corpus: BenchmarkCorpus,
) -> None:
    """A second, blunter guard on the same concern: even at the cap, the total
    quantity of someone else's prose in the repository stays a sample. The
    ceiling is generous against the current fixture and would still catch an
    order-of-magnitude expansion."""
    total_text = sum(len(c.text) for c in corpus.chunks)

    assert total_text <= 200_000


# --------------------------------------------------------------------------
# Identity and provenance
# --------------------------------------------------------------------------


def test_chunk_ids_are_unique(corpus: BenchmarkCorpus) -> None:
    ids = [c.chunk_id for c in corpus.chunks]

    assert len(set(ids)) == len(ids)


def test_a_duplicated_chunk_id_is_refused(tmp_path: Path) -> None:
    """Mutant: the same id twice. ``relevance.jsonl`` addresses chunks by id
    and tasks 6.6 and 6.7 report against them, so an ambiguous id would make a
    relevance judgement refer to two different texts."""
    destination = _fixture_copy_with_mutated_chunks(
        tmp_path, lambda records: records + [dict(records[0])]
    )

    with pytest.raises(CorpusFixtureError, match="duplicate"):
        load_corpus(destination)


def test_every_chunk_records_the_publication_and_article_it_came_from(
    corpus: BenchmarkCorpus,
) -> None:
    """Task 6.1: "Record each chunk's source publication and article so the
    composition requirement 6.5 asks for is real"."""
    for chunk in corpus.chunks:
        assert chunk.publication in EXPECTED_PUBLICATIONS
        assert chunk.article_id
        assert chunk.article_title
        assert chunk.chunk_id.startswith(f"{chunk.publication}/{chunk.article_id}#")


def test_a_chunk_id_that_disagrees_with_its_own_provenance_is_refused(
    tmp_path: Path,
) -> None:
    """The id is derived from the publication and article, so the two could
    drift apart silently. They are cross-checked instead."""
    destination = _fixture_copy_with_mutated_chunks(
        tmp_path,
        lambda records: _replace_first(records, "publication", "somewhere-else"),
    )

    with pytest.raises(CorpusFixtureError, match="does not match"):
        load_corpus(destination)


def test_chunk_text_is_prose_rather_than_markup(corpus: BenchmarkCorpus) -> None:
    """Task 6.1 asks that markdown scaffolding be stripped so the text is
    prose. Markup left in place would be embedded as content and would show up
    in retrieval quality as noise attributable to nothing."""
    for chunk in corpus.chunks:
        assert "](" not in chunk.text, chunk.chunk_id
        assert "http://" not in chunk.text and "https://" not in chunk.text
        assert not chunk.text.startswith("#")
        assert "![" not in chunk.text
        assert chunk.text == chunk.text.strip()


# --------------------------------------------------------------------------
# The query set (requirement 6.4's input, built here under 6.5)
# --------------------------------------------------------------------------


def test_every_relevance_judgement_names_a_committed_chunk(
    corpus: BenchmarkCorpus,
) -> None:
    known = {c.chunk_id for c in corpus.chunks}
    dangling = sorted(
        chunk_id
        for query in corpus.queries
        for chunk_id in query.relevant_chunk_ids
        if chunk_id not in known
    )

    assert dangling == []


def test_a_relevance_judgement_naming_an_absent_chunk_is_refused(
    tmp_path: Path,
) -> None:
    """Mutant: one relevance id pointed at a chunk that does not ship. Task
    6.6 would score it as an unretrievable relevant document and read a
    permanently depressed score as a property of the *model*."""
    destination = tmp_path / "benchmark-corpus"
    shutil.copytree(FIXTURE_DIR, destination)
    path = destination / "relevance.jsonl"
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    records[0]["relevant_chunk_ids"] = ["nowhere/at-all#0"]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    with pytest.raises(CorpusFixtureError, match="nowhere/at-all#0"):
        load_corpus(destination)


def test_the_query_set_is_large_enough_and_every_query_has_judgements(
    corpus: BenchmarkCorpus,
) -> None:
    assert len(corpus.queries) >= MINIMUM_QUERIES
    assert len({q.query_id for q in corpus.queries}) == len(corpus.queries)
    for query in corpus.queries:
        assert query.text.strip() == query.text and query.text
        assert len(query.relevant_chunk_ids) >= 2, query.query_id
        assert len(set(query.relevant_chunk_ids)) == len(query.relevant_chunk_ids)


def test_a_query_with_no_relevant_chunks_is_refused(tmp_path: Path) -> None:
    destination = tmp_path / "benchmark-corpus"
    shutil.copytree(FIXTURE_DIR, destination)
    path = destination / "relevance.jsonl"
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    records[0]["relevant_chunk_ids"] = []
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    with pytest.raises(CorpusFixtureError, match="no relevant"):
        load_corpus(destination)


def test_relevance_judgements_reach_across_publications(
    corpus: BenchmarkCorpus,
) -> None:
    """A query set whose judgements all sat inside one publication would let a
    model score well by learning the publication's register rather than the
    query's meaning."""
    publication_of = {c.chunk_id: c.publication for c in corpus.chunks}
    reached = {
        publication_of[chunk_id]
        for query in corpus.queries
        for chunk_id in query.relevant_chunk_ids
    }

    assert reached == EXPECTED_PUBLICATIONS


# --------------------------------------------------------------------------
# The query set is not satisfiable by lexical overlap
#
# Task 6.6's Observable is that "a model whose dense stage is omitted scores
# visibly worse". That only discriminates if the query set cannot be answered
# without understanding the text. A query sharing obvious keywords with its
# relevant chunks would be answered by any bag of words, and the measurement
# would report nothing about embeddings at all.
#
# The check below is deliberately three-part, because a low score from a broken
# scorer proves nothing: the same scorer must be shown to score *well* on a
# query set that genuinely does overlap lexically, and an oracle ranking must
# score perfectly, before the low score on the real query set is evidence.
# --------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """
    a about after all also an and any are as at be because been before being but by
    can could did do does doing down during each few for from further had has have
    having he her here hers him his how i if in into is it its itself just me more
    most my no nor not of off on once only or other our out over own same she should
    so some such than that the their them then there these they this those through
    to too under until up very was we were what when where which while who whom why
    will with would you your
    """.split()
)


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _lexical_ranking(
    query: str, chunks: Sequence[CorpusChunk], document_frequency: Counter[str]
) -> list[str]:
    """An IDF-weighted bag-of-words ranking - the "obvious literal keyword"
    baseline the query set must defeat."""
    query_terms = _terms(query)
    total = len(chunks)
    scored: list[tuple[float, str]] = []
    for chunk in chunks:
        overlap = query_terms & _terms(f"{chunk.article_title} {chunk.text}")
        score = sum(
            math.log(1 + total / (1 + document_frequency[term])) for term in overlap
        )
        scored.append((score, chunk.chunk_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [chunk_id for _score, chunk_id in scored]


def _recall_at_k(ranking: Iterable[str], relevant: Iterable[str], k: int) -> float:
    top = list(ranking)[:k]
    relevant_ids = set(relevant)
    return len(relevant_ids & set(top)) / len(relevant_ids)


def _document_frequency(chunks: Sequence[CorpusChunk]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for chunk in chunks:
        frequency.update(_terms(f"{chunk.article_title} {chunk.text}"))
    return frequency


def test_the_lexical_baseline_finds_overlap_when_overlap_exists(
    corpus: BenchmarkCorpus,
) -> None:
    """Control 1 for the check below. Queries built by copying the opening of
    a relevant chunk *are* satisfiable by keyword overlap; the baseline must
    score near-perfectly on them, or its low score on the real query set would
    only mean the scorer is broken."""
    frequency = _document_frequency(corpus.chunks)
    scores = [
        _recall_at_k(
            _lexical_ranking(chunk.text[:200], corpus.chunks, frequency),
            [chunk.chunk_id],
            5,
        )
        for chunk in corpus.chunks[:40]
    ]

    assert sum(scores) / len(scores) >= 0.95


def test_an_oracle_ranking_scores_perfectly_under_the_same_metric(
    corpus: BenchmarkCorpus,
) -> None:
    """Control 2. If ``_recall_at_k`` could not reach 1.0 the threshold below
    would be unreachable for reasons that have nothing to do with the data."""
    for query in corpus.queries[:5]:
        oracle = list(query.relevant_chunk_ids) + [
            c.chunk_id
            for c in corpus.chunks
            if c.chunk_id not in query.relevant_chunk_ids
        ]
        assert _recall_at_k(oracle, query.relevant_chunk_ids, 5) == pytest.approx(
            min(1.0, 5 / len(query.relevant_chunk_ids))
        )


def test_the_query_set_is_not_satisfiable_by_keyword_overlap(
    corpus: BenchmarkCorpus,
) -> None:
    """The property that makes task 6.6's Observable meaningful.

    With the baseline shown able to score 1.0 on lexically-overlapping queries
    and the metric shown able to reach its ceiling, a low score here is a fact
    about the *queries*: their relevant chunks are topically rather than
    literally related, so retrieving them requires a semantic representation.
    """
    frequency = _document_frequency(corpus.chunks)
    per_query = [
        _recall_at_k(
            _lexical_ranking(query.text, corpus.chunks, frequency),
            query.relevant_chunk_ids,
            5,
        )
        for query in corpus.queries
    ]
    mean_recall = sum(per_query) / len(per_query)

    assert mean_recall <= 0.35, (
        f"the lexical baseline already answers this query set "
        f"(recall@5 = {mean_recall:.2f}); it would not discriminate between a "
        f"working embedding and a semantically broken one"
    )
    assert max(per_query) < 1.0, "at least one query is fully answered lexically"


def test_no_query_repeats_a_phrase_from_its_own_relevant_chunks(
    corpus: BenchmarkCorpus,
) -> None:
    """A blunter, per-query form of the same concern that does not depend on a
    ranking metric: no query may lift a run of words out of a chunk it marks
    relevant."""
    by_id = {c.chunk_id: c for c in corpus.chunks}
    for query in corpus.queries:
        query_grams = _ngrams(query.text, 4)
        for chunk_id in query.relevant_chunk_ids:
            shared = query_grams & _ngrams(by_id[chunk_id].text, 4)
            assert not shared, f"{query.query_id} lifts {shared} from {chunk_id}"


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _cosine(a: set[str], b: set[str], document_frequency: Counter[str], total: int) -> float:
    def weight(term: str) -> float:
        return math.log(1 + total / (1 + document_frequency[term]))

    numerator = sum(weight(term) for term in a & b)
    denominator = math.sqrt(sum(weight(t) for t in a)) * math.sqrt(
        sum(weight(t) for t in b)
    )
    return numerator / denominator if denominator else 0.0


def _mean_intra_query_coherence(
    queries: Sequence[RelevanceQuery], chunks: Sequence[CorpusChunk]
) -> float:
    """Mean similarity between chunks that share a relevance label, expressed as
    a multiple of the mean similarity between two chunks picked at random."""
    frequency = _document_frequency(chunks)
    total = len(chunks)
    vectors = {c.chunk_id: _terms(f"{c.article_title} {c.text}") for c in chunks}
    baseline = [
        _cosine(vectors[a.chunk_id], vectors[b.chunk_id], frequency, total)
        for a, b in combinations(chunks, 2)
    ]
    corpus_mean = sum(baseline) / len(baseline)
    per_query = [
        sum(
            _cosine(vectors[a], vectors[b], frequency, total)
            for a, b in combinations(query.relevant_chunk_ids, 2)
        )
        / max(1, len(list(combinations(query.relevant_chunk_ids, 2))))
        for query in queries
    ]
    return (sum(per_query) / len(per_query)) / corpus_mean


def test_the_relevance_labels_are_not_arbitrary(corpus: BenchmarkCorpus) -> None:
    """The opposite failure to the one above, and it has to be checked too.

    A query set can defeat a keyword baseline simply by labelling chunks that
    have nothing to do with the query - which would make task 6.6 unanswerable
    rather than discriminating, and no test written so far would notice. Chunks
    that share a relevance label are therefore required to resemble each other
    more than two chunks drawn at random do.

    The control is the second assertion: relabelling the same queries with the
    same *number* of chunks, drawn deterministically from elsewhere in the
    sample, must collapse the statistic to roughly the corpus baseline. Without
    it, a threshold above 1.0 could be met by an artefact of the metric.
    """
    real = _mean_intra_query_coherence(corpus.queries, corpus.chunks)

    shuffled = [
        RelevanceQuery(
            query_id=query.query_id,
            text=query.text,
            relevant_chunk_ids=tuple(
                corpus.chunks[(offset * 37 + position * 53) % len(corpus.chunks)].chunk_id
                for position in range(len(query.relevant_chunk_ids))
            ),
        )
        for offset, query in enumerate(corpus.queries)
    ]
    arbitrary = _mean_intra_query_coherence(shuffled, corpus.chunks)

    assert real >= 2.0, (
        f"chunks sharing a relevance label are only {real:.2f}x as similar to "
        f"each other as two random chunks; the judgements may be arbitrary"
    )
    assert arbitrary < 1.5, (
        f"the arbitrary-label control scored {arbitrary:.2f}x, so the statistic "
        f"does not discriminate and the assertion above proves nothing"
    )


# --------------------------------------------------------------------------
# Loader shape
# --------------------------------------------------------------------------


def test_the_loaded_types_are_frozen_value_objects(corpus: BenchmarkCorpus) -> None:
    assert isinstance(corpus.chunks[0], CorpusChunk)
    assert isinstance(corpus.queries[0], RelevanceQuery)
    assert isinstance(corpus.composition, SampleComposition)
    with pytest.raises(AttributeError):
        corpus.chunks[0].text = "mutated"  # type: ignore[misc]


def test_chunks_can_be_addressed_by_id(corpus: BenchmarkCorpus) -> None:
    """Tasks 6.6 and 6.7 report against chunk ids, so resolving one is the
    loader's job rather than every consumer's."""
    chunk = corpus.chunks[3]

    assert corpus.chunk_by_id(chunk.chunk_id) is chunk
    with pytest.raises(KeyError):
        corpus.chunk_by_id("no-such-chunk")


def test_chunk_order_is_the_committed_file_order(corpus: BenchmarkCorpus) -> None:
    """Determinism matters for 6.7: the benchmark reports against ids, and a
    load order that varied would make two runs incomparable for reasons
    unrelated to the models."""
    lines = (FIXTURE_DIR / "chunks.jsonl").read_text("utf-8").splitlines()
    committed = [json.loads(line)["chunk_id"] for line in lines if line.strip()]

    assert [c.chunk_id for c in corpus.chunks] == committed


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _fixture_copy_with_mutated_chunks(
    tmp_path: Path,
    mutate: object,
) -> Path:
    destination = tmp_path / "benchmark-corpus"
    shutil.copytree(FIXTURE_DIR, destination)
    path = destination / "chunks.jsonl"
    records = [
        json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()
    ]
    assert callable(mutate)
    mutated = mutate(records)
    path.write_text(
        "\n".join(json.dumps(record) for record in mutated) + "\n", encoding="utf-8"
    )
    return destination


def _replace_first(
    records: list[dict[str, object]], field: str, value: object
) -> list[dict[str, object]]:
    head = dict(records[0])
    head[field] = value
    return [head, *records[1:]]
