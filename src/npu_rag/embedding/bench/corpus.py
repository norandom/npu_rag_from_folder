"""Load the committed benchmark fixture (task 6.1, requirement 6.5).

Requirement 6.5 says the runtime "shall use a representative sample of the
target corpus for benchmarking, and shall state the sample's size and
composition, so that the benchmark does not depend on a completed corpus
index". design.md's Out of Boundary section fixes the mechanism: the benchmark
"does not read ``.md`` or ``.pdf`` files. It consumes a small pre-extracted
fixture committed at ``tests/fixtures/benchmark-corpus/``, generated once by an
ad hoc script and thereafter treated as static test data. This keeps extraction
wholly inside ``document-ingest`` and lets this spec be implemented in parallel
with it."

This module is the reading half of that arrangement. The writing half is
``tools/build_benchmark_corpus.py``, which is throwaway tooling: it is committed
so the fixture's provenance is reproducible, and it is never imported from here
or from anywhere else under ``src/``. Nothing in this module knows where the
source archive is, which is what makes task 6.1's Observable structural rather
than a promise.

Where this module sits, and what it deliberately is not
-------------------------------------------------------

design.md's Requirements Traceability maps 6.5 to ``bench/harness.py`` and a
``SampleSpec`` type. That file belongs to **task 6.3**, which owns the
model-by-provider matrix. Creating it here to hold a fixture loader would take
6.3's file away from it and force a merge, so the loader lives in its own
``bench/corpus.py`` and names its composition record ``SampleComposition``
rather than ``SampleSpec``. When 6.3 writes ``harness.py`` it can build whatever
run-shaped ``SampleSpec`` it needs from a ``BenchmarkCorpus`` this module
returns; nothing here anticipates the shape of a benchmark run.

No new layer rank was needed for this file: ``bench`` already sits at the top of
design.md's dependency direction, and the package-wide guard in
``tests/embedding/providers/test_base.py`` keys a module on its top-level
package directory, so ``bench/corpus.py`` is placed by ``bench``'s existing
entry (Implementation Note 5.1).

Why the loader validates rather than trusts
-------------------------------------------

The recurring defect in this project is a fixture that makes a check trivially
true. A loader that simply parsed JSON would let every one of the following
through without a sound, and each of them corrupts a *later* task's conclusion
rather than this one's:

- a chunk longer than the excerpt cap - a licensing problem in a public
  repository, and the reason the cap exists;
- more than ``MAX_CHUNKS_PER_ARTICLE`` chunks from one article - the same
  concern, in the dimension a per-chunk cap cannot see;
- a duplicate chunk id - ``relevance.jsonl`` addresses chunks by id, so an
  ambiguous id makes a relevance judgement refer to two different texts;
- a relevance judgement naming a chunk that does not ship - task 6.6 would score
  it as an unretrievable relevant document and read the permanently depressed
  score as a property of the *model*;
- a composition record that has drifted from the data it describes - the record
  is exactly the statement requirement 6.5 asks the benchmark to make, and task
  6.7 renders it into the benchmark document.

So the invariants are enforced here, at load, on every run - not in the
generator, which is throwaway and whose output outlives it.

The excerpt cap costs the benchmark nothing measurable
------------------------------------------------------

The NPU graph is compiled at a static ``(batch_size, 512)``, so every input is
padded to the compiled length and consumes identical compute regardless of its
real text length. Throughput, latency, energy and peak memory are therefore
insensitive to excerpt length. Only retrieval quality (task 6.6) reads the text
at all, and a ~500-character excerpt is a substantive paragraph that embeds
meaningfully. Raising the cap would buy no realism and would put more of someone
else's writing into a public repository.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_CHUNKS_PER_ARTICLE",
    "MAX_CHUNK_CHARACTERS",
    "BenchmarkCorpus",
    "CorpusChunk",
    "CorpusFixtureError",
    "RelevanceQuery",
    "SampleComposition",
    "default_corpus_directory",
    "load_corpus",
]

#: The excerpt cap, in characters. Decided by the project owner on 2026-09-07:
#: this repository pushes to public GitHub and the archive is third-party
#: Substack content, so the fixture carries short excerpts only.
MAX_CHUNK_CHARACTERS = 500

#: How many chunks one source article may contribute. A per-chunk cap bounds
#: each excerpt; only this bounds how much of a single article the fixture
#: reproduces in total.
MAX_CHUNKS_PER_ARTICLE = 2

CHUNKS_FILE = "chunks.jsonl"
RELEVANCE_FILE = "relevance.jsonl"
COMPOSITION_FILE = "composition.json"


class CorpusFixtureError(ValueError):
    """The committed fixture is absent, malformed, or violates its own policy.

    A ``ValueError`` rather than a member of ``errors.EmbeddingRuntimeError``:
    house style (Implementation Note 5.4) puts construction invariants on
    ``ValueError``, while requirement 8.1's provider/model/stage obligation
    attaches to a failing embedding *operation*. Loading static test data has
    neither a resolved provider nor a preparation stage.
    """


@dataclass(frozen=True)
class CorpusChunk:
    """One excerpt, with the provenance requirement 6.5's "composition" needs."""

    chunk_id: str
    publication: str
    article_id: str
    article_title: str
    text: str


@dataclass(frozen=True)
class RelevanceQuery:
    """One hand-built query and the chunks a reader judged relevant to it.

    Requirement 6.4 asks for "a fixed query set drawn from the target corpus
    with pre-identified relevant results". The judgements are made by reading,
    not derived by a rule - a rule would only measure itself.
    """

    query_id: str
    text: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class SampleComposition:
    """The statement requirement 6.5 asks the benchmark to make about its sample.

    It is committed rather than recomputed, because it records the *policy* the
    sample was drawn under - the caps, the archive, the generator, how articles
    were allocated and spread - which no amount of counting the data recovers.
    A committed record can drift from its data, so ``load_corpus`` cross-checks
    every counted field against the chunks that actually ship.
    """

    source: str
    generated_by: str
    sampling: str
    excerpt_policy: str
    chunk_count: int
    article_count: int
    publication_count: int
    publications: tuple[str, ...]
    chunks_per_publication: Mapping[str, int]
    articles_per_publication: Mapping[str, int]
    max_chunk_characters: int
    max_chunks_per_article: int
    observed_min_chunk_characters: int
    observed_max_chunk_characters: int
    observed_mean_chunk_characters: float
    observed_total_chunk_characters: int
    observed_max_chunks_per_article: int
    query_count: int
    relevant_chunks_per_query_min: int
    relevant_chunks_per_query_max: int
    relevant_chunks_per_query_mean: float


@dataclass(frozen=True)
class BenchmarkCorpus:
    """The whole fixture: the sample, the query set, and the statement about it."""

    chunks: tuple[CorpusChunk, ...]
    queries: tuple[RelevanceQuery, ...]
    composition: SampleComposition

    def chunk_by_id(self, chunk_id: str) -> CorpusChunk:
        """Resolve a chunk id. Tasks 6.6 and 6.7 report against ids, so this
        belongs to the loader rather than to every consumer."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(chunk_id)


def default_corpus_directory() -> Path:
    """``tests/fixtures/benchmark-corpus`` beside this checkout.

    A convenience for a caller working in the repository - the benchmark, and
    tasks 6.3 and 6.6. It is resolved from this file rather than from the
    working directory, and it is *not* a default parameter anywhere: design.md's
    Out of Boundary section says callers pass explicit parameters, and a loader
    that silently found its own data would be reaching outside the directory it
    was handed.
    """
    return Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "benchmark-corpus"


def load_corpus(directory: Path) -> BenchmarkCorpus:
    """Read the fixture from ``directory`` and enforce every policy it claims.

    Reads only files inside ``directory``. Raises ``CorpusFixtureError`` rather
    than returning a corpus that would quietly mislead a later measurement.
    """
    chunks = _read_chunks(directory / CHUNKS_FILE)
    queries = _read_queries(directory / RELEVANCE_FILE, {c.chunk_id for c in chunks})
    composition = _read_composition(directory / COMPOSITION_FILE, chunks, queries)
    return BenchmarkCorpus(
        chunks=tuple(chunks), queries=tuple(queries), composition=composition
    )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CorpusFixtureError(f"the benchmark fixture is missing {path.name}: {path}")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:  # pragma: no cover - malformed commit
            raise CorpusFixtureError(f"{path.name} line {number} is not JSON") from error
        if not isinstance(record, dict):
            raise CorpusFixtureError(f"{path.name} line {number} is not an object")
        records.append(record)
    if not records:
        raise CorpusFixtureError(f"{path.name} is empty")
    return records


def _field(record: Mapping[str, Any], name: str, path: Path) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise CorpusFixtureError(f"{path.name}: {name!r} must be a non-empty string")
    return value


def _read_chunks(path: Path) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    seen: set[str] = set()
    for record in _read_json_lines(path):
        chunk = CorpusChunk(
            chunk_id=_field(record, "chunk_id", path),
            publication=_field(record, "publication", path),
            article_id=_field(record, "article_id", path),
            article_title=_field(record, "article_title", path),
            text=_field(record, "text", path),
        )
        if chunk.chunk_id in seen:
            raise CorpusFixtureError(
                f"duplicate chunk id {chunk.chunk_id!r}: relevance judgements "
                f"address chunks by id, so a repeated id names two texts"
            )
        seen.add(chunk.chunk_id)
        _check_identity(chunk)
        _check_excerpt_cap(chunk)
        chunks.append(chunk)
    _check_per_article_ceiling(chunks)
    return chunks


def _check_identity(chunk: CorpusChunk) -> None:
    expected = f"{chunk.publication}/{chunk.article_id}#"
    if not chunk.chunk_id.startswith(expected):
        raise CorpusFixtureError(
            f"chunk id {chunk.chunk_id!r} does not match its own provenance "
            f"(expected it to begin {expected!r})"
        )


def _check_excerpt_cap(chunk: CorpusChunk) -> None:
    if len(chunk.text) > MAX_CHUNK_CHARACTERS:
        raise CorpusFixtureError(
            f"chunk {chunk.chunk_id!r} exceeds the excerpt cap: "
            f"{len(chunk.text)} > {MAX_CHUNK_CHARACTERS} characters"
        )
    if chunk.text != chunk.text.strip():
        raise CorpusFixtureError(f"chunk {chunk.chunk_id!r} carries surrounding space")


def _check_per_article_ceiling(chunks: Sequence[CorpusChunk]) -> None:
    counts = Counter((chunk.publication, chunk.article_id) for chunk in chunks)
    for (publication, article), count in sorted(counts.items()):
        if count > MAX_CHUNKS_PER_ARTICLE:
            raise CorpusFixtureError(
                f"{count} chunks from one article ({publication}/{article}); "
                f"at most {MAX_CHUNKS_PER_ARTICLE} are permitted, so that no "
                f"article is substantially reproduced"
            )


def _read_queries(path: Path, known_chunk_ids: set[str]) -> list[RelevanceQuery]:
    queries: list[RelevanceQuery] = []
    seen: set[str] = set()
    for record in _read_json_lines(path):
        query_id = _field(record, "query_id", path)
        text = _field(record, "text", path)
        raw = record.get("relevant_chunk_ids")
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise CorpusFixtureError(
                f"{path.name}: {query_id!r} must carry a list of chunk ids"
            )
        if query_id in seen:
            raise CorpusFixtureError(f"duplicate query id {query_id!r}")
        seen.add(query_id)
        if not raw:
            raise CorpusFixtureError(
                f"query {query_id!r} has no relevant chunks, so no ranking can "
                f"score better or worse on it"
            )
        if len(set(raw)) != len(raw):
            raise CorpusFixtureError(f"query {query_id!r} repeats a relevant chunk id")
        for chunk_id in raw:
            if chunk_id not in known_chunk_ids:
                raise CorpusFixtureError(
                    f"query {query_id!r} names {chunk_id!r}, which is not in "
                    f"{CHUNKS_FILE}: a relevant document that cannot be "
                    f"retrieved would depress every model's score equally"
                )
        queries.append(
            RelevanceQuery(query_id=query_id, text=text, relevant_chunk_ids=tuple(raw))
        )
    return queries


def _read_composition(
    path: Path, chunks: Sequence[CorpusChunk], queries: Sequence[RelevanceQuery]
) -> SampleComposition:
    if not path.is_file():
        raise CorpusFixtureError(f"the benchmark fixture is missing {path.name}: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise CorpusFixtureError(f"{path.name} is not a JSON object")
    try:
        composition = SampleComposition(
            source=_field(record, "source", path),
            generated_by=_field(record, "generated_by", path),
            sampling=_field(record, "sampling", path),
            excerpt_policy=_field(record, "excerpt_policy", path),
            chunk_count=int(record["chunk_count"]),
            article_count=int(record["article_count"]),
            publication_count=int(record["publication_count"]),
            publications=tuple(record["publications"]),
            chunks_per_publication=dict(record["chunks_per_publication"]),
            articles_per_publication=dict(record["articles_per_publication"]),
            max_chunk_characters=int(record["max_chunk_characters"]),
            max_chunks_per_article=int(record["max_chunks_per_article"]),
            observed_min_chunk_characters=int(record["observed_min_chunk_characters"]),
            observed_max_chunk_characters=int(record["observed_max_chunk_characters"]),
            observed_mean_chunk_characters=float(
                record["observed_mean_chunk_characters"]
            ),
            observed_total_chunk_characters=int(
                record["observed_total_chunk_characters"]
            ),
            observed_max_chunks_per_article=int(
                record["observed_max_chunks_per_article"]
            ),
            query_count=int(record["query_count"]),
            relevant_chunks_per_query_min=int(record["relevant_chunks_per_query_min"]),
            relevant_chunks_per_query_max=int(record["relevant_chunks_per_query_max"]),
            relevant_chunks_per_query_mean=float(
                record["relevant_chunks_per_query_mean"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CorpusFixtureError(f"{path.name} is missing or malformed: {error}") from error
    _check_composition(composition, chunks, queries)
    return composition


def _check_composition(
    composition: SampleComposition,
    chunks: Sequence[CorpusChunk],
    queries: Sequence[RelevanceQuery],
) -> None:
    """Recompute every counted field from the data and refuse a disagreement.

    Without this, ``composition.json`` would be prose that happens to be JSON:
    task 6.7 renders it into the benchmark document as the statement requirement
    6.5 demands, so a stale count is a false claim in a published deliverable.
    """
    lengths = [len(chunk.text) for chunk in chunks]
    per_article = Counter((chunk.publication, chunk.article_id) for chunk in chunks)
    publications = sorted({chunk.publication for chunk in chunks})
    relevant = [len(query.relevant_chunk_ids) for query in queries]
    expected: dict[str, object] = {
        "chunk_count": len(chunks),
        "article_count": len(per_article),
        "publication_count": len(publications),
        "publications": tuple(publications),
        "chunks_per_publication": {
            name: sum(1 for chunk in chunks if chunk.publication == name)
            for name in publications
        },
        "articles_per_publication": {
            name: len(
                {chunk.article_id for chunk in chunks if chunk.publication == name}
            )
            for name in publications
        },
        "max_chunk_characters": MAX_CHUNK_CHARACTERS,
        "max_chunks_per_article": MAX_CHUNKS_PER_ARTICLE,
        "observed_min_chunk_characters": min(lengths),
        "observed_max_chunk_characters": max(lengths),
        "observed_mean_chunk_characters": round(sum(lengths) / len(lengths), 1),
        "observed_total_chunk_characters": sum(lengths),
        "observed_max_chunks_per_article": max(per_article.values()),
        "query_count": len(queries),
        "relevant_chunks_per_query_min": min(relevant),
        "relevant_chunks_per_query_max": max(relevant),
        "relevant_chunks_per_query_mean": round(sum(relevant) / len(relevant), 2),
    }
    disagreements = [
        f"{name}: recorded {getattr(composition, name)!r}, data says {value!r}"
        for name, value in expected.items()
        if getattr(composition, name) != value
    ]
    if disagreements:
        raise CorpusFixtureError(
            f"{COMPOSITION_FILE} disagrees with the committed data - "
            + "; ".join(disagreements)
        )
