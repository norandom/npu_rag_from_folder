"""Generate the committed benchmark fixture from the source archive (task 6.1).

This is the "ad hoc script" design.md's Out of Boundary section names when it
says the benchmark "does not read ``.md`` or ``.pdf`` files. It consumes a small
pre-extracted fixture committed at ``tests/fixtures/benchmark-corpus/``,
generated once by an ad hoc script and thereafter treated as static test data."

It is committed for provenance, not for reuse. Nothing in ``src/npu_rag/``
imports it, and the benchmark must never import it: the fixture's whole purpose
is to sever this spec's dependence on text extraction, which belongs to the
``document-ingest`` spec. It lives under ``tools/`` for the reason Implementation
Note 1.2 records - ``tools/`` is deliberately outside the package, because
design.md's Out of Boundary says the package detects and reports but never
mutates the system, and this script writes files.

Usage (the archive path is required and is never defaulted, because the archive
is not part of this repository)::

    uv run python -m tools.build_benchmark_corpus \
        --archive <path to the archive> \
        --out tests/fixtures/benchmark-corpus

It writes ``chunks.jsonl`` and ``composition.json``. It deliberately does **not**
write ``relevance.jsonl``: requirement 6.4's query set has "pre-identified
relevant results", and identifying them is a judgement about meaning that no
extraction rule can make. That file is hand-built and hand-maintained.

What the script commits, and what it refuses to
-----------------------------------------------

The archive is 1205 markdown files from three third-party Substack publications
and this repository pushes to public GitHub, so the fixture carries **short
excerpts only**: at most ``MAX_CHUNK_CHARACTERS`` per chunk and at most
``MAX_CHUNKS_PER_ARTICLE`` chunks from any one article, so that no article is
substantially reproduced. Each chunk records the publication and article it came
from, which is what makes requirement 6.5's "composition" a real statement
rather than a count.

**The cap costs the benchmark nothing measurable.** The NPU graph is compiled at
a static ``(batch_size, 512)``, so every input is padded to the compiled length
and consumes identical compute regardless of its actual text length. Throughput,
latency, energy and peak memory are insensitive to excerpt length; only
retrieval quality (task 6.6) reads the text at all, and a ~500-character excerpt
is a substantive paragraph that embeds meaningfully. Do not "improve" the
fixture later by lengthening chunks on the assumption that longer inputs are
more realistic. They are not more expensive, and they are more of someone else's
writing.

What "representative" means here
--------------------------------

Requirement 6.5 asks for "a representative sample of the target corpus". That
word is given four concrete commitments, each of which is checked by
``tests/embedding/bench/test_corpus.py`` against the committed data:

1. **Every publication contributes.** Sampling whichever directory sorts first
   would be a sample of one voice, not of the archive.
2. **The smaller publications are not token.** Allocation is proportional to
   each publication's share of the archive, subject to a floor
   (``MINIMUM_ARTICLES_PER_PUBLICATION``), so the two smaller publications carry
   enough articles to support relevance judgements about their subject matter.
3. **Selection spreads across each publication's history**, by walking the
   date-ordered filenames at an even stride rather than taking a contiguous run.
   A month of one publication would satisfy every count and sample one moment.
4. **Chunks are drawn from different positions inside an article**, one from the
   opening quarter and one from the closing quarter, so the sample is not made
   entirely of introductions.

Selection is fully deterministic - a stride over a sorted list, with no random
number generator - so regenerating from the same archive reproduces the same
fixture, and a diff means the archive changed.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")

# --------------------------------------------------------------------------
# Policy. These are duplicated in npu_rag.embedding.bench.corpus, which
# ENFORCES them at load time; here they only shape what gets written. The
# duplication is forced - the package must not import from tools/ - and the two
# are pinned to each other by tests/tools/test_build_benchmark_corpus.py, which
# may import both sides because it is not the package.
# --------------------------------------------------------------------------

MAX_CHUNK_CHARACTERS = 500
MAX_CHUNKS_PER_ARTICLE = 2

#: A chunk shorter than this is a caption or a stray sentence, not a paragraph
#: that embeds meaningfully.
MINIMUM_CHUNK_CHARACTERS = 260

#: Total articles to draw. 120 articles at up to 2 chunks each is a pool of
#: roughly 190 excerpts - enough for a retrieval ranking to be informative,
#: while staying well inside "a sample" of a 1205-article archive.
TARGET_ARTICLES = 120

#: The floor from commitment 2 above.
MINIMUM_ARTICLES_PER_PUBLICATION = 24

#: Substack furniture that appears verbatim across hundreds of posts. Left in,
#: it would dominate the sample's term statistics and make several chunks near
#: duplicates of each other - noise in the retrieval measurement attributable to
#: nothing about the models.
BOILERPLATE_MARKERS = (
    "reader-supported publication",
    "paid subscriber",
    "paid subscription",
    "free subscriber",
    "subscribe now",
    "read full story",
    "thanks for reading",
    "share this post",
    "leave a comment",
    "upgrade to paid",
    "you are receiving this",
    "unsubscribe",
    "click here to",
    "sponsored by",
    "this post is for",
)

_ARCHIVE_FILENAME = re.compile(r"^(?P<stamp>\d{8})_(?P<time>\d{6})_(?P<slug>.+)$")
_IMAGE_LINK = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_INLINE_LINK = re.compile(r"\[([^\]\[]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"<?https?://\S+?>?(?=[\s)]|$)")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_RULE = re.compile(r"^\s*([*\-_]\s*){3,}$")
_LIST_MARKER = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,2}[.)])\s+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*")
_QUOTE = re.compile(r"^\s{0,3}>\s?")
_UNDERSCORE_EMPHASIS = re.compile(r"(?<![A-Za-z0-9_])_+|_+(?![A-Za-z0-9_])")
_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!\"'>~|])")
_EMPTY_PARENS = re.compile(r"\(\s*\)|\[\s*\]")
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)]?\s")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    publication: str
    article_id: str
    article_title: str
    text: str


# --------------------------------------------------------------------------
# Markdown to prose
#
# The rule is strip scaffolding, never touch the words. Link *text* survives
# because it is the author's prose; the target URL does not, because it is
# markup. Nothing here paraphrases, reorders or substitutes.
# --------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Fold the invisible characters Substack exports leave behind.

    Zero-width spaces appear inside link text in this archive, so a chunk that
    kept them would embed differently from the same words typed by hand.
    """
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    return "".join(
        ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C"
    )


def strip_markup(line: str) -> str:
    """Remove markdown scaffolding from a single line."""
    line = _IMAGE_LINK.sub("", line)
    line = _IMAGE.sub("", line)
    previous = None
    while previous != line:  # nested link text, e.g. [**bold** word](url)
        previous = line
        line = _INLINE_LINK.sub(r"\1", line)
    line = _BARE_URL.sub("", line)
    line = _HTML_TAG.sub("", line)
    line = _HEADING.sub("", line)
    line = _QUOTE.sub("", line)
    line = _LIST_MARKER.sub("", line)
    line = line.replace("**", "").replace("`", "").replace("*", "")
    line = _UNDERSCORE_EMPHASIS.sub("", line)
    # Markdown escapes are scaffolding too: an author who wrote \" meant ",
    # and a chunk carrying the backslash would embed a character the author
    # never wrote.
    line = _MARKDOWN_ESCAPE.sub(r"\1", line)
    line = _EMPTY_PARENS.sub("", line)
    line = re.sub(r"\(\s+", "(", line)
    line = re.sub(r"\s+([)\].,;:!?])", r"\1", line)
    return re.sub(r"[ \t]+", " ", line).strip()


def paragraphs(markdown: str) -> Iterator[str]:
    """Yield the document's prose paragraphs, scaffolding removed."""
    buffer: list[str] = []
    for raw in normalise(markdown).splitlines():
        if _RULE.match(raw) or not raw.strip():
            if buffer:
                yield " ".join(buffer)
                buffer = []
            continue
        cleaned = strip_markup(raw)
        if cleaned:
            buffer.append(cleaned)
    if buffer:
        yield " ".join(buffer)


def is_boilerplate(paragraph: str) -> bool:
    lowered = paragraph.lower()
    return any(marker in lowered for marker in BOILERPLATE_MARKERS)


def is_prose(paragraph: str) -> bool:
    """Reject fragments that are punctuation, code or a caption rather than
    sentences. ``document-ingest`` will do this properly; here it only has to
    keep the sample readable."""
    letters = sum(ch.isalpha() or ch.isspace() for ch in paragraph)
    if not paragraph or letters / len(paragraph) < 0.85:
        return False
    return paragraph.count(" ") >= 20 and paragraph[0].isupper()


def cap(paragraph: str) -> str:
    """Cut a paragraph to the excerpt cap at the last sentence boundary, or at
    a word boundary when the paragraph holds no sentence end inside the cap."""
    if len(paragraph) <= MAX_CHUNK_CHARACTERS:
        return paragraph
    window = paragraph[: MAX_CHUNK_CHARACTERS + 1]
    ends = [match.end() for match in _SENTENCE_END.finditer(window)]
    if ends and ends[-1] >= MINIMUM_CHUNK_CHARACTERS:
        return window[: ends[-1]].strip()
    return window[: window.rindex(" ")].strip()


def article_title(markdown: str, fallback: str) -> str:
    for raw in normalise(markdown).splitlines():
        if raw.lstrip().startswith("#"):
            title = strip_markup(raw)
            if title:
                return title
    return fallback.replace("-", " ")


def candidates(markdown: str) -> list[str]:
    seen: list[str] = []
    for paragraph in paragraphs(markdown):
        if is_boilerplate(paragraph) or not is_prose(paragraph):
            continue
        excerpt = cap(paragraph)
        if len(excerpt) >= MINIMUM_CHUNK_CHARACTERS and excerpt not in seen:
            seen.append(excerpt)
    return seen


def positions(available: int, wanted: int) -> list[int]:
    """Which candidate paragraphs to take: spread through the article rather
    than taking its opening, so the sample is not made of introductions."""
    if available <= wanted:
        return list(range(available))
    chosen = sorted({(available * (2 * i + 1)) // (2 * wanted) for i in range(wanted)})
    return chosen[:wanted]


# --------------------------------------------------------------------------
# Selection across the archive
# --------------------------------------------------------------------------


def slug_of(path: Path) -> str:
    match = _ARCHIVE_FILENAME.match(path.stem)
    return match.group("slug") if match else path.stem


def article_id_of(path: Path) -> str:
    """``YYYYMMDD-slug``. The date leads so that ids sort chronologically and a
    reader can see at a glance which part of a publication's history a chunk
    came from; the test suite reads the leading year to check the spread."""
    match = _ARCHIVE_FILENAME.match(path.stem)
    if not match:
        return path.stem
    return f"{match.group('stamp')}-{match.group('slug')}"


def allocate(sizes: dict[str, int], total: int, floor: int) -> dict[str, int]:
    """Split ``total`` articles between publications in proportion to their size,
    but never below ``floor``. The floor is what stops the two smaller
    publications from being represented by a handful of posts that could not
    support a relevance judgement about their subject matter; the remainder is
    absorbed by the largest publication, which has articles to spare."""
    corpus_size = sum(sizes.values())
    if corpus_size == 0:
        raise ValueError(
            "no article in the archive yielded a usable paragraph; check the "
            "archive path and the extraction rules before committing anything"
        )
    allocation = {
        name: min(sizes[name], max(floor, round(total * size / corpus_size)))
        for name, size in sizes.items()
    }
    largest = max(sorted(allocation), key=lambda name: sizes[name])
    allocation[largest] = min(
        sizes[largest], allocation[largest] + total - sum(allocation.values())
    )
    return allocation


def ordered_selection(items: Sequence[_T], wanted: int) -> list[_T]:
    """Walk a date-ordered list at an even stride, so the selection covers the
    publication's whole history instead of a contiguous run."""
    if wanted >= len(items):
        return list(items)
    picked = sorted({(len(items) * (2 * i + 1)) // (2 * wanted) for i in range(wanted)})
    return [items[i] for i in picked]


def dedup_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


@dataclass(frozen=True)
class Article:
    publication: str
    path: Path
    article_id: str
    title: str
    candidates: tuple[str, ...]


def read_articles(archive: Path) -> dict[str, list[Article]]:
    """Read every article in the archive and extract its usable paragraphs.

    The whole archive is read rather than only the articles the stride will
    pick, because cross-article boilerplate can only be recognised by seeing
    every article: this publication repeats a "get some help via ..." paragraph
    verbatim across many posts, and two chunks carrying identical text would be
    a duplicate document in the retrieval pool, scored as if it were two.
    """
    by_publication: dict[str, list[Article]] = {}
    for directory in sorted(p for p in archive.iterdir() if p.is_dir()):
        articles: list[Article] = []
        for path in sorted(directory.rglob("*.md"), key=lambda p: p.name):
            markdown = path.read_text(encoding="utf-8", errors="replace")
            articles.append(
                Article(
                    publication=directory.name,
                    path=path,
                    article_id=article_id_of(path),
                    title=article_title(markdown, slug_of(path)),
                    candidates=tuple(candidates(markdown)),
                )
            )
        by_publication[directory.name] = articles
    return by_publication


def drop_repeated_across_articles(
    by_publication: dict[str, list[Article]],
) -> dict[str, list[Article]]:
    seen: Counter[str] = Counter()
    for articles in by_publication.values():
        for article in articles:
            seen.update({dedup_key(text) for text in article.candidates})
    return {
        name: [
            Article(
                publication=article.publication,
                path=article.path,
                article_id=article.article_id,
                title=article.title,
                candidates=tuple(
                    text
                    for text in article.candidates
                    if seen[dedup_key(text)] == 1
                ),
            )
            for article in articles
        ]
        for name, articles in by_publication.items()
    }


def build(archive: Path) -> list[Chunk]:
    by_publication = drop_repeated_across_articles(read_articles(archive))
    # Stride over the articles that actually yielded prose, so the even spacing
    # describes the selection rather than being disturbed by articles that turn
    # out to be all captions and lists.
    usable = {
        name: [article for article in articles if article.candidates]
        for name, articles in by_publication.items()
    }
    allocation = allocate(
        {name: len(articles) for name, articles in usable.items()},
        TARGET_ARTICLES,
        MINIMUM_ARTICLES_PER_PUBLICATION,
    )

    collected: list[Chunk] = []
    for name in sorted(usable):
        for article in ordered_selection(usable[name], allocation[name]):
            collected.extend(
                Chunk(
                    chunk_id=f"{name}/{article.article_id}#{index}",
                    publication=name,
                    article_id=article.article_id,
                    article_title=article.title,
                    text=article.candidates[index],
                )
                for index in positions(
                    len(article.candidates), MAX_CHUNKS_PER_ARTICLE
                )
            )
    return collected


def composition(
    chunks: Sequence[Chunk],
    queries: Sequence[Mapping[str, Any]],
    archive: Path,
) -> dict[str, object]:
    lengths = [len(chunk.text) for chunk in chunks]
    per_article = Counter((chunk.publication, chunk.article_id) for chunk in chunks)
    publications = sorted({chunk.publication for chunk in chunks})
    relevant_counts = [len(query["relevant_chunk_ids"]) for query in queries]
    return {
        "source": (
            f"{archive.name} - {len(publications)} Substack publications, "
            "markdown export, text only"
        ),
        "generated_by": "tools/build_benchmark_corpus.py",
        "sampling": (
            "Articles are allocated to publications in proportion to their share "
            "of the archive, subject to a floor, then drawn at an even stride "
            "over the date-ordered filenames so the selection spans each "
            "publication's history. Within an article, chunks are taken from "
            "spread positions rather than from the opening. Selection is "
            "deterministic: no random number generator is involved."
        ),
        "excerpt_policy": (
            "Third-party writing in a public repository: each chunk is capped at "
            f"{MAX_CHUNK_CHARACTERS} characters and no article contributes more "
            f"than {MAX_CHUNKS_PER_ARTICLE} chunks, so no article is "
            "substantially reproduced. The cap does not affect throughput, "
            "latency, energy or memory, because the NPU graph is compiled at a "
            "static (batch_size, 512) and every input is padded to that length."
        ),
        "chunk_count": len(chunks),
        "article_count": len(per_article),
        "publication_count": len(publications),
        "publications": publications,
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
        "relevant_chunks_per_query_min": min(relevant_counts) if queries else 0,
        "relevant_chunks_per_query_max": max(relevant_counts) if queries else 0,
        "relevant_chunks_per_query_mean": (
            round(sum(relevant_counts) / len(relevant_counts), 2) if queries else 0.0
        ),
    }


def write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    chunks = build(args.archive)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.out / "chunks.jsonl",
        [
            {
                "chunk_id": chunk.chunk_id,
                "publication": chunk.publication,
                "article_id": chunk.article_id,
                "article_title": chunk.article_title,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    )

    relevance_path = args.out / "relevance.jsonl"
    queries: list[Mapping[str, Any]] = []
    if relevance_path.is_file():
        queries = [
            json.loads(line)
            for line in relevance_path.read_text("utf-8").splitlines()
            if line.strip()
        ]

    (args.out / "composition.json").write_text(
        json.dumps(composition(chunks, queries, args.archive), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
