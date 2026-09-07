# Benchmark corpus fixture

Static test data for the embedding benchmark (task 6.1, requirement 6.5). The
machine-readable statement of the sample's size and composition is
`composition.json`, not this file — task 6.7 renders that record into
`docs/embedding-benchmark.md`, and `npu_rag.embedding.bench.corpus.load_corpus`
cross-checks every counted field in it against the data that ships beside it.
This file is the human-readable provenance note.

## Files

| File | What it is |
|------|------------|
| `chunks.jsonl` | The sample. One excerpt per line: `chunk_id`, `publication`, `article_id`, `article_title`, `text`. |
| `relevance.jsonl` | The query set. One query per line: `query_id`, `text`, `relevant_chunk_ids`. **Hand-built** — the judgements are made by reading, not derived by a rule. |
| `composition.json` | The size, composition and sampling policy, cross-checked on load. |

## Why the fixture exists

design.md's Out of Boundary section: the benchmark "does not read `.md` or
`.pdf` files. It consumes a small pre-extracted fixture committed at
`tests/fixtures/benchmark-corpus/`, generated once by an ad hoc script and
thereafter treated as static test data. This keeps extraction wholly inside
`document-ingest` and lets this spec be implemented in parallel with it."

## Where it came from

`tools/build_benchmark_corpus.py`, run once against a local archive of 1205
markdown files exported from three Substack publications. The archive is **not**
part of this repository and nothing in `src/` knows where it is. To reproduce:

```
uv run python -m tools.build_benchmark_corpus --archive <path> --out tests/fixtures/benchmark-corpus
```

Selection is deterministic — a stride over sorted filenames, no random number
generator — so regenerating from the same archive reproduces the same file, and
a diff means the archive changed. The generator reads `relevance.jsonl` when
recomputing `composition.json` but never writes it.

## The excerpt cap, and why it costs the benchmark nothing

The source is third-party writing and this repository pushes to public GitHub,
so the fixture carries **short excerpts only**: at most **500 characters** per
chunk and at most **2 chunks** from any one article. Both limits are enforced by
`load_corpus` on every read, not merely by the generator's good behaviour — the
generator is throwaway and the data outlives it.

The cap does not weaken any measurement. The NPU graph is compiled at a static
`(batch_size, 512)`, so every input is padded to the compiled length and
consumes identical compute regardless of its real text length. Throughput,
latency, energy and peak memory are insensitive to excerpt length. Only
retrieval quality (task 6.6) reads the text at all, and a ~500-character excerpt
is a substantive paragraph that embeds meaningfully.

**Do not lengthen the chunks later** on the assumption that longer inputs are
more realistic. They are not more expensive, and they are more of someone else's
writing.

## What "representative" means here (requirement 6.5)

Four commitments, each checked against the committed data by
`tests/embedding/bench/test_corpus.py`:

1. Every publication in the archive contributes.
2. None contributes by a token article — allocation is proportional to each
   publication's share of the archive, subject to a floor.
3. Selection spans each publication's history, by walking date-ordered
   filenames at an even stride rather than taking a contiguous run.
4. Chunks come from spread positions inside an article, not only its opening.

## Why the queries read the way they do

Task 6.6's Observable is that "a model whose dense stage is omitted scores
visibly worse". That only discriminates if the query set cannot be answered
without understanding the text, so the queries are written to be **topically but
not lexically** related to their relevant chunks: they paraphrase rather than
quote, and no query shares a four-word run with any chunk it marks relevant.

Measured against the committed data by an IDF-weighted bag-of-words baseline,
mean recall@5 is **0.08**. The same baseline scores **1.00** on control queries
copied out of the chunks themselves, which is what makes the low figure evidence
about the queries rather than about a broken scorer. See
`test_the_query_set_is_not_satisfiable_by_keyword_overlap`.

The text is quoted verbatim from the source with markdown scaffolding removed —
image and link markup, URLs, heading and list markers, emphasis characters and
subscription boilerplate. No wording is paraphrased or altered.
