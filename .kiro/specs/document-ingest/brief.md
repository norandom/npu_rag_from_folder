# Brief: document-ingest

## Problem

The source material is a scraped archive: 5,207 files totalling 2.4 GB, of which only 1,343 are text-bearing (1,205 `.md`, 135 `.pdf`, 3 `.txt`) and roughly 3,800 are images. The text is heterogeneous — Markdown with embedded HTML, image references, and front-matter, alongside PDFs whose 237 MB is mostly fonts and raster content wrapping a comparatively small amount of extractable prose.

None of that can be embedded directly. It must be found, extracted, cleaned, and cut into pieces that fit a fixed token budget — and it must be re-runnable cheaply, because the archive grows as new posts are downloaded. Re-embedding 30k chunks to pick up five new articles is the difference between a tool that gets used and one that does not.

## Current State

- The archive exists at `D:\Source\substack_dl\archive`, organized per author (e.g. `archive/<author>/...`), populated by a separate `substack_dl` tool outside this project's control.
- No code exists.
- Measured content: ~1.0M words of Markdown. PDF text volume not yet measured — 237 MB on disk is not a useful proxy, since most of it is fonts and raster images.

## Desired Outcome

- A single call walks a configurable root and returns clean, chunked, metadata-tagged text records ready for embedding.
- Re-running after new files appear embeds only what is new or changed, and removes records for deleted files.
- Extraction failures on individual files are reported and skipped, never fatal to the run.
- The module is pure and hardware-free: fully testable without an NPU, a model, or a vector store.

## Approach

A pipeline of small, independently testable stages: discover, extract, normalize, chunk, hash. Each format gets its own extractor behind a common interface, so PDF handling can be swapped without touching Markdown handling.

Incremental behaviour is built on a per-file content hash persisted in a file-state table. On each run, files are classified as new, changed, unchanged, or deleted, and only the first two categories produce work. The hash covers file content plus the parameters that affect chunking, so changing the chunk size correctly invalidates prior work rather than silently mixing incompatible chunks.

Chunking targets a token budget that is injected, not decided here — the correct value is the compiled static sequence length published by `npu-embedding-runtime`.

## Scope

- **In**: Recursive walk of a configurable root with include/exclude rules; Markdown extraction (strip HTML, image refs, front-matter; preserve headings and structure); PDF text extraction; plain-text handling; Unicode and whitespace normalization; token-aware chunking with configurable size and overlap; heading and section-aware splitting where structure permits; per-chunk metadata (source path, author derived from directory structure, title, position, character offsets); per-file content hashing and a persisted file-state table; new/changed/deleted classification; per-file error isolation and reporting.
- **Out**: Images of any kind — no OCR, no captioning, no image metadata (~3,800 files skipped by design). Embedding (`npu-embedding-runtime`). Storage of chunks or vectors (`vector-index`). Deciding the token budget value. Scraping or downloading — the archive is produced externally. Any network access.

## Boundary Candidates

- Filesystem discovery and filtering
- Per-format text extraction (Markdown / PDF / plain text)
- Normalization and cleaning
- Token-aware chunking and metadata attachment
- File-state tracking and change classification

## Out of Boundary

- The value of the chunk token budget — an input from `npu-embedding-runtime`.
- Which tokenizer is authoritative. This spec must use the model's tokenizer, supplied as a dependency, not an approximation of its own.
- Deduplication of near-identical content across files.
- Watch/daemon mode. Incremental re-run on demand is in scope; a long-running watcher is explicitly deferred.

## Upstream / Downstream

- **Upstream**: The `substack_dl` archive on disk (read-only, externally maintained). The tokenizer and max sequence length from `npu-embedding-runtime`.
- **Downstream**: `vector-index` consumes chunk records and metadata. `search-cli` surfaces ingest progress, counts, and skipped-file reports.

## Existing Spec Touchpoints

- **Extends**: None. Greenfield.
- **Adjacent**: `npu-embedding-runtime` — shares the token-budget and tokenizer contract. Both are wave-1, and only one may own tokenization. It is the runtime spec, because the tokenizer belongs to the model.

## Constraints

- Text only: `.md`, `.pdf`, `.txt`. Images are out of scope by explicit decision.
- Source root must be a configuration variable defaulting to `D:\Source\substack_dl\archive`, never hardcoded.
- Chunk token budget is a parameter, constrained by the NPU's fixed compiled sequence length.
- Windows paths, long paths, and non-ASCII filenames must work — the archive contains all three.
- Must run without an NPU present, so ingest is developable and testable before the hardware spike concludes.
- Incremental re-run must be substantially cheaper than a full rebuild. That is the feature's reason for existing.
