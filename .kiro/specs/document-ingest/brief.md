# Brief: document-ingest

> **Redesigned 2026-09-07.** This brief previously scoped a text-only pipeline over `.md`, `.pdf` and `.txt`, with images explicitly excluded ("no OCR, no captioning, no image metadata (~3,800 files skipped by design)"). Two things changed that. First, **spreadsheets are a priority format** and were absent from the original scope entirely. Second, measurement of the archive showed the excluded images are not incidental — they are **90% of the corpus by size and carry most of its quantitative content**. Both prior decisions are reversed here, deliberately and with the evidence below.

## Problem

The source material is heterogeneous and mostly *not* prose. Measured 2026-09-07:

| kind | count | size | share of bytes |
| --- | ---: | ---: | ---: |
| images (`.png`/`.jpeg`/`.gif`/`.webp`) | 3,837 | 2,129 MB | **90%** |
| PDFs | 135 | 237 MB | 10% |
| Markdown | 1,205 | ~1.0M words | <1% |
| plain text | 3 | — | — |
| **Excel** | **0 today** | — | — |

Two facts drive the redesign.

**The images are the content, not decoration.** These are business and quantitative newsletters; the charts carry the numbers. Only **9.8%** of image references in the Markdown have alt text (399 with, 3,688 without), and what exists is thin slugs — `innovation-theory`, `Reddit ARPU`, `facebook-arpu` — not descriptions. So indexing images by their surrounding text covers a tenth of them, badly. Excluding images means excluding most of what a quantitative question would actually ask about.

**Excel is a priority format and is not in the archive yet.** The owner's spreadsheets will be added to the corpus. They are **financial models**, not flat data exports: labelled line items down the left, periods across the top, several labelled regions per sheet, formulas throughout, and embedded charts that matter. Nothing about the measured archive above describes them, so this spec must not be tuned to a snapshot that excludes its priority input.

None of this can be embedded directly. It must be discovered, extracted, cleaned, and cut into pieces that fit a fixed token budget — and re-running must be cheap, because the archive grows.

## Current State

- Archive at `D:\Source\substack_dl\archive`, organised per author, populated by a separate `substack_dl` tool outside this project's control. **The Excel sheets are not there yet** and may live elsewhere; the source root is configuration, and more than one root must be expressible.
- No code exists.
- `npu-embedding-runtime` is complete and closed. It publishes the tokenizer and a compiled sequence length of **512 tokens**, which is the chunk budget. Its default model is `embeddinggemma-300m` (768-dim, licence-gated, has a Dense stage).

## Desired Outcome

- A single call walks configurable roots and returns clean, chunked, metadata-tagged records ready for embedding.
- Re-running after files appear embeds only what is new or changed, and removes records for deleted files.
- Extraction failures on individual files are reported and skipped, never fatal to the run.
- The module needs **no NPU and no network**. It needs local ONNX models for the vision path, and must degrade to text-only when they are absent rather than fail.

## Approach

A pipeline of small, independently testable stages — discover, route, extract, normalize, chunk, hash — with each format behind a common extractor interface.

**Extraction is routed by what a file actually contains, not by its extension alone.** The expensive path must stay rare:

1. **Markdown / plain text** — strip HTML, front-matter and image syntax; preserve headings; keep image references as anchors so an extracted figure can be tied back to its position in the post.
2. **PDF with a usable text layer** — direct extraction. Covers most documents and is orders of magnitude cheaper than OCR. Scientific and financial charts generated as *vector* graphics carry real text objects, so axis labels, legends and titles come out here for free.
3. **PDF without a usable text layer, and raster images** — the vision path below.
4. **Excel** — spreadsheet library, no vision model for the cells themselves.

### The vision path — a hosted API, optional (decided 2026-09-07)

**A local ONNX vision stack was evaluated and rejected on simplicity.** The candidate was the DeepDoc model set RAGFlow uses (`det`/`rec`/`layout`/`tsr`, ~103 MB, Apache 2.0, PaddleOCR-derived). It works, but it means wiring four models, a reading-order heuristic, and their failure modes into a pipeline whose job is text extraction. Rejected in favour of one HTTP call.

**The vision path is an OpenRouter call to a vision-capable chat model, defaulting to a Mistral one.** Mistral's dedicated `/v1/ocr` document endpoint is the better instrument for whole-document parsing, but it is not reachable through OpenRouter and is not practically available to an individual account — so this spec targets **OpenRouter only**, with the model named as a **configuration string** so it can be changed without touching code.

Applies to three inputs, all producing text that the ordinary embedding path then consumes:

- raster image files referenced from Markdown posts
- PDF pages with no usable text layer
- charts and images embedded in spreadsheets

**It is optional and the pipeline is complete without it.** With no API key configured, those inputs are **reported as requiring vision and skipped by name and count** — never silently dropped, and never guessed at. This is requirement-6.8-style discipline inherited from the runtime spec: an omission carries its reason and no substituted value.

Four consequences that must be designed for, not discovered:

1. **Ingest reaches the network.** This reverses the prior "no network access" constraint, deliberately. Query time still never does — the network is touched during ingest and nowhere else.
2. **Content leaves the machine.** Chart images from the newsletters, and figures from the owner's financial models, are sent to a third party. That is an accepted trade, made knowingly; it is recorded here so the decision is not rediscovered later as a surprise.
3. **Results are non-deterministic and the model will change under us.** So every vision result is **cached, keyed by image content hash plus model id plus prompt version**. A re-run costs nothing and does not drift; changing the model or the prompt correctly invalidates and re-fetches. The same three values enter the file-state hash, so switching models re-ingests rather than silently mixing outputs from two different models.
4. **Cost is small but not zero.** At Mistral Small 4 rates (~$0.15/M input tokens) an image is a fraction of a cent, so the 3,837-image archive is on the order of a few dollars once — cheap enough not to be a design factor, but it is per-image and it recurs for genuinely new files.

**What this buys over local OCR**: not just the text inside a chart but a description of it, so an unlabelled diagram is no longer invisible. That folds in what was previously a deferred "Option C" tier.

**The NPU is not involved here, and should not be.** It compiles at one static shape and earns its keep on embedding, which is bulk and repeated. Vision is a one-time ingest cost on someone else's hardware.

### Excel

`openpyxl` primary with a `pandas`/`calamine` fallback. Three decisions, taken 2026-09-07:

- **Chunk by labelled block, not by row count.** These are financial models. A fixed 256-row split would cut a model in half and orphan the row labels that make the numbers mean anything. A chunk is one labelled region carried with its period header row and its row labels intact. Detecting block boundaries — blank-row runs, merged cells, label columns, sub-total rows — is the substantive engineering problem in this format, not an incidental detail.
- **Values *and* formulas are both searchable.** `openpyxl` cannot return both in one load, so this costs a second pass. Formula text is carried as a **distinct chunk kind** rather than inlined with prose, because formula strings embed poorly and would otherwise crowd out the text a query was aiming at.
- **Embedded charts and images are extracted** and routed through the same vision path as PDF figures. Native Excel chart objects additionally carry their source cell ranges, which can be read directly rather than OCR'd — cheaper and exact where available.

### Incremental behaviour

A per-file content hash in a persisted file-state table classifies files as new, changed, unchanged or deleted; only the first two produce work. The hash must cover **file content plus every parameter that affects the output** — chunk budget, extractor version, and, for anything vision-derived, the **model id and prompt version** — so that turning the vision path on, or changing the model behind it, correctly invalidates prior work instead of silently mixing chunks produced by two different models.

Separately from the file-state table, **vision results are cached by image content hash + model id + prompt version.** This is what makes a hosted, non-deterministic, per-call-billed step behave like a pure function: a re-run pays nothing and produces the same text, while a genuine change re-fetches.

## Scope

- **In**: recursive walk of configurable roots with include/exclude rules; per-format extraction for Markdown, plain text, PDF and Excel; routed PDF handling (text layer vs vision path); the optional hosted vision call over raster pages, embedded images and standalone image files, with its result cache; Excel labelled-block chunking with values and formulas; extraction of embedded spreadsheet charts; Unicode and whitespace normalisation; token-aware chunking against the runtime's tokenizer; per-chunk metadata (source path, author from directory structure, title, sheet and cell range, page, chunk kind, and provenance for anything vision-derived); per-file content and parameter hashing; new/changed/deleted classification; per-file error isolation and reporting; graceful degradation to text-only when no API key is configured.
- **Out**: embedding (`npu-embedding-runtime`); storage of chunks or vectors (`vector-index`); deciding the token budget value; scraping or downloading; a local ONNX vision stack (evaluated and rejected, above).

## The network boundary — only vision leaves the machine

**One outbound call exists in this system, and it is image-to-text.** Nothing else may go to OpenRouter or to any other hosted service. In particular:

- **Embedding stays local, on the NPU.** `embeddinggemma-300m` runs on this machine. That is the entire point of the project, and a hosted embedding model would dissolve it.
- **Text extraction, chunking and hashing stay local.** They need no model and must never acquire a network dependency.
- **Query time is wholly offline.** The network is touched during ingest and at no other moment. A search must work with the network unplugged.

This is a boundary that erodes by convenience — once an API key is in the config, reaching for it again is easy, and a plausible-sounding "we are already calling out anyway" is exactly how a local-first system stops being one. Anything that would widen this line needs a decision recorded here, not an implementation detail.

## Boundary Candidates

- Filesystem discovery, routing and filtering
- Per-format text extraction (Markdown / PDF / plain text / Excel)
- The vision path: the hosted image-to-text call, its result cache, and its absent-key degradation
- Normalisation and cleaning
- Token-aware chunking and metadata attachment
- File-state tracking and change classification

## Out of Boundary

- The value of the chunk token budget — an input from `npu-embedding-runtime`.
- Which tokenizer is authoritative. This spec uses the model's tokenizer, supplied as a dependency, never an approximation of its own.
- Choosing or evaluating the vision model. It is a configuration string; comparing candidates is a separate exercise, not this spec's work.
- Any hosted call other than image-to-text. See the network boundary above — that line is a decision, not a default.
- Deduplication of near-identical content across files.
- Watch/daemon mode. Incremental re-run on demand is in scope; a long-running watcher is deferred.

## Upstream / Downstream

- **Upstream**: the archive on disk (read-only, externally maintained), plus the owner's Excel sheets from a root yet to be named. The tokenizer and the 512-token compiled length from `npu-embedding-runtime`. OpenRouter, for the optional vision call only.
- **Downstream**: `vector-index` consumes chunk records and metadata — note it must handle **several chunk kinds** now (prose, table, formula, figure text), not one. `search-cli` surfaces ingest progress, counts and skipped-file reports.

## Existing Spec Touchpoints

- **Extends**: none. Greenfield.
- **Adjacent**: `npu-embedding-runtime` (closed) — owns tokenization and the token budget. `vector-index` — its brief was corrected on 2026-09-07 from 1024 to **768** dimensions; it now also inherits multiple chunk kinds from this spec.

## Constraints

- Formats: `.md`, `.txt`, `.pdf`, `.xlsx`, and raster images. Images are **in** scope, reversing the prior decision.
- Source roots are configuration, defaulting to `D:\Source\substack_dl\archive`, never hardcoded, and more than one root must be expressible since the spreadsheets live elsewhere.
- Chunk token budget is a parameter, constrained by the NPU's compiled sequence length of 512.
- Windows paths, long paths and non-ASCII filenames must work — the archive contains all three.
- Must run with **no NPU**, and must run with **no network** — degrading to text-only, loudly and by report, when no API key is configured. The vision path is an enhancement, never a prerequisite.
- The OpenRouter key is configuration, read from the gitignored `.env` alongside the existing `HF_TOKEN`. It is never printed, never logged, and never committed — the runtime spec's credential discipline applies unchanged.
- The vision model is a configuration string, not a constant. Changing it must invalidate cached results and re-ingest, not silently mix outputs from two models.
- Incremental re-run must be substantially cheaper than a full rebuild. That is the feature's reason for existing, and it matters more now that a full run makes a paid API call per image.

## Known risk inherited from upstream

`npu-embedding-runtime`'s task **8.1 was descoped** when that spec closed. It would have verified that a consumer's token count and the service's truncation decision agree for the same text — and its own Observable named this spec as the consumer that would size its chunks against that contract. **The contract is therefore unverified.** Chunking to 512 tokens on the assumption that the runtime agrees is exactly the kind of untested join this project has been bitten by before; this spec should verify it rather than assume it.
