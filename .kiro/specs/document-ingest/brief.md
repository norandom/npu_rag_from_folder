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
- The module needs **no NPU and no network**. Image-to-text is local OCR (detection plus recognition). It degrades to text-only when the OCR models are absent rather than fail, and it never invents a number.

## Approach

A pipeline of small, independently testable stages — discover, route, extract, normalize, chunk, hash — with each format behind a common extractor interface.

**Extraction is routed by what a file actually contains, not by its extension alone.** The expensive path must stay rare:

1. **Markdown / plain text** — strip HTML, front-matter and image syntax; preserve headings; keep image references as anchors so an extracted figure can be tied back to its position in the post.
2. **PDF with a usable text layer** — direct extraction. Covers most documents and is orders of magnitude cheaper than OCR. Scientific and financial charts generated as *vector* graphics carry real text objects, so axis labels, legends and titles come out here for free.
3. **PDF without a usable text layer, and raster images** — the vision path below.
4. **Excel** — spreadsheet library, no vision model for the cells themselves.

### The vision path — local det+rec OCR, optional (reversed 2026-09-10)

**A hosted vision-LLM call was the 2026-09-07 decision and is withdrawn.** OpenRouter-to-Mistral (or any chat VLM) can paraphrase a chart and invent numbers. For quantitative and engineering material that is a no-go: a retrieved "ARPU 12.4" that was never on the page is worse than a missing figure. The pipeline must transcribe visible text, never describe, never guess.

**Tesseract as it stood a decade ago is not the primary engine.** Tesseract 3 was a different system. Tesseract 5 LSTM is usable as an *independent second vote on digits*, but modern PaddleOCR-derived ONNX detectors and recognizers (the InfiniFlow DeepDoc `det.onnx` + `rec.onnx` pair RAGFlow actually runs, Apache-2.0, ~16 MB together) are the primary: they are built for scene and document text, run on ONNX Runtime CPU with no PaddlePaddle, and do not emit free prose.

**Layout and table-structure models (`layout.onnx`, `tsr.onnx`) stay out.** Those were the complexity that sank the first local-ONNX evaluation. This spec takes only detection and recognition. Native PDF text layers and Excel cell ranges already carry structure; OCR is for pixels that have no text layer.

Applies to three inputs, all producing **transcribed** text that the ordinary embedding path then consumes:

- raster image files referenced from Markdown posts
- PDF pages with no usable text layer
- images **embedded** in spreadsheets (anchored pictures)

**Native Excel chart objects are not OCR'd.** Task 4.6 already reads series source ranges off the chart object. Those ranges are exact. A raster of the chart is not available from openpyxl (the extractor emits a blank placeholder), so sending that raster to any vision model would be inventing a picture. The figure text for a native chart is the source ranges.

**It is optional and the pipeline is complete without it.** With the detector, recognizer, or digit-checker models absent, those inputs are **reported as requiring OCR and skipped by name and count** — never silently dropped, and never guessed at.

**Accuracy for numbers is a two-engine rule, not a confidence slider.** After the primary recognizer reads a box, numeric tokens (integers, decimals, percentages, currency amounts) are kept only when a second local engine — Tesseract 5 LSTM on the same crop — produces the same normalised digits. Disagreement drops the number and is reported; the rest of the box's non-numeric text may still be kept. A number that appears in only one engine is omitted, never promoted. Unlabelled diagrams with no readable text become omissions, not captions.

**Ingest does not reach the network.** Model files are local, like the tokenizer fixture. Query time was already offline; ingest is too. The NPU is not a prerequisite: OCR runs on CPU ONNX Runtime. Compiling `det` onto VitisAI is a later optional spike, not this spec's gate. Embedding keeps the NPU.

Cache key remains `image content hash + primary model id + pipeline version` (the pipeline version names det, rec, and the checker). Changing any of those invalidates the cache. A re-run with nothing changed does no OCR.

### Excel

`openpyxl` primary with a `pandas`/`calamine` fallback. Three decisions, taken 2026-09-07:

- **Chunk by labelled block, not by row count.** These are financial models. A fixed 256-row split would cut a model in half and orphan the row labels that make the numbers mean anything. A chunk is one labelled region carried with its period header row and its row labels intact. Detecting block boundaries — blank-row runs, merged cells, label columns, sub-total rows — is the substantive engineering problem in this format, not an incidental detail.
- **Values *and* formulas are both searchable.** `openpyxl` cannot return both in one load, so this costs a second pass. Formula text is carried as a **distinct chunk kind** rather than inlined with prose, because formula strings embed poorly and would otherwise crowd out the text a query was aiming at.
- **Embedded worksheet images are extracted** and routed through local OCR. Native Excel chart objects carry their source cell ranges and are **not** OCR'd — the ranges are exact and the raster is not available from openpyxl.

### Incremental behaviour

A per-file content hash in a persisted file-state table classifies files as new, changed, unchanged or deleted; only the first two produce work. The hash must cover **file content plus every parameter that affects the output** — chunk budget, extractor version, and, for anything vision-derived, the **model id and prompt version** — so that turning the vision path on, or changing the model behind it, correctly invalidates prior work instead of silently mixing chunks produced by two different models.

Separately from the file-state table, **OCR results are cached by image content hash + primary model id + pipeline version** (pipeline version names det, rec, and the digit checker). A re-run pays no OCR; a genuine model change re-runs.

## Scope

- **In**: recursive walk of configurable roots with include/exclude rules; per-format extraction for Markdown, plain text, PDF and Excel; routed PDF handling (text layer vs local OCR); optional local det+rec OCR over raster pages, embedded worksheet images and standalone image files, with its result cache and a second-engine vote on numeric tokens; Excel labelled-block chunking with values and formulas; native chart source ranges indexed without OCR; Unicode and whitespace normalisation; token-aware chunking against the runtime's tokenizer; per-chunk metadata; per-file content and parameter hashing; new/changed/deleted classification; per-file error isolation and reporting; graceful degradation to text-only when OCR models are absent.
- **Out**: embedding (`npu-embedding-runtime`); storage of chunks or vectors (`vector-index`); deciding the token budget value; scraping or downloading; hosted image-to-text (OpenRouter and every other VLM); DeepDoc layout and table-structure models; compiling OCR onto the NPU (optional later spike).

## The network boundary — ingest does not leave the machine

**No outbound call exists in this system.** The 2026-09-07 exception for image-to-text is closed. In particular:

- **Embedding stays local, on the NPU.** `embeddinggemma-300m` runs on this machine.
- **OCR stays local, on CPU ONNX Runtime.** Detector and recognizer files live on disk. Tesseract, if used as the digit checker, is a local binary.
- **Text extraction, chunking and hashing stay local.**
- **Query time is wholly offline.** A search must work with the network unplugged, and so must ingest.

Anything that would open an HTTP path needs a decision recorded here, not an implementation detail.

## Boundary Candidates

- Filesystem discovery, routing and filtering
- Per-format text extraction (Markdown / PDF / plain text / Excel)
- The vision path: local det+rec OCR, the second-engine vote on numeric tokens, its result cache, and absent-model degradation
- Normalisation and cleaning
- Token-aware chunking and metadata attachment
- File-state tracking and change classification

## Out of Boundary

- The value of the chunk token budget — an input from `npu-embedding-runtime`.
- Which tokenizer is authoritative. This spec uses the model's tokenizer, supplied as a dependency, never an approximation of its own.
- Compiling OCR onto the NPU. CPU ONNX is the contract; VitisAI offload of `det` is a later spike.
- Hosted image-to-text, figure captioning, and any VLM. See the network boundary above.
- Deduplication of near-identical content across files.
- Watch/daemon mode. Incremental re-run on demand is in scope; a long-running watcher is deferred.

## Upstream / Downstream

- **Upstream**: the archive on disk (read-only, externally maintained), plus the owner's Excel sheets from a root yet to be named. The tokenizer and the 512-token compiled length from `npu-embedding-runtime`. Local InfiniFlow DeepDoc `det`/`rec` ONNX files and a Tesseract 5 checker, for optional OCR only.
- **Downstream**: `vector-index` consumes chunk records and metadata — note it must handle **several chunk kinds** now (prose, table, formula, figure text), not one. `search-cli` surfaces ingest progress, counts and skipped-file reports.

## Existing Spec Touchpoints

- **Extends**: none. Greenfield.
- **Adjacent**: `npu-embedding-runtime` (closed) — owns tokenization and the token budget. `vector-index` — its brief was corrected on 2026-09-07 from 1024 to **768** dimensions; it now also inherits multiple chunk kinds from this spec.

## Constraints

- Formats: `.md`, `.txt`, `.pdf`, `.xlsx`, and raster images. Images are **in** scope, reversing the prior decision.
- Source roots are configuration, defaulting to `D:\Source\substack_dl\archive`, never hardcoded, and more than one root must be expressible since the spreadsheets live elsewhere.
- Chunk token budget is a parameter, constrained by the NPU's compiled sequence length of 512.
- Windows paths, long paths and non-ASCII filenames must work — the archive contains all three.
- Must run with **no NPU** and **no network**. OCR degrades to text-only, loudly and by report, when detector, recognizer or digit-checker files are absent. The vision path is an enhancement, never a prerequisite.
- No API key is required for OCR. `OpenRouterCredential` (task 1.5) is withdrawn and must not be consulted.
- Detector, recognizer and checker identities are configuration. Changing any of them must invalidate cached results and re-ingest.
- Incremental re-run must be substantially cheaper than a full rebuild. A full OCR pass is CPU time, not a billed API call, but it is still the expensive path.

## Known risk inherited from upstream

`npu-embedding-runtime`'s task **8.1 was descoped** when that spec closed. It would have verified that a consumer's token count and the service's truncation decision agree for the same text — and its own Observable named this spec as the consumer that would size its chunks against that contract. **The contract is therefore unverified.** Chunking to 512 tokens on the assumption that the runtime agrees is exactly the kind of untested join this project has been bitten by before; this spec should verify it rather than assume it.
