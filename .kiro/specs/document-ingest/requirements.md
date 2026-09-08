# Requirements Document

## Project Description (Input)

Walk configurable source roots and turn a heterogeneous archive into clean, chunked, metadata-tagged text records ready for embedding, with cheap incremental re-runs.

### Who has the problem

The owner of a local document archive who wants to search it semantically on their own machine. The archive is scraped newsletter content plus their own spreadsheets: business and quantitative writing where the substance sits in charts and financial models as much as in prose.

### Current situation

The material cannot be embedded as it stands. Measured 2026-09-07, the archive is **3,837 images (2,129 MB, 90% of bytes)**, **135 PDFs (237 MB)**, **1,205 Markdown files (~1.0M words)** and 3 plain-text files. The owner's **Excel financial models are a priority format and are not in the archive yet**, so they live under a root still to be named.

Only 9.8% of image references in the Markdown carry alt text (399 of 4,087), and what exists is thin slugs — `innovation-theory`, `Reddit ARPU` — not descriptions. Indexing images by surrounding text therefore covers a tenth of them, badly, and the charts are where the numbers are.

`npu-embedding-runtime` is complete and closed. It publishes the tokenizer and a compiled sequence length of **512 tokens**, which is the chunk budget, and its default model `embeddinggemma-300m` is **text-only** — it has no vision modality, so images must become text before anything can embed them.

### What should change

A single call walks configurable roots and returns clean, chunked, metadata-tagged records ready for embedding, re-runnable cheaply as the archive grows.

**Extraction is routed by what a file actually contains**, so the expensive path stays rare: Markdown and plain text directly; PDFs with a usable text layer via a local library; PDFs without one, standalone raster images, and charts embedded in spreadsheets via an **optional hosted vision call to OpenRouter**, whose model is a configuration string defaulting to a Mistral one.

**Excel is chunked by labelled block, not by row count.** The sheets are financial models — line items down the left, periods across the top, several labelled regions per sheet — so a fixed row split would cut a model in half and orphan the labels that give the numbers meaning. Values and formulas are both searchable, with formula text as a distinct chunk kind so it does not crowd prose. Embedded sheet charts route through the vision path.

**Only image-to-text leaves the machine.** Embedding stays local on the NPU, extraction, chunking and hashing stay local, and query time is wholly offline — a search must work with the network unplugged.

**The vision path is optional and never a prerequisite.** With no API key configured, affected files are reported by name and count as requiring vision, never silently dropped and never guessed at. Vision results are cached by image content hash plus model id plus prompt version, so a paid, non-deterministic step behaves like a pure function: a re-run costs nothing and does not drift, while changing the model correctly re-fetches.

**Incremental re-run is the feature's reason for existing.** A per-file hash over content plus every output-affecting parameter classifies files as new, changed, unchanged or deleted, and only the first two produce work. Per-file extraction failures are reported and skipped, never fatal to the run.

The module must run with no NPU and with no network. The source root defaults to `D:\Source\substack_dl\archive`, is never hardcoded, and more than one root must be expressible.

See `.kiro/specs/document-ingest/brief.md`, redesigned 2026-09-07, for the measured corpus profile, the rejected local-ONNX-vision alternative, the network-boundary rationale, and a risk inherited from upstream: `npu-embedding-runtime`'s task 8.1 was descoped, so the token-count contract this spec sizes its chunks against is currently unverified.

## Requirements
<!-- Will be generated in /kiro-spec-requirements phase -->
