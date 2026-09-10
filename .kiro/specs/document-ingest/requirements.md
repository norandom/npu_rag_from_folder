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

A single call walks configurable roots and returns clean, chunked, metadata-tagged records ready for embedding, re-runnable cheaply as the archive grows. Extraction is routed by what a file actually contains, so the expensive path stays rare. Spreadsheets are chunked by labelled block rather than by row count. Image-to-text is local OCR and never invents a number. Per-file failures are reported and skipped, never fatal.

See `.kiro/specs/document-ingest/brief.md`, amended 2026-09-10, for the measured corpus profile, the withdrawn hosted-vision decision, and the local det+rec OCR path.

## Introduction

This feature converts a heterogeneous local archive into embedding-ready chunk records. It discovers files under configurable roots, routes each to an extraction path chosen by what the file actually contains, cuts the result into chunks that fit a token budget it does not choose, attaches metadata sufficient for retrieval and provenance, and persists enough per-file state that a later run does work only for what changed.

Its defining constraint is locality. Embedding, extraction, OCR, chunking and hashing all happen on this machine. Ingest and a later search must both work with the network unplugged. Image-to-text is optional local OCR: with the models absent the pipeline is complete, and the inputs that would have needed it are reported rather than silently dropped. It transcribes visible text and does not invent numbers.

## Boundary Context

- **In scope**: discovery under configurable roots; content-based routing between extraction paths; extraction from Markdown, plain text, PDF and spreadsheet workbooks; optional local detection-and-recognition OCR together with its result cache, a second-engine vote on numeric tokens, and absent-model behaviour; labelled-block chunking of worksheets with values and formulas; native chart source ranges indexed without OCR; token-budget-aware chunking with heading-aware splitting; per-chunk metadata and provenance; per-file state, change classification and deletion handling; per-file failure isolation and an end-of-run report.

- **Out of scope**: producing embedding vectors; storing chunks or vectors; choosing the token budget value; deciding which tokenizer is authoritative; acquiring or scraping source documents; hosted image-to-text or any vision-language model; DeepDoc layout and table-structure models; compiling OCR onto the NPU.

- **Adjacent expectations**: `npu-embedding-runtime` (closed) supplies the tokenizer and the 512-token compiled length, and its default embedding model is text-only. Its task 8.1 — the check that a consumer's token count agrees with the runtime's own truncation decision — was descoped, so **that contract is unverified and this feature must verify it rather than assume it**. `vector-index` consumes the emitted records and must handle several chunk kinds rather than one. `search-cli` surfaces the run report. The source archive is maintained by a separate tool outside this feature's control and is read-only to it.

## Requirements

### Requirement 1: Source Discovery and Configuration

**Objective:** As the archive owner, I want the pipeline to find my documents wherever I keep them, so that content in more than one location is searchable without moving it.

#### Acceptance Criteria

1. The Document Ingest shall accept one or more source roots as configuration and shall contain no source root path as a fixed value.
2. When a run begins, the Document Ingest shall enumerate files recursively beneath every configured root.
3. Where include or exclude rules are configured, the Document Ingest shall apply them to enumerated files before attempting any extraction.
4. If a configured source root does not exist or cannot be read, then the Document Ingest shall report that root as unavailable and shall continue with the remaining roots.
5. The Document Ingest shall process file paths that exceed the platform's traditional path-length limit and file paths containing non-ASCII characters.
6. If two configured roots overlap such that a file is enumerated more than once, then the Document Ingest shall process that file once and shall emit one set of records for it.

### Requirement 2: Extraction Routing

**Objective:** As the archive owner, I want each file handled by the cheapest path that can read it, so that a costly path is used only when it is genuinely needed.

#### Acceptance Criteria

1. When a file is enumerated, the Document Ingest shall select an extraction path based on the file's type together with its inspected content, rather than on its file extension alone.
2. When a page of a paginated document yields extractable text at or above a configured minimum, the Document Ingest shall extract that page locally and shall not submit it for image-to-text conversion.
3. If a page of a paginated document yields extractable text below the configured minimum, then the Document Ingest shall route that page to the image-to-text path.
4. If an enumerated file is of a type the Document Ingest does not support, then it shall record the file as unsupported with its path and shall not fail the run.
5. The Document Ingest shall accept the minimum-extractable-text threshold as configuration.

### Requirement 3: Text-Bearing Document Extraction

**Objective:** As the archive owner, I want prose extracted cleanly and in reading order, so that retrieved passages are readable and their position in the source is recoverable.

#### Acceptance Criteria

1. When extracting a Markdown document, the Document Ingest shall remove front matter, embedded markup and image syntax from the emitted text while preserving heading structure and reading order.
2. When a Markdown document references an image, the Document Ingest shall retain that reference as an anchor which ties any text later derived from the image to the referencing position in the document.
3. When extracting from any text-bearing document, the Document Ingest shall normalise Unicode representation and collapse redundant whitespace without altering the words themselves.
4. When extracting a page of a paginated document locally, the Document Ingest shall preserve reading order within that page.
5. Where a document declares a title, the Document Ingest shall record that title on every chunk derived from that document.

### Requirement 4: Spreadsheet Extraction and Labelled Blocks

**Objective:** As the archive owner, I want my financial models chunked by their labelled sections rather than by row count, so that retrieved numbers arrive with the labels that give them meaning.

#### Acceptance Criteria

1. When extracting a worksheet, the Document Ingest shall treat a run of one or more entirely empty rows as the boundary between labelled blocks.
2. When emitting a labelled block, the Document Ingest shall include the block's label, the block's period header row, and the row label of every data row the block contains.
3. If a single labelled block exceeds the token budget, then the Document Ingest shall split it and shall repeat the block label and the period header row in every resulting chunk.
4. When a cell contains a formula, the Document Ingest shall emit that cell's computed value within the block chunk and shall emit the formula text as a chunk of a distinct kind.
5. If a cell's computed value is unavailable, then the Document Ingest shall record the value as unavailable with its reason and shall not substitute a value in its place.
6. Where a worksheet contains an embedded chart or image, the Document Ingest shall extract it and route it to the image-to-text path.
7. Where an embedded chart declares the cell ranges it draws from, the Document Ingest shall record those ranges and shall not require image-to-text conversion to obtain them.
8. If a worksheet is hidden, then the Document Ingest shall skip it and shall record it as skipped with its name.

### Requirement 5: Optional Local OCR

**Objective:** As the archive owner, I want visible text in charts and scanned pages to become searchable as it appeared, without a vision model inventing numbers, and without that capability being a prerequisite for running at all.

#### Acceptance Criteria

1. Where the configured detector, recognizer and digit-checker files are present, the Document Ingest shall convert each routed image to text by local detection and recognition and shall emit the result as a chunk of a distinct kind carrying a reference to the source image.
2. Where any of the detector, recognizer or digit-checker files is absent, the Document Ingest shall skip every input that requires OCR and shall report those inputs by path and by count, naming the missing capability.
3. The Document Ingest shall exclude from OCR any image whose dimensions fall below a configured minimum, and shall report the number of images so excluded.
4. When an OCR result is obtained, the Document Ingest shall retain it keyed by the image content, the primary recognizer identifier and the OCR pipeline version.
5. When a routed image's content, the configured recognizer identifier and the OCR pipeline version all match a retained result, the Document Ingest shall reuse that result and shall not run OCR again.
6. If OCR of an image fails, then the Document Ingest shall record that input as failed with the reason and shall continue the run.
7. The Document Ingest shall accept the detector, recognizer and digit-checker identities as configuration.
8. The Document Ingest shall mark every chunk derived from OCR as vision-derived, together with the recognizer identifier that produced it.
9. The Document Ingest shall emit a numeric token from OCR only when the primary recognizer and the digit checker agree on that token after normalisation, and shall omit a number that only one engine produced.
10. Where an image reference already carries chart source ranges, the Document Ingest shall emit those ranges as the figure text and shall not run OCR on that reference.

### Requirement 6: Chunking Against the Token Budget

**Objective:** As a downstream consumer of chunks, I want every chunk to fit the embedding model's input length, measured the same way the model measures it, so that nothing is silently truncated at embedding time.

#### Acceptance Criteria

1. The Document Ingest shall accept the chunk token budget as an input and shall not select its value.
2. The Document Ingest shall measure chunk length using the tokenizer published by the embedding runtime, and shall not use an approximation of its own.
3. The Document Ingest shall emit no chunk whose measured token length exceeds the token budget.
4. When emitting chunks from prose, the Document Ingest shall apply a configurable overlap between consecutive chunks.
5. When emitting a spreadsheet block, a table, a formula, or a vision-derived chunk, the Document Ingest shall apply no overlap.
6. Where a document has heading structure, the Document Ingest shall prefer a split at a heading boundary when doing so does not exceed the token budget.
7. When the Document Ingest measures a text as within the token budget, the embedding runtime shall accept that text without truncating it, and when the Document Ingest measures a text as over budget, the embedding runtime shall report it as truncated — for the same text and the same text kind.

### Requirement 7: Chunk Records and Provenance

**Objective:** As a downstream consumer, I want each chunk to carry enough metadata to be filtered, cited and traced back to its origin, so that a result can be explained and re-found.

#### Acceptance Criteria

1. The Document Ingest shall record on every chunk its source path, its chunk kind, and its position within the source document.
2. The Document Ingest shall derive an author for every chunk from the directory structure of its source root.
3. The Document Ingest shall record on every chunk a locator identifying where in its source the chunk originates — the page for a paginated document, and the workbook, sheet name and cell range for a worksheet.
4. The Document Ingest shall assign every chunk an identifier that remains stable when files unrelated to it change between runs.
5. When the same source content produces a chunk on two runs with all output-affecting parameters unchanged, the Document Ingest shall assign that chunk the same identifier on both runs.

### Requirement 8: Incremental Re-Run

**Objective:** As the archive owner, I want a re-run after adding a few files to cost almost nothing, so that keeping the index current is not a reason to stop using the tool.

#### Acceptance Criteria

1. The Document Ingest shall persist per-file state sufficient to classify a file on a later run as new, changed, unchanged or deleted.
2. The per-file state shall incorporate the file's content together with every parameter that affects the file's output, including the token budget, the overlap, the routing thresholds, and — for anything OCR-derived — the recognizer identifier and the OCR pipeline version.
3. When a run encounters a file classified as unchanged, the Document Ingest shall reuse the previously emitted records for that file and shall not extract it again.
4. When a file present in a previous run is absent from the current run, the Document Ingest shall report the records derived from it as removed.
5. When any output-affecting parameter differs from the value recorded in a file's persisted state, the Document Ingest shall classify that file as changed.
6. When a run is repeated with no file and no parameter changed, the Document Ingest shall extract no file, run no OCR, and report that no work was required.

### Requirement 9: Failure Isolation and Run Reporting

**Objective:** As the archive owner, I want one bad file to cost me that file and not the run, and I want to be told what did not make it in.

#### Acceptance Criteria

1. If extraction of a file fails for any reason, then the Document Ingest shall record the failure with the file's path and the reason, and shall continue processing the remaining files.
2. When a run completes, the Document Ingest shall produce a report stating the counts of files processed, unchanged, skipped and failed.
3. When a run completes, the Document Ingest shall list every skipped and every failed file by path together with the reason it was skipped or failed.
4. When an input is skipped because a capability is unavailable rather than because it is unwanted, the Document Ingest shall state which capability was missing.
5. The Document Ingest shall complete a run and produce its report even when every file fails.

### Requirement 10: Locality

**Objective:** As the archive owner, I want a local-first system to stay local, so that optional OCR does not open a network path and does not invent content.

#### Acceptance Criteria

1. The Document Ingest shall complete a run without an NPU present.
2. The Document Ingest shall complete a run with no network access available.
3. The Document Ingest shall issue no network request.
4. The Document Ingest shall not read a hosted-vision credential, and shall not write any secret to any log, report, emitted record or error message.
5. If the detector, recognizer or digit-checker files are absent, then the Document Ingest shall report that OCR is unavailable, naming the missing capability, and shall complete the run using the local extraction paths alone.
