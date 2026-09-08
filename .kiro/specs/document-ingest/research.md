# Research & Design Decisions

## Summary
- **Feature**: `document-ingest`
- **Discovery Scope**: New Feature (greenfield package) with one Complex Integration (a hosted vision API) and a hard contract dependency on the closed `npu-embedding-runtime` spec.
- **Key Findings**:
  - Every image-bearing input — Markdown image references, PDF pages without a text layer, standalone raster files, charts embedded in worksheets — reduces to one capability: *describe this image as text*. One seam, four producers.
  - `openpyxl` can read an existing chart's source cell ranges and a worksheet's anchored images, but only through private attributes (`ws._charts`, `ws._images`). It never evaluates formulas: `data_only=True` yields `None` for any formula cell in a file not last saved by Excel, which will be common.
  - The hosted call is non-deterministic even at `temperature=0`, per OpenRouter's own documentation. Reproducibility therefore comes from caching by input, never from re-running.
  - `httpx` (0.28.1) and `markdown-it-py` (4.2.0) are already installed transitively, so the HTTP client and the Markdown parser cost no new dependency — but both must be declared explicitly, since importing a transitive dependency directly breaks the day `huggingface-hub` drops it.

## Research Log

### Runtime contract this feature consumes
- **Context**: Requirements 6.1–6.3 and 6.7 bind chunking to the embedding runtime's own tokenizer and truncation decision; 5.1/7.1 bind vision-derived text to the runtime's `DocumentText` shape.
- **Sources Consulted**: `src/npu_rag/embedding/tokenize.py`, `service.py`, `types.py`, `models/acquire.py`; `tests/embedding/test_package_baseline.py`; `npu-embedding-runtime/tasks.md` Implementation Notes.
- **Findings**:
  - `ModelTokenizer` exposes `count_tokens(text: str | DocumentText, kind: TextKind) -> int`, `exceeds_limit(...) -> bool`, `render(...)`, `encode_documents(...) -> EncodedBatch` (with `truncated_indices`), and `max_input_tokens`. Counting is over the **rendered** text — the document template with its title slot included — so a chunk must be measured as `DocumentText(content, title)`, not as bare content, or the count is wrong for exactly the titled documents this corpus produces.
  - `DocumentText(content: str, title: str | None = None)`; a blank title normalises to `None`, and an absent title renders as the literal sentinel `MISSING_TITLE_SENTINEL`. `TextKind` has exactly `DOCUMENT` and `QUERY`.
  - `EmbeddingContract` carries `model_id`, `dimension`, `max_input_tokens` (the compiled 512, never the architectural limit) and `tokenizer_id`.
  - The credential pattern is `HfCredential`: the secret is reachable only via `reveal()`, renders as `REDACTED = "<redacted>"` everywhere else, and hub errors are re-raised `from None` so no chained traceback can carry it. `find_dotenv` walks upward from cwd; `parse_dotenv` is a plain parser. All are public.
  - The outer package guard forbids `npu_rag.embedding` importing any sibling; the reverse direction is unconstrained. `npu_rag.ingest` importing `npu_rag.embedding` is the intended dependency direction.
  - The runtime's task **8.1 was descoped**: the check that a consumer's token count agrees with the runtime's truncation decision was never run, and its Observable named this spec as the consumer.
- **Implications**: The chunker takes a `ModelTokenizer` as a dependency and measures with the title attached. Requirement 6.7 becomes a test that drives the real tokenizer both ways. The OpenRouter credential mirrors `HfCredential` exactly, so 10.4 holds by construction.

### Spreadsheet reading — `openpyxl`
- **Context**: Requirements 4.1–4.8: labelled blocks, values and formulas, hidden sheets, embedded charts and images, chart source ranges without OCR.
- **Sources Consulted**: openpyxl 3.1 documentation (images, worksheet, chart.series API); openpyxl-users threads on `data_only` and `_images`; a working `ws._charts` example.
- **Findings**:
  - `load_workbook(data_only=True)` returns the **cached** value Excel wrote; openpyxl never evaluates. A formula cell in a file saved by another tool, or by openpyxl itself, reads as `None`. Obtaining both the value and the formula text requires **two loads** of the workbook.
  - `ws.sheet_state` ∈ {`visible`, `hidden`, `veryHidden`} and `ws.merged_cells.ranges` are public and stable.
  - `ws._images` (anchored images) and `ws._charts` (parsed chart objects) work on load but are **private**. A chart's series expose `series.val.numRef.f` and `series.cat.numRef.f` as plain range strings. Images embedded *in cells* (a newer Excel feature) are not in `ws._images`.
  - Current stable: 3.1.5.
- **Implications**: A two-load extractor; `None` for a formula cell is requirement 4.5's "value unavailable" case and is reported, never substituted. The two private attributes are touched in exactly one adapter function with a committed fixture workbook as a smoke test, so an openpyxl minor release breaking them fails one test rather than silently dropping every chart. "Entirely empty row" must treat a merged region's non-anchor cells as belonging to their anchor, or a merged label spanning rows would read as a block boundary.

### Fallback spreadsheet reader — `python-calamine`
- **Context**: The brief carried RAGFlow's `pandas`/`calamine` fallback forward.
- **Sources Consulted**: PyPI (0.8.2, MIT, cp312 win_amd64 wheels present).
- **Findings**: Whether it exposes cached formula values or formula text is **unverified**; it almost certainly reads no images or charts.
- **Implications**: Dropped from the design. No requirement demands a second reader, the primary one covers every requirement, and an unverified dependency added for a hypothetical fallback is the speculative abstraction the synthesis rules say to remove. Recorded under Simplification below.

### PDF text layer and page rendering — `pypdfium2`
- **Context**: Requirements 2.2/2.3/2.5 (route by extractable text per page), 3.4 (reading order), 5.x (render textless pages for the vision path).
- **Sources Consulted**: pypdfium2 readthedocs (5.13.0, Apache-2.0 / BSD-3-Clause), two independent extractor benchmarks.
- **Findings**:
  - Text: `page.get_textpage().get_text_range()`; the character count of the stripped result is the cheapest "is there a text layer" probe. Reading order follows content-stream order — adequate for single-column prose, not guaranteed for multi-column layouts.
  - Rendering: `page.render(scale=...)` → bitmap with `.to_pil()`; `scale=1` is 72 DPI.
  - `page.get_objects()` enumerates page objects, including images.
  - Roughly 10–30× faster than `pdfplumber` for text; no layout analysis. `pdfplumber` (pdfminer.six) does layout/table analysis at that speed cost.
- **Implications**: One dependency covers probing, extraction and rasterisation. `pdfplumber` is rejected: the structure it would recover is the structure the vision path is for, and the lean-dependency constraint is explicit. Multi-column misordering is accepted and recorded as a known limitation; the vision route is the remedy if it proves material.

### Hosted vision — OpenRouter chat completions
- **Context**: Requirements 5.1–5.8 and 10.2–10.5.
- **Sources Consulted**: OpenRouter multimodal docs, API reference, limits page, parameters guide, Mistral provider page.
- **Findings**:
  - Request: `messages[].content` is an array of parts; `{"type":"text","text":...}` then `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`. Public HTTPS URLs also accepted. MIME: png, jpeg, webp, gif. Text part first.
  - Headers: `Authorization: Bearer <key>`; optional `HTTP-Referer`, `X-Title`.
  - 429 carries `X-RateLimit-*` headers and, when present, `Retry-After`, which the docs say to treat as binding.
  - **Determinism is not guaranteed at `temperature=0`.**
  - Documented maximum image size: **unverified** — not found; to be established empirically against real archive images.
  - Vision-capable Mistral ids and rough $/M input: `mistral-small-3.2-24b-instruct` ~0.075, `pixtral-12b` ~0.15, `mistral-small-2603` ~0.15, `mistral-small-3.1-24b-instruct` ~0.35, `mistral-medium-3-5` ~1.50.
- **Implications**: The model id is configuration with `mistralai/mistral-small-3.2-24b-instruct` as the default. Results are cached by `(sha256(image bytes), model_id, prompt_version)`; a re-run never re-calls. The client honours `Retry-After`, bounds concurrency, and never assumes identical output. Image-size limits are a follow-up to verify during implementation.

### Markdown structure — `markdown-it-py`
- **Context**: Requirements 3.1, 3.2, 3.5, 6.6: strip front matter, HTML and image syntax; keep headings and reading order; keep image references as position anchors.
- **Sources Consulted**: markdown-it-py 4.2.0 docs (token API), mistune 3.3.4.
- **Findings**: Flat token stream with paired open/close tokens; block tokens carry `token.map = [line_begin, line_end]`; inline children (including `image`) carry no map of their own. GFM tables are available via the `table` rule. Already installed transitively.
- **Implications**: Heading path and reading order come straight from the token stream; an image anchor is the enclosing block's line range plus the image's ordinal within it, which is sufficient for requirement 3.2. Tables become `TABLE` chunks, giving requirement 6.5's "table" kind a concrete producer. `mistune` and regex are rejected — regex cannot distinguish a heading from other `#`-prefixed text or handle nested HTML.

### HTTP client and image dimensions
- **Findings**: `httpx` 0.28.1 is installed (via `huggingface-hub`); it has timeouts, and `MockTransport` for tests without a network. `Pillow` is not installed; `pypdfium2`'s `.to_pil()` needs it, and reading an image's dimensions for the size threshold (5.3) is a header read via `Image.open`, which is lazy.
- **Implications**: Declare `httpx` explicitly; add `pillow`. Use `MockTransport` so the OpenRouter adapter is fully tested offline.

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Staged pipeline with per-format extractors behind one protocol | discover → route → extract → describe → chunk → persist → report; extractors emit a common segment list | Mirrors the runtime's ports-and-adapters style; each stage testable alone; extractors parallel-implementable | Orchestrator is the one place all stages meet and must own per-file isolation | **Selected** |
| Extractors call the vision service directly | Each extractor resolves its own images | Fewer types | Four copies of caching, thresholds and credential handling; network reachable from four modules, so requirement 10.3 is enforced in four places | Rejected |
| Streaming/event pipeline | Files flow as events through async stages | Natural concurrency for the vision calls | Async everywhere for a batch job that is CPU-bound except one stage; harder to keep per-file failure isolation legible | Rejected; concurrency is confined to the describer |
| Plugin registry for extractors | Discover extractors dynamically | Extensible | Nothing in scope needs a fifth format; a registry is indirection with one consumer | Rejected — Simplification |

## Design Decisions

### Decision: One `ImageRef` seam for every image producer
- **Context**: Markdown references, textless PDF pages, standalone raster files and worksheet charts all need image-to-text (5.1, 4.6, 2.3, 3.2).
- **Alternatives Considered**: 1. Each extractor calls the describer. 2. Extractors emit `ImageRef` segments; the pipeline resolves them through one describer before chunking.
- **Selected Approach**: 2. Extractors never touch the network or the cache. The pipeline collects `ImageRef`s, applies the size threshold, consults the cache, calls the describer for misses, and replaces each ref with a `FIGURE` segment (or an `Omission`).
- **Rationale**: Requirement 10.3 ("no request other than image-to-text") becomes checkable at a single module; 5.3–5.5 have one implementation; extractors stay pure and offline.
- **Trade-offs**: One more segment kind and a resolution pass. Accepted.
- **Follow-up**: The layer guard must forbid `extract/*` importing `vision`.

### Decision: Blank-row runs are the block boundary, with merged cells attributed to their anchor
- **Context**: Requirement 4.1, decided by the owner: a run of one or more entirely empty rows ends a block.
- **Alternatives Considered**: heading-row rule; both with precedence; heuristic detection with reporting.
- **Selected Approach**: A row is empty iff every cell's value is `None` **and** no merged range with a value covers it. Consecutive non-empty rows form a block; the first row carries the label, the first row containing period-like values is the header.
- **Rationale**: The owner's sheets follow this convention; it is the simplest rule that is testable, and merged-cell handling is the one place it would otherwise be wrong on real models.
- **Trade-offs**: Tightly packed sheets with no separators produce one block per sheet. Accepted; revisit only if real workbooks show it.
- **Follow-up**: Fixture workbook must contain a merged label spanning rows, so the guard is non-vacuous.

### Decision: Values and formulas via two workbook loads; `None` is an omission, never a substitute
- **Context**: Requirements 4.4, 4.5; openpyxl cannot return both in one load and never evaluates.
- **Selected Approach**: Load once with `data_only=True` for values and once without for formula text. A formula cell whose cached value is `None` is emitted as unavailable with the reason "no cached value; the workbook was not saved by an application that computes formulas", and the formula text still goes to its `FORMULA` chunk.
- **Rationale**: Requirement 4.5 and the runtime's inherited 6.8 discipline: state the omission, never guess.
- **Trade-offs**: Two parses per workbook. Workbooks are few and small relative to the 3,837 images; accepted.

### Decision: Cache by input, not by re-running
- **Context**: Requirements 5.4, 5.5, 8.6; the hosted model is non-deterministic and billed per call.
- **Selected Approach**: `vision_cache(image_sha256, model_id, prompt_version) → text`, persisted in the same SQLite store as file state. A hit never issues a request. The three keys also enter the file's parameter fingerprint, so a changed model or prompt reclassifies affected files as changed.
- **Rationale**: Makes a paid, drifting step behave as a pure function of its inputs; 8.6's zero-work re-run holds even for vision-derived content.
- **Trade-offs**: Cache grows with the archive (text only, a few KB per image). Accepted.

### Decision: SQLite via the standard library for state and cache
- **Alternatives Considered**: JSON file; a directory of content-addressed files; SQLite.
- **Selected Approach**: One SQLite file at a configurable path, tables `file_state`, `chunk_registry`, `vision_cache`.
- **Rationale**: Transactional per-file commits (a crashed run leaves prior rows intact — 9.x), queryable for deletion detection (8.4), no dependency.
- **Trade-offs**: A binary artefact rather than a diffable file. Accepted; it is derived state.

### Decision: Chunk identity is a function of the file's own content and parameters only
- **Context**: Requirements 7.4, 7.5.
- **Selected Approach**: `chunk_id = sha256(root_id, relative_path, kind, locator, ordinal_within_file, params_fingerprint)`. No run-global counter enters it.
- **Rationale**: Stable when unrelated files change (7.4); identical across runs when nothing relevant changed (7.5).

### Decision: Measure chunks with the title attached
- **Context**: Requirements 6.2, 6.3, 6.7; the runtime counts over the rendered template.
- **Selected Approach**: Every candidate chunk is measured as `DocumentText(content=chunk_text, title=document_title)` under `TextKind.DOCUMENT`.
- **Rationale**: Measuring bare content would under-count every titled document by the title's token cost and violate 6.3 at embedding time — precisely the untested join this project has been bitten by.

### Synthesis outcomes
- **Generalisation**: four image producers → one `ImageRef` seam; five chunk kinds → one `ChunkRecord` with a `kind` discriminator; seven ways a file can be skipped or fail → one `Omission` record shape carrying a category and a reason.
- **Build vs adopt**: adopt `markdown-it-py`, `pypdfium2`, `openpyxl`, `httpx`, `pillow`, stdlib `sqlite3`/`hashlib`; build only block detection (no library does it) and the orchestration.
- **Simplification**: no `calamine` fallback; no extractor registry; no separate normaliser component (a pure function in `chunk.py`); no daemon; concurrency confined to the describer.

## Risks & Mitigations
- **openpyxl private attributes change** — one adapter function, pinned version, committed fixture workbook with an image and a chart as a smoke test.
- **Formula cells read as `None` in most owner-authored workbooks** — first-class omission path with a specific reason; the report counts them so the owner sees it after the first run.
- **Multi-column PDFs extract in the wrong order** — accepted for text-layer pages; recorded as a limitation; the vision route is the remedy if material.
- **Unverified OpenRouter image-size limit** — verify against the largest archive images during implementation; downscale before sending if a limit is found.
- **Rate limiting during a 3,837-image first run** — bounded concurrency, `Retry-After` honoured, per-image failures recorded and the run continues (5.6, 9.1).
- **The 512-token contract is unverified upstream** — requirement 6.7 is a two-way test against the real tokenizer; it runs in the standing suite with the ungated control tokenizer and opt-in against the gated default.
- **Credential leakage via tracebacks or logs** — mirror `HfCredential`: `reveal()` at one call site, errors re-raised `from None`, no request logging.

## References
- [openpyxl images](https://openpyxl.readthedocs.io/en/3.1/images.html) — write path is documented; read path is private.
- [openpyxl chart series API](https://openpyxl.readthedocs.io/en/stable/api/openpyxl.chart.series.html) — `numRef.f` range strings.
- [openpyxl worksheet API](https://openpyxl.readthedocs.io/en/stable/api/openpyxl.worksheet.worksheet.html) — `sheet_state`, `merged_cells`.
- [pypdfium2](https://pypdfium2.readthedocs.io/en/stable/readme.html) — text, render, objects.
- [OpenRouter multimodal images](https://openrouter.ai/docs/docs/overview/multimodal/images) — request shape.
- [OpenRouter limits](https://openrouter.ai/docs/api_reference/limits) — 429 and `Retry-After`.
- [OpenRouter parameters](https://openrouter.ai/docs/api_reference/parameters) — determinism not guaranteed.
- [OpenRouter Mistral models](https://openrouter.ai/mistralai) — ids and pricing.
- [markdown-it-py token API](https://markdown-it-py.readthedocs.io/en/latest/api/markdown_it.token.html) — `token.map`.
- [python-calamine](https://pypi.org/project/python-calamine/) — evaluated, not adopted.
