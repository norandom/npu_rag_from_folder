# Technical Design: document-ingest

## Overview

**Purpose**: This feature turns a heterogeneous local archive — Markdown, plain text, PDF, Excel financial models and several thousand chart images — into chunk records that fit the embedding runtime's 512-token input, carry enough provenance to be cited and re-found, and can be regenerated cheaply as the archive grows.

**Users**: The archive owner runs it on demand after new files arrive. `vector-index` consumes its records; `search-cli` surfaces its run report.

**Impact**: Greenfield. It adds a sibling package `npu_rag.ingest` beside the closed `npu_rag.embedding` runtime and consumes that runtime's tokenizer and text types as a library. It introduces the project's only outbound network call — image-to-text via OpenRouter — and confines it to one module.

### Goals
- One call walks configured roots and yields embedding-ready `ChunkRecord`s with a stable identity and a source locator on every record.
- Extraction is routed by inspected content so the paid, hosted path is used only for images and textless pages.
- Spreadsheets are chunked by labelled block with their labels and period headers intact; values and formulas are both searchable.
- A re-run with nothing changed extracts nothing, calls nothing and says so.
- The pipeline is complete with no NPU, no network and no credential; what it cannot do is reported by path and reason.

### Non-Goals
- Producing vectors, storing chunks or vectors, choosing the token budget, or deciding which tokenizer is authoritative — all owned upstream or downstream.
- A local OCR or layout model stack (evaluated and rejected; see `research.md`).
- Any hosted call other than image-to-text. A daemon or file watcher. Near-duplicate detection across files.
- Reading-order recovery for multi-column PDF text layers; the vision route is the remedy if it proves material.

## Boundary Commitments

### This Spec Owns
- Discovery under configurable roots, include/exclude filtering, and de-duplication across overlapping roots.
- Routing each file to an extraction path by type **and** inspected content, including the per-page text-layer decision inside PDFs.
- Extraction for Markdown, plain text, PDF, Excel workbooks and standalone raster images, each behind the one `Extractor` protocol, each emitting the one `Segment` vocabulary.
- The image-to-text seam: size threshold, result cache, the OpenRouter adapter, absent-credential degradation, and provenance marking of everything vision-derived.
- Token-budget chunking against the runtime's tokenizer, the per-kind overlap policy, heading-aware splitting, and block splitting with repeated headers.
- Chunk identity and per-file state: content and parameter fingerprints, new/changed/unchanged/deleted classification, the chunk registry that makes deletions reportable, and the vision cache.
- Per-file failure isolation and the run report.
- The `IngestConfig` contract and the `ChunkRecord` / `RunReport` output contracts that `vector-index` and `search-cli` build against.

### Out of Boundary
- Embedding, vector storage, retrieval — `npu-embedding-runtime`, `vector-index`.
- The value of the token budget and the tokenizer itself — supplied by the runtime via `ModelTokenizer`; never re-implemented or approximated here.
- Acquiring, scraping or modifying source documents; the archive is read-only to this feature.
- Choosing or benchmarking vision models; the model id is configuration.
- Removing records from any downstream store on deletion. This feature **reports** removed chunk ids (8.4); `vector-index` acts on them.
- Any second spreadsheet reader, any OCR model, any per-figure captioning beyond the one describer call.

### Allowed Dependencies
- `npu_rag.embedding` — `ModelTokenizer`, `DocumentText`, `TextKind`, `MISSING_TITLE_SENTINEL`, `find_dotenv`, `parse_dotenv`, `REDACTED`. This is the intended direction; the runtime's own guard forbids the reverse and stays untouched.
- Standard library: `sqlite3`, `hashlib`, `pathlib`, `json`, `base64`, `concurrent.futures`.
- Third-party, declared in `[project]`: `markdown-it-py` (already transitively installed), `httpx` (already transitively installed), `openpyxl`, `pypdfium2`, `pillow`.
- Network: **only** `vision.py` may open a connection, and only to the configured OpenRouter base URL. No other module imports `httpx`.
- Constraint: modules import only leftward along the dependency direction below; `extract/*` may not import `vision`; nothing imports `pipeline` except the caller.

### Revalidation Triggers
- `ChunkRecord` or `ChunkKind` changes shape or gains a member → `vector-index` must re-check its schema and its per-kind handling.
- `RunReport` or `Omission` changes shape → `search-cli` must re-check its rendering.
- The runtime changes `ModelTokenizer.count_tokens`, the document template, or the compiled length → requirement 6.7's contract test must be re-run and `params_fingerprint` inputs re-checked.
- The vision prompt text or `PROMPT_VERSION` changes → every cached result is invalidated by design; the owner must expect a paid re-run over all images.
- The default model id changes → same as above.
- A sixth chunk kind or a fifth extractor is added → the layer guard's `LAYER_ORDER` and the no-overlap policy table must be updated together.

## Architecture

### Architecture Pattern & Boundary Map

A staged pipeline in the runtime's ports-and-adapters style. Extractors are adapters behind one protocol and never touch the network; the pipeline is the single place every stage meets and the single place per-file isolation is enforced.

```mermaid
graph TB
    subgraph Inputs
        Roots[Configured roots]
        Env[dotenv credential]
    end
    subgraph Ingest
        Discover[Discoverer]
        Route[Router]
        Extract[Extractors]
        Resolve[ImageRef resolution]
        Describe[OpenRouterDescriber]
        Chunk[Chunker]
        State[StateStore]
        Pipeline[IngestPipeline]
        Report[RunReport]
    end
    subgraph Runtime
        Tok[ModelTokenizer]
    end
    subgraph Downstream
        VI[vector index]
        CLI[search cli]
    end
    OR[OpenRouter]

    Roots --> Discover --> Route --> Extract --> Resolve --> Chunk --> Pipeline
    Env --> Describe
    Resolve --> State
    Resolve --> Describe --> OR
    Chunk --> Tok
    Pipeline --> State
    Pipeline --> Report
    Report --> CLI
    Pipeline --> VI
```

**Architecture Integration**:
- Selected pattern: staged pipeline with per-format adapters; one `ImageRef` seam for all four image producers.
- Domain boundaries: extraction is pure and offline; resolution owns cost, cache and credential; chunking owns the budget; state owns persistence; the pipeline owns sequencing and isolation.
- Existing patterns preserved: redacting credential object with a single `reveal()`; errors carrying a `stage`; omissions carrying a reason and never a substituted value; a package-wide layer guard.
- Steering compliance: uv-managed Python 3.12, lean default dependencies, Windows-first paths, no NPU or network required to run.

**Dependency direction** (each module imports only from modules to its left; enforced by a guard test):

```
types, errors → config → credential → identity → state → discover → route → extract → vision → chunk, report → pipeline
```

### Technology Stack

| Layer | Choice / Version | Role in Feature | Notes |
|-------|------------------|-----------------|-------|
| Language / runtime | Python 3.12, uv | as the rest of the project | no conda, PowerShell tooling |
| Markdown | `markdown-it-py` 4.2 | token stream with line provenance; headings, tables, image refs | already installed transitively; declare explicitly |
| PDF | `pypdfium2` 5.13 | per-page text probe and extraction; page rasterisation for the vision path | one dependency for both jobs; `pdfplumber` rejected |
| Spreadsheets | `openpyxl` 3.1.5 | cells, merged ranges, hidden sheets, formulas; anchored images and chart source ranges via one adapter over private attributes | two loads per workbook; `calamine` dropped |
| Images | `pillow` | dimensions for the size threshold; encoding for the request payload | new dependency |
| HTTP | `httpx` 0.28 | the single outbound call; `MockTransport` for offline tests | already installed transitively; declare explicitly |
| State / cache | stdlib `sqlite3` | file state, chunk registry, vision cache in one file | transactional per file; no dependency |
| Tokenizer | `npu_rag.embedding.tokenize.ModelTokenizer` | every chunk measurement | never approximated |

## File Structure Plan

### Directory Structure
```
src/npu_rag/ingest/
├── __init__.py            # Public surface: run_ingest, IngestConfig, ChunkRecord, RunReport
├── types.py               # SourceFile, Locator kinds, Segment kinds incl. ImageRef, ChunkKind, ChunkRecord, Omission, RunReport
├── errors.py              # IngestError taxonomy with stage and path: DiscoveryError, ExtractionError, VisionError, StateError
├── config.py              # IngestConfig: roots, include/exclude, token budget, overlap, thresholds, model id, prompt version, state path
├── credential.py          # OpenRouterCredential mirroring HfCredential; discover_openrouter_credential via the runtime's dotenv helpers
├── identity.py            # params_fingerprint, file content hash, chunk_id derivation
├── state.py               # StateStore over sqlite3: file_state, chunk_registry, vision_cache; classification queries
├── discover.py            # Discoverer: walk roots with long-path support, include/exclude, de-dup across overlapping roots, author from directory
├── route.py               # SourceFile -> ExtractionPath by extension plus content sniff; unsupported types recorded
├── extract/
│   ├── __init__.py
│   ├── base.py            # Extractor protocol, Extracted result, shared normalisation helpers
│   ├── markdown.py        # markdown-it-py: front matter, HTML, headings, tables, ImageRef anchors
│   ├── text.py            # plain text
│   ├── pdf.py             # pypdfium2: per-page text or ImageRef of the rendered page; page locators
│   ├── excel.py           # openpyxl: blank-row blocks, merged-cell attribution, values+formulas, hidden sheets, charts and images adapter
│   └── image.py           # standalone raster file -> one ImageRef with dimensions
├── vision.py              # VisionDescriber protocol; OpenRouterDescriber over httpx; size threshold; cache-through via StateStore; PROMPT_VERSION
├── chunk.py               # Chunker over ModelTokenizer: budget, per-kind overlap, heading-aware split, block split with repeated header
├── report.py              # RunReport counting and human-readable rendering
└── pipeline.py            # IngestPipeline: discover -> route -> extract -> resolve ImageRefs -> chunk -> persist -> report; per-file isolation

tests/ingest/
├── test_package_baseline.py   # layer guard for npu_rag.ingest; extract/* may not import vision; only vision imports httpx
├── test_types.py, test_errors.py, test_config.py, test_credential.py, test_identity.py, test_state.py, test_route.py
├── extract/test_markdown.py, test_text.py, test_pdf.py, test_excel.py, test_image.py
├── test_vision.py             # fake describer; OpenRouterDescriber over httpx MockTransport; 429 and Retry-After; redaction
├── test_chunk.py              # budget, overlap policy, heading split, block split; measured with the title attached
├── test_pipeline.py           # per-file isolation, no-op re-run, deletion reporting, absent credential
├── test_token_contract.py     # requirement 6.7 both ways against the real runtime tokenizer
└── fixtures/
    ├── markdown/              # front matter, HTML, tables, images with and without alt text
    ├── pdf/                   # a text-layer page and a textless page
    ├── excel/                 # models.xlsx: blank-row blocks, a merged label spanning rows, formulas with and without cached values, a hidden sheet, one chart, one image
    └── images/                # a chart above the threshold and an icon below it
```

### Modified Files
- `pyproject.toml` — add `openpyxl`, `pypdfium2`, `pillow` to `[project] dependencies`; declare `httpx` and `markdown-it-py` explicitly there (currently transitive only). No new group; nothing here is heavyweight.
- `.gitignore` — the default state path `.npu_rag/ingest.sqlite`.
- `tests/embedding/test_package_baseline.py` — **unchanged**; it forbids `embedding` importing siblings, which this design respects.

## System Flows

One file through the pipeline, showing where the paid path is gated:

```mermaid
sequenceDiagram
    participant P as IngestPipeline
    participant S as StateStore
    participant R as Router
    participant X as Extractor
    participant V as ImageRef resolution
    participant D as OpenRouterDescriber
    participant C as Chunker
    participant T as ModelTokenizer

    P->>S: classify(path, content_hash, params_fingerprint)
    S-->>P: new or changed or unchanged
    alt unchanged
        P->>S: reuse prior chunk ids
    else new or changed
        P->>R: route(source_file)
        R-->>P: extraction path or unsupported
        P->>X: extract(source_file)
        X-->>P: segments incl ImageRefs
        loop each ImageRef
            P->>V: resolve(image_ref)
            V->>S: cache lookup by image hash, model id, prompt version
            alt hit
                S-->>V: text
            else miss and credential present and above threshold
                V->>D: describe(image)
                D-->>V: text or failure
                V->>S: cache store
            else miss and no credential or below threshold
                V-->>P: Omission with reason
            end
        end
        P->>C: chunk(segments, title)
        C->>T: count_tokens(DocumentText, DOCUMENT)
        C-->>P: ChunkRecords
        P->>S: commit file_state and chunk_registry
    end
```

Flow-level decisions: classification happens before any extraction, so an unchanged file costs one hash and one query. Image resolution is the only stage that can spend money and it is gated three ways — cache, credential, threshold — before a request is made. The per-file `commit` is atomic, so a crash mid-run leaves every completed file consistent and every incomplete file classified as new on the next run.

File classification is a small state machine:

```mermaid
stateDiagram-v2
    [*] --> Unseen
    Unseen --> New: first observed
    New --> Unchanged: content and params match on later run
    Unchanged --> Changed: content or params differ
    Changed --> Unchanged: re-extracted and committed
    Unchanged --> Deleted: absent from this run
    Changed --> Deleted: absent from this run
    Deleted --> New: reappears
```

## Requirements Traceability

| Requirement | Summary | Components | Interfaces | Flows |
|-------------|---------|------------|------------|-------|
| 1.1, 1.2, 1.3 | roots as config, recursive walk, include/exclude | `IngestConfig`, `Discoverer` | `Discoverer.discover` | — |
| 1.4 | unreadable root reported, run continues | `Discoverer`, `RunReport` | `Omission(category=ROOT_UNAVAILABLE)` | — |
| 1.5 | long and non-ASCII paths | `Discoverer`, `SourceFile` | `\\?\` prefixed absolute paths on Windows | — |
| 1.6 | overlapping roots de-duplicated | `Discoverer` | de-dup by resolved absolute path | — |
| 2.1 | route by type and inspected content | `Router` | `Router.route` | sequence |
| 2.2, 2.3, 2.5 | per-page text-layer threshold | `PdfExtractor`, `IngestConfig.min_page_chars` | emits `ProseSegment` or `ImageRef` per page | sequence |
| 2.4 | unsupported type recorded | `Router`, `RunReport` | `Omission(category=UNSUPPORTED)` | — |
| 3.1, 3.5 | Markdown cleaned, headings kept, title recorded | `MarkdownExtractor` | `Extracted.title`, `ProseSegment.heading_path` | — |
| 3.2 | image reference kept as anchor | `MarkdownExtractor` | `ImageRef.locator = MarkdownLocator(line_range, ordinal)` | — |
| 3.3 | Unicode and whitespace normalisation | `extract/base.normalise` | pure function | — |
| 3.4 | reading order within a page | `PdfExtractor` | content-stream order; limitation recorded | — |
| 4.1 | blank-row runs bound blocks | `ExcelExtractor` | `segment_blocks` | — |
| 4.2, 4.3 | label and header carried; oversize block split with header repeated | `ExcelExtractor`, `Chunker` | `BlockSegment`; `Chunker.split_block` | — |
| 4.4, 4.5 | value in block, formula as its own kind; unavailable value reported | `ExcelExtractor` | two loads; `FormulaSegment`; `Omission(category=VALUE_UNAVAILABLE)` | — |
| 4.6 | embedded charts and images routed to vision | `ExcelExtractor` | `ImageRef` from the anchored-object adapter | sequence |
| 4.7 | chart source ranges read directly | `ExcelExtractor` | `ChartRanges` on the `ImageRef` | — |
| 4.8 | hidden sheets skipped and named | `ExcelExtractor` | `Omission(category=HIDDEN_SHEET)` | — |
| 5.1 | vision result emitted as a distinct kind with its image reference | ImageRef resolution, `Chunker` | `FigureSegment` → `ChunkKind.FIGURE` | sequence |
| 5.2 | no credential: skip and report by path and count | ImageRef resolution, `RunReport` | `Omission(category=VISION_UNAVAILABLE)` | sequence |
| 5.3 | size threshold, count reported | ImageRef resolution, `IngestConfig.min_image_pixels` | `Omission(category=BELOW_THRESHOLD)`, counted | sequence |
| 5.4, 5.5 | cache by image, model, prompt version; hit issues no request | `StateStore.vision_cache`, ImageRef resolution | `StateStore.cached_description` | sequence |
| 5.6 | request failure recorded, run continues | `OpenRouterDescriber` | `VisionError` → `Omission(category=VISION_FAILED)` | sequence |
| 5.7 | model id is configuration | `IngestConfig.vision_model` | — | — |
| 5.8 | vision-derived chunks marked with model id | `ChunkRecord.provenance` | `Provenance(vision_model, prompt_version)` | — |
| 6.1, 6.2, 6.3 | budget is input; runtime tokenizer; never over budget | `Chunker` | `Chunker.__init__(tokenizer, budget)`; `count_tokens(DocumentText, DOCUMENT)` | sequence |
| 6.4, 6.5 | overlap on prose only | `Chunker` | `OVERLAP_POLICY: dict[ChunkKind, bool]` | — |
| 6.6 | heading-aware split | `Chunker` | splits at `heading_path` change first | — |
| 6.7 | two-way agreement with the runtime's truncation | `test_token_contract.py` | `count_tokens` vs `encode_documents().truncated_indices` | — |
| 7.1, 7.2, 7.3 | path, kind, position, author, locator on every chunk | `ChunkRecord`, `Discoverer` (author) | `Locator` union | — |
| 7.4, 7.5 | stable identity | `identity.chunk_id` | hash of file-local inputs only | — |
| 8.1, 8.2 | persisted state over content plus all output-affecting parameters | `StateStore`, `identity.params_fingerprint` | `file_state` table | state |
| 8.3 | unchanged file reused, not re-extracted | `IngestPipeline`, `StateStore` | `classify` before extract | sequence |
| 8.4 | deleted files' records reported as removed | `StateStore.chunk_registry`, `RunReport` | `RunReport.removed_chunk_ids` | state |
| 8.5 | parameter change reclassifies as changed | `identity.params_fingerprint` | fingerprint compared in `classify` | state |
| 8.6 | no-op re-run: nothing extracted, nothing called, reported | `IngestPipeline`, `RunReport` | `RunReport.no_work_required` | — |
| 9.1 | per-file failure isolated | `IngestPipeline` | catches `IngestError` and `Exception` per file | sequence |
| 9.2, 9.3, 9.4 | counts; skipped and failed listed with reason; missing capability named | `RunReport`, `report.render` | `Omission.reason`, `Omission.missing_capability` | — |
| 9.5 | run completes when every file fails | `IngestPipeline` | report always produced | — |
| 10.1 | no NPU required | whole package | no import of providers or sessions | — |
| 10.2, 10.3 | no network without credential; only image-to-text | `vision.py` sole importer of `httpx`; layer guard | `test_package_baseline` | — |
| 10.4 | credential never written out | `OpenRouterCredential` | `reveal()` at one call site; `from None` re-raise | — |
| 10.5 | absent or rejected credential degrades to local paths | ImageRef resolution, `OpenRouterDescriber` | 401/403 → `VISION_UNAVAILABLE` for the run | sequence |

## Components and Interfaces

| Component | Domain/Layer | Intent | Req Coverage | Key Dependencies (P0/P1) | Contracts |
|-----------|--------------|--------|--------------|--------------------------|-----------|
| `IngestConfig` | config | every tunable in one frozen object | 1.1, 1.3, 2.5, 5.3, 5.7, 6.1, 6.4 | — | State |
| `OpenRouterCredential` | credential | a secret that cannot render itself | 10.4, 10.5 | runtime dotenv helpers (P0) | Service |
| `Discoverer` | discovery | roots → `SourceFile`s, de-duplicated, author derived | 1.1–1.6, 7.2 | `IngestConfig` (P0) | Service |
| `Router` | routing | `SourceFile` → extraction path | 2.1, 2.4 | — | Service |
| `Extractor` protocol + 5 adapters | extraction | file → `Extracted` segments, offline | 2.2, 2.3, 3.x, 4.x | `pypdfium2`, `openpyxl`, `markdown-it-py` (P0) | Service |
| ImageRef resolution | vision seam | gate, cache, describe, mark | 5.1–5.6, 5.8, 10.5 | `StateStore`, `VisionDescriber` (P0) | Service |
| `OpenRouterDescriber` | vision adapter | the one outbound call | 5.6, 10.2, 10.3, 10.4 | `httpx` (P0), `OpenRouterCredential` (P0) | API |
| `Chunker` | chunking | segments → `ChunkRecord`s within budget | 4.2, 4.3, 6.1–6.6 | `ModelTokenizer` (P0) | Service |
| `identity` | identity | fingerprints and chunk ids | 7.4, 7.5, 8.2, 8.5 | — | Service |
| `StateStore` | persistence | file state, chunk registry, vision cache | 5.4, 5.5, 8.1–8.5 | `sqlite3` (P0) | State |
| `IngestPipeline` | orchestration | sequencing, isolation, report | 8.3, 8.6, 9.1–9.5 | all of the above (P0) | Batch |
| `RunReport` / `report` | reporting | counts, omissions, removed ids | 1.4, 2.4, 5.2, 5.3, 9.2–9.4 | — | State |

### Extraction

#### `Extractor` protocol and the five adapters

| Field | Detail |
|-------|--------|
| Intent | Turn one `SourceFile` into an ordered list of `Segment`s without touching the network or the state store |
| Requirements | 2.2, 2.3, 3.1–3.5, 4.1–4.8 |

**Responsibilities & Constraints**
- Pure with respect to the outside world: reads the file, returns segments, raises `ExtractionError` with a `stage` and the path on failure. Never imports `vision`, `state` or `httpx` (guarded).
- Emits `ImageRef` for anything that needs image-to-text; never decides whether that will happen.
- `PdfExtractor` applies `min_page_chars` per page: at or above → `ProseSegment(page=n)`; below → `ImageRef(locator=PageLocator(n), bytes=rendered PNG)`.
- `ExcelExtractor` loads twice (values, then formulas); segments blocks by blank-row runs with merged-cell attribution; emits `BlockSegment`, `FormulaSegment`, `ImageRef` for anchored images and charts, and `Omission`s for hidden sheets and unavailable values. All access to `ws._images` / `ws._charts` lives in one function, `anchored_objects(ws)`, with a committed fixture as its smoke test.
- `MarkdownExtractor` uses the token stream: front matter and HTML blocks dropped; headings maintain a `heading_path`; GFM tables become `TableSegment`; image tokens become `ImageRef(locator=MarkdownLocator(line_range, ordinal))` and are resolved from the file's directory.

**Dependencies**
- Inbound: `IngestPipeline` — calls `extract` (P0)
- Outbound: `extract/base.normalise` (P0); `IngestConfig` thresholds (P0)
- External: `pypdfium2`, `openpyxl`, `markdown-it-py`, `pillow` (P0)

**Contracts**: Service [x]

##### Service Interface
```python
class Extractor(Protocol):
    def extract(self, source: SourceFile, config: IngestConfig) -> Extracted: ...

@dataclass(frozen=True)
class Extracted:
    title: str | None
    segments: tuple[Segment, ...]      # in reading order
    omissions: tuple[Omission, ...]    # hidden sheets, unavailable values

Segment = ProseSegment | TableSegment | BlockSegment | FormulaSegment | ImageRef
```
- Preconditions: `source.path` exists and `Router` selected this extractor.
- Postconditions: segments are in reading order; every segment carries a `Locator`; no segment text contains front matter, HTML, or image syntax.
- Invariants: no network; no state; an `ImageRef` carries raw bytes, MIME, width and height, so resolution needs nothing from the extractor afterwards.

**Implementation Notes**
- Integration: `BlockSegment` carries `label`, `header_row: tuple[str, ...]`, `rows: tuple[tuple[str, ...], ...]` with the row label as column 0, so the chunker can split and re-head it without re-parsing.
- Validation: the Excel fixture must contain a merged label spanning rows, so a naive "all cells `None`" rule is proven wrong by a test; the PDF fixture needs one textless page so the threshold branch is reachable.
- Risks: openpyxl private attributes; multi-column PDF ordering (accepted, recorded).

### Vision seam

#### ImageRef resolution and `OpenRouterDescriber`

| Field | Detail |
|-------|--------|
| Intent | Replace each `ImageRef` with a `FigureSegment` or an `Omission`, spending money only after cache, credential and threshold have all said yes |
| Requirements | 5.1–5.8, 10.2–10.5 |

**Responsibilities & Constraints**
- Resolution order per `ImageRef`: threshold (5.3) → cache (5.5) → credential (5.2) → describe (5.1) → cache store (5.4). Each refusal is an `Omission` with its own category, so the report can say which gate closed.
- `OpenRouterDescriber` is the **only** module that imports `httpx` and the only module that may open a connection (10.3, guarded). Request: `POST {base_url}/chat/completions` with a text part (the versioned prompt) then one `image_url` part carrying a base64 data URL; `temperature=0` for stability, without assuming determinism.
- Honours `Retry-After` on 429; bounded concurrency (`vision_concurrency`, default 4); per-image timeout. 401/403 marks vision unavailable **for the rest of the run** rather than retrying every image (10.5). Any other failure is a per-image `VisionError` (5.6).
- The credential is revealed at exactly one line, inside the request construction; `httpx` exceptions are re-raised as `VisionError` **`from None`** so no chained traceback carries headers (10.4).

**Dependencies**
- Inbound: `IngestPipeline` (P0)
- Outbound: `StateStore.vision_cache` (P0), `OpenRouterCredential` (P0), `IngestConfig` (P0)
- External: OpenRouter chat-completions API (P1 — optional, degradable)

**Contracts**: Service [x] / API [x]

##### Service Interface
```python
class VisionDescriber(Protocol):
    def describe(self, image: ImageRef) -> str: ...     # raises VisionError; raises VisionUnavailable on 401/403

@dataclass(frozen=True)
class Resolved:
    figure: FigureSegment | None
    omission: Omission | None                            # exactly one of the two is set

def resolve(ref: ImageRef, *, store: StateStore, describer: VisionDescriber | None,
            config: IngestConfig) -> Resolved: ...
```
- Preconditions: `ref.width`/`ref.height` known.
- Postconditions: on success the returned `FigureSegment.provenance` names the model id and `PROMPT_VERSION`; on any refusal `omission.category` is one of `BELOW_THRESHOLD`, `VISION_UNAVAILABLE`, `VISION_FAILED`.
- Invariants: a cache hit never constructs an HTTP client; `describer is None` never raises.

##### API Contract
| Method | Endpoint | Request | Response | Errors |
|--------|----------|---------|----------|--------|
| POST | `{base_url}/chat/completions` | `{model, temperature: 0, messages: [{role: user, content: [{type: text, text: PROMPT}, {type: image_url, image_url: {url: data URL}}]}]}` | `choices[0].message.content` | 401/403 → unavailable for run; 429 → wait `Retry-After` then retry once; 4xx/5xx/timeout → `VisionError` |

**Implementation Notes**
- Integration: `PROMPT` asks for, in order, a one-paragraph description, a verbatim transcription of all visible text, and any table rendered as Markdown. Its text is a module constant paired with `PROMPT_VERSION`; changing one without the other is a review failure.
- Validation: `MockTransport` tests for the payload shape, the `Retry-After` path, the 401 run-wide latch, and an assertion that the rendered request's headers never reach an exception message.
- Risks: unverified image-size limit — measure the archive's largest images during implementation; downscale before sending if a limit is found. Cost is per image and recurs only for new images.

### Chunking

#### `Chunker`

| Field | Detail |
|-------|--------|
| Intent | Turn segments into `ChunkRecord`s that the runtime will accept unmodified |
| Requirements | 4.2, 4.3, 6.1–6.6 |

**Responsibilities & Constraints**
- Measures **every** candidate as `tokenizer.count_tokens(DocumentText(content=text, title=title), TextKind.DOCUMENT)` — with the document title attached, because the runtime counts over the rendered template and a titled document costs more than its content.
- Prose: accumulate paragraphs up to the budget, splitting at a `heading_path` change first (6.6); apply `prose_overlap_tokens` between consecutive prose chunks (6.4). Tables, blocks, formulas, figures: one segment → one or more chunks, never overlapping (6.5).
- Blocks over budget are split by rows with `label` and `header_row` repeated in every piece (4.3).
- A single indivisible unit over budget (one paragraph, one table row) is truncated at a token boundary and the chunk marked `truncated=True`, so 6.3 holds without silently dropping content.

**Dependencies**
- Inbound: `IngestPipeline` (P0)
- Outbound: `identity.chunk_id` (P0)
- External: `ModelTokenizer` from the runtime (P0)

**Contracts**: Service [x]

##### Service Interface
```python
class Chunker:
    def __init__(self, tokenizer: ModelTokenizer, *, budget: int, prose_overlap_tokens: int) -> None: ...
    def chunk(self, source: SourceFile, extracted: Extracted, *, params: ParamsFingerprint) -> tuple[ChunkRecord, ...]: ...

OVERLAP_POLICY: Final[dict[ChunkKind, bool]] = {
    ChunkKind.PROSE: True, ChunkKind.TABLE: False, ChunkKind.BLOCK: False,
    ChunkKind.FORMULA: False, ChunkKind.FIGURE: False,
}
```
- Preconditions: `budget == tokenizer.max_input_tokens` is asserted at construction — a mismatch is a configuration error, not something to reconcile silently.
- Postconditions: every record satisfies `tokenizer.exceeds_limit(DocumentText(record.text, record.title), DOCUMENT) is False`.
- Invariants: `OVERLAP_POLICY` covers every `ChunkKind` member (pinned by test, so a sixth kind cannot be added without deciding its policy).

### Persistence

#### `StateStore`

| Field | Detail |
|-------|--------|
| Intent | Make incremental re-runs and cache hits possible with one file and no dependency |
| Requirements | 5.4, 5.5, 8.1–8.5 |

**Contracts**: State [x]

##### State Management
- State model: three tables in one SQLite file — `file_state(root_id, relative_path, content_hash, params_fingerprint, status, last_seen_run)`, `chunk_registry(chunk_id, root_id, relative_path)`, `vision_cache(image_sha256, model_id, prompt_version, text, created_at)`.
- Persistence & consistency: one transaction per file covering `file_state` and its `chunk_registry` rows; a run id stamps `last_seen_run`, and files whose `last_seen_run` is older than the current run at the end are the deleted set (8.4).
- Concurrency strategy: single writer; the vision cache is read and written under the same connection from the describer's worker threads via a lock. Two simultaneous ingest processes are unsupported and detected by SQLite's lock.

##### Service Interface
```python
class StateStore:
    def classify(self, file: SourceFile, content_hash: str, params: ParamsFingerprint) -> FileStatus: ...
    def commit_file(self, file: SourceFile, content_hash: str, params: ParamsFingerprint, chunk_ids: Sequence[str], run_id: str) -> None: ...
    def deleted_since(self, run_id: str) -> tuple[tuple[SourceFile, tuple[str, ...]], ...]: ...
    def cached_description(self, image_sha256: str, model_id: str, prompt_version: str) -> str | None: ...
    def store_description(self, image_sha256: str, model_id: str, prompt_version: str, text: str) -> None: ...
```

### Orchestration

#### `IngestPipeline`

| Field | Detail |
|-------|--------|
| Intent | Run the stages in order for every file, isolate failures, and produce the report |
| Requirements | 8.3, 8.6, 9.1–9.5 |

**Contracts**: Batch [x]

##### Batch / Job Contract
- Trigger: `run_ingest(config: IngestConfig, tokenizer: ModelTokenizer, *, describer: VisionDescriber | None = None) -> RunReport`. The caller passes the runtime's tokenizer; the pipeline builds the describer from the discovered credential unless one is injected (tests inject a fake).
- Input / validation: `IngestConfig` validated at construction; `budget` cross-checked against the tokenizer.
- Output / destination: `RunReport` (counts, omissions, removed chunk ids, `no_work_required`) returned in memory and rendered by `report.render`; `ChunkRecord`s yielded to the caller via `RunReport.records` — this feature stores none of them.
- Idempotency & recovery: a re-run is a no-op for unchanged files by construction; a crash leaves committed files committed and uncommitted files classified as new; the vision cache survives crashes because it commits per image.
- Isolation: each file is wrapped so that `IngestError` and any other `Exception` becomes `Omission(category=FAILED, reason=...)` and the loop continues; `KeyboardInterrupt` propagates.

### Summary-only components
- **`IngestConfig`** — frozen dataclass; `roots: tuple[Path, ...]` (≥1), `include`/`exclude` glob tuples, `token_budget`, `prose_overlap_tokens` (default 64), `min_page_chars` (default 50), `min_image_pixels` (default 200, applied to the shorter side), `vision_model` (default `mistralai/mistral-small-3.2-24b-instruct`), `vision_base_url`, `vision_concurrency` (4), `state_path`. Validation refuses an empty `roots`, a non-positive budget, or overlap ≥ budget.
- **`OpenRouterCredential`** — same shape as `HfCredential`: `reveal()` only; `__repr__`/`__str__` return `REDACTED`; `discover_openrouter_credential(env, start)` reads `OPENROUTER_API_KEY` from the environment or the nearest `.env` via the runtime's `find_dotenv`/`parse_dotenv`.
- **`Discoverer`** — resolves each root, walks with `\\?\`-prefixed paths on Windows, applies include/exclude, de-duplicates by resolved absolute path, derives `author` as the first directory component beneath the root, and emits an `Omission(ROOT_UNAVAILABLE)` per unreadable root.
- **`Router`** — extension plus a content sniff (PDF magic, zip signature for `.xlsx`, image header via `pillow`); unmatched → `Omission(UNSUPPORTED)`.
- **`identity`** — `params_fingerprint(config, tokenizer_id, extractor_versions)` as a SHA-256 over canonical JSON; `content_hash(path)`; `chunk_id(root_id, relative_path, kind, locator, ordinal, params)`.
- **`report`** — counts by status and category; a human-readable rendering that lists every omission by path and reason and names the missing capability where one applies (9.4).

## Data Models

### Domain Model
- **Aggregate: the source file.** A `SourceFile` and everything derived from it — its `Extracted` segments, its `ChunkRecord`s, its `file_state` row and its `chunk_registry` rows — change together and are committed together. Nothing derived from one file references another.
- **Value objects**: `Locator` (a discriminated union: `MarkdownLocator(line_range, ordinal)`, `PageLocator(page)`, `SheetLocator(sheet, cell_range)`, `ImageFileLocator()`), `Provenance(vision_model, prompt_version)`, `ParamsFingerprint`, `ChunkKind` (`PROSE`, `TABLE`, `BLOCK`, `FORMULA`, `FIGURE`), `Omission(category, path, reason, missing_capability)`.
- **Invariants**: a `ChunkRecord` has exactly one `Locator`; a `FIGURE` record has a non-`None` `Provenance` and no other kind does; `Omission.missing_capability` is set iff `category` is `VISION_UNAVAILABLE`; `Resolved` has exactly one of `figure`/`omission`.

### Logical Data Model
```python
@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str                 # stable, file-local derivation
    source_path: Path
    root_id: str
    author: str
    title: str | None
    kind: ChunkKind
    text: str
    locator: Locator
    ordinal: int                  # position within the source file
    token_count: int              # as measured with the title attached
    truncated: bool
    provenance: Provenance | None
```
`ChunkRecord` is the contract `vector-index` builds against; `RunReport` is the contract `search-cli` builds against. Both are exported from `npu_rag.ingest.__init__`.

### Physical Data Model
SQLite, one file at `IngestConfig.state_path`:

```sql
CREATE TABLE file_state (
  root_id TEXT NOT NULL, relative_path TEXT NOT NULL,
  content_hash TEXT NOT NULL, params_fingerprint TEXT NOT NULL,
  status TEXT NOT NULL, last_seen_run TEXT NOT NULL,
  PRIMARY KEY (root_id, relative_path));
CREATE TABLE chunk_registry (
  chunk_id TEXT PRIMARY KEY, root_id TEXT NOT NULL, relative_path TEXT NOT NULL);
CREATE INDEX chunk_registry_file ON chunk_registry (root_id, relative_path);
CREATE TABLE vision_cache (
  image_sha256 TEXT NOT NULL, model_id TEXT NOT NULL, prompt_version TEXT NOT NULL,
  text TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (image_sha256, model_id, prompt_version));
```

## Error Handling

### Error Strategy
`IngestError(message, *, stage: str, path: Path | None)` mirrors the runtime's taxonomy: `DiscoveryError`, `ExtractionError`, `VisionError`, `VisionUnavailable`, `StateError`. Errors are raised where they occur and converted to `Omission`s at exactly one place — the per-file loop in `IngestPipeline` — so isolation is a property of the orchestrator, not of every extractor remembering to catch.

### Error Categories and Responses
- **Input problems** (unreadable root, unsupported type, hidden sheet, image below threshold): recorded as omissions with a category; never raised.
- **Extraction failures** (corrupt PDF, malformed workbook, undecodable image): `ExtractionError` → `Omission(FAILED)`; the file is not committed, so the next run retries it.
- **Vision failures**: 401/403 → `VisionUnavailable` latched for the run, every remaining `ImageRef` becomes `VISION_UNAVAILABLE` with `missing_capability="vision credential rejected"`; 429 → wait and retry once; anything else → `VisionError` → `Omission(VISION_FAILED)`, file still committed with its other chunks, the image retried next run.
- **State failures** (locked database, schema mismatch): `StateError` aborts the run before any file is processed; this is the one fatal category, because continuing would produce records nothing can classify later.

### Monitoring
The `RunReport` is the observability surface: counts by status and by omission category, every omission by path, the number of vision requests issued versus served from cache, and `no_work_required`. No logging framework is introduced; the report is returned and rendered.

## Testing Strategy

The runtime's standing lesson applies: every fixture must be able to tell right from wrong, and each load-bearing guard is proved by planting its negation.

- **Unit — extraction**: a Markdown fixture with front matter, an HTML block, a heading hierarchy, a GFM table, and images with and without alt text, asserting heading paths, `TableSegment` emission and `ImageRef` line anchors (3.1, 3.2, 3.5). An Excel fixture with two blank-row-separated blocks, **a merged label spanning three rows**, a formula with a cached value and one without, a hidden sheet, one chart and one image — asserting block boundaries (a naive all-`None` rule must fail), `FormulaSegment`s, the `VALUE_UNAVAILABLE` omission, the hidden-sheet omission, and chart source ranges read without OCR (4.1–4.8). A PDF fixture with one text page and one textless page, asserting the threshold branch on both sides (2.2, 2.3).
- **Unit — chunking**: budget never exceeded when measured **with the title attached** (a fixture whose content alone fits but content-plus-title does not); overlap present on prose and absent on every other kind, with `OVERLAP_POLICY` covering every `ChunkKind`; a block over budget split with label and header repeated in every piece (4.3, 6.2–6.6).
- **Unit — vision seam**: a fake describer proving the gate order threshold → cache → credential → describe; `OpenRouterDescriber` over `httpx.MockTransport` for payload shape, `Retry-After` handling, the 401 run-wide latch, and that no exception message or `repr` contains the key (5.x, 10.4).
- **Unit — identity and state**: `chunk_id` unchanged when an unrelated file changes and identical across two runs (7.4, 7.5); `params_fingerprint` changes when any listed parameter changes and only then (8.2, 8.5); `deleted_since` returns exactly the files absent this run with their chunk ids (8.4).
- **Integration — pipeline**: a temp root with mixed files where one is corrupt — the run completes, the corrupt file is a `FAILED` omission, every other file is committed (9.1, 9.5); a second identical run extracts nothing, issues no describer call, and reports `no_work_required` (8.3, 8.6); the same run with no credential reports every `ImageRef` as `VISION_UNAVAILABLE` naming the capability, and the run has no network access at all (5.2, 9.4, 10.2).
- **Contract — requirement 6.7**: for a sample of real archive chunks, `tokenizer.count_tokens(DocumentText(text, title), DOCUMENT) <= budget` implies the index is absent from `encode_documents([...]).truncated_indices`, and `> budget` implies it is present — both directions, in the standing suite with the ungated control tokenizer, and opt-in against the gated default.
- **Guard**: `tests/ingest/test_package_baseline.py` walks every module and asserts the dependency direction, that only `vision.py` imports `httpx`, and that `extract/*` never imports `vision` or `state` (10.3).

## Security Considerations
- The only secret is the OpenRouter key. It is read from the environment or `.env`, held in `OpenRouterCredential`, revealed at one line inside request construction, and never enters a log, report, record, `repr`, or chained traceback (10.4). A test greps the rendered exception for the key's value.
- Content leaves the machine only through `vision.py`, only when a credential is configured, and only as images. This is the owner's accepted trade, recorded in the brief; the layer guard makes any second exit path a test failure rather than a code-review judgement.
- The archive is opened read-only; the state file is the only thing written, at a configurable path outside the roots.

## Performance & Scalability
- First run over the current archive: ~1,343 text files extract locally in seconds to minutes; up to 3,837 images go through the vision seam at bounded concurrency, dominated by API latency — expect tens of minutes and a few dollars, once. Subsequent runs cost one hash per file plus work for new files only.
- The cache turns cost from per-run into per-new-image; a prompt or model change is the one event that re-spends across the whole archive, and it is called out as a revalidation trigger for that reason.
- Memory: one file at a time; a rendered PDF page or a workbook is the largest transient. No whole-archive structure is held.
