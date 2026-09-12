# Research Document: search-cli

## 1. CLI Framework Evaluation

### Context & Goals
The CLI is the user's primary interface to `npu_rag_from_folder`. A user querying on a GPD Pocket 4 demands near-zero startup latency (< 50 ms for `--help` or `status`), robust Windows terminal compatibility, and zero dependency friction.

### Comparison
1. **Standard Library `argparse`**:
   - Built into Python 3.12 standard library.
   - Import latency: **~2–4 ms**.
   - Zero external dependencies.
   - Fully supports subcommands (`index`, `search`, `status`, `tags`, `benchmark`), custom type converters, mutually exclusive groups, and clean help generation.
2. **`click`**:
   - External dependency (~15 ms import).
   - Good ergonomics, but adds unnecessary dependency weight for a utility that already wraps rich domain objects.
3. **`typer`**:
   - Requires `click`, `pydantic` or `rich`. Import latency **~70–120 ms**.
   - Unacceptable for quick terminal lookups.

### Decision
Use **Python 3.12 standard library `argparse`**. It guarantees the fastest startup, zero packaging bloat, and full native control over help formatting and error exits.

---

## 2. Configuration Resolution Hierarchy

### Precedence Order
Configuration values must be resolved in strict order of specificity:
1. **Explicit CLI Flags** (e.g. `--folder "Bhagwati Ugra Tara Sadhana"`, `--provider npu`, `--mode hybrid`)
2. **Environment Variables** prefixed with `NPU_RAG_` (e.g. `NPU_RAG_ARCHIVE_PATH`, `NPU_RAG_PROVIDER`, `NPU_RAG_INDEX_PATH`)
3. **Project / User TOML Configuration File** (`npu-rag.toml` in current directory, repository root, or `~/.config/npu-rag/config.toml`)
4. **Built-in Defaults** (default index path `./.npu-rag/index`, hybrid mode, cosine metric, RRF fusion, k=60)

### TOML Parsing
Python 3.12 includes `tomllib` in the standard library. Reading `npu-rag.toml` adds zero external dependencies.

---

## 3. Tag Taxonomy & Folder Resolution Architecture

### The Domain Model
- **Folders**: The primary organizational unit (representing notebooks, lecture series, seminar archives).
- **Tags**: High-level domains (e.g. `siddha`, `finance`, `ai`, `psychology`, `economics`).
- **Relationship**: Many-to-many. $N$ folders can be assigned to a tag; a folder can possess multiple tags.

### Specification in `npu-rag.toml`
```toml
[tags]
siddha = [
    "Bhagwati Ugra Tara Sadhana",
    "Bhagwan Garuda*",
]

finance = [
    "Axia_*",
    "BM_*",
    "B_ Dynamic*",
    "B_ Marketing*",
    "B_ Restabilization of German Statutory Pension Insurance",
    "B_ The Price of Time",
    "B_ Wavelets in Finance",
]

ai = [
    "AI and Our Economic Future",
    "AI_ Durable SEO Strategy in the Age of AI",
    "AI_ The Infinite Leader*",
]
```

### Query-Time SQL Pushdown Compilation
When a user specifies `--tag <tag_name>` (e.g. `--tag siddha`), the CLI:
1. Looks up `tag_name` in the tag registry.
2. Expands any glob patterns against the discovered folders or index metadata.
3. Compiles a LanceDB SQL WHERE predicate:
   ```sql
   (author = 'Bhagwati Ugra Tara Sadhana' OR file_path LIKE '%Bhagwati Ugra Tara Sadhana%' OR file_path LIKE '%Bhagwan Garuda%')
   ```
4. Injects this into `SearchOptions.filter_sql`. LanceDB executes this as a pushdown pre-filter during retrieval, ensuring that only chunks within the tagged folders are scanned and ranked.

---

## 4. Order Relevance Per Folder & Chronological Grouping

### The Problem
Traditional vector retrieval scores every chunk independently. For a structured folder like `Bhagwati Ugra Tara Sadhana` containing consecutive sessions (Day 1 Intro, Day 1 Session 2, Day 2, Mahasiddha Ishaputra Part 1, Part 2), pure relevance ranking scatters pieces of the lecture randomly across the results list.

### The Solution: Sequential Presentation & Grouping
The CLI provides an order-preserving presentation mode:
1. **`--group-by-folder`**: Results are partitioned by folder. Within each folder, hits are arranged in **narrative sequence**:
   - First sorted by document chronological/filename order (`Day 1` before `Day 2`, `Part 1` before `Part 2`, dates sorted chronologically).
   - Second sorted by intra-document chunk ordinal (`chunk_index` 0, 1, 2...).
2. **Contextual Expansion (`--context N`)**:
   Calls `ContextExpander` from `vector-index` to pull $N$ preceding and succeeding chunks in exact sequence, restoring the flow of the discourse.
3. **Sequential Breadcrumb Attribution**:
   Each snippet is rendered with its exact position in the series:
   `[Folder: Bhagwati Ugra Tara Sadhana] [Doc 1/13: First Day - Part 1] [Chunk #14]`

---

## 5. Terminal Formatting & Windows Console Compatibility

### ANSI & UTF-8 Handling
- Windows Terminal and modern Windows 11 `conhost` support full VT100 ANSI escape sequences.
- To prevent garbled text when redirected:
  - Check `sys.stdout.isatty()`. If output is piped to a file or another command (e.g. `npu-rag search ... | grep`), strip all ANSI escape sequences automatically.
  - Support a `--no-color` flag (and respect `NO_COLOR` environment variable per the no-color.org standard).
  - Explicitly set `sys.stdout.reconfigure(encoding="utf-8")` on Windows to handle Devanagari, German umlauts (`ä, ö, ü, ß`), Chinese characters, and mathematical symbols without CP1252 `UnicodeEncodeError`.

---

## 6. Lazy Subsystem Imports (Fast Startup)

### Measurement & Strategy
- Importing `torch`: ~1,200 ms
- Importing `onnxruntime`: ~250 ms
- Importing `lancedb`: ~180 ms
- Importing `transformers`: ~600 ms

### Architectural Invariant
`src/npu_rag/cli/main.py` and top-level modules must **never** import `onnxruntime`, `lancedb`, `torch`, or `transformers` at module scope.
- Subcommand parsers and help handlers resolve purely within `argparse`.
- Subsystem imports are strictly deferred into the execution body of each command handler:
  - `cmd_status`: only loads LanceDB store to inspect metadata.
  - `cmd_tags`: only reads config and folder registry (0 heavy imports).
  - `cmd_search`: imports `VectorIndex` and embedder only after arguments are fully validated.
  - `cmd_index`: imports `IndexSynchronizer`, ingest, and embedder only upon starting indexing.
