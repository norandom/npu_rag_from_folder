# Requirements Document: search-cli

## Introduction

This feature provides the unified command-line interface for the `npu_rag_from_folder` system, packaged via `uvx` / `[project.scripts]` as the `npu-rag` command. It exposes the capabilities of `document-ingest`, `npu-embedding-runtime`, and `vector-index` to the user and automation scripts.

Its architecture is strictly **folder-first**: personal archives are organized as folders (notebooks, course modules, sermon/sadhana discourses, financial and economic collections), governed by a **tag taxonomy** ($N$ folders per tag, many-to-many), and presented with full respect for **folder chronological and narrative ordering**.

---

## Boundary Context

- **In scope**:
  - `[project.scripts]` packaging and `npu-rag` command dispatch.
  - Subcommands: `search`, `index`, `status`, `tags`, `benchmark`.
  - Configuration resolution hierarchy (CLI flags > environment variables > TOML file > defaults).
  - Folder scoping (`--folder`) and tag taxonomy resolution (`--tag`, `npu-rag tags`).
  - Order-preserving presentation (`--group-by-folder`) and context expansion (`--context`).
  - Search algorithm parameter controls (`--mode`, `--metric`, `--fusion`, `--mmr`, `--rollup`, etc.).
  - Explicit provider execution and reporting (`--provider npu|cpu|auto`, zero silent fallback).
  - Incremental indexing orchestration and progress reporting.
  - Terminal output formatting (Windows UTF-8, ANSI color auto-detection, and `--json` format).
  - Fast startup through deferred subsystem imports.

- **Out of scope**:
  - Core retrieval, fusion, and MMR algorithms (owned by `vector-index`).
  - Low-level document extraction and token chunking (owned by `document-ingest`).
  - Model ONNX export, NPU compilation, and Vitis AI session management (owned by `npu-embedding-runtime`).
  - MCP server protocol (owned by `mcp-server`).
  - Audio transcription pipelines (handled by specialized tools).

---

## Requirements

### Requirement 1: Packaging, Entry Points, and Fast Startup

**Objective:** As an archive user, I want to run `npu-rag` instantly from anywhere without activating virtual environments manually, with zero startup lag on lookups and help messages.

#### Acceptance Criteria
1. The package shall define a console entry point in `pyproject.toml` under `[project.scripts]` named `npu-rag`, pointing to `npu_rag.cli.main:main`.
2. Running `npu-rag --help`, `npu-rag --version`, or any invalid command invocation shall complete in under 50 milliseconds by deferring all heavy imports (`onnxruntime`, `torch`, `lancedb`, `transformers`).
3. The CLI shall provide five primary subcommands:
   - `search`: Query retrieval across the archive with filtering and ranking controls.
   - `index`: Ingest, chunk, embed, and synchronize local documents into LanceDB.
   - `status`: Display index health, chunk counts, model architecture, and freshness.
   - `tags`: Inspect and validate the configured tag-to-folder taxonomy.
   - `benchmark`: Execute the embedding throughput and latency benchmark harness.

---

### Requirement 2: Configuration Precedence & Resolution

**Objective:** As a user or sysadmin, I want predictable configuration resolution so that CLI flags override environment variables, which in turn override local TOML files.

#### Acceptance Criteria
1. The CLI shall resolve configuration parameters in the following strict order of precedence (highest to lowest):
   - Explicit command-line arguments.
   - Environment variables prefixed with `NPU_RAG_` (e.g. `NPU_RAG_ARCHIVE_PATH`, `NPU_RAG_PROVIDER`, `NPU_RAG_INDEX_PATH`).
   - Project configuration file `npu-rag.toml` discovered in the current directory, parent directories, or `~/.config/npu-rag/config.toml`.
   - Built-in system defaults.
2. The configuration system shall parse `npu-rag.toml` using Python 3.12's standard library `tomllib`.
3. If a specified configuration file is malformed, the CLI shall exit with an informative error message and code 1, without crashing with an unhandled exception trace.

---

### Requirement 3: Folder-First Scoping & Tag Taxonomy

**Objective:** As a researcher managing distinct notebooks and topics, I want to scope queries to specific folders or across entire tagged subject areas ($N$ folders per tag) without re-indexing my database.

#### Acceptance Criteria
1. The `search` command shall support a `--folder <name_or_glob>` flag that restricts retrieval to chunks belonging to the specified folder name, relative subfolder, or glob pattern.
2. The system shall support a `[tags]` section in `npu-rag.toml` defining many-to-many mappings between tag names and folder names/globs (e.g. `siddha = ["Bhagwati Ugra Tara Sadhana", "Bhagwan Garuda*"]`).
3. The `search` command shall support a `--tag <tag_name>` flag (and multiple comma-separated tags) that automatically expands the tag into its constituent folders and pushes down a combined SQL filter into the vector index.
4. When a user requests `--tag <tag_name>` for an undefined tag, the CLI shall report an actionable error listing all available tags.
5. The `npu-rag tags` subcommand shall list all configured tags, their resolved folders, and the count of indexed chunks in each folder.

---

### Requirement 4: Order Relevance & Chronological Presentation

**Objective:** As a practitioner studying structured lecture series, courses, or ritual sadhanas, I want search results within a folder presented in their natural sequential order rather than scattered by arbitrary score fragments.

#### Acceptance Criteria
1. The `search` command shall support a `--group-by-folder` flag.
2. When `--group-by-folder` is enabled, search results shall be grouped under their parent folder and ordered by natural document sequence (chronological dates or session numbers like `Day 1`, `Part 1`) and intra-document `chunk_index`.
3. The `search` command shall support a `--context <N>` flag (default 0) that invokes `ContextExpander` to attach $N$ preceding and $N$ succeeding sequential chunks to each hit.
4. Terminal output shall render full hierarchical breadcrumb attribution for every hit: `[Folder] -> [Document (Index/Total)] -> [Chunk #Ordinal | Page/Line Locator]`.

---

### Requirement 5: Search Algorithm & Retrieval Controls

**Objective:** As an empirical search practitioner, I want complete access to all hybrid retrieval and reranking parameters via CLI flags.

#### Acceptance Criteria
1. The `search` command shall support all core retrieval flags:
   - `--mode dense|fts|hybrid` (default `hybrid`).
   - `--metric cosine|dot|l2` (default `cosine`).
   - `--fusion rrf|linear` (default `rrf`).
   - `--rrf-k <int>` (default 60).
   - `--linear-alpha <float>` (default 0.5, dense weight).
   - `--top-k <int>` (default 10, final returned results).
   - `--fetch-k <int>` (default 20, candidate pool per retriever).
   - `--mmr <float>` (diversity balance $\lambda$, opt-in).
   - `--max-chunks-per-doc <int>` (cap chunks per document).
   - `--rollup` (enable document-level score aggregation).
   - `--filter <sql>` (arbitrary SQL WHERE predicate).
2. All numeric arguments shall be strictly validated at parse time with clear error messages for out-of-range values.

---

### Requirement 6: Provider Accountability (Zero Silent Fallback)

**Objective:** As an NPU operator, I want guaranteed transparency regarding which hardware processor executed my query or indexing run.

#### Acceptance Criteria
1. The CLI shall support `--provider npu|cpu|auto` across `index`, `search`, and `benchmark`.
2. When `--provider npu` is specified and the Vitis AI execution provider is unavailable or fails to initialize, the CLI shall exit with an error code and diagnostic instructions; it shall **never** silently fall back to CPU.
3. Every execution shall visibly report the active execution provider (e.g. `[Provider: VitisAIExecutionProvider (NPU)]` or `[Provider: CPUExecutionProvider]`) in terminal output and in `--json` payloads.

---

### Requirement 7: Indexing Command & Progress Reporting

**Objective:** As an archive curator, I want an efficient indexing command that skips unchanged files and reports live progress.

#### Acceptance Criteria
1. The `index` subcommand shall scan the configured archive roots, extract supported documents (`.md`, `.pdf`, `.txt`), chunk them, generate embeddings, and upsert them into LanceDB.
2. By default, `index` shall run incrementally, skipping any file whose SHA256 content hash matches the stored record.
3. The `index` subcommand shall support a `--full-rebuild` flag that drops the existing index table and re-indexes the entire corpus from scratch.
4. During indexing, the CLI shall display a dynamic console progress indicator showing: files scanned/processed, chunks embedded, elapsed time, and processing throughput (chunks/sec).
5. At the conclusion of indexing, a summary report shall display: files added/updated/skipped, chunks created, omissions by category, and total run duration.

---

### Requirement 8: Terminal Rendering & JSON Output

**Objective:** As both an interactive user and a script author, I want beautiful terminal formatting when interactive and clean JSON when piping to downstream tools.

#### Acceptance Criteria
1. The CLI shall format interactive terminal output with clean UTF-8 characters, colors, and snippet highlights.
2. The CLI shall automatically detect if stdout is not a TTY (or if `NO_COLOR` is set) and strip all ANSI styling.
3. The CLI shall provide a `--json` flag on `search`, `status`, `tags`, and `index` that emits valid, structured JSON to stdout for piping to tools like `jq`.
4. The CLI shall enforce UTF-8 stdout encoding on Windows, preventing charmap encode crashes on Devanagari, Chinese, or German characters.
