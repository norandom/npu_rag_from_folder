# Brief: search-cli

## Problem

The ingest pipeline, NPU embedder, and vector index are libraries — they do nothing on their own. Someone sitting at a GPD Pocket 4 who wants to find a half-remembered essay, financial analysis, or sadhana transcript needs a command they can type. Someone who wants to know whether the NPU is actually being used needs the tool to tell them plainly rather than leave them guessing.

Furthermore, this tool is fundamentally **folder-based** (as named in `npu_rag_from_folder`). Documents are organized by folders (notebooks, lecture series, seminar archives, and topical categories like `Bhagwati Ugra Tara Sadhana`, `Axia_ Market Profile Masterclass`, or `B_ Restabilization of German Statutory Pension Insurance`). 

Two structural realities govern retrieval over this archive:
1. **Topical Tagging**: Folders belong to high-level domains (`siddha`, `finance`, `ai`, `psychology`, `economics`). A user needs to search across all $N$ folders associated with a tag (`--tag siddha`), and $N$ folders must be able to share tags without requiring data re-indexing.
2. **Order Relevance Per Folder**: Within a folder, order is paramount. Documents follow a sequential or chronological progression (Session 1 $\rightarrow$ Session 2, Part 1 $\rightarrow$ Part 2, Day 1 $\rightarrow$ Day 2). Shuffling chunks purely by raw cosine similarity destroys the narrative thread. The CLI must allow grouping hits by folder and preserving chronological/discourse order, with neighbouring context expansion (`--context N`).

Finally, retrieval quality is an empirical question. Whether hybrid beats pure vector, whether MMR helps, or which metric performs best on this specific corpus requires that every search knob be reachable from CLI flags.

## Current State

- No CLI exists yet.
- `uv` 0.12.5 is installed; `uvx` is the required distribution mechanism.
- The foundational libraries are complete and verified:
  - `npu-embedding-runtime`: Vitis AI NPU / CPU execution providers, BF16 static shapes, benchmark harness.
  - `document-ingest`: Recursive folder walking, markdown/pdf/txt extraction, token-budget chunking, content hash tracking.
  - `vector-index`: LanceDB storage, dense/lexical/hybrid search, RRF/linear fusion, MMR diversity, neighbouring chunk context expansion, index sync.

## Desired Outcome

- `uvx` or `uv run npu-rag` runs indexing and search without manual environment activation.
- The CLI is strictly **folder-first**: `--folder` filters to specific notebooks/folders or path globs.
- A clean **tag taxonomy** (`[tags]` in `npu-rag.toml`) allows querying $N$ folders per tag via `--tag <name>`.
- Search presentation respects **folder order**: `--group-by-folder` preserves chronological/discourse sequence within each notebook folder.
- A user can index the archive, search it, and see which provider ran and how long each phase took.
- Provider reporting is first-class, not a debug log line: `--provider npu|cpu|auto` forbids silent fallback to CPU.
- Startup is instantaneous: heavy ONNX, PyTorch, or LanceDB imports are deferred so that `--help`, `status`, or `tags` execute in milliseconds.

## Approach

A lean, robust command-line application using standard library `argparse` (or lightweight parser) with lazy subsystem imports and unified configuration resolution (CLI flags > Environment variables > TOML config > Defaults).

### Core Commands
1. **`index`**: Walk folders, extract, chunk, embed (via NPU or CPU), and upsert into LanceDB. Incremental by default via SHA256 hashes, with an explicit `--full-rebuild` flag.
2. **`search`**: The complete retrieval flag surface:
   - Folder scoping: `--folder <name_or_glob>`, `--tag <tag_name>`
   - Order & Context: `--group-by-folder`, `--context <N>`
   - Retrieval mode: `--mode dense|fts|hybrid`, `--metric cosine|dot|l2`
   - Fusion: `--fusion rrf|linear`, `--rrf-k`, `--linear-alpha`
   - Diversity: `--mmr <diversity>`, `--max-chunks-per-doc <N>`, `--rollup`
   - Provider: `--provider npu|cpu|auto`
   - Formats: Terminal colored rendering and `--json`
3. **`status`**: Corpus and index health: chunk count, file count, folders indexed, model name and embedding dimensions, index freshness timestamp, and skipped/omitted files.
4. **`tags`**: List all configured tags, their mapped folders, and chunk distribution.
5. **`benchmark`**: Run the NPU vs CPU embedding throughput and latency benchmark harness directly from the CLI.

## Scope

- **In**:
  - `uvx`-compatible packaging with `[project.scripts]` entry point (`npu-rag`).
  - Configuration resolution precedence: CLI flags > `NPU_RAG_*` env vars > `npu-rag.toml` > defaults.
  - Folder-based filtering (`--folder`) and tag taxonomy mapping (`[tags]` in config and `--tag`).
  - Order-preserving presentation (`--group-by-folder`) and context expansion (`--context N`).
  - Explicit provider execution and reporting (`--provider npu|cpu|auto`) with zero silent fallback.
  - Progress reporting during indexing (file counter, chunk counter, throughput).
  - Human-friendly terminal rendering (Windows ANSI/color safe) and structured `--json` output.
  - Clean exit codes and actionable diagnostics.
- **Out**:
  - Retrieval logic itself (owned by `vector-index`).
  - Extraction and chunking logic (owned by `document-ingest`).
  - Audio transcription (handled by specialized tools like `tools/transcribe_audio.py`).
  - MCP JSON-RPC protocol surface (owned by `mcp-server`).
  - TUI, GUI, or interactive REPL.

## Boundary Candidates

- CLI entry point and lazy-loading dispatch (`src/npu_rag/cli/main.py`)
- Configuration resolution hierarchy (`src/npu_rag/cli/config.py`)
- Tag taxonomy resolver (`src/npu_rag/cli/tags.py`)
- Output rendering and ANSI terminal formatting (`src/npu_rag/cli/render.py`)
- Command handlers: `cmd_search.py`, `cmd_index.py`, `cmd_status.py`, `cmd_tags.py`, `cmd_benchmark.py`

## Constraints

- Windows-first: Console encoding (CP1252 vs UTF-8), ANSI color support, and long paths must behave flawlessly.
- Startup latency: Subcommands must not import `torch`, `onnxruntime`, or `transformers` until an actual embed or search execution is needed.
- No network access at query time (100% offline retrieval).
