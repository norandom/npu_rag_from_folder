# Task Plan: search-cli

## Phase 1: Foundation & Configuration

- [ ] 1.1 Package Entry Point & Fast Startup Dispatcher
  - Register `npu-rag = "npu_rag.cli.main:main"` under `[project.scripts]` in `pyproject.toml`.
  - Create `src/npu_rag/cli/__init__.py` and `src/npu_rag/cli/main.py`.
  - Implement top-level `ArgumentParser` and lazy command routing.
  - Add test asserting `--help`, `--version`, and top-level entry take < 50 ms and perform zero heavy imports (`torch`, `onnxruntime`, `lancedb`).
  - *Files:* `pyproject.toml`, `src/npu_rag/cli/main.py`, `tests/cli/test_cli_baseline.py`

- [ ] 1.2 Configuration Resolution Hierarchy
  - Implement `src/npu_rag/cli/config.py` defining `CliConfig` and `load_config()`.
  - Parse `npu-rag.toml` via standard library `tomllib`.
  - Enforce precedence: CLI Flags > `NPU_RAG_*` Env Vars > `npu-rag.toml` > Defaults.
  - Handle malformed config files gracefully with exit code 1 and actionable errors.
  - *Files:* `src/npu_rag/cli/config.py`, `tests/cli/test_cli_config.py`

---

## Phase 2: Tag Taxonomy & Folder Ordering

- [ ] 2.1 Tag Taxonomy & Pushdown Filter Builder
  - Implement `TagRegistry` in `src/npu_rag/cli/tags.py`.
  - Support $N$ folders per tag and $N$ folders sharing tags via `[tags]` in config.
  - Implement `build_filter_sql()` combining `--folder`, `--tag`, and custom `--filter` into LanceDB SQL WHERE expressions.
  - Validate unknown tag diagnostics listing valid available tags.
  - *Files:* `src/npu_rag/cli/tags.py`, `tests/cli/test_cli_tags.py`

- [ ] 2.2 Folder Chronological Sequence & Narrative Grouping
  - Implement `src/npu_rag/cli/order.py` with `natural_sort_key` (parsing session prefixes like `Day 1`, `Part 2`, dates).
  - Implement `group_and_order_results()` grouping hits by folder, sorted by document sequence and `chunk_index`.
  - Validate that ordered presentation maintains narrative flow across session parts.
  - *Files:* `src/npu_rag/cli/order.py`, `tests/cli/test_cli_order.py`

---

## Phase 3: Presentation & Rendering

- [ ] 3.1 Terminal Formatter, Windows UTF-8, and JSON Serialization
  - Implement `src/npu_rag/cli/render.py`.
  - Enforce UTF-8 console output encoding on Windows.
  - Implement ANSI styling with automatic stripping when output is redirected (non-TTY) or `NO_COLOR` is set.
  - Implement hierarchical breadcrumb formatting: `[Folder] -> [Document] -> [Chunk # | Locator]`.
  - Implement structured `--json` formatting for machine consumption.
  - *Files:* `src/npu_rag/cli/render.py`, `tests/cli/test_cli_render.py`

---

## Phase 4: Search & Inspection Commands

- [ ] 4.1 Search Command Implementation
  - Implement `cmd_search.py` wiring all CLI search flags (`--mode`, `--metric`, `--fusion`, `--mmr`, `--rollup`, `--top-k`, `--fetch-k`).
  - Wire `--folder`, `--tag`, `--group-by-folder`, and `--context` window expansion.
  - Enforce `--provider npu|cpu|auto` with visible provider attribution and zero silent fallback.
  - Verify exit codes (0 for success, 1 for argument/tag error, 2 for provider/runtime error).
  - *Files:* `src/npu_rag/cli/cmd_search.py`, `tests/cli/test_cli_search.py`

- [ ] 4.2 Status & Tags Inspection Commands
  - Implement `cmd_status.py` querying `VectorIndex.stats()`.
  - Implement `cmd_tags.py` listing configured tags, assigned folders, and indexed chunk counts.
  - Support both formatted terminal tables and `--json` outputs.
  - *Files:* `src/npu_rag/cli/cmd_status.py`, `src/npu_rag/cli/cmd_tags.py`, `tests/cli/test_cli_inspection.py`

---

## Phase 5: Ingest Synchronization & End-to-End Packaging

- [ ] 5.1 Incremental Indexing Command
  - Implement `cmd_index.py` connecting discovery, extraction, chunking, embedding, and `IndexSynchronizer`.
  - Provide incremental sync by default and `--full-rebuild` flag.
  - Render a dynamic console progress counter (files, chunks, throughput).
  - Produce final summary report with files processed/skipped and omissions.
  - *Files:* `src/npu_rag/cli/cmd_index.py`, `tests/cli/test_cli_index.py`

- [ ] 5.2 Benchmark Subcommand & E2E Validation
  - Implement `cmd_benchmark.py` delegating to `npu_rag.embedding.benchmark`.
  - Validate end-to-end execution of `npu-rag index` and `npu-rag search` in a temporary corpus.
  - Verify `uvx`-compatibility and `[project.scripts]` execution.
  - *Files:* `src/npu_rag/cli/cmd_benchmark.py`, `tests/cli/test_cli_e2e.py`
