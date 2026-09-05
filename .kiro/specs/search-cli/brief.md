# Brief: search-cli

## Problem

The ingest pipeline, NPU embedder, and index are libraries — they do nothing on their own. Someone sitting at a GPD Pocket 4 who wants to find a half-remembered essay needs a command they can type, and someone who wants to know whether the NPU is actually being used needs the tool to tell them plainly rather than leave them guessing.

Retrieval quality is also not a fixed thing. Whether hybrid beats pure vector, whether reranking is worth its latency, whether MMR helps or hurts — these are empirical questions on this specific corpus. If the search algorithm is hardcoded, none of them can be answered. The CLI is where that experimentation happens.

## Current State

- No CLI exists.
- `uv` 0.12.5 is installed; `uvx` is the required distribution mechanism.
- The full search-algorithm surface has been designed (modes, ANN variants, metrics, fusion, reranking, MMR, granularity, filters) and is owned by `vector-index`. This spec exposes it.

## Desired Outcome

- `uvx` runs indexing and search without a manually activated environment.
- A user can index the archive, search it, and see which provider ran and how long each phase took.
- Every retrieval knob is reachable from flags, so search strategy can be compared empirically rather than assumed.
- Re-running index after new posts arrive is fast and reports exactly what changed.
- The benchmark from `npu-embedding-runtime` is runnable as a subcommand.

## Approach

A small command surface over the libraries, with configuration resolution owned here and injected downward — libraries read no global state.

Commands: `index` (walk, extract, chunk, embed, upsert, with incremental behaviour by default and an explicit full-rebuild flag), `search` (the full retrieval flag surface), `status` (corpus and index health: chunk counts, model and dimension, index freshness, last run, skipped files), and `benchmark` (delegating to the runtime spec's harness rather than reimplementing it).

Provider reporting is treated as a first-class output, not a log line. Because `--provider` forbids implicit fallback, the CLI must state which provider served the run and fail clearly when the requested one is unavailable.

## Scope

- **In**: `uvx`-compatible packaging with `[project.scripts]` entry points; configuration resolution (CLI flags over environment variables over TOML file over defaults) including `ARCHIVE_PATH`; the four commands above; the full search flag surface (`--mode`, `--ann`, `--metric`, `--fusion`, `--rerank`, `--top-k`, `--fetch-k`, `--mmr`, `--granularity`, `--context`, `--filter`, `--prefilter/--postfilter`, `--nprobes`, `--refine-factor`); `--provider npu|cpu|auto` with explicit reporting and no silent fallback; progress reporting for long ingests; human-readable and JSON output formats; result rendering with source path, author, title, score, and snippet; clear exit codes and actionable error messages.
- **Out**: Retrieval logic itself (`vector-index` owns ranking; this spec must not reimplement or diverge from it). Embedding (`npu-embedding-runtime`). Extraction and chunking (`document-ingest`). The MCP protocol surface (`mcp-server`). Any TUI, interactive REPL, or GUI. Answer generation.

## Boundary Candidates

- Packaging and entry points
- Configuration resolution and precedence
- Command definitions and argument parsing
- Output rendering and formatting
- Progress and provider reporting

## Out of Boundary

- Deciding default retrieval parameters. Defaults are proposed by `vector-index`; this spec surfaces and can override them.
- Reimplementing the benchmark harness.
- Installing the Ryzen AI SDK or NPU driver — that is provisioning, owned by `npu-embedding-runtime`. The CLI may *detect* and report a bad environment, but does not fix it.

## Upstream / Downstream

- **Upstream**: `vector-index` for the query-orchestration API, `document-ingest` for the ingest pipeline, `npu-embedding-runtime` for the embedder and benchmark harness.
- **Downstream**: The user. Also shell scripting and automation, which is why JSON output matters.

## Existing Spec Touchpoints

- **Extends**: None. Greenfield.
- **Adjacent**: `mcp-server` is the sibling surface. Both consume the same retrieval API, and both must produce identical rankings for identical parameters. Shared config and result-formatting helpers are a seam to watch for duplication.

## Constraints

- Must be launchable via `uvx` — this is the packaging requirement that conflicts with AMD's conda-based install model and may be affected by the outcome of the `npu-embedding-runtime` spike.
- Python 3.12, `uv`-managed.
- Windows-first. Console encoding, ANSI colour support, and long paths must all behave.
- `--provider` must never fall back silently.
- Startup latency matters. A search that takes seconds to begin is a search that does not get used; heavy imports should be deferred until needed.
- No network access at query time.
