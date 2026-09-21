# Task Plan: search-cli

## Phase 1: Foundation & Configuration

- [x] 1.1 Package Entry Point & Fast Startup Dispatcher
  - Register `npu-rag = "npu_rag.cli.main:main"` under `[project.scripts]` in `pyproject.toml`.
  - Create `src/npu_rag/cli/__init__.py` and `src/npu_rag/cli/main.py`.
  - Implement top-level `ArgumentParser` and lazy command routing.
  - Add test asserting `--help`, `--version`, and top-level entry take < 50 ms and perform zero heavy imports (`torch`, `onnxruntime`, `lancedb`).
  - *Files:* `pyproject.toml`, `src/npu_rag/cli/main.py`, `tests/cli/test_cli_baseline.py`

- [x] 1.2 Configuration Resolution Hierarchy
  - Implement `src/npu_rag/cli/config.py` defining `CliConfig` and `load_config()`.
  - Parse `npu-rag.toml` via standard library `tomllib`.
  - Enforce precedence: CLI Flags > `NPU_RAG_*` Env Vars > `npu-rag.toml` > Defaults.
  - Handle malformed config files gracefully with exit code 1 and actionable errors.
  - *Files:* `src/npu_rag/cli/config.py`, `tests/cli/test_cli_config.py`

---

## Phase 2: Tag Taxonomy & Folder Ordering

- [x] 2.1 Tag Taxonomy & Pushdown Filter Builder
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
  - **CONFLICTS WITH REALITY, flagged 2026-09-21 at preflight.** `npu_rag.embedding.benchmark` **does not exist**. What exists is `npu_rag.embedding.bench.harness` (`BenchmarkHarness.run(cells, sample) -> MatrixResult`, runtime task 6.3), plus `bench/corpus.py` and `bench/power.py`. Runtime tasks **6.4-6.7 were descoped**, so there is no repetition/variance, no fidelity or quality scoring, **no report renderer**, and no matrix entry point. Requirement 1.3 still lists `benchmark` as one of five subcommands and requirement 6.1 applies `--provider` to it. Resolve before implementing: either point this task at `bench.harness` and accept that the subcommand prints a raw `MatrixResult`, or block it and revive the descoped renderer. Do not invent a `benchmark` module to satisfy the task text.

---

## Implementation Notes

- 1.1: **Handler contract for tasks 4.1-5.2.** Each `cmd_<name>.py` must expose `run(args: argparse.Namespace) -> int`. `COMMAND_MODULES` and `COMMAND_SUMMARIES` in `main.py` must be extended **together** — a module row without a summary raises `KeyError` at parser-build time. A handler that is absent, or whose `run` is missing or not callable, exits 2 with a diagnostic on stderr; the handler's own return value is passed through unchanged.
- 1.1: **`src/npu_rag/cli/__init__.py` must stay empty of re-exports**, against design §2's "Public package export". `from npu_rag.cli.main import main` binds `main` on the package as a *function*, permanently shadowing the `main` submodule, and makes `python -m npu_rag.cli.main` emit a `RuntimeWarning` (fatal under `-W error`). Verified by planting it: 10 tests die. The console script resolves through the module, so nothing needs the re-export.
- 1.1: **argparse's default exit code collides with design §4.** argparse exits **2** on a usage error; §4 reserves 2 for hardware/runtime failure and assigns 1 to invalid arguments. `_CliParser.error()` overrides it to 1, so a typo and an unavailable NPU stay distinguishable to a caller. Subparsers inherit it. Restoring argparse's default kills 4 tests.
- 1.1: **Requirement 1.2's "under 50 milliseconds" is not satisfiable as a subprocess measurement on this machine and should be amended.** Measured 2026-09-21: bare `python -c pass` median **1226 ms** (min 158, max 1243 - extreme variance, most likely Defender scanning `.venv`), `python -m npu_rag.cli.main --help` median **1253 ms**. The CLI's own cost is **~27 ms** by difference and **22.4 ms** in-process (13.6 import + 8.8 parser). No implementation could pass a 50 ms subprocess stopwatch here, including an empty `main()`. The **load-bearing assertion is therefore structural**: a fresh interpreter runs the fast paths and asserts `onnxruntime`, `torch`, `lancedb`, `transformers` are all absent from `sys.modules`. The wall-clock check survives only as a generously-bounded smoke check, named as such. This follows `npu-embedding-runtime`'s standing lesson rules 6-8: an assertion about the machine is not a test of your code.
- 1.1: **a misspelled guard name cannot be caught by resolving it.** `HEAVY_MODULES` is pinned against a literal, which looks redundant and is not: an unresolvable name is indistinguishable from a legitimately absent optional dependency (`torch` lives in the `export` extra), so no `find_spec` check can tell `"torhc"` from `"torch is not installed here"`. A `find_spec`-based fix was written and planted against, and the typo mutant survived it. Only the literal comparison kills it.
- 1.1: **BINDING ON 2.1 - design §3.2's `build_filter_sql` is SQL-injectable and will break on real folder names.** It interpolates user input directly: `f"(author = '{folder}' OR file_path LIKE '%{folder}%')"`. `--folder`, `--tag` and the `[tags]` table in `npu-rag.toml` are all user-controlled, and a folder name containing an apostrophe breaks the query outright. Quote or parameterise; do not copy the design's snippet verbatim.
- 1.1: **BINDING ON 3.1 - do not call `sys.stdout.reconfigure(...)` unconditionally** as design §3.4 prescribes. `sys.stdout` is typed `TextIO`, which has no `reconfigure`; only `io.TextIOWrapper` does, so strict mypy rejects the bare call and it raises when stdout is redirected to a non-wrapper. Narrow with `isinstance(sys.stdout, io.TextIOWrapper)` — the same bug was fixed in `tools/transcribe_audio.py` in commit `e828554`.
- 1.2: **BINDING ON 4.1 - `dataclasses.replace()` on a `CliConfig` bypasses every check.** `CliConfig` validates only inside `load_config()`, deliberately **unlike** `IngestConfig`, which validates in `__post_init__` as the runtime's 13 other sites do. The reason is requirement 2.3's word *actionable*: the message must name where a bad value came from, and provenance is irrecoverably lost by the time `__post_init__` runs - `IngestConfig` can only say `token_budget must be positive`, which is exactly the unactionable message 2.3 forbids, while `load_config` says `npu-rag.toml: key 'provider' must be one of 'npu', 'cpu', 'auto'; got 'gpu'`. The distinction is the input's provenance, not the object's shape: `__post_init__` for objects built in code from trusted values, origin-aware validation for the terminus of an untrusted pipeline. **The cost lands on 4.1**: `replace(cfg, default_limit=args.top_k)` is the natural thing to write when wiring flags, and it validates nothing. Requirement 5.2 demands numeric flags be validated at parse time - 4.1 must do that itself.
- 1.2: **BINDING ON 3.1 - `emit_config_error()` already owns the config-error output path; `render.py` must not duplicate it.** A config error occurs *before* a `CliConfig` exists, so `ansi_colors` is not yet known and `render.py` cannot style it. Its format is byte-identical to `main._fail`, and it deliberately does not import `main`, to avoid a cycle.
- 1.2: **BINDING ON 2.1 - degenerate `[tags]` entries are accepted silently.** `alpha = []` yields `()` and `alpha = ""` yields `('',)`; a list of blank strings likewise. Requirements 3.2/3.4 do not say what a tag expanding to nothing, or to a blank folder name, should do - and 2.1 compiles the pushdown filter, so it is 2.1's call. A tag resolving to `('',)` would build a filter matching everything or nothing depending on the SQL. Decide it there rather than leaving it to chance.
- 1.2: design §3.1 says the config walk stops **"up to git root"**; the implementation walks the **full parent chain**, reusing the `(origin, *origin.parents)` shape of `npu_rag.embedding.models.acquire.find_dotenv` byte-identically. A git-root stop needs a git dependency or a `.git` sniff, and would make a config unreachable precisely where the archive is not a repository - which is the product. Accepted on review, but it has a cost: **the walk reaches the filesystem root, so a `npu-rag.toml` anywhere above `tmp_path` pollutes tests.** Planting one at `%TEMP%` broke 7 tests. `find_config_file()` and `load_config()` therefore take an optional `stop: Path | None` that bounds the walk **inclusively**; production stays unbounded, tests pass `stop=tmp_path`. Verified by a four-way experiment: polluted+bound passes, polluted+unbound fails 5.
- 1.2: **a mutation score offered in place of a RED phase must state its coverage surface, or it is unfalsifiable.** Round 1 reported "24 mutants, zero survivors" and review then found **9 real survivors**, all clustered in value-validation guards - because the 24 had probed the precedence and discovery core thoroughly and barely reached the coercers. The count was accurate and the claim was still misleading. The reviewer's own sweep missed the `[tags]` value-type guard for the same reason. **Enumerate the properties a sweep probed and name those it did not reach**; both gaps would have surfaced before review.
- 1.1: `tests/**/__init__.py` markers are mandatory (commit `e828554`); `tests/cli/__init__.py` was added with the rest. Five test basenames already collide across sibling directories, and mypy aborts with "Duplicate module named ..." **before checking anything** — it silently checked zero files for nine days. Any new test directory needs a marker.
- 2.1: **BINDING ON ANY TASK THAT EMITS SQL - lancedb 0.38.0 has no parameterisation.** `where()` takes `Union[str, Expr]`; a tuple/dict/int is rejected by a pydantic validator, and `Expr` exposes no `LIKE` (`and_, cast, contains, eq, gt, gte, isin, lower, lt, lte, ne, or_, to_sql, upper`), so requirement 3.1's `*`/`?` globs **cannot** be expressed without emitting SQL text. Escaping is the only option, and its dialect is narrow: `'O''Brien'` works while `'O\'Brien'` is a hard parse error ("Unterminated string literal"), and `ESCAPE` accepts **only** backslash (`ESCAPE '!'` raises). Anything building a predicate must reuse `tags.py`'s quoting rather than re-deriving it.
- 2.1: **BINDING ON 2.2 - `file_path` is `str(Path)`, so it is backslash-separated on Windows.** A `--folder "Parent/Child"` spec would have returned a silent empty result; the fix emits both separator spellings per folder. `order.py` parses folder identity out of the same column and must not assume `/`. One third of every predicate is consequently dead on this platform and **no test here can notice if the `/` branch breaks** - it was verified only by constructing POSIX-style values by hand.
- 2.1: **a folder filter needs a component-boundary anchor, not a bare substring.** Design §3.2's unanchored `LIKE '%spec%'` let `--folder "Vol 1"` match `Vol 10` and `--folder "Parent"` match the *file* `Parent.md`. The predicate now requires a separator after the match. Residual, inherited from the design and unaddressed by any requirement: the term matches any path component **including ancestors of the archive root**, so a folder named like an ancestor directory selects everything.
- 2.1: resolves the 1.2 note - degenerate `[tags]` entries are **errors** (`EmptyTagError(ConfigError)`, exit 1), raised **lazily** at `resolve_tag`. Silence was worse in a way the user cannot see: a blank folder compiles to `LIKE '%%'`, so a flag whose only job is to narrow would widen to the whole archive. Lazy because a `TagRegistry` is built on every command - eager validation would break an unrelated `--folder` search - and because 4.2 must still be able to *list* the broken entry.
- 2.1: **test the predicate through the route it is used in, not only through a scan.** Every test ran `table.search().where(...)` - a plain scan - while production reaches it via the FTS prefilter, the ANN prefilter and `SearchOptions.filter_sql`. This is the exact "both halves tested, the join untested" shape that produced a confirmed defect in `npu-embedding-runtime` 5.4. Review closed it: 9 predicates x 7 routes agreed with the scan, including against a real IVF_PQ index so the ANN planner parsed it, and a 200-folder / 37,011-char expansion. Clean - but it was closed in review, not in the task.
- 2.1: **FOR 4.1** - two docstring sentences in `src/npu_rag/cli/tags.py` overstate their guarantee and should be corrected when 4.1 touches the file. (a) `:24-27`/`:367-371` claim parenthesising `--filter` means "its operators cannot escape their clause"; an *unbalanced* value such as `chunk_index = 1) OR (1=1` still defeats it by AND-over-OR precedence. Not a security boundary - requirement 5.1 grants arbitrary SQL over the user's own index - but the sentence is wrong. (b) `:208-209` claims the `author` equality "prunes better than a LIKE"; OR-ed to a non-sargable `LIKE '%...%'` the planner scans regardless.
