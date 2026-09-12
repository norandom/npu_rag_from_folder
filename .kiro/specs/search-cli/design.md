# Design Document: search-cli

## 1. Overview & Architecture

The `search-cli` package is the user-facing operational surface of `npu_rag_from_folder`. It provides the `npu-rag` command-line executable, orchestrating the underlying ingest, embedding, and vector storage engines.

### Architectural Principles
1. **Folder-First Organization**: Everything is indexed, scoped, and displayed through the primary lens of the **folder** (the notebook / document collection).
2. **Tag Taxonomy**: Folders are classified into high-level subject domains ($N$ folders per tag, many-to-many) via a declarative `[tags]` section in `npu-rag.toml`, resolved at query time into database pushdown filters with zero re-indexing cost.
3. **Folder Order Preservation**: Inside folders representing courses, sadhanas, or lecture series, search hits are presented in chronological/session narrative sequence rather than disorienting random similarity shards.
4. **Hardware Accountability**: Explicit reporting of the executing processor (`VitisAIExecutionProvider` vs `CPUExecutionProvider`) with zero silent fallback.
5. **Fast Startup**: Zero heavy imports (`torch`, `onnxruntime`, `transformers`, `lancedb`) during top-level entry, parsing, `--help`, or `tags` listing.

---

## 2. Component Layout & Directory Structure

```
src/npu_rag/cli/
├── __init__.py          # Public package export
├── main.py              # Entry point, argparse setup, lazy dispatch
├── config.py            # TOML config, env var, and flag precedence resolver
├── tags.py              # Tag taxonomy registry, folder globs, and SQL filter builder
├── order.py             # Folder chronological and session sequence sorting
├── render.py            # ANSI terminal rendering, breadcrumb formatting, and JSON output
├── cmd_search.py        # Search command execution handler
├── cmd_index.py         # Index command execution handler and progress display
├── cmd_status.py        # Index health and statistics command handler
├── cmd_tags.py          # Tag registry inspection command handler
└── cmd_benchmark.py      # Benchmark runner delegating to embedding benchmark
```

---

## 3. Detailed Component Design

### 3.1 `config.py` — Precedence & Configuration Model

Resolves configuration with strict priority:
**CLI Arguments > Environment Variables (`NPU_RAG_*`) > `npu-rag.toml` > Defaults**

```python
@dataclass(frozen=True)
class CliConfig:
    archive_path: Path
    index_path: Path
    model_name: str
    provider: str  # "npu", "cpu", "auto"
    tags: dict[str, tuple[str, ...]]
    default_mode: str
    default_limit: int
    ansi_colors: bool
```

`load_config()` uses `tomllib` to discover and parse `npu-rag.toml` from the current working directory, parent directories up to git root, or `~/.config/npu-rag/config.toml`.

### 3.2 `tags.py` — Tag Taxonomy & Pushdown Filter Builder

Translates user `--tag` and `--folder` inputs into LanceDB SQL WHERE expressions.

```python
class TagRegistry:
    def __init__(self, tags: dict[str, Sequence[str]]) -> None:
        self._tags: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in tags.items()}

    def resolve_tag(self, tag_name: str) -> tuple[str, ...]:
        """Return list of folder names/patterns for a tag."""
        if tag_name not in self._tags:
            raise UnknownTagError(f"Unknown tag '{tag_name}'. Available: {list(self._tags.keys())}")
        return self._tags[tag_name]

    def build_filter_sql(
        self,
        *,
        folder: str | None = None,
        tags: Sequence[str] | None = None,
        custom_filter: str | None = None,
    ) -> str | None:
        """Compile a combined LanceDB SQL predicate."""
        predicates: list[str] = []
        if folder:
            predicates.append(f"(author = '{folder}' OR file_path LIKE '%{folder}%')")
        if tags:
            tag_patterns = []
            for t in tags:
                tag_patterns.extend(self.resolve_tag(t))
            sub = " OR ".join(f"(author = '{p}' OR file_path LIKE '%{p}%')" for p in tag_patterns)
            predicates.append(f"({sub})")
        if custom_filter:
            predicates.append(f"({custom_filter})")

        return " AND ".join(predicates) if predicates else None
```

### 3.3 `order.py` — Folder Chronological & Narrative Ordering

Provides natural sequence keys for documents and chunks within a folder:

```python
def natural_sort_key(file_path: str) -> tuple[Any, ...]:
    """Parse file path into chronological / session tokens (e.g. 'Day 1', 'Part 2', '2026-03-29')."""
    ...

def group_and_order_results(
    results: Sequence[SearchResult],
) -> dict[str, list[SearchResult]]:
    """Group SearchResult items by folder, sorted by natural document sequence and chunk ordinal."""
    grouped: dict[str, list[SearchResult]] = {}
    for res in results:
        folder = _extract_folder(res.file_path)
        grouped.setdefault(folder, []).append(res)

    for folder, items in grouped.items():
        items.sort(key=lambda r: (natural_sort_key(r.file_path), r.chunk_index))
    return grouped
```

### 3.4 `render.py` — Terminal Output & JSON Formatter

- **Windows Console**: Always invokes `sys.stdout.reconfigure(encoding="utf-8")` at startup.
- **TTY & Color**: If `not sys.stdout.isatty()` or `NO_COLOR` environment variable is set or `--no-color` is passed, strips ANSI escapes.
- **Visual Output**:
  ```text
  ================================================================================
  [Folder: Bhagwati Ugra Tara Sadhana] (Tag: siddha)
  ================================================================================
    ▶ [Doc 1/13: Savarni Nath - First Day.md] [Chunk #12 | Lines 45-80] [Score: 0.892]
      "पवित्रीकरण जल छिड़क लीजिए... ॐ अपवित्रः पवित्रो वा..."
      
    ▶ [Doc 1/13: Savarni Nath - First Day.md] [Chunk #13 | Context Continuation]
      "अब आचमनी में जल लीजिए और मेरे साथ कहिए ओम आम आम ह्रीम..."
  ```
- **JSON Output**: Emits clean, machine-readable JSON matching `SearchResponse` schema when `--json` is supplied.

### 3.5 Subcommands

#### `npu-rag search <query>`
- Flags: `--folder`, `--tag`, `--group-by-folder`, `--context`, `--mode`, `--metric`, `--fusion`, `--rrf-k`, `--linear-alpha`, `--mmr`, `--max-chunks-per-doc`, `--rollup`, `--top-k`, `--fetch-k`, `--filter`, `--provider`, `--json`, `--no-color`.
- Execution:
  1. Lazy-loads `VectorIndex` and embedder.
  2. Compiles `filter_sql` via `TagRegistry`.
  3. Executes `index.search()`.
  4. If `--group-by-folder`, routes through `order.group_and_order_results()`.
  5. Renders terminal or JSON output.

#### `npu-rag index`
- Flags: `--archive-path`, `--full-rebuild`, `--provider`, `--json`.
- Execution:
  1. Lazy-loads `document-ingest`, `IndexSynchronizer`, and embedder.
  2. Runs discovery and sync, reporting live console updates (file count, chunk count, throughput).
  3. Emits final summary report.

#### `npu-rag status`
- Inspects LanceDB table metadata and reports chunk counts, model dimension, last indexed timestamp, and omissions.

#### `npu-rag tags`
- Lists all tags in `npu-rag.toml`, their mapped folders, and the count of chunks currently indexed in each folder.

#### `npu-rag benchmark`
- Delegates directly to `npu_rag.embedding.benchmark` harness.

---

## 4. Error Taxonomy & Exit Codes

| Exit Code | Condition | Behavior |
| :--- | :--- | :--- |
| `0` | Success | Clean output rendered to stdout. |
| `1` | Configuration or User Error | Missing folder, unknown tag, malformed config, invalid argument values. Prints actionable diagnostic to stderr. |
| `2` | Hardware / Runtime Error | Requested NPU provider is unavailable, model compilation failure, corrupted index database. Prints diagnostic to stderr. |

---

## 5. Testing & Verification Plan

1. **`test_cli_baseline.py`**: Entry point verification, `--help`, `--version`, argument parsing, and verify zero heavy imports (< 50 ms startup).
2. **`test_cli_config.py`**: Precedence resolution: CLI flags > Env vars > TOML file > Defaults.
3. **`test_cli_tags.py`**: Tag taxonomy resolution, glob matching, unknown tag error handling, SQL pushdown compilation.
4. **`test_cli_order.py`**: Natural sort key extraction, chronological session sequencing, and folder grouping.
5. **`test_cli_render.py`**: ANSI color handling, non-TTY stripping, Windows UTF-8 formatting, and JSON serialization.
6. **`test_cli_commands.py`**: End-to-end integration tests for `search`, `index`, `status`, `tags`, and `benchmark` commands using simulated or real index stores.
