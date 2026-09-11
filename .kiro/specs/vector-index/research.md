# Research & Design Decisions: vector-index

## Summary
- **Feature**: `vector-index`
- **Discovery Scope**: New Feature (greenfield package `npu_rag.index`) providing durable vector + lexical storage, hybrid search, ranking/fusion, diversity, context expansion, and a unified query orchestration API for downstream CLI and MCP surfaces.
- **Key Findings**:
  - `lancedb` (0.38.0) installs and runs cleanly in our Python 3.12 environment under the `numpy<2` restriction with zero conflict with AMD ONNX Runtime vendor wheels.
  - Storage is file-backed and memory-mapped on local disk; exact flat search over 10k–30k 768-dim float32 vectors executes in ~2 ms on CPU with 100% recall and no stale-index drift on incremental updates.
  - LanceDB's full-text search is native (Apache Lance FTS); `use_tantivy` is deprecated/dropped. Modern index creation uses `table.create_index("text", config=lancedb.index.FTS(), replace=True)`.
  - Reciprocal Rank Fusion (RRF) with constant $k=60$ merges dense cosine distance rankings and sparse BM25 scores without requiring calibration or normalization across disparate score scales.
  - Query vector generation requires the upstream embedding model's asymmetric query prefix template (`embed_query(query)` from `npu-embedding-runtime`), distinct from the document template.

---

## Research Log

### 1. LanceDB 0.38.0 API & Windows Compatibility
- **Context**: Embedded vector storage without external daemons or heavy runtime requirements, suitable for in-process `uvx` execution.
- **Sources Consulted**: LanceDB 0.38 documentation, local spike script [`spike_lancedb.py`](file:///C:/Users/mariu/.gemini/antigravity-cli/brain/c1225090-f990-45f4-a25d-0c53a95a0485/scratch/spike_lancedb.py).
- **Findings**:
  - `lancedb.connect(uri)` connects directly to a local directory path (supports standard Windows paths).
  - Schema can be defined cleanly via `lancedb.pydantic.LanceModel` with `vector: Vector(768)` or via PyArrow Schema.
  - Record insertion via `table.add(...)` supports dicts, Pydantic objects, and Arrow tables.
  - Table search: `table.search(query_vector).metric("cosine").limit(k).where(filter_sql).to_list()` runs fast and returns dictionary records containing `_distance`.
  - Record deletion: `table.delete(predicate)` supports SQL predicates like `chunk_id = 'c1'` or `file_path = '...'`.
- **Implications**: LanceDB is the ideal embedded engine: zero daemon overhead, memory-mapped storage with flat resident RAM footprint, and clean Pydantic integration.

### 2. Native Full-Text Search (FTS)
- **Context**: Lexical BM25 search over chunk text to complement dense semantic embeddings for exact terms, names, and numbers.
- **Findings**:
  - `table.create_fts_index(...)` was deprecated as of 0.25.0 in favor of `table.create_index("text", config=lancedb.index.FTS(), replace=True)`.
  - Querying: `table.search(query_text, query_type="fts").limit(k).to_list()` returns records scored by BM25 under `_score`.
  - No Tantivy C++ binaries or external search engines are required.
- **Implications**: The package will use `lancedb.index.FTS()` directly. When new chunks are indexed or deleted, the FTS index can be created or updated.

### 3. Hybrid Search & Fusion: RRF vs. Linear Combination
- **Context**: Dense retrieval yields semantic neighbors but misses exact keywords; BM25 catches exact identifiers but misses paraphrases. Combining them is essential.
- **Findings**:
  - Dense cosine search produces distance $d \in [0, 2]$ (where 0 is identical for unit-norm vectors).
  - BM25 produces unbounded positive relevance scores.
  - Reciprocal Rank Fusion (RRF):
    $$RRF(d) = \sum_{m \in \{dense, fts\}} \frac{w_m}{k + \text{rank}_m(d)}$$
    Standard constant $k = 60$. RRF relies solely on rank order, completely eliminating score-calibration errors.
  - Linear combination requires normalising scores (e.g. min-max or sigmoid scaling) and tuning a weight parameter $\alpha \in [0, 1]$.
- **Implications**: RRF is the robust default ($k=60$, $w_{dense}=1.0$, $w_{fts}=1.0$). Linear combination is supported as an alternative scoring mode.

### 4. Diversity Control: Maximal Marginal Relevance (MMR) & Document Rollup
- **Context**: A Substack post produces 4–8 consecutive chunks. Naive top-10 retrieval frequently returns 4–5 chunks from a single article.
- **Findings**:
  - **MMR (Maximal Marginal Relevance)**: Given candidates $R$ and selected set $S$, select:
    $$d^* = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$
    where $\lambda \in [0, 1]$ balances query relevance against novelty relative to already selected results.
  - **Document Rollup**: Grouping chunks by `file_path` (or `title`). The orchestrator can either select the top-scoring chunk per document (with a collapse count) or roll up multiple chunks under their parent document.
- **Implications**: Both MMR and Document Rollup will be implemented as post-retrieval ranking pipeline stages.

### 5. Neighbouring-Chunk Context Expansion
- **Context**: When a chunk matches a query, surrounding context (preceding and succeeding paragraphs or rows) often carries essential context (e.g. table headers, introductory definitions).
- **Findings**:
  - Ingest attaches `chunk_index: int` and `file_path: str` to every `ChunkRecord`.
  - When context expansion is requested with window size $w \ge 1$, the index queries the store for chunks with matching `file_path` and `chunk_index \in [\text{target} - w, \text{target} + w]$ and merges their text.
- **Implications**: Fast secondary lookup using the index's existing metadata fields, without re-embedding.

### 6. Upstream Contracts: Ingest & Embedding Runtime
- **Context**: Seamless handoff from `document-ingest` and `npu-embedding-runtime`.
- **Findings**:
  - `document-ingest` provides `ChunkRecord(chunk_id, file_path, file_hash, chunk_index, kind, title, headings, locator, text, token_count)`.
  - `npu-embedding-runtime` provides `EmbeddingService` with:
    - `embed_documents(docs)`: embeds corpus chunks with title slot populated.
    - `embed_query(query)`: embeds query string using the model's asymmetric query prefix!
- **Implications**: Ingest-to-index sync takes `IngestReport` and `EmbeddingService`, embeds new/changed chunks, writes them to LanceDB, and removes deleted chunk IDs.

---

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Verdict |
|---|---|---|---|---|
| **Pipelined Orchestration (Selected)** | Modular stages: `Store` (LanceDB) $\to$ `Retriever` (Dense, FTS) $\to$ `Fusion` (RRF, Linear) $\to$ `PostProcessor` (MMR, Rollup, Context) $\to$ `Reranker` | Clear boundaries, testable in isolation, customizable per query | Small composition overhead | **Selected** |
| **Monolithic Store** | Everything embedded in one massive class | Fewer files initially | Hard to test RRF or MMR in isolation without mocking LanceDB | Rejected |
| **Server / Daemon API** | Background HTTP / gRPC service managing LanceDB | Multi-process sharing | Fragile process supervision, conflicts with `uvx` constraint | Rejected |

---

## Design Decisions

1. **Storage Backend**: Embedded LanceDB table named `chunks` under a user-configured directory.
2. **Vector Dimension**: Pinned to 768 dimensions (matching `embeddinggemma-300m`, `gte-modernbert-base`, and `nomic-embed-text-v1.5`).
3. **Exact Search as Default**: Exact flat kNN search by default; opt-in ANN index creation for large archives.
4. **Native FTS**: Use `lancedb.index.FTS()`.
5. **Score Fusion**: Default to RRF ($k=60$).
6. **Query Embedding**: Mandatory use of `embed_query()` to ensure asymmetric prefixes are applied.
7. **Reranker Hook**: Protocol `Reranker(Protocol)` allowing pluggable cross-encoders with no-op default.

---

## Risks & Mitigations

- **LanceDB table lock on Windows**: LanceDB uses file locks. Ensure single-writer discipline and clean connection management.
- **Asymmetric query template omission**: Hard requirement that `VectorIndex` query path routes through `EmbeddingService.embed_query`.
- **Empty FTS search**: If query contains words with no lexical match, FTS returns empty list; RRF seamlessly handles uneven candidate sets.
