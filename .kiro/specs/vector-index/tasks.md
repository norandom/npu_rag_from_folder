# Implementation Plan: vector-index

- [x] 1. Foundation: Baseline, Dependencies, Types, and Boundaries
- [x] 1.1 Declare dependencies and establish package baseline
  - Add `lancedb>=0.38.0` to `pyproject.toml` dependencies; create `src/npu_rag/index/` package structure
  - Establish layer-order boundary test asserting `npu_rag.index` may import `npu_rag.embedding` and `npu_rag.ingest`, but neither upstream package imports `npu_rag.index`
  - Observable: `import npu_rag.index` succeeds, and boundary tests enforce clean one-way dependency direction
  - _Requirements: 1.1, 8.1_
  - _Boundary: package baseline_

- [x] 1.2 Define domain types, search options, and error taxonomy
  - Define `SearchMode` (DENSE_ONLY, LEXICAL_ONLY, HYBRID), `DistanceMetric` (COSINE, DOT, L2), `FusionMethod` (RRF, LINEAR)
  - Define `SearchOptions`, `SearchResult`, `SearchResponse`, `SyncSummary`, `IndexStats`
  - Define error hierarchy rooted in `IndexError` (`StoreError`, `QueryError`, `SyncError`)
  - Observable: all domain types, dataclasses, and error classes are importable, type-checked by mypy, and distinguishable by type
  - _Requirements: 1.1, 5.1, 8.1, 8.4_
  - _Boundary: types and errors_

- [x] 2. Schema and Durable Embedded Store
- [x] 2.1 Define LanceDB schema and record conversion
  - Implement `ChunkSchema` using `lancedb.pydantic.LanceModel` with 768-dim float32 `Vector` field
  - Implement conversion functions between `ChunkRecord` (from `document-ingest`) and `ChunkSchema`
  - Encode and decode locator structures (`locator_type`, `locator_json`) faithfully
  - Observable: converting a `ChunkRecord` to `ChunkSchema` and back preserves all fields and locator identity without loss
  - _Requirements: 1.2_
  - _Boundary: schema_

- [x] 2.2 Implement `LanceIndexStore` table operations and lifecycle
  - Connect to embedded LanceDB directory path; initialize or open `chunks` table
  - Support batch insert and atomic upsert keyed by `chunk_id`
  - Support deletion by chunk IDs, by `file_path`, and by SQL predicate
  - Implement table compaction and record statistics
  - Observable: inserted records persist across store re-open; deleting by `file_path` removes all chunks for that file; upserting updates existing `chunk_id`s in place
  - _Requirements: 1.1, 1.3, 1.4, 1.5_
  - _Boundary: store_

- [x] 2.3 Implement native FTS index lifecycle
  - Create and maintain Lance-native full-text index using `lancedb.index.FTS()` on the `text` field
  - Handle FTS index rebuild / refresh after batch updates
  - Observable: FTS index is created without tantivy warnings and indexes chunk text for lexical querying
  - _Requirements: 4.1_
  - _Boundary: store / fts_

- [x] 3. Retrieval Primitives
- [x] 3.1 Implement `DenseRetriever` with exact kNN and metric selection
  - Execute exact brute-force flat search over vectors with `cosine` (default), `dot`, and `l2` metrics
  - Support SQL metadata pre-filtering and post-filtering
  - Support opt-in ANN indexing parameters (`nprobes`, `refine_factor`)
  - Observable: dense retrieval returns top-k nearest neighbors sorted by distance; SQL filter strictly excludes non-matching rows
  - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - _Boundary: retriever_

- [x] 3.2 Implement `LexicalRetriever` with native BM25 FTS
  - Execute lexical BM25 queries over the LanceDB FTS index
  - Extract and normalize BM25 relevance scores
  - Return empty results gracefully when a query contains zero matching terms
  - Support SQL metadata filtering on lexical queries
  - Observable: searching for an exact keyword returns matching records with positive scores; querying a non-existent word returns `[]` without error
  - _Requirements: 4.2, 4.3, 4.4_
  - _Boundary: retriever_

- [x] 4. Hybrid Retrieval and Score Fusion
- [x] 4.1 Implement Reciprocal Rank Fusion (RRF)
  - Implement RRF algorithm merging dense rank and lexical rank with configurable constant $k$ (default 60) and weights
  - Handle asymmetric result sets (items appearing in only dense or only lexical candidates)
  - Observable: a document appearing in top positions of both candidate sets ranks highest; items appearing in only one set are scored correctly without failure
  - _Requirements: 5.2, 5.4_
  - _Boundary: fusion_

- [x] 4.2 Implement Linear Combination fusion
  - Implement min-max score normalisation for dense distances and BM25 scores
  - Compute weighted combination $\alpha \cdot s_{dense} + (1 - \alpha) \cdot s_{fts}$
  - Observable: normalized fusion produces scores in $[0, 1]$; changing $\alpha$ dynamically shifts weight between dense and lexical rankings
  - _Requirements: 5.3, 5.4_
  - _Boundary: fusion_

- [x] 5. Diversity Control and Context Expansion
- [x] 5.1 Implement Maximal Marginal Relevance (MMR) diversity reranking
  - Implement MMR algorithm balancing query similarity with pairwise candidate dissimilarity using $\lambda \in [0, 1]$
  - Observable: given clustered candidates with high mutual similarity, MMR promotes dissimilar candidates into top-k
  - _Requirements: 6.1_
  - _Boundary: diversity_

- [x] 5.2 Implement Document Rollup and per-document chunk capping
  - Implement grouping of chunk hits by source `file_path`
  - Support `max_chunks_per_document` cap (e.g. at most 2 chunks per article)
  - Support collapsed document-level hit aggregation
  - Observable: result list contains at most `max_chunks_per_document` entries per file; document rollup returns distinct articles
  - _Requirements: 6.2, 6.3_
  - _Boundary: diversity_

- [x] 5.3 Implement neighbouring-chunk context expansion
  - Look up adjacent chunks within window $[-w, +w]$ by `file_path` and `chunk_index`
  - Concatenate adjacent chunk text in reading order
  - Handle document boundary edge cases (chunk 0, final chunk) cleanly
  - Observable: a matched chunk with $w=1$ returns merged text of chunks $i-1$, $i$, and $i+1$; boundary chunks do not raise index errors
  - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - _Boundary: expansion_

- [x] 6. Synchronization Bridge
- [x] 6.1 Implement `IndexSynchronizer` from `IngestReport` and `EmbeddingService`
  - Purge removed chunk IDs specified in `IngestReport.removed_chunk_ids`
  - Skip unchanged files without re-embedding
  - Transform new and changed chunks to `DocumentText(content, title)`, embed via `embed_service.embed_documents`, and upsert into store
  - Rebuild or update FTS index and return `SyncSummary`
  - Observable: a second sync with an unchanged `IngestReport` calls `embed_documents` zero times; modified files are re-embedded and updated; deleted files are purged
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - _Boundary: sync_

- [x] 7. Query Orchestration and Public Surface
- [x] 7.1 Implement `VectorIndex` facade and public package API
  - Coordinate query embedding via `embed_service.embed_query(query)` (enforcing asymmetric query prefix)
  - Execute retrieval, fusion, reranking, diversity, and context expansion pipeline
  - Expose pluggable `Reranker` protocol with `PassThroughReranker` default
  - Export public API from `src/npu_rag/index/__init__.py`
  - Observable: calling `index.search(...)` executes full pipeline end-to-end and returns structured `SearchResponse`; verifying query embedding uses `embed_query`
  - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - _Boundary: service_

- [x] 7.2 Validate end-to-end search, sync, and diversity
  - Test end-to-end workflow: ingest document sample, embed and index, perform hybrid search with exact lexical queries and semantic queries
  - Validate MMR diversity and context expansion over real multi-chunk articles
  - Observable: hybrid search retrieves documents that both dense-only and lexical-only fail to rank at top; incremental re-run reuses index completely
  - _Requirements: 1.1, 2.1, 3.1, 4.2, 5.2, 6.1, 7.1, 8.1_
  - _Boundary: validation_

---

## Follow-on (post-closure amendments)

- [x] 8.1 Make the retrieval candidate depth configurable (`SearchOptions.fetch_k`)
  - Add `fetch_k: int | None = None` to `SearchOptions`; honour it in `VectorIndex.search`, falling back to the derived `max(limit * 4, 30)` when `None`.
  - *Files:* `src/npu_rag/index/types.py`, `src/npu_rag/index/service.py`, `tests/index/test_service.py`
  - **Why this is an amendment and not a new feature.** `design.md` section 3 already writes `fetch_k` into the retrieval pseudocode (lines 118 and 122), but the implementation computed it inline at `service.py:142` with **no seam** - no subclass, property or keyword could reach it. `search-cli`'s requirement 5.1 lists a `--fetch-k` flag, so task 4.1 there had a choice between advertising a flag that silently did nothing and patching this closed spec. It did neither: it omitted the flag, pinned the omission with a test, and routed the gap here. This closes it.
  - **Compatibility.** `None` is the default and reproduces the previous behaviour exactly, so no existing caller changes. The floor of 30 is deliberately **not** applied to an explicit value: `max(30, fetch_k)` would look correct for every value above 30 and would quietly turn the parameter into a suggestion, when a caller asking for a pool of 3 on a large index means 3.
  - Implemented 2026-10-11. Five mutants planted and all killed (ignore the field entirely; apply the floor to an explicit value; drop the floor; drop the product; test falsiness rather than `None`, which would silently reinterpret a programmatic `fetch_k=0`), each restored and SHA256-verified. Both arms of the derived default are asserted separately, since a mutant dropping either one still satisfies the other.

### Known gap, not yet scheduled

- **The index records no embedding-model identity.** `ChunkSchema` has no `model_id`/`model_name` column, and `VECTOR_DIMENSION` is a module constant. A *dimension* change fails at write time, but a **same-dimension different model does not**: EmbeddingGemma 2 is also 768-d, so repointing `model_name` at it over an existing index would silently produce one table holding two incompatible vector spaces - no error, no way to tell the rows apart afterwards, and every cross-boundary similarity score meaningless but plausible. A `model_id` column or a sidecar manifest is a precondition for any model migration. Raised 2026-10-11 during `search-cli` 4.2 preflight.
- LANCEDB SCAN LIMITS, measured 2026-10-11 on 0.38.0 (relevant to 5.1 and to any full-table read): **a plain scan is not truncated; a vector query is.**
  - `table.search().select([...]).to_arrow()` with **no** `.limit()` returned **2000 of 2000** rows - and still 2000 after an FTS index and an IVF_PQ index were both created.
  - `table.search(query_vector).to_arrow()` with no `.limit()` returned **10**.
  So the default-10 belongs to the **vector-query** path, not to the scan path. `DenseRetriever.retrieve` always passes `limit=fetch_k` explicitly, so it is unaffected.
  **Consequence:** `sync.py:97` (the `chunk_id`/`text` read that classifies new vs updated vs unchanged) and `sync.py:163` (the `file_path` read behind `files_indexed`) have **no** explicit limit and are **safe** - incremental sync is *not* silently wrong above ten rows, which was the worry raised when 8.2 landed. Verified by direct measurement rather than by reading the lancedb docs. Do not "fix" those two call sites on the assumption that an unbounded scan truncates; do, however, keep passing an explicit limit on any **vector** query.
- 8.2 RESOLVED a worry it had itself raised: `sync.py:97` and `:163` scan with no explicit limit, and that is **correct**. Measured on lancedb 0.38.0 - an unlimited plain scan returns every row (42/42 and 2000/2000); the default of 10 belongs to the *vector-query* path only. Recorded so the next reader does not re-raise it. The converse is the real lesson: `model_identity_counts` originally computed `len(table)` and passed it as `.limit(total)`, which bought nothing and **introduced** a check-then-read window in which a table growing between the two calls is scanned short - letting the identity gate pass a mixed index because the second model's rows fell outside the page. Removing the limit removed the bug. Defensive code is not free.
