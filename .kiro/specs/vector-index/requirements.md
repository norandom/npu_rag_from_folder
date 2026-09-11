# Requirements Document: vector-index

## Introduction

This feature provides the durable vector and lexical storage, hybrid retrieval, ranking fusion, diversity control, and unified query orchestration for the NPU-accelerated RAG system. It bridges the output of `document-ingest` (chunked records) and `npu-embedding-runtime` (dense embeddings), persisting them in an embedded LanceDB database on local disk and exposing a single, consistent search API to the downstream CLI (`search-cli`) and MCP server (`mcp-server`).

Its core mission is to make personal document archives reliably searchable through both semantic concepts ("incentive mechanisms in open source") and exact lexical terms (names, formulas, numbers, jargon), while ensuring search results are diverse across documents rather than clustered in a single article.

---

## Boundary Context

- **In scope**:
  - Embedded LanceDB schema, lifecycle, connection management, and compaction.
  - Ingest-to-index synchronization bridge taking `IngestReport` and `EmbeddingService`.
  - Dense vector retrieval with exact flat search (default) and opt-in ANN indices.
  - Native LanceDB full-text search (FTS / BM25) over chunk text.
  - Hybrid retrieval with Reciprocal Rank Fusion (RRF, default) and Linear Combination.
  - Diversity ranking via Maximal Marginal Relevance (MMR) and document-level rollup.
  - Neighbouring-chunk context expansion (windowed retrieval around matched chunk ordinals).
  - SQL metadata pre-filtering and post-filtering.
  - Pluggable cross-encoder reranking hook.
  - Unified `VectorIndex` query orchestration API.

- **Out of scope**:
  - Model acquisition, ONNX export, and NPU execution (owned by `npu-embedding-runtime`).
  - Document parsing, format detection, OCR, and chunk budget sizing (owned by `document-ingest`).
  - User-facing CLI interface and flags (owned by `search-cli`).
  - MCP JSON-RPC / SSE server endpoints (owned by `mcp-server`).
  - Third-party cloud rerankers or hosted search services (retrieval is 100% local and offline).

---

## Requirements

### Requirement 1: Schema and Embedded Storage

**Objective:** As the archive owner, I want my chunks and embeddings stored durably in an embedded database on my local disk with zero background daemon overhead, so that the index is immediately accessible and memory-efficient.

#### Acceptance Criteria

1. The Vector Index shall connect to an embedded LanceDB database at a configurable directory path without requiring any background service or network socket.
2. The Vector Index shall maintain a primary table named `chunks` storing the following fields:
   - `chunk_id` (string, unique primary identifier)
   - `file_path` (string, canonical relative path of source document)
   - `file_hash` (string, SHA256 content hash of source document)
   - `chunk_index` (integer, 0-based chunk ordinal within source document)
   - `chunk_kind` (string, e.g. `prose`, `table`, `figure`)
   - `title` (nullable string, document or article title)
   - `headings` (list of strings, hierarchical heading trail)
   - `locator_type` (string, locator discriminator)
   - `locator_json` (string, serialized locator payload)
   - `text` (string, extracted text content)
   - `vector` (fixed-size float32 array of 768 dimensions)
   - `token_count` (integer, measured token length)
   - `created_at` (float, UTC timestamp)
3. When upserting chunks, the Vector Index shall write new records and update existing records matching `chunk_id` atomically.
4. When deleting records, the Vector Index shall support deletion by list of `chunk_id`s, by `file_path`, and by SQL predicate.
5. The Vector Index shall provide a compaction operation that optimizes table data files and cleans up superseded version fragments.

---

### Requirement 2: Ingest and Embedding Synchronization

**Objective:** As the archive owner, I want a synchronization bridge between document ingest and the vector index, so that new and changed files are embedded and indexed incrementally while unchanged files incur zero re-embedding cost.

#### Acceptance Criteria

1. The Vector Index shall provide a synchronization entry point accepting an `IngestReport` and an `EmbeddingService`.
2. For files marked unchanged in the `IngestReport`, the synchronization bridge shall perform no re-embedding and make no modifications to their existing index records.
3. For new and changed files in the `IngestReport`, the synchronization bridge shall construct `DocumentText(content=chunk.text, title=chunk.title)` records, generate embeddings via `EmbeddingService.embed_documents(...)`, and write the resulting records to the index.
4. For files deleted from the archive (or chunk IDs present in `IngestReport.removed_chunk_ids`), the synchronization bridge shall purge their records from the index.
5. The synchronization bridge shall return a `SyncSummary` reporting counts of chunks added, chunks updated, chunks deleted, and total records remaining in the index.

---

### Requirement 3: Dense Vector Retrieval

**Objective:** As a search consumer, I want fast, exact semantic vector retrieval by default, with opt-in approximate nearest neighbor indexing for large archives.

#### Acceptance Criteria

1. The Vector Index shall perform exact flat (brute-force) kNN vector search by default, guaranteeing 100% recall over stored vectors.
2. The Vector Index shall support configurable distance metrics including `cosine` (default), `dot`, and `l2`.
3. The Vector Index shall support optional ANN index creation (such as `IVF_PQ` or `IVF_HNSW_SQ`) with configurable `nprobes` and `refine_factor` tuning.
4. The Vector Index shall allow filtering by SQL metadata predicates during dense search, supporting both pre-filtering (filtering candidate set prior to vector comparison) and post-filtering.

---

### Requirement 4: Native Full-Text Lexical Retrieval

**Objective:** As a search consumer, I want exact keyword matching using BM25, so that specialized names, code symbols, formulas, and numeric identifiers can be found precisely.

#### Acceptance Criteria

1. The Vector Index shall create and maintain a Lance-native Full-Text Search (FTS) index over the `text` field using `lancedb.index.FTS()`.
2. The Vector Index shall provide a lexical retrieval mode that executes BM25 search over the FTS index and returns ranked chunks with relevance scores.
3. The Vector Index shall support metadata filtering in conjunction with FTS queries.
4. If a lexical query matches no records in the FTS index, the lexical retriever shall return an empty list without raising an error.

---

### Requirement 5: Hybrid Retrieval and Score Fusion

**Objective:** As a search consumer, I want semantic and lexical results combined seamlessly, so that queries with mixed conceptual and exact terms return optimal rankings.

#### Acceptance Criteria

1. The Vector Index shall support three search modes: `DENSE_ONLY`, `LEXICAL_ONLY`, and `HYBRID` (default).
2. Under `HYBRID` mode, the Vector Index shall execute dense vector search and lexical FTS search, and combine the candidate lists using Reciprocal Rank Fusion (RRF) by default:
   $$RRF(d) = \frac{w_{dense}}{k + \text{rank}_{dense}(d)} + \frac{w_{fts}}{k + \text{rank}_{fts}(d)}$$
   where $k$ is configurable with default value 60, and $w_{dense}, w_{fts}$ are configurable weights with default 1.0.
3. The Vector Index shall support an alternative linear combination fusion mode with score normalization and configurable weight $\alpha \in [0, 1]$.
4. The fused result list shall be ranked in descending order of combined score and truncated to the requested limit $k$.

---

### Requirement 6: Result Diversity and Document Rollup

**Objective:** As a search consumer, I want diverse results across different source documents, so that the top results are not dominated by multiple chunks from a single document.

#### Acceptance Criteria

1. The Vector Index shall implement Maximal Marginal Relevance (MMR) ranking:
   $$d^* = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$
   with configurable diversity factor $\lambda \in [0, 1]$ (default 0.7).
2. The Vector Index shall support document-level rollup, grouping matched chunks by `file_path` and returning either the single highest-scoring chunk per document or an aggregated document hit.
3. The Vector Index shall support a configurable `max_chunks_per_document` limit, capping how many chunks from any single document may appear in the final result list.

---

### Requirement 7: Neighbouring-Chunk Context Expansion

**Objective:** As a search consumer, I want the option to view surrounding context around a retrieved chunk, so that sentences split by chunk boundaries or dependent context can be understood.

#### Acceptance Criteria

1. The Vector Index shall support context expansion by a configurable window size $w \ge 1$.
2. When context expansion is enabled, the Vector Index shall retrieve adjacent chunks with the same `file_path` and `chunk_index \in [\text{target} - w, \text{target} + w]$.
3. The Vector Index shall assemble the expanded context text in proper reading order, while maintaining provenance and distinct identification of the primary matched chunk.
4. If a matched chunk is at the beginning or end of a document, context expansion shall retrieve available adjacent chunks without error or out-of-bounds indexing.

---

### Requirement 8: Unified Query Orchestration API

**Objective:** As a developer building the CLI or MCP server, I want a single, cohesive query orchestration API, so that search behavior and ranking logic are identical across all interfaces.

#### Acceptance Criteria

1. The Vector Index shall provide a unified `VectorIndex` facade class exposing:
   - `search(query: str, options: SearchOptions) -> SearchResponse`
   - `sync(ingest_report: IngestReport, embed_service: EmbeddingService) -> SyncSummary`
   - `stats() -> IndexStats`
2. For all queries requiring vector comparison, the Vector Index shall generate the query embedding strictly via `EmbeddingService.embed_query(query)`, ensuring the model's asymmetric query template is applied.
3. The Vector Index shall provide a `Reranker` protocol enabling optional cross-encoder reranking over the fused candidate set, with a default pass-through implementation when no reranker is supplied.
4. The `SearchResponse` shall contain ranked `SearchResult` items, each providing:
   - `chunk_id`, `file_path`, `title`, `chunk_kind`, `chunk_index`, `headings`, `locator`
   - `score` (and individual sub-scores: dense score, lexical score, fusion score)
   - `text` (primary chunk text)
   - `expanded_context` (assembled surrounding context text when requested)
