# Design Document: vector-index

## Architecture Overview

`vector-index` (implemented under `src/npu_rag/index/`) is the central storage and retrieval engine of the local RAG system. It sits downstream from `document-ingest` and `npu-embedding-runtime`, and upstream from `search-cli` and `mcp-server`.

```
                  ┌──────────────────────┐
                  │   document-ingest    │
                  └──────────┬───────────┘
                             │ IngestReport (ChunkRecords)
                             ▼
 ┌──────────────────────┐  ┌───────────────────────┐
 │npu-embedding-runtime │─▶│   IndexSynchronizer   │
 └──────────────────────┘  └───────────┬───────────┘
   (EmbeddingService)                  │ Upsert / Delete
                                       ▼
                             ┌───────────────────┐
                             │  LanceIndexStore  │
                             │ (Embedded LanceDB)│
                             └─────────┬─────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         ┌────────────────────┐                ┌─────────────────────┐
         │   DenseRetriever   │                │  LexicalRetriever   │
         │(Exact kNN / Cosine)│                │     (Native FTS)    │
         └──────────┬─────────┘                └──────────┬──────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                             ┌───────────────────┐
                             │  FusionPipeline   │
                             │   (RRF / Linear)  │
                             └─────────┬─────────┘
                                       ▼
                             ┌───────────────────┐
                             │ DiversityPipeline │
                             │  (MMR / Rollup)   │
                             └─────────┬─────────┘
                                       ▼
                             ┌───────────────────┐
                             │  ContextExpander  │
                             │(Windowed Context) │
                             └─────────┬─────────┘
                                       ▼
                             ┌───────────────────┐
                             │  Reranker Hook    │
                             │   (PassThrough)   │
                             └─────────┬─────────┘
                                       ▼
                             ┌───────────────────┐
                             │   SearchResponse  │
                             └───────────────────┘
```

---

## Component Layout & Modules

```
src/npu_rag/index/
├── __init__.py           # Public exports (VectorIndex, SearchOptions, SearchResponse, SearchResult)
├── types.py              # Domain types, Enums (SearchMode, Metric, FusionMethod), dataclasses
├── errors.py             # Error hierarchy (IndexError, StoreError, QueryError, SyncError)
├── schema.py             # LanceModel definitions & ChunkRecord conversion
├── store.py              # LanceIndexStore: LanceDB connection, table ops, FTS index, compaction
├── sync.py               # IndexSynchronizer: IngestReport + EmbeddingService -> store sync
├── retriever.py          # DenseRetriever & LexicalRetriever implementations
├── fusion.py             # ReciprocalRankFusion & LinearCombination
├── diversity.py          # MaximalMarginalRelevance (MMR) & DocumentRollup
├── expansion.py          # ContextExpander: windowed adjacent chunk expansion
├── rerank.py             # Reranker Protocol & PassThroughReranker
└── service.py            # VectorIndex facade coordinating all stages
```

---

## Schema Design

### LanceDB Table: `chunks`

Defined via `lancedb.pydantic.LanceModel`:

```python
from lancedb.pydantic import LanceModel, Vector

class ChunkSchema(LanceModel):
    chunk_id: str                   # Primary identifier (e.g. SHA256 of file+kind+ordinal)
    file_path: str                  # Canonical relative document path
    file_hash: str                  # Content hash of source file
    chunk_index: int                # 0-based chunk ordinal within source file
    chunk_kind: str                 # "prose", "table", "figure", etc.
    title: str | None = None        # Document/article title (or None)
    headings: list[str] = []        # Hierarchical heading trail
    locator_type: str               # "markdown", "page", "sheet", "image_file"
    locator_json: str               # JSON-serialized locator dict
    text: str                       # Raw text content for FTS and snippets
    vector: Vector(768)             # 768-dim float32 embedding vector
    token_count: int                # Token length measured by upstream tokenizer
    created_at: float               # Unix epoch timestamp
```

---

## Retrieval & Query Pipeline

### 1. Vector Generation
- For any query in `DENSE_ONLY` or `HYBRID` mode, the query vector is produced via:
  ```python
  query_vector = embed_service.embed_query(query)
  ```
  This is required to apply the model's asymmetric query prefix (e.g. `query: ` or `Given a search query...`).

### 2. Candidate Retrieval
- **Dense Retriever**:
  ```python
  results = table.search(query_vector).metric(metric).where(filter_sql).limit(fetch_k).to_list()
  ```
- **Lexical Retriever**:
  ```python
  results = table.search(query, query_type="fts").where(filter_sql).limit(fetch_k).to_list()
  ```
- If an FTS query produces 0 hits, it returns an empty list without raising.

### 3. Score Fusion
- **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \frac{w_{dense}}{k + \text{rank}_{dense}(d)} + \frac{w_{fts}}{k + \text{rank}_{fts}(d)}$$
  Default $k = 60$. If a document appears in only one candidate list, the other rank component is 0.
- **Linear Combination**:
  Normalizes scores to $[0, 1]$ and computes $\alpha \cdot s_{dense} + (1 - \alpha) \cdot s_{fts}$.

### 4. Diversity Control (MMR & Rollup)
- **MMR**:
  Calculates cosine similarity between candidate vectors and the query vector, and pairwise among selected candidates.
  Iteratively selects candidates maximizing marginal relevance until `limit` is reached.
- **Document Rollup / Capping**:
  Limits results to at most `max_chunks_per_document` hits per unique `file_path`.

### 5. Context Expansion
- When `options.context_window > 0`:
  For each returned chunk $(path, i)$, queries the store for chunks where:
  ```sql
  file_path = '{path}' AND chunk_index >= {i - w} AND chunk_index <= {i + w}
  ```
  Sorted by `chunk_index`, concatenates their text into `expanded_context`.

### 6. Reranking Hook
- Protocol `Reranker`:
  ```python
  class Reranker(Protocol):
      def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]: ...
  ```
- Default: `PassThroughReranker`.

---

## Synchronization Workflow

`IndexSynchronizer.sync(ingest_report, embed_service)`:
1. Identify deleted files and removed chunk IDs from `ingest_report.removed_chunk_ids`.
   - Purge from `store.delete_chunks(removed_chunk_ids)`.
2. Extract new and changed chunk records from `ingest_report.records`.
3. Filter out chunks belonging to unchanged files that are already present in the index.
4. Prepare `DocumentText(content=c.text, title=c.title)` for new/changed chunks.
5. Batch embed via `embed_service.embed_documents(doc_texts, progress_callback=...)`.
6. Construct `ChunkSchema` records with generated vectors.
7. Upsert records into LanceDB `chunks` table.
8. Rebuild / update the FTS index `table.create_index("text", config=lancedb.index.FTS(), replace=True)`.
9. Return `SyncSummary`.

---

## Dependency & Layer Architecture

```
src/npu_rag/index  ──▶  src/npu_rag/embedding (EmbeddingService, types)
src/npu_rag/index  ──▶  src/npu_rag/ingest (ChunkRecord, IngestReport)
```
- Layer rule: `embedding` and `ingest` do not know about `index`.
- `index` imports from both upstream packages cleanly.
