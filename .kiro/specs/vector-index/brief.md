# Brief: vector-index

## Problem

Embeddings are useless until they can be searched, and pure vector search alone is a poor fit for this corpus. A personal archive of essays needs both semantic recall ("that piece about incentive design") and exact lexical matching (a person's name, a coined term, a specific phrase). Dense retrieval is bad at the latter; BM25 is bad at the former. Either one alone produces visibly disappointing results.

There is a second, subtler problem: one Substack post chunks into four to six pieces, so naive top-10 ranking returns the same three articles repeated. Without diversity control the result list looks broken even when the ranking is technically correct.

## Current State

- No store exists.
- Corpus sizing is known: roughly 10–30k chunks, 30–92 MB of vectors at **768 dimensions**. The target machine has 23.6 GB RAM. *(Corrected 2026-09-07: this said 1024 dimensions, which was `bge-large-en-v1.5`'s width. That candidate was replaced by `gte-modernbert-base` under requirement 4.1, and **all three shipping models are now 768-dimensional** — see `npu-embedding-runtime/profiles.py`. The default is `embeddinggemma-300m`, also 768. Float32 at 768 dims is 3,072 bytes per vector.)*
- LanceDB is the selected store (Apache 2.0, embedded, memory-mapped, actively maintained), with sqlite-vec as the documented fallback.
- At this scale exact brute-force kNN costs about 61 MFLOP over roughly 123 MB — around 2 ms, bandwidth-bound rather than compute-bound. ANN indexing is not yet warranted.

## Desired Outcome

- Chunks and their vectors are stored durably with queryable metadata.
- A single retrieval API serves both surfaces identically, so CLI and MCP results can never diverge.
- Hybrid semantic and lexical search works by default and beats either mode alone on real queries.
- Results are diverse across source documents rather than clustered in one article.
- Incremental upserts and deletes work without a full rebuild.

## Approach

LanceDB as an embedded store with one table carrying vectors and metadata, plus a native full-text index. Retrieval defaults to hybrid: dense search and BM25 run in parallel and fuse by Reciprocal Rank Fusion, which merges by rank and so sidesteps the incomparable-scales problem between cosine distance and BM25 scores without loading any model.

Vector search defaults to exact rather than ANN. This is deliberate and evidence-backed: at 30k chunks exact search takes about 2 ms with 100% recall, needs no index build, and never goes stale on incremental ingest. IVF and HNSW variants are exposed as opt-in for when the archive grows past a few hundred thousand chunks, at which point memory bandwidth — not FLOPs — makes them worthwhile.

Post-ranking, MMR provides diversity and optional score rollup returns documents rather than chunks. All of this is exposed as one orchestration API so the surfaces stay thin.

## Scope

- **In**: LanceDB schema for chunks, vectors, and metadata; upsert and delete keyed to `document-ingest`'s file-state; native full-text index; dense vector search with configurable metric (cosine, dot, l2); exact-by-default with opt-in IVF_PQ / IVF_HNSW_SQ / IVF_SQ / IVF_FLAT plus `nprobes` and `refine_factor` tuning; BM25 keyword search; hybrid fusion via RRF and linear combination; MMR diversity; chunk-versus-document granularity rollup; neighbouring-chunk context expansion; SQL metadata filters with pre-filter and post-filter control; a hook for cross-encoder reranking; the unified query-orchestration API.
- **Out**: Producing embeddings (`npu-embedding-runtime`). Producing chunks (`document-ingest`). Cross-encoder inference itself — this spec defines the hook, the runtime spec would own the model. All cloud rerankers (Cohere, Jina, OpenAI, Voyage), since this is an offline project. Answer generation. CLI or MCP presentation concerns.

## Boundary Candidates

- Schema and storage lifecycle (create, upsert, delete, compact)
- Index management (vector index and full-text index)
- Retrieval primitives (dense, lexical)
- Ranking and fusion (RRF, linear combination, MMR, rollup)
- The unified query-orchestration API consumed by both surfaces

## Out of Boundary

- Choosing the embedding model or its dimension — consumed as configuration.
- Embedding the query text — delegated to the `Embedder` from `npu-embedding-runtime`.
- Flag naming and CLI ergonomics.
- Migration between embedding models. Changing models invalidates the index; a documented rebuild is acceptable.

## Upstream / Downstream

- **Upstream**: `npu-embedding-runtime` for vectors, vector dimension, and query embedding. `document-ingest` for chunk records, metadata, and file-state.
- **Downstream**: `search-cli` and `mcp-server` both consume the query-orchestration API. Any future reranking or generation spec builds on the same API.

## Existing Spec Touchpoints

- **Extends**: None. Greenfield.
- **Adjacent**: Both wave-1 specs feed it and both wave-3 surfaces consume it. This spec is the project's structural centre, and its API is the most important seam to get right.

## Constraints

- LanceDB (Apache 2.0). Its full-text search is now Lance-native; `use_tantivy` is no longer accepted by the index creation APIs.
- Must run embedded and in-process — no server, no daemon — to remain compatible with `uvx` launching.
- Resident memory must stay flat and independent of corpus size, relying on memory-mapping rather than holding vectors in RAM. This is precisely why Chroma and FAISS were rejected.
- Exact search is the default; ANN is opt-in.
- Queries must use the model's asymmetric query prefix, distinct from the document prefix. Getting this wrong degrades retrieval silently with no error, so it is a hard correctness requirement rather than an option.
- sqlite-vec remains the documented fallback if LanceDB proves unsuitable on Windows.
