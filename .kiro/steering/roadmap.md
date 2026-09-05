# Roadmap

## Overview

A local, low-energy retrieval system over a folder of documents. Text is extracted from `.md`, `.pdf`, and `.txt` files under a configurable source root (default `D:\Source\substack_dl\archive`), chunked, and embedded on the AMD XDNA2 NPU of a Ryzen AI 9 HX 370 via ONNX Runtime + the VitisAI Execution Provider. Embeddings land in a local embedded vector store (LanceDB) alongside a full-text index, and are queried through two surfaces: a `uvx`-packaged CLI and an MCP server over streamable HTTP.

The project is **retrieval-only**. It contains no LLM and generates no answers — it returns ranked passages with source attribution, and any synthesis is the job of the MCP client. This keeps the entire NPU budget on embedding and (optionally) cross-encoder reranking, which is where NPU acceleration actually pays.

## Approach Decision

- **Chosen**: Approach C — Pluggable `Embedder` interface, in-process by default. A single Python package where embedding sits behind a narrow protocol with swappable backends (`VitisAIEmbedder` on NPU, `CpuOnnxEmbedder`, `SubprocessEmbedder` as an isolation fallback), selected explicitly per run.
- **Why**:
  1. The `--provider npu|cpu|auto` requirement already forces this abstraction — naming it up front costs nothing.
  2. Benchmarking three candidate models requires one interface with swappable backends. Same seam.
  3. It is the only option that survives the primary technical risk (see Constraints) without rearchitecting.
- **Rejected alternatives**:
  - **A — Monolith with direct VitisAI EP calls.** Smallest, but makes the unresolved standalone-wheel problem a single point of total project failure with no escape hatch.
  - **B — Embedding daemon + thin clients.** Pays process-supervision cost up front for a fallback that may never be needed. Remains available as a later refactor if a shared, always-warm NPU session proves necessary; the `Embedder` seam makes that refactor cheap.

## Scope

- **In**: Text extraction from `.md`, `.pdf`, `.txt`; token-budget chunking; NPU/CPU embedding with explicit provider selection; a documented energy and throughput benchmark; LanceDB storage with hybrid vector + full-text retrieval; incremental re-indexing via content hash; a `uvx`-packaged CLI; an MCP server over streamable HTTP.
- **Out**: Images of any kind (the source archive holds ~3,800 PNG/JPEG files — explicitly not indexed; no OCR, no captioning, no multimodal embeddings). Answer generation, chat, or any LLM. Cloud services of any kind, including cloud rerankers. Multi-user access, authentication, or network exposure beyond localhost. GPU (iGPU/Radeon 890M) execution paths.

## Constraints

- **Language / packaging**: Python 3.12 end-to-end, dependency-managed with `uv`. CLI and MCP server must be launchable via `uvx` with `[project.scripts]` entry points.
- **Primary technical risk**: `uvx` conflicts with AMD's install model. The Ryzen AI installer creates a **conda** environment; using VitisAI EP from an arbitrary venv requires the standalone wheels, and [RyzenAI-SW issue #213](https://github.com/amd/RyzenAI-SW/issues/213) reports that installing `onnxruntime_vitisai` into a user environment left the EP unregistered. The issue is open and unresolved. This must be spiked first and carries a designed fallback (`SubprocessEmbedder`).
- **Environment gap**: Ryzen AI Software is not installed on the target machine, and the NPU driver is `32.0.20102.3930` against a Ryzen AI 1.8 minimum of `32.0.203.280`. Provisioning is in scope for the first spec.
- **NPU compilation**: VitisAI EP requires static tensor shapes. Sequence length is fixed at compile time and batch size is 1 by default (dynamic batch only since 1.8's predecessors). This constrains chunking and batching design, not just model loading.
- **Model support**: Strix (STX) supports CNN INT8/BF16, NLP BF16, and LLM via OGA. BERT-style encoders are NPU-eligible on this silicon; INT8-only parts (Phoenix/Hawk Point) are not a target.
- **Licensing**: `embeddinggemma-300m` is under Google's **Gemma Terms of Use** (gated, acceptable-use restrictions), not a permissive license. `bge-large-en-v1.5` is MIT; `nomic-embed-text-v1.5` is Apache 2.0. Acceptable for personal use; constrains redistribution of a bundled model.
- **Offline**: No network dependency at query time. Model download is a one-time provisioning step.

## Boundary Strategy

- **Why this split**: The four domains fail independently and are reviewed differently. NPU provisioning and model compilation is hardware-coupled research work with a real chance of not working at all; document extraction is pure, testable, hardware-free logic; the index is a storage and ranking concern; the CLI and MCP server are thin presentation layers over a shared retrieval API. Putting the risky hardware work in its own spec means the project's central premise gets validated or falsified before the other three specs commit to it.
- **Shared seams to watch**:
  - **Config ownership.** Libraries take explicit parameters and read no global config. The CLI and MCP server own configuration resolution (env var, TOML file, flag precedence, `ARCHIVE_PATH`) and inject values downward. This prevents a dependency inversion where foundational specs reach up into the surfaces.
  - **Chunk token budget.** `document-ingest` chunks to a token budget that is an *input parameter*, while the correct value is an *output* of `npu-embedding-runtime` (the compiled static sequence length). The contract is the parameter, not the number.
  - **Retrieval API.** `vector-index` must expose a single query-orchestration API consumed identically by both surfaces, so ranking behaviour cannot drift between CLI and MCP.
  - **Benchmark harness.** Lives in `npu-embedding-runtime` as an importable module; `search-cli` later exposes it as a subcommand rather than reimplementing it.

## Specs (dependency order)

- [ ] npu-embedding-runtime -- Provision the Ryzen AI stack, define the pluggable Embedder interface with NPU/CPU/subprocess backends, and produce a documented benchmark of three candidate embedding models. Dependencies: none
- [ ] document-ingest -- Walk a configurable source root, extract text from md/pdf/txt, chunk to a token budget, and track per-file content hashes for incremental re-indexing. Dependencies: none
- [ ] vector-index -- LanceDB schema and storage, hybrid vector + full-text retrieval, fusion/rerank/MMR/filters, exposed as one query-orchestration API. Dependencies: npu-embedding-runtime, document-ingest
- [ ] search-cli -- uvx-packaged CLI with index/search/status/benchmark commands and the full search-algorithm flag surface. Dependencies: vector-index
- [ ] mcp-server -- MCP server over streamable HTTP exposing retrieval tools to any MCP client. Dependencies: vector-index
