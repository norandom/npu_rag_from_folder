# Brief: npu-embedding-runtime

## Problem

Embedding a document corpus is the single most compute-intensive part of a local retrieval system, and on a handheld device it is also the most energy-intensive. Running it on the CPU of a GPD Pocket 4 means hours of full-load compute, heat, fan noise, and battery drain for what should be a background task. The machine has an idle XDNA2 NPU rated for 50 TOPS that is designed precisely for this workload, but nothing in the standard Python embedding ecosystem targets it.

The pain is concrete: for a ~30k-chunk corpus, embedding is roughly 9 PFLOP of work. On CPU that is hours; on NPU it should be minutes, at a fraction of the wattage. Nothing downstream in this project matters if that gap cannot be realized in practice.

## Current State

- The NPU is present and healthy (`PCI\VEN_1022&DEV_17F0`, XDNA2, Strix Point) on a Ryzen AI 9 HX 370.
- **Ryzen AI Software is not installed.** `C:\Program Files\RyzenAI` does not exist.
- The installed NPU driver is `32.0.20102.3930`; Ryzen AI 1.8 requires `32.0.203.280` or newer.
- No code exists. Python 3.12 and `uv` 0.12.5 are available.
- AMD's supported install path creates a **conda** environment, which is in direct tension with this project's `uvx` packaging requirement.

## Desired Outcome

- The Ryzen AI stack is provisioned and verified on this machine, with the driver gap resolved.
- An `Embedder` protocol exists with working `VitisAIEmbedder` (NPU), `CpuOnnxEmbedder`, and `SubprocessEmbedder` (isolation fallback) implementations, selected explicitly and never silently.
- Three candidate models are converted to ONNX, quantized to BF16, compiled for the NPU, and measured.
- **A written benchmark document exists** comparing them on throughput, energy, latency, and retrieval quality — sufficient to justify the model choice to a future reader.
- It is known, with evidence, whether `uvx` + standalone wheels works, or whether the subprocess fallback is required.

## Approach

Approach C from discovery: a narrow `Embedder` protocol with swappable backends, in-process by default.

Work proceeds risk-first. The very first deliverable is a minimal spike that loads *any* model onto the NPU via VitisAI EP from a `uv`-managed environment and confirms the EP actually registers. That single result determines whether the rest of the project uses in-process embedding or the subprocess worker, and it must land before any downstream spec commits.

Models are handled behind one interface so that benchmarking three candidates is a matter of configuration rather than three code paths. Static-shape constraints (fixed sequence length, batch size 1 by default) are treated as first-class design inputs, not late surprises.

## Scope

- **In**: Ryzen AI SDK and driver provisioning with documented steps; NPU capability detection; model acquisition; ONNX export and BF16 quantization; VitisAI EP session construction with compilation caching; static-shape handling; the `Embedder` protocol and its three backends; explicit provider selection with no implicit fallback; asymmetric query/document prefix handling; a reusable benchmark harness; the written benchmark document.
- **Out**: Chunking and tokenization policy (owned by `document-ingest` — this spec exposes the max sequence length as a contract). Storage of the resulting vectors (`vector-index`). CLI surface (`search-cli` exposes the benchmark as a subcommand later). Cross-encoder reranking *inference* is a downstream extension, though the interface should not preclude it. Any LLM. iGPU execution.

## Boundary Candidates

- Environment provisioning and capability detection (does this machine have a working NPU stack?)
- Model asset pipeline (acquire, export to ONNX, quantize to BF16, compile, cache)
- The `Embedder` protocol and backend implementations
- Provider selection and reporting policy
- Benchmark harness and its written output

## Out of Boundary

- Deciding the final production model. The benchmark *informs* it; this spec does not hardcode a winner beyond a documented default.
- Chunk sizing policy. This spec publishes the sequence-length constraint; it does not decide how text is split.
- Retrieval quality evaluation beyond what is needed to compare the three models against each other.

## Upstream / Downstream

- **Upstream**: AMD Ryzen AI Software 1.8 and the NPU driver. ONNX Runtime with VitisAI EP. Hugging Face for model weights.
- **Downstream**: `vector-index` consumes embeddings and the vector dimension. `document-ingest` consumes the max sequence length. `search-cli` exposes the benchmark. A future reranking spec would extend the same interface.

## Existing Spec Touchpoints

- **Extends**: None. Greenfield; this is the foundation spec.
- **Adjacent**: `document-ingest` shares the token-budget contract. Both are wave-1 and must not both define tokenization.

## Constraints

- **`uvx` vs conda is the defining risk.** [RyzenAI-SW #213](https://github.com/amd/RyzenAI-SW/issues/213) reports VitisAI EP failing to register when the standalone wheel is installed into a user environment; the issue is open and unresolved. The `SubprocessEmbedder` fallback exists specifically for this.
- Driver must be upgraded from `32.0.20102.3930` to `32.0.203.280`+.
- VitisAI EP requires **static tensor shapes**; sequence length is fixed at compile time, batch size 1 by default.
- Strix supports NLP BF16 — BERT-style encoders are NPU-eligible. INT8-only targets are out of scope.
- `--provider npu|cpu|auto` with **no implicit silent fallback**; the active provider must always be reported.
- Candidate models are gated by license: `embeddinggemma-300m` under Gemma Terms of Use (gated, restricted), `nomic-embed-text-v1.5` Apache 2.0, `gte-modernbert-base` Apache 2.0 and ungated. *(Amended 2026-09-06: the third candidate was `bge-large-en-v1.5`; see requirement 4.1.)*
- All measurements must be reproducible on this machine; the benchmark document records hardware, driver, and SDK versions alongside results.

## Benchmark Deliverable

An explicit, written artifact — not just numbers in a terminal. It must record:

- **Primary candidate**: `embeddinggemma-300m` (project owner's stated preference). **Comparators**: `nomic-embed-text-v1.5`, then `gte-modernbert-base`. *(Amended 2026-09-06: `bge-large-en-v1.5` replaced - see requirement 4.1 for the dense-architecture and size constraints that shaped the choice.)*
- **Per model × per provider (NPU, CPU)**: chunks/sec throughput; wall-clock for a fixed representative batch; single-query embed latency (p50/p95); average and peak power draw plus energy per 1k chunks; peak RSS; model compile time and cache size on first vs. subsequent runs.
- **Quality**: retrieval quality on a small hand-built query set drawn from the actual archive, so the energy/quality trade-off is visible rather than assumed.
- **Fidelity**: BF16-vs-FP32 embedding divergence (cosine similarity against a CPU FP32 reference) to confirm quantization has not degraded the vectors.
- **Methodology**: how measured, how power was sampled, how many runs, variance. Documented well enough to re-run and to trust.
- **Verdict**: a recommended default model with stated reasoning, and the conditions under which a different choice would be better.

