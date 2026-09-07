# Requirements Document

## Project Description (Input)

### Who has the problem

The owner of a GPD Pocket 4 (AMD Ryzen AI 9 HX 370) who wants to search a personal document archive locally, without cloud services, and without paying hours of CPU load, heat, fan noise, and battery drain to build the index.

Embedding is the single most compute-intensive part of a local retrieval system. For a corpus of roughly 30,000 chunks the work is on the order of 9 PFLOP. On CPU that is hours at full load; on the machine's idle 50-TOPS XDNA2 NPU it should be minutes, at a fraction of the wattage. Nothing else in this project matters if that gap cannot be realized in practice.

### Current situation

- The NPU is present and healthy: `PCI\VEN_1022&DEV_17F0`, XDNA2, Strix Point.
- **Ryzen AI Software is not installed.** `C:\Program Files\RyzenAI` does not exist.
- Driver-level XRT tooling **is** present without the SDK: `xrt-smi.exe` and `pyxrt.pyd` at `C:\Windows\System32\AMD`. Verified working — it enumerates the device as `NPU Strix` (8 columns) and reports estimated power, so energy measurement does not depend on the SDK.
- The installed NPU driver reports as `32.0.20102.3930` (XRT 2.21.0, firmware 1.1.2.64, dated 2026-05-07). The 1.8 documentation states a minimum of `32.0.203.280`. These version schemes are **not directly comparable** — the third component differs in width and, read component-wise, the installed value is the larger. Driver adequacy is therefore an open question to be settled by whether the execution provider registers, not by string comparison.
- Verified: the stock `onnxruntime` wheel exposes only `AzureExecutionProvider` and `CPUExecutionProvider`; `onnxruntime-vitisai` is absent from PyPI and AMD's `voe` package there is a labelled dummy. The provider exists only inside the vendor installer.
- No source code exists anywhere in this project. Python 3.12 and `uv` 0.12.5 are available.
- This machine is **uv/venv only — conda is not installed and must not be assumed** (user directive). AMD's documented install path defaults to conda, but that path is not used here: the provider wheels come from AMD's package index at `https://pypi.amd.com/simple` into uv-managed venvs. [RyzenAI-SW issue #213](https://github.com/amd/RyzenAI-SW/issues/213) reported the EP failing to register in a user environment; the 2026-09-04 probe on this machine showed that failure is avoidable (see research.md): with `numpy<2` and the `voe` wheel's stranded DLL payload relocated, the EP registers and is selected in a pure uv venv. The unresolved remainder is the BF16 compiler (`vaiml.dll`), which those wheels do not ship.

### What should change

- The Ryzen AI stack is provisioned and verified on this machine, with the driver gap closed and the steps documented.
- An `Embedder` protocol exists with working NPU, CPU, and isolated-execution implementations, selected explicitly and never silently.
- Three candidate embedding models are prepared for the NPU and measured.
- A **written benchmark document** exists comparing them on throughput, energy, latency, and retrieval quality — detailed enough to justify the model choice to a future reader and to be re-run.
- It is known, with evidence, whether the application's own packaging can host NPU execution directly, or whether isolated execution is required. This answer gates the four downstream specs.

### Approach

Approach C from discovery: a narrow `Embedder` protocol with swappable backends, in-process by default.

Work proceeds **risk-first**. The first deliverable is a minimal spike that loads any model onto the NPU from the application's own managed environment and confirms the execution provider actually registers. That single result determines whether the rest of the project uses in-process embedding or isolated execution, and it must land before downstream specs commit.

Models sit behind one interface so that benchmarking three candidates is a matter of configuration rather than three code paths. Static-shape constraints (fixed sequence length, batch size 1 by default) are treated as first-class design inputs, not late surprises.

**Ad hoc experimentation is explicitly sanctioned.** Small throwaway scripts may be written and run at any point to pretest an assumption — provider registration, a quantization step, a session option, a power-measurement method — before it is committed to the design. Verifying against real hardware is preferred over reasoning about documentation, particularly given that the primary risk is an open, unresolved upstream issue.

### Constraints

- **Application packaging versus vendor packaging is the defining risk**, tracked via RyzenAI-SW issue #213. Isolated execution exists specifically for this outcome.
- Driver must be upgraded from `32.0.20102.3930` to `32.0.203.280` or newer.
- NPU execution requires **static tensor shapes**: sequence length fixed at preparation time, batch size 1 by default.
- Strix (STX) supports NLP BF16, so BERT-style encoders are NPU-eligible. INT8-only targets (Phoenix, Hawk Point) are not a target.
- Provider selection is `npu`, `cpu`, or `auto`, with no implicit silent fallback; the active provider is always reported.
- Model licensing differs: `embeddinggemma-300m` is under Google's Gemma Terms of Use (gated, acceptable-use restrictions); `nomic-embed-text-v1.5` is Apache 2.0; `gte-modernbert-base` is Apache 2.0 and **not gated**, so it needs no credential. Acceptable for personal use; the Gemma terms constrain redistribution of a bundled model. *(Amended 2026-09-06: the third candidate was `bge-large-en-v1.5`, MIT — see requirement 4.1.)*
- Python 3.12, `uv`-managed, Windows 11.
- All measurements must be reproducible on this machine. The benchmark document records hardware, driver, and runtime versions alongside results.

## Introduction

This feature turns an idle NPU into the workhorse of a local retrieval system. It provisions and verifies the vendor AI runtime on the target machine, exposes text embedding through a single interface backed by explicitly selected compute providers, prepares three candidate models for NPU execution, and produces a written benchmark that decides which model becomes the project default.

It is the foundation spec and the project's principal risk. Its central question — can the NPU be reached from this application's own environment — has no confirmed answer upstream, and the four downstream specs depend on the result. Accordingly, the feature is required to make its environment state, its provider selection, and its failures explicit and legible at every point, so that a silent degradation to CPU can never masquerade as success.

## Boundary Context

- **In scope**: Environment provisioning and capability verification; acquisition and preparation of candidate models for NPU execution; the embedding interface and its NPU, CPU, and isolated-execution providers; explicit provider selection and reporting; query-versus-document input handling; preparation-artifact caching; benchmark measurement of throughput, latency, energy, memory, vector fidelity, and retrieval quality; the written benchmark document; diagnostics for every failure mode above.
- **Out of scope**: Deciding how documents are split into chunks, and which tokenizer boundaries apply to a corpus — this feature publishes the maximum input token length and the tokenizer as a contract, and consumes neither. Persisting or searching vectors. Any command-line surface, which a separate feature provides while reusing this feature's benchmark capability rather than reimplementing it. Reranking inference. Answer generation or any language model. Integrated-GPU execution paths.
- **Adjacent expectations**: The document-ingest feature is expected to size its chunks against the maximum input token length reported here, and to use the tokenizer supplied here rather than an approximation. The vector-index feature is expected to consume vectors and the reported vector dimension without re-normalizing or re-prefixing them. Neither feature is expected to install, detect, or repair the AI runtime environment; this feature owns that and reports its state.

## Requirements

### Requirement 1: Environment Provisioning and Capability Verification

**Objective:** As the operator of this machine, I want the NPU environment provisioned and independently verifiable, so that I can trust acceleration is genuinely available before committing hours to indexing a corpus.

#### Acceptance Criteria

1. The Embedding Runtime shall provide a capability check that reports whether NPU hardware is present, whether the vendor AI runtime is installed, and whether the installed device driver meets the runtime's documented minimum version.
2. When the capability check runs, the Embedding Runtime shall report each condition individually with its own pass or fail state, rather than a single aggregate verdict.
3. If the installed device driver version is below the runtime's documented minimum, then the Embedding Runtime shall report the installed version, the required version, and the remediation step.
4. If the vendor AI runtime is not installed, then the Embedding Runtime shall report its absence and shall not attempt NPU execution.
5. When the capability check completes, the Embedding Runtime shall report whether NPU execution is reachable from the application's own managed environment or only through isolated execution.
6. The Embedding Runtime shall provide provisioning documentation sufficient to reproduce a working environment from a clean machine, including the driver and runtime versions verified.

### Requirement 2: Explicit Provider Selection and Reporting

**Objective:** As a user, I want to state which compute provider performs embedding, so that a silent fall back to CPU can never conceal the failure of the capability this project exists to deliver.

#### Acceptance Criteria

1. The Embedding Runtime shall accept a provider selection of `npu`, `cpu`, or `auto` for every embedding operation.
2. When provider selection is `npu` and the NPU is unavailable for any reason, the Embedding Runtime shall fail with a diagnostic naming the unmet condition, and shall not embed using any other provider.
3. When provider selection is `cpu`, the Embedding Runtime shall embed using the CPU regardless of whether the NPU is available.
4. When provider selection is `auto` and the NPU is available, the Embedding Runtime shall use the NPU.
5. When provider selection is `auto` and the NPU is unavailable, the Embedding Runtime shall report the specific reason the NPU was not used before embedding using the CPU.
6. The Embedding Runtime shall report which provider actually served every embedding operation, including operations that succeed.
7. While an embedding operation is in progress, the Embedding Runtime shall not change provider before the operation completes.

### Requirement 3: Embedding Generation

**Objective:** As a downstream consumer of vectors, I want text embedded correctly and consistently, so that retrieval quality is not silently degraded by a handling mistake that produces no error.

#### Acceptance Criteria

1. When given a batch of input texts, the Embedding Runtime shall return exactly one vector per input, in the same order as the inputs.
2. The Embedding Runtime shall require every embedding request to declare whether its text is corpus content or a search query.
3. If an embedding request does not declare whether its text is corpus content or a search query, then the Embedding Runtime shall reject the request rather than apply a default.
4. When embedding corpus content, the Embedding Runtime shall apply the active model's document-side convention; when embedding a search query, it shall apply the active model's query-side convention.
5. The Embedding Runtime shall return unit-normalized vectors, so that cosine similarity and dot-product similarity are equivalent for downstream consumers.
6. The Embedding Runtime shall report the active model's vector dimension and maximum input token length on request.
7. The Embedding Runtime shall expose the active model's tokenizer, so that a consumer can measure input length by the same rule the runtime applies.
8. When a consumer measures an input's token length using the exposed tokenizer, the Embedding Runtime shall treat that same input as within limits if and only if the consumer's measurement is within the reported maximum.
9. If an input text exceeds the active model's maximum input token length, then the Embedding Runtime shall reduce it to that limit and report that the input was shortened, identifying which input.
10. When the same text is embedded twice under the same model and provider, the Embedding Runtime shall return identical vectors.

### Requirement 4: Candidate Model Preparation and Reuse

**Objective:** As a user, I want candidate models obtainable and reusable without repeating setup cost, so that switching models or re-running a benchmark does not mean waiting through preparation again.

#### Acceptance Criteria

1. The Embedding Runtime shall support three candidate models: `embeddinggemma-300m`, `nomic-embed-text-v1.5`, and `gte-modernbert-base`.

   **Amended 2026-09-06.** The third candidate was `bge-large-en-v1.5`, a 2023 model the project owner ranked last and wanted replaced with something modern and better qualified for the use case. Preference order is now EmbeddingGemma, then nomic, then the third slot. Two constraints shaped the replacement, both verified rather than assumed:
   - **Dense architectures only.** `nomic-embed-text-v2-moe` scores better than v1.5 on BEIR and MIRACL but routes 8 experts top-2 per token; that conditional computation does not export cleanly to ONNX and cannot compile to a static-shape NPU graph. Nomic therefore stays at **v1.5**, whose dense architecture is what makes it exportable at all.
   - **Compile cost, not a size ceiling.** `gte-modernbert-base` is 149M parameters, roughly 600 MB exported — ~~around half the compile time of anything else in the set~~, which matters because every benchmark cell pays it. *(**Falsified by measurement, 2026-09-07, task 5.5.** The prediction was never validated and is wrong by roughly an order of magnitude in the other direction: the real cold NPU compile is **2953.5 s (~49 min)**, against the control model's 160–310 s. The comparison is not like-for-like — 22 layers at sequence 512 versus 6 layers at 128 — but the claim as written was a guess presented as a reason for choosing the model. It does not change the choice: the model exports to a fully static graph and the compiler offloads 1149 of 1152 operators in a single subgraph, which is what actually mattered. It does change the benchmark budget. **Warm reuse is 0.59 s**, so tasks 6.3 and 6.4 must reuse artifacts across cells and repetitions rather than paying cold compiles per cell.)* *(Corrected 2026-09-06: an earlier draft framed this as an ONNX "2 GB ceiling" the model had to fit under. That was wrong. The 2 GB limit is a Protocol Buffers constraint on a **single self-contained** `.onnx` file, not a limit on model size — `onnx.save_model(save_as_external_data=True)` writes tensors to sidecar files and lifts it entirely, which is how far larger models ship. EmbeddingGemma's export sits at 1.22 GB only because task 3.2 passed `external_data=False`. That is a project choice, not a constraint. This machine has 24 GB of system RAM, which the NPU draws from, so nothing here is near a hardware limit either.)*
     **Open question for 5.5, if a larger candidate is ever wanted**: whether the Vitis AI compiler accepts an external-data ONNX model. Untested. If it does, model size stops being a consideration at all; if it does not, `external_data=False` becomes a genuine constraint rather than a default, and that should be recorded as such.

   `gte-modernbert-base` is Apache-2.0 and **not gated**, so unlike the primary candidate it is fetchable without a credential. It carries no Dense stage and uses **symmetric** text handling with no query or document instruction, so its templates are identity.

   **It uses CLS pooling, not mean pooling** — confirmed from the model's own `1_Pooling/config.json` (`pooling_mode_cls_token = True`). `ModelProfile.pooling` is `Literal["mean"]` today and `postprocess.py` implements masked mean pooling only, so adding this candidate requires widening that literal and adding a CLS branch (`tokens[:, 0, :]`). That branch does not consult the attention mask at all, so it cannot reintroduce the mask-blind pooling hazard. Deferred to task 5.5, where a real export and compile happen anyway. *(Reassigned from 6.1 on 2026-09-07: the work is in `profiles.py`/`postprocess.py`, outside 6.1's `bench fixtures` boundary. Note also that `ModelProfile.pooling` currently has no production consumer at all — `postprocess.finalize` hardcodes masked mean pooling — so widening the literal without dispatching on it would leave a CLS profile silently mean-pooled.)*
2. The Embedding Runtime shall treat `embeddinggemma-300m` as the initial default candidate until the benchmark document supersedes that choice.
3. When a model is prepared for a provider for the first time, the Embedding Runtime shall persist the resulting preparation artifacts for reuse.
4. When valid preparation artifacts already exist for the requested model and provider, the Embedding Runtime shall reuse them instead of repeating preparation, and shall report that it did so.
5. If a candidate model requires acceptance of license terms before it can be obtained, then the Embedding Runtime shall report the licensing requirement and the acceptance step, rather than failing with a generic retrieval error.
6. If a candidate model cannot be prepared for the requested provider, then the Embedding Runtime shall report which preparation stage failed, and shall not substitute a different model or provider.
7. When preparation artifacts were produced under a different model version, input length, or provider than the current request, the Embedding Runtime shall treat them as invalid and prepare again.

### Requirement 5: Isolated Execution Fallback

**Objective:** As a user, I want NPU embedding to remain available even when it cannot run inside the application's own environment, so that a packaging incompatibility outside my control does not cost me the feature entirely.

#### Acceptance Criteria

1. Where NPU execution is not reachable from the application's own managed environment, the Embedding Runtime shall provide an isolated execution mode that performs NPU embedding in a separate environment.
2. When isolated execution is active, the Embedding Runtime shall accept the same requests and return results satisfying the same criteria as in-process execution.
3. When isolated execution is active, the Embedding Runtime shall report that fact for every operation it serves.
4. The Embedding Runtime shall demonstrate that vectors produced by isolated execution match those produced by in-process execution for identical inputs, within a stated tolerance.
5. If the isolated execution environment is unavailable, fails to start, or terminates unexpectedly, then the Embedding Runtime shall report the failure and shall not continue using another provider unless that provider was explicitly selected.
6. While isolated execution is serving a batch, the Embedding Runtime shall report progress on the same terms as in-process execution.

### Requirement 6: Benchmark Measurement

**Objective:** As the project owner, I want the candidate models measured on this specific machine, so that the default model is chosen from evidence gathered here rather than from published claims or assumption.

#### Acceptance Criteria

1. The Embedding Runtime shall provide a benchmark capability that measures each candidate model under both the NPU and CPU providers.
2. When the benchmark runs, the Embedding Runtime shall measure throughput in inputs processed per second, single-input latency at the median and 95th percentile, energy consumed per one thousand inputs, peak resident memory, and preparation time on first run versus subsequent runs.
3. When the benchmark runs, the Embedding Runtime shall measure the similarity between NPU-produced vectors and full-precision CPU reference vectors for the same inputs, so that any loss from reduced-precision execution is quantified rather than assumed absent.
4. When the benchmark runs, the Embedding Runtime shall measure retrieval quality for each candidate model against a fixed query set drawn from the target corpus with pre-identified relevant results.
5. The Embedding Runtime shall use a representative sample of the target corpus for benchmarking, and shall state the sample's size and composition, so that the benchmark does not depend on a completed corpus index.
6. When a benchmark run completes, the Embedding Runtime shall record the hardware identification, driver version, and runtime version under which the measurements were taken.
7. The Embedding Runtime shall repeat each measurement more than once and report the number of repetitions together with the observed variation.
8. If a measurement cannot be taken on this machine, then the Embedding Runtime shall record the omission and its reason, and shall not report an estimated or substituted value in its place.

### Requirement 7: Benchmark Document

**Objective:** As a future reader of this project, I want the model comparison written down with its methodology, so that I can understand the decision, trust it, and re-run it, instead of re-deriving it from scratch.

#### Acceptance Criteria

1. When the benchmark completes, the Embedding Runtime shall produce a written benchmark document containing every measurement required by Requirement 6.
2. The benchmark document shall state the measurement methodology, including how energy was sampled, how the corpus sample was selected, and how many repetitions were performed.
3. The benchmark document shall state a recommended default model together with the reasoning that leads to that recommendation.
4. The benchmark document shall state the conditions under which a different candidate model would be the better choice.
5. The benchmark document shall record the hardware identification, driver version, and runtime version under which the measurements were taken.
6. Where a candidate model or a measurement could not be completed, the benchmark document shall state which one and why.
7. The benchmark document shall present results for the NPU and CPU providers side by side, so that the energy and throughput difference between them is directly readable.

### Requirement 8: Diagnostics, Progress, and Failure Reporting

**Objective:** As a user troubleshooting a failure, I want the system to tell me what went wrong and what to do next, so that I am not left guessing at an opaque error during a long-running operation.

#### Acceptance Criteria

1. If any embedding operation fails, then the Embedding Runtime shall report the provider in use, the model in use, and the stage at which the failure occurred.
2. The Embedding Runtime shall distinguish environment failures, model preparation failures, and execution failures from one another in its reporting.
3. While a batch embedding operation is in progress, the Embedding Runtime shall report progress including inputs completed and inputs remaining.
4. If a batch embedding operation is interrupted before completion, then the Embedding Runtime shall report how many inputs were completed.
5. If a preparation or embedding operation is interrupted, then the Embedding Runtime shall not leave partially written artifacts that a later run would treat as valid.
6. When an operation completes successfully, the Embedding Runtime shall report the elapsed time and the number of inputs processed.
