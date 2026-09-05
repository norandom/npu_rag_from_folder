# Research & Design Decisions

## Summary

- **Feature**: `npu-embedding-runtime`
- **Discovery Scope**: New Feature (greenfield) + Complex Integration (vendor hardware runtime)
- **Key Findings**:
  - **NPU power measurement is available and Strix-specific.** `xrt-smi examine --report platform` reports "Estimated Power" in Watts, is documented as pollable in a loop, and is supported **only on STX devices and onwards** — which this machine is. The open question from the requirements review is resolved for the NPU side; the CPU side remains unmeasured by this tool.
  - **An ONNX export is not the whole model.** `optimum-cli export onnx` converts only the transformer trunk, emitting *token* embeddings. EmbeddingGemma's sentence pipeline is Pooling → Dense → Normalize. Those stages must be applied after the graph, which decides where normalization (3.5) is guaranteed.
  - **EmbeddingGemma's prefixes are structured, not decorative.** Documents use `title: {title | "none"} | text: {content}`; retrieval queries use `task: search result | query: {content}`. The document form has a title slot the Substack archive can actually populate.
  - **Two distinct caching mechanisms exist** in Ryzen AI 1.8: the implicit VitisAI EP cache (`cache_dir`/`cache_key`) and the explicit ONNX Runtime **EP context** file (`ep.context_enable`, `ep.context_file_path`). The latter is the documented production mechanism and is an artifact we control.

## Research Log

### Ryzen AI 1.8 model deployment and the BF16 flow

- **Context**: Requirement 4 needs models prepared for the NPU and reused; the constraint set says Strix supports NLP BF16. The precise mechanism was unknown.
- **Sources Consulted**: [Model Run — Ryzen AI 1.8](https://ryzenai.docs.amd.com/en/latest/modelrun.html), [Vitis AI EP — ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html), [Ryzen AI release notes](https://ryzenai.docs.amd.com/en/latest/relnotes.html)
- **Findings**:
  - The EP partitions the graph automatically: NPU-supported subgraphs run on the NPU, the remainder falls back to CPU **within the same session**. This is silent and invisible unless inspected.
  - BF16 targeting uses a JSON `config_file` provider option carrying `optimize_level` (1–3) and `preferred_data_storage`. INT8 uses a `target` option (`X2` for STX/KRK).
  - Session construction: `InferenceSession(model, providers=["VitisAIExecutionProvider"], provider_options=[{...}])`. Options include `cache_dir`, `cache_key`, `log_level`.
  - EP context: `session_options.add_session_config_entry('ep.context_enable', '1')` plus `ep.context_file_path` produces an explicit compiled snapshot. Documented as the production path; the EP cache is described as the development path.
  - Model-type support matrix confirms STX gets CNN INT8/BF16, **NLP BF16**, and LLM via OGA.
- **Implications**:
  - Partial CPU fallback *inside* an NPU session is a real hazard for Requirement 2. "Provider = NPU" at the session level does not prove the model actually ran on the NPU. The design must verify partitioning, not just session creation.
  - EP context files are the right artifact for 4.3/4.4 — explicit, inspectable, and invalidatable, unlike an opaque temp cache.

### NPU power and utilization telemetry on Windows

- **Context**: Requirement 6.2 mandates energy per 1,000 inputs. At requirements time it was unknown whether this was measurable at all; 6.8 was written to let it degrade to a recorded omission.
- **Sources Consulted**: [NPU Management Interface — Ryzen AI 1.8](https://ryzenai.docs.amd.com/en/latest/xrt_smi.html)
- **Findings**:
  - `xrt-smi examine --report platform` reports **Estimated Power in Watts** plus performance mode and device identity.
  - **Power reporting is unsupported on PHX and HPT; available on STX and onwards.** This machine is STX (Strix Point) — supported.
  - The tool does **not** report NPU utilization.
  - Documented as safe to run in a loop during inference for live sampling.
  - Windows binary lives at `C:\Windows\System32\AMD`; the path may need adding to `PATH`.
  - `examine --report aie-partitions` reports partition and column occupancy.
- **Implications**:
  - NPU energy is measurable. 6.8's omission path is retained for the **CPU** provider, where `xrt-smi` gives nothing.
  - Because utilization is unavailable, `aie-partitions` occupancy becomes the practical evidence that the NPU is genuinely engaged — useful for the partitioning check above.
  - Sampling is polling-based, so energy is an integration of sampled power over wall-clock, not a hardware energy counter. Accuracy limits must be stated in the benchmark methodology (7.2).

### EmbeddingGemma model characteristics

- **Context**: Selected as the primary candidate (4.2). Its dimensions, context limit, and prefix conventions drive Requirement 3.
- **Sources Consulted**: [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m), [maxious/embeddinggemma-300m-onnx](https://huggingface.co/maxious/embeddinggemma-300m-onnx)
- **Findings**:
  - 300M parameters, **768-dim** output, **2048-token** context, built on Gemma 3.
  - Matryoshka representation: truncatable to 512, 256, or 128 **with re-normalization**.
  - Document prefix: `title: {title | "none"} | text: {content}`.
  - Query prefix is task-specific; retrieval uses `task: search result | query: {content}`.
  - No official ONNX export in the Google repo. Community exports exist. `optimum-cli export onnx --model ... --task feature-extraction` is the supported route.
  - Gated on Hugging Face behind Gemma Terms of Use acceptance.
- **Implications**:
  - 3.4's "document-side convention" is not a bare string prefix — it is a template with a **title** slot, which `document-ingest` already extracts. The contract this spec publishes must expose that slot rather than hardcode `none`.
  - The 2048-token architectural limit is **not** the value to publish under 3.6. The compiled static length is (see decision below).
  - Gated download makes 4.5 a concrete, expected path rather than an edge case.

### ONNX export scope and the sentence-embedding pipeline

- **Context**: Requirement 3.5 mandates unit-normalized vectors and 5.4 requires cross-backend equivalence.
- **Sources Consulted**: [Sentence Transformers efficiency docs](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html), [Optimum ONNX export](https://huggingface.co/docs/optimum), [optimum issue #1519](https://github.com/huggingface/optimum/issues/1519)
- **Findings**:
  - ONNX export converts only the transformer, producing token embeddings. Pooling, any Dense layer, and normalization are separate pipeline stages.
  - EmbeddingGemma's pipeline includes a **Dense** stage, not just pooling — omitting it produces vectors that are wrong rather than merely unnormalized.
  - Mean pooling must respect the attention mask, which matters acutely under fixed-length padding.
- **Implications**:
  - Decisive for the architecture: post-processing runs **outside** the ONNX graph, identically for every backend. This is what makes 3.5 and 5.4 achievable by construction rather than by testing three separate code paths.

### Reaching the execution provider from a `uv`-managed environment

- **Context**: The project's defining risk, carried from discovery.
- **Sources Consulted**: [RyzenAI-SW #213](https://github.com/amd/RyzenAI-SW/issues/213), [RyzenAI-SW #333](https://github.com/amd/RyzenAI-SW/issues/333), [Ryzen AI installation](https://ryzenai.docs.amd.com/en/latest/inst.html)
- **Findings**:
  - The installer creates a conda environment at `C:\Program Files\RyzenAI\1.8.0`; standalone wheels are offered for custom environments.
  - #213 reports the EP unregistered after installing the standalone wheel into a user environment. Open, unresolved.
  - The EP depends on native runtime assets located via `RYZEN_AI_INSTALLATION_PATH` and `XLNX_VART_FIRMWARE`.
- **Implications**:
  - The failure is plausibly environmental (unset variables, missing native DLLs on `PATH`) rather than fundamental, which means the spike must test the *documented* variable set explicitly before concluding failure.
  - `onnxruntime.get_available_providers()` is the cheap, decisive probe and belongs in the capability check (1.5).

### Live probe on the target machine (2026-09-01)

- **Context**: Feasibility questions raised before implementation. Answered with throwaway code in a scratch `uv` environment rather than from documentation, per the sanctioned ad hoc practice.
- **Method**: `xrt-smi` invoked directly; a scratch `uv` venv with `onnxruntime` 1.29.0, `onnx` 1.22.0, `numpy` 2.5.2; a synthetic fixed-shape `(1, 512, 768)` graph; anonymous HTTP against the model host.
- **Findings**:
  - **Driver-level XRT is present without the SDK.** `xrt-smi.exe` and `pyxrt.pyd` ship in `C:\Windows\System32\AMD`. XRT 2.21.0, hash dated 2026-05-07, NPU firmware 1.1.2.64. The device enumerates as `NPU Strix` at `[00c6:00:01.1]` with 8 columns.
  - **Power reporting works**: `examine --report platform` returned `Estimated Power: 0.001 Watts` at idle. Requirement 6.2 is confirmed feasible on real hardware, and it does not depend on installing the SDK.
  - **The driver may not be too old.** XRT self-reports `32.0.20102.3930`; the docs state a `32.0.203.280` minimum. The third component differs in width, so the schemes are likely different branches; component-wise the installed value is larger. The earlier "driver is below minimum" premise is **withdrawn as unverified**.
  - **No package-index route to the provider exists.** Stock `onnxruntime` exposes only `AzureExecutionProvider` and `CPUExecutionProvider`. `onnxruntime-vitisai` returns 404 on PyPI. AMD's `voe` package on PyPI self-describes as "some common util for vaip dev - Dummy".
  - **Session creation is not evidence of provider selection.** Requesting `VitisAIExecutionProvider` with the EP absent **succeeded**, warned only via `UserWarning`, and ran on the CPU. `provider_options` did not change this. A fabricated provider name produced a louder `EP Error ... Falling back` message — so the *known-but-unavailable* case is the quieter and more dangerous one.
  - **Static shapes are enforceable**: the fixed `(1, 512, 768)` graph rejected a `(1, 256, 768)` input with `InvalidArgument`, confirming the compiled-length contract is verifiable at runtime.
  - **The primary candidate is genuinely gated**: `gated: "manual"`, and `config.json` returns HTTP 401 anonymously. No credential is configured on this machine. Both comparators return 200. The repository contains no ONNX export, confirming self-export is required.
- **What the SDK actually adds, established by inspection**:
  - Already present from the driver package: NPU firmware 1.1.2.64, XRT 2.21.0, `vitis-ai-runtime.dll` and `vitis-ai-runtime2.dll` in `System32`, and roughly forty pre-built `.xclbin` overlays.
  - Missing and only obtainable from the SDK: the **Vitis AI execution provider compiled into an AMD build of ONNX Runtime**, the **VAIP graph compiler** that partitions an ONNX graph and compiles subgraphs for the NPU, the op-support configuration driving that partitioning, and the Quark quantizer.
  - Execution providers are compiled into the ONNX Runtime binary at build time — confirmed by dumping the provider-name string table of the stock wheel, which lists CPU, Azure, CUDA, CoreML, CANN, ACL and others but **not** VitisAI. No `onnxruntime.dll` anywhere on this machine contains it, including AMD's own noise-suppression build, which uses DirectML instead. The provider therefore cannot be added beside a stock wheel.
  - `vitis-ai-runtime.dll` is the execution layer, not the compilation layer: its symbols concern loading a pre-built xclbin and dispatching kernels, and it carries the error string `Fingerprint of xclbin does not match subgraph's fingerprint`. It expects an already-compiled subgraph.
- **Implications**:
  - Because the low-level runtime and overlays already sit in `System32`, which is on the default DLL search path, issue #213 looks **more environmental than fundamental**. The spike should first try the standalone wheels with `RYZEN_AI_INSTALLATION_PATH` and `XLNX_VART_FIRMWARE` set explicitly, since missing native dependencies are unlikely to be the cause.
  - Provider verification must be a **two-guard** pattern — pre-check the available-provider list, post-check the session's selected provider — because the obvious `try`/`except` detects nothing. Partition-share verification is a third, finer check layered on top, not the first line of defence.
  - Task 1.2 must not assume a driver upgrade; provider registration is the real adequacy test.
  - Task 1.3's spike is specifically about the installer's standalone wheels, since no dependency declaration can obtain the provider.
  - Model-host credentials become an explicit prerequisite task rather than an error path discovered at first download.

### Second live probe: the uv route works (2026-09-04)

- **Context**: The user reported the Ryzen AI software installed and required the setup to be venv-based (uv only, conda excluded by directive). Probing established what was actually installed and then executed the task 1.3 spike ad hoc.
- **What "installed" turned out to mean**: three similarly named AMD products exist on the machine — the **AMD AI Bundle** (LM Studio, Ollama, ComfyUI: consumer apps), the **ROCm 7.14 SDK** (iGPU compute wheels via AMD Install Manager), and the NPU **driver+XRT**. None of these is the Ryzen AI SDK; no `vaip`, no VitisAI wheel, and both vendor env vars unset in every scope.
- **The decisive discovery**: AMD hosts a real package index at **`https://pypi.amd.com/simple`** carrying `onnxruntime-vitisai` (1.23.2, cp312), `voe` (1.7.0), `ryzenai-dynamic-dispatch`, `ryzenai-onnx-utils`, and others. This is a pure-wheel route with **no exe installer** — exactly matching the uv/venv constraint.
- **Three obstacles, all solved in sequence**:
  1. **uv cannot resolve the index** (`no versions of onnxruntime-vitisai`) — its simple-index pages deviate from what uv expects. Workaround: install by **direct wheel URL**, which works cleanly.
  2. **NumPy ABI**: the AMD build is compiled against NumPy 1.x; with 2.5.2 the import fails. Pin **`numpy<2`** (1.26.4 verified).
  3. **`voe` wheel packaging bug**: the wheel claims version `1.7.0` but its `.data` directory is named `voe-1.7.0.dev20260117...+g0198366.data`. The version mismatch makes installers treat the payload as an opaque directory, stranding four DLLs (~303 MB: `onnxruntime_vitisai_ep.dll`, `aiecompiler_client.dll`, `dyn_dispatch_core.dll`, `onnxruntime_vitis_ai_custom_ops.dll`) under `site-packages/voe-...data/data/lib/site-packages/onnxruntime/capi/` instead of merging them into the real `onnxruntime/capi/`. Without them, session creation dies with a native access violation *after* successful provider registration. Fix: copy the four DLLs beside the bridge. This is almost certainly the mechanism behind upstream issue #213.
- **Result with all three fixes** (ort `1.23.2.dev20260117`, pure uv venv, exit 0):
  - Guard 1: `get_available_providers()` → `['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']` ✅
  - Guard 2: `session.get_providers()` → `['VitisAIExecutionProvider', 'CPUExecutionProvider']` ✅
  - A fixed-shape `(1,512,768)` MatMul executed with max abs error 1.4e-4 versus NumPy.
- **Remaining gaps, honestly stated**:
  - `Cannot load vaiml.dll` is logged at fatal level during session creation (session continues). **VAIML is the BF16 compiler** — the flow our NLP encoder models need on STX. The wheels do not ship it; whether it comes only from the SDK exe installer or from another index package is the open question that keeps task 1.3 from closing.
  - **vaiml.dll provenance resolved (2026-09-05), exhaustively**: it is loaded by bare name (standard DLL search order — dropping it beside `onnxruntime_vitisai_ep.dll` suffices) and the EP itself embeds the full VAIML pass pipeline, so the DLL is the *only* missing piece. It ships in exactly two places per AMD docs: `%RYZEN_AI_INSTALLATION_PATH%\deployment\vaiml.dll`, and inside the **`flexml`** package (`flexml/flexml_extras/lib/vaiml.dll`). Every scriptable route is closed: no accessible AMD-index wheel contains it (verified across all NPU-relevant wheels, including a ranged read of the 190 MB voe dev wheel's zip directory); `flexml` and `quark` exist on `pypi.amd.com` but return **403 — auth-gated**; the exe installer and the **NuGet zip** (`ryzen_ai_nuget_1.8.0.zip`) both sit behind AMD's account EULA form at `account.amd.com`, and no unauthenticated `download.amd.com/opendownload` mirror exists (probed, 404). **Resolution: a one-time user click-through of the AMD EULA to download the NuGet zip**, from which the DLL is extracted into the venv — no installer executed, no conda, uv purity preserved.
  - **Partition verification was not performed** — the matmul may well have executed on the CPU inside the EP. Two-guard success proves the *plumbing*, not NPU *execution*. `xrt-smi examine --report aie-partitions` during a sustained workload is the outstanding evidence.
- **Execution-mode implication**: `IN_PROCESS` is the working verdict. The isolated backend (task 4.4) is very probably not built; if the BF16 flow ever demands a separate environment it would be a second **uv venv**, never conda (user directive, recorded in memory).
- **Hugging Face state**: token stored in `.env` (gitignored), validated as user `wishinet`. Gated `embeddinggemma-300m` still returns 403 with the token — the Gemma licence has not been accepted yet. The acceptance form is embedded at the top of the model card when logged in; there is no separate terms page.

### Third probe: NPU execution CONFIRMED end to end (2026-09-05)

- **Context**: Closing task 1.3's two open questions after the user supplied the Gemma licence acceptance and the Ryzen AI NuGet package.
- **`vaiml.dll` sourced**: extracted from `ryzen_ai_nuget_1.8.0` at `RyzenAI_Deployment.1.8.0/runtimes/win-x64/native/vaiml.dll` (266 MB) and copied beside `onnxruntime_vitisai_ep.dll`. The same folder also carries **`vaip_config.json`** (850 KB), which is the `config_file` provider option. No installer was executed; uv purity preserved.
- **Gemma gate cleared**: `config.json` returns 200 with the `.env` token.

**The synthetic probe was a false negative.** A bare FP32 MatMul, and a bare BF16 MatMul, both compiled to *zero* offloaded operators. Throughput A/B against CPU: ratio 0.97 and 1.02 — the NPU was doing nothing. VAIML fuses *structures* (attention, LayerNorm, GEMM chains); an isolated primitive matches no pattern. Any spike built on a synthetic graph would have wrongly concluded the NPU was unusable.

**Three artifacts disagree, and two of them are misleading:**

| Source | Reported | Truth |
|---|---|---|
| Console `100.00% of operations will run on AIE` | 100% AIE | ❌ fail-safe *plan*, not a verdict |
| `vaiml_partition_fe.flexml/fail_safe_summary.json` → `{"AIE":100,"CPU":0}` | 100% AIE | ❌ same plan, same lie |
| **`preliminary-vaiml-pass-summary.txt`** → `operators supported by VAIML: 0(0.000%)` | 0% | ✅ **authoritative** |

That file also reports `Model data type` and `Device data type`, which is how we confirmed `config_file` switches the target to `bfloat16`.

**With a real encoder (`all-MiniLM-L6-v2`, ONNX, shapes pinned to batch 1 / seq 128), everything works:**
- Compiler instantiated genuine BF16 AIE kernels (e.g. `SubAttributeBroadcastingBf163D` with `bfloat16,bfloat16,bfloat16`)
- Produced a **13.3 MB compiled binary `minilm.rai`** in the cache — the synthetic graph produced only JSON
- **Throughput: 132.8 inf/s on NPU vs 25.3 inf/s on CPU — a 5.24× speedup**
- Sustained 45 s run (6047 inferences) with `xrt-smi examine --report aie-partitions` showing a **live hardware context on Partition 0, columns [0..7]**, and estimated power rising from **0.001 W idle to ~0.44 W under load**

**Conclusions for the design:**
- Execution mode verdict is **`IN_PROCESS`, confirmed**. Task 4.4 (isolated worker) is not needed.
- The partition check must parse `preliminary-vaiml-pass-summary.txt`, never the console percentage or `fail_safe_summary.json`.
- A CPU throughput A/B and the `aie-partitions` hardware-context check are the two independent runtime confirmations; both belong in the capability/benchmark path.
- Energy measurement is viable and discriminating: idle-versus-load power differs by a factor of ~440, so per-1000-input energy will be meaningful.
- **The compiled artifact is `.rai`**, cached under `<cache_dir>/<cache_key>/`. This is the preparation artifact the ArtifactStore manages; its presence (rather than JSON alone) is itself a signal that compilation genuinely succeeded.
- Static shapes must be pinned before compilation — HF exports carry `batch_size`/`sequence_length` symbolic dims that must be rewritten to concrete values.

### Telemetry findings from implementing the wrapper (2026-09-05, task 1.4)

Two behaviours of `xrt-smi` surfaced during implementation and were independently reproduced by the reviewer. Both change how telemetry must be consumed.

- **`Estimated Power` intermittently reads `N/A` even where power reporting is supported.** Measured at idle on this machine: the implementer saw 4 consecutive `N/A` then 4 consecutive `0.002 W` in 8 polls; the reviewer independently saw `N/A` in 2 of 39 polls. Earlier probes recorded only steady `0.001 W` idle / `0.44 W` load, so this is new. Consequence: `N/A` is a *transient* condition, not only the permanent PHX/HPT/Linux unsupported signal, and it must never be folded to `0.0 W` — doing so would silently deflate requirement 6.2's integrated energy. Power therefore needs **three** states, not two: reported, unavailable-this-sample, and unsupported-by-platform.
- **`xrt-smi` exits 0 even when it rejects the request.** `examine --report bogus-report` prints an error and returns exit code 0. The exit status carries no validity information whatsoever; only parsing the output can establish whether a reading was obtained. Any code that trusts the return code will treat garbage as success.

Implication for task 6.2: the `PowerSampler` protocol in this design exposes `supported() -> bool` and `sample_watts() -> float | None`, which cannot express the middle state. When mapping onto `Measurement(value, unavailable_reason)`, the reason text must preserve whether a sample was *unsupported* or merely *unavailable*, so requirement 6.8's recorded omission stays specific. A mid-run `N/A` should be dropped from the integral with a missed-sample count reported in the methodology (7.2) — never interpolated.

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Ports & Adapters | `Embedder` port with backend adapters per provider | Matches 2.1 and 5.1 exactly; backends swap without touching callers; benchmark treats all three uniformly | One indirection layer | **Selected.** The requirements already force this seam |
| Direct EP calls | Session construction inline at call sites | Fewest moving parts | Cannot satisfy 5.1 without rewrite; benchmark needs three code paths | Rejected — Approach A from discovery |
| Always-on local service | Daemon owns the session; callers are clients | Session compiled once and shared | Process supervision is a new failure domain not required by any requirement | Rejected as default; retained as the *isolated* adapter only, where 5.1 conditionally demands it |

## Design Decisions

### Decision: Post-process pooling, Dense, and normalization outside the ONNX graph

- **Context**: 3.5 requires unit-normalized vectors; 5.4 requires isolated and in-process backends to agree within a tolerance.
- **Alternatives Considered**:
  1. Fold pooling/Dense/normalize into the exported graph so each backend emits final vectors.
  2. Run the transformer on the selected backend, then apply pooling/Dense/normalize identically in NumPy on the CPU.
- **Selected Approach**: Option 2. Backends return token embeddings plus the attention mask; a single shared post-processing stage produces the final vector.
- **Rationale**: Makes 3.5 true by construction for every backend, and reduces 5.4's equivalence surface to the transformer alone. Folding post-processing into the graph would multiply it across three backends and risk BF16 rounding differences in the normalization itself.
- **Trade-offs**: A small amount of CPU work per batch — negligible against a transformer pass (the pooling and Dense stages are a few thousand FLOPs against hundreds of GFLOPs). In exchange, normalization can never differ by provider.
- **Follow-up**: Confirm the Dense stage's weights are exported alongside the graph and loaded correctly; a missing Dense layer produces plausible-looking but wrong vectors, which no shape check would catch.

### Decision: Publish the compiled sequence length, not the model's architectural limit

- **Context**: 3.6 requires reporting maximum input token length; 3.7–3.8 make it a contract with `document-ingest`.
- **Alternatives Considered**:
  1. Report the model's architectural limit (2048 for EmbeddingGemma).
  2. Report the fixed length the graph was actually compiled at.
- **Selected Approach**: Option 2. Each model profile declares a `compiled_seq_len` (default 512); that value is what 3.6 reports.
- **Rationale**: NPU compilation fixes the shape. A consumer chunking to 2048 would produce inputs that are silently truncated at 512 by 3.7 — losing three quarters of every chunk while reporting success. The published number must be the one that is actually enforced.
- **Trade-offs**: Forfeits EmbeddingGemma's long-context capability. Justified: retrieval chunks are deliberately small, and shorter fixed lengths cost less compute per chunk.
- **Follow-up**: Validate that padding every input to 512 does not dominate throughput for short chunks; if it does, consider a second compiled profile at 256.

### Decision: Use ONNX Runtime EP context files as the preparation artifact

- **Context**: 4.3, 4.4, and 4.7 require persisted, reusable, invalidatable preparation artifacts.
- **Alternatives Considered**:
  1. The implicit VitisAI EP cache via `cache_dir`/`cache_key`.
  2. Explicit EP context files via `ep.context_enable` / `ep.context_file_path`.
- **Selected Approach**: Option 2, with an accompanying sidecar manifest recording model identity, compiled sequence length, batch size, provider, and toolchain versions.
- **Rationale**: 4.4 requires *reporting* that artifacts were reused, and 4.7 requires detecting staleness across model version, input length, and provider. An implicit cache keyed on a model hash can express none of that. AMD documents EP context as the production mechanism.
- **Trade-offs**: More explicit lifecycle code than letting the EP cache manage itself.
- **Follow-up**: Verify EP context files remain valid across Ryzen AI point releases; if not, the toolchain version in the manifest is the invalidation key.

### Decision: Verify NPU partitioning, not merely session creation

- **Context**: 2.2 forbids embedding on another provider when `npu` is selected; 2.6 requires reporting the provider that actually served the work.
- **Alternatives Considered**:
  1. Treat successful session creation with `VitisAIExecutionProvider` as proof of NPU execution.
  2. Additionally inspect graph partitioning and NPU occupancy after session creation.
- **Selected Approach**: Option 2. After preparation, the runtime records how much of the graph was assigned to the NPU and treats a below-threshold assignment as an NPU failure under `npu` selection.
- **Rationale**: The EP silently falls back to CPU *per subgraph*. Without this check, a model whose operators are largely unsupported would report "provider: npu" while running mostly on the CPU — precisely the silent degradation Requirement 2 exists to prevent.
- **Trade-offs**: Depends on EP diagnostics whose format is not contractually stable; the check must degrade to a warning rather than a hard failure if diagnostics are unavailable.
- **Follow-up**: Establish the partitioning threshold empirically during the spike; record the observed value in the benchmark document.

### Decision: Length-prefixed binary framing for the isolated backend

- **Context**: 5.1–5.4 require an isolated backend producing equivalent vectors across a process boundary between two different Python interpreters.
- **Alternatives Considered**:
  1. `multiprocessing.connection` (pickle-based).
  2. JSON lines carrying base64-encoded arrays.
  3. Length-prefixed frames: 4-byte length, JSON header describing shape and dtype, then a raw `float32` payload.
- **Selected Approach**: Option 3 over a loopback socket.
- **Rationale**: The worker runs under the vendor's conda interpreter, a different Python from the application's. Pickle across interpreter versions is a compatibility hazard for a channel that must be reliable, and base64 inflates payloads by a third for no benefit. Raw frames are trivially verifiable and interpreter-agnostic.
- **Trade-offs**: A small amount of hand-written framing code instead of a stdlib call.
- **Follow-up**: 5.4's tolerance must be measured, not assumed — identical ONNX and identical post-processing should yield bitwise-identical results, so any divergence indicates a real defect.

### Generalization outcomes

- Requirements 2, 4, and 5 are variations of one problem: **resolve a (model, provider) pair into a prepared, callable session**. A single resolution path serves all three; backends differ only in how they execute a prepared graph.
- Requirements 3.4 and 4.1 are variations of **model-specific behavior**. Captured as a declarative `ModelProfile` (prefix templates, pooling strategy, Dense presence, dimension, compiled length, batch size) so three candidates require three data entries, not three code paths.
- Requirements 6 and 7 are **consumers** of the embedding service under instrumentation, not a mode of it. The benchmark gets no privileged access, which keeps the measured path identical to the production path.

### Simplification outcomes

- No model registry, plugin discovery, or configuration DSL. Three profiles in a module-level mapping.
- No abstraction over power sampling beyond one narrow interface with a real implementation and an explicit "unavailable" case, because 6.8 already defines the degraded behavior.
- The isolated backend is designed but its implementation is **gated on the spike outcome**. If the EP registers in the application environment, it is not built. Designing it costs a protocol definition; building it speculatively would violate the executability review.
- No retry, circuit-breaker, or rate-limiting machinery. This is a local, single-user, offline batch process.

## Risks & Mitigations

- **The EP does not register in a `uv` environment (#213)** — Mitigation: the spike is the first task and gates everything downstream; the isolated backend is a designed, costed fallback; the capability check reports which mode is viable (1.5).
- **Silent per-subgraph CPU fallback inside an NPU session** — Mitigation: partitioning verification with a threshold, reported per run.
- **Missing Dense stage yields wrong-but-plausible vectors** — Mitigation: retrieval-quality measurement (6.4) against a known-relevant query set catches semantically broken vectors that shape and norm checks cannot.
- **CPU-provider energy is unmeasurable by `xrt-smi`** — Mitigation: 6.8 records the omission explicitly; wall-clock and throughput remain comparable, and the benchmark states the limitation rather than estimating.
- **Polled power sampling understates short bursts** — Mitigation: state sampling interval and integration method in the methodology (7.2); report variance across repetitions (6.7).
- **Gated model download blocks an unattended run** — Mitigation: 4.5 requires reporting the licensing step specifically; the capability check surfaces credential state before a long run begins.
- **Driver upgrade may regress or fail on this OEM handheld** — Mitigation: record the working driver version in the benchmark document (7.5) so a regression is identifiable.

## References

- [Ryzen AI Software 1.8 — Installation](https://ryzenai.docs.amd.com/en/latest/inst.html) — driver minimum, installer behavior, standalone wheels
- [Ryzen AI Software 1.8 — Model Run](https://ryzenai.docs.amd.com/en/latest/modelrun.html) — BF16/INT8 flows, provider options, EP context caching
- [Ryzen AI Software 1.8 — NPU Management Interface](https://ryzenai.docs.amd.com/en/latest/xrt_smi.html) — `xrt-smi` power reporting, STX-only support, polling guidance
- [Ryzen AI Software 1.8 — Release Notes](https://ryzenai.docs.amd.com/en/latest/relnotes.html) — model-type support matrix, supported processors
- [Vitis AI Execution Provider — ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html) — Python session API and provider options
- [RyzenAI-SW issue #213](https://github.com/amd/RyzenAI-SW/issues/213) — EP unregistered in user-managed Python environment (open)
- [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) — dimensions, Matryoshka, context, prompt templates, licensing
- [Sentence Transformers — Speeding up Inference](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html) — ONNX export scope, pooling and normalization as separate stages
