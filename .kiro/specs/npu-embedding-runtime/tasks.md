# Implementation Plan

> **Gating note**: Task 1.3 decides whether NPU execution is reachable in-process or only in isolation. Task 4.4 is conditional on its outcome and must not be implemented speculatively. Everything from major task 3 onward assumes 1.3 has produced a verdict.

- [x] 1. Foundation: environment, provisioning, and the gating spike

- [x] 1.1 Establish the Python project baseline
  - Create the project manifest with a uv-managed dependency set and the package skeleton for the embedding domain
  - Declare the baseline dependency group; keep vendor-runtime packages out of the default group so packaging stays viable for downstream consumers
  - Configure the test runner and type checking so later tasks have somewhere to put tests
  - Observable: a clean checkout installs with uv, imports the embedding package, and runs an empty test suite green
  - _Requirements: 4.1_

- [x] 1.2 Provision the vendor runtime wheels into the uv environment
  - Install the provider wheels from the vendor's package index (verified working 2026-09-04: direct wheel URLs from pypi.amd.com, since uv cannot resolve that index's pages), with the numpy-below-2 pin the vendor build requires
  - Apply the voe wheel workaround: relocate the four backend libraries stranded by the wheel's mismatched data-directory version into the runtime's provider directory, and encode this step so it survives environment rebuilds
  - This machine is uv/venv only — no conda anywhere, and no exe installer unless the BF16 compiler library proves unobtainable any other way
  - Record the install steps, wheel versions, and workaround in reproducible form
  - Observable: a freshly created uv environment reaches provider registration by following the recorded steps alone, and the workaround is applied automatically rather than by hand
  - _Requirements: 1.6_

- [x] 1.3 Spike: confirm NPU execution from a uv-managed environment — **CLOSED 2026-09-05, verdict IN_PROCESS**
  - Provider registers and is selected in a pure uv venv; the BF16 compiler library was sourced from the vendor NuGet package (no installer run) together with the provider configuration file it needs
  - Confirmed with a real encoder at pinned static shapes: genuine reduced-precision kernels instantiated, a compiled binary artifact produced, and roughly five times the CPU throughput
  - Confirmed at the hardware level: a live context across all accelerator columns during a sustained run, with power rising from idle by more than two orders of magnitude
  - Established that a synthetic single-operator graph is a false negative — the compiler fuses structures, not isolated primitives — and that the console offload percentage and the fail-safe summary both misreport; only the preliminary pass summary is authoritative
  - Observable: verdict recorded in the research log with throughput ratio, compiled-artifact presence, and captured hardware-context evidence
  - _Requirements: 1.5_

- [x] 1.4 Wrap the vendor management utility for telemetry reads
  - Invoke the platform report to read estimated power, and the partition report to read column occupancy
  - Parse both into typed values, treating an absent utility or unsupported reading as data rather than an exception
  - Support repeated polling so a caller can sample during a running workload
  - Observable: polling returns a plausible watt value on this machine, and returns an explicit unsupported result rather than raising when power reporting is unavailable
  - _Requirements: 1.1, 6.2_
  - _Boundary: XrtSmiWrapper_

- [x] 1.5 Implement the environment capability check
  - Evaluate device presence, runtime installation, driver minimum, provider registration, and required environment variables as independent conditions
  - Carry observed value, required value, and a remediation step on every unsatisfied condition
  - Derive the execution-mode verdict from provider registration, formalizing what the spike established
  - Compare driver versions component-wise rather than lexicographically, covering the installed-versus-required case observed on this machine
  - Observable: the check runs to completion on a machine with no NPU and no vendor runtime, reporting each condition separately instead of raising
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - _Boundary: CapabilityChecker_
  - _Depends: 1.3, 1.4_

- [x] 1.6 Establish model-repository access for the gated primary candidate — **CLOSED 2026-09-05**
  - Credential stored in the project's gitignored environment file and verified against the expected account
  - Licence accepted for the gated primary candidate; access verified as succeeding where it previously returned an authorisation failure
  - Remaining for implementation: load the credential from the environment file in tooling rather than requiring a shell-profile variable
  - Observable: fetching the primary candidate's configuration with the stored credential succeeds, and both open comparators remain fetchable without one
  - _Requirements: 4.5_

- [ ] 2. Foundation: shared contracts

- [ ] 2.1 Define the domain types and the error taxonomy
  - Model text kind, provider choice, execution mode, and the document form that carries a title alongside content
  - Structure errors so environment, preparation, and execution are distinguishable by type rather than by message text
  - Carry provider, model, and failing stage on every error
  - Observable: catching the environment error category succeeds without catching preparation or execution failures, and every error instance exposes provider, model, and stage
  - _Requirements: 8.1, 8.2_

- [ ] 2.2 (P) Declare the three candidate model profiles
  - Capture dimension, compiled sequence length, batch size, pooling strategy, presence of a dense stage, and license gating per model
  - Encode the document and query templates for each model, including the title slot the document form requires
  - Treat the primary candidate as the initial default until the benchmark supersedes it
  - Observable: each profile reports a compiled sequence length distinct from the model's architectural context limit, and the primary candidate's templates match the published conventions exactly
  - _Requirements: 3.4, 3.6, 4.1, 4.2_
  - _Boundary: ModelProfiles_
  - _Depends: 2.1_

- [ ] 2.3 (P) Build progress reporting and run summaries
  - Provide a progress callback carrying completed and remaining counts, usable identically by every backend
  - Summarize a completed run with elapsed time, input count, provider served, execution mode, and truncation count
  - Report completed count when an operation is interrupted before finishing
  - Observable: a long batch emits monotonically increasing progress, and an interrupted batch reports how many inputs finished
  - _Requirements: 2.6, 5.6, 8.3, 8.4, 8.6_
  - _Boundary: reporting_
  - _Depends: 2.1_

- [ ] 3. Model preparation

- [ ] 3.1 Implement model acquisition with license-gate handling
  - Download model weights and record the resolved revision so a silently changed upstream artifact is detectable
  - Detect a gated repository whose terms have not been accepted, and surface the acceptance requirement specifically
  - Observable: requesting the gated primary candidate without accepted terms raises the licensing error carrying the acceptance step, not a generic transport failure
  - _Requirements: 4.5_
  - _Boundary: ArtifactStore_

- [ ] 3.2 Export models to ONNX at a fixed sequence length
  - Export the transformer trunk at the profile's compiled sequence length and batch size, keeping the graph at full precision
  - Extract and persist the dense-stage weights separately, since the export covers only the trunk
  - Report the failing stage when export cannot complete, without substituting another model
  - Observable: the exported graph accepts exactly the profile's declared input shape and rejects any other, and dense weights are present for every profile that declares a dense stage
  - _Requirements: 4.6_
  - _Boundary: ArtifactStore_

- [ ] 3.3 Build the artifact store with manifest-based invalidation
  - Compile the exported graph for the NPU and persist the resulting context snapshot
  - Write a manifest recording model identity and revision, compiled sequence length, batch size, provider, and toolchain versions
  - Decide reuse by comparing the manifest against the current profile and toolchain, and report when artifacts were reused
  - Write to a temporary location and rename on success, so nothing partial survives an interruption
  - Observable: changing any one manifest field independently forces a rebuild, a repeated preparation reuses artifacts with near-zero elapsed time and says so, and a run killed mid-preparation leaves no directory a later run accepts
  - _Requirements: 4.3, 4.4, 4.6, 4.7, 8.5_
  - _Boundary: ArtifactStore_

- [ ] 4. Execution backends

- [ ] 4.1 Define the backend protocol and provider resolution policy
  - Specify a backend contract that returns token embeddings and the attention mask, and performs no pooling or normalization
  - Resolve an explicit NPU request to a failure when the NPU is unusable, never to a substitute
  - Prefer the NPU under automatic selection and return the reason whenever the CPU is used instead
  - Bind the resolved backend for the whole operation so it cannot change mid-run
  - Observable: requesting the NPU on a machine where it is unavailable raises rather than returning CPU vectors, and automatic selection returns a non-empty reason string exactly when it falls back
  - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.7_
  - _Boundary: TransformerBackend_

- [ ] 4.2 (P) Implement the CPU backend as the full-precision reference
  - Execute the exported graph through the default runtime provider at full precision
  - Serve forced CPU selection regardless of NPU availability
  - Observable: the CPU backend produces vectors for the same inputs the NPU backend accepts, at the same shape, and is selectable even when the NPU is present and healthy
  - _Requirements: 2.3_
  - _Boundary: CpuBackend_
  - _Depends: 3.3, 4.1_

- [ ] 4.3 (P) Implement the NPU backend with partition verification
  - Construct the session against the vendor provider with reduced-precision targeting supplied through the provider configuration
  - Measure how much of the graph was assigned to the NPU after preparation, and treat a below-threshold assignment as a failure under explicit NPU selection
  - Treat unverifiable partitioning as a failure under explicit NPU selection and a recorded warning under automatic selection
  - Report whether verification actually happened, so an assumed NPU run is distinguishable from a verified one
  - Observable: a run under explicit NPU selection either reports verified partitioning above threshold or fails, and never silently proceeds with the graph mostly on the CPU
  - _Requirements: 2.2, 2.6_
  - _Boundary: VitisAIBackend_
  - _Depends: 3.3, 4.1_

- [ ] 4.4 Implement the isolated backend and its worker
  - **Conditional on task 1.3.** If the spike concluded in-process execution works, record this task as not applicable and skip it rather than building an unused path
  - Exchange length-prefixed frames carrying a descriptive header and a raw numeric payload over a loopback socket, avoiding any serialization that assumes matching interpreter versions
  - Exchange a version handshake in the first frame so a mismatched worker fails loudly rather than returning subtly wrong vectors
  - Resolve the worker-side deployment and dependency pinning, which the design deliberately deferred to this point
  - Report isolated execution on every operation it serves, emit progress on the same terms as in-process execution, and report worker failure without promoting another provider
  - Observable: the isolated backend returns vectors for the same inputs as the in-process path, reports its execution mode, and a killed worker surfaces a worker error rather than CPU results
  - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.6_
  - _Boundary: IsolatedBackend_
  - _Depends: 1.3, 4.1_

- [ ] 5. Embedding service

- [ ] 5.1 (P) Expose the tokenizer and the length-measurement contract
  - Publish the active model's tokenizer so a consumer can measure input length by the same rule the service applies
  - Provide token counting that accounts for the prefix template, since the template consumes budget the caller cannot see
  - Observable: counting tokens for a text and then embedding that same text agree on whether it exceeds the limit, across a sample of real content
  - _Requirements: 3.7, 3.8_
  - _Boundary: tokenize_
  - _Depends: 2.2_

- [ ] 5.2 (P) Implement post-processing shared by all backends
  - Apply mean pooling that respects the attention mask, so padding contributes nothing under fixed-length inputs
  - Apply the dense stage where the profile declares one, then normalize to unit length
  - Keep this path independent of which backend produced the token embeddings
  - Observable: pooled output for a padded input matches a hand-computed reference, every produced vector has unit norm within floating-point tolerance, and embedding the same text twice returns identical values
  - _Requirements: 3.5, 3.10_
  - _Boundary: postprocess_
  - _Depends: 2.1_

- [ ] 5.3 Assemble the embedding service over backends and post-processing
  - Offer separate entry points for corpus content and for queries, so omitting the text kind is impossible rather than merely rejected
  - Apply the profile's template per kind, populating the title slot for corpus content
  - Report the contract values, publishing the compiled sequence length rather than the model's architectural limit
  - Shorten over-length inputs and identify which ones were shortened
  - Return vectors in input order, one per input, alongside the provider served, execution mode, partition verification state, and elapsed time
  - Observable: the service returns exactly one vector per input in order, names the serving provider on success as well as failure, and reports the compiled length as its maximum
  - _Requirements: 2.6, 2.7, 3.1, 3.2, 3.3, 3.4, 3.6, 3.9, 5.3, 8.3, 8.4, 8.6_
  - _Boundary: EmbeddingService_
  - _Depends: 4.1, 5.1, 5.2, 2.3_

- [ ] 6. Benchmark instrumentation

- [ ] 6.1 (P) Build the committed benchmark fixture
  - Generate a small pre-extracted text sample from the target archive using a throwaway script, then commit the result as static test data
  - Hand-build a query set with pre-identified relevant entries drawn from that sample
  - Record the sample's size and composition alongside the data
  - Observable: the fixture loads without touching any source document, so the benchmark carries no dependency on document extraction and this spec stays independently implementable
  - _Requirements: 6.5_
  - _Boundary: bench fixtures_

- [ ] 6.2 (P) Implement energy sampling with an explicit unavailable case
  - Sample NPU power by polling the management utility concurrently with the measured workload, never inside the measured call
  - Integrate sampled power over wall-clock into energy per unit of work, recording the sampling interval used
  - Model a measurement so that a value and an unavailability reason are mutually exclusive, and record the CPU provider's energy as unavailable with that reason
  - Observable: an NPU run yields a non-null energy figure while the equivalent CPU run yields an explicit unavailable reason, and neither is ever an estimate
  - _Requirements: 6.2, 6.8_
  - _Boundary: PowerSampler_
  - _Depends: 1.4_

- [ ] 6.3 Drive the model and provider matrix and capture per-run metrics
  - Drive each candidate model against each provider through the ordinary service interface, with no privileged access, so the measured path is the production path
  - Measure throughput, single-input latency at median and 95th percentile, peak resident memory, and preparation time cold versus warm
  - Observable: a single pass produces one metric record per model and provider combination, each carrying all four metric families
  - _Requirements: 6.1, 6.2, 6.5_
  - _Boundary: BenchmarkHarness_
  - _Depends: 5.3, 6.1, 6.2_

- [ ] 6.4 Add repetition, variance, run provenance, and per-cell failure tolerance
  - Repeat each cell more than once and report the observed variation across repetitions
  - Capture hardware identity, driver version, and runtime version once per run
  - Continue the matrix when one cell fails, recording that cell's failure rather than aborting the run
  - Observable: a run in which one model is deliberately made unusable still completes every other cell, and each completed cell reports its repetition count and observed spread
  - _Requirements: 6.6, 6.7_
  - _Boundary: BenchmarkHarness_

- [ ] 6.5 (P) Measure reduced-precision fidelity and cross-backend equivalence
  - Compare NPU-produced vectors against full-precision CPU reference vectors for identical inputs
  - Compare isolated-backend vectors against in-process vectors where the isolated path exists, expecting exact agreement given a shared graph and shared post-processing
  - Observable: fidelity against the full-precision reference is reported as a similarity distribution rather than a single number, and any cross-backend divergence is surfaced as a defect rather than absorbed into a tolerance
  - _Requirements: 5.4, 6.3_
  - _Boundary: bench fidelity_
  - _Depends: 6.3_

- [ ] 6.6 (P) Score retrieval quality per candidate model
  - Rank the fixture sample against the fixture query set in memory, without any persistent index
  - Score each candidate model so the energy-versus-quality trade-off is visible rather than assumed
  - Observable: a model whose dense stage is omitted scores visibly worse, confirming the measurement detects semantically broken vectors that shape and norm checks cannot
  - _Requirements: 6.4_
  - _Boundary: bench quality_
  - _Depends: 6.3_

- [ ] 6.7 Render the benchmark document
  - Present every recorded measurement with NPU and CPU side by side so the throughput and energy difference is directly readable
  - State the methodology, covering power sampling, sample selection, and repetition count
  - State a recommended default model with its reasoning, and the conditions under which a different candidate would be better
  - Record hardware, driver, and runtime versions, and state any model or measurement that could not be completed
  - Observable: the rendered document contains no estimated values, every omission carries its reason, and the recommendation is traceable to figures present in the same document
  - _Requirements: 6.8, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_
  - _Boundary: BenchmarkReport_
  - _Depends: 6.4, 6.5, 6.6_

- [ ] 7. Integration and deliverable production

- [ ] 7.1 Verify the explicit-provider contract end to end
  - Exercise the full path from capability check through preparation to embedding, under each provider selection
  - Confirm that an explicit NPU request on a machine with the NPU made unavailable produces no vectors at all
  - Confirm that automatic selection reports its fallback reason and still names the serving provider on success
  - Observable: the forced-NPU path fails without producing vectors, proving no silent fallback exists anywhere in the assembled system
  - _Requirements: 2.2, 2.6_
  - _Depends: 5.3, 4.3_

- [ ] 7.2 Execute the full benchmark and produce the written deliverable
  - Run the complete matrix on the target machine and render the document from the recorded results
  - Confirm the recommended default is supported by the measured figures rather than by prior assumption
  - Observable: the benchmark document exists, covers all three candidates across both providers, and states a default model chosen from evidence gathered on this machine
  - _Requirements: 6.1, 7.1, 7.3_
  - _Depends: 6.7, 7.1_

- [ ] 8. Validation

- [ ] 8.1 Verify the tokenizer contract against a content sample
  - Assert that consumer-side token counting and the service's truncation decision agree for the same text and kind across a sample of real content
  - Cover text that sits near the compiled length boundary, where prefix overhead decides the outcome
  - Observable: agreement holds across the sample, establishing the contract the document-ingest spec will size its chunks against
  - _Requirements: 3.8_
  - _Depends: 5.1, 5.3_

- [ ] 8.2 (P) Verify artifact lifecycle and preparation failure handling
  - Confirm each manifest field independently forces a rebuild, and that a warm run reports reuse
  - Confirm an interrupted preparation leaves nothing a later run treats as valid
  - Confirm gated acquisition without accepted terms raises the licensing error rather than a transport error
  - Observable: every invalidation dimension is covered by a passing test, and the interrupted-preparation case leaves a clean artifact directory
  - _Requirements: 4.3, 4.4, 4.5, 4.7, 8.5_
  - _Boundary: ArtifactStore_
  - _Depends: 3.3_

- [ ] 8.3 (P) Verify error categorization and padding-sensitive pooling
  - Confirm the three error categories are separable by type and that each instance carries provider, model, and stage
  - Confirm masked mean pooling ignores padding under fixed-length inputs, comparing against a hand-computed reference
  - Confirm repeated embedding of identical text returns identical vectors
  - Observable: pooling output matches the reference for a deliberately short input padded to full compiled length, the case where mean pooling silently breaks
  - _Requirements: 3.5, 3.10, 8.1, 8.2_
  - _Boundary: postprocess, errors_
  - _Depends: 5.2, 2.1_

## Implementation Notes
- 1.1: canonical validation set = `uv sync`, `uv run pytest`, `uv run mypy`, `uv run python -c "import npu_rag.embedding"`; vendor group via `uv sync --group npu`.
- 1.1: `numpy<2` is pinned project-wide (not only in the npu group) so adding vendor wheels later cannot force a NumPy 2.x ABI break.
- 1.1: stock `onnxruntime` and vendor `onnxruntime-vitisai` both own the `onnxruntime` import package - task 1.2 must decide replacement vs coexistence, not install both blindly.
- 1.1: `optimum` resolved to 2.x and `transformers` to 5.x, beyond design.md assumptions - task 3.2 must re-verify the ONNX export API or pin `optimum<2`.
- 1.1: tasks.md cites `_Requirements: 4.1_` for 1.1 but design.md maps 4.1 to profiles.py (task 2.2); 1.1's true anchor is the Boundary Commitments pyproject bullet. Fix the citation when 2.2 lands.
- 1.1: reviewer note - the import-boundary AST guard in tests skips relative imports and passes vacuously if the package is deleted; harden when providers/base.py lands.
- 1.2: provisioning lives in `tools/`, NOT in the package, because design.md Out of Boundary says this spec detects and reports environment state and does not mutate the system. design.md's File Structure Plan has no `tools/` entry and says "Modified Files: None" - that is known drift, not unmanaged scope. Record it in design.md when a later task touches that section.
- 1.2: an explicit `uv sync` DROPS the npu group and leaves `onnxruntime` unimportable (`AttributeError: no attribute '__version__'`). Re-run `uv run python -m tools.provision_npu` to repair. Measured correction: `uv run pytest` and `uv run mypy` do NOT de-provision - only an explicit `uv sync` does.
- 1.2: the NuGet native dir ships its own build of the four stranded DLLs, DIFFERENT from the voe wheel's (EP is 119 MB vs 184 MB). The provenance split - four DLLs from the voe wheel, vaiml.dll + vaip_config.json from NuGet - is load-bearing and must not be "simplified" later.
- 1.2: stock `onnxruntime` and vendor `onnxruntime-vitisai` own the same import package; resolution is REPLACEMENT, enforced by verify-and-repair rather than install order. A stale `onnxruntime-<stock>.dist-info` residue persists claiming stock ownership over vendor files and cannot be removed without deleting vendor files.
- 1.2: `vaip_config.json` resolves at `Path(onnxruntime.__file__).parent / 'capi' / 'vaip_config.json'` - this is the `config_file` provider option contract for task 4.3.
- 1.4: `xrt-smi` exits 0 even when it REJECTS the requested report - the exit code carries no validity information, only parsing does. Never trust returncode from this tool.
- 1.4: `Estimated Power` intermittently reads `N/A` at idle even where power reporting IS supported (2/39 polls measured). `N/A` must never be folded to 0.0 W. Power needs three states: reported / unavailable-this-sample / unsupported-by-platform.
- 1.4: for task 6.2 - design.md's `PowerSampler` (supported()->bool, sample_watts()->float|None) cannot express the middle state. When mapping to `Measurement`, keep UNSUPPORTED vs UNAVAILABLE distinguishable in the reason text so requirement 6.8's omission stays specific; drop mid-run N/A samples from the integral and report a missed-sample count, never interpolate.
- 1.4: the AST guard in tests/embedding/test_package_baseline.py enforces only the OUTER package boundary (no non-embedding `npu_rag.*`); it permits ANY intra-embedding import, so it does not enforce design.md's layer order. A module-scoped layer guard lives in test_xrt.py; a package-wide one belongs with the deferred hardening when providers/base.py lands.
- 1.5: design.md drift - the File Structure Plan lists `CapabilityReport` under `types.py`, but the CapabilityChecker section sketches `ExecutionMode`/`Condition`/`CapabilityReport` inline. They live in `capability.py` because `types.py` belongs to task 2.1. When 2.1 lands, decide: move and re-export, or record the drift in design.md.
- 1.5: `VENDOR_PAYLOAD_FILES` in capability.py duplicates the payload filenames in tools/provision_npu.py. The duplication is FORCED by design.md Out of Boundary (the package must not depend on tools/). The two lists must be changed together.
- 1.5: reviewer left 5 non-blocking test-hardening suggestions (empty-string remediation invariant, driver-exactly-equal boundary, hollow-onnxruntime branch, `_payload_state(None)`, and wrapping the `XrtSmiWrapper()` construction in a guard so the never-raise promise is structural rather than dependent on xrt.py internals). Pick these up if a later task hardens `environment/`.
