# Implementation Plan

> **Gating note**: Task 1.3 decides whether NPU execution is reachable in-process or only in isolation. Task 4.4 is conditional on its outcome and must not be implemented speculatively. Everything from major task 3 onward assumes 1.3 has produced a verdict.

- [x] 1. Foundation: environment, provisioning, and the gating spike

- [x] 1.1 Establish the Python project baseline
  - Create the project manifest with a uv-managed dependency set and the package skeleton for the embedding domain
  - Declare the baseline dependency group; keep vendor-runtime packages out of the default group so packaging stays viable for downstream consumers
  - Configure the test runner and type checking so later tasks have somewhere to put tests
  - Observable: a clean checkout installs with uv, imports the embedding package, and runs an empty test suite green
  - _Requirements: 4.1_
  - Citation note 2026-09-05: 4.1 ("support three candidate models") is a poor anchor for scaffolding; design.md maps 4.1 to profiles.py, i.e. task 2.2, now complete. Task 1.1's true anchor is the Boundary Commitments bullet "Creation of pyproject.toml with baseline project metadata and this feature's dependency group". The citation is retained so requirement-coverage tooling stays whole, and the mismatch is recorded rather than silently reassigned.

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

- [x] 2. Foundation: shared contracts

- [x] 2.1 Define the domain types and the error taxonomy
  - Model text kind, provider choice, execution mode, and the document form that carries a title alongside content
  - Structure errors so environment, preparation, and execution are distinguishable by type rather than by message text
  - Carry provider, model, and failing stage on every error
  - Observable: catching the environment error category succeeds without catching preparation or execution failures, and every error instance exposes provider, model, and stage
  - _Requirements: 8.1, 8.2_

- [x] 2.2 (P) Declare the three candidate model profiles
  - Capture dimension, compiled sequence length, batch size, pooling strategy, presence of a dense stage, and license gating per model
  - Encode the document and query templates for each model, including the title slot the document form requires
  - Treat the primary candidate as the initial default until the benchmark supersedes it
  - Observable: each profile carries the compiled sequence length and the architectural context limit as separate fields and publishes the compiled one as the maximum input token length, and the primary candidate's templates match the published conventions exactly
  - Spec correction 2026-09-05: the original wording required the two lengths to be "distinct". That is unsatisfiable for bge-large-en-v1.5, whose architectural limit genuinely IS 512, so meeting it literally would have required falsifying real model data. Requirement 3.6 asks only that the maximum be reported, and research.md's decision requires that the reported value be the compiled length; neither requires the two to differ.
  - _Requirements: 3.4, 3.6, 4.1, 4.2_
  - _Boundary: ModelProfiles_
  - _Depends: 2.1_

- [x] 2.3 (P) Build progress reporting and run summaries
  - Provide a progress callback carrying completed and remaining counts, usable identically by every backend
  - Summarize a completed run with elapsed time, input count, provider served, execution mode, and truncation count
  - Report completed count when an operation is interrupted before finishing
  - Observable: a long batch emits monotonically increasing progress, and an interrupted batch reports how many inputs finished
  - _Requirements: 2.6, 5.6, 8.3, 8.4, 8.6_
  - _Boundary: reporting_
  - _Depends: 2.1_

- [x] 3. Model preparation

- [x] 3.1 Implement model acquisition with license-gate handling
  - Download model weights and record the resolved revision so a silently changed upstream artifact is detectable
  - Detect a gated repository whose terms have not been accepted, and surface the acceptance requirement specifically
  - Observable: requesting the gated primary candidate without accepted terms raises the licensing error carrying the acceptance step, not a generic transport failure
  - _Requirements: 4.5_
  - _Boundary: ArtifactStore_

- [x] 3.2 Export models to ONNX at a fixed sequence length
  - Export the transformer trunk at the profile's compiled sequence length and batch size, keeping the graph at full precision
  - Extract and persist the dense-stage weights separately, since the export covers only the trunk
  - Report the failing stage when export cannot complete, without substituting another model
  - Observable: the exported graph accepts exactly the profile's declared input shape and rejects any other, and dense weights are present for every profile that declares a dense stage
  - _Requirements: 4.6_
  - _Boundary: ArtifactStore_

- [x] 3.3 Build the artifact store with manifest-based invalidation
  - Compile the exported graph for the NPU and persist the resulting context snapshot
  - Write a manifest recording model identity and revision, compiled sequence length, batch size, provider, and toolchain versions
  - Decide reuse by comparing the manifest against the current profile and toolchain, and report when artifacts were reused
  - Write to a temporary location and rename on success, so nothing partial survives an interruption
  - Observable: changing any one manifest field independently forces a rebuild, a repeated preparation reuses artifacts with near-zero elapsed time and says so, and a run killed mid-preparation leaves no directory a later run accepts
  - _Requirements: 4.3, 4.4, 4.6, 4.7, 8.5_
  - _Boundary: ArtifactStore_

- [x] 4. Execution backends

- [x] 4.1 Define the backend protocol and provider resolution policy
  - Specify a backend contract that returns token embeddings and the attention mask, and performs no pooling or normalization
  - Resolve an explicit NPU request to a failure when the NPU is unusable, never to a substitute
  - Prefer the NPU under automatic selection and return the reason whenever the CPU is used instead
  - Bind the resolved backend for the whole operation so it cannot change mid-run
  - Observable: requesting the NPU on a machine where it is unavailable raises rather than returning CPU vectors, and automatic selection returns a non-empty reason string exactly when it falls back
  - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.7_
  - _Boundary: TransformerBackend_

- [x] 4.2 (P) Implement the CPU backend as the full-precision reference
  - Execute the exported graph through the default runtime provider at full precision
  - Serve forced CPU selection regardless of NPU availability
  - Observable: the CPU backend produces vectors for the same inputs the NPU backend accepts, at the same shape, and is selectable even when the NPU is present and healthy
  - _Requirements: 2.3_
  - _Boundary: CpuBackend_
  - _Depends: 3.3, 4.1_

- [x] 4.3 (P) Implement the NPU backend with partition verification
  - Construct the session against the vendor provider with reduced-precision targeting supplied through the provider configuration
  - Measure how much of the graph was assigned to the NPU after preparation, and treat a below-threshold assignment as a failure under explicit NPU selection
  - Treat unverifiable partitioning as a failure under explicit NPU selection and a recorded warning under automatic selection
  - Report whether verification actually happened, so an assumed NPU run is distinguishable from a verified one
  - Observable: a run under explicit NPU selection either reports verified partitioning above threshold or fails, and never silently proceeds with the graph mostly on the CPU
  - _Requirements: 2.2, 2.6_
  - _Boundary: VitisAIBackend_
  - _Depends: 3.3, 4.1_

- [x] 4.4 Implement the isolated backend and its worker - **NOT APPLICABLE, closed 2026-09-06 without implementation**
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

- [x] 5.1 (P) Expose the tokenizer and the length-measurement contract
  - Publish the active model's tokenizer so a consumer can measure input length by the same rule the service applies
  - Provide token counting that accounts for the prefix template, since the template consumes budget the caller cannot see
  - Observable: counting tokens for a text and then embedding that same text agree on whether it exceeds the limit, across a sample of real content
  - _Requirements: 3.7, 3.8_
  - _Boundary: tokenize_
  - _Depends: 2.2_

- [x] 5.2 (P) Implement post-processing shared by all backends
  - Apply mean pooling that respects the attention mask, so padding contributes nothing under fixed-length inputs
  - Apply the dense stage where the profile declares one, then normalize to unit length
  - Keep this path independent of which backend produced the token embeddings
  - Observable: pooled output for a padded input matches a hand-computed reference, every produced vector has unit norm within floating-point tolerance, and embedding the same text twice returns identical values
  - _Requirements: 3.5, 3.10_
  - _Boundary: postprocess_
  - _Depends: 2.1_

- [x] 5.3 Assemble the embedding service over backends and post-processing
  - Offer separate entry points for corpus content and for queries, so omitting the text kind is impossible rather than merely rejected
  - Apply the profile's template per kind, populating the title slot for corpus content
  - Report the contract values, publishing the compiled sequence length rather than the model's architectural limit
  - Shorten over-length inputs and identify which ones were shortened
  - Return vectors in input order, one per input, alongside the provider served, execution mode, partition verification state, and elapsed time
  - Observable: the service returns exactly one vector per input in order, names the serving provider on success as well as failure, and reports the compiled length as its maximum
  - _Requirements: 2.6, 2.7, 3.1, 3.2, 3.3, 3.4, 3.6, 3.9, 5.3, 8.3, 8.4, 8.6_
  - _Boundary: EmbeddingService_
  - _Depends: 4.1, 5.1, 5.2, 2.3_

- [x] 5.4 Close the provider-resolution and assembly defects found at feature validation
  - Added 2026-09-07 from `/kiro-validate-impl`'s NO-GO. Each item is a confirmed defect in already-shipped code that no open task owned; none is new scope.
  - Make an isolated-execution verdict a state provider resolution can actually act on. Task 4.4 closed without building an isolated backend, but the resolution policy still treats that verdict as "the NPU can serve", so `auto` binds the NPU factory, pays a full compile, and only then fails. Under that verdict `auto` must report the specific reason and fall back to the CPU without preparing anything, and an explicit `npu` request must be refused at resolution rather than at session construction
  - Refuse to construct the embedding service with an empty Dense stage for a profile that declares one. The trunk width equals the published dimension, so an omitted Dense stage yields correctly-shaped, correctly-normalised, semantically wrong vectors that no shape, dtype or norm check can catch — and task 6.3 adds a second construction site
  - Bring the capability check's vendor payload list into agreement with what provisioning installs. The check covers three files where provisioning installs six; the three it omits are those whose absence lets the provider register and then die in native code at session creation — the exact failure the check exists to pre-empt
  - Observable: given a capability report naming isolated execution, `auto` returns CPU-served vectors carrying the reason and performs no preparation, explicit `npu` fails before preparing, and constructing the service for a Dense-stage profile without its layers is refused
  - _Requirements: 1.1, 1.4, 2.2, 2.4, 2.5_
  - _Boundary: provider resolution, EmbeddingService, CapabilityChecker_
  - _Depends: 4.1, 4.3, 5.3_

- [ ] 5.5 Replace the third candidate model and generalise pooling
  - Moved here 2026-09-07 from the "Model lineup amended" appendix, which assigned it to task 6.1. It does not fit that task's `bench fixtures` boundary: this work is in `profiles.py` and `postprocess.py`, and it is what requirement 4.1 actually asks for.
  - Dispatch pooling from the profile instead of hardcoding it, and add a CLS branch taking the first position without consulting the attention mask, so it cannot reintroduce the mask-blind pooling hazard. A profile declaring a rule the post-processor does not implement must fail loudly rather than be silently mean-pooled — the field has no production consumer today, so widening the literal alone would change nothing
  - Replace the bge-large profile with `gte-modernbert-base`: 768-dimensional, 8192 architectural context compiled at the fixed length, Apache-2.0 and ungated, no Dense stage, symmetric identity templates on both sides
  - Export and NPU-compile it once to confirm the path works end to end, as tasks 3.2 and 3.3 did for the control model
  - Update the profile-set test and every remaining bge-large reference, including the live tests that fetch it over the network
  - Observable: a fixture where CLS and mean pooling give different vectors proves the branch is selected by the profile rather than reachable only in principle; the identity-template case exercises the template machinery's no-op path, which nothing currently does
  - _Requirements: 3.4, 3.5, 4.1_
  - _Boundary: profiles, postprocess_
  - _Depends: 2.2, 3.2, 3.3, 5.2_

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
  - _Depends: 5.3, 5.5, 6.1, 6.2_

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
- 1.5: design.md drift - the File Structure Plan lists `CapabilityReport` under `types.py`, but the CapabilityChecker section sketches `ExecutionMode`/`Condition`/`CapabilityReport` inline. They live in `capability.py` because `types.py` belongs to task 2.1. RESOLVED by 2.1: moved to types.py and re-exported from capability.py; identity preserved. design.md's File Structure Plan now matches reality for these three types.
- 1.5: `VENDOR_PAYLOAD_FILES` in capability.py duplicates the payload filenames in tools/provision_npu.py. The duplication is FORCED by design.md Out of Boundary (the package must not depend on tools/). The two lists must be changed together. *(Corrected by 5.4, 2026-09-07: they were never the same list and had already diverged 3-vs-6, silently, for the whole life of the project - the capability check omitted the three DLLs whose absence lets the provider register and then die in native code, which is the exact failure it exists to pre-empt. "Must be changed together" was a convention with nothing enforcing it. The relation is now a test in `tests/tools/test_provision_npu.py`, which may import both sides because it is not the package.)*
- 1.5: reviewer left 5 non-blocking test-hardening suggestions (empty-string remediation invariant, driver-exactly-equal boundary, hollow-onnxruntime branch, `_payload_state(None)`, and wrapping the `XrtSmiWrapper()` construction in a guard so the never-raise promise is structural rather than dependent on xrt.py internals). Pick these up if a later task hardens `environment/`.
- 2.1: `CONDITION_PROVIDER_REGISTERED` had to move to types.py with `CapabilityReport` because that report's `__post_init__` keys on it; leaving it behind would force a right-to-left import and a cycle. The other four condition names stay in capability.py. Re-exported and guarded by an AST test against local re-definition.
- 2.1: `EmbedResult`/`EmbeddingContract` deliberately DEFERRED to task 5.3 - their fields are claims about a completed operation whose invariants only the producer can enforce, so declaring them now would be a stub. design.md's File Structure Plan is a whole-feature inventory, not a per-task mandate (it also lists ModelProfile, which is 2.2).
- 2.1: error `stage` is never absent - every class carries a `default_stage`, so an error built knowing nothing still reports one. `provider`/`model_id` are None when unknown, with blank strings normalising to None so absence has one form. No error constructor raises.
- 2.1: reviewer left 2 non-blocking test gaps in 2.1-owned code (the no-constructor-raises invariant is described in a docstring but not pinned; `DocumentText` accepting the literal title "none" is unpinned). Pick up if types/errors are hardened later.
- 2.2: `ModelProfile` lives in `profiles.py`, NOT `types.py`. design.md was self-contradictory (File Structure Plan listed it under types.py while the profiles.py line assigns "the three ModelProfile entries" to profiles.py). Resolved toward profiles.py; verified no layering inversion since nothing imports ModelProfile from types. design.md corrected to match.
- 2.2: `license_acceptance_url` exceeds design.md's original ModelProfile sketch but is REQUIRED - `LicenseAcceptanceRequired.acceptance_url` has no default, so requirement 4.5 cannot be satisfied without the profile supplying it. Invariant-bound to `license_gated` in both directions. design.md's sketch updated.
- 2.2: the benchmark (6.x) may want `all-MiniLM-L6-v2` as a known-good NPU control - the model task 1.3 proved at 5.24x CPU. It would isolate "this candidate does not offload" from "the NPU path is broken". NOT added: requirement 4.1 names exactly three candidates and a test pins the set, so adding it needs a requirements change. Decide at 6.1.
- 2.3: TASK 4.1 MUST ADD `fallback_reason` to `RunSummary`. design.md's Requirements Traceability maps 2.4/2.5 to `RunSummary.fallback_reason`, but those requirements belong to task 4.1, so 2.3 correctly omitted the field. Do not lose this: without it 2.5 (auto must report why the NPU was not used) has no carrier.
- 2.3: design.md's `EmbedResult` (flat) duplicates four fields verbatim with `RunSummary` - provider_served, execution_mode, elapsed_seconds, input_count. Task 5.3 should consider composing EmbedResult around a RunSummary so 2.6/8.6 have ONE producer, but that is a design amendment 5.3 must raise, not something 2.3 could decide.
- 2.3: a clean early exit from a run is NOT success. `_interruption()` returns None only when the run both raised nothing AND finished; exiting early without an error records "ended without an error after N of M inputs". Found by a second RED phase and independently confirmed correct by review (reverting it fails 10 tests). Requirement 8.6 reports COMPLETION, so success must never be claimed over a partial batch.
- 2.3: a progress callback that raises inside `RunTracker.__enter__` means `__exit__` never runs, so no summary is built or delivered. Harmless today (zero work done at that point) but task 5.3 should know.
- 2.3: `RunTracker` is SINGLE-USE - re-entering raises RuntimeError. Reuse previously produced a false success (a second partial run reported interruption=None over a full count). Task 5.3 must construct a tracker per embed() call, never cache one on the service; this aligns with 2.7's invariant that a backend is bound once per operation.
- 3.1: A GATED REPO'S METADATA IS PUBLIC. `model_info(gated_repo, token=False)` returns 200 with `gated='manual'` and a valid sha; the gate closes on FILE FETCH, not metadata. Confirmed independently by review. Task 3.3 must not assume acquisition fails before any bytes move.
- 3.1: a commit-pinned file already in the HF cache is served with NO request, so gate tests are cache-order-dependent. Live gate tests must use an isolated `cache_dir` or they pass cold and fail warm.
- 3.1: anonymity must be requested as `token=False`, NOT `token=None` - None lets huggingface_hub silently pick up an ambient token, which would make an anonymous-gate test pass for the wrong reason.
- 3.1: `AcquiredModel.revision` is the 40-hex commit the manifest should record (task 3.3). `HfCredential.reveal()` is the single credential unwrap point in the package; the credential renders as `<redacted>` everywhere else and mapped errors raise `from None` so no chained cause can print it.
- 3.1: acquisition deliberately does NOT use `RunTracker` - RunSummary requires provider_served and execution_mode, which acquisition genuinely lacks; forcing one would fabricate a 2.6 attribution. It uses a bare ProgressCallback instead. Confirmed correct by review.
- 3.1: two non-blocking test gaps left open - the 40-hex revision boundary is not pinned (relaxing to {39,41} survives), and `parse_dotenv`'s deliberate refusal to strip inline `#` comments is documented but not tested (adding `.split('#')[0]` survives, and would silently truncate a credential). Close both if models/ is hardened later.
- 3.1: `models/__init__.py` still exports nothing, so `acquire_model` is reachable only by full module path. Task 3.3 may want the re-export.
- 3.2: `optimum` is DROPPED and design.md corrected in three places (lines 54, 138, 297). Verified non-destructively: `optimum-onnx` downgrades transformers 5.16.1->4.57.6 and huggingface-hub 1.30.0->0.36.2 (the stack acquire.py needs), and `optimum<2` resolves to 1.27.0 which imports `is_tf_available`, absent from all of transformers 5.16.1. Export runs directly through `torch.onnx.export` with `dynamic_shapes=None`, which pins every dim from the concrete example in one step.
- 3.2: `torch`+`onnxscript` live in `[project.optional-dependencies].export` (an EXTRA, not a PEP 735 group) because groups never reach wheel metadata, so a consumer that must run preparation can opt in via `npu-rag[export]`. Default deps stay lean and `export.py` imports WITHOUT loading torch.
- 3.2: `export.py` and its tests were INHERITED from an earlier interrupted run, not written under this task's TDD cycle. The inherited code shipped the requirement 4.6 anti-substitution guard as dead code (`if False and ...`) and its live test had substituted `all-MiniLM-L6-v2` - a model with NO dense stage - so the load-bearing extraction was never proven on real weights. Both fixed. Treat any other inherited code with the same suspicion: sweep for `if False`, `and False`, `or True`, unreachable branches.
- 3.2: `model.onnx` for EmbeddingGemma is a SINGLE 1.22 GB protobuf because this task passed `external_data=False`. *(Corrected 2026-09-06: this note previously called that "against ONNX's 2 GB ceiling ... tight". Misleading. 2 GB caps a single self-contained protobuf, not a model - `save_as_external_data=True` splits tensors into sidecars and removes the cap. The 1.22 GB figure is a consequence of our flag. What it actually costs is load and compile time, not headroom: see Note 4.3 on `_load_graph` parsing all 1.22 GB to count nodes.)* Measured: acquisition 126s cold, export 63s, dense.npz 19 MB.
- 3.2: `DENSE_ORDER_KEY` in dense.npz is the PIPELINE order task 5.2 must iterate to apply the projections. A sorted order applies 768->3072->768 in the wrong sequence: right shape, right norm, WRONG MEANING. Task 5.2 must consume `order` and never re-derive it by sorting the weight names.
- 3.2: three non-blocking test gaps left open on defensive branches (reviewer's N6/N10/N12): the int64 input-dtype check (export.py:611), the real reader's `activation` fidelity (export.py:538-541 - all three candidates use Identity today), and the symbolic hidden-width check (export.py:668). No fixture reaches any of them. Pick up when task 5.2 starts consuming `activation`, or if models/ is hardened later.
- 3.3: **BINDING ON TASK 4.3** - the compiler's diagnostics do NOT survive the EP-context flow. `preliminary-vaiml-pass-summary.txt` exists during compilation and is GONE by the time the session returns, so `observed_partition_share` is null in every published manifest. design.md's "unverifiable partitioning fails under explicit npu" policy would therefore fire on EVERY run. design.md is amended: derive the offload verdict from the published `context.onnx`'s own node mix (a live MiniLM artifact reads {EPContext:1, Cast:1, Gather:1, GatherND:1}). Confirmed twice on real hardware, by implementer and reviewer independently.
- 3.3: the EP context snapshot is TWO files - `context.onnx` plus a `context.onnx_VITISAI.bin` sidecar, referenced by BARE filename. That is what makes the atomic directory rename safe; an absolute path would have silently broken every published artifact. design.md's Physical Data Model corrected from four files to five.
- 3.3: NEVER read the ONNX Runtime version from `importlib.metadata` - the stale stock dist-info residue (Note 1.2) reports 1.29.0 while the loaded runtime is 1.23.2.dev20260117. Use `onnxruntime.__version__`. An AST test pins this. A wrong version in the fingerprint would silently defeat 4.7.
- 3.3: `observed_partition_share` is recorded but deliberately EXCLUDED from the reuse comparison - there is no "current" share to compare against without performing the very compile reuse exists to avoid, so the comparison is undefined rather than merely undesirable. Every other manifest field forces a rebuild. Reviewer confirmed this is correct scoping, not a narrowing of the Observable.
- 3.3: `PUBLICATION_STAGE` is a fifth preparation stage beyond errors.py's documented four. `stage` is a free-form str and 4.6 does not enumerate stages, so a rename failure genuinely needs its own.
- 3.3: the vendor compiler writes `original-info-signature.txt` and `original-model-signature.txt` into the process CWD on every compile. Tasks 4.3/6.x should expect this litter from every benchmark cell and clean it up.
- 3.3: two non-blocking test gaps on proven-LIVE defensive guards (not dead code - reachability probed): the `PreparedArtifact` path guard (artifacts.py:770-781) and `_optional_text`'s absence-vs-null branch (artifacts.py:379-380). Also `mapping.get("files", [])` is lenient where `_optional_text` is strict.
- 3.3: measured compile cost - MiniLM at batch 1 x seq 128 takes ~160-310s cold and 0.40s warm, a ~400x ratio. EmbeddingGemma at 1.22 GB / seq 512 was deliberately not compiled. The live compile test is opt-in via NPU_RAG_LIVE_COMPILE=1.
- 4.1: **TASK 5.3 MUST THREAD `fallback_reason` INTO THE TRACKER.** 4.1 added the field to `RunSummary` and produces the string in `resolve_backend`, but `RunTracker.__exit__` builds every summary with `fallback_reason=None` - the field was added under a "change nothing else in reporting.py" authorisation, so no tracker keyword was added. Requirement 2.5 therefore has a carrier with no producer path until 5.3 either threads the reason through `RunTracker` or constructs the `RunSummary` itself. This is the same carrier-without-producer situation Note 2.3 was written to prevent, one level on. Also per Note 2.3: construct ONE tracker per `embed()` call - it is single-use.
- 4.1: design.md corrected in two places. The TransformerBackend prose said `run` returns "token embeddings plus the attention mask", contradicting its own Service Interface sketch AND its own Postconditions. The sketch wins: the service holds the mask it passed in, so returning a copy would create a second mask that could disagree after a padding change - exactly where masked mean pooling silently breaks - and the isolated adapter would have to serialize it back over the socket for nothing, widening the divergence surface 5.4 requires to be zero.
- 4.1: `resolve_backend` takes a required keyword-only `factories` beyond design.md's three-arg sketch, now recorded in design.md. The port must not construct its own adapters (import cycle inside one layer, against the ports-and-adapters seam), and it is REQUIRED rather than defaulted because a default pair is exactly the silent substitution 2.2 forbids. It is also what lets both branches of requirement 2 be exercised on a machine whose NPU works.
- 4.1: the package-wide layer guard deferred by Notes 1.1 and 1.4 is now DONE - it walks every module, resolves relative imports against each file's containing package, and has a non-vacuity check. Reviewer verified it by planting a real `models` -> `service` inversion, which was caught.
- 4.1: design.md's Requirements Traceability maps 2.7 to `service.py`, but `BoundBackend` correctly lives in `providers/base.py` per the task text and the section's own Invariants line. Touch up the traceability row when design.md is next edited; not a defect.

- 4.2: SUITE-TIME DECISION POINT FOR 4.3. Three live test files (test_export_live, test_artifacts_live, test_cpu_live) each perform real MiniLM exports; the suite is now ~2 min and providers/ alone is ~17-31 s. Each module-scoped tmp dir means the export cannot be shared. Reviewer measured a session-scoped conftest fixture would save ~20 s today but would couple models/ and providers/ test modules through a shared conftest - not worth it yet. REVISIT AT 4.3/4.4: a third and fourth real export pushes the saving to ~60-90 s, and a suite people skip protects nothing.
- 4.2: CI CONSEQUENCE, applies to every task from here. GitHub runners have NO NPU, so every `*_live.py` test will skip there. Any assertion whose only real coverage is a live test is silently uncovered in CI. Task 4.2 hit exactly this: an all-ones mask fixture made a unit-level mask assertion vacuous, catchable only by the live partial-mask test. Keep unit-level coverage genuinely independent of the live suite.
- 4.2: guard one failing (CPUExecutionProvider unregistered) raises `EnvironmentError_`, NOT `ExecutionError` - a missing built-in CPU provider is a broken ORT installation, not a graph that would not run, and miscategorising it would blunt 8.2. Reviewer ruled this correct; 8.1 is preserved because EnvironmentError_ carries default_stage="environment".
- 4.2: no CPU `BackendFactory` ships. `BackendFactory` is `(profile, capability) -> TransformerBackend` with nowhere to get an artifact root, so TASK 5.3 must wire `ensure_prepared(profile, ProviderChoice.CPU, root) -> CpuBackend(artifact, profile)`.
- 4.2: the CPU `Session`/`SessionFactory` seam deliberately CANNOT express `SessionOptions` or `provider_options` - that is what makes "reduces precision nowhere" structural rather than promised. Task 4.3 needs both (the `config_file` option is what engages BF16), so vitisai.py must define its OWN factory rather than widening this one.


- 4.3: **LOAD-TIME PROVIDER OPTIONS MUST BE `config_file` ONLY.** Passing `cache_dir`/`cache_key` when LOADING an EP-context snapshot makes the Vitis AI EP call `abort()` - the interpreter dies, uncatchable by any try/except - because the key differs from the one baked in at compile time. Cache options are correct at COMPILE time (3.3) and forbidden at LOAD time. Found only by the reviewer's live run; the implementer had skipped the live suite. A unit test must pin the load-time factory receives no cache options, since CI has no NPU.
- 4.3: **FOR 4.4** - the EP's failure mode on option mismatch is a hard `abort()`, uncatchable in-process. An isolated worker hitting it would DIE, not error. Any worker protocol must treat a vanished worker as a possible EP abort, not only a crash.
- 4.3: **TASK 5.3 MUST CLOSE THE NPU FACTORY OVER THE CALLER'S SELECTION.** `VitisAIBackend` takes `requested: ProviderChoice` (npu or auto; cpu rejected) because design.md places the fail-vs-warn policy in the adapter. But `BackendFactory` is `(profile, capability)` and `resolve_backend` calls `factories.npu` identically for npu and auto, so 5.3 must build it as e.g. `lambda p, c: VitisAIBackend(artifact, p, choice, ...)`. If it ever defaults, default to NPU (fail-closed). Cleaner alternative if base.py is reopened: apply the threshold policy in `_bind`, which already knows `requested`.
- 4.3: `partition_verified` is derived SERVICE-SIDE: `None` for CPU, else `share is not None and share >= MINIMUM_PARTITION_SHARE`. 5.3 imports the threshold from `providers.vitisai` (downward, permitted). Reviewer ruled this correct given BoundBackend's four-member delegation.
- 4.3: measured `npu_partition_share` for MiniLM is **0.988** (trunk 251 nodes, residue {Cast:1, Gather:1, GatherND:1}). The "~0.97" quoted in code was an unmeasured estimate written before any live run - corrected.
- 4.3: **FOR 6.x** - the node mix has a BLIND SPOT: it cannot see CPU fallback INSIDE the EPContext blob (the vendor fail-safe partitioning). The throughput A/B against a CPU-only session on the identical graph is the mandatory independent backstop; ratio ~1.0 means the NPU is idle regardless of the node mix.
- 4.3: confirmed on hardware - the sidecar resolves relative to the model file (a foreign CWD works), and a copied `context.onnx` WITHOUT its sidecar fails with a catchable ORT `NotImplemented`, so `ExecutionError(stage="session")` is the right path for that case.
- 4.3: EmbeddingGemma's `model.onnx` stores weights INLINE (Note 3.2), so `onnx.load(..., load_external_data=False)` does nothing for it - counting nodes parses the full 1.22 GB. Count more cheaply if backend construction cost matters.
- 4.3: two non-blocking coverage gaps left open (shipped behaviour correct in both): a snapshot with one EPContext node and ZERO residue is unpinned (`residue == 0 -> None` survives; code correctly returns 1.0), and the guards-before-verification ordering design.md line 374 mandates is unpinned (swapping `_open`/`_verify` survives; a regression would surface an unregistered provider as PartitionShareTooLow instead of EnvironmentError_, blunting 8.2). Pick up if providers/ is hardened later.
- 4.4: CLOSED AS NOT APPLICABLE, no code written. The task's own first bullet made it conditional: "If the spike concluded in-process execution works, record this task as not applicable and skip it rather than building an unused path." Task 1.3's verdict was IN_PROCESS, and that has since been confirmed twice more on hardware - task 4.3's live suite constructs a real Vitis AI session in-process and runs it at ~81 inputs/s with a live xrt-smi hardware context. There is no isolation requirement left to serve.
- 4.4: requirement 5.1 is a "Where..." conditional ("Where NPU execution is not reachable from the application's own managed environment..."), so it is satisfied vacuously - its precondition is false on this machine. Requirements 5.2-5.6 hang off the same conditional. Nothing in 2.x, 3.x, 4.1-4.3 or 5.x-8.x depends on an isolated backend existing. *(**FALSIFIED by 5.4, 2026-09-07.** `npu_unavailable_reason` did depend on it: it returned `None` - "the NPU can serve" - for every mode but `UNAVAILABLE`, citing requirement 5.1 by name. With no isolated adapter built, `auto` bound the NPU factory, paid a 160-310 s compile and only then raised, instead of falling back to the CPU with a reason. Requirements 2.4, 2.5 and 1.4 were unmet on any ISOLATED machine - reachable via the bare-`uv sync` foot-gun of Note 1.2. The lesson is about the shape of the claim, not the oversight: "nothing depends on X" was asserted by search over tasks and requirements, when the dependency was a policy branch in code that named X only through a requirement number.)*
- 4.4: if it is ever revived, two hard-won facts must carry over. (1) An EP option mismatch calls abort(), so a worker would DIE rather than error - the protocol must treat a vanished worker as a possible abort, not only a crash, and the parent cannot rely on catching anything. (2) The wire protocol was designed in design.md as length-prefixed frames with a JSON header and a raw float32 payload, deliberately avoiding pickle because the worker would run under a different interpreter; a version handshake in the first frame was made a precondition by design validation.
- 4.4: `EmbedResult.execution_mode` and `ExecutionMode.ISOLATED` remain in the type system and are correct to keep - `resolve_backend` already treats ISOLATED as NPU-available, so the vocabulary is ready if the situation ever changes. No dead code ships as a result of this closure. *(**Corrected by 5.4, 2026-09-07.** Keeping the vocabulary was right; the reason given for it recorded the defect as a feature. "`resolve_backend` already treats ISOLATED as NPU-available" was precisely the bug - the policy claimed a route that closing 4.4 had just removed. The vocabulary is now kept explicitly instead: `SERVABLE_EXECUTION_MODES` in providers/base.py is the one place naming which verdicts an adapter exists for, so reviving 4.4 means adding `ISOLATED` to that set rather than rediscovering why the policy disagrees with reality. `IsolatedWorkerError` is still dead code, which the original note also denied.)*

- 5.1: **THE PACKAGE-WIDE LAYER GUARD WAS NOT PACKAGE-WIDE.** From 4.1 until 2026-09-06 `test_every_module_in_the_package_respects_the_layer_order` did `if own not in LAYER_OF: continue` - a SILENT SKIP - so any module absent from `LAYER_ORDER` was invisible to it both as importer and as target. 4.1's non-vacuity proof could not detect this: it counted only modules the table already knew. `tokenize.py` was the first to fall through; `postprocess.py` (5.2) and `service.py` (5.3) were next. Fixed by the controller: `tokenize` placed at the `providers` rank (both sit above `models`, below `service`, neither imports the other), and an unplaced module now FAILS with a message naming it. Verified by planting a decoy module and confirming the named failure, then confirming green after removal.
- 5.1: consequence for every future module - adding a file under `src/npu_rag/embedding/` now requires a deliberate `LAYER_ORDER` placement or the suite fails. That is the intent; do not "fix" it by deleting the check.
- 5.1: requirement 3.8 is made STRUCTURAL, not tested-for: one measurement (`_measure`), one boundary (`_over_limit`, the module's only `>` comparison), and `EncodedBatch.__post_init__` recomputes the truncation set from its own counts and refuses to construct if they disagree. Agreement is unrepresentable rather than merely verified.
- 5.1: the shared-code-path design has one specific weakness worth remembering - a SHARED mutation preserves agreement, so agreement tests alone pass it (the implementer's M2b: drop special tokens from both paths). It is only caught because the counting tests compare against a reference computed independently in the test, calling the raw tokenizer rather than the module. Never let that reference reuse the module's helpers.
- 5.1: `count_tokens` accepts `str | DocumentText`, wider than design.md's `text: str` sketch. Necessary and upheld on review - a title demonstrably changes the count, and a str-only signature could not measure a titled document exactly, breaking 3.8 for precisely the titled inputs document-ingest produces.
- 5.1: NEVER assert that a document count and a query count for the same text differ. Nomic's `search_document: ` and `search_query: ` both tokenize to exactly 4 tokens, so they legitimately coincide; `!=` is also insufficient since swapped templates would pass it. Compare each count to its own independent reference and assert the rendered strings differ.
- 5.1: for 5.3 - `EncodedBatch.token_counts` carries untruncated lengths and `truncated_indices` is `EmbedResult.truncated_indices`' source; `tokenizer_id` is `ModelTokenizer.tokenizer_id` (`model_id@40-hex-commit`), do not recompute it. The batch is `(n, 512)`; slicing into `profile.batch_size` groups is 5.3's job.


- 5.3: **NOTES 4.1, 4.2 AND 4.3'S WIRING OBLIGATIONS ARE DISCHARGED *AND TESTED*.** `RunTracker` takes `fallback_reason` and emits it, so requirement 2.5's carrier finally has a producer; `default_backend_builder` wires `ensure_prepared` to both adapters and closes the NPU factory over the caller's selection; `partition_verified` is derived service-side (`None` for CPU, else `share is not None and share >= MINIMUM_PARTITION_SHARE`). Do not re-open these as carriers.
- 5.3: **THE FIRST ROUND SHIPPED THREE PUBLIC FUNCTIONS WITH ZERO COVERAGE AND WAS REJECTED FOR IT.** `load_dense_layers`, `default_backend_builder` and `build_service` could each have been deleted wholesale with the full 1107-test suite still green. Five mutants inside them survived, including a CPU factory preparing an NPU artifact and `VitisAIBackend` hardcoded to `auto` - the latter recreating the requirement 2.2 substitution Note 4.3 exists to prevent. The excuse would have been "it needs hardware". It does not: `ArtifactPreparer` is an injectable seam and the adapters are module attributes `monkeypatch` can replace. **The lesson generalises past fixtures: a function whose only claim to correctness is that it reads correctly is untested, and "the seam is hard to reach" is a hypothesis to check before it is a reason.** Tasks 6.x own more wiring of exactly this shape.
- 5.3: `elapsed_seconds` was asserted as `>= 0.0`, which a hardcoded literal `0.0` satisfies - requirement 8.6 asserted vacuously. `EmbeddingService.__init__` takes an injectable `clock` and `RunTracker` reads it exactly three times (construction, entry, exit), so a fixed-step fake makes the duration exactly one step. **Where a seam already exists to make an assertion exact, an inequality is a choice, not a constraint.**
- 5.3: an equality assertion against a `ProviderChoice` proves nothing about conversion - it is a `StrEnum`, so a raw `"cpu"` compares equal to the member while every branch downstream is written with `is`. Assert identity, or `type(x) is ProviderChoice`.
- 5.3: **A PROTOCOL PARAMETER DEFAULT IS A PERMISSION, NOT AN OBLIGATION.** A default was briefly put on `BackendBuilder.__call__`'s `requested` with a note claiming "every implementation is held to it". False, and verified false: mypy accepts an implementation whose own default is `AUTO` against a protocol whose default is `NPU` - it checks arity and types, never the value. It also widened the contract in the wrong direction, since design.md's Preconditions for this boundary say "a provider choice is always explicit, with no default". Reverted: `BackendBuilder.requested` is required, and the fail-closed `NPU` default lives on `default_backend_builder`'s concrete `build`, typed as `DefaultingBackendBuilder` and pinned by test rather than by annotation.
- 5.3: **NOTE 2.3 IS DISCHARGED WITHOUT A DESIGN AMENDMENT.** 2.3 asked 5.3 to consider composing `EmbedResult` around `RunSummary` so requirements 2.6 and 8.6 have ONE producer, and to raise it if not. Composing was rejected - design.md publishes `EmbedResult` flat and downstream consumers read it directly - but the intent is met instead: every field the two types share (`provider_served`, `execution_mode`, `input_count`, `elapsed_seconds`, `fallback_reason`) is READ OFF THE SUMMARY rather than re-derived from `bound` or `texts`. This is not cosmetic; review found `execution_mode` taken from the capability report instead of the backend that served, invisible because every fixture had the two agreeing. A test now pins the agreement, and one where the two sources genuinely differ pins requirement 5.3. One of the five, `input_count`, is an EQUIVALENT mutant: the tracker is constructed with `input_count=len(texts)`, so reading it back off the summary and recomputing `len(texts)` are equal by construction and no test can tell them apart. Recorded rather than papered over with a contrived assertion - the single-producer rule still applies to it, but its mutant is unkillable, like the filler-row `token_ids[-1:]` case.
- 5.3: two documented deviations from design.md's `EmbeddingService` sketch, both accepted at review. `count_tokens` accepts `str | DocumentText` because a document's length includes its rendered title slot, and measuring content alone would disagree with the truncation decision for exactly the titled documents (requirement 3.8's drift). The embed methods take an `on_finish` summary callback because requirement 8.4's "how many inputs completed" is otherwise unreachable: an interrupted run raises and never returns its `EmbedResult`.
- 5.3: the service pads a short final batch to `batch_size` and slices the filler off BEFORE post-processing. `masked_mean_pool`'s zero-mask refusal is the backstop, not the mechanism - a mutant that pools first fails 29 tests. Filler repeats the LAST real row so its ids are certainly in-vocabulary; that its content is unobservable is deliberate, and the mutant repeating the first row instead correctly survives.
- 5.3: **ALL THREE SHIPPING PROFILES COMPILE AT `batch_size=1`, WHERE THE PADDING PATH IS UNREACHABLE.** `tests/embedding/test_service.py` declares its own profile at batch 3 for that reason. Any future batching test that reuses a real profile tests nothing; regressing the fixture to batch 1 fails 5 tests, which is the check that it still discriminates.


- 5.4: **`SERVABLE_EXECUTION_MODES` IS NOW THE ONE PLACE THAT SAYS WHICH VERDICTS AN ADAPTER EXISTS FOR.** `providers/base.py` holds `frozenset({ExecutionMode.IN_PROCESS})`; `npu_unavailable_reason` returns `None` only for a member of it and otherwise prefixes the isolation sentence ahead of the unmet conditions. `resolve_backend` needed no branch change. **If task 4.4 is ever revived, adding `ISOLATED` to that set is the whole switch** - do not reintroduce a policy that infers servability from "not UNAVAILABLE".
- 5.4: the fix went in the POLICY, not in the factory, and the reason is structural rather than stylistic. A guard inside `default_backend_builder`'s npu factory can only *raise*: `_bind` calls the factory with no `try`/`except`, so `resolve_backend` would never obtain the reason string requirement 2.5 needs to fall back with. Option (b) could not satisfy 2.5 at all. Independently confirmed by review. `factories` is also caller-supplied, so a guard in one builder would not bind any other factory pair.
- 5.4: `default_backend_builder`'s npu closure still accepts `capability` and ignores it. Deliberate - one policy location - and safe only because `resolve_backend` is the sole caller of `factories.npu`/`factories.cpu` across `src/` and `tools/`, and `EmbeddingService._run` is the sole caller of `resolve_backend`. **If a later task adds a second call site for the factories, that gate is bypassed and this decision must be revisited.**
- 5.4: `EmbeddingService.__init__` now refuses `has_dense_stage` with no dense layers. **TASK 6.3 MUST PASS `dense=load_dense_layers(...)`** when it constructs a service for the benchmark harness; that is the guard working, not an obstacle. The hazard it closes is invisible to every mechanical check - the trunk hidden width equals `profile.dimension` (768->3072->768), so an omitted Dense stage yields right shape, right norm, wrong meaning, and even a dimension assertion cannot see it. The guard keys on the profile's DECLARATION, never on emptiness; a mutant keyed on emptiness fails 20+ tests.
- 5.4: the guard raises `ValueError`, not the `ExecutionError` taxonomy. House style, upheld on review: `EmbedResult.__post_init__`, `RunSummary.__post_init__`, `RunTracker.__init__` and `RuntimeProbe.__post_init__` all raise `ValueError` for construction invariants, while requirement 8.1's provider/model/stage obligation attaches to a failing *operation*. A constructor has neither a resolved provider nor a stage.
- 5.4: **MEMBERSHIP CANNOT SEE POSITION.** Review's mutation pass replaced `parts.insert(0, _NO_ISOLATED_BACKEND)` with `parts.append(...)` and it survived all 1144 tests, because the test asserted only that both facts were *in* the string. The ordering is load-bearing - an operator who reads the provisioning fault first will try to fix it, when what actually stops the run is that no isolated adapter was ever built - so it is now asserted with an index comparison. Generalises: for any assertion of the form `x in s`, ask whether the arrangement of `s` also carries meaning.
- 5.4: two pre-existing gaps confirmed and deliberately NOT closed here, both outside this task's boundary. (1) Requirement 2.5's "before" is fully met at `resolve_backend` but only partly at the service port: `ProgressUpdate` carries no `fallback_reason`, so an `embed_documents` caller first sees the reason in `EmbedResult`/`RunSummary` *after* the CPU run. The new ISOLATED reason inherits that timing. (2) `build_service` (service.py:641) prepares an artifact for `provider` - defaulting to NPU - *before* constructing the service, so on that entry point the "prepares nothing" property does not hold under an ISOLATED or UNAVAILABLE report. No production caller yet. Both want a task.


## Standing lesson: vacuous fixtures are this project's recurring defect

Six tasks have now shipped a test whose data or assertion made a check trivially true. 5.3 widens the class: its fixtures discriminated, but a published value was asserted only for its sign, and three public functions were shipped with no assertion at all. Review found all of them by mutation, never by reading.

**5.4 widens it again, and this is the sharpest form so far: a fixture can be non-vacuous against a system that was never assembled that way.** Task 4.1's tests for the ISOLATED policy were careful, parametrized and discriminating - against a stub backend that reports `ExecutionMode.ISOLATED`. No such adapter exists in the shipped system, because 4.4 closed without building one. Both halves were individually correct and individually tested; the join was untested and wrong, and the production code was wrong this time rather than merely unprotected. **The question to ask of a test double is not only "does this discriminate?" but "does the assembled system actually contain a collaborator shaped like this?"** The remedy is cheap and was available all along: 5.4's tests drive the real `default_backend_builder` with `ensure_prepared` replaced at its own injectable seam, plus a non-vacuity control proving the identical wiring *does* prepare under an IN_PROCESS verdict - so "nothing was prepared" cannot pass by way of a dead seam.

| Task | Fixture | What it could not see |
| --- | --- | --- |
| 2.2 | near-miss (caught pre-merge) | - |
| 3.2 | dense stage names all alphabetical | `sorted()` == pipeline order, so a reversed 768->3072->768 chain passed |
| 4.2 | attention mask all ones | a fabricated `ones_like(mask)` passed all 38 unit tests |
| 5.2 | both dense weights symmetric AND bias aligned with its weight | bias-before-projection numerically identical; weight orientation pinned by nothing |
| 5.3 | `elapsed_seconds >= 0.0` on a value the service itself publishes | a hardcoded `0.0` passed all 1107 tests (req 8.6) |
| 5.4 | ISOLATED policy proved against a stub backend reporting ISOLATED | the shipped system contains no such adapter, so `auto` compiled and then raised (reqs 2.4, 2.5) |
| 5.4 | `"no isolated backend" in reason` — membership without position | `parts.append` instead of `parts.insert(0, ...)` survived all 1144 tests |

**The 5.2 case is the instructive one**: that file's own docstring cites 3.2's lesson, and its `2_Dense`/`10_Dense` ordering fixture was deliberately built to be discriminating - and the same fixture was still degenerate along two *other* axes. Fixing one vacuity does not make a fixture non-vacuous.

**Rules for every remaining task:**
1. Non-vacuity is **per property, not per fixture**. Ask separately of each assertion: what wrong implementation would this data fail to distinguish?
2. Prefer fixture values that are **generic** for the property under test - a non-symmetric matrix where orientation matters, a bias not aligned with its weight, a partial mask, non-alphabetical names.
3. Add an explicit **non-vacuity test** asserting the fixture can tell right from wrong (task 4.2's `test_the_fixtures_can_tell_a_real_batch_from_a_fabricated_one` is the model). It fails loudly if someone later "simplifies" the fixture and silently disarms the assertion it protects.
4. Hand-computed references must be **literals**, never produced by the module under test - task 5.2's reviewer re-derived them by hand to confirm.
5. CI has no NPU, so a gap covered only by a live test is not covered at all.


## Model lineup amended 2026-09-06 - obligations for task 5.5

Requirement 4.1's third candidate changed from `bge-large-en-v1.5` to `gte-modernbert-base`. Spec documents are amended; **the code is not**. `profiles.py` still declares bge-large and a test pins the set to exactly the three it knows, so 5.5 must land all of the following together. *(Reassigned from 6.1 to 5.5 on 2026-09-07: `/kiro-validate-impl` found this work sits outside 6.1's `bench fixtures` boundary, and that requirement 4.1 was cited only by the closed task 2.2 — so nothing gated the swap.)*

1. **Widen `ModelProfile.pooling`** from `Literal["mean"]` to include `"cls"`, and add a CLS branch to `postprocess.py` - `tokens[:, 0, :]`, taking the first position. Verified from the model's own `1_Pooling/config.json`: `pooling_mode_cls_token = True`. The CLS path does NOT consult the attention mask, so it cannot reintroduce the mask-blind pooling hazard; but it must still be pinned by a test with a fixture where CLS and mean differ, or the branch is vacuous (see the standing lesson above).
2. **Replace the bge-large profile** with gte-modernbert-base: 149M params, ~600 MB exported, 768-dim, 8192 architectural context compiled at 512, Apache-2.0, **not gated**, **no Dense stage**, **symmetric** - document and query templates are both identity passthrough. That identity case is worth a test of its own: it exercises the template machinery's no-op path, which nothing currently does.
3. **Export and NPU-compile it once** to confirm the path works end to end, as 3.2/3.3 did for MiniLM. At ~600 MB it should compile faster than anything else in the set.
4. Update the profile-count test and any bge-large reference in tests.

Two constraints that shaped this and must not be relitigated silently:
- **Dense architectures only.** `nomic-embed-text-v2-moe` beats v1.5 on BEIR and MIRACL but routes 8 experts top-2 per token; conditional computation does not export cleanly to ONNX and will not compile to a static-shape NPU graph. Nomic stays at **v1.5**.
- **Model size is NOT currently constrained** (corrected 2026-09-06). An earlier note called the ONNX 2 GB protobuf limit a ceiling on candidate size. It is not: 2 GB is a Protocol Buffers limit on a SINGLE self-contained `.onnx` file, and `onnx.save_model(save_as_external_data=True)` lifts it by writing tensors to sidecars, which is how much larger models ship. EmbeddingGemma sits at 1.22 GB only because `export.py` passes `external_data=False` - a project choice, not a constraint. The machine has 24 GB of system RAM, which the NPU draws from, so there is no hardware pressure either. What size actually costs is COMPILE TIME, paid per benchmark cell. **Untested open question**: whether the Vitis AI compiler accepts an external-data model. If it does, size stops mattering; if it does not, `external_data=False` becomes a real constraint and should be recorded as one.


