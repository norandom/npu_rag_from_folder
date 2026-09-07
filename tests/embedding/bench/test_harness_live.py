"""The harness against a real service, a real graph and a real session.

``test_harness.py`` proves the arithmetic and the attribution against scripted
collaborators, which is where an exact assertion is possible at all. What it
cannot establish is that the seams line up with the real components behind them:
that `default_service_factory` builds a service the harness can actually drive,
that a real preparation reports cold and warm figures the harness can tell
apart, and that this platform answers a resident-memory read.

The provider is the **CPU**, deliberately and per task 6.3's scope. A cold NPU
compile was measured at 2953.5 s for one candidate (Implementation Note 5.5) and
task 7.2 owns that budget; nothing here compiles anything, because
``ensure_prepared`` compiles only for the NPU. The NPU limb of the harness is
covered in the unit file by injecting at the same seams this file wires to real
components.

The model is ``all-MiniLM-L6-v2`` at batch 1 x sequence 128, for the reason
tasks 3.2, 3.3 and 4.2 gave: it is the model this machine is proven on and it is
an order of magnitude cheaper than any of requirement 4.1's candidates. It is
deliberately not added to ``PROFILES`` - which is the point, since the harness
takes its cell list as data and has no opinion about which models exist. Its
document template carries a **title slot**, so the title Implementation Note 6.1
requires travels all the way into a real tokenizer here rather than only into a
`DocumentText`.

Two categories, kept apart per the standing lesson's rule 6. Everything below is
a property of the harness's code driving real components. The one claim about
the machine - that it can report its own working set - is named as such and
skips where it cannot.
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from huggingface_hub import HfApi

from npu_rag.embedding.bench.corpus import default_corpus_directory, load_corpus
from npu_rag.embedding.bench.harness import (
    BenchmarkCell,
    BenchmarkHarness,
    CellMetrics,
    MatrixResult,
    ResidentMemorySource,
    SampleSpec,
    default_service_factory,
    matrix_cells,
    sample_from_corpus,
)
from npu_rag.embedding.bench.power import CPU_ENERGY_UNAVAILABLE_REASON
from npu_rag.embedding.models.acquire import (
    DEFAULT_REVISION,
    AcquiredModel,
    acquire_model,
)
from npu_rag.embedding.models.artifacts import (
    PreparedArtifact,
    Toolchain,
    ensure_prepared,
)
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.reporting import ProgressCallback
from npu_rag.embedding.types import (
    CONDITION_PROVIDER_REGISTERED,
    CapabilityReport,
    Condition,
    ExecutionMode,
    ProviderChoice,
)

CONTROL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CONTROL_PROFILE = ModelProfile(
    model_id=CONTROL_MODEL,
    dimension=384,
    compiled_seq_len=128,
    architectural_context_limit=256,
    batch_size=1,
    pooling="mean",
    has_dense_stage=False,
    # A title slot, so Implementation Note 6.1's obligation is exercised against
    # a real tokenizer rather than only against the value object.
    document_template="{title}: {content}",
    query_template="{content}",
    license_gated=False,
)

#: Only what a trunk export reads (task 4.2): the repository also ships ONNX,
#: OpenVINO and TensorFlow copies this project never uses.
PATTERNS = ("*.json", "*.txt", "model.safetensors", "1_Pooling/*")

SAMPLE_SIZE = 3
LATENCY_PROBES = 3


def _hub_reachable() -> bool:
    try:
        HfApi().model_info(CONTROL_MODEL, token=False, timeout=10)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hub_reachable(), reason="the Hugging Face Hub is not reachable"
)


def _acquirer(
    profile: ModelProfile,
    *,
    revision: str | None,
    progress: ProgressCallback | None,
) -> AcquiredModel:
    return acquire_model(
        profile,
        revision=revision or DEFAULT_REVISION,
        allow_patterns=list(PATTERNS),
        progress=progress,
    )


def _capability() -> CapabilityReport:
    """A report describing a machine with no reachable NPU.

    The cell is a CPU one, so this is the honest description, and it also means
    the run cannot accidentally reach for an NPU adapter.
    """
    return CapabilityReport(
        conditions=(
            Condition(
                name=CONDITION_PROVIDER_REGISTERED,
                satisfied=False,
                observed="not registered in this interpreter",
                required="registered",
                remediation="install the vendor runtime",
            ),
        ),
        execution_mode=ExecutionMode.UNAVAILABLE,
        driver_version=None,
        runtime_version=str(ort.__version__),
        device_name=None,
        power_reporting_supported=False,
    )


@pytest.fixture(scope="module")
def sample() -> SampleSpec:
    return sample_from_corpus(
        load_corpus(default_corpus_directory()), limit=SAMPLE_SIZE
    )


@pytest.fixture(scope="module")
def pass_result(
    tmp_path_factory: pytest.TempPathFactory, sample: SampleSpec
) -> MatrixResult:
    """One real pass: acquire, export, publish, then embed on the CPU.

    No compilation happens - ``ensure_prepared`` compiles only for the NPU,
    which is exactly why the CPU path stays available on a machine with none.
    """
    root = tmp_path_factory.mktemp("benchmark-artifacts")

    def prepare(
        profile: ModelProfile, provider: ProviderChoice, artifacts: Path, /
    ) -> PreparedArtifact:
        return ensure_prepared(
            profile,
            provider,
            artifacts,
            toolchain=Toolchain(
                onnxruntime_version=str(ort.__version__),
                ryzen_ai_version=None,
                driver_version=None,
            ),
            acquirer=_acquirer,
        )

    harness = BenchmarkHarness(
        _capability(),
        artifact_root=root,
        prepare=prepare,
        services=default_service_factory(root, prepare=prepare),
        memory_source=ResidentMemorySource(),
        latency_samples=LATENCY_PROBES,
    )
    cells = matrix_cells((CONTROL_PROFILE,), (ProviderChoice.CPU,))
    return harness.run(cells, sample)


@pytest.fixture(scope="module")
def metrics(pass_result: MatrixResult) -> CellMetrics:
    return pass_result.metrics_for(CONTROL_MODEL, ProviderChoice.CPU)


def test_one_pass_produces_one_record_for_the_cell_it_was_given(
    pass_result: MatrixResult, sample: SampleSpec
) -> None:
    """Task 6.3's Observable against real components."""
    assert len(pass_result.cells) == 1
    assert pass_result.sample is sample
    assert pass_result.cells[0].key == (CONTROL_MODEL, ProviderChoice.CPU)


def test_the_cell_was_served_by_the_provider_it_named(metrics: CellMetrics) -> None:
    assert metrics.provider_served is ProviderChoice.CPU
    assert metrics.execution_mode is ExecutionMode.IN_PROCESS
    # The question does not apply to the CPU, and 'assumed' would be a claim.
    assert metrics.partition_verified is None
    assert metrics.input_count == SAMPLE_SIZE


def test_throughput_is_inputs_over_the_workloads_own_wall_clock(
    metrics: CellMetrics,
) -> None:
    value = metrics.throughput.value

    assert value is not None
    assert value > 0.0
    assert value == pytest.approx(metrics.input_count / metrics.workload_seconds)


def test_single_input_latency_is_measured_and_ordered(metrics: CellMetrics) -> None:
    """Requirement 6.2's median and 95th percentile, from real calls. The
    ordering is a property of the two statistics, not of the machine: a
    nearest-rank p95 is an observation at or above the median's rank."""
    assert metrics.latency_samples == LATENCY_PROBES
    median = metrics.latency_median.value
    p95 = metrics.latency_p95.value

    assert median is not None and p95 is not None
    assert median > 0.0
    assert p95 >= median


def test_preparation_reports_a_cold_first_run_and_a_much_cheaper_warm_one(
    metrics: CellMetrics,
) -> None:
    """The artifact store's whole reason for existing, measured. The first run
    acquires and exports; the second reads a manifest."""
    first = metrics.preparation.first_run.value
    warm = metrics.preparation.warm.value

    assert metrics.preparation.first_run_reused is False
    assert first is not None and warm is not None
    assert warm < first
    assert Path(metrics.preparation.artifact_directory).is_dir()


def test_the_cpu_row_records_energy_as_a_stated_omission(
    metrics: CellMetrics,
) -> None:
    """xrt-smi reports NPU power only, so there is nothing to sample here and
    no sampler runs - which is why the window is absent rather than zero."""
    assert metrics.energy.energy.value is None
    assert metrics.energy.energy.unavailable_reason == CPU_ENERGY_UNAVAILABLE_REASON
    assert metrics.sampled_window_seconds is None


def test_peak_resident_memory_was_actually_sampled_on_this_machine(
    metrics: CellMetrics,
) -> None:
    """A claim about the machine, named as one: this platform must expose its
    own working set for the figure to exist at all."""
    peak = metrics.peak_resident_memory.value
    baseline = metrics.baseline_resident_memory.value

    if peak is None:
        pytest.skip(
            "this platform exposes no process memory counters: "
            f"{metrics.peak_resident_memory.unavailable_reason}"
        )
    assert baseline is not None
    assert peak >= baseline > 0.0
    # At least the synchronous read at each end of the window, so the figure is
    # never a single reading standing in for a window.
    assert metrics.memory_samples >= 2


def test_the_documents_the_run_embedded_carried_their_titles(
    sample: SampleSpec,
) -> None:
    """Implementation Note 6.1, at the end of the wire: this profile's template
    renders a title, so a sample built without one would embed every input with
    the missing-title sentinel in a slot the model reads."""
    corpus = load_corpus(default_corpus_directory())

    for document, chunk in zip(sample.documents, corpus.chunks, strict=False):
        assert document.title == chunk.article_title
        assert CONTROL_PROFILE.render_document(document).startswith(
            chunk.article_title
        )


def test_a_second_pass_reuses_the_artifact_rather_than_preparing_again(
    tmp_path_factory: pytest.TempPathFactory, sample: SampleSpec
) -> None:
    """Note 5.5 is binding on this task: a cold compile is ~49 minutes for a
    real candidate, so repetitions must not pay preparation.

    Counted against a real ``ensure_prepared``. The first pass costs three
    calls: the harness's own first run, its warm comparison, and one warm
    manifest read from inside the embedding call, because `resolve_backend`
    binds a backend per operation (requirement 2.7) and each factory prepares
    for the provider it serves. The second pass costs **one** - the per-call
    read alone. A harness that re-prepared per pass would cost three again.
    """
    root = tmp_path_factory.mktemp("reuse-artifacts")
    calls: list[tuple[str, ProviderChoice]] = []

    def prepare(
        profile: ModelProfile, provider: ProviderChoice, artifacts: Path, /
    ) -> PreparedArtifact:
        calls.append((profile.model_id, provider))
        return ensure_prepared(
            profile,
            provider,
            artifacts,
            toolchain=Toolchain(
                onnxruntime_version=str(ort.__version__),
                ryzen_ai_version=None,
                driver_version=None,
            ),
            acquirer=_acquirer,
        )

    harness = BenchmarkHarness(
        _capability(),
        artifact_root=root,
        prepare=prepare,
        services=default_service_factory(root, prepare=prepare),
        latency_samples=0,
    )
    cell = [BenchmarkCell(profile=CONTROL_PROFILE, provider=ProviderChoice.CPU)]

    first = harness.run(cell, sample)
    first_pass = len(calls)
    second = harness.run(cell, sample)
    second_pass = len(calls) - first_pass

    assert first_pass == 3
    assert second_pass == 1
    # The figures belong to the artifact, not to the pass, so they survive
    # rather than quietly becoming a warm number on the second run.
    assert second.cells[0].preparation == first.cells[0].preparation
    assert second.cells[0].preparation.first_run.value is not None
