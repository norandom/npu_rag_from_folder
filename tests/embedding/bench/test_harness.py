"""The model-by-provider matrix driver (task 6.3, requirements 6.1, 6.2, 6.5).

Everything here runs without an NPU, without a network, and without a compile.
The harness reaches the outside world through four injectable seams that the
project already provides - an `ArtifactPreparer`, a `ServiceFactory`, a
`PowerSource` and a `MemorySource` - so an NPU cell is expressible as data on a
machine with no NPU at all, which is exactly what task 6.3's scope asks for:
6.3 builds and proves the harness, and 7.2 pays for the real matrix run.

Two categories of test live here, and the standing lesson says to keep them
apart. Almost everything below is a **property of the harness's arithmetic and
attribution**, pinned exactly against a controlled clock and a scripted machine.
The handful that need a real thread - peak memory and energy - do not assert
that the machine is fast; they wait on the *property* (a poll has landed) with a
ceiling that fails loudly, per rule 7.

The fixtures are deliberately asymmetric per cell. A metrics fixture where every
cell produces the same numbers cannot distinguish per-cell attribution from one
measurement copied across rows, so every scripted cell here differs in its
throughput, its latencies, its resident memory and its wattage, and
`test_the_cells_in_these_fixtures_really_do_differ` fails loudly if a later edit
flattens them.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from npu_rag.embedding.bench.corpus import default_corpus_directory, load_corpus
from npu_rag.embedding.bench.harness import (
    DEFAULT_LATENCY_SAMPLES,
    LATENCY_METHODOLOGY,
    LATENCY_UNIT,
    MEMORY_METHODOLOGY,
    MEMORY_UNIT,
    NO_POWER_SOURCE_REASON,
    PREPARATION_UNIT,
    THROUGHPUT_UNIT,
    BenchmarkCell,
    BenchmarkHarness,
    CellMetrics,
    MatrixResult,
    PreparationTiming,
    ProcessMemory,
    ResidentMemorySampler,
    ResidentMemorySource,
    SampleSpec,
    default_process_memory_reader,
    default_service_factory,
    matrix_cells,
    median_seconds,
    percentile_seconds,
    sample_from_corpus,
)
from npu_rag.embedding.bench.power import (
    CPU_ENERGY_UNAVAILABLE_REASON,
    ENERGY_UNIT,
    UNSUPPORTED_PLATFORM_REASON,
    Measurement,
    cpu_energy_unavailable,
)
from npu_rag.embedding.environment.xrt import PowerReading, PowerStatus
from npu_rag.embedding.models.artifacts import (
    ArtifactIdentity,
    ArtifactManifest,
    PreparedArtifact,
)
from npu_rag.embedding.models.export import (
    ACTIVATION_PREFIX,
    BIAS_PREFIX,
    DENSE_ORDER_KEY,
    WEIGHT_PREFIX,
)
from npu_rag.embedding.postprocess import DenseLayer
from npu_rag.embedding.profiles import PROFILES, ModelProfile
from npu_rag.embedding.service import EmbedResult, EmbeddingService
from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.embedding.types import (
    CONDITION_PROVIDER_REGISTERED,
    CapabilityReport,
    Condition,
    DocumentText,
    ExecutionMode,
    ProviderChoice,
)

# --------------------------------------------------------------------------
# Profiles. Two of them, differing in dimension, so a record that carried the
# wrong model's vectors would be visible rather than merely wrong.
# --------------------------------------------------------------------------

ALPHA = ModelProfile(
    model_id="fixtures/alpha-embed",
    dimension=8,
    compiled_seq_len=32,
    architectural_context_limit=64,
    batch_size=1,
    pooling="mean",
    has_dense_stage=False,
    document_template="title: {title} | text: {content}",
    query_template="query: {content}",
    license_gated=False,
)

BETA = ModelProfile(
    model_id="fixtures/beta-embed",
    dimension=4,
    compiled_seq_len=16,
    architectural_context_limit=16,
    batch_size=1,
    pooling="cls",
    has_dense_stage=False,
    document_template="{content}",
    query_template="{content}",
    license_gated=False,
)

DENSE_PROFILE = replace(ALPHA, model_id="fixtures/dense-embed", has_dense_stage=True)


def _capability(
    *, power_reporting_supported: bool = True
) -> CapabilityReport:
    return CapabilityReport(
        conditions=(
            Condition(
                name=CONDITION_PROVIDER_REGISTERED,
                satisfied=True,
                observed=None,
                required=None,
                remediation=None,
            ),
        ),
        execution_mode=ExecutionMode.IN_PROCESS,
        driver_version="32.0.20102.3930",
        runtime_version="1.23.2",
        device_name="NPU Strix",
        power_reporting_supported=power_reporting_supported,
    )


# --------------------------------------------------------------------------
# A controllable machine: one clock, one resident-memory figure, one wattage
# --------------------------------------------------------------------------


class FakeClock:
    """A clock that moves only when something asks it to.

    Note 5.3: where a seam exists to make an assertion exact, an inequality is a
    choice rather than a constraint. Every duration below is therefore an exact
    number rather than a bound.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class Machine:
    """What the scripted sources read. Mutated by the scripted service."""

    clock: FakeClock = field(default_factory=FakeClock)
    resident_bytes: int = 100_000_000
    watts: float | None = None
    memory_reads: int = 0
    power_reads: int = 0
    #: Set when the harness under test was given a power source, so an NPU cell
    #: can wait for a poll it knows will come rather than one that never will.
    await_power: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def read_memory(self) -> int:
        with self._lock:
            self.memory_reads += 1
            return self.resident_bytes

    def read_watts(self) -> float | None:
        with self._lock:
            self.power_reads += 1
            return self.watts

    def counts(self) -> tuple[int, int]:
        with self._lock:
            return self.memory_reads, self.power_reads

    def set_resident(self, value: int) -> None:
        with self._lock:
            self.resident_bytes = value

    def set_watts(self, value: float | None) -> None:
        with self._lock:
            self.watts = value


class ScriptedMemorySource:
    def __init__(self, machine: Machine) -> None:
        self._machine = machine

    def read_resident_bytes(self) -> int | None:
        return self._machine.read_memory()


class ScriptedPowerSource:
    def __init__(self, machine: Machine) -> None:
        self._machine = machine

    def read_power(self) -> PowerReading:
        watts = self._machine.read_watts()
        if watts is None:
            return PowerReading(
                watts=None,
                status=PowerStatus.UNAVAILABLE,
                reason="the scripted platform reported N/A for this poll",
            )
        return PowerReading(watts=watts, status=PowerStatus.REPORTED, reason=None)


WAIT_CEILING_SECONDS = 20.0


def _wait_for(condition: Callable[[], bool], *, what: str) -> None:
    """Wait on the property, with a ceiling that fails loudly (rule 7).

    Picking a constant sleep would only be a bet on how fast this machine is,
    and task 6.2 measured that the machine a benchmark runs on is a busy one.
    """
    deadline = time.perf_counter() + WAIT_CEILING_SECONDS
    while not condition():
        if time.perf_counter() > deadline:
            raise AssertionError(
                f"{what} did not happen within {WAIT_CEILING_SECONDS:g} s"
            )
        time.sleep(0.001)


class ScriptedService:
    """A stand-in for `EmbeddingService`, exercised through its own interface.

    It accepts exactly what `MeasuredService` declares - a sequence of
    `DocumentText` and a `ProviderChoice` - so a harness that reached for a
    privileged entry point would not compile against it.
    """

    def __init__(
        self,
        machine: Machine,
        profile: ModelProfile,
        provider: ProviderChoice,
        *,
        batch_seconds: float,
        latency_seconds: Sequence[float],
        peak_bytes: int | None = None,
        watts: float | None = None,
        provider_served: ProviderChoice | None = None,
    ) -> None:
        self._machine = machine
        self._profile = profile
        self._provider = provider
        self._batch_seconds = batch_seconds
        self._latency = list(latency_seconds)
        self._peak_bytes = peak_bytes
        self._served = provider_served or provider
        self.calls: list[tuple[tuple[DocumentText, ...], ProviderChoice]] = []
        # Set at construction, which happens before this cell's samplers start,
        # so every poll inside this cell's window carries this cell's wattage.
        if watts is not None:
            machine.set_watts(watts)

    def embed_documents(
        self, texts: Sequence[DocumentText], provider: ProviderChoice, /
    ) -> EmbedResult:
        self.calls.append((tuple(texts), provider))
        if len(texts) == 1 and self._latency:
            seconds = self._latency.pop(0)
        else:
            seconds = self._batch_seconds
            self._occupy_machine()
        self._machine.clock.advance(seconds)
        return self._result(len(texts), seconds)

    def _occupy_machine(self) -> None:
        """Raise resident memory for the duration of the batch, and prove both
        samplers observed the cell before it ends.

        The count is captured *after* the resident figure is raised, never
        before: a poll that had already taken the lock would otherwise satisfy
        the wait while having read the baseline.
        """
        if self._peak_bytes is not None:
            baseline = self._machine.resident_bytes
            self._machine.set_resident(self._peak_bytes)
            memory_before = self._machine.counts()[0]
            _wait_for(
                lambda: self._machine.counts()[0] > memory_before,
                what="a memory poll during the elevated window",
            )
            self._machine.set_resident(baseline)
        if self._machine.await_power and self._provider is ProviderChoice.NPU:
            power_before = self._machine.counts()[1]
            _wait_for(
                lambda: self._machine.counts()[1] > power_before,
                what="a power poll during the measured window",
            )

    def _result(self, count: int, seconds: float) -> EmbedResult:
        served = self._served
        return EmbedResult(
            vectors=np.zeros((count, self._profile.dimension), dtype=np.float32),
            provider_served=served,
            execution_mode=ExecutionMode.IN_PROCESS,
            partition_verified=None if served is ProviderChoice.CPU else True,
            fallback_reason=None,
            truncated_indices=(),
            elapsed_seconds=seconds,
            input_count=count,
        )


# --------------------------------------------------------------------------
# A scripted preparer: real files on disk, controlled timings
# --------------------------------------------------------------------------


def _write_artifact(root: Path, profile: ModelProfile, provider: ProviderChoice) -> Path:
    directory = root / profile.name / provider.value
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.onnx").write_bytes(b"not really a graph")
    return directory


def _artifact(
    root: Path, profile: ModelProfile, provider: ProviderChoice, *, reused: bool,
    elapsed: float,
) -> PreparedArtifact:
    directory = _write_artifact(root, profile, provider)
    identity = ArtifactIdentity(
        model_id=profile.model_id,
        revision="0" * 40,
        provider=provider.value,
        compiled_seq_len=profile.compiled_seq_len,
        batch_size=profile.batch_size,
        onnxruntime_version="1.23.2",
        ryzen_ai_version=None,
        driver_version=None,
    )
    return PreparedArtifact(
        directory=directory,
        onnx_path=directory / "model.onnx",
        context_path=None,
        dense_path=None,
        manifest=ArtifactManifest(identity=identity, observed_partition_share=None),
        reused=reused,
        reason="scripted",
        elapsed_seconds=elapsed,
    )


class ScriptedPreparer:
    """Counts its calls, and advances the clock by a different amount cold."""

    def __init__(
        self,
        machine: Machine,
        root: Path,
        *,
        cold_seconds: float = 30.0,
        warm_seconds: float = 0.5,
        already_prepared: bool = False,
    ) -> None:
        self._machine = machine
        self._root = root
        self._cold = cold_seconds
        self._warm = warm_seconds
        self._already = already_prepared
        self.calls: list[tuple[str, ProviderChoice]] = []

    def __call__(
        self, profile: ModelProfile, provider: ProviderChoice, root: Path, /
    ) -> PreparedArtifact:
        key = (profile.model_id, provider)
        first = key not in {(m, p) for m, p in self.calls}
        self.calls.append((profile.model_id, provider))
        cold = first and not self._already
        seconds = self._cold if cold else self._warm
        self._machine.clock.advance(seconds)
        return _artifact(
            self._root, profile, provider, reused=not cold, elapsed=seconds
        )


class ScriptedFactory:
    """One scripted service per cell, keyed the way the harness caches them."""

    def __init__(self, machine: Machine, scripts: dict[tuple[str, str], dict[str, Any]]):
        self._machine = machine
        self._scripts = scripts
        self.built: list[tuple[str, ProviderChoice]] = []
        self.services: dict[tuple[str, str], ScriptedService] = {}

    def __call__(
        self,
        profile: ModelProfile,
        provider: ProviderChoice,
        artifact: PreparedArtifact,
        capability: CapabilityReport,
        /,
    ) -> ScriptedService:
        self.built.append((profile.model_id, provider))
        key = (profile.model_id, provider.value)
        service = ScriptedService(
            self._machine, profile, provider, **self._scripts[key]
        )
        self.services[key] = service
        return service


# --------------------------------------------------------------------------
# The sample
# --------------------------------------------------------------------------


def _documents(count: int) -> tuple[DocumentText, ...]:
    return tuple(
        DocumentText(content=f"body {index}", title=f"Article {index}")
        for index in range(count)
    )


def _sample(count: int = 4) -> SampleSpec:
    corpus = load_corpus(default_corpus_directory())
    return SampleSpec(
        documents=_documents(count),
        chunk_ids=tuple(f"fixtures/article-{index}#0" for index in range(count)),
        composition=corpus.composition,
        selection=f"{count} fabricated documents standing in for the fixture",
    )


def _harness(
    machine: Machine,
    preparer: ScriptedPreparer,
    factory: ScriptedFactory,
    root: Path,
    *,
    power: bool = False,
    latency_samples: int = 0,
    power_reporting_supported: bool = True,
) -> BenchmarkHarness:
    machine.await_power = power
    return BenchmarkHarness(
        _capability(power_reporting_supported=power_reporting_supported),
        artifact_root=root,
        prepare=preparer,
        services=factory,
        memory_source=ScriptedMemorySource(machine),
        power_source=ScriptedPowerSource(machine) if power else None,
        clock=machine.clock,
        latency_samples=latency_samples,
        sampling_interval_seconds=0.02,
        memory_interval_seconds=0.005,
    )


# --------------------------------------------------------------------------
# Cells are data
# --------------------------------------------------------------------------


def test_the_matrix_takes_its_cell_list_as_data() -> None:
    """Task 6.3's scope bullet: 7.2 hands the full set in unchanged, so the
    harness must not know which models or providers exist."""
    cells = matrix_cells(
        (ALPHA, BETA), (ProviderChoice.NPU, ProviderChoice.CPU)
    )

    assert [(cell.profile.model_id, cell.provider) for cell in cells] == [
        (ALPHA.model_id, ProviderChoice.NPU),
        (ALPHA.model_id, ProviderChoice.CPU),
        (BETA.model_id, ProviderChoice.NPU),
        (BETA.model_id, ProviderChoice.CPU),
    ]


def test_the_real_lineup_goes_through_the_same_door() -> None:
    """The 3x2 matrix task 7.2 will run, built from `PROFILES` by the caller
    rather than from anything the harness knows."""
    cells = matrix_cells(
        PROFILES.values(), (ProviderChoice.NPU, ProviderChoice.CPU)
    )

    assert len(cells) == len(PROFILES) * 2
    assert {cell.profile.model_id for cell in cells} == {
        profile.model_id for profile in PROFILES.values()
    }


def test_the_harness_names_no_model_and_no_provider_of_its_own() -> None:
    """A source scan, because "takes its cell list as data" is a claim about
    what the module does *not* contain and no fixture can show that."""
    from npu_rag.embedding.bench import harness as module

    source = Path(str(module.__file__)).read_text(encoding="utf-8")

    assert len(PROFILES) >= 3
    for profile in PROFILES.values():
        assert profile.model_id not in source
        assert profile.name not in source


def test_a_cell_refuses_auto_because_auto_is_a_request_not_an_outcome() -> None:
    with pytest.raises(ValueError, match="auto"):
        BenchmarkCell(profile=ALPHA, provider=ProviderChoice.AUTO)


def test_a_cell_refuses_a_pooling_rule_the_post_processor_cannot_apply() -> None:
    """Note 5.5's cheap gate at this second construction site. An unimplemented
    rule otherwise surfaces at embed time - after a ~49 minute cold compile."""
    unimplemented = replace(ALPHA, pooling=cast(Any, "max"))

    with pytest.raises(ValueError, match="max"):
        BenchmarkCell(profile=unimplemented, provider=ProviderChoice.CPU)


def test_a_cell_whose_pooling_rule_is_implemented_is_accepted() -> None:
    """The non-vacuity control for the gate above: it must not refuse
    everything."""
    assert BenchmarkCell(profile=BETA, provider=ProviderChoice.CPU).provider is (
        ProviderChoice.CPU
    )


# --------------------------------------------------------------------------
# The Observable: one record per model-and-provider, all four families
# --------------------------------------------------------------------------


SCRIPTS: dict[tuple[str, str], dict[str, Any]] = {
    (ALPHA.model_id, "npu"): {
        "batch_seconds": 2.0,
        "latency_seconds": [0.10, 0.20, 0.30, 0.40],
        "peak_bytes": 900_000_000,
        "watts": 12.5,
    },
    (ALPHA.model_id, "cpu"): {
        "batch_seconds": 8.0,
        # Deliberately right-skewed, which is the shape real latency has: the
        # median of these five is 3.0 and their arithmetic mean is 22.0, so a
        # median computed as a mean is visible end to end and not only in the
        # helper's own unit test.
        "latency_seconds": [1.0, 2.0, 3.0, 4.0, 100.0],
        "peak_bytes": 400_000_000,
    },
    (BETA.model_id, "npu"): {
        "batch_seconds": 5.0,
        "latency_seconds": [0.5, 0.6, 0.7, 0.8],
        "peak_bytes": 300_000_000,
        "watts": 3.0,
    },
    (BETA.model_id, "cpu"): {
        "batch_seconds": 16.0,
        "latency_seconds": [2.0, 2.5, 3.0, 3.5],
        "peak_bytes": 200_000_000,
    },
}


def test_the_cells_in_these_fixtures_really_do_differ() -> None:
    """Rule 3's explicit non-vacuity control. A metrics fixture whose cells all
    produce the same numbers cannot tell per-cell attribution from a single
    measurement copied across every row, so if a later edit flattens these
    scripts this fails rather than silently disarming five assertions."""
    assert len({script["batch_seconds"] for script in SCRIPTS.values()}) == 4
    assert len({script["peak_bytes"] for script in SCRIPTS.values()}) == 4
    assert len(
        {tuple(script["latency_seconds"]) for script in SCRIPTS.values()}
    ) == 4
    watts = [s["watts"] for s in SCRIPTS.values() if "watts" in s]
    assert len(set(watts)) == len(watts) == 2


def test_one_pass_produces_one_record_per_cell_carrying_all_four_families(
    tmp_path: Path,
) -> None:
    """Task 6.3's Observable, whole."""
    machine = Machine()
    preparer = ScriptedPreparer(machine, tmp_path)
    factory = ScriptedFactory(machine, SCRIPTS)
    harness = _harness(
        machine, preparer, factory, tmp_path, power=True, latency_samples=4
    )
    cells = matrix_cells((ALPHA, BETA), (ProviderChoice.NPU, ProviderChoice.CPU))

    result = harness.run(cells, _sample())

    assert isinstance(result, MatrixResult)
    assert len(result.cells) == len(cells) == 4
    assert {(m.model_id, m.provider) for m in result.cells} == {
        (cell.profile.model_id, cell.provider) for cell in cells
    }
    for metrics in result.cells:
        published = metrics.measurements()
        assert set(published) == {
            "throughput",
            "latency_median",
            "latency_p95",
            "peak_resident_memory",
            "baseline_resident_memory",
            "preparation_first_run",
            "preparation_warm",
            "energy",
        }
        assert all(isinstance(value, Measurement) for value in published.values())
        assert published["throughput"].unit == THROUGHPUT_UNIT
        assert published["latency_median"].unit == LATENCY_UNIT
        assert published["peak_resident_memory"].unit == MEMORY_UNIT
        assert published["preparation_warm"].unit == PREPARATION_UNIT
        assert published["energy"].unit == ENERGY_UNIT


def test_a_pass_refuses_to_measure_one_combination_twice(
    tmp_path: Path,
) -> None:
    """`run`'s own guard, which fires before any preparation is paid."""
    machine = Machine()
    preparer = ScriptedPreparer(machine, tmp_path)
    factory = ScriptedFactory(machine, SCRIPTS)
    harness = _harness(machine, preparer, factory, tmp_path)
    cell = BenchmarkCell(profile=ALPHA, provider=ProviderChoice.CPU)

    with pytest.raises(ValueError, match="once"):
        harness.run([cell, cell], _sample())
    assert preparer.calls == []


def test_the_result_itself_refuses_two_records_for_one_model_and_provider() -> None:
    """"One metric record per combination" made unconstructible to violate.

    Constructed directly, not through `run`. The two guards otherwise mask each
    other - deleting either alone leaves the suite green, because a pass is
    caught by `run` first - and this is the half that matters downstream: tasks
    6.4 and 6.7 build a `MatrixResult` from records they hold, bypassing `run`
    entirely, so the Observable would rest on a guard nothing exercised.
    """
    duplicate = _fabricated_metrics(ALPHA.model_id, ProviderChoice.CPU)
    other = _fabricated_metrics(BETA.model_id, ProviderChoice.CPU)
    sample = _sample()

    with pytest.raises(ValueError, match="once"):
        MatrixResult(sample=sample, cells=(duplicate, duplicate))

    # The non-vacuity control: two genuinely different combinations are
    # accepted, so the guard is not refusing every pair it is handed.
    accepted = MatrixResult(sample=sample, cells=(duplicate, other))
    assert len(accepted.cells) == 2


def _fabricated_metrics(model_id: str, provider: ProviderChoice) -> CellMetrics:
    """One plausible record, built without running anything."""
    return CellMetrics(
        model_id=model_id,
        provider=provider,
        provider_served=provider,
        execution_mode=ExecutionMode.IN_PROCESS,
        partition_verified=None if provider is ProviderChoice.CPU else True,
        fallback_reason=None,
        input_count=4,
        truncated_count=0,
        workload_seconds=2.0,
        sampled_window_seconds=None,
        throughput=Measurement(value=2.0, unit=THROUGHPUT_UNIT, unavailable_reason=None),
        latency_median=Measurement(
            value=0.5, unit=LATENCY_UNIT, unavailable_reason=None
        ),
        latency_p95=Measurement(value=0.9, unit=LATENCY_UNIT, unavailable_reason=None),
        latency_samples=5,
        peak_resident_memory=Measurement(
            value=1.0, unit=MEMORY_UNIT, unavailable_reason=None
        ),
        baseline_resident_memory=Measurement(
            value=1.0, unit=MEMORY_UNIT, unavailable_reason=None
        ),
        memory_samples=2,
        preparation=PreparationTiming(
            first_run=Measurement(
                value=30.0, unit=PREPARATION_UNIT, unavailable_reason=None
            ),
            warm=Measurement(
                value=0.5, unit=PREPARATION_UNIT, unavailable_reason=None
            ),
            first_run_reused=False,
            artifact_directory="fabricated",
        ),
        energy=cpu_energy_unavailable(),
    )


def test_a_run_needs_at_least_one_cell(tmp_path: Path) -> None:
    machine = Machine()
    harness = _harness(
        machine,
        ScriptedPreparer(machine, tmp_path),
        ScriptedFactory(machine, SCRIPTS),
        tmp_path,
    )

    with pytest.raises(ValueError, match="at least one"):
        harness.run([], _sample())


# --------------------------------------------------------------------------
# Throughput
# --------------------------------------------------------------------------


def _metrics(
    tmp_path: Path,
    profile: ModelProfile,
    provider: ProviderChoice,
    *,
    machine: Machine | None = None,
    latency_samples: int = 0,
    power: bool = False,
    power_reporting_supported: bool = True,
    inputs: int = 4,
    preparer: ScriptedPreparer | None = None,
    factory: ScriptedFactory | None = None,
) -> CellMetrics:
    machine = machine or Machine()
    preparer = preparer or ScriptedPreparer(machine, tmp_path)
    factory = factory or ScriptedFactory(machine, SCRIPTS)
    harness = _harness(
        machine,
        preparer,
        factory,
        tmp_path,
        power=power,
        latency_samples=latency_samples,
        power_reporting_supported=power_reporting_supported,
    )
    result = harness.run(
        [BenchmarkCell(profile=profile, provider=provider)], _sample(inputs)
    )
    return result.cells[0]


def test_throughput_is_inputs_per_second_and_not_seconds_per_input(
    tmp_path: Path,
) -> None:
    """4 inputs in 8 s is 0.5 inputs/s. The inverted figure is 2.0, and the two
    are distinguishable only because the fixture avoids the fixed point of
    ``x -> 1/x``."""
    metrics = _metrics(tmp_path, ALPHA, ProviderChoice.CPU, inputs=4)

    assert metrics.workload_seconds == pytest.approx(8.0)
    assert metrics.throughput.value == pytest.approx(0.5)
    assert metrics.throughput.value != pytest.approx(2.0)
    assert metrics.throughput.unit == THROUGHPUT_UNIT


def test_throughput_divides_by_the_workload_not_by_the_energy_window(
    tmp_path: Path,
) -> None:
    """Obligation from Note 6.2: the workload is timed separately from the
    sampler. The sampler keeps its own clock - a real one - while the workload
    is timed on the harness's injected clock, so the two windows here are
    genuinely different numbers and a denominator swap cannot hide behind their
    being equal."""
    metrics = _metrics(
        tmp_path, ALPHA, ProviderChoice.NPU, power=True, inputs=4
    )

    assert metrics.workload_seconds == pytest.approx(2.0)
    assert metrics.sampled_window_seconds is not None
    assert metrics.sampled_window_seconds != pytest.approx(
        metrics.workload_seconds, abs=1e-6
    )
    assert metrics.throughput.value == pytest.approx(4 / metrics.workload_seconds)


def test_a_workload_that_took_no_measurable_time_reports_no_throughput(
    tmp_path: Path,
) -> None:
    machine = Machine()
    scripts = {(ALPHA.model_id, "cpu"): {"batch_seconds": 0.0, "latency_seconds": []}}
    metrics = _metrics(
        tmp_path,
        ALPHA,
        ProviderChoice.CPU,
        machine=machine,
        factory=ScriptedFactory(machine, scripts),
    )

    assert metrics.throughput.value is None
    # The claim, not a digit: `"0" in reason` is satisfied by any omission text
    # containing a zero anywhere, including the three other reasons this field
    # never carries but a mutant could substitute.
    reason = metrics.throughput.unavailable_reason or ""
    assert "completed in 0 s of measured wall clock" in reason
    assert "undefined rather than infinite" in reason


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------


def test_the_percentile_helper_is_not_the_median() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    assert median_seconds(values) == pytest.approx(5.5)
    assert percentile_seconds(values, 0.95) == pytest.approx(10.0)
    assert median_seconds(values) != percentile_seconds(values, 0.95)


def test_the_percentile_helper_uses_nearest_rank_on_a_short_series() -> None:
    values = [4.0, 1.0, 3.0, 2.0]

    assert median_seconds(values) == pytest.approx(2.5)
    assert percentile_seconds(values, 0.95) == pytest.approx(4.0)
    assert percentile_seconds(values, 0.5) == pytest.approx(2.0)


def test_the_median_is_not_the_arithmetic_mean() -> None:
    """The third coincidence of the same family as the two already guarded here
    (throughput avoiding the fixed point of ``x -> 1/x``, and p95 against the
    median). Every symmetric series makes ``sum / count`` and the middle
    observation equal, and a mean would defeat the whole reason a median is
    reported: latency is right-skewed, so one slow call drags a mean and leaves
    a median where the typical call actually sits."""
    skewed = [1.0, 2.0, 3.0, 4.0, 100.0]
    mean = sum(skewed) / len(skewed)

    assert median_seconds(skewed) == pytest.approx(3.0)
    assert mean == pytest.approx(22.0)
    assert median_seconds(skewed) != pytest.approx(mean)


def test_the_percentile_helper_refuses_an_empty_series() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile_seconds([], 0.95)
    with pytest.raises(ValueError, match="empty"):
        median_seconds([])


def test_single_input_latency_reports_median_and_p95_separately(
    tmp_path: Path,
) -> None:
    """Five single-input calls at 1, 2, 3, 4 and 100 seconds - the right-skewed
    shape real latency has. The median is 3.0, the 95th percentile is 100.0 and
    the arithmetic mean is 22.0, so all three published-figure mistakes are
    distinguishable here: a p95 computed as the median reports 3.0, and a median
    computed as a mean reports 22.0."""
    metrics = _metrics(tmp_path, ALPHA, ProviderChoice.CPU, latency_samples=5)

    assert metrics.latency_samples == 5
    assert metrics.latency_median.value == pytest.approx(3.0)
    assert metrics.latency_p95.value == pytest.approx(100.0)
    assert metrics.latency_median.value != pytest.approx(22.0)
    assert metrics.latency_median.value != metrics.latency_p95.value


def test_latency_is_measured_one_input_at_a_time(tmp_path: Path) -> None:
    """Requirement 6.2 says *single-input* latency. A harness that timed the
    batch and divided would never call the service with one document."""
    machine = Machine()
    factory = ScriptedFactory(machine, SCRIPTS)
    _metrics(
        tmp_path,
        ALPHA,
        ProviderChoice.CPU,
        machine=machine,
        factory=factory,
        latency_samples=3,
    )
    service = factory.services[(ALPHA.model_id, "cpu")]

    sizes = [len(texts) for texts, _ in service.calls]
    assert sizes == [4, 1, 1, 1]


def test_no_latency_probes_means_a_stated_omission_not_a_zero(
    tmp_path: Path,
) -> None:
    metrics = _metrics(tmp_path, ALPHA, ProviderChoice.CPU, latency_samples=0)

    assert metrics.latency_median.value is None
    assert metrics.latency_p95.value is None
    # Pinned by CLAIM, not keyword. `"no single-input"` alone was satisfied by
    # a reason degraded to the bare string "no single-input." - review's G8.
    # What must survive is the reason 6.2 rejects the obvious substitute: a
    # batch's wall clock divided by its input count is a different measurement,
    # not a cheaper way to get this one.
    reason = metrics.latency_median.unavailable_reason or ""
    assert "no single-input embedding call was timed" in reason
    assert "would answer a different question" in reason
    assert "No figure is substituted in its place." in reason


def test_the_default_latency_sample_count_supports_a_p95_at_all() -> None:
    """A p95 over fewer than five observations is just the maximum."""
    assert DEFAULT_LATENCY_SAMPLES >= 5


# --------------------------------------------------------------------------
# Peak resident memory
# --------------------------------------------------------------------------


def test_the_memory_source_reads_the_current_working_set_not_the_process_peak() -> (
    None
):
    """The whole hazard, in one assertion. ``PeakWorkingSetSize`` is a
    high-water mark the OS never resets, so a source that returned it would
    report the largest cell's figure for every cell that follows it."""
    source = ResidentMemorySource(
        reader=lambda: ProcessMemory(
            working_set_bytes=100, peak_working_set_bytes=900
        )
    )

    assert source.read_resident_bytes() == 100


def test_the_memory_source_reports_an_absent_reading_rather_than_a_zero() -> None:
    source = ResidentMemorySource(reader=lambda: None)

    assert source.read_resident_bytes() is None


def test_the_sampler_takes_the_maximum_over_the_window_it_covers() -> None:
    readings = iter([10, 90, 30])
    last = [30]

    def source_read() -> int | None:
        try:
            last[0] = next(readings)
        except StopIteration:
            pass
        return last[0]

    sampler = ResidentMemorySampler(
        _CallableSource(source_read), interval_seconds=0.001
    )
    with sampler:
        _wait_for(
            lambda: sampler.samples_taken >= 3, what="three memory polls"
        )

    assert sampler.baseline_bytes == 10
    assert sampler.peak_bytes == 90


def test_the_memory_sampler_measures_one_cell_and_refuses_reuse() -> None:
    sampler = ResidentMemorySampler(
        _CallableSource(lambda: 1), interval_seconds=0.001
    )
    with sampler:
        pass

    with pytest.raises(RuntimeError, match="reused"):
        with sampler:
            pass


class _CallableSource:
    def __init__(self, read: Callable[[], int | None]) -> None:
        self._read = read

    def read_resident_bytes(self) -> int | None:
        return self._read()


def test_peak_memory_is_measured_per_cell_not_as_a_run_wide_high_water_mark(
    tmp_path: Path,
) -> None:
    """The first cell peaks at 900 MB and the second at 300 MB. A high-water
    mark that never resets reports 900 MB for both, and a peak taken once for
    the whole run reports one number for every row."""
    machine = Machine()
    preparer = ScriptedPreparer(machine, tmp_path)
    factory = ScriptedFactory(machine, SCRIPTS)
    harness = _harness(machine, preparer, factory, tmp_path)

    result = harness.run(
        [
            BenchmarkCell(profile=ALPHA, provider=ProviderChoice.NPU),
            BenchmarkCell(profile=BETA, provider=ProviderChoice.NPU),
        ],
        _sample(),
    )

    first = result.metrics_for(ALPHA.model_id, ProviderChoice.NPU).peak_resident_memory
    second = result.metrics_for(BETA.model_id, ProviderChoice.NPU).peak_resident_memory
    assert first.value == pytest.approx(900_000_000)
    assert second.value == pytest.approx(300_000_000)
    assert first.value is not None and second.value is not None
    assert second.value < first.value


def test_each_cell_reports_the_floor_it_started_from(tmp_path: Path) -> None:
    """A per-cell peak in one process is not a model's footprint - it includes
    everything the process was already holding. The baseline is published beside
    it so a reader can see the floor rather than being invited to subtract."""
    metrics = _metrics(tmp_path, ALPHA, ProviderChoice.NPU)

    assert metrics.baseline_resident_memory.value == pytest.approx(100_000_000)
    assert metrics.peak_resident_memory.value == pytest.approx(900_000_000)


def _clauses(text: str) -> list[str]:
    """Split a methodology string into clauses, for position-aware assertions.

    Sentences by ``". "`` rather than ``"."`` because these constants carry
    decimal figures, and then by ``";"``, because a semicolon joins two
    independent claims and a membership test over the pair cannot tell which of
    them a word belongs to. That is not a hypothetical: asserting ``"not"`` and
    ``"inference-latency comparison"`` as separate memberships let the central
    claim be *inverted* while both stayed true, because a later clause in the
    same sentence says "does not pay it per input".
    """
    parts: list[str] = []
    for sentence in text.split(". "):
        parts.extend(clause.strip() for clause in sentence.split(";"))
    return [clause for clause in parts if clause]


def test_the_memory_methodology_states_the_process_wide_caveat() -> None:
    """Named for the caveat, so it must test the caveat.

    ``"process" in text`` and ``"working set" in text`` are both satisfied by
    the constant's *first* sentence, which merely defines the measurement - so
    the entire load-bearing passage could be deleted with both still green.
    What has to survive an edit is the scope claim and requirement 6.8's two
    refusals: do not subtract, and do not estimate.
    """
    text = MEMORY_METHODOLOGY.lower()

    scope = next(c for c in _clauses(text) if "process-wide" in c)
    assert "interpreter" in scope
    assert "earlier cell has not released" in scope

    # 6.8 on subtraction: the floor is published beside the peak, never taken
    # off it, because the difference is not a figure anything measured.
    assert "not a model's memory cost and is not reported as one" in text
    # 6.8 on a spike between two reads: an under-measurement, not an estimate.
    assert "under-measures rather than estimates" in text


def test_the_latency_methodology_names_the_confound_it_measures_through() -> None:
    """The published median is dominated by per-call session construction, and
    task 6.7 has no other channel to learn it: memory has
    `MEMORY_METHODOLOGY` and energy has `EnergyMeasurement.methodology`, so the
    one figure with the largest confound must not be the one with no statement.

    Every assertion here is positional, because task 7.2 will re-measure on the
    NPU and edit this exact prose. A suite that cannot tell a corrected number
    from a deleted caveat - or from an inverted claim - fails silently at
    precisely the moment someone is editing it.
    """
    text = LATENCY_METHODOLOGY.lower()
    clauses = _clauses(text)

    # The claim that discharges the finding, asserted inside its own clause.
    # As two independent memberships this was invertible: flipping "is NOT an
    # inference-latency comparison" to "is an inference-latency comparison"
    # left both words present elsewhere in the constant.
    claim = next(c for c in clauses if "inference-latency comparison" in c)
    assert " not " in claim

    # What each probe contains, and the requirement that puts it there.
    assert "requirement 2.7 binds a backend once per operation" in text
    assert (
        "one warm artifact-manifest read and one onnx runtime session "
        "construction" in text
    )

    # The measured share. Deliberately NOT pinned to the literal 77.8%: that is
    # CPU-control-model data and task 7.2 will legitimately replace it. What
    # must survive such an edit is that a share is stated at all, and stated
    # against inference so the comparison a reader needs is present.
    share = next(c for c in clauses if "session construction (" in c)
    # A bare `"%" in share` admitted a clause that names no figure at all
    # ("an unstated %") - review's G10. A bare `\d+%` anywhere in the clause
    # does NOT close it: the clause states BOTH shares, so stripping the
    # construction figure still matches the inference one. That is this task's
    # own membership-without-position defect at one more level, and it is why
    # each figure is bound to the term it quantifies rather than counted.
    # Tolerant by construction: 7.2 re-measures on the NPU and changes the
    # numbers, not the shape.
    assert re.search(r"session construction \(\d+(\.\d+)?%\)", share)
    assert re.search(r"inference \(\d+(\.\d+)?%\)", share)

    # And that throughput carries one such construction for the whole batch.
    assert "throughput understates steady-state" in text


def test_a_memory_source_that_answers_nothing_records_an_omission(
    tmp_path: Path,
) -> None:
    machine = Machine()
    quiet = {(ALPHA.model_id, "cpu"): {"batch_seconds": 1.0, "latency_seconds": []}}
    harness = BenchmarkHarness(
        _capability(),
        artifact_root=tmp_path,
        prepare=ScriptedPreparer(machine, tmp_path),
        services=ScriptedFactory(machine, quiet),
        memory_source=_CallableSource(lambda: None),
        power_source=None,
        clock=machine.clock,
        latency_samples=0,
        memory_interval_seconds=0.005,
    )

    metrics = harness.run(
        [BenchmarkCell(profile=ALPHA, provider=ProviderChoice.CPU)], _sample()
    ).cells[0]

    assert metrics.peak_resident_memory.value is None
    # Bare truthiness was the weakest assertion in either file - a reason
    # degraded to the single character "x" passed it (review's G19). Requirement
    # 6.8's refusal to substitute is the claim worth holding.
    reason = metrics.peak_resident_memory.unavailable_reason or ""
    assert "no reading of this process's working set could be taken" in reason
    assert "No figure is substituted in its place." in reason


# --------------------------------------------------------------------------
# Preparation, cold versus warm
# --------------------------------------------------------------------------


def test_preparation_reports_the_first_run_and_the_warm_run_separately(
    tmp_path: Path,
) -> None:
    """30 s cold against 0.5 s warm - measured for real at 2953.5 s against
    0.59 s (Note 5.5). Collapsing the two reports one number twice."""
    metrics = _metrics(tmp_path, ALPHA, ProviderChoice.CPU)
    preparation = metrics.preparation

    assert isinstance(preparation, PreparationTiming)
    assert preparation.first_run.value == pytest.approx(30.0)
    assert preparation.warm.value == pytest.approx(0.5)
    assert preparation.first_run_reused is False
    assert preparation.first_run.value != preparation.warm.value


def test_a_first_run_that_found_artifacts_already_there_records_no_cold_figure(
    tmp_path: Path,
) -> None:
    """Requirement 6.8: a warm figure wearing a cold label is a substituted
    value. 7.2 runs cold; a re-run over the same artifact root does not."""
    machine = Machine()
    preparer = ScriptedPreparer(machine, tmp_path, already_prepared=True)
    metrics = _metrics(
        tmp_path, ALPHA, ProviderChoice.CPU, machine=machine, preparer=preparer
    )

    assert metrics.preparation.first_run.value is None
    # Both halves of the omission: that artifacts predated the run, and why the
    # figure that *is* available must not be published under a cold label.
    reason = metrics.preparation.first_run.unavailable_reason or ""
    assert "already existed before this run" in reason
    assert "warm measurement wearing a cold label" in reason
    assert metrics.preparation.warm.value == pytest.approx(0.5)
    assert metrics.preparation.first_run_reused is True


def test_artifacts_are_prepared_once_and_reused_across_cells_and_repetitions(
    tmp_path: Path,
) -> None:
    """Note 5.5 is binding here: a cold NPU compile is ~49 minutes, so a second
    pass over the same cell must cost nothing. Two calls per key - the first run
    and the warm comparison - and never a third."""
    machine = Machine()
    preparer = ScriptedPreparer(machine, tmp_path)
    factory = ScriptedFactory(machine, SCRIPTS)
    harness = _harness(machine, preparer, factory, tmp_path)
    cells = matrix_cells((ALPHA,), (ProviderChoice.NPU, ProviderChoice.CPU))

    harness.run(cells, _sample())
    harness.run(cells, _sample())

    assert preparer.calls.count((ALPHA.model_id, ProviderChoice.NPU)) == 2
    assert preparer.calls.count((ALPHA.model_id, ProviderChoice.CPU)) == 2
    assert len(factory.built) == 2


def test_a_repetition_reports_the_same_preparation_figures(tmp_path: Path) -> None:
    """6.4 will repeat cells on top of this. The preparation figures belong to
    the artifact, not to the repetition, so they must survive reuse rather than
    silently becoming a warm number on the second pass."""
    machine = Machine()
    preparer = ScriptedPreparer(machine, tmp_path)
    factory = ScriptedFactory(machine, SCRIPTS)
    harness = _harness(machine, preparer, factory, tmp_path)
    cell = [BenchmarkCell(profile=ALPHA, provider=ProviderChoice.CPU)]

    first = harness.run(cell, _sample()).cells[0]
    second = harness.run(cell, _sample()).cells[0]

    assert second.preparation == first.preparation
    assert second.preparation.first_run.value == pytest.approx(30.0)


# --------------------------------------------------------------------------
# Energy
# --------------------------------------------------------------------------


def test_the_cpu_row_states_why_energy_was_not_measured(tmp_path: Path) -> None:
    metrics = _metrics(tmp_path, ALPHA, ProviderChoice.CPU, power=True)

    assert metrics.energy.energy.value is None
    assert metrics.energy.energy.unavailable_reason == CPU_ENERGY_UNAVAILABLE_REASON
    assert metrics.energy.sampling_interval_seconds is None
    assert metrics.sampled_window_seconds is None


def test_each_measured_cell_gets_its_own_power_sampler(tmp_path: Path) -> None:
    """`PowerSampler` raises on reuse, exactly like `RunTracker`. A harness that
    cached one would fail on the second NPU cell rather than blending two
    windows into one integral - so two NPU cells in one pass is the test."""
    machine = Machine()
    preparer = ScriptedPreparer(machine, tmp_path)
    factory = ScriptedFactory(machine, SCRIPTS)
    harness = _harness(machine, preparer, factory, tmp_path, power=True)

    result = harness.run(
        [
            BenchmarkCell(profile=ALPHA, provider=ProviderChoice.NPU),
            BenchmarkCell(profile=BETA, provider=ProviderChoice.NPU),
        ],
        _sample(),
    )

    for metrics in result.cells:
        assert metrics.energy.samples_reported >= 1
        assert metrics.energy.energy.value is not None


def test_energy_is_attributed_to_the_cell_whose_sampler_measured_it(
    tmp_path: Path,
) -> None:
    """Cell alpha draws 12.5 W and cell beta 3.0 W, constant across each cell's
    own window, so each cell's joules are its own wattage times its own measured
    seconds. A record carrying another cell's `EnergyMeasurement` - value and
    accounting together - fails this, because the accounting travels with it."""
    machine = Machine()
    preparer = ScriptedPreparer(machine, tmp_path)
    factory = ScriptedFactory(machine, SCRIPTS)
    harness = _harness(machine, preparer, factory, tmp_path, power=True)
    sample = _sample()

    result = harness.run(
        [
            BenchmarkCell(profile=ALPHA, provider=ProviderChoice.NPU),
            BenchmarkCell(profile=BETA, provider=ProviderChoice.NPU),
        ],
        sample,
    )

    for model_id, watts in ((ALPHA.model_id, 12.5), (BETA.model_id, 3.0)):
        metrics = result.metrics_for(model_id, ProviderChoice.NPU)
        expected = watts * metrics.energy.measured_seconds / sample.input_count * 1000
        assert metrics.energy.energy.value == pytest.approx(expected)
        assert metrics.energy.measured_seconds > 0.0


def test_the_platform_support_flag_comes_from_the_capability_report(
    tmp_path: Path,
) -> None:
    """Note 6.2 is binding: hardcoding ``True`` gives a PHX machine, which
    cannot report power at all, the wrong omission text. Both arms poll a source
    that answers nothing, so only the flag decides which reason is published."""
    # No wattage in the script, so the scripted source answers N/A on every
    # poll - the state a PHX part and a momentary blank both produce.
    silent = {(ALPHA.model_id, "npu"): {"batch_seconds": 1.0, "latency_seconds": []}}

    machine = Machine()
    supported = _metrics(
        tmp_path / "a",
        ALPHA,
        ProviderChoice.NPU,
        machine=machine,
        factory=ScriptedFactory(machine, silent),
        power=True,
        power_reporting_supported=True,
    )

    machine_off = Machine()
    unsupported = _metrics(
        tmp_path / "b",
        ALPHA,
        ProviderChoice.NPU,
        machine=machine_off,
        factory=ScriptedFactory(machine_off, silent),
        power=True,
        power_reporting_supported=False,
    )

    assert supported.energy.samples_taken >= 1
    assert unsupported.energy.samples_taken >= 1
    assert supported.energy.energy.value is None
    assert unsupported.energy.energy.value is None
    assert unsupported.energy.energy.unavailable_reason == UNSUPPORTED_PLATFORM_REASON
    assert supported.energy.energy.unavailable_reason != UNSUPPORTED_PLATFORM_REASON


def test_an_npu_cell_with_no_power_source_records_the_omission(
    tmp_path: Path,
) -> None:
    metrics = _metrics(tmp_path, ALPHA, ProviderChoice.NPU, power=False)

    assert metrics.energy.energy.value is None
    assert metrics.energy.energy.unavailable_reason == NO_POWER_SOURCE_REASON


# --------------------------------------------------------------------------
# The production path
# --------------------------------------------------------------------------


def test_the_harness_drives_the_service_through_its_ordinary_interface(
    tmp_path: Path,
) -> None:
    """design.md: the harness consumes the service as an ordinary caller, with
    no privileged access, so the measured path is the production path. It hands
    over `DocumentText` values and a `ProviderChoice`, and nothing else."""
    machine = Machine()
    factory = ScriptedFactory(machine, SCRIPTS)
    sample = _sample(4)
    _metrics(
        tmp_path,
        ALPHA,
        ProviderChoice.NPU,
        machine=machine,
        factory=factory,
        latency_samples=2,
    )
    service = factory.services[(ALPHA.model_id, "npu")]

    texts, provider = service.calls[0]
    assert texts == sample.documents
    assert provider is ProviderChoice.NPU
    assert all(isinstance(text, DocumentText) for text in texts)


def test_a_cell_served_by_a_provider_it_did_not_name_is_a_failure(
    tmp_path: Path,
) -> None:
    """Attribution is the whole point of a matrix row. A cell that asked for the
    NPU and got the CPU has no business being recorded as an NPU measurement."""
    machine = Machine()
    scripts = {
        (ALPHA.model_id, "npu"): {
            "batch_seconds": 1.0,
            "latency_seconds": [],
            "provider_served": ProviderChoice.CPU,
        }
    }

    with pytest.raises(ValueError, match="served"):
        _metrics(
            tmp_path,
            ALPHA,
            ProviderChoice.NPU,
            machine=machine,
            factory=ScriptedFactory(machine, scripts),
        )


# --------------------------------------------------------------------------
# The sample (requirement 6.5) and the title obligation (Note 6.1)
# --------------------------------------------------------------------------


def test_every_document_carries_its_articles_title() -> None:
    """Note 6.1 is binding: without the title every fixture chunk embeds with
    the literal sentinel ``none`` in the slot requirement 3.4's document
    template renders."""
    corpus = load_corpus(default_corpus_directory())

    sample = sample_from_corpus(corpus)

    assert len(sample.documents) == len(corpus.chunks)
    for document, chunk in zip(sample.documents, corpus.chunks, strict=True):
        assert document.content == chunk.text
        assert document.title == chunk.article_title


def test_the_fixture_titles_would_actually_show_a_dropped_title() -> None:
    """The non-vacuity control for the assertion above. If the fixture's titles
    were all absent, or all equal to the sentinel a missing title renders as,
    then dropping the title would change nothing and the check would pass on a
    harness that never passed one."""
    from npu_rag.embedding.profiles import MISSING_TITLE_SENTINEL

    corpus = load_corpus(default_corpus_directory())
    titles = {chunk.article_title for chunk in corpus.chunks}

    assert len(titles) > 1
    assert MISSING_TITLE_SENTINEL not in titles
    rendered = ALPHA.render_document(
        DocumentText(content="body", title=corpus.chunks[0].article_title)
    )
    assert corpus.chunks[0].article_title in rendered
    assert MISSING_TITLE_SENTINEL not in rendered


def test_the_sample_states_its_size_and_its_selection() -> None:
    corpus = load_corpus(default_corpus_directory())

    whole = sample_from_corpus(corpus)
    part = sample_from_corpus(corpus, limit=5)

    assert whole.input_count == corpus.composition.chunk_count
    assert part.input_count == 5
    assert part.selection != whole.selection
    # A partial selection states both numbers - how much was used and of what -
    # since the fixture's own size alone would read as the whole having been
    # measured (requirement 6.5 asks the run to state the size it measured).
    assert "first 5" in part.selection
    assert str(len(corpus.chunks)) in part.selection
    assert part.composition is corpus.composition


def test_a_sample_of_nothing_is_refused() -> None:
    corpus = load_corpus(default_corpus_directory())

    with pytest.raises(ValueError, match="at least one"):
        sample_from_corpus(corpus, limit=0)


def test_the_sample_carries_the_chunk_id_of_every_document() -> None:
    corpus = load_corpus(default_corpus_directory())

    sample = sample_from_corpus(corpus, limit=3)

    assert sample.chunk_ids == tuple(c.chunk_id for c in corpus.chunks[:3])
    with pytest.raises(ValueError, match="one chunk id"):
        SampleSpec(
            documents=_documents(2),
            chunk_ids=("only-one",),
            composition=corpus.composition,
            selection="mismatched",
        )


# --------------------------------------------------------------------------
# The default service factory: the Dense-stage obligation (Note 5.4)
# --------------------------------------------------------------------------


def _dense_artifact(root: Path) -> PreparedArtifact:
    directory = root / "dense"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.onnx").write_bytes(b"graph")
    weight = np.eye(DENSE_PROFILE.dimension, dtype=np.float32) * 2.0
    bias = np.zeros(DENSE_PROFILE.dimension, dtype=np.float32)
    np.savez(
        directory / "dense.npz",
        **{
            DENSE_ORDER_KEY: np.array(["2_Dense"]),
            f"{WEIGHT_PREFIX}2_Dense": weight,
            f"{BIAS_PREFIX}2_Dense": bias,
            f"{ACTIVATION_PREFIX}2_Dense": np.array("Identity"),
        },
    )
    identity = ArtifactIdentity(
        model_id=DENSE_PROFILE.model_id,
        revision="0" * 40,
        provider=ProviderChoice.CPU.value,
        compiled_seq_len=DENSE_PROFILE.compiled_seq_len,
        batch_size=DENSE_PROFILE.batch_size,
        onnxruntime_version="1.23.2",
        ryzen_ai_version=None,
        driver_version=None,
    )
    return PreparedArtifact(
        directory=directory,
        onnx_path=directory / "model.onnx",
        context_path=None,
        dense_path=directory / "dense.npz",
        manifest=ArtifactManifest(identity=identity, observed_partition_share=None),
        reused=False,
        reason="built for this test",
        elapsed_seconds=1.0,
    )


class _StubTokenizer:
    """Enough of `ModelTokenizer` for construction; nothing embeds here."""


def _tokenizer_source(profile: ModelProfile, /) -> ModelTokenizer:
    return cast(ModelTokenizer, _StubTokenizer())


def test_the_default_service_factory_passes_the_declared_dense_stage(
    tmp_path: Path,
) -> None:
    """Note 5.4 is binding on this task: an omitted Dense stage yields vectors
    of the right width, the right dtype and unit norm that mean something else,
    and no shape, dtype or norm check can see it."""
    artifact = _dense_artifact(tmp_path)
    factory = default_service_factory(
        tmp_path, prepare=_unused_preparer, tokenizers=_tokenizer_source
    )

    service = factory(
        DENSE_PROFILE, ProviderChoice.CPU, artifact, _capability()
    )

    assert isinstance(service, EmbeddingService)
    layers = _dense_of(service)
    assert len(layers) == 1
    assert layers[0].weight.shape == (
        DENSE_PROFILE.dimension,
        DENSE_PROFILE.dimension,
    )


def test_omitting_the_dense_stage_would_be_refused(tmp_path: Path) -> None:
    """The non-vacuity control for the test above: the guard it relies on is
    real, so a factory that dropped ``dense=`` raises rather than shipping
    semantically wrong vectors."""
    with pytest.raises(ValueError, match="Dense"):
        EmbeddingService(
            DENSE_PROFILE,
            _capability(),
            cast(ModelTokenizer, _StubTokenizer()),
            backends=cast(Any, lambda requested: None),
        )


def _dense_of(service: EmbeddingService) -> tuple[DenseLayer, ...]:
    return service._dense  # noqa: SLF001


def _unused_preparer(
    profile: ModelProfile, provider: ProviderChoice, root: Path, /
) -> PreparedArtifact:
    raise AssertionError(
        "the service factory is handed an artifact and must not prepare again"
    )


def test_the_default_service_factory_loads_one_tokenizer_per_model(
    tmp_path: Path,
) -> None:
    """Both providers for one model share a tokenizer: acquiring it twice would
    pay a network round trip per cell for a file that cannot differ."""
    calls: list[str] = []

    def counting(profile: ModelProfile, /) -> ModelTokenizer:
        calls.append(profile.model_id)
        return cast(ModelTokenizer, _StubTokenizer())

    artifact = _dense_artifact(tmp_path)
    factory = default_service_factory(
        tmp_path, prepare=_unused_preparer, tokenizers=counting
    )

    for provider in (ProviderChoice.CPU, ProviderChoice.NPU):
        factory(DENSE_PROFILE, provider, artifact, _capability())

    assert calls == [DENSE_PROFILE.model_id]


# --------------------------------------------------------------------------
# The real reader, on whatever platform this is
# --------------------------------------------------------------------------


def test_the_default_process_memory_reader_reports_a_working_set_under_its_peak() -> (
    None
):
    """A structural relation, not a claim that the machine cooperates: the
    current working set cannot exceed the high-water mark of itself. Where the
    platform offers no reader at all this records that instead of guessing."""
    reading = default_process_memory_reader()()

    if reading is None:
        pytest.skip("this platform exposes no process memory counters")
    assert reading.working_set_bytes > 0
    assert reading.working_set_bytes <= reading.peak_working_set_bytes


def test_vectors_are_never_read_by_the_harness(tmp_path: Path) -> None:
    """6.5's fidelity and 6.6's quality own the vectors. This task measures the
    cost of producing them, so a cell records counts and timings only."""
    metrics = _metrics(tmp_path, ALPHA, ProviderChoice.CPU)

    assert metrics.input_count == 4
    assert not hasattr(metrics, "vectors")
