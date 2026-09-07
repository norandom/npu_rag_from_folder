"""Drive the model-by-provider matrix and record what each cell cost.

Task 6.3, requirements 6.1 ("measures each candidate model under both the NPU
and CPU providers"), 6.2 ("throughput ..., single-input latency at the median
and 95th percentile, energy ..., peak resident memory, and preparation time on
first run versus subsequent runs") and 6.5 (the sample and its stated size and
composition travel with the run).

**This module builds and proves the harness. Task 7.2 pays for the run.**
Nothing is persistently prepared today, so a full pass means a cold NPU compile
of every candidate - measured at 2953.5 s for one of them (Implementation Note
5.5) - and that belongs in one deliberate window rather than in the middle of a
task queue. Three consequences shape the design:

- **The cell list is data.** `BenchmarkCell` names a profile and a provider, and
  `run` takes a sequence of them. Nothing here knows which models exist or which
  providers there are, so 7.2 hands in the full set unchanged and a test on a
  machine with no NPU hands in a fabricated one.
- **Artifacts and services are prepared once and reused** across every cell and
  every repetition that names the same model and provider. Warm reuse is 0.59 s
  against 2953.5 s cold, so this is the difference between a benchmark pass and
  a working day. Task 6.4 adds repetitions on top of this and inherits the
  property.
- **Every outside collaborator is an injectable seam** the project already
  provides: an `ArtifactPreparer`, a `ServiceFactory`, a `PowerSource` and a
  `MemorySource`. An NPU cell is therefore expressible without an NPU.

**The measured path is the production path.** design.md: the harness "consumes
`EmbeddingService` as an ordinary caller. No privileged access." It calls
`embed_documents` with `DocumentText` values and a `ProviderChoice`, and it
reads nothing off the service that a downstream consumer could not read. It
never inspects a vector: reduced-precision fidelity is task 6.5's measurement
and retrieval quality is task 6.6's.

Four obligations recorded by earlier tasks are discharged here
-------------------------------------------------------------

1. **The Dense stage is passed** (Note 5.4). `default_service_factory` builds
   the service with ``dense=load_dense_layers(...)``. Omitting it yields
   correctly shaped, correctly normalised, semantically wrong vectors that no
   shape, dtype or norm check can see, which is why
   `EmbeddingService.__init__` refuses the omission outright.
2. **The title is passed** (Note 6.1). `sample_from_corpus` constructs
   ``DocumentText(content=chunk.text, title=chunk.article_title)``, so the slot
   requirement 3.4's document template renders carries the article's real title
   rather than the literal sentinel a missing title becomes.
3. **Power support comes from the capability report** (Note 6.2). It is never
   hardcoded: a part that cannot report estimated power at all and a part that
   momentarily did not emit the same token, and only the capability check has
   the evidence to tell them apart.
4. **The workload is timed separately from the sampler** (Note 6.2). Throughput
   is measured on this module's own clock around the embedding call. The energy
   denominator is the sampler's own window, which *brackets* that call - it
   opens before and closes after, by well under a millisecond on real hardware -
   so the energy figure marginally overstates the time the work took rather than
   understating it. The two windows are published side by side
   (``workload_seconds`` and ``sampled_window_seconds``) so a reader never has
   to assume they are the same number.

What peak resident memory means here
------------------------------------

Resident memory is a **process-wide** quantity and there is no per-model
accounting to be had from the operating system. Two consequences had to be
decided rather than assumed, and both are visible in what this module records:

- **It is sampled, not read once.** The obvious primitive - Windows'
  ``PeakWorkingSetSize`` - is a high-water mark the OS never resets, so in a
  process that measures several cells in sequence it reports the largest earlier
  cell's figure forever after. `ResidentMemorySource` therefore reads
  ``WorkingSetSize``, the *current* working set, and
  `ResidentMemorySampler` takes the maximum over one cell's window. That number
  falls again when a later cell needs less, which a high-water mark cannot do.
- **It is a floor plus a peak, not a footprint.** The peak still includes
  everything the process was already holding: the interpreter, the runtime, the
  tokenizer, and any session an earlier cell has not released. So the reading
  taken as the cell opens is published beside the peak as
  ``baseline_resident_memory``. The difference is *not* reported as the model's
  memory cost, because it is not one - subtracting would be exactly the
  estimated value requirement 6.8 forbids.

Polling can miss a spike between two reads, which makes the figure an
under-measurement rather than an estimate - the same stance `bench/power.py`
takes for a missed poll. `MEMORY_METHODOLOGY` states it for the document task
6.7 renders.

This module sits in the ``bench`` layer, the top of design.md's dependency
direction. It reads everything to its left and nothing may import it.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self

from npu_rag.embedding.bench.corpus import BenchmarkCorpus, SampleComposition
from npu_rag.embedding.bench.power import (
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    ENERGY_UNIT,
    EnergyMeasurement,
    Measurement,
    PowerSampler,
    PowerSource,
    cpu_energy_unavailable,
    energy_per_thousand_inputs,
)
from npu_rag.embedding.models.artifacts import PreparedArtifact, ensure_prepared
from npu_rag.embedding.postprocess import POOLING_RULES
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.service import (
    ArtifactPreparer,
    EmbedResult,
    EmbeddingService,
    default_backend_builder,
    load_dense_layers,
)
from npu_rag.embedding.tokenize import ModelTokenizer, load_tokenizer
from npu_rag.embedding.types import (
    CapabilityReport,
    DocumentText,
    ExecutionMode,
    ProviderChoice,
)

__all__ = [
    "DEFAULT_LATENCY_SAMPLES",
    "DEFAULT_MEMORY_INTERVAL_SECONDS",
    "LATENCY_METHODOLOGY",
    "LATENCY_UNIT",
    "MEMORY_METHODOLOGY",
    "MEMORY_UNIT",
    "NO_POWER_SOURCE_REASON",
    "PREPARATION_UNIT",
    "THROUGHPUT_UNIT",
    "BenchmarkCell",
    "BenchmarkHarness",
    "CellMetrics",
    "MatrixResult",
    "MeasuredService",
    "MemorySource",
    "PreparationTiming",
    "ProcessMemory",
    "ProcessMemoryReader",
    "ResidentMemorySampler",
    "ResidentMemorySource",
    "SampleSpec",
    "ServiceFactory",
    "TokenizerSource",
    "default_process_memory_reader",
    "default_service_factory",
    "matrix_cells",
    "median_seconds",
    "percentile_seconds",
    "sample_from_corpus",
]

#: Requirement 6.2's four units. They travel on every `Measurement` so a reader
#: of the rendered document never has to infer one.
THROUGHPUT_UNIT: Final = "inputs/s"
LATENCY_UNIT: Final = "s"
MEMORY_UNIT: Final = "bytes"
PREPARATION_UNIT: Final = "s"

#: How many single-input calls a cell times by default. A 95th percentile over
#: fewer than five observations is simply the maximum, so this is a floor rather
#: than a preference; the count travels with every record because a caller
#: paying an NPU session construction per probe may well choose a smaller one.
DEFAULT_LATENCY_SAMPLES: Final = 10

#: Reading the current working set costs a system call, not a subprocess, so it
#: can be sampled an order of magnitude more finely than estimated power without
#: perturbing the workload it is measuring.
DEFAULT_MEMORY_INTERVAL_SECONDS: Final = 0.05

MEMORY_METHODOLOGY: Final = (
    "Peak resident memory is the largest process working set observed while the "
    "cell ran, sampled on a separate thread rather than read once at the end. "
    "The operating system's own peak counter is a high-water mark it never "
    "resets, so in a process that measures several cells in sequence it would "
    "report the largest earlier cell's figure for every later one. The number "
    "is process-wide: it includes the interpreter, the runtime and anything an "
    "earlier cell has not released, so the working set observed as the cell "
    "opened is reported beside it as a floor. The difference between the two is "
    "not a model's memory cost and is not reported as one. Sampling can miss a "
    "spike between two reads, so the figure under-measures rather than "
    "estimates."
)

LATENCY_METHODOLOGY: Final = (
    "Single-input latency is the end-to-end wall clock of one "
    "embed_documents call carrying one document - the whole service call a "
    "consumer makes, measured with no privileged access to its internals. "
    "Requirement 2.7 binds a backend once per operation, so every probe "
    "includes one warm artifact-manifest read and one ONNX Runtime session "
    "construction alongside the forward pass. That per-call construction "
    "dominates the figure. Measured on this machine's CPU control model over "
    "10 probes with the machine otherwise idle: a published median of 0.2426 s "
    "was 0.1889 s of session construction (77.8%) against 0.0299 s of "
    "inference (12.3%), with the warm manifest read at 0.000292 s and "
    "tokenization at 0.000343 s - an overhead-to-inference ratio of 6.3x. On "
    "the NPU the construction share is expected to be larger still, because "
    "opening a compiled context snapshot scales with artifact size rather than "
    "with compute. This figure is therefore a per-call service cost and is NOT "
    "an inference-latency comparison between models; a consumer that binds "
    "once and embeds many times does not pay it per input. The same "
    "construction is paid once inside the batch that throughput is measured "
    "over, so throughput understates steady-state embedding rate, and it "
    "understates it most at small sample sizes where one construction is "
    "amortised over few inputs."
)

NO_POWER_SOURCE_REASON: Final = (
    "Energy was not measured: this benchmark run was given no power source, so "
    "nothing polled the NPU while the cell ran. This is a configuration of the "
    "run rather than a property of the machine - supply a power source (the "
    "vendor management-utility wrapper satisfies one) to measure it. No figure "
    "is substituted in its place."
)

_NO_LATENCY_REASON: Final = (
    "Latency was not measured: no single-input embedding call was timed for "
    "this cell, so there is no series to take a median or a 95th percentile "
    "over. Requirement 6.2 asks for single-input latency specifically, and "
    "dividing a batch's wall clock by its input count would answer a different "
    "question. No figure is substituted in its place."
)

_NO_MEMORY_REASON: Final = (
    "Peak resident memory was not measured: no reading of this process's "
    "working set could be taken on this platform while the cell ran. No figure "
    "is substituted in its place."
)

_ZERO_WORKLOAD_REASON: Final = (
    "Throughput was not measured: the workload completed in 0 s of measured "
    "wall clock, so inputs per second is undefined rather than infinite. Time a "
    "larger sample, or one on a clock with finer resolution."
)

_ALREADY_PREPARED_REASON: Final = (
    "First-run preparation time was not measured: valid artifacts for this "
    "model and provider already existed before this run, so the first "
    "preparation reused them. Reporting that figure as a first-run cost would "
    "be a warm measurement wearing a cold label. Remove the artifact directory "
    "and run again to measure a cold preparation."
)

_NO_REUSE_REASON: Final = (
    "Warm preparation time was not measured: the second preparation rebuilt "
    "rather than reused, so it is not a subsequent-run figure. Requirement 4.4 "
    "expects valid artifacts to be reused; a rebuild here means the manifest "
    "did not match what was just written."
)


# --------------------------------------------------------------------------
# The sample (requirement 6.5)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSpec:
    """What a run measured, and the statement requirement 6.5 asks it to make.

    ``composition`` is the committed fixture's own record - the caps it was
    drawn under, its publications, its article counts. ``selection`` says which
    part of that fixture *this* run used, because a run over a subset that
    published the whole fixture's counts would be stating a size it did not
    measure.
    """

    documents: tuple[DocumentText, ...]
    chunk_ids: tuple[str, ...]
    composition: SampleComposition
    selection: str

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError(
                "a benchmark sample needs at least one document: an empty "
                "sample has no throughput and no latency to measure"
            )
        if len(self.chunk_ids) != len(self.documents):
            raise ValueError(
                "every document needs exactly one chunk id so a measurement can "
                f"be traced back to the fixture: {len(self.documents)} "
                f"documents against {len(self.chunk_ids)} ids"
            )
        if not self.selection.strip():
            raise ValueError(
                "a sample states which part of the fixture it used; a blank "
                "selection states nothing (requirement 6.5)"
            )

    @property
    def input_count(self) -> int:
        return len(self.documents)


def sample_from_corpus(
    corpus: BenchmarkCorpus, *, limit: int | None = None
) -> SampleSpec:
    """Turn the committed fixture into the documents a cell embeds.

    **The title is passed** (Implementation Note 6.1). `CorpusChunk` carries
    ``article_title`` because requirement 3.4's document convention has a title
    slot and `DocumentText` carries one; a sample built from ``chunk.text``
    alone would embed every input with the literal sentinel a missing title
    renders as, silently, in a slot the model was trained to read.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"a sample limit cannot be negative, got {limit!r}")
    total = len(corpus.chunks)
    chunks = corpus.chunks if limit is None else corpus.chunks[:limit]
    if limit is None or limit >= total:
        selection = f"all {total} chunks of the committed fixture, in fixture order"
    else:
        selection = (
            f"the first {limit} of the fixture's {total} chunks, in fixture order"
        )
    return SampleSpec(
        documents=tuple(
            DocumentText(content=chunk.text, title=chunk.article_title)
            for chunk in chunks
        ),
        chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        composition=corpus.composition,
        selection=selection,
    )


# --------------------------------------------------------------------------
# Cells, and the matrix as data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCell:
    """One model measured under one provider.

    Two refusals, both cheap and both here because this is the earliest place
    the mistake is visible:

    - ``auto`` is a *request*, not an outcome. A matrix row attributes a
      measurement to a provider, and a row that might have been served by
      either attributes nothing (requirement 2.6, and `EmbedResult` refuses the
      same value for the same reason).
    - a pooling rule the post-processor does not implement fails at embed time,
      which on a cold cell is on the far side of a compile measured at 2953.5 s
      (Implementation Note 5.5). The gate belongs where a cell is named.
    """

    profile: ModelProfile
    provider: ProviderChoice

    def __post_init__(self) -> None:
        if self.provider is ProviderChoice.AUTO:
            raise ValueError(
                "a benchmark cell names the provider that will serve it; 'auto' "
                "is a request, not an outcome, and a row served by either "
                "provider attributes its measurement to neither"
            )
        if self.profile.pooling not in POOLING_RULES:
            raise ValueError(
                f"{self.profile.model_id} declares the pooling rule "
                f"{self.profile.pooling!r}, which the post-processor does not "
                f"implement (known: {sorted(POOLING_RULES)}). Refused here "
                "rather than at embed time, which is on the far side of this "
                "cell's model preparation"
            )

    @property
    def key(self) -> tuple[str, ProviderChoice]:
        return (self.profile.model_id, self.provider)


def matrix_cells(
    profiles: Iterable[ModelProfile], providers: Iterable[ProviderChoice]
) -> tuple[BenchmarkCell, ...]:
    """The full product, model-major.

    This is the whole of "the cell list is data": task 7.2 calls it with the
    real profile set and both providers and hands the result straight to `run`,
    while a test calls it with two fabricated profiles. Neither the harness nor
    this function knows which models exist.
    """
    ordered = tuple(providers)
    return tuple(
        BenchmarkCell(profile=profile, provider=provider)
        for profile in profiles
        for provider in ordered
    )


# --------------------------------------------------------------------------
# Process memory
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessMemory:
    """This process's current working set, and the high-water mark of it.

    Both are carried so the difference between them is a fact this module can
    state rather than a distinction it merely documents. Only
    ``working_set_bytes`` is ever reported as a per-cell figure.
    """

    working_set_bytes: int
    peak_working_set_bytes: int


class ProcessMemoryReader(Protocol):
    def __call__(self) -> ProcessMemory | None: ...


class MemorySource(Protocol):
    """One resident-memory reading, or ``None`` where none could be taken.

    ``None`` is an absent sample rather than zero, for the reason
    `bench/power.py` refuses to fold an absent power reading to 0.0 W: a zero
    would integrate into a plausible-looking figure that nothing could detect.
    """

    def read_resident_bytes(self) -> int | None: ...


def _read_windows_process_memory() -> ProcessMemory | None:
    """``GetProcessMemoryInfo`` for the current process.

    ``WorkingSetSize`` is what this module reports. ``PeakWorkingSetSize`` sits
    beside it in the same structure and is *not* reported per cell: the OS never
    resets it, so it would pin every later cell to the largest earlier one.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declared rather than inferred: the process pseudo-handle is a 64-bit
        # value, and ctypes' default int return would truncate it, which the
        # call then rejects.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        ok = kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            ctypes.sizeof(_Counters),
        )
    except (AttributeError, OSError):  # pragma: no cover - platform defensive
        return None
    if not ok:  # pragma: no cover - platform defensive
        return None
    return ProcessMemory(
        working_set_bytes=int(counters.WorkingSetSize),
        peak_working_set_bytes=int(counters.PeakWorkingSetSize),
    )


def _read_proc_status_process_memory() -> ProcessMemory | None:
    """``VmRSS`` and ``VmHWM`` from ``/proc/self/status``, where there is one."""
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        label, _, rest = line.partition(":")
        if label in {"VmRSS", "VmHWM"}:
            parts = rest.split()
            if len(parts) >= 1 and parts[0].isdigit():
                values[label] = int(parts[0]) * 1024
    current = values.get("VmRSS")
    if current is None:
        return None
    return ProcessMemory(
        working_set_bytes=current,
        peak_working_set_bytes=values.get("VmHWM", current),
    )


def default_process_memory_reader() -> ProcessMemoryReader:
    """Whichever reader this platform actually answers."""

    def read() -> ProcessMemory | None:
        windows = _read_windows_process_memory()
        if windows is not None:
            return windows
        return _read_proc_status_process_memory()

    return read


class ResidentMemorySource:
    """The current working set, never the process's high-water mark."""

    def __init__(self, *, reader: ProcessMemoryReader | None = None) -> None:
        self._reader = reader if reader is not None else default_process_memory_reader()

    def read_resident_bytes(self) -> int | None:
        reading = self._reader()
        return None if reading is None else reading.working_set_bytes


class ResidentMemorySampler:
    """Poll resident memory on its own thread for the duration of one cell.

    Single-use, like `PowerSampler` and `RunTracker`: a reused sampler would
    blend two cells' windows into one maximum, which is the run-wide high-water
    mark this class exists to avoid.

    One reading is taken synchronously as the window opens and one as it closes,
    so a cell always has at least two samples however the thread is scheduled -
    the baseline is a fact rather than a race.
    """

    def __init__(
        self,
        source: MemorySource,
        *,
        interval_seconds: float = DEFAULT_MEMORY_INTERVAL_SECONDS,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise ValueError(
                f"interval_seconds must be a positive finite number, got "
                f"{interval_seconds!r}"
            )
        self._source = source
        self._interval = interval_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._readings: list[int] = []
        self._baseline: int | None = None
        self._thread: threading.Thread | None = None
        self._entered = False

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError(
                "a ResidentMemorySampler measures one cell and cannot be "
                "reused; a reused one reports a run-wide high-water mark"
            )
        self._entered = True
        self._baseline = self._read()
        thread = threading.Thread(
            target=self._loop, name="npu-rag-memory-sampler", daemon=True
        )
        self._thread = thread
        thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type
        del exc
        del traceback
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=30.0)
        self._read()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._read()
            if self._stop.wait(self._interval):
                return

    def _read(self) -> int | None:
        try:
            value = self._source.read_resident_bytes()
        except Exception:  # noqa: BLE001 - a dead thread would lose the window
            return None
        if value is None:
            return None
        with self._lock:
            self._readings.append(value)
        return value

    @property
    def samples_taken(self) -> int:
        with self._lock:
            return len(self._readings)

    @property
    def baseline_bytes(self) -> int | None:
        """The working set as the window opened - the floor, not a footprint."""
        return self._baseline

    @property
    def peak_bytes(self) -> int | None:
        with self._lock:
            return max(self._readings) if self._readings else None


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def median_seconds(values: Sequence[float]) -> float:
    """The middle observation, averaging the two middles for an even count."""
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("a median needs at least one observation, got an empty series")
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def percentile_seconds(values: Sequence[float], fraction: float) -> float:
    """The nearest-rank percentile: the smallest observation at or above rank.

    Nearest rank rather than an interpolating definition, because interpolation
    invents a value between two observations - defensible for a large sample,
    and exactly the substituted figure requirement 6.8 forbids for the handful
    of probes a benchmark cell can afford.
    """
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError(
            "a percentile needs at least one observation, got an empty series"
        )
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction!r}")
    rank = math.ceil(fraction * count)
    return ordered[max(1, rank) - 1]


# --------------------------------------------------------------------------
# What a cell records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparationTiming:
    """Requirement 6.2's "preparation time on first run versus subsequent runs".

    Two measurements, never one figure reported twice: the ratio between them is
    the whole reason the artifact store exists, measured at 2953.5 s against
    0.59 s for one candidate (Implementation Note 5.5).

    ``first_run`` is an omission - not a number - when valid artifacts already
    existed before the run, because the figure available in that case is a warm
    one and labelling it cold would be a substituted value (requirement 6.8).
    """

    first_run: Measurement
    warm: Measurement
    first_run_reused: bool
    artifact_directory: str


@dataclass(frozen=True)
class CellMetrics:
    """One model under one provider: every figure requirement 6.2 asks for."""

    model_id: str
    provider: ProviderChoice
    provider_served: ProviderChoice
    execution_mode: ExecutionMode
    partition_verified: bool | None
    fallback_reason: str | None
    input_count: int
    truncated_count: int
    workload_seconds: float
    #: The sampler's own window, which brackets the measured call. ``None``
    #: where nothing was sampled - the CPU row, or a run with no power source.
    sampled_window_seconds: float | None
    throughput: Measurement
    latency_median: Measurement
    latency_p95: Measurement
    latency_samples: int
    peak_resident_memory: Measurement
    baseline_resident_memory: Measurement
    memory_samples: int
    preparation: PreparationTiming
    energy: EnergyMeasurement

    def __post_init__(self) -> None:
        if self.provider is ProviderChoice.AUTO:
            raise ValueError(
                "a metric record names the provider the cell asked for; 'auto' "
                "is a request, not an outcome"
            )
        if self.provider_served is not self.provider:
            raise ValueError(
                f"this cell asked for {self.provider.value!r} and was served by "
                f"{self.provider_served.value!r}; a row attributed to a provider "
                "that did not serve it is a mis-measurement, not a fallback "
                "worth recording"
            )
        if self.input_count <= 0:
            raise ValueError(
                f"a measured cell embedded at least one input, got "
                f"{self.input_count!r}"
            )
        if not math.isfinite(self.workload_seconds) or self.workload_seconds < 0.0:
            raise ValueError(
                f"workload_seconds must be finite and not negative, got "
                f"{self.workload_seconds!r}"
            )
        if self.latency_samples < 0:
            raise ValueError(
                f"latency_samples must not be negative, got {self.latency_samples!r}"
            )

    @property
    def key(self) -> tuple[str, ProviderChoice]:
        return (self.model_id, self.provider)

    def measurements(self) -> Mapping[str, Measurement]:
        """Every published figure, for task 6.7's renderer.

        The four families requirement 6.2 names - throughput, latency, memory
        and preparation - plus 6.2's energy, each as a `Measurement` that is
        either a value or a stated reason there is none.
        """
        return {
            "throughput": self.throughput,
            "latency_median": self.latency_median,
            "latency_p95": self.latency_p95,
            "peak_resident_memory": self.peak_resident_memory,
            "baseline_resident_memory": self.baseline_resident_memory,
            "preparation_first_run": self.preparation.first_run,
            "preparation_warm": self.preparation.warm,
            "energy": self.energy.energy,
        }


@dataclass(frozen=True)
class MatrixResult:
    """One pass: exactly one record per model-and-provider combination."""

    sample: SampleSpec
    cells: tuple[CellMetrics, ...]

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("a matrix result carries at least one measured cell")
        seen: set[tuple[str, ProviderChoice]] = set()
        for metrics in self.cells:
            if metrics.key in seen:
                raise ValueError(
                    f"{metrics.model_id} under {metrics.provider.value} appears "
                    "twice; a model and provider combination is measured once "
                    "per pass"
                )
            seen.add(metrics.key)

    def metrics_for(self, model_id: str, provider: ProviderChoice) -> CellMetrics:
        for metrics in self.cells:
            if metrics.key == (model_id, provider):
                return metrics
        raise KeyError((model_id, provider))


# --------------------------------------------------------------------------
# The seams
# --------------------------------------------------------------------------


class MeasuredService(Protocol):
    """What the harness needs of `EmbeddingService`, and nothing more.

    Exactly the public document-side entry point a downstream consumer calls.
    Narrowing the harness to this is what makes design.md's "no privileged
    access, so the measured path is the production path" structural: there is no
    wider handle to reach for.
    """

    def embed_documents(
        self, texts: Sequence[DocumentText], provider: ProviderChoice, /
    ) -> EmbedResult: ...


class ServiceFactory(Protocol):
    """Builds the service for one cell, from an artifact already prepared."""

    def __call__(
        self,
        profile: ModelProfile,
        provider: ProviderChoice,
        artifact: PreparedArtifact,
        capability: CapabilityReport,
        /,
    ) -> MeasuredService: ...


class TokenizerSource(Protocol):
    def __call__(self, profile: ModelProfile, /) -> ModelTokenizer: ...


def _load_tokenizer(profile: ModelProfile, /) -> ModelTokenizer:
    return load_tokenizer(profile)


def default_service_factory(
    root: Path,
    *,
    prepare: ArtifactPreparer = ensure_prepared,
    tokenizers: TokenizerSource = _load_tokenizer,
) -> ServiceFactory:
    """Assemble a real `EmbeddingService` for a cell.

    Deliberately not `build_service`: that entry point prepares an artifact of
    its own, and the harness has already prepared one and timed the preparation.
    What is copied from it, and must never be dropped, is
    ``dense=load_dense_layers(...)`` - Implementation Note 5.4 records this task
    as the second construction site that guard exists for.

    One tokenizer per model, shared by that model's providers: it is acquired
    over the network and cannot differ between them.
    """
    cache: dict[str, ModelTokenizer] = {}

    def build(
        profile: ModelProfile,
        provider: ProviderChoice,
        artifact: PreparedArtifact,
        capability: CapabilityReport,
        /,
    ) -> MeasuredService:
        tokenizer = cache.get(profile.model_id)
        if tokenizer is None:
            tokenizer = tokenizers(profile)
            cache[profile.model_id] = tokenizer
        return EmbeddingService(
            profile,
            capability,
            tokenizer,
            backends=default_backend_builder(root, prepare=prepare),
            dense=load_dense_layers(artifact, profile, provider),
        )

    return build


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


def _unsampled_energy(reason: str) -> EnergyMeasurement:
    return EnergyMeasurement(
        energy=Measurement(value=None, unit=ENERGY_UNIT, unavailable_reason=reason),
        sampling_interval_seconds=None,
        samples_taken=0,
        samples_reported=0,
        samples_missed=0,
        measured_seconds=0.0,
        unmeasured_seconds=0.0,
    )


class BenchmarkHarness:
    """Measure a list of cells, reusing everything expensive across them.

    The harness owns two caches keyed by model and provider: the prepared
    artifact and the service built over it. A second cell, or a second
    repetition, naming the same pair pays nothing - which is the difference
    between a benchmark pass and a multi-hour one, since a cold NPU compile was
    measured at 2953.5 s against 0.59 s warm.

    A failing cell aborts the run. Per-cell failure tolerance, repetitions and
    run provenance are task 6.4's, and building them here would take that task's
    work rather than leaving it its seam.
    """

    def __init__(
        self,
        capability: CapabilityReport,
        *,
        artifact_root: Path,
        prepare: ArtifactPreparer = ensure_prepared,
        services: ServiceFactory | None = None,
        power_source: PowerSource | None = None,
        memory_source: MemorySource | None = None,
        clock: Callable[[], float] = time.perf_counter,
        latency_samples: int = DEFAULT_LATENCY_SAMPLES,
        sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
        memory_interval_seconds: float = DEFAULT_MEMORY_INTERVAL_SECONDS,
    ) -> None:
        if latency_samples < 0:
            raise ValueError(
                f"latency_samples must not be negative, got {latency_samples!r}"
            )
        self._capability = capability
        self._root = artifact_root
        self._prepare = prepare
        self._services: ServiceFactory = (
            services
            if services is not None
            else default_service_factory(artifact_root, prepare=prepare)
        )
        self._power_source = power_source
        self._memory_source: MemorySource = (
            memory_source if memory_source is not None else ResidentMemorySource()
        )
        self._clock = clock
        self._latency_samples = latency_samples
        self._sampling_interval = sampling_interval_seconds
        self._memory_interval = memory_interval_seconds
        self._artifacts: dict[
            tuple[str, ProviderChoice], tuple[PreparedArtifact, PreparationTiming]
        ] = {}
        self._built: dict[tuple[str, ProviderChoice], MeasuredService] = {}

    # -- the pass ----------------------------------------------------------

    def run(
        self, cells: Iterable[BenchmarkCell], sample: SampleSpec
    ) -> MatrixResult:
        """One pass: one metric record per cell, in the order given."""
        ordered = tuple(cells)
        if not ordered:
            raise ValueError(
                "a benchmark pass needs at least one cell: the matrix is data, "
                "and an empty one measures nothing"
            )
        seen: set[tuple[str, ProviderChoice]] = set()
        for cell in ordered:
            if cell.key in seen:
                raise ValueError(
                    f"{cell.profile.model_id} under {cell.provider.value} is "
                    "named twice; a model and provider combination is measured "
                    "once per pass"
                )
            seen.add(cell.key)
        return MatrixResult(
            sample=sample,
            cells=tuple(self.measure_cell(cell, sample) for cell in ordered),
        )

    def measure_cell(self, cell: BenchmarkCell, sample: SampleSpec) -> CellMetrics:
        """Prepare, serve and measure one cell."""
        artifact, preparation = self._prepared(cell)
        service = self._service(cell, artifact)

        memory = ResidentMemorySampler(
            self._memory_source, interval_seconds=self._memory_interval
        )
        with memory:
            result, workload_seconds, energy, window = self._workload(
                cell, service, sample
            )
            latencies = self._latencies(cell, service, sample)

        return CellMetrics(
            model_id=cell.profile.model_id,
            provider=cell.provider,
            provider_served=result.provider_served,
            execution_mode=result.execution_mode,
            partition_verified=result.partition_verified,
            fallback_reason=result.fallback_reason,
            input_count=result.input_count,
            truncated_count=result.truncated_count,
            workload_seconds=workload_seconds,
            sampled_window_seconds=window,
            throughput=_throughput(result.input_count, workload_seconds),
            latency_median=_latency(latencies, median_seconds),
            latency_p95=_latency(
                latencies, lambda values: percentile_seconds(values, 0.95)
            ),
            latency_samples=len(latencies),
            peak_resident_memory=_memory(memory.peak_bytes),
            baseline_resident_memory=_memory(memory.baseline_bytes),
            memory_samples=memory.samples_taken,
            preparation=preparation,
            energy=energy,
        )

    # -- preparation -------------------------------------------------------

    def _prepared(
        self, cell: BenchmarkCell
    ) -> tuple[PreparedArtifact, PreparationTiming]:
        """Prepare once per model and provider, and time both runs.

        Two calls, exactly: the first is requirement 6.2's first-run figure and
        the second is its subsequent-run figure. Every later cell and every
        later repetition reads the cache and pays neither.
        """
        cached = self._artifacts.get(cell.key)
        if cached is not None:
            return cached

        started = self._clock()
        first = self._prepare(cell.profile, cell.provider, self._root)
        first_seconds = self._clock() - started

        started = self._clock()
        again = self._prepare(cell.profile, cell.provider, self._root)
        warm_seconds = self._clock() - started

        timing = PreparationTiming(
            first_run=(
                Measurement(
                    value=None,
                    unit=PREPARATION_UNIT,
                    unavailable_reason=_ALREADY_PREPARED_REASON,
                )
                if first.reused
                else Measurement(
                    value=max(0.0, first_seconds),
                    unit=PREPARATION_UNIT,
                    unavailable_reason=None,
                )
            ),
            warm=(
                Measurement(
                    value=max(0.0, warm_seconds),
                    unit=PREPARATION_UNIT,
                    unavailable_reason=None,
                )
                if again.reused
                else Measurement(
                    value=None,
                    unit=PREPARATION_UNIT,
                    unavailable_reason=_NO_REUSE_REASON,
                )
            ),
            first_run_reused=first.reused,
            artifact_directory=str(again.directory),
        )
        self._artifacts[cell.key] = (again, timing)
        return again, timing

    def _service(
        self, cell: BenchmarkCell, artifact: PreparedArtifact
    ) -> MeasuredService:
        service = self._built.get(cell.key)
        if service is None:
            service = self._services(
                cell.profile, cell.provider, artifact, self._capability
            )
            self._built[cell.key] = service
        return service

    # -- the measured work -------------------------------------------------

    def _workload(
        self, cell: BenchmarkCell, service: MeasuredService, sample: SampleSpec
    ) -> tuple[EmbedResult, float, EnergyMeasurement, float | None]:
        """Embed the whole sample once, sampling power beside it where there is
        power to sample.

        The workload is timed on this harness's own clock, *inside* the
        sampler's context. The sampler keeps its own clock and its window
        therefore brackets the call - it opens before and closes after - so the
        energy denominator is marginally larger than the time the work took and
        the figure marginally overstates. Both windows are published.
        """
        if cell.provider is ProviderChoice.CPU:
            result, seconds = self._timed(service, sample.documents, cell.provider)
            return result, seconds, cpu_energy_unavailable(), None
        if self._power_source is None:
            result, seconds = self._timed(service, sample.documents, cell.provider)
            return result, seconds, _unsampled_energy(NO_POWER_SOURCE_REASON), None

        # One sampler per measured cell: it raises on reuse, like `RunTracker`,
        # and a shared one would blend two cells into a single integral.
        sampler = PowerSampler(
            self._power_source, interval_seconds=self._sampling_interval
        )
        with sampler:
            result, seconds = self._timed(service, sample.documents, cell.provider)
        trace = sampler.trace()
        energy = energy_per_thousand_inputs(
            trace,
            input_count=len(sample.documents),
            # From the capability report, never hardcoded: a part that cannot
            # report power at all and a part that momentarily did not emit the
            # same token (Implementation Note 6.2).
            power_reporting_supported=self._capability.power_reporting_supported,
        )
        return result, seconds, energy, trace.duration_seconds

    def _latencies(
        self, cell: BenchmarkCell, service: MeasuredService, sample: SampleSpec
    ) -> tuple[float, ...]:
        """Time single-input calls, one document at a time.

        Requirement 6.2 asks for *single-input* latency specifically, which is a
        different question from a batch's wall clock divided by its input count:
        a batched run amortises everything the service does once per call.

        What that call contains, and why the figure is not an inference time,
        is stated in `LATENCY_METHODOLOGY` for task 6.7 to render: requirement
        2.7 binds a backend per operation, so each probe pays a session
        construction that measured 77.8% of the published median on this
        machine's CPU control model. The measurement is faithful to what a
        consumer pays; it is the reading of it that needs the caveat.
        """
        documents = sample.documents
        timings: list[float] = []
        for index in range(self._latency_samples):
            document = documents[index % len(documents)]
            _, seconds = self._timed(service, (document,), cell.provider)
            timings.append(seconds)
        return tuple(timings)

    def _timed(
        self,
        service: MeasuredService,
        texts: Sequence[DocumentText],
        provider: ProviderChoice,
    ) -> tuple[EmbedResult, float]:
        started = self._clock()
        result = service.embed_documents(texts, provider)
        return result, max(0.0, self._clock() - started)


def _throughput(input_count: int, seconds: float) -> Measurement:
    """Requirement 6.2's "inputs processed per second" - inputs over seconds."""
    if seconds <= 0.0:
        return Measurement(
            value=None, unit=THROUGHPUT_UNIT, unavailable_reason=_ZERO_WORKLOAD_REASON
        )
    return Measurement(
        value=input_count / seconds, unit=THROUGHPUT_UNIT, unavailable_reason=None
    )


def _latency(
    timings: Sequence[float], reduce: Callable[[Sequence[float]], float]
) -> Measurement:
    if not timings:
        return Measurement(
            value=None, unit=LATENCY_UNIT, unavailable_reason=_NO_LATENCY_REASON
        )
    return Measurement(
        value=reduce(timings), unit=LATENCY_UNIT, unavailable_reason=None
    )


def _memory(value: int | None) -> Measurement:
    if value is None:
        return Measurement(
            value=None, unit=MEMORY_UNIT, unavailable_reason=_NO_MEMORY_REASON
        )
    return Measurement(
        value=float(value), unit=MEMORY_UNIT, unavailable_reason=None
    )
