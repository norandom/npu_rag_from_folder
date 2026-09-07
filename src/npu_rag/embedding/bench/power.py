"""Energy sampling for the benchmark (task 6.2, requirements 6.2 and 6.8).

Requirement 6.2 asks for "energy consumed per one thousand inputs". This machine
exposes no energy counter, so the figure is **integrated**: ``xrt-smi`` is polled
for estimated Watts while the workload runs, and each reading is weighted by the
wall-clock window it actually covers. Requirement 6.8 governs the other half -
"if a measurement cannot be taken on this machine, record the omission and its
reason, and do not report an estimated or substituted value in its place".

Four decisions carry the whole module.

**Polling happens beside the measured call, never inside it.** design.md's
BenchmarkHarness section states this directly: "samples power concurrently with
the measured call, never inside it". ``PowerSampler`` is a context manager that
runs its loop on its own thread, so the timed call cannot contain a subprocess
spawn. The structural consequence is testable: the poll arrives on a different
thread than the one being measured.

**Three states, not two.** design.md sketched ``PowerSampler`` as
``supported() -> bool`` plus ``sample_watts() -> float | None``. That shape
cannot express the middle state, and the middle state is real: Implementation
Note 1.4 measured ``Estimated Power`` reading ``N/A`` on 2 of 39 idle polls of a
part where power reporting *is* supported. So this module consumes
``PowerReading`` - the wrapper's own three-state value (``REPORTED`` /
``UNAVAILABLE`` / ``UNSUPPORTED``) - and keeps the distinction all the way out to
requirement 6.8's omission text:

- *reported*: the poll carried Watts, and it is integrated.
- *unavailable this sample*: the poll came back empty. It is **dropped from the
  integral and counted**, never interpolated and never folded to 0.0 W - which
  would drag the energy figure silently toward zero.
- *unsupported by platform*: the report has no power field at all. AMD documents
  this for PHX/HPT parts and for Linux. Polling harder will not help, and saying
  "this run's polls came back empty" would misdescribe it.

Whether the platform reports power at all is **not** re-derived here.
``CapabilityReport.power_reporting_supported`` already answers exactly that
question, and it is passed in.

**A value and an unavailability reason are mutually exclusive, structurally.**
``Measurement.__post_init__`` refuses to construct an object carrying both or
neither, in the same style as ``Condition``, ``EmbedResult``, ``RunSummary`` and
``PowerReading`` itself. Requirement 6.8 is then unrepresentable to violate at
this boundary rather than merely tested for.

**The CPU row is an omission, not a failure.** ``xrt-smi`` reports NPU power
only; there is no equivalent source for CPU-provider energy on this machine. That
is recorded once, as its own reason, and no number is substituted (6.8).

Integration method, stated because requirement 7.2 asks the benchmark document to
state it: **left-endpoint rectangles over the observed windows.** Each sample
holds until the next poll, or until the sampled window closes for the last one.
A poll that carried no reading holds nothing, so its window is excluded from the
integral and accumulated as ``unmeasured_seconds``; the energy figure therefore
*understates* rather than estimates, and the size of the gap is published beside
it. Trapezoidal integration was rejected for exactly this reason: across a
dropped sample it would interpolate a value that was never observed.

This module sits in the ``bench`` layer - the top of design.md's dependency
direction - and reaches down into ``environment`` for the telemetry types.
Nothing below ``bench`` may import it.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

from npu_rag.embedding.environment.xrt import PowerReading, PowerStatus

__all__ = [
    "CPU_ENERGY_UNAVAILABLE_REASON",
    "DEFAULT_SAMPLING_INTERVAL_SECONDS",
    "ENERGY_UNIT",
    "UNSUPPORTED_PLATFORM_REASON",
    "EnergyMeasurement",
    "Measurement",
    "PowerIntegral",
    "PowerSample",
    "PowerSampler",
    "PowerSource",
    "PowerTrace",
    "cpu_energy_unavailable",
    "energy_per_thousand_inputs",
    "integrate_power",
]

#: One ``xrt-smi examine --report platform`` costs about 0.16 s on this machine
#: (measured 2026-09-07 over 30 consecutive polls). A default interval near that
#: would have the sampler spawning subprocesses back to back and perturbing the
#: very workload it is measuring, so the default leaves the poll at roughly a
#: third duty cycle. It is a default, never a constant: a long run can afford a
#: coarser one, and a short run needs a finer one to see anything at all - which
#: is why the interval actually used travels with every measurement.
DEFAULT_SAMPLING_INTERVAL_SECONDS = 0.5

#: Requirement 6.2's unit of energy per unit of work.
ENERGY_UNIT = "J/1000 inputs"

#: How long ``__exit__`` waits for the sampling thread before declaring the
#: shutdown non-deterministic. Generous against the ~0.16 s a poll costs.
DEFAULT_JOIN_TIMEOUT_SECONDS = 30.0

CPU_ENERGY_UNAVAILABLE_REASON = (
    "Energy was not measured for the CPU provider. xrt-smi reports estimated "
    "power for the NPU only, and this machine exposes no equivalent counter for "
    "CPU-side energy, so the measurement could not be taken rather than merely "
    "failing. No figure is substituted in its place: throughput, latency and "
    "wall-clock remain directly comparable between the providers, and energy "
    "does not."
)

UNSUPPORTED_PLATFORM_REASON = (
    "Energy was not measured: estimated power reporting is unsupported on this "
    "platform. The xrt-smi platform report carries no 'Estimated Power' field "
    "at all, which AMD documents for PHX/HPT parts and for Linux. Sampling for "
    "longer or more often would not produce a reading."
)


def _positive_interval(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number, got {value!r}")
    return number


def _finite(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class Measurement:
    """One benchmark figure, or a typed account of why there isn't one.

    design.md, Data Models: "exactly one of ``value`` and ``unavailable_reason``
    is set". Enforced here rather than asserted elsewhere, so requirement 6.8's
    "shall not report an estimated or substituted value" has no representable
    counter-example at this boundary.
    """

    value: float | None
    unit: str
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        has_value = self.value is not None
        if has_value is (self.unavailable_reason is not None):
            raise ValueError(
                "exactly one of value and unavailable_reason must be set, got "
                f"value={self.value!r}, unavailable_reason="
                f"{self.unavailable_reason!r}"
            )
        if not self.unit.strip():
            raise ValueError("a measurement must name its unit")
        if self.unavailable_reason is not None and not self.unavailable_reason.strip():
            # A blank reason is an omission with no reason, which is precisely
            # what requirement 6.8 exists to prevent.
            raise ValueError("an unavailable measurement must carry a reason")
        if self.value is not None:
            if not math.isfinite(self.value):
                raise ValueError(f"value must be finite, got {self.value!r}")
            if self.value < 0.0:
                raise ValueError(f"value must not be negative, got {self.value!r}")


@dataclass(frozen=True)
class PowerSample:
    """One poll: when it was taken, and what came back."""

    at_seconds: float
    reading: PowerReading


@dataclass(frozen=True)
class PowerTrace:
    """Every poll taken across one sampled window, with the window itself.

    The window is the sampler's own, read from its injected clock immediately
    before the sampling thread starts and immediately after it is joined. It
    therefore brackets the measured call rather than coinciding with it - a
    caveat that costs a sub-millisecond either side and buys a guarantee that no
    sample falls outside the interval the energy is attributed to.
    """

    started_at: float
    stopped_at: float
    interval_seconds: float
    samples: tuple[PowerSample, ...]

    def __post_init__(self) -> None:
        started = _finite(self.started_at, label="started_at")
        stopped = _finite(self.stopped_at, label="stopped_at")
        _positive_interval(self.interval_seconds, label="interval_seconds")
        if stopped < started:
            raise ValueError(
                f"a sampled window cannot run backwards: started_at={started!r}, "
                f"stopped_at={stopped!r}"
            )
        previous = started
        for index, sample in enumerate(self.samples):
            at = _finite(sample.at_seconds, label=f"samples[{index}].at_seconds")
            if at < previous:
                raise ValueError(
                    f"samples must be ordered by time; samples[{index}] at {at!r} "
                    f"precedes {previous!r}"
                )
            if at > stopped:
                raise ValueError(
                    f"samples[{index}] at {at!r} falls outside the sampled window "
                    f"[{started!r}, {stopped!r}]"
                )
            previous = at

    @property
    def duration_seconds(self) -> float:
        return self.stopped_at - self.started_at


@dataclass(frozen=True)
class PowerIntegral:
    """The result of integrating one trace, with its own coverage accounting.

    ``joules`` alone would be a number with no way to judge it. ``missed_samples``
    and ``unmeasured_seconds`` say how much of the window the integral could not
    see, which is the difference between an under-measurement a reader can
    discount and an estimate requirement 6.8 forbids.
    """

    joules: float
    measured_seconds: float
    unmeasured_seconds: float
    reported_samples: int
    missed_samples: int
    unsupported_samples: int


@dataclass(frozen=True)
class EnergyMeasurement:
    """Requirement 6.2's energy figure, or requirement 6.8's omission.

    The sampling interval and the coverage counts travel with it because
    ``Measurement`` cannot carry them: it holds a value **or** a reason, never a
    value plus a caveat. Task 6.3 records this per matrix cell and task 6.7
    renders ``methodology`` into the document's methodology section, which
    requirement 7.2 requires to state "how energy was sampled".
    """

    energy: Measurement
    #: ``None`` only where nothing was sampled at all - the CPU row.
    sampling_interval_seconds: float | None
    samples_taken: int
    samples_reported: int
    samples_missed: int
    measured_seconds: float
    unmeasured_seconds: float

    def __post_init__(self) -> None:
        counts = {
            "samples_taken": self.samples_taken,
            "samples_reported": self.samples_reported,
            "samples_missed": self.samples_missed,
        }
        for label, count in counts.items():
            if count < 0:
                raise ValueError(f"{label} must not be negative, got {count!r}")
        if self.samples_reported + self.samples_missed > self.samples_taken:
            raise ValueError(
                "reported and missed polls cannot exceed the polls taken: "
                f"{self.samples_reported} + {self.samples_missed} > "
                f"{self.samples_taken}"
            )
        for label, seconds in (
            ("measured_seconds", self.measured_seconds),
            ("unmeasured_seconds", self.unmeasured_seconds),
        ):
            value = _finite(seconds, label=label)
            if value < 0.0:
                raise ValueError(f"{label} must not be negative, got {seconds!r}")
        if self.sampling_interval_seconds is not None:
            _positive_interval(
                self.sampling_interval_seconds, label="sampling_interval_seconds"
            )
        if self.energy.value is not None and self.samples_reported <= 0:
            raise ValueError(
                "an energy value must come from at least one reported poll, got "
                f"samples_reported={self.samples_reported!r}"
            )

    @property
    def methodology(self) -> str:
        """One sentence for requirement 7.2's methodology section."""
        interval = self.sampling_interval_seconds
        if interval is None:
            return self.energy.unavailable_reason or ""
        head = (
            f"Estimated NPU power was polled every {interval:g} s and integrated "
            "over wall clock as left-endpoint rectangles"
        )
        if self.energy.value is None:
            return f"{head}. {self.energy.unavailable_reason}"
        body = (
            f"{head}: {self.samples_reported} of {self.samples_taken} polls "
            f"carried a reading, covering {self.measured_seconds:g} s"
        )
        if self.samples_missed:
            return (
                f"{body}. {self.samples_missed} poll(s) returned no reading; the "
                f"{self.unmeasured_seconds:g} s they would have covered are "
                "excluded from the integral rather than interpolated, so the "
                "figure understates rather than estimates."
            )
        return f"{body}."


class PowerSource(Protocol):
    """Where one power reading comes from.

    Deliberately shaped as ``XrtSmiWrapper.read_power`` so the production
    telemetry path is the one under test, rather than an in-memory analogue of
    it (Implementation Note 5.4). The three-state ``PowerReading`` is the whole
    point of the seam: a ``float | None`` here would collapse "this poll came
    back empty" into "this platform cannot report power" and requirement 6.8's
    omission would stop being specific.
    """

    def read_power(self) -> PowerReading: ...


def integrate_power(trace: PowerTrace) -> PowerIntegral:
    """Weight each reading by the window it actually covers.

    Left-endpoint rectangles over *observed* windows, not over the nominal
    sampling interval: a sampler whose poll overran, or whose thread was
    descheduled, produces uneven gaps, and multiplying by the nominal interval
    would attribute energy to time that did not elapse.
    """
    samples = trace.samples
    if not samples:
        return PowerIntegral(
            joules=0.0,
            measured_seconds=0.0,
            unmeasured_seconds=trace.duration_seconds,
            reported_samples=0,
            missed_samples=0,
            unsupported_samples=0,
        )

    joules = 0.0
    measured = 0.0
    # Nothing covers the stretch before the first poll landed.
    unmeasured = max(0.0, samples[0].at_seconds - trace.started_at)
    reported = 0
    missed = 0
    unsupported = 0

    last = len(samples) - 1
    for index, sample in enumerate(samples):
        boundary = (
            samples[index + 1].at_seconds if index < last else trace.stopped_at
        )
        span = max(0.0, boundary - sample.at_seconds)
        watts = sample.reading.watts
        if watts is None:
            # Dropped, never folded to 0.0 W and never interpolated across
            # (Implementation Note 1.4). The span is published as a gap instead.
            unmeasured += span
            if sample.reading.status is PowerStatus.UNSUPPORTED:
                unsupported += 1
            else:
                missed += 1
            continue
        reported += 1
        measured += span
        joules += watts * span

    return PowerIntegral(
        joules=joules,
        measured_seconds=measured,
        unmeasured_seconds=unmeasured,
        reported_samples=reported,
        missed_samples=missed,
        unsupported_samples=unsupported,
    )


def _no_reading_reason(
    trace: PowerTrace, integral: PowerIntegral, *, power_reporting_supported: bool
) -> str:
    """Requirement 6.8's omission, kept specific across three distinct causes."""
    if integral.reported_samples + integral.missed_samples + (
        integral.unsupported_samples
    ) == 0:
        return (
            "Energy was not measured: no poll completed during the "
            f"{trace.duration_seconds:g} s sampled window at a "
            f"{trace.interval_seconds:g} s sampling interval. The workload "
            "finished before the sampler could take a reading, so a finer "
            "interval is needed to measure a run this short."
        )
    if not power_reporting_supported or integral.unsupported_samples > 0:
        return UNSUPPORTED_PLATFORM_REASON
    polls = integral.missed_samples
    return (
        f"Energy was not measured: all {polls} poll(s) taken during the "
        f"{trace.duration_seconds:g} s sampled window came back with no reading. "
        "xrt-smi prints 'N/A' for estimated power intermittently even on a part "
        "that does report it, so this is an omission for this particular run "
        "rather than a property of the platform, and a repeat run may well "
        "succeed. No value is inferred from the polls that came back empty."
    )


def energy_per_thousand_inputs(
    trace: PowerTrace, *, input_count: int, power_reporting_supported: bool
) -> EnergyMeasurement:
    """Requirement 6.2's "energy consumed per one thousand inputs".

    ``power_reporting_supported`` comes from
    ``CapabilityReport.power_reporting_supported``, which already decides whether
    this platform reports estimated power **at all**. It is passed in rather than
    re-derived from the samples because a part that cannot report power and a
    part that momentarily did not both answer with the same token, and only the
    capability check has the evidence to tell them apart.

    Precedence, where the two disagree: an observation outranks the flag. If any
    poll actually carried Watts, the integral is reported even under
    ``power_reporting_supported=False``, because a reading that exists is
    evidence and a capability flag is only a prediction; the flag decides solely
    which omission text applies when *no* poll carried a reading.
    """
    if input_count <= 0:
        raise ValueError(
            f"energy per one thousand inputs needs a positive input count, got "
            f"{input_count!r}"
        )
    integral = integrate_power(trace)

    if integral.reported_samples == 0:
        energy = Measurement(
            value=None,
            unit=ENERGY_UNIT,
            unavailable_reason=_no_reading_reason(
                trace, integral, power_reporting_supported=power_reporting_supported
            ),
        )
    else:
        energy = Measurement(
            value=integral.joules / input_count * 1000.0,
            unit=ENERGY_UNIT,
            unavailable_reason=None,
        )

    return EnergyMeasurement(
        energy=energy,
        sampling_interval_seconds=trace.interval_seconds,
        samples_taken=len(trace.samples),
        samples_reported=integral.reported_samples,
        samples_missed=integral.missed_samples,
        measured_seconds=integral.measured_seconds,
        unmeasured_seconds=integral.unmeasured_seconds,
    )


def cpu_energy_unavailable() -> EnergyMeasurement:
    """The CPU row's energy: a stated omission, never a substituted number.

    Task 6.2: "record the CPU provider's energy as unavailable with that reason".
    This is not a failure path - nothing went wrong - so it is produced directly
    rather than derived from an empty trace, and it carries no sampling interval
    because no sampling was attempted.
    """
    return EnergyMeasurement(
        energy=Measurement(
            value=None,
            unit=ENERGY_UNIT,
            unavailable_reason=CPU_ENERGY_UNAVAILABLE_REASON,
        ),
        sampling_interval_seconds=None,
        samples_taken=0,
        samples_reported=0,
        samples_missed=0,
        measured_seconds=0.0,
        unmeasured_seconds=0.0,
    )


class PowerSampler:
    """Poll a power source on its own thread for the duration of a measured call.

    Used as a context manager wrapped **around** the workload::

        with PowerSampler(wrapper) as sampler:
            result = service.embed_documents(batch)
        measured = energy_per_thousand_inputs(
            sampler.trace(), input_count=len(batch),
            power_reporting_supported=capability.power_reporting_supported,
        )

    The sampler never calls the workload and the workload never calls the
    sampler, which is how design.md's "never inside it" becomes structural rather
    than a convention. ``__exit__`` sets the stop flag and **joins**, so the last
    poll is complete before the trace window closes and no poll can land after a
    measurement has been read - shutdown is deterministic rather than eventual.

    Single-use, for the same reason ``RunTracker`` is (Implementation Note 2.3):
    one sampler measures one operation, and a reused one would silently blend two
    windows into a single integral.
    """

    def __init__(
        self,
        source: PowerSource,
        *,
        interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
        join_timeout_seconds: float = DEFAULT_JOIN_TIMEOUT_SECONDS,
    ) -> None:
        self._source = source
        self._interval = _positive_interval(interval_seconds, label="interval_seconds")
        self._join_timeout = _positive_interval(
            join_timeout_seconds, label="join_timeout_seconds"
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._samples: list[PowerSample] = []
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._entered = False

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError(
                "a PowerSampler measures one operation and cannot be reused; "
                "construct a new one per measured call"
            )
        self._entered = True
        self._started_at = self._clock()
        thread = threading.Thread(
            target=self._loop, name="npu-rag-power-sampler", daemon=True
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
        del exc
        del traceback
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._join_timeout)
            if thread.is_alive() and exc_type is None:
                # Only raised when the measured call itself did not fail, so a
                # hung sampler never masks the workload's own exception.
                raise RuntimeError(
                    "the power sampling thread did not stop within "
                    f"{self._join_timeout:g} s; the trace window cannot be closed"
                )
        self._stopped_at = self._clock()

    def _loop(self) -> None:
        while not self._stop.is_set():
            at = self._clock()
            try:
                reading = self._source.read_power()
            except Exception as exc:  # noqa: BLE001 - a dead thread loses the trace
                # ``XrtSmiWrapper`` never raises, but the source is a Protocol.
                # An exception becomes data, exactly as an absent reading does,
                # and sampling stops because a source that raised once will
                # almost certainly raise again every interval.
                self._record(
                    at,
                    PowerReading(
                        watts=None,
                        status=PowerStatus.UNAVAILABLE,
                        reason=f"the power source raised {exc!r}",
                    ),
                )
                return
            self._record(at, reading)
            if self._stop.wait(self._interval):
                return

    def _record(self, at: float, reading: PowerReading) -> None:
        with self._lock:
            self._samples.append(PowerSample(at_seconds=at, reading=reading))

    def trace(self) -> PowerTrace:
        """Every poll taken, with the window they were taken in.

        Available only after the context has closed: before then the window has
        no end, and a trace with a guessed end is the kind of invented number
        requirement 6.8 forbids.
        """
        started = self._started_at
        stopped = self._stopped_at
        if started is None or stopped is None:
            raise RuntimeError(
                "the sampled window is not closed yet; read trace() after the "
                "PowerSampler context exits"
            )
        with self._lock:
            samples = tuple(self._samples)
        return PowerTrace(
            started_at=started,
            stopped_at=stopped,
            interval_seconds=self._interval,
            samples=samples,
        )
