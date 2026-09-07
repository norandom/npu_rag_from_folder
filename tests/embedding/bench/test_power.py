"""Unit tests for benchmark energy sampling (task 6.2, requirements 6.2, 6.8).

Nothing here needs an NPU. The concurrency tests drive ``PowerSampler`` against
in-memory sources so shutdown is deterministic, and the *seam* tests drive it
against a **real** ``XrtSmiWrapper`` whose only stand-in is the subprocess call
itself - the argument vector, the parser and the three-state ``PowerReading``
are the production ones. That distinction is Implementation Note 5.4's: a
fixture can be non-vacuous against a system that was never assembled that way,
so at least one test must assemble the real one.

The live counterpart is ``test_power_live.py``, which polls the real binary and
skips when it is absent. Per Implementation Note 4.2 CI has no NPU, so nothing
asserted here may depend on that file running.

**The hand-computed energy literal.** ``SERIES`` below is deliberately
non-uniform in *both* dimensions - the wattages differ from each other and the
gaps between polls differ from each other and from the nominal sampling
interval. A uniform series cannot tell a correct integral from ``mean x
duration``, from ``first x duration``, or from a sum that forgot the interval;
``test_the_energy_literal_discriminates`` pins that it can, and fails loudly if
someone later "simplifies" the fixture.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from npu_rag.embedding.bench.power import (
    CPU_ENERGY_UNAVAILABLE_REASON,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    ENERGY_UNIT,
    EnergyMeasurement,
    Measurement,
    PowerSample,
    PowerSampler,
    PowerTrace,
    cpu_energy_unavailable,
    energy_per_thousand_inputs,
    integrate_power,
)
from npu_rag.embedding.environment.xrt import (
    CommandResult,
    PowerReading,
    PowerStatus,
    XrtSmiWrapper,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "xrt"

#: ``platform_load.txt``'s own ``Estimated Power``. Named here so a seam test
#: asserts the wrapper carried the fixture's number through, rather than merely
#: producing *a* number.
LOAD_FIXTURE_WATTS = 0.438


def fixture(name: str) -> str:
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Reading constructors
# --------------------------------------------------------------------------


def reported(watts: float) -> PowerReading:
    return PowerReading(watts=watts, status=PowerStatus.REPORTED, reason=None)


def na() -> PowerReading:
    """The mid-run ``N/A`` measured 2 times in 39 polls on this Strix part."""
    return PowerReading(
        watts=None,
        status=PowerStatus.UNAVAILABLE,
        reason="xrt-smi reported Estimated Power as 'N/A'.",
    )


def unsupported() -> PowerReading:
    return PowerReading(
        watts=None,
        status=PowerStatus.UNSUPPORTED,
        reason="the platform report carries no 'Estimated Power' field.",
    )


def trace_of(
    readings: Sequence[tuple[float, PowerReading]],
    *,
    started_at: float,
    stopped_at: float,
    interval_seconds: float = 0.5,
) -> PowerTrace:
    return PowerTrace(
        started_at=started_at,
        stopped_at=stopped_at,
        interval_seconds=interval_seconds,
        samples=tuple(
            PowerSample(at_seconds=at, reading=reading) for at, reading in readings
        ),
    )


# --------------------------------------------------------------------------
# The hand-computed series
# --------------------------------------------------------------------------

#: Window [10.0, 13.0]; nominal interval 0.4 s, which matches NONE of the actual
#: gaps (0.5 and 1.0), so an implementation that multiplies by the nominal
#: interval instead of the observed window is killed. It is also deliberately
#: *not* ``DEFAULT_SAMPLING_INTERVAL_SECONDS``: at the default, an implementation
#: that hardcoded the module default instead of reading the trace's own interval
#: would be indistinguishable from a correct one, and
#: ``test_the_sampling_interval_used_is_recorded`` would be vacuous. That
#: separation is itself asserted there, so flattening this back onto the default
#: fails loudly.
SERIES_START = 10.0
SERIES_END = 13.0
SERIES_INTERVAL = 0.4

#: (timestamp, reading). Windows are 0.5, 1.0, 0.5, 1.0 s.
SERIES: tuple[tuple[float, PowerReading], ...] = (
    (10.0, reported(2.0)),
    (10.5, reported(6.0)),
    (11.5, na()),
    (12.0, reported(4.0)),
)

#: Hand-computed, by hand, as literals:
#:   2.0 W x 0.5 s =  1.0 J
#:   6.0 W x 1.0 s =  6.0 J
#:   N/A  x 0.5 s  =  -   (dropped, never interpolated, never 0.0 W)
#:   4.0 W x 1.0 s =  4.0 J
#:                   ------
#:                   11.0 J
SERIES_JOULES = 11.0
SERIES_MEASURED_SECONDS = 2.5
SERIES_UNMEASURED_SECONDS = 0.5
SERIES_INPUT_COUNT = 250
#: 11.0 J over 250 inputs -> 44.0 J per 1000 inputs.
SERIES_JOULES_PER_THOUSAND = 44.0


@pytest.fixture
def series() -> PowerTrace:
    return trace_of(
        SERIES,
        started_at=SERIES_START,
        stopped_at=SERIES_END,
        interval_seconds=SERIES_INTERVAL,
    )


# --------------------------------------------------------------------------
# Measurement: a value and an unavailability reason are mutually exclusive
# --------------------------------------------------------------------------


def test_a_measurement_refuses_both_a_value_and_a_reason() -> None:
    """design.md, Unit Tests: "``Measurement`` rejects construction with both
    ``value`` and ``unavailable_reason`` set, or neither (6.8)". Structural, in
    the style of ``Condition``/``EmbedResult``/``RunSummary``."""
    with pytest.raises(ValueError):
        Measurement(value=1.0, unit=ENERGY_UNIT, unavailable_reason="no NPU here")


def test_a_measurement_refuses_neither_a_value_nor_a_reason() -> None:
    with pytest.raises(ValueError):
        Measurement(value=None, unit=ENERGY_UNIT, unavailable_reason=None)


def test_a_measurement_refuses_a_blank_reason() -> None:
    """An empty reason is an omission with no reason, which is what 6.8 forbids."""
    with pytest.raises(ValueError):
        Measurement(value=None, unit=ENERGY_UNIT, unavailable_reason="   ")


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
def test_a_measurement_refuses_an_impossible_value(value: float) -> None:
    with pytest.raises(ValueError):
        Measurement(value=value, unit=ENERGY_UNIT, unavailable_reason=None)


def test_a_measurement_refuses_a_blank_unit() -> None:
    with pytest.raises(ValueError):
        Measurement(value=1.0, unit="  ", unavailable_reason=None)


def test_a_coherent_measurement_constructs() -> None:
    both_ways = (
        Measurement(value=0.0, unit=ENERGY_UNIT, unavailable_reason=None),
        Measurement(value=None, unit=ENERGY_UNIT, unavailable_reason="no source"),
    )
    assert both_ways[0].value == 0.0
    assert both_ways[1].unavailable_reason == "no source"


# --------------------------------------------------------------------------
# The integral
# --------------------------------------------------------------------------


def test_each_reading_is_weighted_by_its_own_observed_window(
    series: PowerTrace,
) -> None:
    """Requirement 6.2's integral, against a literal computed by hand."""
    integral = integrate_power(series)

    assert integral.joules == pytest.approx(SERIES_JOULES)
    assert integral.measured_seconds == pytest.approx(SERIES_MEASURED_SECONDS)
    assert integral.unmeasured_seconds == pytest.approx(SERIES_UNMEASURED_SECONDS)
    assert integral.reported_samples == 3
    assert integral.missed_samples == 1
    assert integral.unsupported_samples == 0


def test_the_energy_literal_discriminates(series: PowerTrace) -> None:
    """Non-vacuity control (Implementation Notes, standing lesson rule 3).

    Every wrong integration this fixture is meant to catch must produce a
    different number from the literal. If someone flattens the series later,
    this test fails and says so rather than silently disarming the one above.
    """
    watts = [
        sample.reading.watts
        for sample in series.samples
        if sample.reading.watts is not None
    ]
    duration = series.stopped_at - series.started_at
    wrong = {
        "mean x duration": (sum(watts) / len(watts)) * duration,
        "mean x measured": (sum(watts) / len(watts)) * SERIES_MEASURED_SECONDS,
        "first x duration": watts[0] * duration,
        "last x duration": watts[-1] * duration,
        "sum of watts, interval dropped": sum(watts),
        "sum x nominal interval": sum(watts) * SERIES_INTERVAL,
        "sum x duration": sum(watts) * duration,
        "uniform windows": sum(watts) * (duration / len(series.samples)),
    }
    for name, value in wrong.items():
        assert value != pytest.approx(SERIES_JOULES), name


def test_an_absent_reading_is_dropped_from_the_integral_never_folded_to_zero(
    series: PowerTrace,
) -> None:
    """Implementation Note 1.4: "``N/A`` must never be folded to 0.0 W".

    Folding is invisible in the joules - 0.0 W over any window adds 0.0 J - so
    the discriminator is the *accounting*: a folded sample would count as
    reported and its window as measured. Both traces are asserted, side by side,
    so the property is pinned by the difference rather than by one number.
    """
    folded = trace_of(
        (
            (10.0, reported(2.0)),
            (10.5, reported(6.0)),
            (11.5, reported(0.0)),
            (12.0, reported(4.0)),
        ),
        started_at=SERIES_START,
        stopped_at=SERIES_END,
        interval_seconds=SERIES_INTERVAL,
    )

    honest = integrate_power(series)
    zeroed = integrate_power(folded)

    assert zeroed.joules == pytest.approx(honest.joules)
    assert honest.missed_samples == 1
    assert zeroed.missed_samples == 0
    assert honest.reported_samples == 3
    assert zeroed.reported_samples == 4
    assert honest.measured_seconds == pytest.approx(2.5)
    assert zeroed.measured_seconds == pytest.approx(3.0)
    assert honest.unmeasured_seconds == pytest.approx(0.5)
    assert zeroed.unmeasured_seconds == pytest.approx(0.0)


def test_the_gap_before_the_first_poll_is_unmeasured() -> None:
    """A second hand-computed literal, on a differently shaped window.

      leading gap 0.0 -> 1.0 s : unmeasured
      3.0 W x 1.0 s            :  3.0 J
      5.0 W x 2.0 s            : 10.0 J
                                 -------
                                 13.0 J
    """
    integral = integrate_power(
        trace_of(
            ((1.0, reported(3.0)), (2.0, reported(5.0))),
            started_at=0.0,
            stopped_at=4.0,
        )
    )

    assert integral.joules == pytest.approx(13.0)
    assert integral.measured_seconds == pytest.approx(3.0)
    assert integral.unmeasured_seconds == pytest.approx(1.0)


def test_a_trace_with_no_polls_measures_nothing() -> None:
    integral = integrate_power(trace_of((), started_at=5.0, stopped_at=8.0))

    assert integral.joules == 0.0
    assert integral.reported_samples == 0
    assert integral.measured_seconds == pytest.approx(0.0)
    assert integral.unmeasured_seconds == pytest.approx(3.0)


def test_unsupported_and_unavailable_polls_are_counted_apart() -> None:
    integral = integrate_power(
        trace_of(
            ((0.0, na()), (1.0, unsupported()), (2.0, reported(1.0))),
            started_at=0.0,
            stopped_at=3.0,
        )
    )

    assert integral.missed_samples == 1
    assert integral.unsupported_samples == 1
    assert integral.reported_samples == 1


def test_a_trace_refuses_a_window_that_runs_backwards() -> None:
    with pytest.raises(ValueError):
        trace_of((), started_at=3.0, stopped_at=1.0)


def test_a_trace_refuses_a_sample_outside_its_window() -> None:
    with pytest.raises(ValueError):
        trace_of(((9.0, reported(1.0)),), started_at=0.0, stopped_at=1.0)


def test_a_trace_refuses_samples_out_of_order() -> None:
    with pytest.raises(ValueError):
        trace_of(
            ((2.0, reported(1.0)), (1.0, reported(1.0))),
            started_at=0.0,
            stopped_at=3.0,
        )


@pytest.mark.parametrize("interval", [0.0, -1.0, math.nan])
def test_a_trace_refuses_a_non_positive_sampling_interval(interval: float) -> None:
    with pytest.raises(ValueError):
        trace_of((), started_at=0.0, stopped_at=1.0, interval_seconds=interval)


# --------------------------------------------------------------------------
# Energy per one thousand inputs
# --------------------------------------------------------------------------


def test_energy_is_normalised_per_thousand_inputs(series: PowerTrace) -> None:
    """Requirement 6.2 asks for "energy consumed per one thousand inputs"."""
    measured = energy_per_thousand_inputs(
        series, input_count=SERIES_INPUT_COUNT, power_reporting_supported=True
    )

    assert measured.energy.value == pytest.approx(SERIES_JOULES_PER_THOUSAND)
    assert measured.energy.unit == ENERGY_UNIT
    assert measured.energy.unavailable_reason is None


def test_the_normalisation_actually_divides_by_the_input_count(
    series: PowerTrace,
) -> None:
    """Non-vacuity: 250 inputs is not 1000, so the raw joules and the normalised
    figure differ, and halving the input count doubles the answer."""
    at_250 = energy_per_thousand_inputs(
        series, input_count=250, power_reporting_supported=True
    )
    at_500 = energy_per_thousand_inputs(
        series, input_count=500, power_reporting_supported=True
    )

    assert at_250.energy.value != pytest.approx(SERIES_JOULES)
    assert at_250.energy.value == pytest.approx(2.0 * (at_500.energy.value or 0.0))
    assert at_500.energy.value == pytest.approx(22.0)


def test_the_sampling_interval_used_is_recorded(series: PowerTrace) -> None:
    """Task 6.2: "recording the sampling interval used". Requirement 7.2 renders
    it into the methodology, so it must survive into the measurement rather than
    staying in the trace."""
    # Non-vacuity control (standing lesson rule 3). Both assertions below are
    # satisfied by an implementation that hardcodes the module default whenever
    # the fixture's interval happens to equal it, so the separation is the
    # precondition of the test rather than an incidental property of the fixture.
    assert SERIES_INTERVAL != DEFAULT_SAMPLING_INTERVAL_SECONDS

    measured = energy_per_thousand_inputs(
        series, input_count=SERIES_INPUT_COUNT, power_reporting_supported=True
    )

    assert measured.sampling_interval_seconds == pytest.approx(SERIES_INTERVAL)
    assert f"{SERIES_INTERVAL:g}" in measured.methodology


def test_the_missed_poll_count_survives_into_the_measurement(
    series: PowerTrace,
) -> None:
    """The caveat has nowhere else to live: ``Measurement`` carries a value XOR a
    reason, so a partially-sampled integral must disclose the gap here."""
    measured = energy_per_thousand_inputs(
        series, input_count=SERIES_INPUT_COUNT, power_reporting_supported=True
    )

    assert measured.samples_taken == 4
    assert measured.samples_reported == 3
    assert measured.samples_missed == 1
    assert measured.measured_seconds == pytest.approx(SERIES_MEASURED_SECONDS)
    assert measured.unmeasured_seconds == pytest.approx(SERIES_UNMEASURED_SECONDS)
    assert "1" in measured.methodology


def test_a_platform_that_reports_no_power_at_all_yields_an_unsupported_reason() -> None:
    """Implementation Note 1.4's third state, taken from the capability report's
    ``power_reporting_supported`` rather than re-derived here."""
    measured = energy_per_thousand_inputs(
        trace_of(
            ((0.0, unsupported()), (1.0, unsupported())),
            started_at=0.0,
            stopped_at=2.0,
        ),
        input_count=100,
        power_reporting_supported=False,
    )

    assert measured.energy.value is None
    reason = measured.energy.unavailable_reason or ""
    assert "unsupported" in reason.lower()
    assert "estimated power" in reason.lower()


def test_the_capability_flag_alone_makes_the_omission_unsupported() -> None:
    """A PHX part answers every poll with ``N/A`` - the same token this Strix
    part emits intermittently. The platform verdict is what tells them apart, so
    it must be consulted even when the samples look identical."""
    samples = ((0.0, na()), (1.0, na()))
    on_a_part_that_cannot = energy_per_thousand_inputs(
        trace_of(samples, started_at=0.0, stopped_at=2.0),
        input_count=100,
        power_reporting_supported=False,
    )
    on_a_part_that_can = energy_per_thousand_inputs(
        trace_of(samples, started_at=0.0, stopped_at=2.0),
        input_count=100,
        power_reporting_supported=True,
    )

    assert on_a_part_that_cannot.energy.unavailable_reason is not None
    assert on_a_part_that_can.energy.unavailable_reason is not None
    assert (
        on_a_part_that_cannot.energy.unavailable_reason
        != on_a_part_that_can.energy.unavailable_reason
    )


def test_polls_that_all_came_back_empty_stay_distinct_from_unsupported() -> None:
    """Requirement 6.8's omission must be *specific*. "the platform cannot report
    power" and "every poll in this run came back empty" are different facts and a
    reader acts on them differently: the first is permanent, the second is not."""
    unavailable = energy_per_thousand_inputs(
        trace_of(((0.0, na()), (1.0, na())), started_at=0.0, stopped_at=2.0),
        input_count=100,
        power_reporting_supported=True,
    ).energy.unavailable_reason
    unsupported_reason = energy_per_thousand_inputs(
        trace_of(
            ((0.0, unsupported()), (1.0, unsupported())),
            started_at=0.0,
            stopped_at=2.0,
        ),
        input_count=100,
        power_reporting_supported=False,
    ).energy.unavailable_reason

    assert unavailable is not None
    assert unsupported_reason is not None
    assert unavailable != unsupported_reason
    assert unsupported_reason not in unavailable
    assert unavailable not in unsupported_reason
    assert "unsupported" not in unavailable.lower()
    assert "unsupported" in unsupported_reason.lower()


def test_a_run_with_no_poll_at_all_says_so(series: PowerTrace) -> None:
    """Distinct from both: a workload shorter than one sampling interval."""
    del series
    measured = energy_per_thousand_inputs(
        trace_of((), started_at=0.0, stopped_at=0.01, interval_seconds=0.5),
        input_count=10,
        power_reporting_supported=True,
    )

    assert measured.energy.value is None
    reason = measured.energy.unavailable_reason or ""
    assert "no poll" in reason.lower()
    assert measured.samples_taken == 0


def test_energy_is_never_reported_when_no_poll_carried_a_number() -> None:
    """The Observable's "neither is ever an estimate", from the NPU side."""
    for supported in (True, False):
        measured = energy_per_thousand_inputs(
            trace_of(((0.0, na()),), started_at=0.0, stopped_at=2.0),
            input_count=100,
            power_reporting_supported=supported,
        )
        assert measured.energy.value is None
        assert measured.energy.unavailable_reason is not None


@pytest.mark.parametrize("count", [0, -1])
def test_a_non_positive_input_count_is_refused(
    series: PowerTrace, count: int
) -> None:
    with pytest.raises(ValueError):
        energy_per_thousand_inputs(
            series, input_count=count, power_reporting_supported=True
        )


def test_an_energy_measurement_refuses_incoherent_accounting() -> None:
    with pytest.raises(ValueError):
        EnergyMeasurement(
            energy=Measurement(value=1.0, unit=ENERGY_UNIT, unavailable_reason=None),
            sampling_interval_seconds=0.5,
            samples_taken=1,
            samples_reported=5,
            samples_missed=0,
            measured_seconds=1.0,
            unmeasured_seconds=0.0,
        )


def test_an_energy_value_cannot_exist_without_a_poll_behind_it() -> None:
    """Requirement 6.8, structurally. A value with no reported poll behind it is
    precisely the estimated-or-substituted figure 6.8 forbids: nothing was ever
    read, yet a number is on offer. The sibling guard above catches incoherent
    *counts*; this one catches a coherent count of zero under a real value, and
    they are different mistakes."""
    with pytest.raises(ValueError):
        EnergyMeasurement(
            energy=Measurement(value=1.0, unit=ENERGY_UNIT, unavailable_reason=None),
            sampling_interval_seconds=0.5,
            samples_taken=0,
            samples_reported=0,
            samples_missed=0,
            measured_seconds=0.0,
            unmeasured_seconds=0.0,
        )


# --------------------------------------------------------------------------
# The CPU row: an omission, not a failure
# --------------------------------------------------------------------------


def test_cpu_energy_is_unavailable_with_its_own_specific_reason() -> None:
    """Task 6.2: "record the CPU provider's energy as unavailable with that
    reason". design.md: "``xrt-smi`` reports NPU power only"."""
    measured = cpu_energy_unavailable()

    assert measured.energy.value is None
    assert measured.energy.unit == ENERGY_UNIT
    assert measured.energy.unavailable_reason == CPU_ENERGY_UNAVAILABLE_REASON
    assert "xrt-smi" in CPU_ENERGY_UNAVAILABLE_REASON
    assert "cpu" in CPU_ENERGY_UNAVAILABLE_REASON.lower()
    assert measured.sampling_interval_seconds is None
    assert measured.samples_taken == 0


def test_the_cpu_reason_is_not_reused_for_either_npu_omission() -> None:
    npu_reasons = {
        energy_per_thousand_inputs(
            trace_of(((0.0, na()),), started_at=0.0, stopped_at=1.0),
            input_count=100,
            power_reporting_supported=True,
        ).energy.unavailable_reason,
        energy_per_thousand_inputs(
            trace_of(((0.0, unsupported()),), started_at=0.0, stopped_at=1.0),
            input_count=100,
            power_reporting_supported=False,
        ).energy.unavailable_reason,
        energy_per_thousand_inputs(
            trace_of((), started_at=0.0, stopped_at=1.0),
            input_count=100,
            power_reporting_supported=True,
        ).energy.unavailable_reason,
    }

    assert len(npu_reasons) == 3
    assert CPU_ENERGY_UNAVAILABLE_REASON not in npu_reasons


def test_an_npu_run_and_the_equivalent_cpu_run_land_on_opposite_sides(
    series: PowerTrace,
) -> None:
    """Task 6.2's Observable, stated as one assertion."""
    npu = energy_per_thousand_inputs(
        series, input_count=SERIES_INPUT_COUNT, power_reporting_supported=True
    )
    cpu = cpu_energy_unavailable()

    assert npu.energy.value is not None
    assert npu.energy.unavailable_reason is None
    assert cpu.energy.value is None
    assert cpu.energy.unavailable_reason is not None
    assert npu.energy.unit == cpu.energy.unit


# --------------------------------------------------------------------------
# The sampler: concurrent with the measured call, never inside it
# --------------------------------------------------------------------------


class RecordingSource:
    """A power source that records every poll and the thread it arrived on."""

    def __init__(self, readings: Sequence[PowerReading]) -> None:
        self._readings = list(readings)
        self._lock = threading.Lock()
        self.polls = 0
        self.thread_ids: set[int] = set()
        self.polled = threading.Event()

    def read_power(self) -> PowerReading:
        with self._lock:
            reading = self._readings[min(self.polls, len(self._readings) - 1)]
            self.polls += 1
            self.thread_ids.add(threading.get_ident())
        self.polled.set()
        return reading


class ExplodingSource:
    def __init__(self) -> None:
        self.polls = 0
        self.polled = threading.Event()

    def read_power(self) -> PowerReading:
        self.polls += 1
        self.polled.set()
        raise OSError("telemetry exploded")


class SteppingClock:
    """A monotonic clock with a fixed step, safe to read from two threads."""

    def __init__(self, *, start: float = 0.0, step: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._now = start
        self._step = step

    def __call__(self) -> float:
        with self._lock:
            value = self._now
            self._now += self._step
            return value


def test_polling_happens_during_the_measured_call_not_after_it() -> None:
    """Task 6.2: "concurrently with the measured workload, never inside the
    measured call". A sampler that polled only on exit would leave
    ``polls_during`` at zero."""
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(source, interval_seconds=0.001)

    with sampler:
        assert source.polled.wait(timeout=10.0)
        polls_during = source.polls

    assert polls_during >= 1


def test_the_sampler_polls_on_a_thread_other_than_the_measured_call() -> None:
    """The structural form of "never inside the measured call": the poll cannot
    be on the measuring thread, so it cannot be inside the call being timed."""
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(source, interval_seconds=0.001)

    with sampler:
        assert source.polled.wait(timeout=10.0)
        measuring_thread = threading.get_ident()

    assert source.thread_ids
    assert measuring_thread not in source.thread_ids


def test_the_sampler_stops_polling_once_the_context_exits() -> None:
    """Deterministic shutdown: ``__exit__`` joins the sampling thread, so no
    poll can land after it. With a 1 ms interval an unjoined thread would add
    tens of polls across the settle window below."""
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(source, interval_seconds=0.001)

    with sampler:
        assert source.polled.wait(timeout=10.0)

    assert sampler.is_running is False
    settled = source.polls
    time.sleep(0.1)
    assert source.polls == settled


def test_the_trace_holds_exactly_what_was_polled() -> None:
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(source, interval_seconds=0.001)

    with sampler:
        assert source.polled.wait(timeout=10.0)
    trace = sampler.trace()

    assert len(trace.samples) == source.polls
    assert trace.interval_seconds == pytest.approx(0.001)


def test_the_trace_window_brackets_every_sample() -> None:
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(source, interval_seconds=0.001)

    with sampler:
        assert source.polled.wait(timeout=10.0)
    trace = sampler.trace()

    assert trace.samples
    assert trace.started_at <= min(sample.at_seconds for sample in trace.samples)
    assert max(sample.at_seconds for sample in trace.samples) <= trace.stopped_at
    assert trace.duration_seconds == pytest.approx(trace.stopped_at - trace.started_at)


def test_the_trace_window_comes_from_the_injected_clock() -> None:
    """The project already injects a ``clock`` into ``EmbeddingService`` for
    exactly this reason (Implementation Note 5.3): where a seam exists to make an
    assertion exact, an inequality is a choice rather than a constraint."""
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(
        source, interval_seconds=0.001, clock=SteppingClock(start=100.0, step=1.0)
    )

    with sampler:
        assert source.polled.wait(timeout=10.0)
    trace = sampler.trace()

    assert trace.started_at == 100.0
    assert trace.stopped_at > trace.started_at


def test_the_trace_is_unavailable_before_the_context_exits() -> None:
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(source, interval_seconds=0.001)

    with pytest.raises(RuntimeError):
        sampler.trace()

    with sampler:
        with pytest.raises(RuntimeError):
            sampler.trace()


def test_the_sampler_is_single_use() -> None:
    """Implementation Note 2.3's lesson, applied to the second thing in this
    package that measures one operation."""
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(source, interval_seconds=0.001)

    with sampler:
        assert source.polled.wait(timeout=10.0)

    with pytest.raises(RuntimeError), sampler:
        pass  # pragma: no cover - the context body never runs


def test_a_source_that_raises_becomes_an_unavailable_reading() -> None:
    """A sampling thread that dies takes the measurement's provenance with it.
    ``xrt-smi``'s own wrapper never raises, but the seam is a Protocol and this
    is the only place that can hold that line for any implementation of it."""
    source = ExplodingSource()
    sampler = PowerSampler(source, interval_seconds=0.001)

    with sampler:
        assert source.polled.wait(timeout=10.0)
    trace = sampler.trace()

    assert len(trace.samples) == 1
    reading = trace.samples[0].reading
    assert reading.status is PowerStatus.UNAVAILABLE
    assert reading.watts is None
    assert "telemetry exploded" in (reading.reason or "")
    assert source.polls == 1


@pytest.mark.parametrize("interval", [0.0, -0.5, math.nan])
def test_the_sampler_refuses_a_non_positive_interval(interval: float) -> None:
    with pytest.raises(ValueError):
        PowerSampler(RecordingSource([reported(1.0)]), interval_seconds=interval)


def test_the_sampler_waits_its_interval_between_polls() -> None:
    """The interval is honoured rather than spun through.

    A sampler that ignored it would poll continuously and perturb the very
    workload task 6.2 asks it not to disturb - the concern the test below states
    as a number. The property was previously pinned only by ``test_power_live.py``
    (``len(trace.samples) <= 1`` at a 30 s interval), and per Implementation Note
    4.2 that file never runs on CI, so it was effectively uncovered.
    """
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(source, interval_seconds=30.0)

    with sampler:
        assert source.polled.wait(timeout=10.0)

    # One poll at entry, then the wait - and ``__exit__``'s stop event cuts the
    # wait short, so this returns immediately rather than in 30 s.
    assert source.polls == 1
    assert len(sampler.trace().samples) == 1


def test_the_default_interval_is_coarser_than_one_poll_costs() -> None:
    """Measured on this machine 2026-09-07: one ``xrt-smi`` platform report costs
    about 0.16 s. A default interval below that would have the sampler polling
    back to back, which is exactly the perturbation task 6.2 warns against."""
    assert DEFAULT_SAMPLING_INTERVAL_SECONDS > 0.16


def test_an_exception_in_the_measured_call_still_stops_the_sampler() -> None:
    source = RecordingSource([reported(1.0)])
    sampler = PowerSampler(source, interval_seconds=0.001)

    with pytest.raises(ZeroDivisionError):
        with sampler:
            assert source.polled.wait(timeout=10.0)
            raise ZeroDivisionError("the measured workload failed")

    assert sampler.is_running is False


# --------------------------------------------------------------------------
# The real telemetry seam (Implementation Note 5.4)
# --------------------------------------------------------------------------


class CyclingRunner:
    """Stands in for the subprocess call ONLY.

    Everything above it - the argument vector, ``parse_platform_report``, the
    three-state ``PowerReading`` - is the production wrapper, so this exercises
    the system as it is actually assembled rather than an in-memory analogue of
    it.
    """

    def __init__(self, outputs: Sequence[str], *, signal_after: int) -> None:
        self._outputs = list(outputs)
        self._signal_after = signal_after
        self._lock = threading.Lock()
        self.calls = 0
        self.ready = threading.Event()

    def __call__(
        self, argv: Sequence[str], *, timeout_seconds: float
    ) -> CommandResult:
        del argv, timeout_seconds
        with self._lock:
            text = self._outputs[self.calls % len(self._outputs)]
            self.calls += 1
            if self.calls >= self._signal_after:
                self.ready.set()
        return CommandResult(returncode=0, stdout=text, stderr="")


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    path = tmp_path / "xrt-smi.exe"
    path.write_bytes(b"not a real binary")
    return path


def test_the_sampler_drives_a_real_xrt_smi_wrapper(fake_binary: Path) -> None:
    """The three states, produced by the real parser from captured report text,
    carried through the real wrapper into a real trace."""
    runner = CyclingRunner(
        [
            fixture("platform_load"),
            fixture("platform_power_na"),
            fixture("platform_no_power_field"),
        ],
        signal_after=3,
    )
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)
    sampler = PowerSampler(wrapper, interval_seconds=0.001)

    with sampler:
        assert runner.ready.wait(timeout=10.0)
    trace = sampler.trace()

    assert [sample.reading.status for sample in trace.samples][:3] == [
        PowerStatus.REPORTED,
        PowerStatus.UNAVAILABLE,
        PowerStatus.UNSUPPORTED,
    ]
    watts = {
        sample.reading.watts
        for sample in trace.samples
        if sample.reading.watts is not None
    }
    assert watts == {LOAD_FIXTURE_WATTS}

    integral = integrate_power(trace)
    assert integral.reported_samples >= 1
    assert integral.missed_samples >= 1
    assert integral.unsupported_samples >= 1
    assert integral.joules > 0.0


def test_the_seam_produces_an_energy_figure_end_to_end(fake_binary: Path) -> None:
    """From captured ``xrt-smi`` text to a joules-per-thousand-inputs value,
    through nothing but production code."""
    runner = CyclingRunner([fixture("platform_load")], signal_after=2)
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)
    sampler = PowerSampler(wrapper, interval_seconds=0.001)

    with sampler:
        assert runner.ready.wait(timeout=10.0)

    measured = energy_per_thousand_inputs(
        sampler.trace(), input_count=1000, power_reporting_supported=True
    )

    assert measured.energy.value is not None
    assert measured.energy.value > 0.0
    assert measured.samples_missed == 0
    assert measured.energy.unavailable_reason is None


def test_a_wrapper_that_cannot_find_the_binary_yields_an_unavailable_run(
    tmp_path: Path,
) -> None:
    """A machine with no NPU is the case the whole taxonomy exists for, and it
    must reach an omission with a reason rather than an exception or a zero."""
    wrapper = XrtSmiWrapper(tmp_path / "absent" / "xrt-smi.exe")
    assert wrapper.available is False
    sampler = PowerSampler(wrapper, interval_seconds=0.001)

    with sampler:
        pass

    measured = energy_per_thousand_inputs(
        sampler.trace(), input_count=100, power_reporting_supported=False
    )

    assert measured.energy.value is None
    assert measured.energy.unavailable_reason is not None
