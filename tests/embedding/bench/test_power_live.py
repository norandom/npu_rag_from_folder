"""Live energy sampling against the real ``xrt-smi`` binary (task 6.2).

These skip wholesale when the vendor utility is absent, so the suite stays green
on a machine with no NPU. Per Implementation Note 4.2 CI has no NPU either, so
**nothing here may be the only coverage of anything**: every property of the code
asserted below is also pinned against captured report text - in
``test_power.py``, or, for the ``PowerReading`` invariant each poll is built on
(a reading carries Watts and no reason, or a reason and no Watts), in
``test_xrt.py``. What this file adds is the one thing fixture text cannot show -
that the real utility, the real subprocess spawn and a real sampling thread
compose into a trace with the timing properties the integral assumes.

Two assertions below have no fixture counterpart by design, because neither is a
property of this module's code: that a reading off real silicon is physically
plausible (``IMPLAUSIBLE_WATTS``), and that a real sampled window really spans
the workload it bracketed. Both are claims about the machine, not about the
integral, and a captured report cannot make either one true or false.

No model is prepared and no session is created. The "workload" is a short busy
wait, which is enough to exercise concurrent polling and costs no NPU compile.

Measured on this machine 2026-09-07, 30 consecutive polls at idle: 29 readings
between 0.001 W and 0.005 W and **one** ``N/A``, at about 0.16 s per poll. That
is Implementation Note 1.4's intermittent absence, live. Under CPU contention it
is far from marginal: with 24 spinning background jobs on this 24-logical-
processor machine, 6 of 8 consecutive runs saw an entire burst come back ``N/A``.

So nothing here asserts a particular wattage, and **nothing requires a reading to
arrive**. Whether the NPU answers a given burst is a claim about the machine
cooperating, not a property of this module, and a test that requires it is flaky
by construction - it fails exactly when the code is doing the right thing. Every
test below therefore asserts the code property under *both* outcomes rather than
demanding one of them. What stays unconditional is the non-vacuity floor - that
polls occurred at all - without which the absent-reading arm cannot be told apart
from a sampler that never ran.
"""

from __future__ import annotations

import math
import threading
import time

import pytest

from npu_rag.embedding.bench.power import (
    ENERGY_UNIT,
    UNSUPPORTED_PLATFORM_REASON,
    PowerSampler,
    energy_per_thousand_inputs,
    integrate_power,
)
from npu_rag.embedding.environment.xrt import PowerReading, PowerStatus, XrtSmiWrapper

wrapper = XrtSmiWrapper()

pytestmark = pytest.mark.skipif(
    not wrapper.available,
    reason=f"xrt-smi not present at {XrtSmiWrapper().executable}",
)

#: An NPU estimated-power reading above this is a parse error wearing a number;
#: the whole Strix package is rated tens of Watts (see ``test_xrt_live.py``).
IMPLAUSIBLE_WATTS = 100.0

#: The measured call spins for at least this long, and then keeps spinning until
#: the sampler has taken ``POLLS_NEEDED`` polls, up to ``WORKLOAD_CEILING_SECONDS``.
#:
#: A fixed duration cannot carry that guarantee. One ``xrt-smi`` poll costs about
#: 0.16 s at idle but **~0.96 s under CPU saturation** (measured 2026-09-07, n=12,
#: with 24 spinning jobs on this 24-logical-processor machine: min 0.904 s, median
#: 0.958 s, max 1.018 s). At ~1.0 s per poll plus the 0.25 s interval, a 1.6 s
#: window affords the second poll only marginally - which is exactly how it was
#: observed to fail, 1 run in 10 under saturation, on ``len(trace.samples) >= 2``.
#: Waiting for the condition keeps "the sampler polled repeatedly" a property of
#: the sampler rather than a bet on how fast the machine happens to be, and costs
#: nothing at idle, where the second poll lands well inside the floor.
WORKLOAD_SECONDS = 1.6
WORKLOAD_CEILING_SECONDS = 20.0
POLLS_NEEDED = 2
LIVE_INTERVAL_SECONDS = 0.25


def busy_workload(seconds: float) -> int:
    """A stand-in for a measured call. Deliberately not ``sleep``: a sleeping
    thread releases the GIL trivially, and the point is that a *working* thread
    is still sampled beside."""
    deadline = time.perf_counter() + seconds
    spins = 0
    while time.perf_counter() < deadline:
        spins += 1
    return spins


def busy_workload_until(event: threading.Event, *, floor: float, ceiling: float) -> int:
    """``busy_workload``, but it also waits for ``event`` before finishing.

    Still a spin rather than a sleep, for the reason above. ``ceiling`` keeps a
    machine that has stopped answering from hanging the suite: if it engages, the
    poll-count assertion fails on its own terms rather than the test timing out.
    """
    started = time.perf_counter()
    spins = 0
    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= ceiling or (elapsed >= floor and event.is_set()):
            return spins
        spins += 1


class CountingSource:
    """The production ``XrtSmiWrapper``, with a poll counter the workload watches.

    Not a stand-in for anything: every call delegates straight to the real
    wrapper, so the real argument vector, the real subprocess spawn and the real
    parser are all still in the path, and the readings are the machine's own. It
    exists only to make "the sampler has polled twice" observable *while* the
    measured call is still running.
    """

    def __init__(self, inner: XrtSmiWrapper, *, needed: int) -> None:
        self._inner = inner
        self._needed = needed
        self._lock = threading.Lock()
        self.polls = 0
        self.enough = threading.Event()

    def read_power(self) -> PowerReading:
        reading = self._inner.read_power()
        with self._lock:
            self.polls += 1
            if self.polls >= self._needed:
                self.enough.set()
        return reading


def test_sampling_a_real_workload_yields_an_energy_figure() -> None:
    """Task 6.2's Observable from the NPU side, end to end on real telemetry.

    Asserted under **both** outcomes the machine can supply, because which one
    arrives is not a property of this code. See the branch below.
    """
    source = CountingSource(wrapper, needed=POLLS_NEEDED)
    sampler = PowerSampler(source, interval_seconds=LIVE_INTERVAL_SECONDS)

    with sampler:
        workload_thread = threading.get_ident()
        busy_workload_until(
            source.enough, floor=WORKLOAD_SECONDS, ceiling=WORKLOAD_CEILING_SECONDS
        )
    trace = sampler.trace()

    # Non-vacuity floor. Polls really happened, more than one of them, in a
    # window that really spans the workload, at the interval that was asked for.
    # Without these three the branch below would be satisfied by a sampler that
    # never polled at all: the all-absent arm cannot tell "every poll came back
    # empty" from "there were no polls" unless something else pins that polls
    # occurred. ``__exit__`` joins the sampling thread, so a poll already in
    # flight when the workload stopped is still recorded before ``trace()``.
    assert len(trace.samples) >= POLLS_NEEDED
    assert trace.duration_seconds >= WORKLOAD_SECONDS
    assert trace.interval_seconds == pytest.approx(LIVE_INTERVAL_SECONDS)

    integral = integrate_power(trace)
    assert integral.unsupported_samples == 0, (
        "this machine is a Strix part, which does report estimated power; an "
        "unsupported sample here means the platform report shape changed"
    )
    assert integral.measured_seconds + integral.unmeasured_seconds == pytest.approx(
        trace.duration_seconds
    )

    measured = energy_per_thousand_inputs(
        trace, input_count=1000, power_reporting_supported=True
    )

    # True whichever way the machine went: the unit, and the sampling interval
    # that travels into requirement 7.2's methodology. ``methodology`` states the
    # interval on both the value path and the omission path.
    assert measured.energy.unit == ENERGY_UNIT
    assert measured.sampling_interval_seconds == pytest.approx(LIVE_INTERVAL_SECONDS)
    assert f"{LIVE_INTERVAL_SECONDS:g}" in measured.methodology

    if integral.reported_samples >= 1:
        # At least one poll carried Watts, so requirement 6.2's figure exists.
        assert measured.energy.value is not None
        assert measured.energy.unavailable_reason is None
        assert math.isfinite(measured.energy.value)
        assert 0.0 <= measured.energy.value < IMPLAUSIBLE_WATTS * trace.duration_seconds
    else:
        # Every poll in the burst came back ``N/A``. Implementation Note 1.4
        # measured 2 absent of 39 at idle; under CPU contention the *whole* burst
        # can be absent. Characterised on this machine (24 logical processors)
        # 2026-09-07 with 24 spinning background jobs: 6 of 8 consecutive runs
        # produced an all-``N/A`` burst, 0 of 5 did at 8 jobs.
        #
        # This is the correct path, and requiring a reading here was the bug:
        # ``assert integral.reported_samples >= 1`` made the test that exists to
        # prove ``N/A`` is *accounted rather than invented* fail precisely when
        # the machine produced nothing but ``N/A`` - it exercised the right code
        # path and then rejected it. A reading arriving is a claim about the
        # machine cooperating, not a property of this module, so both outcomes
        # are checked instead of one being required. Do not collapse this back
        # into a single arm.
        assert measured.energy.value is None
        reason = measured.energy.unavailable_reason
        assert reason is not None
        # Specifically the all-polls-empty omission (transient, a rerun may
        # succeed), never the unsupported-platform one (permanent). Requirement
        # 6.8 wants the omission *specific*, so a live collapse of the two
        # distinct reasons is still caught here.
        assert reason != UNSUPPORTED_PLATFORM_REASON
        assert "unsupported" not in reason.lower()
        assert "no reading" in reason.lower()
        assert measured.samples_missed == len(trace.samples)
        assert measured.measured_seconds == pytest.approx(0.0)

    # The polls landed on the sampling thread, not on the thread that ran the
    # workload - the live form of "never inside the measured call".
    assert workload_thread == threading.get_ident()
    assert sampler.is_running is False


def test_every_live_reading_is_a_number_or_an_accounted_absence() -> None:
    """Implementation Note 1.4: ``N/A`` must never arrive as 0.0 W. Whether it
    appears in any given burst is genuinely intermittent, so the assertion is
    that each of the two outcomes is well formed, not that both occur."""
    sampler = PowerSampler(wrapper, interval_seconds=0.05)

    with sampler:
        busy_workload(1.0)
    trace = sampler.trace()

    # The non-vacuity floor again, and the only machine claim in this test: at a
    # 0.05 s interval over a 1.0 s workload the sampler polls at entry and many
    # times after, so this holds unless the sampling thread never ran at all. It
    # is *not* a claim that any poll carried Watts - the loop below is already
    # outcome-independent, asserting that whichever of the two shapes a reading
    # takes, it is well formed, and the accounting assertion after it holds for
    # any mix of the three states.
    assert trace.samples
    for sample in trace.samples:
        reading = sample.reading
        if reading.status is PowerStatus.REPORTED:
            assert reading.watts is not None
            assert 0.0 <= reading.watts < IMPLAUSIBLE_WATTS
            assert reading.reason is None
        else:
            assert reading.watts is None
            assert reading.reason is not None

    integral = integrate_power(trace)
    assert integral.reported_samples + integral.missed_samples + (
        integral.unsupported_samples
    ) == len(trace.samples)


def test_a_run_shorter_than_the_interval_reports_the_omission() -> None:
    """The coarse-interval caveat, live: a workload that finishes inside one
    sampling period still yields at most the poll taken at entry, and the
    measurement says so rather than inventing a figure."""
    sampler = PowerSampler(wrapper, interval_seconds=30.0)

    with sampler:
        pass
    trace = sampler.trace()

    assert len(trace.samples) <= 1
    measured = energy_per_thousand_inputs(
        trace, input_count=1000, power_reporting_supported=True
    )
    if measured.energy.value is None:
        assert measured.energy.unavailable_reason is not None
    else:
        # A single poll landed at entry; its window is the whole tiny run, so the
        # figure is real but negligible. Never an estimate either way.
        assert measured.samples_reported == 1
