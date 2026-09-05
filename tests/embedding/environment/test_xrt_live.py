"""Live telemetry reads against the real ``xrt-smi`` binary (task 1.4).

These are task 1.4's Observable: *polling returns a plausible watt value on this
machine*. They skip wholesale when the vendor utility is absent, so the suite
stays green on a machine with no NPU - which is exactly the condition the
unit tests in ``test_xrt.py`` cover from fixture text instead.

Nothing here asserts a specific wattage. Estimated power on an idle Strix part
sits around 0.001-0.002 W and rises to roughly 0.44 W under a sustained NPU
workload (research.md, "Third probe"); the assertion is that the reading is a
finite, non-negative number well below the part's package power, not that it
equals any particular value.
"""

from __future__ import annotations

import math

import pytest

from npu_rag.embedding.environment.xrt import PowerStatus, XrtSmiWrapper

wrapper = XrtSmiWrapper()

pytestmark = pytest.mark.skipif(
    not wrapper.available,
    reason=f"xrt-smi not present at {XrtSmiWrapper().executable}",
)

#: An NPU estimated-power reading above this is not a reading, it is a parse
#: error wearing a number. The whole Strix package is rated tens of Watts and
#: the NPU is a fraction of it.
IMPLAUSIBLE_WATTS = 100.0


def test_device_identity_reports_versions() -> None:
    identity = wrapper.read_identity()

    assert identity.unavailable_reason is None
    assert identity.xrt_version is not None
    assert identity.npu_driver_version is not None
    assert identity.npu_firmware_version is not None
    assert identity.device_name is not None
    assert identity.bdf is not None


def test_platform_report_reads_mode_and_columns() -> None:
    telemetry = wrapper.read_platform()

    assert telemetry.total_columns is not None
    assert telemetry.total_columns >= 1
    assert telemetry.power_mode is not None
    assert telemetry.device_name is not None


def test_polling_returns_a_plausible_watt_value() -> None:
    """Task 1.4's Observable.

    ``N/A`` is tolerated per sample - it occurs intermittently on this machine
    even though power reporting is supported here - but it must never arrive as
    a number, and at least one poll in a short burst must yield a real value.
    """
    samples = [wrapper.read_power() for _ in range(12)]

    for sample in samples:
        if sample.status is PowerStatus.REPORTED:
            assert sample.watts is not None
            assert math.isfinite(sample.watts)
            assert 0.0 <= sample.watts < IMPLAUSIBLE_WATTS
            assert sample.reason is None
        else:
            assert sample.watts is None
            assert sample.reason is not None

    reported = [s.watts for s in samples if s.status is PowerStatus.REPORTED]
    assert reported, (
        "no poll in a burst of 12 returned a watt value; "
        f"statuses were {[s.status for s in samples]}"
    )


def test_partition_report_answers_liveness_without_raising() -> None:
    occupancy = wrapper.read_partitions()

    assert occupancy.unavailable_reason is None
    assert occupancy.context_live in (True, False)
    if occupancy.context_live:
        assert occupancy.occupied_columns
    else:
        assert occupancy.partitions == ()
