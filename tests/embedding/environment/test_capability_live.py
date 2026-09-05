"""The capability check against this machine's real environment (task 1.5).

Task 1.3 settled the execution-mode verdict on this hardware empirically:
``IN_PROCESS``, confirmed by provider registration, genuine BF16 AIE kernels, a
5.24x throughput ratio over CPU, and a live hardware context across all eight
accelerator columns. This file asserts that the checker *formalizes* that
verdict rather than merely being capable of producing it - the unit tests can
only show that a fake environment maps to the right answer.

Everything here skips wholesale when the execution provider is not registered in
this interpreter, which is the state of any machine without the vendor runtime
and also the state this repository's own environment falls into after a bare
``uv sync`` (which drops the ``npu`` dependency group; repair with
``uv run python -m tools.provision_npu``).
"""

from __future__ import annotations

import pytest

from npu_rag.embedding.environment.capability import (
    CONDITION_DRIVER_MINIMUM,
    CONDITION_NPU_DEVICE,
    CONDITION_PROVIDER_REGISTERED,
    CONDITION_VENDOR_RUNTIME,
    DOCUMENTED_MINIMUM_DRIVER_VERSION,
    VITISAI_PROVIDER,
    ExecutionMode,
    check_capability,
    probe_onnx_runtime,
)
from npu_rag.embedding.environment.xrt import XrtSmiWrapper

_probe = probe_onnx_runtime()
_providers = _probe.available_providers or ()

provider_registered = pytest.mark.skipif(
    VITISAI_PROVIDER not in _providers,
    reason=(
        f"{VITISAI_PROVIDER} is not registered in this interpreter "
        f"(available: {list(_providers)})"
    ),
)

device_present = pytest.mark.skipif(
    not XrtSmiWrapper().available,
    reason=f"xrt-smi not present at {XrtSmiWrapper().executable}",
)


@provider_registered
@device_present
def test_this_machine_reports_in_process_execution() -> None:
    """Task 1.5's derivation of task 1.3's verdict."""
    report = check_capability()

    assert report.execution_mode is ExecutionMode.IN_PROCESS
    assert report.condition(CONDITION_PROVIDER_REGISTERED).satisfied is True
    assert report.condition(CONDITION_NPU_DEVICE).satisfied is True
    assert report.device_name is not None
    assert report.driver_version is not None
    assert report.runtime_version is not None


@provider_registered
@device_present
def test_this_machines_driver_is_not_reported_as_too_old() -> None:
    """The installed driver reads lower than the documented minimum as a
    *string* and higher component-wise. On this machine the provider registers
    and the NPU executes, so any report of an inadequate driver here would be
    false."""
    report = check_capability()

    driver = report.condition(CONDITION_DRIVER_MINIMUM)
    assert driver.satisfied is True
    assert driver.observed is not None
    assert report.driver_version is not None
    assert report.driver_version in driver.observed
    assert driver.required is not None
    assert DOCUMENTED_MINIMUM_DRIVER_VERSION in driver.required


@provider_registered
def test_a_provisioned_environment_reports_no_unsatisfied_condition() -> None:
    report = check_capability()

    unsatisfied = [c.name for c in report.conditions if not c.satisfied]
    assert unsatisfied == [], (
        "a fully provisioned machine reported unsatisfied conditions: "
        f"{[(c.name, c.observed) for c in report.conditions if not c.satisfied]}"
    )
    assert report.condition(CONDITION_VENDOR_RUNTIME).satisfied is True


@device_present
def test_power_reporting_is_detected_on_this_strix_part() -> None:
    """Strix reports estimated power; PHX/HPT and Linux do not. Requirement 6.2
    depends on this being read from the platform report rather than assumed."""
    assert check_capability().power_reporting_supported is True
