"""Unit tests for the environment capability check (task 1.5).

Every test here drives ``check_capability`` through **injected fakes**, so the
whole file passes on a machine with no NPU, no vendor runtime, and no
``xrt-smi``. That is task 1.5's Observable stated as a test suite: *the check
runs to completion on a machine with no NPU and no vendor runtime, reporting
each condition separately instead of raising*.

The live counterpart lives in ``test_capability_live.py`` and skips when the
execution provider is not registered in this interpreter.

Two of these tests exist because the corresponding mistakes were actually made
somewhere in this problem space and would be invisible without them:

- ``32.0.20102.3930 < 32.0.203.280`` is **True** as strings and **False**
  component-wise. The installed driver on the reference machine is the former
  and AMD's documented 1.8 minimum is the latter, so a lexicographic comparison
  reports a perfectly working driver as too old.
- Both documented vendor environment variables are **unset** on the reference
  machine and the provider registers and executes on the NPU anyway. A checker
  that treats them as hard requirements contradicts a working environment.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from npu_rag.embedding.environment.capability import (
    CONDITION_DRIVER_MINIMUM,
    CONDITION_ENVIRONMENT_VARIABLES,
    CONDITION_NPU_DEVICE,
    CONDITION_PROVIDER_REGISTERED,
    CONDITION_VENDOR_RUNTIME,
    DOCUMENTED_MINIMUM_DRIVER_VERSION,
    VENDOR_ENVIRONMENT_VARIABLES,
    VENDOR_PAYLOAD_FILES,
    VITISAI_PROVIDER,
    CapabilityReport,
    Condition,
    ExecutionMode,
    RuntimeProbe,
    check_capability,
    compare_versions,
    probe_onnx_runtime,
)
from npu_rag.embedding.environment.xrt import (
    DeviceIdentity,
    PlatformTelemetry,
    PowerReading,
    PowerStatus,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "npu_rag"
    / "embedding"
    / "environment"
    / "capability.py"
)

#: Measured on the reference machine, 2026-09-05 (research.md, first probe).
INSTALLED_DRIVER_VERSION = "32.0.20102.3930"

ALL_CONDITIONS = frozenset(
    {
        CONDITION_NPU_DEVICE,
        CONDITION_VENDOR_RUNTIME,
        CONDITION_DRIVER_MINIMUM,
        CONDITION_PROVIDER_REGISTERED,
        CONDITION_ENVIRONMENT_VARIABLES,
    }
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


@dataclass
class FakeTelemetry:
    """A stand-in for ``XrtSmiWrapper``, structurally typed.

    The real wrapper never raises; ``raises`` exists so the checker's own
    never-raise guarantee can be tested independently of that promise being
    kept.
    """

    identity: DeviceIdentity
    platform: PlatformTelemetry
    raises: bool = False

    def read_identity(self) -> DeviceIdentity:
        if self.raises:
            raise OSError("telemetry exploded")
        return self.identity

    def read_platform(self) -> PlatformTelemetry:
        if self.raises:
            raise OSError("telemetry exploded")
        return self.platform


def identity_present(
    driver_version: str | None = INSTALLED_DRIVER_VERSION,
) -> DeviceIdentity:
    return DeviceIdentity(
        bdf="00c6:00:01.1",
        device_name="NPU Strix",
        xrt_version="2.21.0",
        npu_driver_version=driver_version,
        npu_firmware_version="1.1.2.64",
        processor="AMD Ryzen AI 9 HX 370",
        unavailable_reason=None,
    )


def identity_absent() -> DeviceIdentity:
    return DeviceIdentity(
        bdf=None,
        device_name=None,
        xrt_version=None,
        npu_driver_version=None,
        npu_firmware_version=None,
        processor=None,
        unavailable_reason="xrt-smi was not found at C:\\Windows\\System32\\AMD",
    )


def platform_reporting(watts: float = 0.002) -> PlatformTelemetry:
    return PlatformTelemetry(
        power=PowerReading(watts=watts, status=PowerStatus.REPORTED, reason=None),
        power_mode="Default",
        total_columns=8,
        device_name="NPU Strix",
        bdf="00c6:00:01.1",
    )


def platform_power_na() -> PlatformTelemetry:
    """The measured intermittent ``N/A`` on a part that *does* report power."""
    return PlatformTelemetry(
        power=PowerReading(
            watts=None, status=PowerStatus.UNAVAILABLE, reason="reported N/A"
        ),
        power_mode="Default",
        total_columns=8,
        device_name="NPU Strix",
        bdf="00c6:00:01.1",
    )


def platform_power_unsupported() -> PlatformTelemetry:
    return PlatformTelemetry(
        power=PowerReading(
            watts=None, status=PowerStatus.UNSUPPORTED, reason="no power field"
        ),
        power_mode="Default",
        total_columns=4,
        device_name="NPU Phoenix",
        bdf="00c5:00:01.1",
    )


def platform_absent() -> PlatformTelemetry:
    return PlatformTelemetry(
        power=PowerReading(
            watts=None, status=PowerStatus.UNAVAILABLE, reason="xrt-smi not found"
        ),
        power_mode=None,
        total_columns=None,
        device_name=None,
        bdf=None,
    )


def probe_vendor() -> RuntimeProbe:
    """The reference machine after ``tools/provision_npu.py``."""
    return RuntimeProbe(
        version="1.23.2.dev20260117",
        available_providers=(
            VITISAI_PROVIDER,
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ),
        payload_present=VENDOR_PAYLOAD_FILES,
        payload_missing=(),
        unavailable_reason=None,
    )


def probe_stock() -> RuntimeProbe:
    """Stock ONNX Runtime: imports fine, carries no Vitis AI provider."""
    return RuntimeProbe(
        version="1.23.0",
        available_providers=("AzureExecutionProvider", "CPUExecutionProvider"),
        payload_present=(),
        payload_missing=VENDOR_PAYLOAD_FILES,
        unavailable_reason=None,
    )


def probe_absent() -> RuntimeProbe:
    return RuntimeProbe(
        version=None,
        available_providers=None,
        payload_present=(),
        payload_missing=VENDOR_PAYLOAD_FILES,
        unavailable_reason=(
            "ONNX Runtime could not be imported: "
            "ModuleNotFoundError(\"No module named 'onnxruntime'\")"
        ),
    )


def report_for(
    *,
    identity: DeviceIdentity,
    probe: Callable[[], RuntimeProbe],
    platform: PlatformTelemetry | None = None,
    environ: Mapping[str, str] | None = None,
    raises: bool = False,
) -> CapabilityReport:
    telemetry = FakeTelemetry(
        identity=identity,
        platform=platform if platform is not None else platform_reporting(),
        raises=raises,
    )
    return check_capability(
        telemetry=telemetry,
        runtime_probe=probe,
        environ={} if environ is None else environ,
    )


def bare_machine() -> CapabilityReport:
    """No NPU, no vendor runtime, no ``xrt-smi``, no vendor variables."""
    return report_for(
        identity=identity_absent(), probe=probe_absent, platform=platform_absent()
    )


# --------------------------------------------------------------------------
# Requirement 1.2 - each condition reported individually
# --------------------------------------------------------------------------


def test_reports_all_five_conditions_individually() -> None:
    report = bare_machine()

    assert {condition.name for condition in report.conditions} == ALL_CONDITIONS
    assert len(report.conditions) == len(ALL_CONDITIONS)


def test_conditions_are_not_collapsed_into_one_verdict() -> None:
    """Requirement 1.2. Hardware present, vendor runtime absent: the device
    condition must stay satisfied while the provider condition fails."""
    report = report_for(identity=identity_present(), probe=probe_stock)

    assert report.condition(CONDITION_NPU_DEVICE).satisfied is True
    assert report.condition(CONDITION_PROVIDER_REGISTERED).satisfied is False
    assert report.condition(CONDITION_VENDOR_RUNTIME).satisfied is False


# --------------------------------------------------------------------------
# Requirement 1.4 and the Observable - absence is data, never an exception
# --------------------------------------------------------------------------


def test_bare_machine_completes_without_raising() -> None:
    report = bare_machine()

    assert report.execution_mode is ExecutionMode.UNAVAILABLE
    assert report.condition(CONDITION_NPU_DEVICE).satisfied is False
    assert report.condition(CONDITION_VENDOR_RUNTIME).satisfied is False
    assert report.condition(CONDITION_PROVIDER_REGISTERED).satisfied is False
    assert report.driver_version is None
    assert report.runtime_version is None
    assert report.device_name is None
    assert report.power_reporting_supported is False


def test_runtime_probe_that_raises_still_yields_a_report() -> None:
    """The checker's never-raise guarantee cannot rest on its collaborators
    keeping theirs."""

    def exploding_probe() -> RuntimeProbe:
        raise RuntimeError("DLL load failed while importing onnxruntime_pybind11_state")

    report = report_for(identity=identity_absent(), probe=exploding_probe)

    assert {condition.name for condition in report.conditions} == ALL_CONDITIONS
    runtime = report.condition(CONDITION_VENDOR_RUNTIME)
    assert runtime.satisfied is False
    assert runtime.observed is not None
    assert "DLL load failed" in runtime.observed
    assert report.execution_mode is ExecutionMode.UNAVAILABLE


def test_telemetry_that_raises_still_yields_a_report() -> None:
    report = report_for(
        identity=identity_present(), probe=probe_vendor, raises=True
    )

    assert {condition.name for condition in report.conditions} == ALL_CONDITIONS
    device = report.condition(CONDITION_NPU_DEVICE)
    assert device.satisfied is False
    assert device.observed is not None
    assert "telemetry exploded" in device.observed
    assert report.power_reporting_supported is False


def test_the_real_default_check_runs_on_any_machine() -> None:
    """``check_capability()`` with no arguments - the design.md signature -
    against whatever this machine actually is."""
    report = check_capability()

    assert {condition.name for condition in report.conditions} == ALL_CONDITIONS
    assert isinstance(report.execution_mode, ExecutionMode)


@pytest.mark.parametrize(
    "scenario",
    [
        "bare",
        "device-only",
        "stock-runtime",
        "old-driver",
        "unknown-driver",
        "fully-provisioned",
    ],
)
def test_every_unsatisfied_condition_carries_observed_required_and_remediation(
    scenario: str,
) -> None:
    """Requirement 1.3, swept across the whole state space rather than asserted
    once. The dataclass enforces this too; this proves the enforcement is
    exercised by real reports rather than by construction alone."""
    reports = {
        "bare": lambda: bare_machine(),
        "device-only": lambda: report_for(
            identity=identity_present(), probe=probe_absent
        ),
        "stock-runtime": lambda: report_for(
            identity=identity_present(), probe=probe_stock
        ),
        "old-driver": lambda: report_for(
            identity=identity_present(driver_version="32.0.100.100"),
            probe=probe_stock,
        ),
        "unknown-driver": lambda: report_for(
            identity=identity_present(driver_version=None), probe=probe_stock
        ),
        "fully-provisioned": lambda: report_for(
            identity=identity_present(), probe=probe_vendor
        ),
    }
    report = reports[scenario]()

    for condition in report.conditions:
        if condition.satisfied:
            continue
        assert condition.observed, f"{condition.name} has no observed value"
        assert condition.required, f"{condition.name} has no required value"
        assert condition.remediation, f"{condition.name} has no remediation"


# --------------------------------------------------------------------------
# Requirement 1.3 - component-wise version comparison
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # The reference machine. Lexicographically "less", component-wise
        # greater. This single row is the whole point of the function.
        (INSTALLED_DRIVER_VERSION, DOCUMENTED_MINIMUM_DRIVER_VERSION, 1),
        (DOCUMENTED_MINIMUM_DRIVER_VERSION, INSTALLED_DRIVER_VERSION, -1),
        ("1.10.0", "1.9.0", 1),
        ("1.9.0", "1.10.0", -1),
        ("32.0.203.280", "32.0.203.280", 0),
        ("32.0.203.280", "32.0.203.281", -1),
        ("32.0.203.281", "32.0.203.280", 1),
        # Differing component counts compare by value, not by length.
        ("32.0.203", "32.0.203.0", 0),
        ("32.0.203.280", "32.0.203", 1),
        ("32.0.203", "32.0.203.280", -1),
        # Leading zeros are digits, not text.
        ("32.0.0203.280", "32.0.203.280", 0),
        # Unparsable on either side yields no ordering rather than a guess.
        ("32.0.203.280", "32.0.203.280-rc1", None),
        ("", "32.0.203.280", None),
        ("32.0.203.280", "", None),
    ],
)
def test_compare_versions_is_component_wise(
    left: str, right: str, expected: int | None
) -> None:
    assert compare_versions(left, right) == expected


def test_string_comparison_would_get_the_reference_machine_backwards() -> None:
    """Documents the trap this function exists to avoid: if the implementation
    ever degrades to comparing strings, the assertion below is what it would be
    computing."""
    assert INSTALLED_DRIVER_VERSION < DOCUMENTED_MINIMUM_DRIVER_VERSION
    assert compare_versions(
        INSTALLED_DRIVER_VERSION, DOCUMENTED_MINIMUM_DRIVER_VERSION
    ) == 1


def test_installed_driver_is_never_reported_as_too_old() -> None:
    """design.md, CapabilityChecker Risks: *the design must not assert the
    driver is too old*. Checked with the provider absent, so nothing else can
    be carrying the verdict."""
    report = report_for(identity=identity_present(), probe=probe_stock)

    driver = report.condition(CONDITION_DRIVER_MINIMUM)
    assert driver.satisfied is True
    assert driver.observed is not None
    assert INSTALLED_DRIVER_VERSION in driver.observed
    assert driver.required is not None
    assert DOCUMENTED_MINIMUM_DRIVER_VERSION in driver.required
    assert driver.remediation is None


def test_driver_below_minimum_reports_installed_required_and_remediation() -> None:
    """Requirement 1.3's literal text, for a driver that really is lower."""
    report = report_for(
        identity=identity_present(driver_version="32.0.100.100"), probe=probe_stock
    )

    driver = report.condition(CONDITION_DRIVER_MINIMUM)
    assert driver.satisfied is False
    assert driver.observed is not None and "32.0.100.100" in driver.observed
    assert driver.required is not None
    assert DOCUMENTED_MINIMUM_DRIVER_VERSION in driver.required
    assert driver.remediation


def test_registered_provider_settles_driver_adequacy() -> None:
    """The only reliable evidence. A driver below the documented minimum whose
    provider registers and runs is adequate in fact, whatever the numbers say."""
    report = report_for(
        identity=identity_present(driver_version="32.0.100.100"), probe=probe_vendor
    )

    assert report.condition(CONDITION_DRIVER_MINIMUM).satisfied is True


def test_incomparable_driver_version_draws_no_conclusion() -> None:
    """A version that is not purely numeric cannot be ordered, and guessing an
    order is exactly the failure this whole condition guards against."""
    report = report_for(
        identity=identity_present(driver_version="32.0.203.280-rc1"),
        probe=probe_stock,
    )

    driver = report.condition(CONDITION_DRIVER_MINIMUM)
    assert driver.satisfied is False
    assert driver.observed is not None
    assert "32.0.203.280-rc1" in driver.observed
    assert driver.remediation


def test_unknown_driver_version_is_reported_unknown_not_assumed_adequate() -> None:
    report = report_for(
        identity=identity_present(driver_version=None), probe=probe_stock
    )

    driver = report.condition(CONDITION_DRIVER_MINIMUM)
    assert driver.satisfied is False
    assert driver.observed is not None
    assert "unknown" in driver.observed.lower()
    assert driver.remediation


# --------------------------------------------------------------------------
# Requirement 1.5 - the execution-mode verdict
# --------------------------------------------------------------------------


def test_execution_mode_is_in_process_when_the_provider_registers() -> None:
    report = report_for(identity=identity_present(), probe=probe_vendor)

    assert report.execution_mode is ExecutionMode.IN_PROCESS
    assert report.condition(CONDITION_PROVIDER_REGISTERED).satisfied is True


def test_execution_mode_is_isolated_when_hardware_is_present_but_unreachable() -> None:
    report = report_for(identity=identity_present(), probe=probe_stock)

    assert report.execution_mode is ExecutionMode.ISOLATED


def test_execution_mode_is_unavailable_without_hardware() -> None:
    report = report_for(identity=identity_absent(), probe=probe_stock)

    assert report.execution_mode is ExecutionMode.UNAVAILABLE


def test_provider_registration_condition_names_the_provider_and_what_was_seen() -> None:
    report = report_for(identity=identity_present(), probe=probe_stock)

    provider = report.condition(CONDITION_PROVIDER_REGISTERED)
    assert provider.observed is not None
    assert "CPUExecutionProvider" in provider.observed
    assert provider.required is not None and VITISAI_PROVIDER in provider.required
    assert provider.remediation is not None
    assert "provision_npu" in provider.remediation


# --------------------------------------------------------------------------
# Requirement 1.1 / 1.4 - vendor runtime installation
# --------------------------------------------------------------------------


def test_vendor_runtime_condition_names_the_missing_payload() -> None:
    report = report_for(identity=identity_present(), probe=probe_stock)

    runtime = report.condition(CONDITION_VENDOR_RUNTIME)
    assert runtime.satisfied is False
    assert runtime.observed is not None
    for name in VENDOR_PAYLOAD_FILES:
        assert name in runtime.observed
    assert runtime.remediation is not None
    assert "provision_npu" in runtime.remediation


def test_vendor_runtime_satisfied_only_with_the_full_payload() -> None:
    """The stranded-DLL packaging bug (research.md, second probe) registers the
    provider and *then* dies in native code. A partial payload is not an
    installed runtime."""

    def probe_partial() -> RuntimeProbe:
        return RuntimeProbe(
            version="1.23.2.dev20260117",
            available_providers=(VITISAI_PROVIDER, "CPUExecutionProvider"),
            payload_present=("onnxruntime_vitisai_ep.dll",),
            payload_missing=("vaiml.dll", "vaip_config.json"),
            unavailable_reason=None,
        )

    report = report_for(identity=identity_present(), probe=probe_partial)

    assert report.condition(CONDITION_VENDOR_RUNTIME).satisfied is False
    # Registration is still its own condition, and it is still satisfied.
    assert report.condition(CONDITION_PROVIDER_REGISTERED).satisfied is True


def test_vendor_runtime_condition_satisfied_on_a_provisioned_machine() -> None:
    report = report_for(identity=identity_present(), probe=probe_vendor)

    runtime = report.condition(CONDITION_VENDOR_RUNTIME)
    assert runtime.satisfied is True
    assert runtime.remediation is None


# --------------------------------------------------------------------------
# The vendor environment variables - diagnostic, not gating
# --------------------------------------------------------------------------


def test_unset_variables_do_not_fail_a_working_environment() -> None:
    """Measured on the reference machine: both variables unset, provider
    registered, NPU executing. Reporting a failure here would contradict an
    environment that demonstrably works."""
    report = report_for(identity=identity_present(), probe=probe_vendor, environ={})

    variables = report.condition(CONDITION_ENVIRONMENT_VARIABLES)
    assert variables.satisfied is True
    assert variables.remediation is None
    assert [c.name for c in report.conditions if not c.satisfied] == []


def test_unset_variables_become_a_remediation_lead_when_the_provider_is_absent() -> None:
    report = report_for(identity=identity_present(), probe=probe_stock, environ={})

    variables = report.condition(CONDITION_ENVIRONMENT_VARIABLES)
    assert variables.satisfied is False
    assert variables.observed is not None
    assert variables.required is not None
    assert variables.remediation is not None
    for name in VENDOR_ENVIRONMENT_VARIABLES:
        assert name in variables.observed
        assert name in variables.required
    assert "provision_npu" in variables.remediation


def test_set_variables_satisfy_the_condition_even_without_a_provider() -> None:
    report = report_for(
        identity=identity_present(),
        probe=probe_stock,
        environ={name: "C:\\vendor" for name in VENDOR_ENVIRONMENT_VARIABLES},
    )

    assert report.condition(CONDITION_ENVIRONMENT_VARIABLES).satisfied is True


def test_a_blank_variable_is_not_a_set_variable() -> None:
    report = report_for(
        identity=identity_present(),
        probe=probe_stock,
        environ={name: "   " for name in VENDOR_ENVIRONMENT_VARIABLES},
    )

    assert report.condition(CONDITION_ENVIRONMENT_VARIABLES).satisfied is False


# --------------------------------------------------------------------------
# Report provenance fields
# --------------------------------------------------------------------------


def test_report_carries_versions_and_device_name() -> None:
    report = report_for(identity=identity_present(), probe=probe_vendor)

    assert report.driver_version == INSTALLED_DRIVER_VERSION
    assert report.runtime_version == "1.23.2.dev20260117"
    assert report.device_name == "NPU Strix"


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        (platform_reporting(), True),
        # An intermittent N/A on a part that produced a platform report is a
        # missing *sample*, not a missing capability.
        (platform_power_na(), True),
        (platform_power_unsupported(), False),
        (platform_absent(), False),
    ],
)
def test_power_reporting_support_distinguishes_sample_from_capability(
    platform: PlatformTelemetry, expected: bool
) -> None:
    report = report_for(
        identity=identity_present(), probe=probe_vendor, platform=platform
    )

    assert report.power_reporting_supported is expected


# --------------------------------------------------------------------------
# Construction invariants - programming errors, and only those, raise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("omitted", ["observed", "required", "remediation"])
def test_an_unsatisfied_condition_must_carry_all_three_fields(omitted: str) -> None:
    fields: dict[str, str | None] = {
        "observed": "seen",
        "required": "wanted",
        "remediation": "do this",
    }
    fields[omitted] = None

    with pytest.raises(ValueError):
        Condition(name="whatever", satisfied=False, **fields)


def test_a_satisfied_condition_may_omit_them() -> None:
    condition = Condition(
        name="whatever", satisfied=True, observed=None, required=None, remediation=None
    )

    assert condition.satisfied is True


def test_in_process_requires_the_registration_condition_to_be_satisfied() -> None:
    """design.md, Data Models: ``execution_mode`` is ``IN_PROCESS`` only when EP
    registration succeeded in this interpreter."""
    unregistered = Condition(
        name=CONDITION_PROVIDER_REGISTERED,
        satisfied=False,
        observed="CPUExecutionProvider",
        required=VITISAI_PROVIDER,
        remediation="provision",
    )

    with pytest.raises(ValueError):
        CapabilityReport(
            conditions=(unregistered,),
            execution_mode=ExecutionMode.IN_PROCESS,
            driver_version=None,
            runtime_version=None,
            device_name=None,
            power_reporting_supported=False,
        )


def test_duplicate_condition_names_are_a_programming_error() -> None:
    condition = Condition(
        name=CONDITION_NPU_DEVICE,
        satisfied=True,
        observed=None,
        required=None,
        remediation=None,
    )

    with pytest.raises(ValueError):
        CapabilityReport(
            conditions=(condition, condition),
            execution_mode=ExecutionMode.UNAVAILABLE,
            driver_version=None,
            runtime_version=None,
            device_name=None,
            power_reporting_supported=False,
        )


def test_condition_lookup_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError):
        bare_machine().condition("no such condition")


# --------------------------------------------------------------------------
# The default runtime probe
# --------------------------------------------------------------------------


def test_probe_onnx_runtime_never_raises_and_reports_consistently() -> None:
    """Runs against whatever ONNX Runtime this interpreter has, including none.

    The invariant asserted is the one the dataclass declares: a probe either
    carries a version and a provider list, or carries a reason it does not.
    """
    probe = probe_onnx_runtime()

    if probe.unavailable_reason is None:
        assert probe.version is not None
        assert probe.available_providers is not None
    else:
        assert probe.version is None or probe.available_providers is None
    assert set(probe.payload_present) | set(probe.payload_missing) == set(
        VENDOR_PAYLOAD_FILES
    )


# --------------------------------------------------------------------------
# Boundary guards
# --------------------------------------------------------------------------

#: design.md, Architecture: "types, errors -> reporting -> profiles ->
#: environment -> models -> providers -> service -> bench". Duplicated from
#: ``test_xrt.py`` rather than shared, because a cross-test-module import would
#: bind these two files together for no benefit.
MODULE_PACKAGE = "npu_rag.embedding.environment"
LAYERS_RIGHT_OF_ENVIRONMENT = ("models", "providers", "service", "bench")


def absolute_imports_of(source: str, package: str) -> list[str]:
    """Every name ``source`` imports, as an absolute dotted path."""
    parts = package.split(".")
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                root = node.module or ""
            else:
                base = ".".join(parts[: len(parts) - node.level + 1])
                root = f"{base}.{node.module}" if node.module else base
            names.append(root)
            names.extend(f"{root}.{alias.name}" for alias in node.names)
    return names


def test_module_imports_nothing_from_a_later_layer() -> None:
    imported = absolute_imports_of(
        MODULE_PATH.read_text(encoding="utf-8"), MODULE_PACKAGE
    )

    forbidden = sorted(
        {
            name
            for name in imported
            for layer in LAYERS_RIGHT_OF_ENVIRONMENT
            if name == f"npu_rag.embedding.{layer}"
            or name.startswith(f"npu_rag.embedding.{layer}.")
        }
    )
    assert forbidden == []


def test_every_remediation_names_the_supported_repair() -> None:
    """This project has exactly one supported way to repair the vendor runtime,
    and every remediation that concerns it must point there. The complementary
    guard - that no owned file names an environment manager this project does
    not use - is repository-wide and lives in
    ``tests/tools/test_provision_npu.py``."""
    report = bare_machine()

    remediations = [c.remediation for c in report.conditions if not c.satisfied]
    assert remediations
    assert all(text for text in remediations)
    runtime_remediations = [
        text
        for name, text in ((c.name, c.remediation) for c in report.conditions)
        if text is not None and name != CONDITION_NPU_DEVICE
    ]
    assert runtime_remediations
    assert all(
        "uv run python -m tools.provision_npu" in text
        for text in runtime_remediations
    )
