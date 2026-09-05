"""Report what this machine and this interpreter can actually do (task 1.5).

This is design.md's ``CapabilityChecker``. It answers five questions, each on
its own terms, and then draws one verdict from them:

1. **Is there an NPU?** Read from ``xrt-smi`` through ``XrtSmiWrapper``.
2. **Is the vendor runtime installed here?** An ONNX Runtime that imports, plus
   the native payload the Vitis AI provider needs beside it.
3. **Does the driver meet the documented minimum?** See the trap below.
4. **Is the execution provider registered?** ``get_available_providers()``.
5. **Are the documented vendor environment variables set?** Informational - see
   below - not a gate.

Requirement 1.2 forbids collapsing these into a single verdict, so each is a
separate ``Condition`` with its own pass/fail state, and requirement 1.3 makes
every unsatisfied one carry what was observed, what was required, and what to
do about it. That last rule is enforced by the dataclass rather than by
convention: constructing an unsatisfied ``Condition`` without all three raises.

**Absence is data, not an exception** (requirement 1.4, and task 1.5's
Observable). On a machine with no NPU, no vendor runtime, and no ``xrt-smi``
this module produces a full five-condition report. It guards its collaborators
even where they promise not to raise, because the guarantee it makes is
stronger than theirs: only a programming error - a self-contradictory
``Condition`` or ``CapabilityReport`` - raises from here.

**The driver-version trap.** Measured on this machine: XRT self-reports
``32.0.20102.3930``; AMD's 1.8 documentation states a minimum of
``32.0.203.280``. As *strings* the installed value sorts lower, so a
lexicographic comparison declares a driver that demonstrably works to be too
old. Component-wise - the third field is ``20102`` against ``203`` - the
installed value is the larger, and the two are plausibly different numbering
branches rather than an old-versus-new pair. This module therefore compares
component-wise, reports both values side by side, and **defers the adequacy
verdict to whether the provider actually registers**, which is the only
reliable evidence (design.md, CapabilityChecker Risks).

**The vendor environment variables are diagnostic, not required.** AMD
documents ``RYZEN_AI_INSTALLATION_PATH`` and ``XLNX_VART_FIRMWARE`` as locating
the runtime's native assets, and RyzenAI-SW issue #213 attributes
non-registration to missing assets. But on this machine both are **unset** and
the provider registers, is selected, and executes on the NPU - because
``tools/provision_npu.py`` places those assets beside the runtime instead. So
an unset variable is reported as a lead when the provider is missing, and never
as a failure when the provider is present: a check that contradicts a working
environment is worse than no check.

This module sits in the ``environment`` layer of design.md's dependency
direction (``types, errors -> reporting -> profiles -> environment -> models ->
providers -> service -> bench``). It takes ``ExecutionMode``, ``Condition`` and
``CapabilityReport`` from ``types`` - the leftmost layer, which owns them per
the File Structure Plan - and re-exports them here, so importers written against
this module keep working. Within its own layer it uses ``xrt``; it imports
nothing from any layer to its right.
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from npu_rag.embedding.environment.xrt import (
    DeviceIdentity,
    PlatformTelemetry,
    PowerStatus,
    XrtSmiWrapper,
)
from npu_rag.embedding.types import (
    CONDITION_PROVIDER_REGISTERED,
    CapabilityReport,
    Condition,
    ExecutionMode,
)

__all__ = [
    "CONDITION_DRIVER_MINIMUM",
    "CONDITION_ENVIRONMENT_VARIABLES",
    "CONDITION_NPU_DEVICE",
    "CONDITION_PROVIDER_REGISTERED",
    "CONDITION_VENDOR_RUNTIME",
    "DOCUMENTED_MINIMUM_DRIVER_VERSION",
    "VENDOR_ENVIRONMENT_VARIABLES",
    "VENDOR_PAYLOAD_FILES",
    "VITISAI_PROVIDER",
    "CapabilityReport",
    "Condition",
    "ExecutionMode",
    "RuntimeProbe",
    "TelemetrySource",
    "check_capability",
    "compare_versions",
    "probe_onnx_runtime",
]

#: The execution provider whose presence in this interpreter *is* the
#: in-process verdict. Requesting an absent provider from ONNX Runtime succeeds
#: and silently runs on the CPU with only a ``UserWarning``, so the available
#: -provider list is the reliable pre-check (research.md, first probe). This is
#: guard one of design.md's two; guard two - ``session.get_providers()`` after
#: construction - belongs to the backend, not here.
VITISAI_PROVIDER = "VitisAIExecutionProvider"

#: What AMD's Ryzen AI 1.8 documentation states. Reported as context beside the
#: installed version, never applied as a lexicographic threshold - see the
#: module docstring.
DOCUMENTED_MINIMUM_DRIVER_VERSION = "32.0.203.280"

#: Documented by AMD as locating the vendor runtime's native assets, and unset
#: on this machine while the provider works. Diagnostic, never gating.
VENDOR_ENVIRONMENT_VARIABLES = ("RYZEN_AI_INSTALLATION_PATH", "XLNX_VART_FIRMWARE")

#: The native payload that must sit beside ONNX Runtime for the Vitis AI
#: provider to be genuinely installed rather than merely present in name.
#: ``onnxruntime_vitisai_ep.dll`` is the provider bridge, stranded by the
#: ``voe`` wheel's mismatched data-directory version and relocated by
#: ``tools/provision_npu.py``; ``vaiml.dll`` is the BF16 compiler the NLP
#: encoder flow needs on Strix; ``vaip_config.json`` is the ``config_file``
#: provider option that switches the device data type to bfloat16. All three
#: come from outside PyPI, and a build missing any of them registers the
#: provider and then fails in native code (research.md, second probe).
VENDOR_PAYLOAD_FILES = (
    "onnxruntime_vitisai_ep.dll",
    "vaiml.dll",
    "vaip_config.json",
)

#: The condition names this checker produces. ``CONDITION_PROVIDER_REGISTERED``
#: is not among them: it is imported from ``types`` above, because
#: ``CapabilityReport``'s invariant keys on it and the two must not be able to
#: drift apart.
CONDITION_NPU_DEVICE = "npu_device_present"
CONDITION_VENDOR_RUNTIME = "vendor_runtime_installed"
CONDITION_DRIVER_MINIMUM = "driver_meets_documented_minimum"
CONDITION_ENVIRONMENT_VARIABLES = "vendor_environment_variables"

#: The supported repair for every environment fault this check can report. It
#: is a uv/venv-only project: nothing here ever suggests another environment
#: manager, because none is installed and AMD's default install path is not the
#: path this project uses.
_PROVISION_COMMAND = "uv run python -m tools.provision_npu"
_PROVISION_HINT = (
    f"Run `{_PROVISION_COMMAND}` (see docs/provisioning.md). Do not run a bare "
    "`uv sync` first: it drops the `npu` dependency group and leaves the "
    "vendor runtime unimportable."
)

_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class RuntimeProbe:
    """What ONNX Runtime reports about itself in *this* interpreter.

    Separated from the conditions it feeds so that every environment shape -
    absent, hollow, stock, partially provisioned, fully provisioned - is
    reachable in a test without the vendor wheels.

    Invariants: ``payload_present`` and ``payload_missing`` partition
    ``VENDOR_PAYLOAD_FILES``, and ``unavailable_reason`` is set exactly when the
    runtime could not be inspected - that is, when either the version or the
    provider list is missing.
    """

    version: str | None
    #: ``None`` means the list could not be obtained at all, which is a
    #: different fact from an empty or CPU-only list.
    available_providers: tuple[str, ...] | None
    payload_present: tuple[str, ...]
    payload_missing: tuple[str, ...]
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        covered = set(self.payload_present) | set(self.payload_missing)
        if covered != set(VENDOR_PAYLOAD_FILES):
            raise ValueError(
                "payload_present and payload_missing must partition "
                f"{VENDOR_PAYLOAD_FILES}, got present={self.payload_present}, "
                f"missing={self.payload_missing}"
            )
        inspected = self.version is not None and self.available_providers is not None
        if inspected is (self.unavailable_reason is not None):
            raise ValueError(
                "unavailable_reason must be set exactly when ONNX Runtime could "
                f"not be inspected, got version={self.version!r}, providers="
                f"{self.available_providers!r}, reason={self.unavailable_reason!r}"
            )

    @property
    def registers(self) -> bool:
        """Whether the Vitis AI provider is registered here (guard one)."""
        return VITISAI_PROVIDER in (self.available_providers or ())


class TelemetrySource(Protocol):
    """The slice of ``XrtSmiWrapper`` this module needs.

    A protocol rather than the concrete class so a test can supply a machine
    this one is not - including one whose telemetry raises, which the real
    wrapper never does.
    """

    def read_identity(self) -> DeviceIdentity: ...

    def read_platform(self) -> PlatformTelemetry: ...


def compare_versions(left: str, right: str) -> int | None:
    """Order two dotted numeric versions **component-wise**.

    Returns ``-1``, ``0`` or ``1``, or ``None`` when either side is not a purely
    numeric dotted version and therefore cannot be ordered at all. Missing
    trailing components count as zero, so ``32.0.203`` equals ``32.0.203.0``.

    Never a string comparison. ``"32.0.20102.3930" < "32.0.203.280"`` is
    ``True`` as text and false in fact, and that one inversion would report this
    machine's working driver as below AMD's documented minimum.
    """
    first = _components(left)
    second = _components(right)
    if first is None or second is None:
        return None
    width = max(len(first), len(second))
    padded_first = first + (0,) * (width - len(first))
    padded_second = second + (0,) * (width - len(second))
    return (padded_first > padded_second) - (padded_first < padded_second)


def _components(version: str) -> tuple[int, ...] | None:
    parts = version.strip().split(".")
    if not parts or any(_DIGITS.fullmatch(part.strip()) is None for part in parts):
        return None
    return tuple(int(part) for part in parts)


def probe_onnx_runtime() -> RuntimeProbe:
    """Inspect ONNX Runtime in this interpreter without ever raising.

    Imported by name through ``importlib`` rather than with a module-level
    ``import onnxruntime``, because this module must be importable - and this
    function callable - on a machine that has no ONNX Runtime at all. The
    failure modes seen in practice are all covered here: no distribution, a
    native load failure inside the extension module, and the hollow package
    left behind when ``uv sync`` drops the vendor group (it imports, and has no
    ``__version__``).
    """
    try:
        module = importlib.import_module("onnxruntime")
    except Exception as exc:  # noqa: BLE001 - absence and native load failures alike
        return _uninspectable_runtime(f"ONNX Runtime could not be imported: {exc!r}")

    version = getattr(module, "__version__", None)
    module_file = getattr(module, "__file__", None)
    present, missing = _payload_state(
        Path(str(module_file)).parent / "capi" if module_file else None
    )

    if not isinstance(version, str) or not version:
        return _uninspectable_runtime(
            "the imported onnxruntime package reports no __version__, so it is "
            "not a usable runtime. This is the residue left when the `npu` "
            "dependency group is dropped.",
            present=present,
            missing=missing,
        )

    lister = getattr(module, "get_available_providers", None)
    if not callable(lister):
        return _uninspectable_runtime(
            f"onnxruntime {version} exposes no get_available_providers(), so "
            "provider registration cannot be established.",
            present=present,
            missing=missing,
        )
    try:
        providers = tuple(str(provider) for provider in lister())
    except Exception as exc:  # noqa: BLE001 - a broken native runtime is data
        return _uninspectable_runtime(
            f"onnxruntime {version}.get_available_providers() failed: {exc!r}",
            present=present,
            missing=missing,
        )

    return RuntimeProbe(
        version=version,
        available_providers=providers,
        payload_present=present,
        payload_missing=missing,
        unavailable_reason=None,
    )


def _uninspectable_runtime(
    reason: str,
    *,
    present: tuple[str, ...] = (),
    missing: tuple[str, ...] = VENDOR_PAYLOAD_FILES,
) -> RuntimeProbe:
    return RuntimeProbe(
        version=None,
        available_providers=None,
        payload_present=present,
        payload_missing=missing,
        unavailable_reason=reason,
    )


def _payload_state(capi: Path | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Which vendor payload files are beside the runtime, and which are not."""
    if capi is None:
        return (), VENDOR_PAYLOAD_FILES
    present: list[str] = []
    missing: list[str] = []
    for name in VENDOR_PAYLOAD_FILES:
        try:
            found = (capi / name).is_file()
        except OSError:  # pragma: no cover - unreadable path is still just data
            found = False
        (present if found else missing).append(name)
    return tuple(present), tuple(missing)


def check_capability(
    *,
    telemetry: TelemetrySource | None = None,
    runtime_probe: Callable[[], RuntimeProbe] = probe_onnx_runtime,
    environ: Mapping[str, str] | None = None,
) -> CapabilityReport:
    """Evaluate every environment condition and derive the execution mode.

    Read-only and non-raising: it detects and reports, never installs, and it
    completes on a machine with no NPU, no vendor runtime and no ``xrt-smi``.
    All three collaborators are injectable so that every environment shape is
    reachable from a test on hardware that is only ever one of them.
    """
    source: TelemetrySource = XrtSmiWrapper() if telemetry is None else telemetry
    variables = os.environ if environ is None else environ

    identity, identity_failure = _read_identity(source)
    platform = _read_platform(source)
    probe = _run_probe(runtime_probe)

    device_present = identity.device_name is not None or identity.bdf is not None
    registered = probe.registers

    conditions = (
        _device_condition(identity, identity_failure, present=device_present),
        _runtime_condition(probe),
        _driver_condition(identity, probe, registered=registered),
        _provider_condition(probe, registered=registered),
        _variables_condition(variables, registered=registered),
    )
    return CapabilityReport(
        conditions=conditions,
        execution_mode=_execution_mode(registered=registered, device=device_present),
        driver_version=identity.npu_driver_version,
        runtime_version=probe.version,
        device_name=identity.device_name,
        power_reporting_supported=_power_reporting_supported(platform),
    )


def _execution_mode(*, registered: bool, device: bool) -> ExecutionMode:
    """Requirement 1.5, decided by evidence rather than by expectation.

    Registration in *this* interpreter is what makes NPU execution reachable
    in-process; task 1.3 confirmed that verdict on this machine end to end. With
    hardware present but the provider absent here, a separate environment could
    still reach it, which is what the isolated mode exists for. With no hardware
    at all, neither route exists.
    """
    if registered:
        return ExecutionMode.IN_PROCESS
    return ExecutionMode.ISOLATED if device else ExecutionMode.UNAVAILABLE


def _read_identity(source: TelemetrySource) -> tuple[DeviceIdentity, str | None]:
    """Device identity, plus the reason the read itself failed, if it did."""
    try:
        return source.read_identity(), None
    except Exception as exc:  # noqa: BLE001 - a broken telemetry source is data
        reason = f"reading NPU telemetry failed: {exc!r}"
        return (
            DeviceIdentity(
                bdf=None,
                device_name=None,
                xrt_version=None,
                npu_driver_version=None,
                npu_firmware_version=None,
                processor=None,
                unavailable_reason=reason,
            ),
            reason,
        )


def _read_platform(source: TelemetrySource) -> PlatformTelemetry | None:
    try:
        return source.read_platform()
    except Exception:  # noqa: BLE001 - platform telemetry is optional context
        return None


def _run_probe(runtime_probe: Callable[[], RuntimeProbe]) -> RuntimeProbe:
    try:
        return runtime_probe()
    except Exception as exc:  # noqa: BLE001 - an unimportable runtime is data
        return _uninspectable_runtime(f"inspecting ONNX Runtime failed: {exc!r}")


def _power_reporting_supported(platform: PlatformTelemetry | None) -> bool:
    if platform is None:
        return False
    status = platform.power.status
    if status is PowerStatus.REPORTED:
        return True
    if status is PowerStatus.UNSUPPORTED:
        return False
    # UNAVAILABLE is ambiguous by itself: it covers both "no report at all" and
    # the intermittent ``N/A`` measured on this Strix part, where power
    # reporting *is* supported. A platform report that carried its other fields
    # distinguishes the two.
    return platform.power_mode is not None or platform.total_columns is not None


def _device_condition(
    identity: DeviceIdentity, failure: str | None, *, present: bool
) -> Condition:
    if present:
        described = " at ".join(
            part for part in (identity.device_name, _bracketed(identity.bdf)) if part
        )
        return Condition(
            name=CONDITION_NPU_DEVICE,
            satisfied=True,
            observed=described,
            required="an AMD XDNA NPU enumerated by xrt-smi",
            remediation=None,
        )
    reason = failure or identity.unavailable_reason or "xrt-smi reported no device"
    return Condition(
        name=CONDITION_NPU_DEVICE,
        satisfied=False,
        observed=f"no NPU was enumerated: {reason}",
        required="an AMD XDNA NPU enumerated by xrt-smi",
        remediation=(
            "Install the AMD NPU driver package, which also provides xrt-smi at "
            "C:\\Windows\\System32\\AMD without any SDK. On a machine whose "
            "processor has no XDNA NPU there is nothing to enable and NPU "
            "execution is unavailable by hardware."
        ),
    )


def _bracketed(bdf: str | None) -> str | None:
    return f"[{bdf}]" if bdf else None


def _runtime_condition(probe: RuntimeProbe) -> Condition:
    required = (
        "an ONNX Runtime build carrying the Vitis AI execution provider, with "
        f"{', '.join(VENDOR_PAYLOAD_FILES)} beside it in onnxruntime/capi"
    )
    if probe.unavailable_reason is not None:
        return Condition(
            name=CONDITION_VENDOR_RUNTIME,
            satisfied=False,
            observed=probe.unavailable_reason,
            required=required,
            remediation=_PROVISION_HINT,
        )
    if probe.payload_missing:
        return Condition(
            name=CONDITION_VENDOR_RUNTIME,
            satisfied=False,
            observed=(
                f"ONNX Runtime {probe.version} is installed, but the vendor "
                f"payload is incomplete: {', '.join(probe.payload_missing)} "
                f"absent, {', '.join(probe.payload_present) or 'nothing'} "
                "present. A build missing any of these can still register the "
                "provider and then fail inside native code."
            ),
            required=required,
            remediation=_PROVISION_HINT,
        )
    return Condition(
        name=CONDITION_VENDOR_RUNTIME,
        satisfied=True,
        observed=(
            f"ONNX Runtime {probe.version} with the full vendor payload: "
            f"{', '.join(probe.payload_present)}"
        ),
        required=required,
        remediation=None,
    )


def _driver_condition(
    identity: DeviceIdentity, probe: RuntimeProbe, *, registered: bool
) -> Condition:
    installed = identity.npu_driver_version
    required = (
        f"AMD's Ryzen AI 1.8 documentation states a minimum of "
        f"{DOCUMENTED_MINIMUM_DRIVER_VERSION}. It is reported here as context "
        "rather than applied as a threshold: the two schemes are not directly "
        "comparable - the third component differs in width - and provider "
        "registration is the only reliable evidence of driver adequacy."
    )
    if registered:
        # Positive evidence outranks version arithmetic. The provider registered
        # in this interpreter, so whatever the numbering branches mean, this
        # driver carries the runtime.
        observed = (
            f"NPU driver {installed}" if installed else "NPU driver version unknown"
        )
        return Condition(
            name=CONDITION_DRIVER_MINIMUM,
            satisfied=True,
            observed=(
                f"{observed}; adequacy established directly by "
                f"{VITISAI_PROVIDER} registering in this interpreter"
            ),
            required=required,
            remediation=None,
        )

    if installed is None:
        reason = identity.unavailable_reason or (
            "xrt-smi produced no NPU Driver Version field"
        )
        return Condition(
            name=CONDITION_DRIVER_MINIMUM,
            satisfied=False,
            observed=f"the installed NPU driver version is unknown: {reason}",
            required=required,
            remediation=(
                "Install the AMD NPU driver package so that xrt-smi can report "
                f"its version, then re-run this check. {_PROVISION_HINT}"
            ),
        )

    ordering = compare_versions(installed, DOCUMENTED_MINIMUM_DRIVER_VERSION)
    observed = (
        f"NPU driver {installed} against the documented minimum "
        f"{DOCUMENTED_MINIMUM_DRIVER_VERSION}"
    )
    if ordering is None:
        return Condition(
            name=CONDITION_DRIVER_MINIMUM,
            satisfied=False,
            observed=(
                f"{observed}: the two cannot be ordered component-wise, so this "
                "check draws no conclusion from them"
            ),
            required=required,
            remediation=(
                "Settle driver adequacy by provider registration rather than by "
                f"version: {_PROVISION_HINT}"
            ),
        )
    if ordering >= 0:
        return Condition(
            name=CONDITION_DRIVER_MINIMUM,
            satisfied=True,
            observed=(
                f"{observed}: component-wise, the installed driver is "
                f"{'the same as' if ordering == 0 else 'newer than'} the "
                "documented minimum"
            ),
            required=required,
            remediation=None,
        )
    return Condition(
        name=CONDITION_DRIVER_MINIMUM,
        satisfied=False,
        observed=f"{observed}: component-wise, the installed driver is lower",
        required=required,
        remediation=(
            f"Update the AMD NPU driver to {DOCUMENTED_MINIMUM_DRIVER_VERSION} "
            "or newer, then confirm the outcome by re-running this check: "
            f"{VITISAI_PROVIDER} appearing in "
            "onnxruntime.get_available_providers() is the decisive evidence, "
            "not the version string."
        ),
    )


def _provider_condition(probe: RuntimeProbe, *, registered: bool) -> Condition:
    required = (
        f"{VITISAI_PROVIDER} present in onnxruntime.get_available_providers(). "
        "Session construction is not an alternative check: requesting an absent "
        "provider succeeds, warns only, and runs on the CPU."
    )
    if registered:
        listed = ", ".join(probe.available_providers or ())
        return Condition(
            name=CONDITION_PROVIDER_REGISTERED,
            satisfied=True,
            observed=f"onnxruntime.get_available_providers() -> {listed}",
            required=required,
            remediation=None,
        )
    if probe.available_providers is None:
        observed = (
            "the available-provider list could not be read: "
            f"{probe.unavailable_reason}"
        )
    else:
        observed = (
            "onnxruntime.get_available_providers() -> "
            f"{', '.join(probe.available_providers) or 'nothing'}; "
            f"{VITISAI_PROVIDER} is absent, so NPU execution must not be "
            "attempted in this interpreter"
        )
    return Condition(
        name=CONDITION_PROVIDER_REGISTERED,
        satisfied=False,
        observed=observed,
        required=required,
        remediation=_PROVISION_HINT,
    )


def _variables_condition(
    environ: Mapping[str, str], *, registered: bool
) -> Condition:
    unset = tuple(
        name
        for name in VENDOR_ENVIRONMENT_VARIABLES
        if not environ.get(name, "").strip()
    )
    observed = "; ".join(
        _describe_variable(environ, name) for name in VENDOR_ENVIRONMENT_VARIABLES
    )
    required = (
        f"AMD documents {' and '.join(VENDOR_ENVIRONMENT_VARIABLES)} as locating "
        "the vendor runtime's native assets. Measured on this machine they are "
        "not required: both are unset and the provider registers and executes "
        "on the NPU, because provisioning places those assets beside the "
        "runtime instead."
    )
    if registered or not unset:
        return Condition(
            name=CONDITION_ENVIRONMENT_VARIABLES,
            satisfied=True,
            observed=observed,
            required=required,
            remediation=None,
        )
    return Condition(
        name=CONDITION_ENVIRONMENT_VARIABLES,
        satisfied=False,
        observed=observed,
        required=required,
        remediation=(
            f"{VITISAI_PROVIDER} is not registered and {', '.join(unset)} "
            f"{'is' if len(unset) == 1 else 'are'} unset. RyzenAI-SW issue #213 "
            "attributes non-registration to native assets the runtime cannot "
            "find. Either point these variables at the unpacked Ryzen AI "
            f"deployment payload, or - the supported route here - {_PROVISION_HINT}"
        ),
    )


def _describe_variable(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    return f"{name}={value}" if value else f"{name} unset"
