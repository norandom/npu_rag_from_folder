"""Unit tests for the ``xrt-smi`` telemetry wrapper (task 1.4).

Every test here parses **captured fixture text** rather than invoking the vendor
binary, so the suite is green on a machine with no NPU and no ``xrt-smi``. That
is not a convenience: design.md's ``CapabilityChecker`` invariant - "never raises
for a missing environment - absence is data, not an exception" - applies to this
wrapper too, and a test that needed the hardware could never demonstrate it.

Fixture provenance (``tests/embedding/fixtures/xrt/``):

- ``examine.txt``, ``platform_idle.txt``, ``platform_power_na.txt``,
  ``aie_partitions_idle.txt``, ``unknown_report.txt`` - captured verbatim from
  ``C:\\Windows\\System32\\AMD\\xrt-smi.exe`` on this machine on 2026-09-05.
- ``platform_load.txt``, ``aie_partitions_load.txt`` - the under-load shapes
  recorded in research.md, "Third probe: NPU execution CONFIRMED end to end",
  during the sustained 45 s / 6047-inference run.
- ``platform_no_power_field.txt`` - synthetic. AMD documents power reporting as
  absent on PHX/HPT parts and on Linux; this machine is Strix, so no captured
  sample of that shape can exist here.

The live counterpart lives in ``test_xrt_live.py`` and skips when the binary is
absent.
"""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from npu_rag.embedding.environment.xrt import (
    DEFAULT_XRT_SMI_PATH,
    CommandResult,
    DeviceIdentity,
    PartitionOccupancy,
    PlatformTelemetry,
    PowerReading,
    PowerStatus,
    XrtSmiWrapper,
    parse_examine,
    parse_partition_report,
    parse_platform_report,
    run_xrt_smi,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "xrt"

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "npu_rag"
    / "embedding"
    / "environment"
    / "xrt.py"
)


def fixture(name: str) -> str:
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Dependency direction
# --------------------------------------------------------------------------

#: The module's own dotted package, used to resolve relative imports.
MODULE_PACKAGE = "npu_rag.embedding.environment"

#: design.md, Architecture: "types, errors -> reporting -> profiles ->
#: environment -> models -> providers -> service -> bench". This module sits in
#: ``environment``, so everything to its right is off limits.
LAYERS_RIGHT_OF_ENVIRONMENT = ("models", "providers", "service", "bench")


def absolute_imports_of(source: str, package: str) -> list[str]:
    """Every name ``source`` imports, as an absolute dotted path.

    Relative imports are resolved rather than skipped: ``from .. import
    providers`` and ``import npu_rag.embedding.providers`` are the same
    violation, and a guard that only recognises the second is trivially evaded
    by writing the first.
    """
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
            # ``from X import Y`` may name a submodule, not just an attribute.
            names.extend(f"{root}.{alias.name}" for alias in node.names)
    return names


def test_module_imports_nothing_from_a_later_layer() -> None:
    """``test_package_baseline.py`` guards the outer boundary - no sibling
    ``npu_rag`` package. It does not order the layers *inside*
    ``npu_rag.embedding``, so that half of design.md's dependency direction is
    asserted here, scoped to this module."""
    imported = absolute_imports_of(MODULE_PATH.read_text(encoding="utf-8"), MODULE_PACKAGE)

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


@pytest.mark.parametrize(
    "statement",
    [
        "import npu_rag.embedding.providers",
        "from npu_rag.embedding.providers import base",
        "from npu_rag.embedding import providers",
        "from .. import providers",
        "from ..providers import base",
        "from ...embedding.service import EmbeddingService",
    ],
)
def test_the_layer_guard_recognises_every_import_spelling(statement: str) -> None:
    """The guard above is only worth having if it cannot be side-stepped by
    choosing a different import form."""
    imported = absolute_imports_of(statement, MODULE_PACKAGE)

    assert any(
        name.startswith(f"npu_rag.embedding.{layer}")
        for name in imported
        for layer in LAYERS_RIGHT_OF_ENVIRONMENT
    ), f"{statement!r} resolved to {imported}"


# --------------------------------------------------------------------------
# Fixture integrity
# --------------------------------------------------------------------------


def test_platform_fixture_retains_trailing_whitespace() -> None:
    """``xrt-smi`` pads its platform values with a trailing space.

    Asserted on the raw bytes because an editor or a formatting hook that
    strips trailing whitespace would silently delete the only coverage the
    parser has for that quirk, leaving the suite green and the parser untested.
    """
    raw = (FIXTURES / "platform_idle.txt").read_bytes()
    assert b"Power Mode             : Default \r\n" in raw or (
        b"Power Mode             : Default \n" in raw
    )


# --------------------------------------------------------------------------
# parse_examine - device identity and versions (requirement 1.1)
# --------------------------------------------------------------------------


def test_parse_examine_reads_versions_and_device_identity() -> None:
    identity = parse_examine(fixture("examine"))

    assert identity.xrt_version == "2.21.0"
    assert identity.npu_driver_version == "32.0.20102.3930"
    assert identity.npu_firmware_version == "1.1.2.64"
    assert identity.device_name == "NPU Strix"
    assert identity.bdf == "00c6:00:01.1"
    assert identity.processor is not None
    assert "Ryzen AI 9 HX 370" in identity.processor
    assert identity.unavailable_reason is None


def test_parse_examine_does_not_confuse_bios_version_with_xrt_version() -> None:
    """``BIOS Version : 2.10`` appears *before* ``Version : 2.21.0`` in the real
    report.

    Exact-key lookup is already safe here - ``BIOS Version`` is a distinct dict
    key, not a qualified spelling of ``Version`` - so this does not guard the
    flat scan. What it does guard is any *loosening* of that key match: a parser
    that scanned for lines containing "Version", or matched keys by
    ``endswith``, would take the BIOS value because it comes first in the
    document. It also pins the section-scoped lookup that keeps this true if a
    future build adds a bare ``Version`` key outside the XRT section.
    """
    identity = parse_examine(fixture("examine"))

    assert identity.xrt_version == "2.21.0"
    assert identity.xrt_version != "2.10"


def test_parse_examine_of_an_empty_report_is_data_not_an_exception() -> None:
    identity = parse_examine("")

    assert identity == DeviceIdentity(
        bdf=None,
        device_name=None,
        xrt_version=None,
        npu_driver_version=None,
        npu_firmware_version=None,
        processor=None,
        unavailable_reason=identity.unavailable_reason,
    )
    assert identity.unavailable_reason is not None


def test_parse_examine_of_a_utility_error_dump_is_unavailable() -> None:
    """``xrt-smi`` exits 0 even when it rejects the requested report, so the
    only signal is that the expected fields are absent from the output."""
    identity = parse_examine(fixture("unknown_report"))

    assert identity.xrt_version is None
    assert identity.npu_driver_version is None
    assert identity.unavailable_reason is not None


# --------------------------------------------------------------------------
# parse_platform_report - power, mode, columns (requirements 1.1, 6.2)
# --------------------------------------------------------------------------


def test_parse_platform_report_reads_a_present_watt_value() -> None:
    telemetry = parse_platform_report(fixture("platform_idle"))

    assert telemetry.power.status is PowerStatus.REPORTED
    assert telemetry.power.watts == pytest.approx(0.002)
    assert telemetry.power.reason is None
    assert telemetry.power_mode == "Default"
    assert telemetry.total_columns == 8
    assert telemetry.device_name == "NPU Strix"
    assert telemetry.bdf == "00c6:00:01.1"


def test_parse_platform_report_reads_the_under_load_watt_value() -> None:
    telemetry = parse_platform_report(fixture("platform_load"))

    assert telemetry.power.status is PowerStatus.REPORTED
    assert telemetry.power.watts == pytest.approx(0.438)


def test_not_available_power_is_never_reported_as_zero_watts() -> None:
    """Requirement 6.2's energy figure is the time-integral of polled Watts.

    ``Estimated Power : N/A`` occurs intermittently on this machine even though
    power reporting *is* supported here - observed alternating with ``0.002
    Watts`` across consecutive polls on 2026-09-05. Folding it to 0.0 would
    silently drag the integrated energy toward zero, so the wrapper must return
    no number at all.
    """
    telemetry = parse_platform_report(fixture("platform_power_na"))

    assert telemetry.power.status is PowerStatus.UNAVAILABLE
    assert telemetry.power.watts is None
    assert telemetry.power.watts != 0.0
    assert telemetry.power.reason is not None
    assert "N/A" in telemetry.power.reason
    # The rest of the report is still readable; one absent field is not a
    # failure of the whole read.
    assert telemetry.total_columns == 8


def test_a_genuine_zero_watt_reading_stays_a_number() -> None:
    """The mirror of the test above: 0.0 W must survive as a value, otherwise
    "unavailable" and "idle" become indistinguishable in the other direction."""
    text = fixture("platform_idle").replace("0.002 Watts", "0.000 Watts")
    telemetry = parse_platform_report(text)

    assert telemetry.power.status is PowerStatus.REPORTED
    assert telemetry.power.watts == 0.0


def test_a_platform_without_a_power_field_reports_unsupported() -> None:
    """AMD reports estimated power only on Strix-class parts and later. A
    platform report that is otherwise well formed but carries no power field is
    the permanent-unsupported case, distinct from a transient ``N/A``."""
    telemetry = parse_platform_report(fixture("platform_no_power_field"))

    assert telemetry.power.status is PowerStatus.UNSUPPORTED
    assert telemetry.power.watts is None
    assert telemetry.power.reason is not None
    assert telemetry.total_columns == 5
    assert telemetry.device_name == "NPU Phoenix"


def test_an_absent_platform_report_is_unavailable_not_unsupported() -> None:
    """No platform section at all means the report was never produced. That is
    not evidence about what the device supports."""
    telemetry = parse_platform_report("")

    assert telemetry.power.status is PowerStatus.UNAVAILABLE
    assert telemetry.power.watts is None
    assert telemetry.total_columns is None


@pytest.mark.parametrize("raw", ["banana", "", "-", "0.4 0.5 Watts"])
def test_an_unparsable_power_value_is_data_not_an_exception(raw: str) -> None:
    text = fixture("platform_idle").replace("0.002 Watts", raw)
    telemetry = parse_platform_report(text)

    assert telemetry.power.status is PowerStatus.UNAVAILABLE
    assert telemetry.power.watts is None
    assert telemetry.power.reason is not None


@pytest.mark.parametrize("raw", ["-0.5 Watts", "-1 Watts", "-0.001 Watts"])
def test_a_negative_wattage_is_refused(raw: str) -> None:
    """``_WATTS`` accepts a leading sign, so a negative reading parses cleanly as
    a number. Requirement 6.2 integrates polled Watts over wall-clock, and a
    negative sample would *subtract* from the accumulated energy - a quieter
    corruption than an absent reading, because it still looks like data."""
    text = fixture("platform_idle").replace("0.002 Watts", raw)
    telemetry = parse_platform_report(text)

    assert telemetry.power.status is PowerStatus.UNAVAILABLE
    assert telemetry.power.watts is None
    assert telemetry.power.reason is not None
    assert "implausible" in telemetry.power.reason


@pytest.mark.parametrize("watts", [float("nan"), float("inf"), float("-inf")])
def test_power_reading_rejects_a_non_finite_value(watts: float) -> None:
    """The ``_WATTS`` regex cannot emit these, so this guards the type rather
    than the parser: any future producer of a ``PowerReading`` is held to a
    value that can actually be integrated."""
    with pytest.raises(ValueError):
        PowerReading(watts=watts, status=PowerStatus.REPORTED, reason=None)


def test_a_non_watt_unit_is_refused_rather_than_silently_rescaled() -> None:
    """Every observed reading is in Watts. If a future build emitted milliwatts,
    accepting the number unscaled would understate energy by a factor of 1000 -
    worse than reporting nothing."""
    text = fixture("platform_idle").replace("0.002 Watts", "438 mW")
    telemetry = parse_platform_report(text)

    assert telemetry.power.status is PowerStatus.UNAVAILABLE
    assert telemetry.power.watts is None


# --------------------------------------------------------------------------
# parse_partition_report - runtime evidence of NPU execution
# --------------------------------------------------------------------------


def test_idle_partition_report_shows_no_live_context() -> None:
    occupancy = parse_partition_report(fixture("aie_partitions_idle"))

    assert occupancy.context_live is False
    assert occupancy.partitions == ()
    assert occupancy.occupied_columns == ()
    assert occupancy.unavailable_reason is None


def test_under_load_partition_report_shows_the_occupied_columns() -> None:
    """Task 1.3 established this as the hardware-level evidence that separates
    genuine NPU execution from a silent CPU fallback inside the EP."""
    occupancy = parse_partition_report(fixture("aie_partitions_load"))

    assert occupancy.context_live is True
    assert len(occupancy.partitions) == 1
    assert occupancy.partitions[0].index == 0
    assert occupancy.partitions[0].columns == (0, 1, 2, 3, 4, 5, 6, 7)
    assert occupancy.occupied_columns == (0, 1, 2, 3, 4, 5, 6, 7)
    assert occupancy.unavailable_reason is None


def test_multiple_partitions_are_all_reported() -> None:
    text = (
        "AIE Partitions\n"
        "Partition Index   : 0\n"
        "  Columns: [0, 1, 2, 3]\n"
        "  HW Contexts:\n"
        "Partition Index   : 1\n"
        "  Columns: [4, 5]\n"
        "  HW Contexts:\n"
    )
    occupancy = parse_partition_report(text)

    assert occupancy.context_live is True
    assert tuple(p.index for p in occupancy.partitions) == (0, 1)
    assert occupancy.occupied_columns == (0, 1, 2, 3, 4, 5)


def test_an_absent_partition_report_leaves_liveness_unknown() -> None:
    """``False`` would assert the NPU is idle. Nothing was read, so the honest
    answer is that liveness is unknown."""
    occupancy = parse_partition_report("")

    assert occupancy.context_live is None
    assert occupancy.partitions == ()
    assert occupancy.unavailable_reason is not None


def test_a_utility_error_dump_leaves_liveness_unknown() -> None:
    occupancy = parse_partition_report(fixture("unknown_report"))

    assert occupancy.context_live is None
    assert occupancy.unavailable_reason is not None


# --------------------------------------------------------------------------
# PowerReading invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("watts", "status", "reason"),
    [
        (None, PowerStatus.REPORTED, None),
        (None, PowerStatus.REPORTED, "missing"),
        (0.5, PowerStatus.UNAVAILABLE, "absent"),
        (0.5, PowerStatus.UNSUPPORTED, None),
        (0.5, PowerStatus.REPORTED, "why"),
    ],
)
def test_power_reading_rejects_contradictory_construction(
    watts: float | None, status: PowerStatus, reason: str | None
) -> None:
    """A value and a reason are mutually exclusive - the same invariant
    design.md places on ``Measurement``. Violating it is a programming error,
    which is the one thing this module is allowed to raise for."""
    with pytest.raises(ValueError):
        PowerReading(watts=watts, status=status, reason=reason)


# --------------------------------------------------------------------------
# XrtSmiWrapper - subprocess orchestration
# --------------------------------------------------------------------------


class RecordingRunner:
    """A stand-in for the subprocess call that records how it was invoked."""

    def __init__(self, results: Sequence[CommandResult | Exception]) -> None:
        self._results = list(results)
        self.calls: list[Sequence[str]] = []
        self.timeouts: list[float] = []

    def __call__(
        self, argv: Sequence[str], *, timeout_seconds: float
    ) -> CommandResult:
        self.calls.append(argv)
        self.timeouts.append(timeout_seconds)
        outcome = self._results[min(len(self.calls) - 1, len(self._results) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok(stdout: str) -> CommandResult:
    return CommandResult(returncode=0, stdout=stdout, stderr="")


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    path = tmp_path / "xrt-smi.exe"
    path.write_bytes(b"not a real binary")
    return path


def test_default_executable_is_the_driver_shipped_path() -> None:
    """``xrt-smi`` ships with the NPU driver, not the SDK, and is not on PATH -
    so it must be resolved explicitly. A default, never a hardcoded constant."""
    assert DEFAULT_XRT_SMI_PATH == Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
    assert XrtSmiWrapper().executable == DEFAULT_XRT_SMI_PATH


def test_reports_are_requested_with_a_fixed_argument_vector(
    fake_binary: Path,
) -> None:
    """design.md, Security Considerations: "xrt-smi is invoked with a fixed
    argument vector and never with shell interpolation"."""
    runner = RecordingRunner([ok(fixture("platform_idle"))])
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)

    wrapper.read_platform()

    argv = runner.calls[0]
    assert not isinstance(argv, str)
    assert list(argv) == [str(fake_binary), "examine", "--report", "platform"]


def test_each_report_uses_its_own_subcommand(fake_binary: Path) -> None:
    runner = RecordingRunner([ok("")])
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)

    wrapper.read_identity()
    wrapper.read_partitions()

    assert list(runner.calls[0]) == [str(fake_binary), "examine"]
    assert list(runner.calls[1]) == [
        str(fake_binary),
        "examine",
        "--report",
        "aie-partitions",
    ]


def test_repeated_polling_re_invokes_and_types_each_sample(
    fake_binary: Path,
) -> None:
    """The benchmark samples power *during* a running workload, so successive
    polls must reflect what the device reported at that moment - including a
    mid-run ``N/A``, which must not corrupt the samples either side of it."""
    runner = RecordingRunner(
        [
            ok(fixture("platform_idle")),
            ok(fixture("platform_power_na")),
            ok(fixture("platform_load")),
        ]
    )
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)

    samples = [wrapper.read_power() for _ in range(3)]

    assert len(runner.calls) == 3
    assert [s.status for s in samples] == [
        PowerStatus.REPORTED,
        PowerStatus.UNAVAILABLE,
        PowerStatus.REPORTED,
    ]
    assert [s.watts for s in samples] == [pytest.approx(0.002), None, pytest.approx(0.438)]


def test_polling_does_not_re_resolve_the_binary(fake_binary: Path) -> None:
    """Resolution happens once, at construction, so a sampling loop pays no
    filesystem cost per poll and cannot change target mid-run."""
    runner = RecordingRunner([ok(fixture("platform_idle"))])
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)
    resolved = wrapper.resolved_executable

    fake_binary.unlink()

    for _ in range(3):
        assert wrapper.read_power().status is PowerStatus.REPORTED
    assert wrapper.resolved_executable == resolved
    assert {tuple(call) for call in runner.calls} == {
        (str(fake_binary), "examine", "--report", "platform")
    }


def test_an_absent_utility_is_data_on_every_read(tmp_path: Path) -> None:
    """The headline constraint: this wrapper must work on a machine with no NPU
    and no ``xrt-smi``, returning typed "unavailable" values rather than
    raising. ``CapabilityChecker`` (task 1.5) depends on that."""
    wrapper = XrtSmiWrapper(tmp_path / "nowhere" / "xrt-smi.exe")

    assert wrapper.available is False
    assert wrapper.resolved_executable is None

    power = wrapper.read_power()
    assert power.status is PowerStatus.UNAVAILABLE
    assert power.watts is None
    assert power.reason is not None
    assert "xrt-smi" in power.reason

    assert isinstance(wrapper.read_platform(), PlatformTelemetry)
    assert wrapper.read_identity().unavailable_reason is not None

    occupancy = wrapper.read_partitions()
    assert isinstance(occupancy, PartitionOccupancy)
    assert occupancy.context_live is None
    assert occupancy.unavailable_reason is not None


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(2, "No such file or directory"),
        PermissionError(13, "Permission denied"),
        OSError("[WinError 216] not compatible with this version of Windows"),
        subprocess.TimeoutExpired(cmd="xrt-smi", timeout=15.0),
    ],
)
def test_a_utility_that_cannot_run_is_data_not_an_exception(
    fake_binary: Path, failure: Exception
) -> None:
    runner = RecordingRunner([failure])
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)

    power = wrapper.read_power()

    assert power.status is PowerStatus.UNAVAILABLE
    assert power.watts is None
    assert power.reason is not None


def test_a_timeout_reason_names_the_timeout(fake_binary: Path) -> None:
    runner = RecordingRunner([subprocess.TimeoutExpired(cmd="xrt-smi", timeout=15.0)])
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)

    reason = wrapper.read_power().reason

    assert reason is not None
    assert "timed out" in reason.lower()


def test_a_failing_exit_code_carries_stderr_into_the_reason(
    fake_binary: Path,
) -> None:
    runner = RecordingRunner(
        [CommandResult(returncode=1, stdout="", stderr="device not found")]
    )
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)

    power = wrapper.read_power()

    assert power.status is PowerStatus.UNAVAILABLE
    assert power.reason is not None
    assert "device not found" in power.reason


def test_a_zero_exit_code_with_an_error_dump_is_still_unavailable(
    fake_binary: Path,
) -> None:
    """Measured on this machine: ``xrt-smi examine --report bogus`` prints an
    error and exits **0**. The exit code carries no information, so the parser
    decides, not the return code."""
    runner = RecordingRunner([ok(fixture("unknown_report"))])
    wrapper = XrtSmiWrapper(fake_binary, runner=runner)

    assert wrapper.read_power().status is PowerStatus.UNAVAILABLE
    assert wrapper.read_partitions().context_live is None


def test_the_configured_timeout_reaches_the_runner(fake_binary: Path) -> None:
    runner = RecordingRunner([ok("")])
    wrapper = XrtSmiWrapper(fake_binary, runner=runner, timeout_seconds=2.5)

    wrapper.read_power()

    assert runner.timeouts == [2.5]


# --------------------------------------------------------------------------
# run_xrt_smi - the call that actually reaches the operating system
# --------------------------------------------------------------------------


class SubprocessSpy:
    """Captures how ``subprocess.run`` was called, without running anything."""

    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}
        self.called = False

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.called = True
        self.args = args
        self.kwargs = kwargs
        argv: Any = args[0] if args else kwargs.get("args", [])
        return subprocess.CompletedProcess(
            args=argv,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr="",
        )


def test_run_xrt_smi_never_invokes_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """design.md, Security Considerations: "xrt-smi is invoked with a fixed
    argument vector and never with shell interpolation".

    The argv test above stops at the injected runner, so it proves how
    ``_invoke`` *composes* the vector but never reaches ``subprocess.run``. This
    is the other half of that control, and it is the half that touches the OS:
    with ``shell=True`` on Windows the same list is handed to ``cmd.exe``, which
    both succeeds and reads identically at every other layer.
    """
    spy = SubprocessSpy(stdout="report text")
    monkeypatch.setattr(subprocess, "run", spy)

    result = run_xrt_smi(
        ("C:/Windows/System32/AMD/xrt-smi.exe", "examine", "--report", "platform"),
        timeout_seconds=3.0,
    )

    assert spy.called
    assert "shell" in spy.kwargs, (
        "shell must be stated explicitly, not left to the subprocess default - "
        "it is the named control in design.md's Security Considerations"
    )
    assert spy.kwargs["shell"] is False

    # Checked in this order so the string rejection is a real assertion rather
    # than one a type narrowing has already made unreachable.
    argv = spy.args[0]
    assert not isinstance(argv, str), (
        "a single string argv is the shell-interpolation shape the design "
        "forbids"
    )
    assert isinstance(argv, list)
    assert argv == [
        "C:/Windows/System32/AMD/xrt-smi.exe",
        "examine",
        "--report",
        "platform",
    ]

    # A failing utility must come back as data, so the call must not raise.
    assert spy.kwargs["check"] is False
    assert spy.kwargs["capture_output"] is True
    assert spy.kwargs["timeout"] == 3.0
    assert result.stdout == "report text"
    assert result.returncode == 0


def test_the_wrappers_default_runner_is_the_guarded_one(
    fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closes the loop between the two argv tests.

    Without this, the wrapper could default to some other runner and the
    ``shell=False`` guard above would be verifying a function nothing calls.
    """
    spy = SubprocessSpy(stdout=fixture("platform_idle"))
    monkeypatch.setattr(subprocess, "run", spy)

    reading = XrtSmiWrapper(fake_binary).read_power()

    assert spy.called
    assert spy.kwargs["shell"] is False
    assert spy.args[0] == [str(fake_binary), "examine", "--report", "platform"]
    assert reading.status is PowerStatus.REPORTED
