"""Read NPU telemetry from the vendor ``xrt-smi`` utility.

This is design.md's ``XrtSmiWrapper``. It answers three questions, each by
invoking the utility with a fixed argument vector and parsing its plain-text
report:

- **Who is this device?** ``xrt-smi examine`` yields the XRT, NPU driver, and
  NPU firmware versions plus the device name and BDF. Requirement 1.1 needs
  these for the capability report, and 6.6 needs them for benchmark run
  provenance.
- **How much power is it drawing?** ``xrt-smi examine --report platform`` yields
  an estimated wattage, which requirement 6.2 integrates over wall-clock into
  energy per one thousand inputs.
- **Is the NPU actually executing?** ``xrt-smi examine --report aie-partitions``
  yields the live hardware contexts and the accelerator columns they occupy.
  Task 1.3 established this as the hardware-level evidence that distinguishes
  genuine NPU execution from a silent CPU fallback inside the execution
  provider - the failure mode the whole feature exists to make impossible.

Three properties of the real utility shape every decision here.

**Absence is data, not an exception.** ``CapabilityChecker`` (task 1.5) must run
to completion on a machine with no NPU and no ``xrt-smi``, reporting each
condition separately instead of raising. So must this wrapper: a missing binary,
a binary that will not execute, a timeout, and a report that does not contain
the expected fields all return typed values carrying a reason. The only thing
this module raises for is a programming error - constructing a ``PowerReading``
that contradicts itself.

**``N/A`` is not zero.** ``Estimated Power`` reads ``N/A`` intermittently even on
this machine, where power reporting *is* supported - observed alternating with
``0.002 Watts`` across consecutive polls on 2026-09-05. AMD also documents power
reporting as absent entirely on PHX/HPT parts and on Linux, so the same token
carries both meanings and this wrapper cannot tell them apart. What it can do is
refuse to invent a number: an absent reading yields ``watts is None``, never
``0.0``. Folding the two together would drag requirement 6.2's integrated energy
silently toward zero.

**The exit code carries no information.** Measured on this machine,
``xrt-smi examine --report bogus`` prints an error and exits **0**. A non-zero
code is therefore treated as a failure, but a zero code is not treated as
success - the parser decides, by whether the fields it needs are present.

This module sits in the ``environment`` layer of design.md's dependency
direction (``types, errors -> reporting -> profiles -> environment -> models ->
providers -> service -> bench``) and imports nothing from any other ``npu_rag``
sub-package.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_XRT_SMI_PATH",
    "AiePartition",
    "CommandResult",
    "CommandRunner",
    "DeviceIdentity",
    "PartitionOccupancy",
    "PlatformTelemetry",
    "PowerReading",
    "PowerStatus",
    "XrtSmiWrapper",
    "parse_examine",
    "parse_partition_report",
    "parse_platform_report",
    "run_xrt_smi",
]

#: ``xrt-smi`` ships with the **NPU driver**, not with the Ryzen AI SDK, and it
#: is not on ``PATH``. Telemetry therefore works on this machine without the SDK
#: installed (design.md, Allowed Dependencies) - but only if the path is
#: resolved explicitly. A default, never a constant: another machine may put it
#: elsewhere, so every entry point takes it as a parameter.
DEFAULT_XRT_SMI_PATH = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")

#: Generous relative to the ~0.2 s a report actually takes. The cost of being
#: wrong in the other direction is a benchmark that hangs mid-run.
DEFAULT_TIMEOUT_SECONDS = 15.0

_EXAMINE: tuple[str, ...] = ("examine",)
_PLATFORM_REPORT: tuple[str, ...] = ("examine", "--report", "platform")
_PARTITION_REPORT: tuple[str, ...] = ("examine", "--report", "aie-partitions")

_SECTION_XRT = "XRT"
_SECTION_PLATFORM = "Platform"
_SECTION_AIE_PARTITIONS = "AIE Partitions"

_POWER_FIELD = "Estimated Power"
_PARTITION_INDEX_FIELD = "Partition Index"
_COLUMNS_FIELD = "Columns"

#: The idle sentinel the partition report prints in place of a partition block.
_NO_CONTEXTS_SENTINEL = "no hardware contexts running on device"

#: Tokens ``xrt-smi`` substitutes for a reading it does not have.
_ABSENT_VALUE_TOKENS = frozenset({"N/A", "NA", "NOT SUPPORTED", "UNSUPPORTED"})

#: ``  Power Mode             : Default `` - note the padding on both sides of
#: the colon and the trailing space after the value, all of which the real
#: utility emits. The key must start alphanumeric so that the device banner
#: ``[00c6:00:01.1] : NPU Strix`` and the table rows are not mistaken for
#: fields, and ``[^:]*?`` stops the key at the first colon so that values
#: containing colons (``Hash Date : 2026-05-07 16:04:09``) survive intact.
_FIELD = re.compile(r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z0-9][^:]*?)[ \t]*:[ \t]*(?P<value>.*?)[ \t]*$")

#: ``[00c6:00:01.1] : NPU Strix`` - the per-device banner above each report.
_DEVICE_BANNER = re.compile(r"^[ \t]*\[(?P<bdf>[^\]]+)\][ \t]*:[ \t]*(?P<name>\S.*?)[ \t]*$")

#: ``|[00c6:00:01.1]  |NPU Strix  |`` - the ``Device(s) Present`` table row.
_DEVICE_ROW = re.compile(r"^[ \t]*\|[ \t]*\[(?P<bdf>[^\]]+)\][ \t]*\|[ \t]*(?P<name>[^|]*?)[ \t]*\|")

#: ``0.001 Watts``. Anchored with ``fullmatch`` so a trailing unit this parser
#: does not understand (``438 mW``) fails rather than being accepted unscaled -
#: a silent factor-of-1000 error in requirement 6.2's energy figure would be
#: worse than reporting nothing at all.
_WATTS = re.compile(
    r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+))[ \t]*(?:W|Watt|Watts)?",
    re.IGNORECASE,
)

_INTEGER = re.compile(r"[-+]?\d+")


class PowerStatus(StrEnum):
    """Why a power reading does or does not carry a number.

    The three states are deliberately not two. ``UNAVAILABLE`` means *this
    reading* is missing - the utility is absent, or it printed ``N/A`` on this
    poll - and a later poll may well succeed. ``UNSUPPORTED`` means the platform
    report was produced and simply has no power field, which is what AMD
    documents for PHX/HPT parts and for Linux; polling harder will not help.
    Requirement 6.8 records omissions with their reason, and those are different
    reasons.
    """

    REPORTED = "reported"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class PowerReading:
    """One estimated-power sample, or a typed account of why there isn't one.

    Invariant, mirroring the one design.md places on ``Measurement``: exactly
    one of ``watts`` and ``reason`` is set, and ``watts`` is present exactly
    when the status is ``REPORTED``. Constructing a reading that says both, or
    neither, is a programming error and raises - it is the only thing in this
    module that does.
    """

    watts: float | None
    status: PowerStatus
    reason: str | None

    def __post_init__(self) -> None:
        has_value = self.watts is not None
        if has_value is not (self.status is PowerStatus.REPORTED):
            raise ValueError(
                f"status {self.status!r} disagrees with watts={self.watts!r}: a "
                "value is present exactly when the status is REPORTED"
            )
        if has_value is (self.reason is not None):
            raise ValueError(
                "exactly one of watts and reason must be set, got "
                f"watts={self.watts!r}, reason={self.reason!r}"
            )
        if self.watts is not None and not math.isfinite(self.watts):
            raise ValueError(f"watts must be finite, got {self.watts!r}")


@dataclass(frozen=True)
class DeviceIdentity:
    """Device and version provenance from ``xrt-smi examine``.

    Feeds requirement 1.1's capability conditions and requirement 6.6's
    benchmark run context. Every field is optional because the whole point is
    that this must be readable - as an answer, not an exception - on a machine
    where none of it exists.
    """

    bdf: str | None
    device_name: str | None
    xrt_version: str | None
    npu_driver_version: str | None
    npu_firmware_version: str | None
    processor: str | None
    #: Non-``None`` exactly when nothing at all could be read.
    unavailable_reason: str | None


@dataclass(frozen=True)
class PlatformTelemetry:
    """The platform report: estimated power plus the device's static shape."""

    power: PowerReading
    power_mode: str | None
    total_columns: int | None
    device_name: str | None
    bdf: str | None


@dataclass(frozen=True)
class AiePartition:
    """One AIE partition and the accelerator columns it occupies."""

    index: int
    columns: tuple[int, ...]


@dataclass(frozen=True)
class PartitionOccupancy:
    """Whether the NPU is executing, and across which columns.

    ``context_live`` is tri-state on purpose. ``False`` is a positive
    observation - the utility ran and said no contexts exist. ``None`` means the
    report could not be read, so liveness is *unknown*; reporting ``False``
    there would assert an idle NPU on no evidence, which is the same class of
    mistake as reporting ``0.0`` Watts for an absent power reading.
    """

    context_live: bool | None
    partitions: tuple[AiePartition, ...]
    #: Non-``None`` exactly when ``context_live`` is ``None``.
    unavailable_reason: str | None

    @property
    def occupied_columns(self) -> tuple[int, ...]:
        """Every column occupied by any partition, sorted and de-duplicated."""
        return tuple(sorted({column for p in self.partitions for column in p.columns}))


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one utility invocation."""

    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """How the wrapper reaches a subprocess.

    Injectable so that argument-vector construction and the failure paths are
    testable without the vendor binary - and so the tests can prove a *fixed
    argument vector* is passed, never a shell string.
    """

    def __call__(
        self, argv: Sequence[str], *, timeout_seconds: float
    ) -> CommandResult: ...


def run_xrt_smi(argv: Sequence[str], *, timeout_seconds: float) -> CommandResult:
    """Invoke the utility with a fixed argument vector.

    ``shell=False`` is stated explicitly rather than left to the default: it is
    the mechanism behind design.md's Security Considerations requirement that
    ``xrt-smi`` "is invoked with a fixed argument vector and never with shell
    interpolation".
    """
    # Fixed argv, shell=False, read-only report: no caller-supplied text ever
    # reaches a shell, and nothing here mutates the system.
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        check=False,
        shell=False,
        timeout=timeout_seconds,
        encoding="utf-8",
        errors="replace",
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


class _Report:
    """A parsed ``xrt-smi`` text report: ordered sections of ``key : value``."""

    def __init__(
        self,
        sections: Mapping[str, Mapping[str, str]],
        devices: Sequence[tuple[str, str]],
    ) -> None:
        self._sections = sections
        self._devices = tuple(devices)

    def has_section(self, name: str) -> bool:
        return name in self._sections

    def get_in(self, section: str, key: str) -> str | None:
        return self._sections.get(section, {}).get(key)

    def get(self, key: str) -> str | None:
        """First match across sections in document order.

        Safe for the keys this module reads because ``xrt-smi`` spells them in
        full - ``BIOS Version`` and ``NPU Driver Version`` are distinct keys,
        not qualified variants of ``Version``.
        """
        for fields in self._sections.values():
            if key in fields:
                return fields[key]
        return None

    def first_device(self) -> tuple[str | None, str | None]:
        if not self._devices:
            return None, None
        bdf, name = self._devices[0]
        return bdf, name


def _parse_report(text: str) -> _Report:
    sections: dict[str, dict[str, str]] = {}
    devices: list[tuple[str, str]] = []
    current: dict[str, str] = {}
    sections[""] = current

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        # A rule of dashes separates the per-device banner from the report body.
        if set(stripped) <= {"-", "="}:
            continue

        banner = _DEVICE_BANNER.match(line)
        if banner is not None:
            devices.append((banner["bdf"].strip(), banner["name"].strip()))
            continue

        row = _DEVICE_ROW.match(line)
        if row is not None:
            name = row["name"].strip()
            # ``name.strip("-")`` rejects a rule row that happens to carry
            # brackets, without rejecting a real name.
            if name.strip("-"):
                devices.append((row["bdf"].strip(), name))
            continue

        if stripped.startswith("|"):
            continue

        match = _FIELD.match(line)
        if match is not None:
            current[match["key"].strip()] = match["value"].strip()
            continue

        # Anything else unindented and colon-free heads a new section.
        if not line[: len(line) - len(line.lstrip())]:
            current = sections.setdefault(stripped, {})

    return _Report(sections, devices)


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = _INTEGER.fullmatch(value.strip())
    return int(match.group()) if match is not None else None


def _as_int_list(value: str) -> tuple[int, ...]:
    """``[0, 1, 2, 3]`` -> ``(0, 1, 2, 3)``; unparsable entries are dropped."""
    inner = value.strip().strip("[]")
    parsed = (_as_int(token) for token in inner.split(",") if token.strip())
    return tuple(number for number in parsed if number is not None)


def _read_power(raw: str | None, *, platform_present: bool) -> PowerReading:
    if raw is None:
        if not platform_present:
            return PowerReading(
                watts=None,
                status=PowerStatus.UNAVAILABLE,
                reason=(
                    "xrt-smi produced no platform report, so no estimated-power "
                    "field could be read."
                ),
            )
        return PowerReading(
            watts=None,
            status=PowerStatus.UNSUPPORTED,
            reason=(
                f"the xrt-smi platform report carries no {_POWER_FIELD!r} field; "
                "estimated power is reported only on Strix-class devices and "
                "later, and not on Linux."
            ),
        )

    value = raw.strip()
    if value.upper() in _ABSENT_VALUE_TOKENS:
        return PowerReading(
            watts=None,
            status=PowerStatus.UNAVAILABLE,
            reason=(
                f"xrt-smi reported {_POWER_FIELD} as {value!r}. This occurs "
                "intermittently even where power reporting is supported, and is "
                "also how an unsupported part reports. It is not 0.0 W and must "
                "not be integrated as one."
            ),
        )

    match = _WATTS.fullmatch(value)
    if match is None:
        return PowerReading(
            watts=None,
            status=PowerStatus.UNAVAILABLE,
            reason=(
                f"could not read {_POWER_FIELD} from {raw!r}: expected a number "
                "of Watts."
            ),
        )

    watts = float(match["value"])
    if not math.isfinite(watts) or watts < 0.0:
        return PowerReading(
            watts=None,
            status=PowerStatus.UNAVAILABLE,
            reason=f"xrt-smi reported an implausible {_POWER_FIELD} of {raw!r}.",
        )
    return PowerReading(watts=watts, status=PowerStatus.REPORTED, reason=None)


def parse_examine(text: str) -> DeviceIdentity:
    """Parse ``xrt-smi examine`` into device and version provenance."""
    report = _parse_report(text)
    bdf, device_name = report.first_device()
    # Section-scoped: ``Version`` under ``XRT`` is the XRT version, and stating
    # the section keeps it that way even if a future build adds a bare
    # ``Version`` key elsewhere in the document.
    xrt_version = report.get_in(_SECTION_XRT, "Version")
    identity = DeviceIdentity(
        bdf=bdf,
        device_name=device_name,
        xrt_version=xrt_version,
        npu_driver_version=report.get("NPU Driver Version"),
        npu_firmware_version=report.get("NPU Firmware Version"),
        processor=report.get("Processor"),
        unavailable_reason=None,
    )
    known: tuple[str | None, ...] = (
        identity.bdf,
        identity.device_name,
        identity.xrt_version,
        identity.npu_driver_version,
        identity.npu_firmware_version,
        identity.processor,
    )
    if any(known):
        return identity
    return _unavailable_identity(
        "xrt-smi examine reported no device or version information. The utility "
        "exits 0 even when it rejects a request, so an empty or unrecognised "
        "report is the only available signal."
    )


def _unavailable_identity(reason: str) -> DeviceIdentity:
    return DeviceIdentity(
        bdf=None,
        device_name=None,
        xrt_version=None,
        npu_driver_version=None,
        npu_firmware_version=None,
        processor=None,
        unavailable_reason=reason,
    )


def parse_platform_report(text: str) -> PlatformTelemetry:
    """Parse ``xrt-smi examine --report platform`` into power and device shape.

    One absent field does not void the rest: a report whose power reads ``N/A``
    still yields the power mode and column count.
    """
    report = _parse_report(text)
    banner_bdf, banner_name = report.first_device()
    platform_present = report.has_section(_SECTION_PLATFORM)
    return PlatformTelemetry(
        power=_read_power(report.get(_POWER_FIELD), platform_present=platform_present),
        power_mode=report.get_in(_SECTION_PLATFORM, "Power Mode") or None,
        total_columns=_as_int(report.get_in(_SECTION_PLATFORM, "Total Columns")),
        device_name=report.get_in(_SECTION_PLATFORM, "Name") or banner_name,
        bdf=banner_bdf,
    )


def parse_partition_report(text: str) -> PartitionOccupancy:
    """Parse ``xrt-smi examine --report aie-partitions`` into live occupancy.

    Scanned line by line rather than through the section map because partition
    blocks repeat the same keys, and each ``Columns`` line belongs to the
    ``Partition Index`` above it.
    """
    lines = text.splitlines()
    if not any(line.strip() == _SECTION_AIE_PARTITIONS for line in lines):
        return PartitionOccupancy(
            context_live=None,
            partitions=(),
            unavailable_reason=(
                "xrt-smi produced no AIE partition report, so whether the NPU "
                "holds a live hardware context is unknown."
            ),
        )

    if any(_NO_CONTEXTS_SENTINEL in line.strip().lower() for line in lines):
        return PartitionOccupancy(
            context_live=False, partitions=(), unavailable_reason=None
        )

    blocks: list[tuple[int, tuple[int, ...]]] = []
    for line in lines:
        match = _FIELD.match(line.rstrip())
        if match is None:
            continue
        key = match["key"].strip()
        value = match["value"].strip()
        if key == _PARTITION_INDEX_FIELD:
            index = _as_int(value)
            if index is not None:
                blocks.append((index, ()))
        elif key == _COLUMNS_FIELD and blocks:
            blocks[-1] = (blocks[-1][0], _as_int_list(value))

    if not blocks:
        return PartitionOccupancy(
            context_live=None,
            partitions=(),
            unavailable_reason=(
                "the AIE partition report contained neither a partition block "
                "nor the idle sentinel, so liveness could not be determined."
            ),
        )
    return PartitionOccupancy(
        context_live=True,
        partitions=tuple(
            AiePartition(index=index, columns=columns) for index, columns in blocks
        ),
        unavailable_reason=None,
    )


def _resolve(candidate: Path) -> Path | None:
    """Locate the utility, without following it to a canonical path.

    ``shutil.which`` is the fallback rather than the primary route: ``xrt-smi``
    is not on ``PATH`` on this machine, so the absolute default is what
    ordinarily succeeds. The candidate is returned as given, not resolved
    through symlinks or short paths, so the argument vector a caller sees is the
    one it asked for.
    """
    if candidate.is_file():
        return candidate
    found = shutil.which(str(candidate))
    return Path(found) if found else None


@dataclass(frozen=True)
class _Invocation:
    """One completed attempt to read a report."""

    stdout: str | None
    reason: str | None = field(default=None)


class XrtSmiWrapper:
    """Typed, non-raising reads of NPU telemetry from ``xrt-smi``.

    The utility is located **once, at construction**, and each read then spends
    exactly one subprocess. That is what makes repeated polling cheap enough to
    sample power during a running workload (requirement 6.2), and it also means
    a sampling loop cannot silently change which binary it is talking to
    part-way through a measurement.

    The corollary is that a wrapper built before the driver is installed stays
    unavailable for its lifetime; construct a new one after the environment
    changes.
    """

    def __init__(
        self,
        executable: Path | str = DEFAULT_XRT_SMI_PATH,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        runner: CommandRunner | None = None,
    ) -> None:
        self._executable = Path(executable)
        self._timeout_seconds = float(timeout_seconds)
        self._runner: CommandRunner = run_xrt_smi if runner is None else runner
        self._resolved = _resolve(self._executable)

    @property
    def executable(self) -> Path:
        """The path this wrapper was asked to use."""
        return self._executable

    @property
    def resolved_executable(self) -> Path | None:
        """Where the utility was found, or ``None`` if it was not."""
        return self._resolved

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def available(self) -> bool:
        """Whether the utility was located. Never an assertion that it works."""
        return self._resolved is not None

    def read_identity(self) -> DeviceIdentity:
        """Device and version provenance (requirements 1.1, 6.6)."""
        outcome = self._invoke(_EXAMINE)
        if outcome.stdout is None:
            return _unavailable_identity(_require_reason(outcome))
        return parse_examine(outcome.stdout)

    def read_platform(self) -> PlatformTelemetry:
        """Estimated power, power mode, and column count (requirement 6.2)."""
        outcome = self._invoke(_PLATFORM_REPORT)
        if outcome.stdout is None:
            return PlatformTelemetry(
                power=PowerReading(
                    watts=None,
                    status=PowerStatus.UNAVAILABLE,
                    reason=_require_reason(outcome),
                ),
                power_mode=None,
                total_columns=None,
                device_name=None,
                bdf=None,
            )
        return parse_platform_report(outcome.stdout)

    def read_power(self) -> PowerReading:
        """One power sample. The call a polling loop makes."""
        return self.read_platform().power

    def read_partitions(self) -> PartitionOccupancy:
        """Live hardware contexts and occupied columns (requirement 1.1)."""
        outcome = self._invoke(_PARTITION_REPORT)
        if outcome.stdout is None:
            return PartitionOccupancy(
                context_live=None,
                partitions=(),
                unavailable_reason=_require_reason(outcome),
            )
        return parse_partition_report(outcome.stdout)

    def _invoke(self, arguments: Sequence[str]) -> _Invocation:
        executable = self._resolved
        if executable is None:
            return _Invocation(
                stdout=None,
                reason=(
                    f"xrt-smi was not found at {self._executable}. It ships with "
                    "the AMD NPU driver rather than the Ryzen AI SDK; on a "
                    "machine with no NPU it is absent and NPU telemetry is "
                    "simply unavailable."
                ),
            )

        argv: tuple[str, ...] = (str(executable), *arguments)
        printable = " ".join(arguments)
        try:
            result = self._runner(argv, timeout_seconds=self._timeout_seconds)
        except subprocess.TimeoutExpired:
            return _Invocation(
                stdout=None,
                reason=(
                    f"xrt-smi {printable} timed out after "
                    f"{self._timeout_seconds:g} s."
                ),
            )
        except OSError as exc:
            return _Invocation(
                stdout=None,
                reason=f"xrt-smi at {executable} could not be executed: {exc}",
            )

        if result.returncode != 0:
            detail = (result.stderr.strip() or result.stdout.strip()) or "no output"
            return _Invocation(
                stdout=None,
                reason=(
                    f"xrt-smi {printable} exited {result.returncode}: {detail}"
                ),
            )
        return _Invocation(stdout=result.stdout)


def _require_reason(outcome: _Invocation) -> str:
    """Every failed invocation carries a reason; this makes that a type."""
    if outcome.reason is None:  # pragma: no cover - construction invariant
        raise ValueError("a failed xrt-smi invocation must carry a reason")
    return outcome.reason
