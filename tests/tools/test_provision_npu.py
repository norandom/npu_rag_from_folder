"""Tests for the NPU provisioning tool.

These cover the parts of provisioning that are pure logic and therefore
unit-testable: wheel-spec parsing, resolution of the ``voe`` wheel's stranded
DLL directory, relocation planning and its idempotency, and the failure message
raised when a required vendor source file is absent.

The side-effecting parts (running ``uv sync``, copying ~570 MB of native
libraries, loading the provider) are deliberately NOT faked here. Their proof is
the end-to-end check in :func:`tools.provision_npu.available_providers`, which
requirement 1.6 and task 1.2 both settle on: ``VitisAIExecutionProvider`` must
appear in ``onnxruntime.get_available_providers()`` of the provisioned
environment.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from npu_rag.embedding.environment.capability import VENDOR_PAYLOAD_FILES
from tools import provision_npu
from tools.provision_npu import (
    NUGET_PAYLOAD,
    REQUIRED_PROVIDER,
    MissingInterpreterError,
    STRANDED_DLLS,
    VENDOR_WHEELS,
    CopyAction,
    ProvisioningError,
    apply_copies,
    copy_reason,
    plan_copies,
    resolve_payload,
    stranded_capi_dir,
    venv_python,
    venv_site_packages,
    wheel_spec_from_url,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The exact directory name observed on this machine (research.md, "Second live
# probe"). Used as a fixture only - provisioning must never hardcode it.
OBSERVED_VOE_DATA_DIR = "voe-1.7.0.dev20260117193805+g019836671.data"
STRANDED_SUBPATH = "data/lib/site-packages/onnxruntime/capi"


def _make_stranded_tree(site_packages: Path, data_dir_name: str) -> Path:
    capi = site_packages / data_dir_name / STRANDED_SUBPATH
    capi.mkdir(parents=True)
    for name in STRANDED_DLLS:
        (capi / name).write_bytes(b"native-payload-" + name.encode())
    return capi


# --------------------------------------------------------------------------
# Wheel specs
# --------------------------------------------------------------------------


def test_wheel_spec_is_parsed_from_the_wheel_filename() -> None:
    spec = wheel_spec_from_url(
        "https://pypi.amd.com/packages/onnxruntime-vitisai/"
        "onnxruntime_vitisai-1.23.2-cp312-cp312-win_amd64.whl"
    )
    assert spec.distribution == "onnxruntime-vitisai"
    assert spec.version == "1.23.2"
    assert spec.python_tag == "cp312"
    assert spec.platform_tag == "win_amd64"


def test_wheel_spec_rejects_a_url_that_is_not_a_wheel() -> None:
    with pytest.raises(ProvisioningError, match="not a wheel"):
        wheel_spec_from_url("https://pypi.amd.com/packages/voe/voe-1.7.0.tar.gz")


@pytest.mark.parametrize("spec", VENDOR_WHEELS, ids=lambda s: s.distribution)
def test_vendor_wheels_target_cpython_312_on_win_amd64(spec: object) -> None:
    """The vendor wheels are ABI-specific; a wheel for another interpreter or
    platform would install and then fail at import time."""
    parsed = wheel_spec_from_url(getattr(spec, "url"))
    assert parsed.python_tag in {"cp312", "py3"}
    assert parsed.platform_tag == "win_amd64"


def test_vendor_wheels_match_the_npu_dependency_group_in_pyproject() -> None:
    """The tool and the manifest must not drift: `uv sync --group npu` is what
    actually installs these, so the URLs have to be the same ones."""
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    declared = manifest["dependency-groups"]["npu"]
    declared_urls = {entry.split("@", 1)[1].strip() for entry in declared}
    assert declared_urls == {spec.url for spec in VENDOR_WHEELS}


def test_numpy_is_pinned_below_2_for_the_vendor_abi() -> None:
    """research.md, "Second live probe": the AMD build is compiled against the
    NumPy 1.x ABI and fails to import under 2.x."""
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    assert "numpy<2" in manifest["project"]["dependencies"]


# --------------------------------------------------------------------------
# Locating the stranded payload
# --------------------------------------------------------------------------


def test_stranded_capi_dir_is_found_by_glob_not_by_hardcoded_version(
    tmp_path: Path,
) -> None:
    expected = _make_stranded_tree(tmp_path, OBSERVED_VOE_DATA_DIR)
    assert stranded_capi_dir(tmp_path) == expected


def test_stranded_capi_dir_survives_a_voe_version_bump(tmp_path: Path) -> None:
    expected = _make_stranded_tree(tmp_path, "voe-9.9.9.dev20991231000000+gdeadbee.data")
    assert stranded_capi_dir(tmp_path) == expected


def test_stranded_capi_dir_reports_actionably_when_voe_is_not_installed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProvisioningError) as excinfo:
        stranded_capi_dir(tmp_path)
    message = str(excinfo.value)
    assert "voe" in message
    assert "uv sync --group npu" in message


def test_stranded_capi_dir_refuses_to_guess_between_two_voe_payloads(
    tmp_path: Path,
) -> None:
    _make_stranded_tree(tmp_path, "voe-1.7.0.dev1+gaaa.data")
    _make_stranded_tree(tmp_path, "voe-1.8.0.dev1+gbbb.data")
    with pytest.raises(ProvisioningError, match="more than one"):
        stranded_capi_dir(tmp_path)


# --------------------------------------------------------------------------
# Relocation planning and idempotency
# --------------------------------------------------------------------------


def test_plan_copies_schedules_every_missing_file(tmp_path: Path) -> None:
    source = _make_stranded_tree(tmp_path / "site-packages", OBSERVED_VOE_DATA_DIR)
    destination = tmp_path / "capi"
    destination.mkdir()

    actions = plan_copies([source / name for name in STRANDED_DLLS], destination)

    assert [action.destination.name for action in actions] == list(STRANDED_DLLS)
    assert {action.reason for action in actions} == {"missing"}


def test_a_second_run_copies_nothing(tmp_path: Path) -> None:
    """Idempotency: ~570 MB must not be re-copied on every provisioning run."""
    source = _make_stranded_tree(tmp_path / "site-packages", OBSERVED_VOE_DATA_DIR)
    destination = tmp_path / "capi"
    destination.mkdir()
    sources = [source / name for name in STRANDED_DLLS]

    apply_copies(plan_copies(sources, destination))

    assert plan_copies(sources, destination) == []


def test_a_truncated_destination_is_recopied(tmp_path: Path) -> None:
    source = _make_stranded_tree(tmp_path / "site-packages", OBSERVED_VOE_DATA_DIR)
    destination = tmp_path / "capi"
    destination.mkdir()
    sources = [source / name for name in STRANDED_DLLS]
    apply_copies(plan_copies(sources, destination))

    victim = destination / STRANDED_DLLS[0]
    victim.write_bytes(b"truncated")

    actions = plan_copies(sources, destination)
    assert [action.destination for action in actions] == [victim]
    assert actions[0].reason == "size-differs"


def test_a_newer_source_supersedes_an_equally_sized_destination(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    destination = tmp_path / "capi"
    destination.mkdir()
    source = source_dir / "vaiml.dll"
    source.write_bytes(b"0123456789")
    (destination / "vaiml.dll").write_bytes(b"9876543210")

    stale_time = 1_000_000_000.0
    import os

    os.utime(destination / "vaiml.dll", (stale_time, stale_time))
    os.utime(source, (stale_time + 3600, stale_time + 3600))

    actions = plan_copies([source], destination)
    assert [action.reason for action in actions] == ["stale"]


def test_copy_reason_is_none_for_an_identical_pair(tmp_path: Path) -> None:
    source = tmp_path / "a.dll"
    source.write_bytes(b"payload")
    destination_dir = tmp_path / "dst"
    destination_dir.mkdir()
    apply_copies([CopyAction(source, destination_dir / "a.dll", "missing")])
    assert copy_reason(source, destination_dir / "a.dll") is None


def test_apply_copies_creates_the_destination_directory(tmp_path: Path) -> None:
    source = tmp_path / "a.dll"
    source.write_bytes(b"payload")
    destination = tmp_path / "not" / "yet" / "there" / "a.dll"
    apply_copies([CopyAction(source, destination, "missing")])
    assert destination.read_bytes() == b"payload"


# --------------------------------------------------------------------------
# Resolving the onnxruntime collision
# --------------------------------------------------------------------------


def test_a_registered_provider_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotency at the package level: no reinstall when the vendor build
    already owns `onnxruntime`."""
    commands: list[list[str]] = []
    monkeypatch.setattr(
        provision_npu, "available_providers", lambda _: [REQUIRED_PROVIDER, "CPU"]
    )
    monkeypatch.setattr(provision_npu, "onnxruntime_version", lambda _: "1.23.2.dev0")
    monkeypatch.setattr(
        provision_npu, "_run", lambda command, cwd: commands.append(list(command))
    )

    providers = provision_npu.ensure_vendor_runtime_active(
        tmp_path / "python.exe", repo_root=tmp_path
    )

    assert REQUIRED_PROVIDER in providers
    assert commands == []


def test_the_vendor_wheel_is_reinstalled_when_stock_won_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two distributions ship the same import package, so within one sync the
    winner is whichever unpacks last. Verify the outcome, repair it, re-verify."""
    commands: list[list[str]] = []
    results = iter([["AzureExecutionProvider", "CPUExecutionProvider"], [REQUIRED_PROVIDER]])
    monkeypatch.setattr(provision_npu, "available_providers", lambda _: next(results))
    monkeypatch.setattr(provision_npu, "onnxruntime_version", lambda _: "1.29.0")
    monkeypatch.setattr(
        provision_npu, "_run", lambda command, cwd: commands.append(list(command))
    )

    providers = provision_npu.ensure_vendor_runtime_active(
        tmp_path / "python.exe", repo_root=tmp_path
    )

    assert providers == [REQUIRED_PROVIDER]
    assert len(commands) == 1
    assert commands[0][:3] == ["uv", "pip", "install"]
    assert "--reinstall-package" in commands[0]
    assert "onnxruntime-vitisai" in commands[0]


def test_an_unimportable_onnxruntime_is_repaired_rather_than_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain `uv sync` drops the vendor wheels and leaves the shared import
    package half-deleted. That is the same repair, not a different failure."""
    commands: list[list[str]] = []
    calls = iter([None, [REQUIRED_PROVIDER]])

    def _providers(_: Path) -> list[str]:
        outcome = next(calls)
        if outcome is None:
            raise ProvisioningError("Could not import onnxruntime\nsecond line")
        return outcome

    monkeypatch.setattr(provision_npu, "available_providers", _providers)
    monkeypatch.setattr(
        provision_npu, "_run", lambda command, cwd: commands.append(list(command))
    )

    providers = provision_npu.ensure_vendor_runtime_active(
        tmp_path / "python.exe", repo_root=tmp_path
    )

    assert providers == [REQUIRED_PROVIDER]
    assert len(commands) == 1


def test_probe_reports_a_missing_interpreter(tmp_path: Path) -> None:
    with pytest.raises(MissingInterpreterError, match="No interpreter"):
        provision_npu.available_providers(tmp_path / "nowhere" / "python.exe")


def test_a_missing_interpreter_is_not_mistaken_for_a_lost_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing can be reinstalled into an environment that does not exist, so the
    repair path must not swallow this diagnostic."""
    commands: list[list[str]] = []
    monkeypatch.setattr(
        provision_npu, "_run", lambda command, cwd: commands.append(list(command))
    )

    with pytest.raises(MissingInterpreterError):
        provision_npu.ensure_vendor_runtime_active(
            tmp_path / "absent" / "python.exe", repo_root=tmp_path
        )

    assert commands == []


def test_a_missing_virtualenv_reports_actionably_from_the_top_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The honest-failure constraint is about what the *operator* sees, so assert
    against the real top-level output rather than against the raised error."""
    exit_code = provision_npu.main(
        ["--venv", str(tmp_path / "absent-venv"), "--skip-sync"]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Provisioning failed." in stderr
    assert "No interpreter" in stderr
    assert str(tmp_path / "absent-venv") in stderr
    assert "uv sync" in stderr
    # The failure it must NOT degrade into: an install against a venv that is
    # not there, whose exit code says nothing about the actual cause.
    assert "uv pip install" not in stderr


# --------------------------------------------------------------------------
# The final provider guard
# --------------------------------------------------------------------------


def _stub_provisioning_steps(
    monkeypatch: pytest.MonkeyPatch, *, final_providers: list[str]
) -> None:
    """Neutralize every side-effecting step so only the closing guard is live."""
    monkeypatch.setattr(provision_npu, "sync_vendor_group", lambda *, repo_root: None)
    monkeypatch.setattr(
        provision_npu,
        "ensure_vendor_runtime_active",
        lambda python_exe, *, repo_root: list(final_providers),
    )
    monkeypatch.setattr(provision_npu, "onnxruntime_version", lambda _: "1.23.2.dev0")
    monkeypatch.setattr(provision_npu, "relocate_stranded_libraries", lambda _: [])
    monkeypatch.setattr(
        provision_npu, "install_nuget_payload", lambda site_packages, native_dir: []
    )
    monkeypatch.setattr(
        provision_npu, "available_providers", lambda _: list(final_providers)
    )


def test_provisioning_fails_when_the_vendor_provider_never_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single guard between a CPU-only environment and a success report.

    requirements.md's Introduction requires that a silent degradation to CPU can
    never masquerade as success, and requesting an absent provider does not
    raise - onnxruntime just runs on the CPU. So reaching the end of
    provisioning without VitisAIExecutionProvider must be an error, never a
    warning.
    """
    _stub_provisioning_steps(
        monkeypatch,
        final_providers=["AzureExecutionProvider", "CPUExecutionProvider"],
    )

    with pytest.raises(ProvisioningError) as excinfo:
        provision_npu.provision(
            venv_dir=tmp_path / ".venv",
            native_dir=tmp_path / "native",
            repo_root=tmp_path,
            run_sync=True,
        )

    message = str(excinfo.value)
    assert REQUIRED_PROVIDER in message
    # Names what was actually available, so the operator can tell a missing
    # driver from a lost install race.
    assert "AzureExecutionProvider" in message
    assert "CPUExecutionProvider" in message


def test_provisioning_succeeds_only_by_reporting_the_registered_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not fire on a genuinely provisioned environment either."""
    registered = [REQUIRED_PROVIDER, "DmlExecutionProvider", "CPUExecutionProvider"]
    _stub_provisioning_steps(monkeypatch, final_providers=registered)

    providers = provision_npu.provision(
        venv_dir=tmp_path / ".venv",
        native_dir=tmp_path / "native",
        repo_root=tmp_path,
        run_sync=True,
    )

    assert providers == registered


# --------------------------------------------------------------------------
# Vendor payload from the Ryzen AI NuGet package
# --------------------------------------------------------------------------


def test_resolve_payload_returns_every_requested_file(tmp_path: Path) -> None:
    for name in NUGET_PAYLOAD:
        (tmp_path / name).write_bytes(b"x")
    assert resolve_payload(tmp_path, NUGET_PAYLOAD) == [
        tmp_path / name for name in NUGET_PAYLOAD
    ]


def test_resolve_payload_names_the_missing_file_and_where_to_get_it(
    tmp_path: Path,
) -> None:
    (tmp_path / "vaip_config.json").write_bytes(b"{}")
    with pytest.raises(ProvisioningError) as excinfo:
        resolve_payload(tmp_path, NUGET_PAYLOAD)
    message = str(excinfo.value)
    assert "vaiml.dll" in message
    assert "vaip_config.json" not in message.split("Missing:")[1].split("\n")[0]
    assert str(tmp_path) in message
    assert "RyzenAI_Deployment" in message


def test_resolve_payload_reports_a_missing_source_directory(tmp_path: Path) -> None:
    absent = tmp_path / "no-such-nuget-dir"
    with pytest.raises(ProvisioningError) as excinfo:
        resolve_payload(absent, NUGET_PAYLOAD)
    assert str(absent) in str(excinfo.value)


def test_bf16_payload_covers_the_compiler_and_its_provider_config() -> None:
    """research.md, "Third probe": vaiml.dll is the BF16 compiler and
    vaip_config.json is the `config_file` provider option that switches the
    device target to bfloat16. Neither ships in the AMD-index wheels."""
    assert set(NUGET_PAYLOAD) == {"vaiml.dll", "vaip_config.json"}


# --------------------------------------------------------------------------
# The forced duplication between provisioning and the capability check
#
# Task 5.4, defect 3. design.md's Out of Boundary says the package detects and
# reports and never mutates the system, so `environment/capability.py` must not
# import from `tools/`. The payload filenames are therefore written down twice.
# What is *not* forced is that the two can drift silently: this file may import
# both sides, so the relation between them is a test rather than a comment.
#
# Implementation Note 1.5 said "the two lists must be changed together". They
# had already diverged when this test was written - the check covered three
# files where provisioning installs six - and the three it omitted were the
# stranded DLLs whose absence lets the provider register and then die inside
# native code, which is the exact failure the check exists to pre-empt.
# --------------------------------------------------------------------------


def test_the_capability_check_covers_every_file_provisioning_installs() -> None:
    assert set(VENDOR_PAYLOAD_FILES) == set(STRANDED_DLLS) | set(NUGET_PAYLOAD), (
        "environment/capability.py's VENDOR_PAYLOAD_FILES and provisioning's "
        "STRANDED_DLLS + NUGET_PAYLOAD have drifted apart; a file provisioning "
        "installs that the check does not verify is a file whose absence the "
        "check will report as a healthy environment"
    )


def test_every_stranded_dll_is_verified_by_the_capability_check() -> None:
    """The half of the equality above that carries the consequence.

    ``tools/provision_npu.py``'s own module docstring on these four: "Without
    these libraries the provider registers and *then* session creation dies with
    a native access violation - a failure that looks like success until it
    crashes." A capability report that called such an environment installed
    would be worse than no report.
    """
    assert set(STRANDED_DLLS) <= set(VENDOR_PAYLOAD_FILES)
    assert len(STRANDED_DLLS) == 4


def test_the_payload_lists_name_no_file_twice() -> None:
    """Non-vacuity for the equality: it compares sets, so a duplicated entry on
    either side would not show up there while still being a mistake."""
    for names in (VENDOR_PAYLOAD_FILES, STRANDED_DLLS, NUGET_PAYLOAD):
        assert len(set(names)) == len(names)
    assert not set(STRANDED_DLLS) & set(NUGET_PAYLOAD), (
        "the provenance split is load-bearing (Note 1.2): four DLLs come from "
        "the voe wheel and two files from the Ryzen AI NuGet package"
    )


# --------------------------------------------------------------------------
# Virtualenv layout
# --------------------------------------------------------------------------


def test_venv_paths_follow_the_windows_layout(tmp_path: Path) -> None:
    assert venv_site_packages(tmp_path) == tmp_path / "Lib" / "site-packages"
    assert venv_python(tmp_path) == tmp_path / "Scripts" / "python.exe"


# --------------------------------------------------------------------------
# Recorded, reproducible form (requirement 1.6)
# --------------------------------------------------------------------------


def test_provisioning_document_records_the_verified_wheel_versions() -> None:
    """Requirement 1.6: documentation sufficient to reproduce a working
    environment from a clean machine, including the versions verified."""
    document = (REPO_ROOT / "docs" / "provisioning.md").read_text("utf-8")
    for spec in VENDOR_WHEELS:
        assert spec.distribution in document
        assert spec.version in document
    for name in STRANDED_DLLS + NUGET_PAYLOAD:
        assert name in document
    assert "VitisAIExecutionProvider" in document


def test_no_owned_file_mentions_conda() -> None:
    """User directive, recorded in requirements.md's Constraints: this machine is
    uv/venv only and conda must not be assumed anywhere."""
    owned = [REPO_ROOT / "pyproject.toml"]
    for directory in ("src", "tests", "tools", "docs"):
        owned.extend(
            path
            for path in (REPO_ROOT / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".md", ".toml", ".cfg"}
        )
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in owned
        if path != Path(__file__) and "conda" in path.read_text("utf-8").lower()
    ]
    assert offenders == []
