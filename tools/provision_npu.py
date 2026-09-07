"""Provision the AMD Ryzen AI vendor runtime into a uv-managed virtualenv.

This is the executable form of ``docs/provisioning.md``. Running it on a machine
with the NPU driver present takes a freshly created uv environment all the way
to ``VitisAIExecutionProvider`` appearing in
``onnxruntime.get_available_providers()``, which is the observable task 1.2 and
requirement 1.6 are settled against.

Four steps are required; omitting any one leaves a broken environment:

1. Install the vendor wheels by **direct URL**. uv cannot resolve
   ``https://pypi.amd.com/simple`` - its simple-index pages deviate from the
   standard and uv reports "no versions of onnxruntime-vitisai" - but it
   installs the very same wheels cleanly when they are named by URL. They live
   in the ``npu`` dependency group so that ``uv sync`` alone never pulls them.
2. Keep NumPy below 2. The vendor build is compiled against the NumPy 1.x ABI
   and fails to import under 2.x. Pinned project-wide in ``pyproject.toml``.
3. Relocate the four native libraries the ``voe`` wheel strands. The wheel
   declares version ``1.7.0`` while its data directory is named for a longer
   development version, and that mismatch makes installers treat the payload as
   an opaque directory instead of merging it into ``onnxruntime/capi/``. Without
   these libraries the provider registers and *then* session creation dies with
   a native access violation - a failure that looks like success until it
   crashes.
4. Place the BF16 compiler and its provider configuration, which the wheels do
   not ship. Both are extracted from the Ryzen AI NuGet package; no installer is
   executed and none is required.

This machine is uv/venv only. Nothing here shells out to any other environment
manager, and the isolated fallback - were it ever built - would be a second uv
virtualenv.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where the Ryzen AI NuGet package was unpacked on the machine this recipe was
#: verified on. A parameter with a default, never a constant: another machine
#: will have unpacked it somewhere else.
#:
#: Written relative to the user's home rather than as an absolute path: this
#: repository is public, and the original literal carried the verifying machine's
#: account name into every clone. Resolves to the same directory here.
DEFAULT_NUGET_NATIVE_DIR = (
    Path.home()
    / "Downloads"
    / "ryzen_ai_nuget_1.8.0"
    / "RyzenAI_Deployment.1.8.0"
    / "runtimes"
    / "win-x64"
    / "native"
)

#: The two files the AMD-index wheels do not ship. ``vaiml.dll`` is the BF16
#: compiler the NLP encoder flow needs on Strix; ``vaip_config.json`` is the
#: ``config_file`` provider option that switches the device data type to
#: bfloat16.
NUGET_PAYLOAD: tuple[str, ...] = ("vaiml.dll", "vaip_config.json")

#: The native libraries stranded inside the ``voe`` wheel's data directory.
STRANDED_DLLS: tuple[str, ...] = (
    "onnxruntime_vitisai_ep.dll",
    "aiecompiler_client.dll",
    "dyn_dispatch_core.dll",
    "onnxruntime_vitis_ai_custom_ops.dll",
)

VENDOR_WHEEL_URLS: tuple[str, ...] = (
    "https://pypi.amd.com/packages/onnxruntime-vitisai/"
    "onnxruntime_vitisai-1.23.2-cp312-cp312-win_amd64.whl",
    "https://pypi.amd.com/packages/voe/voe-1.7.0-py3-none-win_amd64.whl",
)

REQUIRED_PROVIDER = "VitisAIExecutionProvider"

#: Distribution whose files the vendor wheel deliberately overwrites. See
#: docs/provisioning.md, "The onnxruntime collision".
REPLACED_DISTRIBUTION = "onnxruntime"

_WHEEL_FILENAME = re.compile(
    r"^(?P<distribution>.+?)-(?P<version>[^-]+?)"
    r"(?:-(?P<build>\d[^-]*))?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)

CopyReason = Literal["missing", "size-differs", "stale"]


class ProvisioningError(RuntimeError):
    """A provisioning precondition is unmet.

    Always carries what is missing and where to obtain it: continuing past one
    of these produces an environment that registers the provider and then
    crashes, or silently runs on the CPU.
    """


class MissingInterpreterError(ProvisioningError):
    """The target virtualenv has no interpreter to interrogate.

    Distinct from its parent because it is the one probe failure that
    reinstalling cannot repair: there is nothing to reinstall *into*. Provisioning
    must surface it rather than fold it into the vendor-runtime repair path.
    """


@dataclass(frozen=True)
class WheelSpec:
    """A vendor wheel identified by its filename, per PEP 427."""

    distribution: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tag: str
    url: str


@dataclass(frozen=True)
class CopyAction:
    """One file that provisioning intends to place, and why."""

    source: Path
    destination: Path
    reason: CopyReason


def wheel_spec_from_url(url: str) -> WheelSpec:
    """Parse a wheel URL into its distribution, version, and compatibility tags.

    Parsing rather than restating the metadata means a URL edited to the wrong
    interpreter or platform is caught by a test instead of by an ImportError
    after a 600 MB download.
    """
    filename = url.rsplit("/", 1)[-1]
    match = _WHEEL_FILENAME.match(filename)
    if match is None:
        raise ProvisioningError(
            f"{url!r} is not a wheel URL: {filename!r} does not parse as a wheel "
            "filename (distribution-version-python-abi-platform.whl). The vendor "
            "packages must be installed as wheels; source distributions for them "
            "are not published."
        )
    return WheelSpec(
        distribution=match["distribution"].replace("_", "-"),
        version=match["version"],
        python_tag=match["python"],
        abi_tag=match["abi"],
        platform_tag=match["platform"],
        url=url,
    )


VENDOR_WHEELS: tuple[WheelSpec, ...] = tuple(
    wheel_spec_from_url(url) for url in VENDOR_WHEEL_URLS
)


# ---------------------------------------------------------------------------
# Virtualenv layout
# ---------------------------------------------------------------------------


def venv_site_packages(venv_dir: Path) -> Path:
    """Windows virtualenv layout. The vendor wheels are ``win_amd64`` only, so
    no other layout can host them."""
    return venv_dir / "Lib" / "site-packages"


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe"


def onnxruntime_capi_dir(site_packages: Path) -> Path:
    """The directory the provider bridge and its native dependencies must share.

    ``vaiml.dll`` is loaded by bare name through the standard DLL search order,
    so it has to sit beside ``onnxruntime_vitisai_ep.dll`` rather than anywhere
    on ``PATH``.
    """
    capi = site_packages / "onnxruntime" / "capi"
    if not capi.is_dir():
        raise ProvisioningError(
            f"No onnxruntime runtime directory at {capi}. Install the vendor "
            "wheels first: `uv sync --group npu`."
        )
    return capi


def stranded_capi_dir(site_packages: Path) -> Path:
    """Locate the payload the ``voe`` wheel failed to merge.

    Globbed, never hardcoded: the data directory is named for a development
    version (``voe-1.7.0.dev...+g...``) that will change with any vendor rebuild.
    """
    matches = sorted(
        path
        for path in site_packages.glob(
            "voe-*.data/data/lib/site-packages/onnxruntime/capi"
        )
        if path.is_dir()
    )
    if not matches:
        raise ProvisioningError(
            f"No stranded `voe` payload under {site_packages}: expected a "
            "`voe-*.data/data/lib/site-packages/onnxruntime/capi` directory. "
            "Install the vendor wheels first: `uv sync --group npu`. If a "
            "future `voe` release packages its data directory correctly, this "
            "relocation step becomes unnecessary and should be removed rather "
            "than worked around."
        )
    if len(matches) > 1:
        listed = ", ".join(str(path) for path in matches)
        raise ProvisioningError(
            "Found more than one stranded `voe` payload and will not guess "
            f"which is current: {listed}. Remove the stale `voe-*.data` "
            "directory, or rebuild the environment from scratch."
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Relocation planning
# ---------------------------------------------------------------------------


def copy_reason(source: Path, destination: Path) -> CopyReason | None:
    """Why ``destination`` needs rewriting, or ``None`` when it is current.

    Size and modification time are compared rather than content: the payload is
    roughly 570 MB and hashing it on every run would cost more than the copy it
    is meant to avoid.
    """
    if not destination.exists():
        return "missing"
    source_stat = source.stat()
    destination_stat = destination.stat()
    if source_stat.st_size != destination_stat.st_size:
        return "size-differs"
    if source_stat.st_mtime > destination_stat.st_mtime:
        return "stale"
    return None


def plan_copies(sources: Sequence[Path], destination_dir: Path) -> list[CopyAction]:
    """The copies still outstanding. Empty once provisioning has been applied."""
    actions: list[CopyAction] = []
    for source in sources:
        destination = destination_dir / source.name
        reason = copy_reason(source, destination)
        if reason is not None:
            actions.append(CopyAction(source, destination, reason))
    return actions


def apply_copies(actions: Iterable[CopyAction]) -> None:
    """Place each planned file, preserving modification time.

    ``copy2`` is what makes the plan converge: a copied destination carries its
    source's mtime, so the next :func:`plan_copies` reports nothing to do.
    """
    for action in actions:
        action.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(action.source, action.destination)


def resolve_payload(source_dir: Path, filenames: Sequence[str]) -> list[Path]:
    """Resolve required vendor files, failing with what is missing and where it
    comes from."""
    if not source_dir.is_dir():
        raise ProvisioningError(
            f"Vendor payload directory not found: {source_dir}\n"
            f"Expected the Ryzen AI NuGet package unpacked so that "
            f"`RyzenAI_Deployment.<version>/runtimes/win-x64/native` exists. "
            "Download it from AMD (account.amd.com, one-time EULA acceptance); "
            "do not run the exe installer. Then pass the directory with "
            "--nuget-native-dir."
        )
    missing = [name for name in filenames if not (source_dir / name).is_file()]
    if missing:
        raise ProvisioningError(
            f"Missing: {', '.join(missing)}\n"
            f"Searched: {source_dir}\n"
            "These ship only in the Ryzen AI NuGet package, under "
            "`RyzenAI_Deployment.<version>/runtimes/win-x64/native`. They are "
            "not in any AMD-index wheel. Re-unpack the package or point "
            "--nuget-native-dir at the correct directory."
        )
    return [source_dir / name for name in filenames]


# ---------------------------------------------------------------------------
# Environment interrogation
# ---------------------------------------------------------------------------

_PROBE = (
    "import json, onnxruntime as ort; "
    "print(json.dumps({'version': ort.__version__, "
    "'providers': ort.get_available_providers()}))"
)


def available_providers(python_exe: Path) -> list[str]:
    """Execution providers the given interpreter's onnxruntime exposes.

    This is the decisive guard. Requesting a provider that is absent does *not*
    make session construction fail - onnxruntime falls through to the CPU - so
    presence in this list, not a successful session, is what proves the vendor
    runtime is live.
    """
    return _probe(python_exe)[1]


def onnxruntime_version(python_exe: Path) -> str:
    return _probe(python_exe)[0]


def _probe(python_exe: Path) -> tuple[str, list[str]]:
    if not python_exe.is_file():
        raise MissingInterpreterError(
            f"No interpreter at {python_exe}. Create the environment first: "
            "`uv sync`."
        )
    result = subprocess.run(
        [str(python_exe), "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProvisioningError(
            "Could not import onnxruntime in the target environment "
            f"({python_exe}):\n{result.stderr.strip()}\n"
            "A NumPy 2.x resolution is the usual cause - the vendor build is "
            "compiled against the NumPy 1.x ABI. Check that `numpy<2` still "
            "holds in pyproject.toml."
        )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return str(payload["version"]), [str(name) for name in payload["providers"]]


def _run(command: Sequence[str], *, cwd: Path) -> None:
    printable = " ".join(command)
    print(f"  $ {printable}")
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise ProvisioningError(
            f"Command failed with exit code {result.returncode}: {printable}"
        )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def sync_vendor_group(*, repo_root: Path) -> None:
    """Install the vendor wheels through the manifest, not ad hoc.

    Going through the ``npu`` dependency group rather than an ad hoc
    ``uv pip install`` is what makes the installation reproducible and
    lockfile-backed. It still does not make it permanent: a plain ``uv sync``
    drops the group again, which is why provisioning is re-runnable and cheap.
    """
    _run(["uv", "sync", "--group", "npu"], cwd=repo_root)


def ensure_vendor_runtime_active(python_exe: Path, *, repo_root: Path) -> list[str]:
    """Guarantee the vendor build - not the stock wheel - owns ``onnxruntime``.

    ``onnxruntime`` and ``onnxruntime-vitisai`` are separate distributions that
    ship the same import package, so within one ``uv sync`` the winner is
    whichever is unpacked last. Rather than depend on that ordering, verify the
    outcome and reinstall the vendor wheel over the top when the stock build
    won. See docs/provisioning.md, "The onnxruntime collision".

    An ``onnxruntime`` that cannot be imported at all is treated as the same
    condition rather than as a fatal error: a plain ``uv sync`` removes the
    vendor wheels and leaves the shared import package half-deleted, and
    reinstalling is exactly the repair for that too.

    A *missing interpreter* is deliberately not folded into that repair. Nothing
    can be reinstalled into an environment that does not exist, so its
    diagnostic - which names the venv and the command that creates it - must
    reach the caller instead of being demoted to a parenthetical before a
    doomed install.
    """
    try:
        providers = available_providers(python_exe)
        installed = onnxruntime_version(python_exe)
    except MissingInterpreterError:
        raise
    except ProvisioningError as error:
        providers, installed = [], f"unimportable ({error.args[0].splitlines()[0]})"
    if REQUIRED_PROVIDER in providers:
        return providers

    vitisai = next(
        spec for spec in VENDOR_WHEELS if spec.distribution == "onnxruntime-vitisai"
    )
    print(
        f"  the vendor build does not own `{REPLACED_DISTRIBUTION}` "
        f"(found: {installed}); reinstalling the vendor wheel over it"
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_exe),
            "--reinstall-package",
            vitisai.distribution,
            vitisai.url,
        ],
        cwd=repo_root,
    )
    return available_providers(python_exe)


def relocate_stranded_libraries(site_packages: Path) -> list[CopyAction]:
    """Merge the ``voe`` wheel's stranded payload into ``onnxruntime/capi``."""
    source_dir = stranded_capi_dir(site_packages)
    sources = resolve_payload(source_dir, STRANDED_DLLS)
    actions = plan_copies(sources, onnxruntime_capi_dir(site_packages))
    apply_copies(actions)
    return actions


def install_nuget_payload(site_packages: Path, native_dir: Path) -> list[CopyAction]:
    """Place the BF16 compiler and the provider configuration file.

    ``vaip_config.json`` lands beside the runtime deliberately: it keeps an
    850 KB vendor file out of the repository while leaving it resolvable at
    runtime as ``Path(onnxruntime.__file__).parent / "capi" / "vaip_config.json"``.
    """
    sources = resolve_payload(native_dir, NUGET_PAYLOAD)
    actions = plan_copies(sources, onnxruntime_capi_dir(site_packages))
    apply_copies(actions)
    return actions


def _describe(actions: Sequence[CopyAction]) -> str:
    if not actions:
        return "already current, nothing copied"
    total = sum(action.source.stat().st_size for action in actions)
    detail = ", ".join(f"{a.destination.name} ({a.reason})" for a in actions)
    return f"{len(actions)} file(s), {total / 1_048_576:.0f} MiB: {detail}"


def provision(
    *,
    venv_dir: Path,
    native_dir: Path,
    repo_root: Path,
    run_sync: bool,
) -> list[str]:
    """Run every provisioning step and return the resulting provider list."""
    site_packages = venv_site_packages(venv_dir)

    if run_sync:
        print("[1/4] Installing the vendor wheels from AMD's package index")
        sync_vendor_group(repo_root=repo_root)
    else:
        print("[1/4] Skipping wheel installation (--skip-sync)")

    python_exe = venv_python(venv_dir)
    print(f"[2/4] Confirming the vendor build owns `{REPLACED_DISTRIBUTION}`")
    ensure_vendor_runtime_active(python_exe, repo_root=repo_root)
    print(f"      onnxruntime {onnxruntime_version(python_exe)}")

    print("[3/4] Relocating the libraries stranded by the `voe` wheel")
    print(f"      {_describe(relocate_stranded_libraries(site_packages))}")

    print("[4/4] Placing the BF16 compiler and its provider configuration")
    print(f"      {_describe(install_nuget_payload(site_packages, native_dir))}")

    providers = available_providers(python_exe)
    if REQUIRED_PROVIDER not in providers:
        raise ProvisioningError(
            f"{REQUIRED_PROVIDER} did not register. Available: {providers}\n"
            "The environment is not NPU-capable and must not be treated as if "
            "it were - requesting an absent provider does not fail, it silently "
            "runs on the CPU. Check that the NPU driver is present and that "
            "`numpy<2` holds, then re-read docs/provisioning.md."
        )
    return providers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.provision_npu",
        description=(
            "Provision the AMD Ryzen AI vendor runtime into a uv-managed "
            "virtualenv, up to registered VitisAIExecutionProvider."
        ),
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=REPO_ROOT / ".venv",
        help="Virtualenv to provision (default: %(default)s).",
    )
    parser.add_argument(
        "--nuget-native-dir",
        type=Path,
        default=DEFAULT_NUGET_NATIVE_DIR,
        help=(
            "Unpacked Ryzen AI NuGet native directory holding vaiml.dll and "
            "vaip_config.json (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Do not run `uv sync --group npu`; only repair an existing venv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        providers = provision(
            venv_dir=args.venv.resolve(),
            native_dir=args.nuget_native_dir,
            repo_root=REPO_ROOT,
            run_sync=not args.skip_sync,
        )
    except ProvisioningError as error:
        print(f"\nProvisioning failed.\n{error}", file=sys.stderr)
        return 1
    print(f"\nProvisioned. Available providers: {providers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
