# Provisioning the NPU runtime

How to take a clean machine to a Python environment in which
`onnxruntime.get_available_providers()` contains `VitisAIExecutionProvider` and
the BF16 compiler is present, so the NPU can actually execute an encoder.

Satisfies requirement 1.6 (*provisioning documentation sufficient to reproduce a
working environment from a clean machine, including the driver and runtime
versions verified*).

Everything below is automated by `tools/provision_npu.py`. The prose exists so a
reader can reproduce or audit the environment without running the script, and so
the reasoning survives if the script ever has to be rewritten.

## The one command

```powershell
uv sync
uv run python -m tools.provision_npu
```

Optional arguments:

| Flag | Default | Purpose |
|---|---|---|
| `--venv` | `.venv` | Virtualenv to provision |
| `--nuget-native-dir` | `C:\Users\mariu\Downloads\ryzen_ai_nuget_1.8.0\RyzenAI_Deployment.1.8.0\runtimes\win-x64\native` | Where the Ryzen AI NuGet package was unpacked |
| `--skip-sync` | off | Repair an existing environment without re-running `uv sync --group npu` |

The run is idempotent. A second invocation copies nothing: it compares size and
modification time and reports `already current`. That matters because the
payload is roughly 570 MB.

**Re-run it after any `uv sync`.** A plain `uv sync` — which every other
canonical command implies — removes `onnxruntime-vitisai` and `voe`, because
they belong to the `npu` group and not to the default set. Worse, it leaves the
shared `onnxruntime` import package half-deleted, so the environment then has
*no* working runtime at all, vendor or stock:

```
AttributeError: module 'onnxruntime' has no attribute '__version__'
```

That is expected and recoverable; re-running the provisioner restores both. It
is called out here because the symptom looks like a corrupted install rather
than a group membership. See "The onnxruntime collision" below for why the two
distributions interfere.

## Prerequisites

| Item | Verified value on the reference machine |
|---|---|
| Hardware | AMD Ryzen AI 9 HX 370, XDNA2 NPU (`PCI\VEN_1022&DEV_17F0`), Strix Point, 8 columns |
| NPU driver | `32.0.20102.3930` (XRT 2.21.0, firmware 1.1.2.64, 2026-05-07) |
| Driver-level tooling | `xrt-smi.exe`, `pyxrt.pyd` in `C:\Windows\System32\AMD` — present without any SDK |
| OS | Windows 11 |
| Python | 3.12 (the vendor wheels are `cp312` / `win_amd64` only) |
| uv | 0.12.5 |
| Ryzen AI NuGet package | `ryzen_ai_nuget_1.8.0.zip`, unpacked; **no installer executed** |

The AMD documentation states a driver minimum of `32.0.203.280`. That string is
not comparable component-for-component with `32.0.20102.3930` — the third field
differs in width, and read component-wise the installed value is the larger.
Driver adequacy is therefore settled by whether the provider registers, not by
comparing version strings. It registers.

This machine is uv/venv only. No other environment manager is installed,
assumed, or referenced anywhere in this project. AMD's published install path
differs; it is not the path used here.

No `.exe` installer is run at any point. The one manual step is a click-through
of AMD's EULA at `account.amd.com` to download the NuGet zip, which carries the
two files in step 4.

## Verified versions

| Package | Version | Source |
|---|---|---|
| `onnxruntime-vitisai` | 1.23.2 (reports `1.23.2.dev20260117` at runtime) | `https://pypi.amd.com/packages/onnxruntime-vitisai/onnxruntime_vitisai-1.23.2-cp312-cp312-win_amd64.whl` |
| `voe` | 1.7.0 | `https://pypi.amd.com/packages/voe/voe-1.7.0-py3-none-win_amd64.whl` |
| `numpy` | 1.26.4 (`numpy<2`, pinned project-wide) | PyPI |
| Ryzen AI deployment payload | 1.8.0 | `RyzenAI_Deployment.1.8.0/runtimes/win-x64/native` in the NuGet package |

## The four steps, and why each is required

Omitting any one produces an environment that looks provisioned and is not.

### 1. Install the vendor wheels by direct URL

The provider does not exist on PyPI: `onnxruntime-vitisai` is absent there and
AMD's `voe` on PyPI is a labelled dummy. The real packages live on AMD's own
index at `https://pypi.amd.com/simple`.

**uv cannot resolve that index.** Its simple-index pages deviate from the
standard and uv reports `no versions of onnxruntime-vitisai`. uv installs the
very same wheels without complaint when they are named by **direct URL**, so
that is how the `npu` dependency group in `pyproject.toml` declares them:

```toml
npu = [
    "onnxruntime-vitisai @ https://pypi.amd.com/packages/onnxruntime-vitisai/onnxruntime_vitisai-1.23.2-cp312-cp312-win_amd64.whl",
    "voe @ https://pypi.amd.com/packages/voe/voe-1.7.0-py3-none-win_amd64.whl",
]
```

They sit in a group rather than the baseline dependency set so that a plain
`uv sync` — and any downstream consumer of this package — never pulls a
600 MB machine-specific runtime.

### 2. Keep NumPy below 2

The AMD build is compiled against the NumPy 1.x ABI. Under NumPy 2.x the import
fails outright. `numpy<2` is pinned in `pyproject.toml`'s baseline
`dependencies`, not in the `npu` group, so no resolution can reach 2.x and then
break the moment the vendor wheels are added.

### 3. Relocate the four libraries the `voe` wheel strands

The `voe` wheel declares version `1.7.0`, but its data directory is named
`voe-1.7.0.dev20260117193805+g019836671.data`. Installers compare those two
version strings, find them different, and therefore treat the data directory as
opaque payload rather than merging it into the environment. Four native
libraries, about 303 MB, are left at

```
<venv>/Lib/site-packages/voe-*.data/data/lib/site-packages/onnxruntime/capi/
```

instead of in the real `<venv>/Lib/site-packages/onnxruntime/capi/`:

| Library | Size |
|---|---|
| `onnxruntime_vitisai_ep.dll` | 175 MiB |
| `dyn_dispatch_core.dll` | 117 MiB |
| `aiecompiler_client.dll` | 9.6 MiB |
| `onnxruntime_vitis_ai_custom_ops.dll` | 1.3 MiB |

**This is the dangerous failure.** Without them the provider still registers
successfully — `get_available_providers()` looks correct — and then session
creation dies with a native access violation. Almost certainly the mechanism
behind upstream RyzenAI-SW issue #213.

The tool globs `voe-*.data`; it never hardcodes that directory name, so a vendor
version bump does not silently break provisioning. If a future `voe` release
packages its data directory correctly, delete this step rather than keep working
around a bug that no longer exists.

### 4. Place the BF16 compiler and its provider configuration

Two files ship only in the Ryzen AI NuGet package, at
`RyzenAI_Deployment.1.8.0/runtimes/win-x64/native/`:

- **`vaiml.dll`** (266 MiB) — the BF16 compiler. Strix executes NLP encoders
  through the bfloat16 flow, so without this file session creation logs
  `Cannot load vaiml.dll` at fatal level and no BF16 kernel is ever built. It is
  loaded by bare name through the standard DLL search order, so copying it
  beside `onnxruntime_vitisai_ep.dll` in `onnxruntime/capi/` is sufficient.
- **`vaip_config.json`** (850 KB) — the `config_file` provider option. This is
  what switches the device data type to bfloat16; the compiler's
  `preliminary-vaiml-pass-summary.txt` reports `Device data type` and confirms
  it took effect.

Every scriptable route to `vaiml.dll` is closed: no accessible AMD-index wheel
contains it, the `flexml` and `quark` packages on `pypi.amd.com` return 403, and
there is no unauthenticated mirror. Hence the one-time EULA click-through.

`vaip_config.json` is placed in `onnxruntime/capi/` alongside the libraries.
That keeps an 850 KB vendor file out of version control while leaving it
resolvable at runtime:

```python
from pathlib import Path
import onnxruntime
config_file = Path(onnxruntime.__file__).parent / "capi" / "vaip_config.json"
```

That path is the contract the NPU backend consumes when it builds its
`VitisAIExecutionProvider` options.

## The onnxruntime collision

`onnxruntime` and `onnxruntime-vitisai` are two distributions that both ship the
same `onnxruntime` import package. Installing both into one environment means
whichever unpacks last owns the files.

**Decision: in an NPU-provisioned environment the vendor wheel replaces stock
onnxruntime.** The vendor build is a superset — it serves
`CPUExecutionProvider` and `DmlExecutionProvider` as well — so nothing is lost,
and the CPU backend (the full-precision reference path) runs on it unchanged.

Stock `onnxruntime>=1.20` therefore stays in the baseline dependency set: a
consumer who never provisions the NPU gets a working CPU runtime from a plain
`uv sync`, and adding the `npu` group overwrites it in place.

What is *not* acceptable is depending on install order for that outcome. So
provisioning does not trust the overwrite — it **verifies and repairs**:

1. after `uv sync --group npu`, read `get_available_providers()` in the target
   environment;
2. if `VitisAIExecutionProvider` is absent, the stock wheel won the race —
   reinstall the vendor wheel over it with
   `uv pip install --reinstall-package onnxruntime-vitisai <url>`;
3. re-read the provider list, and fail loudly if it is still absent.

A residue remains: `onnxruntime-<stock-version>.dist-info` continues to claim
stock is installed while the files on disk are the vendor build. Removing that
record is not an option — it would delete files the vendor install now owns.
The one consequence is the broken state described under "The one command": when
a plain `uv sync` uninstalls `onnxruntime-vitisai`, the stock record survives
with its files gone. Provisioning treats an unimportable `onnxruntime` as the
same condition as a lost race and repairs it the same way.

Two alternatives were considered and rejected. Declaring mutually exclusive
`cpu` and `npu` dependency groups is deterministic, but dependency groups are
absent from built wheel metadata, so the published package would ship with no
runtime at all. Uninstalling stock `onnxruntime` before installing the vendor
wheel is also deterministic, but the next `uv sync` reinstalls it from the
baseline set and clobbers the vendor files again — a repair loop with no fixed
point.

## Verification

The decisive check, run automatically as the last provisioning step and
repeatable by hand:

```powershell
.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

must contain `VitisAIExecutionProvider`. Observed:

```
['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
```

Provider registration is necessary but **not** sufficient, and the distinction
matters more than it looks:

- A successful `InferenceSession` construction proves nothing. Requesting a
  provider that is not present does not raise — onnxruntime falls through to the
  CPU and returns correct-looking vectors. Never treat session creation as
  evidence.
- Registration without step 3 crashes at session creation.
- Registration without step 4 builds no BF16 kernels.

Full runtime confirmation — partition occupancy and a CPU throughput A/B —
belongs to the capability check and the benchmark, not to provisioning.

## Rebuilding from scratch

```powershell
Remove-Item -Recurse -Force .venv
uv sync
uv run python -m tools.provision_npu
.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expect roughly 600 MB of wheel downloads and 570 MB of local copying on the
first run, and near-zero work on every run after that.

## What provisioning deliberately does not do

It does not install a driver, run a vendor installer, or modify anything outside
the target virtualenv. The `npu_rag.embedding` package does not perform any of
these steps either: it *detects and reports* environment state, which is why
this tool lives in `tools/` and is excluded from the built wheel.
