"""The artifact store: compile, manifest, reuse, invalidation (task 3.3).

This is the third and fourth of design.md's four preparation stages -
acquisition, export, **compile**, verify - and the component that decides
whether any of them need to happen at all. Requirement 4.3 says preparation
artifacts are persisted; 4.4 says valid ones are reused *and the reuse is
reported*; 4.7 says artifacts produced under a different model version, input
length or provider are invalid; 8.5 says an interrupted run leaves nothing a
later run would treat as valid. Those four sentences are the whole module.

## Why an EP context file and not the EP's own cache

research.md, "Decision: Use ONNX Runtime EP context files as the preparation
artifact": Ryzen AI 1.8 offers two caching mechanisms, and only one of them can
carry this feature's requirements. The implicit Vitis AI EP cache
(``cache_dir``/``cache_key``) is keyed on a model hash, which can express
neither 4.4's *report* that artifacts were reused nor 4.7's staleness
dimensions. The explicit EP context file - ``ep.context_enable`` plus
``ep.context_file_path`` - produces a compiled snapshot this code owns, beside a
sidecar manifest that records what it was built from. AMD documents the first as
the development path and the second as the production one.

**Measured on this machine, 2026-09-06**, compiling ``all-MiniLM-L6-v2`` pinned
to batch 1 x sequence 128 (task 1.3's proven configuration) through the Vitis AI
EP, 310 seconds:

- The snapshot is **two files, not one**. With ``ep.context_embed_mode`` at
  ``0`` the EP writes ``context.onnx`` (46.9 MB) plus a sidecar
  ``context.onnx_VITISAI.bin`` (15.4 MB) carrying the compiled AIE binary. Embed
  mode ``0`` is chosen deliberately: EmbeddingGemma's exported trunk is already
  a single 1.22 GB protobuf against ONNX's 2 GB ceiling (Implementation Note
  3.2), and folding a compiled payload into it would push at that ceiling for no
  benefit.
- The sidecar is referenced **relatively**. The ``EPContext`` node records
  ``ep_cache_context = b'context.onnx_VITISAI.bin'``, a bare filename, so
  renaming the directory the two files sit in - which is exactly how publication
  works below - keeps the reference intact. Had it been absolute, the atomic
  rename this module is built on would have silently broken every artifact.
- **The compiler's diagnostics do not survive.** The cache directory the EP was
  given was empty by the time the session finished; ``preliminary-vaiml-pass
  -summary.txt``, the one authoritative partition report (research.md, third
  probe), is gone. `read_partition_share` therefore reports ``None`` on this
  flow far more often than not, and the manifest records that absence honestly
  rather than inventing a number. Task 4.3, which owns the threshold policy,
  should expect an unverifiable share here and has a second, independent signal
  available: the published ``context.onnx`` still carries the non-offloaded
  nodes beside its ``EPContext`` node, so the graph itself shows what the
  compiler kept on the CPU.

## What decides reuse

Everything in `ArtifactIdentity`, and nothing else. The comparison in
`reuse_decision` iterates ``dataclasses.fields(ArtifactIdentity)`` rather than
naming fields one by one, so a dimension added to that record participates the
moment it exists - which is what makes requirement 4.7's "any one field" a
structural property rather than a list somebody has to remember to extend.

Two manifest fields sit outside the identity on purpose:

- ``files`` describes what was written. A manifest naming a file that is not
  there describes nothing, so it still forces a rebuild - through presence, not
  comparison.
- ``observed_partition_share`` is an observation about a compilation that has
  already happened, not a claim about which model, shape, provider or toolchain
  the artifact was built for. Requirement 4.7 names no such dimension, and
  comparing it would rebuild a perfectly good artifact whenever the compiler's
  own reporting drifted. It is recorded because design.md's Physical Data Model
  lists it and because the diagnostic it comes from does not outlive
  preparation.

## What makes an interruption invisible

Everything is built in a sibling ``<name>.partial`` directory and moved into
place with a single rename once complete (8.5). A partial directory is not
merely incomplete - it is *unnamed*: the reuse check looks only at the published
path, so nothing under a temporary name can be mistaken for a prepared artifact
even if it happens to contain a manifest. Publication moves any existing
artifact aside first and restores it if the rename fails, so a failed rebuild
never destroys a working artifact.

This module sits in ``models`` in design.md's dependency direction - ``types,
errors -> reporting -> profiles -> environment -> models -> providers -> service
-> bench`` - so it reads types, errors, reporting, profiles, environment and its
two sibling preparation stages, and nothing to its right.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Protocol

from npu_rag.embedding.environment.capability import check_capability
from npu_rag.embedding.errors import EmbeddingRuntimeError, PreparationError
from npu_rag.embedding.models.acquire import (
    ACQUISITION_STAGE,
    DEFAULT_REVISION,
    AcquiredModel,
    acquire_model,
)
from npu_rag.embedding.models.export import (
    DENSE_FILENAME,
    EXPORT_STAGE,
    ONNX_FILENAME,
    ExportedModel,
    export_model,
)
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.reporting import ProgressCallback, ProgressUpdate
from npu_rag.embedding.types import CapabilityReport, ProviderChoice

__all__ = [
    "COMPILE_STAGE",
    "CONTEXT_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "PREPARATION_STAGE",
    "PUBLICATION_STAGE",
    "RYZEN_AI_DISTRIBUTION",
    "VITISAI_PROVIDER",
    "ArtifactIdentity",
    "ArtifactManifest",
    "CompiledGraph",
    "GraphCompiler",
    "ModelAcquirer",
    "ModelExporter",
    "OrtContextCompiler",
    "PreparedArtifact",
    "ReuseDecision",
    "Toolchain",
    "artifact_directory",
    "current_toolchain",
    "ensure_prepared",
    "read_partition_share",
    "reuse_decision",
]

#: The preparation stages this module reports (4.6, 8.1). ``acquisition`` and
#: ``export`` belong to tasks 3.1 and 3.2 and are re-used from there rather than
#: respelled, so an operator reading a diagnostic sees one vocabulary.
COMPILE_STAGE: Final = "compile"
PUBLICATION_STAGE: Final = "publication"
PREPARATION_STAGE: Final = "preparation"

#: design.md, Physical Data Model. ``model.onnx`` and ``dense.npz`` are task
#: 3.2's and are imported rather than restated.
CONTEXT_FILENAME: Final = "context.onnx"
MANIFEST_FILENAME: Final = "manifest.json"

#: The manifest format. A manifest written under a different number is not read
#: at all, because a field this code cannot interpret is a field it cannot
#: compare, and an uncomparable manifest must not authorise reuse (4.7).
MANIFEST_VERSION: Final = 1

#: The distribution whose version stands for "the Ryzen AI runtime" in the
#: toolchain fingerprint. ``voe`` is the vendor package that carries the Vitis
#: AI provider payload (research.md, second probe); it is queried through
#: distribution metadata because it exposes no importable version.
#:
#: ONNX Runtime's version is deliberately **not** obtained this way. Measured
#: 2026-09-06 on this machine, ``importlib.metadata.version("onnxruntime")``
#: reports ``1.29.0`` - the stale stock ``dist-info`` residue Implementation
#: Note 1.2 records, which cannot be removed without deleting vendor files -
#: while the runtime actually loaded reports ``1.23.2.dev20260117``.
#: Fingerprinting artifacts against the metadata would key them to a version
#: that is not running. The capability report reads ``__version__`` from the
#: imported module, so that is where this takes it from.
RYZEN_AI_DISTRIBUTION: Final = "voe"

#: The provider whose absence makes compilation impossible. This is design.md's
#: *guard one* used as a precondition: requesting an unregistered provider
#: succeeds and silently runs on the CPU (research.md, first probe), which here
#: would mean a "compiled" artifact that was never compiled. Guard two - reading
#: ``session.get_providers()`` back - belongs to the backend, task 4.3.
VITISAI_PROVIDER: Final = "VitisAIExecutionProvider"

#: ONNX Runtime session-configuration keys for the explicit EP context flow.
_CONTEXT_ENABLE: Final = "ep.context_enable"
_CONTEXT_FILE_PATH: Final = "ep.context_file_path"
_CONTEXT_EMBED_MODE: Final = "ep.context_embed_mode"

#: ``0`` keeps the compiled binary in a sidecar rather than folding it into the
#: context protobuf. See the module docstring.
_EMBED_MODE: Final = "0"

#: Where the vendor compiler's scratch goes. It lives *inside* the temporary
#: build directory so an interrupted run cannot leave it beside a published
#: artifact, and it is removed before the manifest is written so it is never
#: published at all.
_CACHE_DIRNAME: Final = ".compile-cache"
_CACHE_KEY: Final = "artifact"

#: The one file that reports a partition *verdict*. The console's "100.00% of
#: operations will run on AIE" and ``fail_safe_summary.json`` both report a
#: fail-safe *plan* and were measured misreporting a graph the compiler could
#: not offload at all (research.md, third probe).
_PASS_SUMMARY: Final = "preliminary-vaiml-pass-summary.txt"
_SUPPORTED_OPERATORS: Final = re.compile(
    r"supported by VAIML:\s*\d+\s*\(\s*([0-9.]+)\s*%\s*\)"
)

#: What a half-finished build is called, and what a superseded artifact is
#: called while its replacement is being moved into place. Neither name is one
#: the reuse check will ever look at.
_PARTIAL_SUFFIX: Final = ".partial"
_SUPERSEDED_SUFFIX: Final = ".superseded"

#: Preparation's steps, for progress reporting: acquire, export, compile,
#: publish.
_STEPS: Final = 4

#: What separates the two halves of a Hugging Face repository id inside a single
#: path segment. The same convention the Hub's own cache uses, so the directory
#: name stays recognisable and two models with the same short name under
#: different owners cannot collide.
_ID_SEPARATOR: Final = "--"


# --------------------------------------------------------------------------
# The toolchain fingerprint
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Toolchain:
    """The versions an artifact's validity depends on (4.7).

    Every field is optional because every one of them can genuinely be unknown:
    a machine with no NPU reports no driver version, an environment without the
    vendor wheels reports no Ryzen AI version, and a runtime that could not be
    inspected reports no ONNX Runtime version. ``None`` is recorded as ``None``
    - a guessed value would make two different environments compare equal, which
    is the one thing this record exists to prevent.
    """

    onnxruntime_version: str | None
    ryzen_ai_version: str | None
    driver_version: str | None


def _distribution_version(name: str) -> str | None:
    """The installed version of ``name``, or ``None`` if it is not installed."""
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 - an absent or broken distribution is data
        return None


def current_toolchain(*, capability: CapabilityReport | None = None) -> Toolchain:
    """Fingerprint this environment's vendor toolchain.

    ``capability`` is accepted so the report has one source (design.md,
    CapabilityChecker: "``RunContext`` for 6.6 and 7.5 is built from this
    report, so version provenance has one source") and so a caller that has
    already paid for the check does not pay again - it shells out to
    ``xrt-smi``, which a warm reuse should not have to wait for.
    """
    report = capability if capability is not None else check_capability()
    return Toolchain(
        onnxruntime_version=report.runtime_version,
        ryzen_ai_version=_distribution_version(RYZEN_AI_DISTRIBUTION),
        driver_version=report.driver_version,
    )


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactIdentity:
    """What an artifact was built for. Every field decides reuse (4.7).

    design.md's Physical Data Model lists exactly these dimensions: model id and
    revision, compiled sequence length, batch size, provider, Ryzen AI runtime
    version, driver version, ONNX Runtime version. Adding a field here adds an
    invalidation dimension automatically, because `reuse_decision` reads this
    record's fields rather than a list of names.

    ``provider`` is the string value of a `ProviderChoice`, not the enum, so a
    manifest read back from JSON is comparable without a decode step that could
    fail differently from the rest of the record.
    """

    model_id: str
    revision: str
    provider: str
    compiled_seq_len: int
    batch_size: int
    onnxruntime_version: str | None
    ryzen_ai_version: str | None
    driver_version: str | None

    def as_mapping(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_mapping(cls, mapping: object) -> ArtifactIdentity:
        if not isinstance(mapping, dict):
            raise ValueError(f"an artifact identity must be an object, got {mapping!r}")
        return cls(
            model_id=_text(mapping, "model_id"),
            revision=_text(mapping, "revision"),
            provider=_text(mapping, "provider"),
            compiled_seq_len=_number(mapping, "compiled_seq_len"),
            batch_size=_number(mapping, "batch_size"),
            onnxruntime_version=_optional_text(mapping, "onnxruntime_version"),
            ryzen_ai_version=_optional_text(mapping, "ryzen_ai_version"),
            driver_version=_optional_text(mapping, "driver_version"),
        )


@dataclass(frozen=True)
class ArtifactManifest:
    """The sidecar that makes an artifact directory self-describing.

    ``identity`` decides reuse. ``files`` is checked for presence rather than
    compared, and ``observed_partition_share`` is neither - see the module
    docstring for why each sits where it does.
    """

    identity: ArtifactIdentity
    observed_partition_share: float | None
    files: tuple[str, ...] = ()
    version: int = MANIFEST_VERSION

    def as_mapping(self) -> dict[str, object]:
        return {
            "version": self.version,
            "identity": self.identity.as_mapping(),
            "files": list(self.files),
            "observed_partition_share": self.observed_partition_share,
        }

    @classmethod
    def from_mapping(cls, mapping: object) -> ArtifactManifest:
        if not isinstance(mapping, dict):
            raise ValueError(f"a manifest must be an object, got {mapping!r}")
        version = _number(mapping, "version")
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"manifest format {version} is not {MANIFEST_VERSION}: a manifest "
                "this code cannot interpret is one it cannot compare, and an "
                "uncomparable manifest must not authorise reuse"
            )
        share = mapping.get("observed_partition_share")
        if share is not None and not isinstance(share, (int, float)):
            raise ValueError(f"observed_partition_share is not a number: {share!r}")
        listed = mapping.get("files", [])
        if not isinstance(listed, list) or any(
            not isinstance(name, str) for name in listed
        ):
            raise ValueError(f"files is not a list of names: {listed!r}")
        return cls(
            identity=ArtifactIdentity.from_mapping(mapping.get("identity")),
            observed_partition_share=None if share is None else float(share),
            files=tuple(str(name) for name in listed),
            version=version,
        )


def _text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string, got {value!r}")
    return value


def _optional_text(mapping: dict[str, Any], key: str) -> str | None:
    if key not in mapping:
        raise ValueError(f"{key} is absent; absence and null are different facts")
    value = mapping[key]
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{key} must be a string or null, got {value!r}")


def _number(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return value


# --------------------------------------------------------------------------
# The reuse decision
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReuseDecision:
    """Whether an artifact may be reused, and why not when it may not.

    ``reason`` is always populated. Requirement 4.4 requires the runtime to
    report that it reused artifacts, and an operator who expected a warm run and
    got a cold one needs the same courtesy in the other direction - the field
    that differed, by name.
    """

    reusable: bool
    reason: str


def reuse_decision(
    stored: ArtifactManifest | None, expected: ArtifactIdentity
) -> ReuseDecision:
    """Compare a stored manifest against the identity the caller is asking for.

    Pure and total: it touches no filesystem and never raises, so the whole of
    requirement 4.7 can be examined as a value. The fields it compares come from
    ``dataclasses.fields(ArtifactIdentity)``, which is what makes "any one field
    invalidates" a property of the record rather than of this function's memory.
    """
    if stored is None:
        return ReuseDecision(
            reusable=False,
            reason="no readable manifest describes this artifact directory",
        )
    differing = [
        f"{field.name} ({was!r} != {now!r})"
        for field, was, now in (
            (
                field,
                getattr(stored.identity, field.name),
                getattr(expected, field.name),
            )
            for field in fields(ArtifactIdentity)
        )
        if was != now
    ]
    if differing:
        return ReuseDecision(
            reusable=False,
            reason="the stored artifact was prepared under a different "
            + ", ".join(differing),
        )
    return ReuseDecision(
        reusable=True, reason="reused the prepared artifact; its manifest matches"
    )


def _load_manifest(path: Path) -> ArtifactManifest | None:
    """The manifest at ``path``, or ``None`` if there is not a usable one there.

    Every way of failing to read it collapses to ``None``, because they all mean
    the same thing to the caller: nothing here authorises reuse. An exception
    would make an absent artifact and a corrupt one look different to code that
    must treat them identically.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ArtifactManifest.from_mapping(json.loads(text))
    except (ValueError, TypeError):
        return None


def _inspect(
    directory: Path, stored: ArtifactManifest | None, expected: ArtifactIdentity
) -> ReuseDecision:
    """Whether the manifest already read from ``directory`` authorises reuse.

    Staleness is decided by the manifest rather than by file presence (design.md,
    "Preparation and artifact reuse"), but a manifest describing files that are
    not there describes nothing - so presence is checked *after* the comparison
    and reported in the same vocabulary.
    """
    decision = reuse_decision(stored, expected)
    if not decision.reusable or stored is None:
        return decision
    missing = [name for name in stored.files if not (directory / name).is_file()]
    if missing:
        return ReuseDecision(
            reusable=False,
            reason="the manifest describes files that are not there: "
            + ", ".join(sorted(missing)),
        )
    return decision


# --------------------------------------------------------------------------
# Where artifacts live
# --------------------------------------------------------------------------


def artifact_directory(
    root: Path, profile: ModelProfile, provider: ProviderChoice
) -> Path:
    """design.md's ``artifacts/<model-id>/<provider>/<compiled-seq-len>/``.

    The model id becomes one path segment rather than two: a repository id
    carries a slash, and letting it nest would make the layout's depth depend on
    the model. ``/`` becomes ``--``, the convention the Hugging Face cache
    itself uses, so two models sharing a short name under different owners still
    get different directories.
    """
    return (
        root
        / _model_directory(profile.model_id)
        / provider.value
        / str(profile.compiled_seq_len)
    )


def _model_directory(model_id: str) -> str:
    return model_id.replace("\\", "/").replace("/", _ID_SEPARATOR)


# --------------------------------------------------------------------------
# The compilation seam
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledGraph:
    """What one compilation produced.

    ``partition_share`` is a fraction between 0 and 1, or ``None`` when the
    compiler's diagnostics could not be read - which is the common case on the
    EP context flow (see the module docstring). ``None`` means *unverified*,
    never *zero*: a measured zero is a measurement, and errors.py's
    `PartitionShareTooLow` keeps the two apart for exactly this reason.
    """

    context_path: Path
    partition_share: float | None


class GraphCompiler(Protocol):
    """The one thing preparation needs from the NPU compiler.

    A seam for the same reason acquisition and export have one: a real
    compilation of a real encoder took 310 seconds on this machine and needs
    hardware, so every unit test drives the lifecycle through a stand-in and
    exactly one live test uses the real implementation.
    """

    def compile(
        self,
        *,
        graph: Path,
        context: Path,
        profile: ModelProfile,
        cache_dir: Path,
    ) -> CompiledGraph:
        """Compile ``graph`` for the NPU, writing the snapshot to ``context``."""
        ...


class OrtContextCompiler:
    """`GraphCompiler` backed by ONNX Runtime's EP context mechanism.

    Constructing one touches nothing: ONNX Runtime is imported inside `compile`
    so preparation for the CPU provider - which never compiles - works in an
    environment that has never seen the vendor wheels.
    """

    def __init__(self, *, config_file: Path | None = None) -> None:
        self._config_file = config_file

    def compile(
        self,
        *,
        graph: Path,
        context: Path,
        profile: ModelProfile,
        cache_dir: Path,
    ) -> CompiledGraph:
        # ``onnxruntime`` ships no annotations, so strict mode sees an untyped
        # import; the ignore is scoped to this line rather than to the module.
        import onnxruntime as ort  # type: ignore[import-untyped]

        available = tuple(str(name) for name in ort.get_available_providers())
        if VITISAI_PROVIDER not in available:
            raise PreparationError(
                f"{VITISAI_PROVIDER} is not registered in this interpreter, so "
                f"nothing here can compile {profile.model_id} for the NPU. "
                f"Available providers: {', '.join(available) or 'none'}. "
                "Requesting an absent provider from ONNX Runtime succeeds and "
                "runs on the CPU, so this is checked before a session is built "
                "rather than after",
                provider=ProviderChoice.NPU,
                model_id=profile.model_id,
                stage=COMPILE_STAGE,
            )

        config_file = self._config_file
        if config_file is None:
            config_file = Path(ort.__file__).parent / "capi" / "vaip_config.json"
        if not config_file.is_file():
            raise PreparationError(
                f"the Vitis AI configuration file {config_file} is missing. It is "
                "the `config_file` provider option that switches the device data "
                "type to bfloat16; without it the NLP flow this feature needs "
                "does not engage",
                provider=ProviderChoice.NPU,
                model_id=profile.model_id,
                stage=COMPILE_STAGE,
            )

        options = ort.SessionOptions()
        options.add_session_config_entry(_CONTEXT_ENABLE, "1")
        options.add_session_config_entry(_CONTEXT_FILE_PATH, str(context))
        options.add_session_config_entry(_CONTEXT_EMBED_MODE, _EMBED_MODE)
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            ort.InferenceSession(
                str(graph),
                options,
                providers=[VITISAI_PROVIDER],
                provider_options=[
                    {
                        "config_file": str(config_file),
                        "cache_dir": str(cache_dir),
                        "cache_key": _CACHE_KEY,
                    }
                ],
            )
        except Exception as error:
            raise PreparationError(
                f"compiling {profile.model_id} at batch {profile.batch_size} x "
                f"sequence {profile.compiled_seq_len} for the NPU failed: "
                f"{type(error).__name__}: {error}",
                provider=ProviderChoice.NPU,
                model_id=profile.model_id,
                stage=COMPILE_STAGE,
            ) from None

        # Session construction is not evidence. The snapshot on disk is: a graph
        # the compiler could not offload produces diagnostics and no binary
        # (research.md, third probe), so an absent context file means nothing
        # was compiled however cleanly the session came back.
        if not context.is_file():
            raise PreparationError(
                f"compiling {profile.model_id} produced no context snapshot at "
                f"{context.name}. A session that constructs without writing one "
                "has not compiled anything, and treating that as success is how "
                "an uncompiled model reaches the NPU backend",
                provider=ProviderChoice.NPU,
                model_id=profile.model_id,
                stage=COMPILE_STAGE,
            )

        return CompiledGraph(
            context_path=context, partition_share=read_partition_share(cache_dir)
        )


def read_partition_share(cache_dir: Path) -> float | None:
    """The fraction of operators VAIML supported, if the compiler said.

    Reads ``preliminary-vaiml-pass-summary.txt`` and nothing else. research.md's
    third probe measured the two obvious alternatives - the console's
    ``100.00% of operations will run on AIE`` and
    ``vaiml_partition_fe.flexml/fail_safe_summary.json`` - both reporting 100%
    AIE for a graph with *zero* supported operators, because both describe a
    fail-safe partition plan rather than a support verdict.

    Returns ``None`` when the file is absent or unparseable, which on the EP
    context flow is the usual outcome: the cache directory is emptied by the
    time the session finishes. ``None`` is not zero, and this never raises -
    absence of a diagnostic is data, and task 4.3 decides what to do about it.
    """
    try:
        candidates = sorted(cache_dir.rglob(_PASS_SUMMARY))
    except OSError:
        return None
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = _SUPPORTED_OPERATORS.search(text)
        if match is None:
            continue
        try:
            return float(match.group(1)) / 100.0
        except ValueError:  # pragma: no cover - the pattern only matches numbers
            continue
    return None


# --------------------------------------------------------------------------
# The acquisition and export seams
# --------------------------------------------------------------------------


class ModelAcquirer(Protocol):
    """Task 3.1's `acquire_model`, narrowed to what preparation asks of it."""

    def __call__(
        self,
        profile: ModelProfile,
        *,
        revision: str | None,
        progress: ProgressCallback | None,
    ) -> AcquiredModel: ...


class ModelExporter(Protocol):
    """Task 3.2's `export_model`, narrowed the same way."""

    def __call__(
        self,
        profile: ModelProfile,
        acquired: AcquiredModel,
        destination: Path,
        *,
        progress: ProgressCallback | None,
    ) -> ExportedModel: ...


def _default_acquirer(
    profile: ModelProfile,
    *,
    revision: str | None,
    progress: ProgressCallback | None,
) -> AcquiredModel:
    return acquire_model(
        profile, revision=revision or DEFAULT_REVISION, progress=progress
    )


def _default_exporter(
    profile: ModelProfile,
    acquired: AcquiredModel,
    destination: Path,
    *,
    progress: ProgressCallback | None,
) -> ExportedModel:
    return export_model(profile, acquired, destination, progress=progress)


# --------------------------------------------------------------------------
# What preparation produces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedArtifact:
    """A model ready for a provider, and how it got that way.

    ``reused`` is requirement 4.4's report: the runtime reused valid artifacts
    instead of repeating preparation, and says so. ``elapsed_seconds`` is the
    corroborating measurement - a warm run does no acquisition, no export and no
    compilation, so it costs a manifest read.
    """

    directory: Path
    onnx_path: Path
    context_path: Path | None
    dense_path: Path | None
    manifest: ArtifactManifest
    reused: bool
    reason: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        for label, path in (
            ("onnx_path", self.onnx_path),
            ("context_path", self.context_path),
            ("dense_path", self.dense_path),
        ):
            if path is not None and not path.is_file():
                raise ValueError(
                    f"{label} {path} does not exist: a prepared artifact reports "
                    "where its files are, so a missing one is a failed "
                    "preparation wearing a success"
                )
        if not self.reason:
            raise ValueError(
                "a preparation states why it reused or rebuilt: requirement 4.4 "
                "is a reporting requirement, not only a caching one"
            )


# --------------------------------------------------------------------------
# Failure mapping and progress
# --------------------------------------------------------------------------


def _guarded[T](
    action: Callable[[], T],
    *,
    stage: str,
    profile: ModelProfile,
    provider: ProviderChoice,
) -> T:
    """Run one step, translating any failure it raises into this vocabulary.

    ``Exception``, not ``BaseException``: an operator interrupting a five-minute
    compilation has not encountered a preparation failure, and dressing their
    ``KeyboardInterrupt`` up as one would be a worse report than the interrupt
    itself. Requirement 8.5's guarantee does not depend on catching it - the
    temporary directory is removed in a ``finally``, which runs for an interrupt
    too, and even a killed process leaves only a directory under a name the
    reuse check never reads.
    """
    try:
        return action()
    except EmbeddingRuntimeError:
        # Already diagnosed in this feature's vocabulary, and by whichever stage
        # actually failed. Re-diagnosing would bury a licensing gate under a
        # generic compile failure.
        raise
    except Exception as error:
        raise PreparationError(
            f"preparing {profile.model_id} for {provider.value} failed at the "
            f"{stage} stage: {type(error).__name__}: {error}",
            provider=provider,
            model_id=profile.model_id,
            stage=stage,
        ) from None


def _emit(
    progress: ProgressCallback | None, operation: str, completed: int
) -> None:
    if progress is None:
        return
    progress(
        ProgressUpdate(operation=operation, completed=completed, total=_STEPS)
    )


def _remove(path: Path) -> None:
    """Remove a directory, and never fail while doing it."""
    shutil.rmtree(path, ignore_errors=True)
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _sweep(directory: Path) -> None:
    """Clear the working names beside ``directory``.

    Anything left under them is the residue of a run that did not finish -
    including one that was killed outright, where no ``finally`` ran. It is
    never *accepted* (the reuse check reads only the published name), but it is
    removed rather than accumulated.
    """
    for suffix in (_PARTIAL_SUFFIX, _SUPERSEDED_SUFFIX):
        _remove(directory.parent / f"{directory.name}{suffix}")


def _publish(temporary: Path, final: Path) -> None:
    """Move a complete build into place, atomically (8.5).

    An existing artifact is moved aside first rather than deleted, and restored
    if the move fails, so a rebuild that goes wrong at the last step does not
    destroy the artifact it was replacing. At no instant does the published name
    refer to a half-built directory.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    superseded = final.parent / f"{final.name}{_SUPERSEDED_SUFFIX}"
    _remove(superseded)
    displaced = False
    if final.exists():
        os.replace(final, superseded)
        displaced = True
    try:
        os.replace(temporary, final)
    except OSError:
        if displaced:
            os.replace(superseded, final)
        raise
    if displaced:
        _remove(superseded)


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


def ensure_prepared(
    profile: ModelProfile,
    provider: ProviderChoice,
    root: Path,
    *,
    toolchain: Toolchain | None = None,
    acquirer: ModelAcquirer | None = None,
    exporter: ModelExporter | None = None,
    compiler: GraphCompiler | None = None,
    expected_revision: str | None = None,
    progress: ProgressCallback | None = None,
) -> PreparedArtifact:
    """Make ``profile`` ready for ``provider``, reusing artifacts where valid.

    Returns immediately with ``reused=True`` when the artifact directory carries
    a manifest matching this profile, this provider and this toolchain, and
    every file that manifest describes is present (4.3, 4.4). Otherwise it
    acquires, exports, compiles and publishes, and the artifact it replaces
    survives untouched until the replacement is complete (8.5).

    ``expected_revision`` is how a caller pins a commit. Supplied, it is
    requested from the repository, compared against the manifest, and a
    repository that answers with a different commit is a failure rather than a
    substitution. Omitted, the artifact's own recorded revision is accepted:
    demanding the upstream revision on every warm run would mean a network round
    trip, which is precisely what 4.4's near-zero reuse forbids. The revision is
    recorded either way, so a silently changed upstream model stays detectable
    by inspection (design.md, Security Considerations).

    Raises `PreparationError` naming the stage that failed (4.6, 8.1), and never
    substitutes a different model or provider - including when the repository
    returns a model or a commit other than the one asked for.
    """
    started = time.perf_counter()
    if provider is ProviderChoice.AUTO:
        raise PreparationError(
            f"{ProviderChoice.AUTO.value!r} is a request, not a provider an "
            "artifact can be prepared for: it names no compute unit, so no "
            "manifest could record which one produced these files (4.7). "
            f"Resolve it to {ProviderChoice.NPU.value!r} or "
            f"{ProviderChoice.CPU.value!r} first",
            model_id=profile.model_id,
            stage=PREPARATION_STAGE,
        )

    resolved_toolchain = (
        toolchain if toolchain is not None else current_toolchain()
    )
    acquire = acquirer if acquirer is not None else _default_acquirer
    export = exporter if exporter is not None else _default_exporter
    compile_graph = compiler if compiler is not None else OrtContextCompiler()

    directory = artifact_directory(root, profile, provider)
    operation = f"prepare:{profile.name}"
    _emit(progress, operation, 0)

    stored = _load_manifest(directory / MANIFEST_FILENAME)
    wanted = _identity(
        profile,
        provider,
        resolved_toolchain,
        revision=_revision_to_compare(expected_revision, stored),
    )
    decision = _inspect(directory, stored, wanted)
    if decision.reusable and stored is not None:
        _emit(progress, operation, _STEPS)
        return _published(
            directory,
            profile,
            provider,
            stored,
            reused=True,
            reason=decision.reason,
            started=started,
        )

    _sweep(directory)
    temporary = directory.parent / f"{directory.name}{_PARTIAL_SUFFIX}"
    try:
        temporary.mkdir(parents=True, exist_ok=True)

        acquired = _guarded(
            lambda: acquire(profile, revision=expected_revision, progress=progress),
            stage=ACQUISITION_STAGE,
            profile=profile,
            provider=provider,
        )
        _refuse_substitution(acquired, profile, provider, expected_revision)
        _emit(progress, operation, 1)

        exported = _guarded(
            lambda: export(profile, acquired, temporary, progress=progress),
            stage=EXPORT_STAGE,
            profile=profile,
            provider=provider,
        )
        _emit(progress, operation, 2)

        share: float | None = None
        if provider is ProviderChoice.NPU:
            cache_dir = temporary / _CACHE_DIRNAME
            compiled = _guarded(
                lambda: compile_graph.compile(
                    graph=exported.onnx_path,
                    context=temporary / CONTEXT_FILENAME,
                    profile=profile,
                    cache_dir=cache_dir,
                ),
                stage=COMPILE_STAGE,
                profile=profile,
                provider=provider,
            )
            if not compiled.context_path.is_file():
                raise PreparationError(
                    f"compiling {profile.model_id} for the NPU reported success "
                    f"but wrote no {CONTEXT_FILENAME}. The compiled snapshot on "
                    "disk is the evidence that compilation happened; a clean "
                    "return without one is not",
                    provider=provider,
                    model_id=profile.model_id,
                    stage=COMPILE_STAGE,
                )
            share = compiled.partition_share
            # Scratch, not an artifact: removed before the manifest lists what
            # is in the directory, so it is never published.
            _remove(cache_dir)
        _emit(progress, operation, 3)

        manifest = ArtifactManifest(
            identity=_identity(
                profile, provider, resolved_toolchain, revision=acquired.revision
            ),
            observed_partition_share=share,
            files=tuple(
                sorted(entry.name for entry in temporary.iterdir() if entry.is_file())
            ),
        )
        _guarded(
            lambda: (temporary / MANIFEST_FILENAME).write_text(
                json.dumps(manifest.as_mapping(), indent=2, sort_keys=True),
                encoding="utf-8",
            ),
            stage=PUBLICATION_STAGE,
            profile=profile,
            provider=provider,
        )
        _guarded(
            lambda: _publish(temporary, directory),
            stage=PUBLICATION_STAGE,
            profile=profile,
            provider=provider,
        )
    finally:
        # After a success the directory has been renamed away and this is a
        # no-op; after a failure - or an interrupt - it is what makes a
        # half-built artifact invisible to the next run (8.5).
        _remove(temporary)

    _emit(progress, operation, _STEPS)
    return _published(
        directory,
        profile,
        provider,
        manifest,
        reused=False,
        reason=decision.reason,
        started=started,
    )


def _identity(
    profile: ModelProfile,
    provider: ProviderChoice,
    toolchain: Toolchain,
    *,
    revision: str,
) -> ArtifactIdentity:
    return ArtifactIdentity(
        model_id=profile.model_id,
        revision=revision,
        provider=provider.value,
        compiled_seq_len=profile.compiled_seq_len,
        batch_size=profile.batch_size,
        onnxruntime_version=toolchain.onnxruntime_version,
        ryzen_ai_version=toolchain.ryzen_ai_version,
        driver_version=toolchain.driver_version,
    )


def _revision_to_compare(
    expected_revision: str | None, stored: ArtifactManifest | None
) -> str:
    """Which revision the reuse comparison should hold the artifact to.

    A pinned revision is compared. An unpinned one adopts whatever the artifact
    records, which is what keeps a warm run off the network; the recorded value
    is still there to be read, so nothing is hidden by not comparing it.
    """
    if expected_revision is not None:
        return expected_revision
    return stored.identity.revision if stored is not None else ""


def _refuse_substitution(
    acquired: AcquiredModel,
    profile: ModelProfile,
    provider: ProviderChoice,
    expected_revision: str | None,
) -> None:
    """Requirement 4.6, at the two points a substitution could pass unnoticed."""
    if acquired.model_id != profile.model_id:
        raise PreparationError(
            f"asked to prepare {profile.model_id} but the repository returned "
            f"{acquired.model_id}. Requirement 4.6 forbids substituting a "
            "different model",
            provider=provider,
            model_id=profile.model_id,
            stage=ACQUISITION_STAGE,
        )
    if expected_revision is not None and acquired.revision != expected_revision:
        raise PreparationError(
            f"asked for {profile.model_id} at {expected_revision} but the "
            f"repository returned {acquired.revision}. A caller that pinned a "
            "commit and silently got another would have exactly the "
            "undetectable upstream change the manifest exists to expose",
            provider=provider,
            model_id=profile.model_id,
            stage=ACQUISITION_STAGE,
        )


def _published(
    directory: Path,
    profile: ModelProfile,
    provider: ProviderChoice,
    manifest: ArtifactManifest,
    *,
    reused: bool,
    reason: str,
    started: float,
) -> PreparedArtifact:
    context_path = (
        directory / CONTEXT_FILENAME if provider is ProviderChoice.NPU else None
    )
    dense_path = directory / DENSE_FILENAME if profile.has_dense_stage else None
    try:
        return PreparedArtifact(
            directory=directory,
            onnx_path=directory / ONNX_FILENAME,
            context_path=context_path,
            dense_path=dense_path,
            manifest=manifest,
            reused=reused,
            reason=reason,
            elapsed_seconds=time.perf_counter() - started,
        )
    except ValueError as error:
        raise PreparationError(
            f"preparing {profile.model_id} for {provider.value} did not produce "
            f"a usable artifact: {error}",
            provider=provider,
            model_id=profile.model_id,
            stage=PUBLICATION_STAGE,
        ) from None
