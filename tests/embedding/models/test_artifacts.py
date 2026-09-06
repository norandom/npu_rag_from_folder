"""The artifact store: compile, manifest, reuse, invalidation (task 3.3).

**Nothing here compiles anything.** NPU compilation of a real encoder takes
minutes and needs hardware; every unit below drives the lifecycle through an
injected `GraphCompiler`, exactly as task 3.2 drove the export through an
injected `TrunkExporter` and task 3.1 drove acquisition through an injected
repository client. The one real compilation lives in ``test_artifacts_live.py``.

Three properties carry requirement 4.7 and design.md's Physical Data Model, and
each is tested structurally rather than by example:

- **Every identity field invalidates on its own.** The sweep below is
  parametrised from ``dataclasses.fields(ArtifactIdentity)`` rather than from a
  hand-written list, so an identity field added later that nothing compares
  fails this suite the moment it appears.
- **A warm run reuses and says so**, and does no acquisition, no export and no
  compilation while doing it - which is a stronger statement than a stopwatch,
  and the stopwatch is asserted too.
- **Nothing partial is ever accepted.** The abrupt-kill case is a real killed
  child process, not an exception: an exception unwinds ``finally`` blocks and
  would prove only that the cleanup path works, which is not what requirement
  8.5 promises.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import subprocess
import sys
import textwrap
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from npu_rag.embedding.errors import (
    EmbeddingRuntimeError,
    LicenseAcceptanceRequired,
    PreparationError,
)
from npu_rag.embedding.models.acquire import AcquiredModel
from npu_rag.embedding.models.artifacts import (
    COMPILE_STAGE,
    CONTEXT_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    PUBLICATION_STAGE,
    RYZEN_AI_DISTRIBUTION,
    ArtifactIdentity,
    ArtifactManifest,
    CompiledGraph,
    OrtContextCompiler,
    PreparedArtifact,
    Toolchain,
    artifact_directory,
    current_toolchain,
    ensure_prepared,
    read_partition_share,
    reuse_decision,
)
from npu_rag.embedding.models.export import DENSE_FILENAME, ONNX_FILENAME
from npu_rag.embedding.profiles import ModelProfile, profile_for
from npu_rag.embedding.reporting import ProgressUpdate
from npu_rag.embedding.types import (
    CapabilityReport,
    ExecutionMode,
    ProviderChoice,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "npu_rag"
    / "embedding"
    / "models"
    / "artifacts.py"
)
MODULE_PACKAGE = "npu_rag.embedding.models"

# design.md -> Architecture -> dependency direction:
# types, errors -> reporting -> profiles -> environment -> models -> providers
# -> service -> bench
LAYERS_RIGHT_OF_MODELS = ("providers", "service", "bench")

GEMMA = profile_for("embeddinggemma-300m")
BGE = profile_for("bge-large-en-v1.5")

REVISION = "a" * 40
OTHER_REVISION = "b" * 40

TOOLCHAIN = Toolchain(
    onnxruntime_version="1.23.2.dev20260117",
    ryzen_ai_version="1.7.0",
    driver_version="32.0.20102.3930",
)


# --------------------------------------------------------------------------
# Layer guard
# --------------------------------------------------------------------------


def absolute_imports_of(source: str, package: str) -> list[str]:
    """Every name ``source`` imports, as an absolute dotted path.

    Relative imports are resolved rather than skipped, so ``from .. import
    providers`` is recognised as the same violation as
    ``import npu_rag.embedding.providers``.
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
            for layer in LAYERS_RIGHT_OF_MODELS
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
    imported = absolute_imports_of(statement, MODULE_PACKAGE)

    assert any(
        name.startswith(f"npu_rag.embedding.{layer}")
        for name in imported
        for layer in LAYERS_RIGHT_OF_MODELS
    )


def test_the_package_never_prints_or_logs() -> None:
    """design.md, Monitoring: progress is a callback so callers choose
    presentation. A library that printed would take that choice away."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    imported = absolute_imports_of(
        MODULE_PATH.read_text(encoding="utf-8"), MODULE_PACKAGE
    )

    assert "print" not in called
    assert "logging" not in imported


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class FakeAcquirer:
    """Stands in for `acquire_model`, which downloads gigabytes."""

    def __init__(
        self, source: Path, *, revision: str = REVISION, error: Exception | None = None
    ) -> None:
        self.source = source
        self.revision = revision
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    def __call__(
        self,
        profile: ModelProfile,
        *,
        revision: str | None,
        progress: Any = None,
    ) -> AcquiredModel:
        self.calls.append((profile.model_id, revision))
        if self.error is not None:
            raise self.error
        return AcquiredModel(
            model_id=profile.model_id,
            revision=revision or self.revision,
            local_path=self.source,
        )


class FakeExporter:
    """Stands in for `export_model`, which needs torch and minutes of tracing.

    It writes real files, because publication renames a real directory and a
    test that pretended otherwise would not exercise it.
    """

    def __init__(self, *, error: Exception | None = None, delay: float = 0.0) -> None:
        self.error = error
        self.delay = delay
        self.calls: list[Path] = []

    def __call__(
        self,
        profile: ModelProfile,
        acquired: AcquiredModel,
        destination: Path,
        *,
        progress: Any = None,
    ) -> Any:
        from npu_rag.embedding.models.export import ExportedModel

        self.calls.append(destination)
        if self.error is not None:
            raise self.error
        if self.delay:
            time.sleep(self.delay)
        destination.mkdir(parents=True, exist_ok=True)
        onnx_path = destination / ONNX_FILENAME
        onnx_path.write_bytes(b"onnx-graph-bytes")
        dense_path: Path | None = None
        if profile.has_dense_stage:
            dense_path = destination / DENSE_FILENAME
            with dense_path.open("wb") as handle:
                np.savez(handle, order=np.array(["dense"]))
        return ExportedModel(
            model_id=profile.model_id,
            revision=acquired.revision,
            onnx_path=onnx_path,
            dense_path=dense_path,
            batch_size=profile.batch_size,
            compiled_seq_len=profile.compiled_seq_len,
            hidden_size=profile.dimension,
        )


class FakeCompiler:
    """Stands in for the Vitis AI EP context compile.

    ``produces`` exists because the presence of a compiled binary is itself the
    evidence that compilation happened (research.md, third probe): a compiler
    that returns cleanly having written nothing is the failure mode this seam
    has to be able to express.
    """

    def __init__(
        self,
        *,
        share: float | None = 0.97,
        error: BaseException | None = None,
        produces: bool = True,
        delay: float = 0.0,
    ) -> None:
        self.share = share
        self.error = error
        self.produces = produces
        self.delay = delay
        self.calls: list[Path] = []

    def compile(
        self,
        *,
        graph: Path,
        context: Path,
        profile: ModelProfile,
        cache_dir: Path,
    ) -> CompiledGraph:
        self.calls.append(graph)
        if self.error is not None:
            raise self.error
        if self.delay:
            time.sleep(self.delay)
        # The real compiler is given a scratch directory and fills it. It is
        # written here too, because a scratch directory that reached the
        # published artifact would be indistinguishable from one of its files.
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "compiler-scratch.log").write_text("noise", encoding="utf-8")
        if self.produces:
            context.write_bytes(b"compiled-context-bytes")
            # ``ep.context_embed_mode`` 0 splits the snapshot in two; the real
            # EP writes the compiled binary beside the graph.
            context.with_name(f"{context.name}_VITISAI.bin").write_bytes(b"aie")
        return CompiledGraph(context_path=context, partition_share=self.share)


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    path = tmp_path / "weights"
    path.mkdir()
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def prepare(
    root: Path,
    source: Path,
    *,
    profile: ModelProfile = BGE,
    provider: ProviderChoice = ProviderChoice.NPU,
    toolchain: Toolchain = TOOLCHAIN,
    acquirer: FakeAcquirer | None = None,
    exporter: FakeExporter | None = None,
    compiler: FakeCompiler | None = None,
    revision: str | None = REVISION,
    progress: Any = None,
) -> PreparedArtifact:
    return ensure_prepared(
        profile,
        provider,
        root,
        toolchain=toolchain,
        acquirer=acquirer or FakeAcquirer(source),
        exporter=exporter or FakeExporter(),
        compiler=compiler or FakeCompiler(),
        expected_revision=revision,
        progress=progress,
    )


# --------------------------------------------------------------------------
# 4.3: preparation artifacts are persisted, where design.md says
# --------------------------------------------------------------------------


def test_preparation_lands_in_the_directory_the_physical_data_model_prescribes(
    root: Path, source_dir: Path
) -> None:
    result = prepare(root, source_dir)

    expected = artifact_directory(root, BGE, ProviderChoice.NPU)
    assert result.directory == expected
    assert expected.parent.name == ProviderChoice.NPU.value
    assert expected.name == str(BGE.compiled_seq_len)
    assert expected.parent.parent.name == "BAAI--bge-large-en-v1.5"


def test_preparation_persists_the_graph_the_context_and_the_manifest(
    root: Path, source_dir: Path
) -> None:
    result = prepare(root, source_dir)

    assert (result.directory / ONNX_FILENAME).is_file()
    assert (result.directory / CONTEXT_FILENAME).is_file()
    assert (result.directory / MANIFEST_FILENAME).is_file()
    assert result.onnx_path == result.directory / ONNX_FILENAME
    assert result.context_path == result.directory / CONTEXT_FILENAME


def test_dense_weights_are_carried_into_the_prepared_directory(
    root: Path, source_dir: Path
) -> None:
    """A profile with a Dense stage keeps it: dropping it produces vectors of
    the right shape and the right norm that mean nothing."""
    result = prepare(root, source_dir, profile=GEMMA)

    assert result.dense_path == result.directory / DENSE_FILENAME
    assert result.dense_path.is_file()


def test_no_dense_file_is_published_for_a_profile_without_the_stage(
    root: Path, source_dir: Path
) -> None:
    result = prepare(root, source_dir, profile=BGE)

    assert result.dense_path is None
    assert not (result.directory / DENSE_FILENAME).exists()


def test_the_cpu_provider_is_prepared_without_a_compiled_context(
    root: Path, source_dir: Path
) -> None:
    """design.md, Physical Data Model: ``context.onnx`` is "EP context snapshot,
    NPU only". The CPU backend runs the exported graph directly."""
    compiler = FakeCompiler()

    result = prepare(
        root, source_dir, provider=ProviderChoice.CPU, compiler=compiler
    )

    assert compiler.calls == []
    assert result.context_path is None
    assert not (result.directory / CONTEXT_FILENAME).exists()
    assert (result.directory / ONNX_FILENAME).is_file()


def test_auto_is_a_request_and_never_an_artifact_identity(
    root: Path, source_dir: Path
) -> None:
    """`ProviderChoice.AUTO` names no compute unit, so it cannot be the provider
    an artifact was produced under (requirement 2.6, 4.7)."""
    with pytest.raises(PreparationError) as caught:
        prepare(root, source_dir, provider=ProviderChoice.AUTO)

    assert "auto" in str(caught.value).lower()


# --------------------------------------------------------------------------
# The manifest (4.3, design.md Physical Data Model)
# --------------------------------------------------------------------------


def test_the_manifest_records_the_model_identity_and_its_revision(
    root: Path, source_dir: Path
) -> None:
    result = prepare(root, source_dir)

    stored = json.loads((result.directory / MANIFEST_FILENAME).read_text("utf-8"))
    assert stored["identity"]["model_id"] == BGE.model_id
    assert stored["identity"]["revision"] == REVISION


def test_the_manifest_records_the_compiled_shape_the_provider_and_the_toolchain(
    root: Path, source_dir: Path
) -> None:
    result = prepare(root, source_dir)

    identity = result.manifest.identity
    assert identity.compiled_seq_len == BGE.compiled_seq_len
    assert identity.batch_size == BGE.batch_size
    assert identity.provider == ProviderChoice.NPU.value
    assert identity.onnxruntime_version == TOOLCHAIN.onnxruntime_version
    assert identity.ryzen_ai_version == TOOLCHAIN.ryzen_ai_version
    assert identity.driver_version == TOOLCHAIN.driver_version


def test_the_manifest_records_the_observed_partition_share(
    root: Path, source_dir: Path
) -> None:
    """design.md's Physical Data Model lists it. It is recorded *here* because
    the diagnostic it comes from lives in a scratch cache that does not survive
    preparation - so task 4.3, which owns the threshold policy, could not
    recover it from a reused artifact otherwise."""
    result = prepare(root, source_dir, compiler=FakeCompiler(share=0.83))

    assert result.manifest.observed_partition_share == pytest.approx(0.83)


def test_the_manifest_lists_exactly_the_files_beside_it(
    root: Path, source_dir: Path
) -> None:
    result = prepare(root, source_dir, profile=GEMMA)

    on_disk = sorted(
        entry.name
        for entry in result.directory.iterdir()
        if entry.is_file() and entry.name != MANIFEST_FILENAME
    )
    assert sorted(result.manifest.files) == on_disk
    assert ONNX_FILENAME in on_disk
    assert CONTEXT_FILENAME in on_disk
    assert DENSE_FILENAME in on_disk


def test_the_compilers_scratch_directory_is_never_published(
    root: Path, source_dir: Path
) -> None:
    """The compiler is given a cache directory inside the build, so an
    interrupted run cannot strand it beside a published artifact. It is removed
    before the manifest is written, so it is not published either."""
    result = prepare(root, source_dir)

    assert [entry.name for entry in result.directory.iterdir() if entry.is_dir()] == []


def test_a_manifest_round_trips_through_json(root: Path, source_dir: Path) -> None:
    result = prepare(root, source_dir)

    text = (result.directory / MANIFEST_FILENAME).read_text("utf-8")
    assert ArtifactManifest.from_mapping(json.loads(text)) == result.manifest


def test_the_manifest_states_the_format_it_was_written_in(
    root: Path, source_dir: Path
) -> None:
    result = prepare(root, source_dir)

    assert result.manifest.version == MANIFEST_VERSION


# --------------------------------------------------------------------------
# 4.4: reuse, and saying so
# --------------------------------------------------------------------------


def test_a_second_preparation_reuses_the_artifacts_and_says_so(
    root: Path, source_dir: Path
) -> None:
    first = prepare(root, source_dir)
    second = prepare(root, source_dir)

    assert first.reused is False
    assert second.reused is True
    assert "reus" in second.reason.lower()
    assert second.directory == first.directory


def test_a_reused_preparation_neither_acquires_nor_exports_nor_compiles(
    root: Path, source_dir: Path
) -> None:
    """The stopwatch below is corroboration; this is the actual claim of 4.4 -
    preparation was not repeated."""
    prepare(root, source_dir)

    acquirer = FakeAcquirer(source_dir)
    exporter = FakeExporter()
    compiler = FakeCompiler()
    second = prepare(
        root, source_dir, acquirer=acquirer, exporter=exporter, compiler=compiler
    )

    assert second.reused is True
    assert acquirer.calls == []
    assert exporter.calls == []
    assert compiler.calls == []


def test_a_reused_preparation_takes_near_zero_time_next_to_the_first(
    root: Path, source_dir: Path
) -> None:
    first = prepare(root, source_dir, compiler=FakeCompiler(delay=0.25))
    second = prepare(root, source_dir, compiler=FakeCompiler(delay=0.25))

    assert first.elapsed_seconds >= 0.25
    assert second.elapsed_seconds < first.elapsed_seconds
    assert second.elapsed_seconds < 0.2


def test_a_run_that_rebuilt_says_why_it_did_not_reuse(
    root: Path, source_dir: Path
) -> None:
    first = prepare(root, source_dir)

    assert first.reused is False
    assert first.reason


def test_a_warm_run_without_a_pinned_revision_adopts_the_recorded_one(
    root: Path, source_dir: Path
) -> None:
    """Requiring the upstream revision on every warm run would mean a network
    round trip, which is exactly what 4.4's near-zero reuse forbids. A caller
    that pins a revision gets it compared; a caller that does not accepts the
    artifact's own recorded identity."""
    prepare(root, source_dir)

    acquirer = FakeAcquirer(source_dir)
    second = prepare(root, source_dir, acquirer=acquirer, revision=None)

    assert second.reused is True
    assert acquirer.calls == []
    assert second.manifest.identity.revision == REVISION


# --------------------------------------------------------------------------
# 4.7: every identity field invalidates independently - the Observable
# --------------------------------------------------------------------------


IDENTITY_FIELDS = tuple(field.name for field in dataclasses.fields(ArtifactIdentity))


def _mutated(value: object) -> object:
    """A different value of the same kind, so the JSON stays well formed."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}-changed"
    if value is None:
        return "was-absent"
    raise AssertionError(f"no mutation defined for {value!r}")


@pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
def test_changing_any_one_identity_field_forces_a_rebuild(
    root: Path, source_dir: Path, field_name: str
) -> None:
    """Requirement 4.7 and this task's Observable, field by field.

    Parametrised from the dataclass rather than from a list: an identity field
    added later that nothing compares fails here the moment it exists.
    """
    first = prepare(root, source_dir)
    manifest_path = first.directory / MANIFEST_FILENAME
    stored = json.loads(manifest_path.read_text("utf-8"))
    stored["identity"][field_name] = _mutated(stored["identity"][field_name])
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")

    compiler = FakeCompiler()
    second = prepare(root, source_dir, compiler=compiler)

    assert second.reused is False, f"{field_name} did not invalidate the artifact"
    assert compiler.calls != []
    assert field_name in second.reason


def test_the_identity_carries_every_dimension_design_names(root: Path) -> None:
    """design.md, Physical Data Model: model id and revision, compiled sequence
    length, batch size, provider, Ryzen AI runtime version, driver version,
    ONNX Runtime version."""
    assert set(IDENTITY_FIELDS) == {
        "model_id",
        "revision",
        "provider",
        "compiled_seq_len",
        "batch_size",
        "onnxruntime_version",
        "ryzen_ai_version",
        "driver_version",
    }


def test_the_observed_partition_share_is_recorded_but_does_not_decide_reuse(
    root: Path, source_dir: Path
) -> None:
    """The one manifest field outside the comparison, deliberately.

    It is an observation about a compilation that already happened, not a claim
    about which model, shape, provider or toolchain the artifact was built for.
    Comparing it would rebuild whenever the compiler's own reporting drifted,
    and 4.7 names no such dimension.
    """
    first = prepare(root, source_dir)
    manifest_path = first.directory / MANIFEST_FILENAME
    stored = json.loads(manifest_path.read_text("utf-8"))
    stored["observed_partition_share"] = 0.11
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")

    second = prepare(root, source_dir)

    assert second.reused is True


def test_a_different_provider_prepares_into_its_own_directory(
    root: Path, source_dir: Path
) -> None:
    npu = prepare(root, source_dir, provider=ProviderChoice.NPU)
    cpu = prepare(root, source_dir, provider=ProviderChoice.CPU)

    assert cpu.reused is False
    assert cpu.directory != npu.directory
    assert (npu.directory / MANIFEST_FILENAME).is_file()


def test_a_different_compiled_length_prepares_into_its_own_directory(
    root: Path, source_dir: Path
) -> None:
    shorter = dataclasses.replace(BGE, compiled_seq_len=256)

    first = prepare(root, source_dir, profile=BGE)
    second = prepare(root, source_dir, profile=shorter)

    assert second.reused is False
    assert second.directory != first.directory


def test_a_different_toolchain_forces_a_rebuild(
    root: Path, source_dir: Path
) -> None:
    prepare(root, source_dir, toolchain=TOOLCHAIN)

    upgraded = dataclasses.replace(TOOLCHAIN, ryzen_ai_version="1.9.0")
    second = prepare(root, source_dir, toolchain=upgraded)

    assert second.reused is False
    assert "ryzen_ai_version" in second.reason


def test_a_missing_manifest_forces_a_rebuild(root: Path, source_dir: Path) -> None:
    first = prepare(root, source_dir)
    (first.directory / MANIFEST_FILENAME).unlink()

    second = prepare(root, source_dir)

    assert second.reused is False
    assert "manifest" in second.reason.lower()


def test_an_unreadable_manifest_forces_a_rebuild(
    root: Path, source_dir: Path
) -> None:
    first = prepare(root, source_dir)
    (first.directory / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    second = prepare(root, source_dir)

    assert second.reused is False


def test_a_manifest_written_in_another_format_version_is_not_trusted(
    root: Path, source_dir: Path
) -> None:
    first = prepare(root, source_dir)
    manifest_path = first.directory / MANIFEST_FILENAME
    stored = json.loads(manifest_path.read_text("utf-8"))
    stored["version"] = MANIFEST_VERSION + 1
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")

    second = prepare(root, source_dir)

    assert second.reused is False


@pytest.mark.parametrize("filename", [ONNX_FILENAME, CONTEXT_FILENAME])
def test_a_manifest_whose_artifacts_have_gone_missing_forces_a_rebuild(
    root: Path, source_dir: Path, filename: str
) -> None:
    """Staleness is decided by the manifest, but a manifest describing files
    that are not there describes nothing."""
    first = prepare(root, source_dir)
    (first.directory / filename).unlink()

    second = prepare(root, source_dir)

    assert second.reused is False
    assert filename in second.reason


def test_a_dense_file_the_profile_requires_but_the_directory_lacks_is_not_reused(
    root: Path, source_dir: Path
) -> None:
    first = prepare(root, source_dir, profile=GEMMA)
    (first.directory / DENSE_FILENAME).unlink()

    second = prepare(root, source_dir, profile=GEMMA)

    assert second.reused is False


# --------------------------------------------------------------------------
# The reuse decision as a value, so the comparison can be examined directly
# --------------------------------------------------------------------------


def identity(**changes: Any) -> ArtifactIdentity:
    base = {
        "model_id": BGE.model_id,
        "revision": REVISION,
        "provider": ProviderChoice.NPU.value,
        "compiled_seq_len": BGE.compiled_seq_len,
        "batch_size": BGE.batch_size,
        "onnxruntime_version": TOOLCHAIN.onnxruntime_version,
        "ryzen_ai_version": TOOLCHAIN.ryzen_ai_version,
        "driver_version": TOOLCHAIN.driver_version,
    }
    base.update(changes)
    return ArtifactIdentity(**base)  # type: ignore[arg-type]


def test_an_identical_identity_is_reusable() -> None:
    manifest = ArtifactManifest(identity=identity(), observed_partition_share=0.9)

    decision = reuse_decision(manifest, identity())

    assert decision.reusable is True


def test_an_absent_manifest_is_not_reusable() -> None:
    decision = reuse_decision(None, identity())

    assert decision.reusable is False
    assert decision.reason


@pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
def test_the_reuse_decision_names_the_field_that_differs(field_name: str) -> None:
    stored = ArtifactManifest(identity=identity(), observed_partition_share=None)
    current = identity(**{field_name: _mutated(getattr(stored.identity, field_name))})

    decision = reuse_decision(stored, current)

    assert decision.reusable is False
    assert field_name in decision.reason


# --------------------------------------------------------------------------
# 8.5: nothing partial survives
# --------------------------------------------------------------------------


def test_a_failed_compilation_publishes_nothing(
    root: Path, source_dir: Path
) -> None:
    with pytest.raises(PreparationError):
        prepare(
            root,
            source_dir,
            compiler=FakeCompiler(error=RuntimeError("the compiler died")),
        )

    assert not artifact_directory(root, BGE, ProviderChoice.NPU).exists()


def test_a_failed_compilation_leaves_no_temporary_directory_behind(
    root: Path, source_dir: Path
) -> None:
    with pytest.raises(PreparationError):
        prepare(
            root,
            source_dir,
            compiler=FakeCompiler(error=RuntimeError("the compiler died")),
        )

    leftovers = [path for path in root.rglob("*") if path.is_dir()]
    assert all(
        not path.name.endswith(".partial") for path in leftovers
    ), [str(p) for p in leftovers]


def test_a_failed_compilation_is_not_reused_by_the_next_run(
    root: Path, source_dir: Path
) -> None:
    with pytest.raises(PreparationError):
        prepare(root, source_dir, compiler=FakeCompiler(error=RuntimeError("no")))

    second = prepare(root, source_dir)

    assert second.reused is False
    assert (second.directory / MANIFEST_FILENAME).is_file()


def test_a_compiler_that_writes_no_context_has_not_compiled_anything(
    root: Path, source_dir: Path
) -> None:
    """research.md, third probe: a synthetic graph produced only JSON and no
    binary. The presence of the compiled artifact is itself the evidence that
    compilation genuinely happened, so a clean return with nothing on disk is a
    failure, not a success."""
    with pytest.raises(PreparationError) as caught:
        prepare(root, source_dir, compiler=FakeCompiler(produces=False))

    assert caught.value.stage == COMPILE_STAGE
    assert not artifact_directory(root, BGE, ProviderChoice.NPU).exists()


def test_an_earlier_valid_artifact_survives_a_failed_rebuild(
    root: Path, source_dir: Path
) -> None:
    """Publication replaces the directory only once the new one is complete."""
    first = prepare(root, source_dir)

    upgraded = dataclasses.replace(TOOLCHAIN, driver_version="99.0.0.0")
    with pytest.raises(PreparationError):
        prepare(
            root,
            source_dir,
            toolchain=upgraded,
            compiler=FakeCompiler(error=RuntimeError("no")),
        )

    assert (first.directory / MANIFEST_FILENAME).is_file()
    assert (first.directory / CONTEXT_FILENAME).is_file()
    reused = prepare(root, source_dir)
    assert reused.reused is True


def test_a_publication_that_fails_halfway_restores_what_it_displaced(
    root: Path, source_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication moves the old artifact aside before moving the new one in.
    If the second move fails the first must be undone, or a rebuild that goes
    wrong at the very last step destroys a working artifact.
    """
    first = prepare(root, source_dir)
    real_replace = os.replace
    attempts = {"count": 0}

    def flaky(src: Any, dst: Any, **kwargs: Any) -> None:
        attempts["count"] += 1
        if attempts["count"] == 2:
            raise OSError("the rename failed")
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", flaky)
    upgraded = dataclasses.replace(TOOLCHAIN, driver_version="99.0.0.0")
    with pytest.raises(PreparationError) as caught:
        prepare(root, source_dir, toolchain=upgraded)
    monkeypatch.undo()

    assert caught.value.stage == PUBLICATION_STAGE
    assert (first.directory / MANIFEST_FILENAME).is_file()
    assert (first.directory / CONTEXT_FILENAME).is_file()
    assert prepare(root, source_dir).reused is True


def test_a_leftover_temporary_directory_is_never_accepted_as_prepared(
    root: Path, source_dir: Path
) -> None:
    """A partial directory is not merely incomplete, it is unnamed: the reuse
    check looks only at the published path."""
    first = prepare(root, source_dir)
    partial = first.directory.parent / f"{first.directory.name}.partial"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / MANIFEST_FILENAME).write_text(
        (first.directory / MANIFEST_FILENAME).read_text("utf-8"), encoding="utf-8"
    )
    (first.directory / MANIFEST_FILENAME).unlink()

    second = prepare(root, source_dir)

    assert second.reused is False


def test_a_stale_temporary_directory_is_swept_away_by_the_next_run(
    root: Path, source_dir: Path
) -> None:
    directory = artifact_directory(root, BGE, ProviderChoice.NPU)
    partial = directory.parent / f"{directory.name}.partial"
    partial.mkdir(parents=True)
    (partial / "junk.bin").write_bytes(b"left by a killed run")

    result = prepare(root, source_dir)

    assert not partial.exists()
    # The sweep is load-bearing, not housekeeping: without it the next run
    # builds *into* the residue and publishes it as part of the artifact.
    assert not (result.directory / "junk.bin").exists()
    assert "junk.bin" not in result.manifest.files


KILLED_CHILD = textwrap.dedent(
    '''
    """Start a preparation, signal that compilation has begun, then hang.

    Run as a real child process so the parent can kill it outright. An
    exception would unwind the ``finally`` blocks and prove only that the
    cleanup path works; requirement 8.5 promises something stronger.
    """
    import sys
    import time
    from pathlib import Path

    import numpy as np

    from npu_rag.embedding.models.acquire import AcquiredModel
    from npu_rag.embedding.models.artifacts import (
        CompiledGraph,
        Toolchain,
        ensure_prepared,
    )
    from npu_rag.embedding.models.export import (
        DENSE_FILENAME,
        ONNX_FILENAME,
        ExportedModel,
    )
    from npu_rag.embedding.profiles import profile_for
    from npu_rag.embedding.types import ProviderChoice

    root = Path(sys.argv[1])
    source = Path(sys.argv[2])
    started = Path(sys.argv[3])
    source.mkdir(parents=True, exist_ok=True)


    def acquirer(profile, *, revision, progress=None):
        return AcquiredModel(
            model_id=profile.model_id, revision="a" * 40, local_path=source
        )


    def exporter(profile, acquired, destination, *, progress=None):
        destination.mkdir(parents=True, exist_ok=True)
        onnx_path = destination / ONNX_FILENAME
        onnx_path.write_bytes(b"onnx-graph-bytes")
        return ExportedModel(
            model_id=profile.model_id,
            revision=acquired.revision,
            onnx_path=onnx_path,
            dense_path=None,
            batch_size=profile.batch_size,
            compiled_seq_len=profile.compiled_seq_len,
            hidden_size=profile.dimension,
        )


    class HangingCompiler:
        def compile(self, *, graph, context, profile, cache_dir):
            context.write_bytes(b"half-written-context")
            started.write_text("compiling", encoding="utf-8")
            time.sleep(600)
            return CompiledGraph(context_path=context, partition_share=1.0)


    ensure_prepared(
        profile_for("bge-large-en-v1.5"),
        ProviderChoice.NPU,
        root,
        toolchain=Toolchain(
            onnxruntime_version="1.23.2.dev20260117",
            ryzen_ai_version="1.7.0",
            driver_version="32.0.20102.3930",
        ),
        acquirer=acquirer,
        exporter=exporter,
        compiler=HangingCompiler(),
        expected_revision="a" * 40,
    )
    '''
)


def test_a_run_killed_mid_preparation_leaves_nothing_a_later_run_accepts(
    root: Path, tmp_path: Path
) -> None:
    """Requirement 8.5, tested the way it is written.

    The child is killed outright - no exception, no ``finally``, no atexit - so
    whatever is on disk afterwards is genuinely what an interrupted run leaves.
    """
    script = tmp_path / "killed_child.py"
    script.write_text(KILLED_CHILD, encoding="utf-8")
    signal = tmp_path / "started.flag"
    source = tmp_path / "child-weights"

    child = subprocess.Popen(
        [sys.executable, str(script), str(root), str(source), str(signal)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src")},
    )
    try:
        deadline = time.monotonic() + 60
        while not signal.exists() and time.monotonic() < deadline:
            if child.poll() is not None:
                raise AssertionError(
                    "the child exited before compiling: "
                    f"{(child.stderr.read().decode() if child.stderr else '')}"
                )
            time.sleep(0.02)
        assert signal.exists(), "the child never reached compilation"
    finally:
        child.kill()
        child.wait(timeout=30)

    directory = artifact_directory(root, BGE, ProviderChoice.NPU)
    assert not (directory / MANIFEST_FILENAME).exists()

    weights = tmp_path / "weights-after-the-kill"
    weights.mkdir()
    compiler = FakeCompiler()
    later = prepare(root, weights, compiler=compiler)

    assert later.reused is False
    assert compiler.calls != []
    assert (later.directory / MANIFEST_FILENAME).is_file()
    assert (later.directory / CONTEXT_FILENAME).read_bytes() != b"half-written-context"


# --------------------------------------------------------------------------
# 4.6: the failing stage is named, and nothing is substituted
# --------------------------------------------------------------------------


def test_a_compilation_failure_names_the_compile_stage(
    root: Path, source_dir: Path
) -> None:
    with pytest.raises(PreparationError) as caught:
        prepare(root, source_dir, compiler=FakeCompiler(error=OSError("boom")))

    assert caught.value.stage == COMPILE_STAGE
    assert caught.value.model_id == BGE.model_id
    assert caught.value.provider == ProviderChoice.NPU


def test_an_already_diagnosed_acquisition_failure_passes_through_unchanged(
    root: Path, source_dir: Path
) -> None:
    """Requirement 4.5's licensing error must not be reburied under a generic
    compile-stage diagnosis."""
    gate = LicenseAcceptanceRequired(
        "terms not accepted",
        acceptance_url="https://huggingface.co/google/embeddinggemma-300m",
        model_id=GEMMA.model_id,
    )

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        prepare(
            root,
            source_dir,
            profile=GEMMA,
            acquirer=FakeAcquirer(source_dir, error=gate),
        )

    assert caught.value is gate


def test_an_export_failure_is_not_re_diagnosed_as_a_compile_failure(
    root: Path, source_dir: Path
) -> None:
    failure = PreparationError("export blew up", stage="export")

    with pytest.raises(PreparationError) as caught:
        prepare(root, source_dir, exporter=FakeExporter(error=failure))

    assert caught.value.stage == "export"


@pytest.mark.parametrize(
    "error", [RuntimeError("native crash"), OSError("disk full"), ValueError("nope")]
)
def test_every_ordinary_failure_becomes_one_of_this_features_errors(
    root: Path, source_dir: Path, error: Exception
) -> None:
    with pytest.raises(EmbeddingRuntimeError):
        prepare(root, source_dir, compiler=FakeCompiler(error=error))


def test_an_interrupt_is_not_dressed_up_as_a_preparation_failure(
    root: Path, source_dir: Path
) -> None:
    with pytest.raises(KeyboardInterrupt):
        prepare(root, source_dir, compiler=FakeCompiler(error=KeyboardInterrupt()))

    assert not artifact_directory(root, BGE, ProviderChoice.NPU).exists()


def test_preparation_never_substitutes_another_provider(
    root: Path, source_dir: Path
) -> None:
    """Requirement 4.6: a failed NPU preparation does not quietly become a CPU
    one."""
    with pytest.raises(PreparationError):
        prepare(root, source_dir, compiler=FakeCompiler(error=RuntimeError("no")))

    assert not artifact_directory(root, BGE, ProviderChoice.CPU).exists()


def test_a_model_other_than_the_one_requested_is_refused(
    root: Path, source_dir: Path
) -> None:
    acquirer = FakeAcquirer(source_dir)

    class WrongModel(FakeAcquirer):
        def __call__(
            self, profile: ModelProfile, *, revision: str | None, progress: Any = None
        ) -> AcquiredModel:
            self.calls.append((profile.model_id, revision))
            return AcquiredModel(
                model_id=GEMMA.model_id,
                revision=revision or REVISION,
                local_path=self.source,
            )

    with pytest.raises(PreparationError) as caught:
        prepare(root, source_dir, profile=BGE, acquirer=WrongModel(source_dir))

    assert BGE.model_id in str(caught.value)
    assert not artifact_directory(root, BGE, ProviderChoice.NPU).exists()
    assert acquirer.calls == []


class IgnoringRevisionAcquirer(FakeAcquirer):
    """An acquirer that returns a different commit than the one asked for."""

    def __call__(
        self, profile: ModelProfile, *, revision: str | None, progress: Any = None
    ) -> AcquiredModel:
        self.calls.append((profile.model_id, revision))
        return AcquiredModel(
            model_id=profile.model_id,
            revision=OTHER_REVISION,
            local_path=self.source,
        )


def test_a_pinned_revision_is_what_the_repository_is_actually_asked_for(
    root: Path, source_dir: Path
) -> None:
    """Pinning must reach the download, not only the comparison. A pin that only
    reached the comparison would fetch the branch head and then reject it - a
    caller could never obtain the commit they named."""
    acquirer = FakeAcquirer(source_dir)

    prepare(root, source_dir, acquirer=acquirer, revision=REVISION)

    assert acquirer.calls == [(BGE.model_id, REVISION)]


def test_an_unpinned_preparation_asks_the_repository_for_no_particular_commit(
    root: Path, source_dir: Path
) -> None:
    acquirer = FakeAcquirer(source_dir)

    prepare(root, source_dir, acquirer=acquirer, revision=None)

    assert acquirer.calls == [(BGE.model_id, None)]


def test_a_revision_other_than_the_one_pinned_is_refused(
    root: Path, source_dir: Path
) -> None:
    """A caller that pinned a commit and silently got another would have exactly
    the undetectable upstream change the manifest exists to expose."""
    with pytest.raises(PreparationError) as caught:
        prepare(root, source_dir, acquirer=IgnoringRevisionAcquirer(source_dir))

    assert OTHER_REVISION in str(caught.value)
    assert not artifact_directory(root, BGE, ProviderChoice.NPU).exists()


# --------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------


def test_progress_is_reported_as_a_callback(root: Path, source_dir: Path) -> None:
    seen: list[ProgressUpdate] = []

    prepare(root, source_dir, progress=seen.append)

    assert seen
    assert [update.completed for update in seen] == sorted(
        update.completed for update in seen
    )
    assert seen[-1].completed == seen[-1].total
    assert all(update.operation == f"prepare:{BGE.name}" for update in seen)


def test_a_reused_preparation_still_reports_completion(
    root: Path, source_dir: Path
) -> None:
    prepare(root, source_dir)
    seen: list[ProgressUpdate] = []

    result = prepare(root, source_dir, progress=seen.append)

    assert result.reused is True
    assert seen
    assert seen[-1].completed == seen[-1].total


# --------------------------------------------------------------------------
# The toolchain fingerprint
# --------------------------------------------------------------------------


def capability(
    *, driver: str | None = None, runtime: str | None = None
) -> CapabilityReport:
    return CapabilityReport(
        conditions=(),
        execution_mode=ExecutionMode.UNAVAILABLE,
        driver_version=driver,
        runtime_version=runtime,
        device_name=None,
        power_reporting_supported=False,
    )


def test_the_toolchain_reads_its_versions_from_the_capability_report() -> None:
    """design.md, CapabilityChecker: the run context is built from this report
    "so version provenance has one source"."""
    toolchain = current_toolchain(
        capability=capability(
            driver="32.0.20102.3930", runtime="1.23.2.dev20260117"
        )
    )

    assert toolchain.onnxruntime_version == "1.23.2.dev20260117"
    assert toolchain.driver_version == "32.0.20102.3930"


def test_the_runtime_version_is_never_taken_from_distribution_metadata() -> None:
    """Measured 2026-09-06 on this machine: ``importlib.metadata.version(
    "onnxruntime")`` reports ``1.29.0`` - the stale stock ``dist-info`` residue
    Implementation Note 1.2 records - while ``onnxruntime.__version__``, which
    the capability report carries, reports the vendor build actually loaded.
    Fingerprinting the toolchain from the metadata would key every artifact to a
    version that is not running.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

    queried = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "version"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert "onnxruntime" not in queried
    assert RYZEN_AI_DISTRIBUTION == "voe"


def test_an_absent_vendor_distribution_is_recorded_as_absent_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guessed version would make two different environments compare equal,
    which is the one thing the fingerprint exists to prevent."""

    def absent(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", absent)

    toolchain = current_toolchain(capability=capability())

    assert toolchain.ryzen_ai_version is None
    assert toolchain.onnxruntime_version is None
    assert toolchain.driver_version is None


def test_the_vendor_distribution_version_is_read_when_it_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metadata, "version", lambda name: f"v-for-{name}")

    toolchain = current_toolchain(capability=capability())

    assert toolchain.ryzen_ai_version == f"v-for-{RYZEN_AI_DISTRIBUTION}"


def test_the_vendor_distribution_is_the_one_actually_installed_here() -> None:
    """Not a tautology: it pins that the name in the fingerprint resolves to a
    real distribution in this environment, so the field is a version and not a
    permanent ``None``."""
    assert metadata.version(RYZEN_AI_DISTRIBUTION)


# --------------------------------------------------------------------------
# The partition-share diagnostic: only one of the three files tells the truth
# --------------------------------------------------------------------------


def test_the_partition_share_is_read_from_the_authoritative_summary(
    tmp_path: Path,
) -> None:
    """research.md, third probe: the console percentage and
    ``fail_safe_summary.json`` both report a fail-safe *plan*; only
    ``preliminary-vaiml-pass-summary.txt`` reports a verdict."""
    cache = tmp_path / "cache" / "key"
    cache.mkdir(parents=True)
    (cache / "preliminary-vaiml-pass-summary.txt").write_text(
        "Model data type: float32\n"
        "Device data type: bfloat16\n"
        "Number of operators supported by VAIML: 412(97.400%)\n",
        encoding="utf-8",
    )

    assert read_partition_share(tmp_path / "cache") == pytest.approx(0.974)


def test_the_misleading_fail_safe_summary_is_never_believed(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "key"
    cache.mkdir(parents=True)
    (cache / "vaiml_partition_fe.flexml").mkdir()
    (cache / "vaiml_partition_fe.flexml" / "fail_safe_summary.json").write_text(
        json.dumps({"AIE": 100, "CPU": 0}), encoding="utf-8"
    )

    assert read_partition_share(tmp_path / "cache") is None


def test_a_measured_zero_share_is_a_measurement_not_an_absence(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache" / "key"
    cache.mkdir(parents=True)
    (cache / "preliminary-vaiml-pass-summary.txt").write_text(
        "Number of operators supported by VAIML: 0(0.000%)\n", encoding="utf-8"
    )

    assert read_partition_share(tmp_path / "cache") == 0.0


def test_an_unreadable_diagnostic_reports_nothing_rather_than_raising(
    tmp_path: Path,
) -> None:
    assert read_partition_share(tmp_path / "nowhere") is None


def test_the_real_compiler_is_constructible_without_the_vendor_runtime() -> None:
    """Constructing it must not import or touch ONNX Runtime: `ensure_prepared`
    builds one by default even for the CPU provider, which never compiles."""
    assert OrtContextCompiler() is not None
