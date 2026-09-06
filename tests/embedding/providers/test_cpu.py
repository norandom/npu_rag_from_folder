"""The CPU backend, the full-precision reference path (task 4.2).

Every test here runs against an **injected session**, so the whole of the
adapter's logic - which graph it opens, which provider it demands, what it
refuses, what it hands back - is examined without an ONNX Runtime session and
without a model. The one thing an injected session cannot show is that a real
graph really executes at full precision on the real CPU provider, and that this
machine's *healthy* NPU does not prevent the CPU from being selected. Both live
in ``test_cpu_live.py``.

The Observable this file has to pin (requirement 2.3, and 6.3's reference role):

- the backend produces vectors for the same inputs, at the same shape, that the
  NPU backend will accept - so a wrong shape is **refused**, never reshaped;
- it is selectable regardless of NPU availability - so nothing in it consults
  NPU state, which is asserted structurally rather than promised;
- it executes at full precision - so nothing in it casts, quantises, or sets a
  precision-reducing session option, which is likewise structural here and
  measured live;
- session construction is not evidence of anything. Requesting a provider
  ONNX Runtime does not have **succeeds** and silently runs on the CPU, so the
  provider that came back is read rather than assumed - benign here, since the
  CPU is what was wanted, but a check that only passes by luck is not a check.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from npu_rag.embedding.errors import (
    EnvironmentError_,
    ExecutionError,
    PreparationError,
)
from npu_rag.embedding.models.artifacts import (
    ArtifactIdentity,
    ArtifactManifest,
    PreparedArtifact,
)
from npu_rag.embedding.models.export import (
    ATTENTION_MASK,
    INPUT_IDS,
    ONNX_FILENAME,
)
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.providers import cpu as cpu_module
from npu_rag.embedding.providers.base import (
    BackendFactories,
    TransformerBackend,
    resolve_backend,
)
from npu_rag.embedding.providers.cpu import (
    CPU_PROVIDER,
    SESSION_STAGE,
    CpuBackend,
)
from npu_rag.embedding.types import (
    CONDITION_PROVIDER_REGISTERED,
    CapabilityReport,
    Condition,
    ExecutionMode,
    ProviderChoice,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "npu_rag"
    / "embedding"
    / "providers"
    / "cpu.py"
)
MODULE_PACKAGE = "npu_rag.embedding"

#: Small enough to write by hand and read in a failure message. The shape is the
#: only thing under test, so it is deliberately not 512.
SEQ_LEN = 4
HIDDEN = 3

PROFILE = ModelProfile(
    model_id="test/reference-model",
    dimension=HIDDEN,
    compiled_seq_len=SEQ_LEN,
    architectural_context_limit=SEQ_LEN * 2,
    batch_size=1,
    pooling="mean",
    has_dense_stage=False,
    document_template="{content}",
    query_template="{content}",
    license_gated=False,
)


# --------------------------------------------------------------------------
# Stand-ins
# --------------------------------------------------------------------------


class FakeSession:
    """An ONNX Runtime session as this adapter is allowed to see it."""

    def __init__(
        self,
        *,
        providers: Sequence[str] = (CPU_PROVIDER,),
        output: object | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._providers = list(providers)
        self._output = output
        self._raises = raises
        self.calls: list[tuple[object, Mapping[str, Any]]] = []

    def get_providers(self) -> Sequence[str]:
        return list(self._providers)

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
        /,
    ) -> Sequence[Any]:
        self.calls.append((output_names, dict(input_feed)))
        if self._raises is not None:
            raise self._raises
        if self._output is None:
            return [np.zeros((PROFILE.batch_size, SEQ_LEN, HIDDEN), np.float32)]
        return [self._output]


class RecordingFactory:
    """Records what the adapter asked for, and hands back a chosen session."""

    def __init__(self, session: FakeSession | None = None) -> None:
        self.session = session if session is not None else FakeSession()
        self.calls: list[tuple[Path, tuple[str, ...]]] = []

    def __call__(
        self, model_path: Path, providers: Sequence[str], /
    ) -> FakeSession:
        self.calls.append((model_path, tuple(providers)))
        return self.session


class ExplodingFactory:
    """A session factory that must never be reached."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.called = False

    def __call__(
        self, model_path: Path, providers: Sequence[str], /
    ) -> FakeSession:
        self.called = True
        if self.error is not None:
            raise self.error
        raise AssertionError("a session was built when none should have been")


def artifact(
    tmp_path: Path, *, with_context: bool = False, profile: ModelProfile = PROFILE
) -> PreparedArtifact:
    """A published artifact directory, with the files a manifest may name."""
    directory = tmp_path / "artifact"
    directory.mkdir(exist_ok=True)
    onnx_path = directory / ONNX_FILENAME
    onnx_path.write_bytes(b"not a real graph; no session is ever built from it")
    context_path: Path | None = None
    if with_context:
        context_path = directory / "context.onnx"
        context_path.write_bytes(b"the compiled NPU snapshot")
    identity = ArtifactIdentity(
        model_id=profile.model_id,
        revision="0" * 40,
        provider=ProviderChoice.CPU.value,
        compiled_seq_len=profile.compiled_seq_len,
        batch_size=profile.batch_size,
        onnxruntime_version="1.23.2",
        ryzen_ai_version=None,
        driver_version=None,
    )
    return PreparedArtifact(
        directory=directory,
        onnx_path=onnx_path,
        context_path=context_path,
        dense_path=None,
        manifest=ArtifactManifest(
            identity=identity,
            observed_partition_share=None,
            files=(ONNX_FILENAME,),
        ),
        reused=False,
        reason="built for this test",
        elapsed_seconds=0.0,
    )


def ids() -> npt.NDArray[np.int64]:
    return np.arange(PROFILE.batch_size * SEQ_LEN, dtype=np.int64).reshape(
        PROFILE.batch_size, SEQ_LEN
    )


#: How many of `mask`'s positions are attended. Strictly between zero and
#: ``SEQ_LEN``, and deliberately **not** half of it: an all-ones mask cannot tell
#: the caller's mask apart from a fabricated ``np.ones_like`` one, and a
#: half-and-half split cannot tell it apart from its own reversal. Three of four
#: distinguishes both, and is what a real padded batch looks like anyway.
ATTENDED = SEQ_LEN - 1


def mask() -> npt.NDArray[np.int64]:
    """A *padded* batch's attention mask: attended positions, then padding."""
    row = np.zeros(SEQ_LEN, dtype=np.int64)
    row[:ATTENDED] = 1
    return np.tile(row, (PROFILE.batch_size, 1))


def backend(
    tmp_path: Path,
    *,
    session: FakeSession | None = None,
    factory: Any | None = None,
    with_context: bool = False,
) -> CpuBackend:
    chosen = factory if factory is not None else RecordingFactory(session)
    return CpuBackend(
        artifact(tmp_path, with_context=with_context),
        PROFILE,
        session_factory=chosen,
    )


def healthy_npu_report() -> CapabilityReport:
    """A report from a machine whose NPU works, fabricated so the CPU branch is
    exercised against a *present, healthy* NPU rather than an absent one."""
    return CapabilityReport(
        conditions=(
            Condition(
                name=CONDITION_PROVIDER_REGISTERED,
                satisfied=True,
                observed="VitisAIExecutionProvider registered",
                required="VitisAIExecutionProvider registered",
                remediation=None,
            ),
        ),
        execution_mode=ExecutionMode.IN_PROCESS,
        driver_version="32.0.203.280",
        runtime_version="1.23.2",
        device_name="NPU Compute Accelerator",
        power_reporting_supported=True,
    )


# --------------------------------------------------------------------------
# The contract (design.md, TransformerBackend)
# --------------------------------------------------------------------------


def test_the_cpu_backend_satisfies_the_transformer_backend_protocol(
    tmp_path: Path,
) -> None:
    assert isinstance(backend(tmp_path), TransformerBackend)


def test_it_reports_the_cpu_provider(tmp_path: Path) -> None:
    """Requirement 2.6: the provider reported is the one that ran."""
    assert backend(tmp_path).provider is ProviderChoice.CPU


def test_it_reports_in_process_execution(tmp_path: Path) -> None:
    """Requirement 5.3: the adapter is the only thing that knows where the work
    happened, and this one happens right here."""
    assert backend(tmp_path).execution_mode is ExecutionMode.IN_PROCESS


def test_it_reports_no_npu_partition_share(tmp_path: Path) -> None:
    """``None`` is *not applicable*, and it is what the protocol requires of a
    CPU adapter - never a number that would read as a verified NPU offload."""
    assert backend(tmp_path).npu_partition_share is None


def test_it_offers_no_member_beyond_the_four_the_protocol_declares(
    tmp_path: Path,
) -> None:
    """No ``pool``, no ``normalize``, no ``embed``: post-processing sits on the
    service side of the port so it cannot differ per backend (task 5.2)."""
    public = {name for name in dir(CpuBackend) if not name.startswith("_")}

    assert public == {"provider", "execution_mode", "npu_partition_share", "run"}


# --------------------------------------------------------------------------
# Executing the prepared graph
# --------------------------------------------------------------------------


def test_the_session_is_built_from_the_prepared_full_precision_graph(
    tmp_path: Path,
) -> None:
    """``model.onnx``, the FP32 trunk - never the NPU's compiled snapshot, which
    is present in an NPU artifact and is not what this backend runs."""
    prepared = artifact(tmp_path, with_context=True)
    factory = RecordingFactory()

    CpuBackend(prepared, PROFILE, session_factory=factory)

    (path, _), = factory.calls
    assert path == prepared.onnx_path
    assert path.name == ONNX_FILENAME
    assert prepared.context_path is not None and path != prepared.context_path


def test_the_session_is_asked_for_the_cpu_provider(tmp_path: Path) -> None:
    factory = RecordingFactory()

    CpuBackend(artifact(tmp_path), PROFILE, session_factory=factory)

    (_, providers), = factory.calls
    assert providers == (CPU_PROVIDER,)


def test_the_fixtures_can_tell_a_real_batch_from_a_fabricated_one() -> None:
    """The non-vacuity check the assertion below depends on.

    An all-ones mask makes ``assert_array_equal(feed[ATTENTION_MASK], mask)``
    trivially true against a backend that fabricated the mask, and this feature
    has already shipped two assertions made vacuous by their own fixture data
    (Implementation Notes 2.2 and 3.2). So the fixtures are held to being
    distinguishable from the three arrays a mutant would reach for - all ones,
    all zeros, and each other - rather than trusted to stay that way.
    """
    token_ids, attention = ids(), mask()

    for label, array in ((INPUT_IDS, token_ids), (ATTENTION_MASK, attention)):
        assert not np.array_equal(array, np.ones_like(array)), label
        assert not np.array_equal(array, np.zeros_like(array)), label
        # Not its own reversal either, so a mutant that fed the batch backwards
        # would be caught by the same equality.
        assert not np.array_equal(array, array[:, ::-1]), label
    assert not np.array_equal(token_ids, attention)
    assert 0 < int(attention.sum()) < attention.size


def test_run_feeds_the_graph_the_callers_own_arrays(tmp_path: Path) -> None:
    session = FakeSession()
    token_ids, attention = ids(), mask()

    backend(tmp_path, session=session).run(token_ids, attention)

    (output_names, feed), = session.calls
    assert output_names is None
    assert set(feed) == {INPUT_IDS, ATTENTION_MASK}
    np.testing.assert_array_equal(feed[INPUT_IDS], token_ids)
    # The mask is padded rather than all ones (see `mask`), so this catches a
    # backend that passed the graph a mask of its own making - the mutant that
    # otherwise survives every unit test and is caught only by the live suite,
    # which will not run on a CI machine with no NPU.
    np.testing.assert_array_equal(feed[ATTENTION_MASK], attention)
    assert feed[INPUT_IDS].dtype == np.int64
    assert feed[ATTENTION_MASK].dtype == np.int64


def test_run_returns_the_graphs_own_output_rather_than_one_it_made_up(
    tmp_path: Path,
) -> None:
    """Identity, not equality. An adapter that fabricated a correctly shaped
    array of zeros would satisfy every shape assertion in this file."""
    produced = np.linspace(
        -1.0, 1.0, PROFILE.batch_size * SEQ_LEN * HIDDEN, dtype=np.float32
    ).reshape(PROFILE.batch_size, SEQ_LEN, HIDDEN)
    session = FakeSession(output=produced)

    result = backend(tmp_path, session=session).run(ids(), mask())

    assert result is produced


def test_the_returned_vectors_keep_the_batch_and_the_compiled_length(
    tmp_path: Path,
) -> None:
    result = backend(tmp_path).run(ids(), mask())

    assert result.shape == (PROFILE.batch_size, SEQ_LEN, HIDDEN)
    assert result.dtype == np.float32


# --------------------------------------------------------------------------
# The provider is verified, never assumed (design.md, guards one and two)
# --------------------------------------------------------------------------


def test_the_cpu_provider_must_be_registered_before_a_session_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard one, as a precondition. Nothing is constructed when the runtime
    cannot offer the provider - the check is *before*, not after."""
    monkeypatch.setattr(cpu_module, "available_providers", lambda: ())
    factory = ExplodingFactory()

    with pytest.raises(EnvironmentError_):
        CpuBackend(artifact(tmp_path), PROFILE, session_factory=factory)

    assert factory.called is False


def test_an_unregistered_cpu_provider_is_an_environment_failure_not_a_run_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 8.2. ONNX Runtime's CPU provider is built into every build,
    so its absence says the installation is incomplete - not that this model
    would not run. Reporting it as an execution failure would send an operator
    looking at the graph instead of at their runtime."""
    monkeypatch.setattr(cpu_module, "available_providers", lambda: ("Other",))

    with pytest.raises(EnvironmentError_) as caught:
        CpuBackend(artifact(tmp_path), PROFILE, session_factory=ExplodingFactory())

    assert not isinstance(caught.value, ExecutionError | PreparationError)
    assert caught.value.provider is ProviderChoice.CPU
    assert caught.value.model_id == PROFILE.model_id
    assert CPU_PROVIDER in str(caught.value)


def test_a_session_that_did_not_come_back_on_the_cpu_is_refused(
    tmp_path: Path,
) -> None:
    """Guard two. Requesting a provider ONNX Runtime does not have *succeeds*
    and silently runs elsewhere, so what came back is read rather than assumed:
    a session reporting some other provider is not this backend's session."""
    session = FakeSession(providers=["VitisAIExecutionProvider", CPU_PROVIDER])

    with pytest.raises(ExecutionError) as caught:
        backend(tmp_path, session=session)

    assert "VitisAIExecutionProvider" in str(caught.value)


def test_a_session_reporting_no_provider_at_all_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(ExecutionError):
        backend(tmp_path, session=FakeSession(providers=[]))


# --------------------------------------------------------------------------
# Failures (8.1, 8.2)
# --------------------------------------------------------------------------


def test_a_session_that_will_not_build_is_an_execution_error_naming_the_stage(
    tmp_path: Path,
) -> None:
    factory = ExplodingFactory(RuntimeError("the graph is corrupt"))

    with pytest.raises(ExecutionError) as caught:
        CpuBackend(artifact(tmp_path), PROFILE, session_factory=factory)

    assert caught.value.stage == SESSION_STAGE
    assert caught.value.provider is ProviderChoice.CPU
    assert caught.value.model_id == PROFILE.model_id
    assert "the graph is corrupt" in str(caught.value)


def test_a_failure_inside_the_graph_is_an_execution_error_naming_the_stage(
    tmp_path: Path,
) -> None:
    session = FakeSession(raises=RuntimeError("RUNTIME_EXCEPTION"))

    with pytest.raises(ExecutionError) as caught:
        backend(tmp_path, session=session).run(ids(), mask())

    assert caught.value.stage == ExecutionError.default_stage
    assert caught.value.provider is ProviderChoice.CPU
    assert caught.value.model_id == PROFILE.model_id
    assert "RUNTIME_EXCEPTION" in str(caught.value)


def test_execution_failures_are_not_environment_or_preparation_failures(
    tmp_path: Path,
) -> None:
    """Requirement 8.2 is a *type* question, and a graph that failed while
    running answers it as an execution failure: the artifact was prepared, the
    runtime offered the provider, and the session came back on it. The one
    failure this backend classifies as environmental is the provider being
    absent from the runtime altogether, which the test above pins - the two
    directions together are what makes the categories mean anything."""
    session = FakeSession(raises=RuntimeError("boom"))

    with pytest.raises(ExecutionError) as caught:
        backend(tmp_path, session=session).run(ids(), mask())

    assert not isinstance(caught.value, EnvironmentError_ | PreparationError)


# --------------------------------------------------------------------------
# The shape contract: refused, never repaired
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (PROFILE.batch_size, SEQ_LEN - 1),
        (PROFILE.batch_size, SEQ_LEN + 1),
        (PROFILE.batch_size + 1, SEQ_LEN),
        (SEQ_LEN,),
    ],
)
def test_a_batch_of_the_wrong_shape_is_refused_rather_than_reshaped(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    """The graph is pinned to one shape and partial batches are padded by the
    *caller*. An adapter that quietly reshaped would embed a different text than
    the one it was handed, at the right shape and the right norm."""
    session = FakeSession()
    adapter = backend(tmp_path, session=session)
    wrong = np.zeros(shape, dtype=np.int64)

    with pytest.raises(ExecutionError) as caught:
        adapter.run(wrong, np.ones(shape, dtype=np.int64))

    assert session.calls == []
    assert str(shape[-1]) in str(caught.value) or str(shape) in str(caught.value)


def test_a_mask_that_does_not_match_the_ids_is_refused(tmp_path: Path) -> None:
    session = FakeSession()
    adapter = backend(tmp_path, session=session)

    with pytest.raises(ExecutionError):
        adapter.run(ids(), np.ones((PROFILE.batch_size, SEQ_LEN - 1), np.int64))

    assert session.calls == []


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_a_batch_of_the_wrong_dtype_is_refused_rather_than_cast(
    tmp_path: Path, dtype: Any
) -> None:
    session = FakeSession()
    adapter = backend(tmp_path, session=session)
    wrong = np.zeros((PROFILE.batch_size, SEQ_LEN), dtype=dtype)

    with pytest.raises(ExecutionError):
        adapter.run(wrong, mask())

    assert session.calls == []


# --------------------------------------------------------------------------
# The postcondition is checked, so a wrong answer is not returned as a right one
# --------------------------------------------------------------------------


def test_an_output_that_is_not_full_precision_is_refused(tmp_path: Path) -> None:
    """6.3 compares NPU vectors against *full-precision* CPU ones. A reference
    that came back at half precision would quietly measure nothing."""
    half = np.zeros((PROFILE.batch_size, SEQ_LEN, HIDDEN), dtype=np.float16)

    with pytest.raises(ExecutionError):
        backend(tmp_path, session=FakeSession(output=half)).run(ids(), mask())


@pytest.mark.parametrize(
    "shape",
    [
        (PROFILE.batch_size, SEQ_LEN - 1, HIDDEN),
        (PROFILE.batch_size + 1, SEQ_LEN, HIDDEN),
        (PROFILE.batch_size, SEQ_LEN),
    ],
)
def test_an_output_that_does_not_match_the_inputs_is_refused(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    produced = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ExecutionError):
        backend(tmp_path, session=FakeSession(output=produced)).run(ids(), mask())


def test_a_graph_returning_more_than_one_tensor_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = FakeSession()
    expected = np.zeros((PROFILE.batch_size, SEQ_LEN, HIDDEN), np.float32)
    monkeypatch.setattr(
        session, "run", lambda names, feed: [expected, expected], raising=True
    )

    with pytest.raises(ExecutionError):
        backend(tmp_path, session=session).run(ids(), mask())


# --------------------------------------------------------------------------
# Requirement 2.3: selectable regardless of what the NPU is doing
# --------------------------------------------------------------------------


def test_it_is_selectable_while_the_npu_is_present_and_healthy(
    tmp_path: Path,
) -> None:
    """Requirement 2.3, against a report that says the NPU is fully usable. The
    NPU factory raises, so a resolution that so much as built one would fail."""
    prepared = artifact(tmp_path)
    npu_factory = ExplodingFactory()

    def cpu_factory(
        profile: ModelProfile, capability: CapabilityReport, /
    ) -> TransformerBackend:
        return CpuBackend(prepared, profile, session_factory=RecordingFactory())

    def npu(
        profile: ModelProfile, capability: CapabilityReport, /
    ) -> TransformerBackend:
        npu_factory.called = True
        raise AssertionError("explicit cpu selection must not build an NPU backend")

    resolved, reason = resolve_backend(
        ProviderChoice.CPU,
        PROFILE,
        healthy_npu_report(),
        factories=BackendFactories(npu=npu, cpu=cpu_factory),
    )

    assert resolved.provider is ProviderChoice.CPU
    assert reason is None
    assert npu_factory.called is False


def test_a_selected_cpu_backend_still_runs_with_a_healthy_npu_reported(
    tmp_path: Path,
) -> None:
    """Selection is half of 2.3; the vectors are the other half."""
    prepared = artifact(tmp_path)
    session = FakeSession()

    def cpu_factory(
        profile: ModelProfile, capability: CapabilityReport, /
    ) -> TransformerBackend:
        return CpuBackend(prepared, profile, session_factory=RecordingFactory(session))

    def npu(
        profile: ModelProfile, capability: CapabilityReport, /
    ) -> TransformerBackend:
        raise AssertionError("explicit cpu selection must not build an NPU backend")

    resolved, _ = resolve_backend(
        ProviderChoice.CPU,
        PROFILE,
        healthy_npu_report(),
        factories=BackendFactories(npu=npu, cpu=cpu_factory),
    )
    result = resolved.run(ids(), mask())

    assert result.shape == (PROFILE.batch_size, SEQ_LEN, HIDDEN)
    assert len(session.calls) == 1


# --------------------------------------------------------------------------
# This module runs a graph; it does not consult the NPU, reduce precision,
# pool, print, or reach into a later layer
# --------------------------------------------------------------------------


def source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def imported_names(source_text: str, package: str) -> list[str]:
    """Every imported name as an absolute dotted path, relatives resolved."""
    parts = package.split(".")
    names: list[str] = []
    for node in ast.walk(ast.parse(source_text)):
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


def called_names(source_text: str) -> set[str]:
    return {
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(source_text))
        if isinstance(node, ast.Call)
    }


def string_constants(source_text: str) -> set[str]:
    """Every string literal *except* docstrings, which explain rather than act."""
    tree = ast.parse(source_text)
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        )
    }
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


def test_the_backend_never_consults_npu_availability(tmp_path: Path) -> None:
    """Requirement 2.3: ``cpu`` is served regardless of the NPU's state, which
    means nothing here may *look*. An adapter that checked would be one release
    away from declining to run when the NPU is busy, absent, or broken."""
    text = source()

    assert [
        name
        for name in imported_names(text, MODULE_PACKAGE)
        if "environment" in name or "capability" in name.lower()
    ] == []
    assert [
        name
        for name in called_names(text)
        if "capability" in name.lower() or "xrt" in name.lower()
    ] == []
    # ``\bnpu\b`` rather than a substring test: "input_ids" contains the three
    # letters and is the name of the tensor this backend feeds.
    assert [
        literal
        for literal in string_constants(text)
        if "vitis" in literal.lower() or re.search(r"\bnpu\b", literal.lower())
    ] == []


PRECISION_REDUCING = (
    "float16",
    "fp16",
    "bfloat16",
    "bf16",
    "int8",
    "uint8",
    "quant",
    "config_file",
    "vaip",
)


def test_the_backend_reduces_precision_nowhere() -> None:
    """design.md: "Precision is a property of the backend, not the export." This
    is the backend that keeps it - requirement 6.3's whole value is that the two
    backends differ in numeric precision alone."""
    text = source()

    lowered = {literal.lower() for literal in string_constants(text)}
    assert [
        literal
        for literal in lowered
        for banned in PRECISION_REDUCING
        if banned in literal
    ] == []

    # ``SessionOptions`` itself, not only the entries set on one. Every knob
    # that could change the numbers is reached through an options object, so
    # refusing to construct one at all is a stronger and simpler guarantee than
    # enumerating the knobs - and it is what `Session`'s two-member protocol
    # already makes structurally true for the injected path.
    called = called_names(text)
    assert [
        name
        for name in called
        if name.endswith(
            ("astype", "add_session_config_entry", "SessionOptions")
        )
        or "quantize" in name.lower()
    ] == []

    attributes = {
        ast.unparse(node)
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Attribute)
    }
    assert [
        name
        for name in attributes
        for banned in PRECISION_REDUCING
        if banned in name.lower()
    ] == []


def test_the_backend_neither_pools_nor_normalizes() -> None:
    """Post-processing is task 5.2's, on the service side of the port, so it
    executes identically no matter which adapter ran the transformer."""
    tree = ast.parse(source())

    assert [
        name
        for name in called_names(source())
        if name.startswith(("np.mean", "np.sum", "np.linalg", "numpy."))
    ] == []

    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    assert sorted(
        name
        for name in defined
        for banned in ("pool", "normal", "dense", "embed")
        if banned in name.lower()
    ) == []


def test_the_backend_neither_prints_nor_logs() -> None:
    text = source()

    assert [
        name
        for name in imported_names(text, MODULE_PACKAGE)
        for banned in ("logging", "sys", "warnings")
        if name == banned or name.startswith(f"{banned}.")
    ] == []
    assert [
        name
        for name in called_names(text)
        if name == "print" or name.startswith(("logging.", "warnings.", "sys.std"))
    ] == []


def test_the_backend_imports_nothing_from_a_later_layer() -> None:
    """design.md's dependency direction: ``providers`` may never reach
    ``service`` or ``bench``."""
    assert [
        name
        for name in imported_names(source(), MODULE_PACKAGE)
        for layer in ("service", "bench")
        if name == f"{MODULE_PACKAGE}.{layer}"
        or name.startswith(f"{MODULE_PACKAGE}.{layer}.")
    ] == []


def test_onnx_runtime_is_imported_lazily_so_the_package_imports_without_it() -> None:
    """``models/artifacts.py`` made the same choice for the same reason: the
    package must import in an environment that has never seen the vendor
    wheels."""
    tree = ast.parse(source())
    module_level = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }

    assert [name for name in module_level if name.startswith("onnxruntime")] == []
    assert "onnxruntime" in source()


def test_the_cpu_backend_is_reachable_from_the_providers_package() -> None:
    module: Any = __import__(
        "npu_rag.embedding.providers.cpu", fromlist=["CpuBackend"]
    )

    assert module.CpuBackend is CpuBackend
