"""The NPU backend and its partition verification (task 4.3).

Every test here runs against an **injected session** and, where the partition
verdict is under test, an **injected node-mix reader** or a **fabricated
``context.onnx``** built in-memory. That is deliberate: CI runners have no NPU,
so ``test_vitisai_live.py`` skips there entirely, and any assertion whose only
real coverage were the live test would be silently uncovered (Implementation
Note 4.2). So the guards, the provider-choice policy, and the node-mix metric
are each pinned at unit level without an ONNX Runtime session and without the
vendor wheels.

The Observable this file has to pin (requirement 2.2, and the task's own):

- a run under **explicit ``npu``** either reports verified partitioning above
  threshold or **fails** - a mutant that always reports "verified", and one that
  proceeds on an unverifiable graph, must both die;
- **``auto``** records the weakness and proceeds - the same weak or unverifiable
  graph that fails under ``npu`` constructs successfully under ``auto``;
- session creation proves nothing, so **guard two** refuses a session that came
  back CPU-first rather than assuming the NPU served;
- the graph that runs is ``context.onnx`` - the compiled snapshot - never
  ``model.onnx`` (which is the CPU backend's graph).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import pytest
from onnx import TensorProto, helper

from npu_rag.embedding.errors import (
    EnvironmentError_,
    ExecutionError,
    PartitionShareTooLow,
    PreparationError,
)
from npu_rag.embedding.models.artifacts import (
    CONTEXT_FILENAME,
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
from npu_rag.embedding.providers.base import TransformerBackend
from npu_rag.embedding.providers.vitisai import (
    EP_CONTEXT_OP,
    MINIMUM_PARTITION_SHARE,
    VITISAI_PROVIDER,
    NpuSession,
    VitisAIBackend,
    read_node_partition_share,
)
from npu_rag.embedding.types import ExecutionMode, ProviderChoice

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "npu_rag"
    / "embedding"
    / "providers"
    / "vitisai.py"
)
MODULE_PACKAGE = "npu_rag.embedding"

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

#: A live MiniLM artifact reads exactly this node mix (design.md, the 2026-09-06
#: amendment): a residue of three cheap wrapper ops beside one EPContext node.
#: Measured against its real 251-node trunk the share is 0.988; the fabricated
#: fixtures below pair it with a 100-node trunk, so they read 0.97 by
#: construction - a fixture value, not the live measurement.
LIVE_MINILM_NODE_MIX = ("EPContext", "Cast", "Gather", "GatherND")


# --------------------------------------------------------------------------
# Stand-ins and fabricated graphs
# --------------------------------------------------------------------------


class FakeSession:
    """An ONNX Runtime session as this adapter is allowed to see it."""

    def __init__(
        self,
        *,
        providers: Sequence[str] = (VITISAI_PROVIDER,),
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
    """Records the provider options the adapter asked for, hands back a session.

    ``**options`` is deliberately open: it records anything *beyond*
    ``config_file`` the adapter tries to pass, so a test can assert that nothing
    beyond it reaches the factory (see the cache-option test below).
    """

    def __init__(self, session: FakeSession | None = None) -> None:
        self.session = session if session is not None else FakeSession()
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, model_path: Path, *, config_file: Path | None, **options: Any
    ) -> FakeSession:
        self.calls.append(
            {"model_path": model_path, "config_file": config_file, **options}
        )
        return self.session


class ExplodingFactory:
    """A session factory that must never be reached, or that fails on demand."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.called = False

    def __call__(
        self, model_path: Path, *, config_file: Path | None, **options: Any
    ) -> FakeSession:
        self.called = True
        if self.error is not None:
            raise self.error
        raise AssertionError("a session was built when none should have been")


def graph_file(path: Path, op_types: Sequence[str]) -> None:
    """Write a structurally-loadable ONNX graph carrying exactly ``op_types``.

    The nodes are not runnable and are never run - the metric under test counts
    ``op_type`` occurrences, which is all ``onnx.load`` needs to expose.
    """
    nodes = [
        helper.make_node(op, [f"in{i}"], [f"out{i}"], name=f"n{i}")
        for i, op in enumerate(op_types)
    ]
    inputs = [helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1])]
    last = f"out{len(op_types) - 1}" if op_types else "in0"
    outputs = [helper.make_tensor_value_info(last, TensorProto.FLOAT, [1])]
    model = helper.make_model(
        helper.make_graph(nodes, "g", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.save(model, str(path))


def artifact(
    tmp_path: Path,
    *,
    with_context: bool = True,
    context_ops: Sequence[str] | None = None,
    model_node_count: int = 100,
) -> PreparedArtifact:
    """A published NPU artifact directory.

    When ``context_ops`` is given the two graphs are real, so the real node-mix
    reader can be exercised; otherwise they are opaque bytes for tests that
    inject their own reader.
    """
    directory = tmp_path / "artifact"
    directory.mkdir(exist_ok=True)
    onnx_path = directory / ONNX_FILENAME
    context_path: Path | None = None
    if context_ops is not None:
        graph_file(onnx_path, ["Add"] * model_node_count)
        graph_file(directory / CONTEXT_FILENAME, list(context_ops))
        context_path = directory / CONTEXT_FILENAME
    else:
        onnx_path.write_bytes(b"not a real graph; no session is built from it")
        if with_context:
            context_path = directory / CONTEXT_FILENAME
            context_path.write_bytes(b"the compiled NPU snapshot")
    identity = ArtifactIdentity(
        model_id=PROFILE.model_id,
        revision="0" * 40,
        provider=ProviderChoice.NPU.value,
        compiled_seq_len=PROFILE.compiled_seq_len,
        batch_size=PROFILE.batch_size,
        onnxruntime_version="1.23.2",
        ryzen_ai_version="1.7.0",
        driver_version=None,
    )
    files = [ONNX_FILENAME]
    if context_path is not None:
        files.append(CONTEXT_FILENAME)
    return PreparedArtifact(
        directory=directory,
        onnx_path=onnx_path,
        context_path=context_path,
        dense_path=None,
        manifest=ArtifactManifest(
            identity=identity,
            observed_partition_share=None,
            files=tuple(files),
        ),
        reused=False,
        reason="built for this test",
        elapsed_seconds=0.0,
    )


def constant_reader(value: float | None) -> Callable[..., float | None]:
    def _read(*, context_path: Path, model_path: Path) -> float | None:
        return value

    return _read


def context_of(prepared: PreparedArtifact) -> Path:
    """The snapshot path, narrowed from ``Path | None`` for the real-reader tests
    that build one deliberately."""
    assert prepared.context_path is not None
    return prepared.context_path


def backend(
    tmp_path: Path,
    *,
    requested: ProviderChoice = ProviderChoice.NPU,
    session: FakeSession | None = None,
    factory: Any | None = None,
    share: float | None = 0.97,
    reader: Any | None = None,
    provider_registered: bool = True,
    with_context: bool = True,
) -> VitisAIBackend:
    chosen = factory if factory is not None else RecordingFactory(session)
    return VitisAIBackend(
        artifact(tmp_path, with_context=with_context),
        PROFILE,
        requested,
        session_factory=chosen,
        partition_reader=reader if reader is not None else constant_reader(share),
        provider_probe=lambda: provider_registered,
    )


def ids() -> npt.NDArray[np.int64]:
    return np.arange(PROFILE.batch_size * SEQ_LEN, dtype=np.int64).reshape(
        PROFILE.batch_size, SEQ_LEN
    )


ATTENDED = SEQ_LEN - 1


def mask() -> npt.NDArray[np.int64]:
    row = np.zeros(SEQ_LEN, dtype=np.int64)
    row[:ATTENDED] = 1
    return np.tile(row, (PROFILE.batch_size, 1))


# --------------------------------------------------------------------------
# The contract (design.md, TransformerBackend)
# --------------------------------------------------------------------------


def test_it_satisfies_the_transformer_backend_protocol(tmp_path: Path) -> None:
    assert isinstance(backend(tmp_path), TransformerBackend)


def test_it_reports_the_npu_provider(tmp_path: Path) -> None:
    assert backend(tmp_path).provider is ProviderChoice.NPU


def test_it_reports_in_process_execution(tmp_path: Path) -> None:
    """The isolated worker (task 4.4) reports ISOLATED; the in-process NPU
    adapter runs right here."""
    assert backend(tmp_path).execution_mode is ExecutionMode.IN_PROCESS


def test_it_populates_the_partition_share_from_the_node_mix(tmp_path: Path) -> None:
    """``npu_partition_share`` is the verified fraction, not ``None`` - the CPU
    backend's ``None`` reads as "not applicable", and this one was verified."""
    assert backend(tmp_path, share=0.97).npu_partition_share == 0.97


# --------------------------------------------------------------------------
# The provider is verified, never assumed (design.md, guards one and two)
# --------------------------------------------------------------------------


def test_guard_one_the_provider_must_be_registered_before_a_session_is_built(
    tmp_path: Path,
) -> None:
    """Reusing capability.py's probe. Nothing is constructed when the runtime
    cannot offer the provider - the check is *before*, not after."""
    factory = ExplodingFactory()

    with pytest.raises(EnvironmentError_):
        VitisAIBackend(
            artifact(tmp_path),
            PROFILE,
            ProviderChoice.NPU,
            session_factory=factory,
            partition_reader=constant_reader(0.97),
            provider_probe=lambda: False,
        )

    assert factory.called is False


def test_guard_two_a_session_that_came_back_cpu_first_is_refused(
    tmp_path: Path,
) -> None:
    """Requesting VitisAI when it is absent *succeeds* and runs on the CPU, so
    the session that came back is read, not assumed: a CPU-first session is not
    this backend's session and must be refused (2.2)."""
    session = FakeSession(providers=["CPUExecutionProvider"])

    with pytest.raises(ExecutionError) as caught:
        backend(tmp_path, session=session)

    assert "CPUExecutionProvider" in str(caught.value)


def test_guard_two_is_first_position_not_mere_membership(tmp_path: Path) -> None:
    """design.md mandates ``session.get_providers()[0] ==`` the provider. A
    session listing VitisAI *second* still runs the graph on the CPU, so a guard
    that only asked "is it in the list" would pass a CPU run as an NPU one."""
    session = FakeSession(providers=["CPUExecutionProvider", VITISAI_PROVIDER])

    with pytest.raises(ExecutionError) as caught:
        backend(tmp_path, session=session)

    assert "CPUExecutionProvider" in str(caught.value)


def test_guard_two_is_refused_under_auto_too(tmp_path: Path) -> None:
    """A session claiming the NPU while running on the CPU is dishonest whatever
    the selection: guard two refuses it rather than proceeding on a lie."""
    session = FakeSession(providers=["CPUExecutionProvider"])

    with pytest.raises(ExecutionError):
        backend(tmp_path, requested=ProviderChoice.AUTO, session=session)


def test_a_session_reporting_no_provider_at_all_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError):
        backend(tmp_path, session=FakeSession(providers=[]))


def test_bf16_targeting_is_supplied_through_the_config_file_provider_option(
    tmp_path: Path,
) -> None:
    """"reduced-precision targeting supplied through the provider configuration":
    the ``config_file`` option is what switches the device data type to
    bfloat16, so the exact path the backend was given must reach the factory -
    not ``None``, which would leave the factory to guess."""
    factory = RecordingFactory()
    config = tmp_path / "vaip_config.json"
    config.write_text("{}", encoding="utf-8")

    VitisAIBackend(
        artifact(tmp_path),
        PROFILE,
        ProviderChoice.NPU,
        session_factory=factory,
        config_file=config,
        partition_reader=constant_reader(0.97),
        provider_probe=lambda: True,
    )

    (call,) = factory.calls
    assert call["config_file"] == config


def test_loading_the_snapshot_passes_no_cache_options_to_the_factory(
    tmp_path: Path,
) -> None:
    """Measured on this machine (2026-09-06, review of this task): passing a
    ``cache_key`` when *loading* an EP-context snapshot whose baked-in key
    differs makes the Vitis AI EP call ``abort()`` - the interpreter dies, no
    exception is raised, and no test after it runs. Task 3.3 passes
    ``cache_dir``/``cache_key`` at *compile* time only. No live test will catch
    a regression in CI, so it is pinned here: nothing beyond ``config_file``
    reaches the factory, and the real factory's source names no cache option."""
    factory = RecordingFactory()

    backend(tmp_path, factory=factory)

    (call,) = factory.calls
    assert set(call) == {"model_path", "config_file"}, call
    assert [
        literal
        for literal in string_constants(source())
        if "cache_dir" in literal or "cache_key" in literal
    ] == []


def test_the_session_is_built_from_the_context_snapshot_not_the_trunk(
    tmp_path: Path,
) -> None:
    """The compiled snapshot is what runs on the NPU; ``model.onnx`` is the CPU
    backend's FP32 graph (task 4.2). A mutant opening the trunk is caught here."""
    prepared = artifact(tmp_path)
    factory = RecordingFactory()

    VitisAIBackend(
        prepared,
        PROFILE,
        ProviderChoice.NPU,
        session_factory=factory,
        partition_reader=constant_reader(0.97),
        provider_probe=lambda: True,
    )

    (call,) = factory.calls
    assert call["model_path"] == prepared.context_path
    assert call["model_path"].name == CONTEXT_FILENAME
    assert call["model_path"] != prepared.onnx_path


# --------------------------------------------------------------------------
# The partition verdict under explicit npu (the Observable)
# --------------------------------------------------------------------------


def test_explicit_npu_with_a_verified_share_above_threshold_constructs(
    tmp_path: Path,
) -> None:
    adapter = backend(
        tmp_path, requested=ProviderChoice.NPU, share=MINIMUM_PARTITION_SHARE + 0.1
    )

    assert adapter.npu_partition_share == pytest.approx(MINIMUM_PARTITION_SHARE + 0.1)


def test_explicit_npu_with_an_unverifiable_share_fails(tmp_path: Path) -> None:
    """The mutant that proceeds on an unverifiable graph under explicit npu dies
    here. ``None`` is *unverified*, and proceeding would reintroduce the silent
    degradation requirement 2 exists to prevent."""
    with pytest.raises(PartitionShareTooLow) as caught:
        backend(tmp_path, requested=ProviderChoice.NPU, share=None)

    assert caught.value.unverifiable is True
    assert caught.value.observed_share is None
    assert caught.value.provider is ProviderChoice.NPU
    assert caught.value.model_id == PROFILE.model_id


def test_explicit_npu_below_threshold_fails_as_a_measured_low(tmp_path: Path) -> None:
    """Measured low is distinct from unverifiable: a number was read and it is
    below the threshold, so the graph is mostly on the CPU."""
    low = MINIMUM_PARTITION_SHARE / 2

    with pytest.raises(PartitionShareTooLow) as caught:
        backend(tmp_path, requested=ProviderChoice.NPU, share=low)

    assert caught.value.unverifiable is False
    assert caught.value.observed_share == pytest.approx(low)
    assert caught.value.minimum_share == pytest.approx(MINIMUM_PARTITION_SHARE)


def test_explicit_npu_exactly_at_the_threshold_is_accepted(tmp_path: Path) -> None:
    """The boundary is inclusive: at the threshold the graph is not *below* it."""
    adapter = backend(
        tmp_path, requested=ProviderChoice.NPU, share=MINIMUM_PARTITION_SHARE
    )

    assert adapter.npu_partition_share == pytest.approx(MINIMUM_PARTITION_SHARE)


def test_the_threshold_is_a_reachable_fraction(tmp_path: Path) -> None:
    """A threshold of 1.0 would reject every real offload (a live MiniLM reads
    ~0.97); one of 0.0 would accept a graph wholly on the CPU. It must sit
    strictly between."""
    assert 0.0 < MINIMUM_PARTITION_SHARE < 1.0


# --------------------------------------------------------------------------
# The partition verdict under auto: recorded, not raised
# --------------------------------------------------------------------------


def test_auto_proceeds_on_an_unverifiable_graph_and_records_it(
    tmp_path: Path,
) -> None:
    """The same ``None`` that fails under explicit npu is a recorded warning
    under auto: the caller already accepted substitution, so execution proceeds
    and the share stays ``None`` for the service to surface."""
    adapter = backend(tmp_path, requested=ProviderChoice.AUTO, share=None)

    assert adapter.npu_partition_share is None
    assert adapter.provider is ProviderChoice.NPU


def test_auto_proceeds_below_threshold_and_records_the_measured_share(
    tmp_path: Path,
) -> None:
    low = MINIMUM_PARTITION_SHARE / 2
    adapter = backend(tmp_path, requested=ProviderChoice.AUTO, share=low)

    assert adapter.npu_partition_share == pytest.approx(low)


def test_auto_with_a_good_share_reports_it_verified(tmp_path: Path) -> None:
    adapter = backend(tmp_path, requested=ProviderChoice.AUTO, share=0.97)

    assert adapter.npu_partition_share == 0.97


def test_the_same_weak_graph_fails_under_npu_but_proceeds_under_auto(
    tmp_path: Path,
) -> None:
    """The provider-choice policy in one assertion: explicit npu is a failure,
    auto is a recorded warning, for the identical unverifiable graph."""
    with pytest.raises(PartitionShareTooLow):
        backend(tmp_path, requested=ProviderChoice.NPU, share=None)

    proceeded = backend(tmp_path, requested=ProviderChoice.AUTO, share=None)
    assert proceeded.npu_partition_share is None


# --------------------------------------------------------------------------
# The reader reads context.onnx against model.onnx, not the other way round
# --------------------------------------------------------------------------


def test_the_reader_is_asked_about_the_context_and_the_trunk(tmp_path: Path) -> None:
    prepared = artifact(tmp_path)
    seen: dict[str, Path] = {}

    def reader(*, context_path: Path, model_path: Path) -> float | None:
        seen["context"] = context_path
        seen["model"] = model_path
        return 0.97

    VitisAIBackend(
        prepared,
        PROFILE,
        ProviderChoice.NPU,
        session_factory=RecordingFactory(),
        partition_reader=reader,
        provider_probe=lambda: True,
    )

    assert seen["context"] == prepared.context_path
    assert seen["model"] == prepared.onnx_path


# --------------------------------------------------------------------------
# The real node-mix reader
# --------------------------------------------------------------------------


def test_the_live_minilm_node_mix_reads_a_high_share(tmp_path: Path) -> None:
    """{EPContext:1, Cast:1, Gather:1, GatherND:1} against a fabricated 100-node
    trunk: a residue of three cheap nodes leaves 0.97 of it offloaded. (The real
    MiniLM trunk has 251 nodes and reads 0.988 - see ``test_vitisai_live.py``.)"""
    prepared = artifact(
        tmp_path, context_ops=LIVE_MINILM_NODE_MIX, model_node_count=100
    )

    share = read_node_partition_share(
        context_path=context_of(prepared), model_path=prepared.onnx_path
    )

    assert share == pytest.approx(0.97)
    assert share is not None and share >= MINIMUM_PARTITION_SHARE


def test_a_context_with_no_ep_context_node_is_unverifiable(tmp_path: Path) -> None:
    """The mutant that reads a graph with *zero* EPContext nodes as fully
    offloaded dies here: no EPContext node means nothing was compiled to the
    NPU, which is unverifiable, never 100%."""
    prepared = artifact(
        tmp_path, context_ops=("Cast", "Gather", "MatMul"), model_node_count=100
    )

    assert (
        read_node_partition_share(
            context_path=context_of(prepared), model_path=prepared.onnx_path
        )
        is None
    )


def test_a_zero_ep_context_graph_fails_under_explicit_npu(tmp_path: Path) -> None:
    """End to end through the real reader: a graph the compiler did not offload
    must not reach the NPU backend as a success."""
    prepared = artifact(
        tmp_path, context_ops=("Cast", "MatMul", "Add"), model_node_count=100
    )

    with pytest.raises(PartitionShareTooLow) as caught:
        VitisAIBackend(
            prepared,
            PROFILE,
            ProviderChoice.NPU,
            session_factory=RecordingFactory(),
            provider_probe=lambda: True,
        )

    assert caught.value.unverifiable is True


def test_a_mostly_cpu_context_reads_below_threshold(tmp_path: Path) -> None:
    """One EPContext node but a large CPU residue: most of the original graph is
    still loose on the CPU, so the share is low."""
    residue = ["MatMul"] * 80
    prepared = artifact(
        tmp_path, context_ops=[EP_CONTEXT_OP, *residue], model_node_count=100
    )

    share = read_node_partition_share(
        context_path=context_of(prepared), model_path=prepared.onnx_path
    )

    assert share is not None
    assert share < MINIMUM_PARTITION_SHARE


def test_more_residue_than_original_nodes_clamps_to_zero(tmp_path: Path) -> None:
    """Compiler glue is not a subset of the original graph, so the residue can
    in principle exceed the original count; the share floors at 0.0 rather than
    going negative, which reads correctly as "mostly on the CPU"."""
    prepared = artifact(
        tmp_path, context_ops=[EP_CONTEXT_OP, *(["Cast"] * 20)], model_node_count=5
    )

    share = read_node_partition_share(
        context_path=context_of(prepared), model_path=prepared.onnx_path
    )

    assert share == 0.0


def test_a_missing_context_file_is_unverifiable(tmp_path: Path) -> None:
    prepared = artifact(tmp_path, context_ops=LIVE_MINILM_NODE_MIX)
    context = context_of(prepared)
    context.unlink()

    assert (
        read_node_partition_share(
            context_path=context, model_path=prepared.onnx_path
        )
        is None
    )


def test_a_missing_trunk_gives_no_denominator_and_is_unverifiable(
    tmp_path: Path,
) -> None:
    prepared = artifact(tmp_path, context_ops=LIVE_MINILM_NODE_MIX)
    prepared.onnx_path.unlink()

    assert (
        read_node_partition_share(
            context_path=context_of(prepared), model_path=prepared.onnx_path
        )
        is None
    )


def test_a_corrupt_context_file_is_unverifiable(tmp_path: Path) -> None:
    prepared = artifact(tmp_path, context_ops=LIVE_MINILM_NODE_MIX)
    context = context_of(prepared)
    context.write_bytes(b"not a protobuf")

    assert (
        read_node_partition_share(
            context_path=context, model_path=prepared.onnx_path
        )
        is None
    )


# --------------------------------------------------------------------------
# A missing context snapshot cannot be run as the NPU graph
# --------------------------------------------------------------------------


def test_an_artifact_without_a_context_snapshot_is_refused(tmp_path: Path) -> None:
    """A CPU artifact has no ``context.onnx``; the NPU backend cannot run one
    that is not there, and says so rather than reaching for ``model.onnx``."""
    prepared = artifact(tmp_path, with_context=False)
    factory = ExplodingFactory()

    with pytest.raises(ExecutionError) as caught:
        VitisAIBackend(
            prepared,
            PROFILE,
            ProviderChoice.NPU,
            session_factory=factory,
            partition_reader=constant_reader(0.97),
            provider_probe=lambda: True,
        )

    # The refusal happens *before* any session is built: a backend that fell
    # back to ``model.onnx`` would have reached the factory, and its exploding
    # assertion would have been folded into an ExecutionError that looks the
    # same from outside - which is why the factory is checked, not just the type.
    assert factory.called is False
    assert CONTEXT_FILENAME in str(caught.value)
    assert caught.value.provider is ProviderChoice.NPU


# --------------------------------------------------------------------------
# Executing the compiled graph (token embeddings only)
# --------------------------------------------------------------------------


def test_the_fixtures_can_tell_a_real_batch_from_a_fabricated_one() -> None:
    """The non-vacuity check the feed assertion depends on (Implementation Note
    4.2): the mask must be distinguishable from the arrays a mutant reaches for."""
    token_ids, attention = ids(), mask()

    for label, array in ((INPUT_IDS, token_ids), (ATTENTION_MASK, attention)):
        assert not np.array_equal(array, np.ones_like(array)), label
        assert not np.array_equal(array, np.zeros_like(array)), label
        assert not np.array_equal(array, array[:, ::-1]), label
    assert not np.array_equal(token_ids, attention)
    assert 0 < int(attention.sum()) < attention.size


def test_run_feeds_the_context_session_the_callers_own_arrays(
    tmp_path: Path,
) -> None:
    session = FakeSession()
    token_ids, attention = ids(), mask()

    backend(tmp_path, session=session).run(token_ids, attention)

    (output_names, feed), = session.calls
    assert output_names is None
    assert set(feed) == {INPUT_IDS, ATTENTION_MASK}
    np.testing.assert_array_equal(feed[INPUT_IDS], token_ids)
    np.testing.assert_array_equal(feed[ATTENTION_MASK], attention)
    assert feed[INPUT_IDS].dtype == np.int64
    assert feed[ATTENTION_MASK].dtype == np.int64


def test_run_returns_the_graphs_own_output_rather_than_one_it_made_up(
    tmp_path: Path,
) -> None:
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
    session = FakeSession()
    adapter = backend(tmp_path, session=session)
    wrong = np.zeros(shape, dtype=np.int64)

    with pytest.raises(ExecutionError):
        adapter.run(wrong, np.ones(shape, dtype=np.int64))

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


def test_an_output_that_is_not_three_dimensional_is_refused(tmp_path: Path) -> None:
    produced = np.zeros((PROFILE.batch_size, SEQ_LEN), dtype=np.float32)

    with pytest.raises(ExecutionError):
        backend(tmp_path, session=FakeSession(output=produced)).run(ids(), mask())


def test_a_graph_returning_more_than_one_tensor_is_refused(tmp_path: Path) -> None:
    session = FakeSession()
    expected = np.zeros((PROFILE.batch_size, SEQ_LEN, HIDDEN), np.float32)
    session.run = lambda names, feed: [expected, expected]  # type: ignore[method-assign]

    with pytest.raises(ExecutionError):
        backend(tmp_path, session=session).run(ids(), mask())


def test_a_failure_inside_the_graph_is_an_execution_error(tmp_path: Path) -> None:
    session = FakeSession(raises=RuntimeError("RUNTIME_EXCEPTION"))

    with pytest.raises(ExecutionError) as caught:
        backend(tmp_path, session=session).run(ids(), mask())

    assert caught.value.provider is ProviderChoice.NPU
    assert caught.value.model_id == PROFILE.model_id
    assert "RUNTIME_EXCEPTION" in str(caught.value)


# --------------------------------------------------------------------------
# Requested selection: npu or auto only
# --------------------------------------------------------------------------


def test_a_cpu_selection_is_rejected(tmp_path: Path) -> None:
    """This is the NPU adapter. A ``cpu`` selection belongs to `CpuBackend`, and
    building this one for it would be a category error resolution never makes."""
    with pytest.raises(ValueError):
        VitisAIBackend(
            artifact(tmp_path),
            PROFILE,
            ProviderChoice.CPU,
            session_factory=RecordingFactory(),
            partition_reader=constant_reader(0.97),
            provider_probe=lambda: True,
        )


# --------------------------------------------------------------------------
# Structural: token embeddings only, no printing, no later layer
# --------------------------------------------------------------------------


def source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def imported_names(source_text: str, package: str) -> list[str]:
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


def defined_names(source_text: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(source_text))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def test_the_backend_neither_pools_nor_normalizes() -> None:
    """Token embeddings only; post-processing is task 5.2's, on the service side
    of the port so it cannot differ per backend."""
    assert [
        name
        for name in called_names(source())
        if name.startswith(("np.mean", "np.sum", "np.linalg", "numpy."))
    ] == []
    assert sorted(
        name
        for name in defined_names(source())
        for banned in ("pool", "normal", "dense", "embed")
        if banned in name.lower()
    ) == []


def test_the_backend_neither_prints_nor_logs() -> None:
    """Warnings under auto are *recorded* as data, never printed."""
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


def test_onnx_runtime_is_imported_lazily_so_the_package_imports_without_it() -> None:
    """The package must import in an environment that has never seen the vendor
    wheels; ONNX Runtime is reached inside the session factory, not at module
    scope."""
    module_level = {
        alias.name
        for node in ast.parse(source()).body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.parse(source()).body
        if isinstance(node, ast.ImportFrom)
    }

    assert [name for name in module_level if name.startswith("onnxruntime")] == []


def test_the_backend_imports_nothing_from_a_later_layer() -> None:
    assert [
        name
        for name in imported_names(source(), MODULE_PACKAGE)
        for layer in ("service", "bench")
        if name == f"{MODULE_PACKAGE}.{layer}"
        or name.startswith(f"{MODULE_PACKAGE}.{layer}.")
    ] == []


def test_the_backend_is_reachable_from_the_providers_package() -> None:
    module: Any = __import__(
        "npu_rag.embedding.providers.vitisai", fromlist=["VitisAIBackend"]
    )

    assert module.VitisAIBackend is VitisAIBackend


def test_the_adapter_defines_its_own_session_seam_expressing_provider_options() -> None:
    """Task 4.2's CPU ``SessionFactory`` cannot express ``provider_options``,
    which is what makes "the CPU reduces precision nowhere" structural. The NPU
    path needs ``config_file`` (bfloat16), so it defines its own factory rather
    than widening the CPU one - and that factory carries the option."""
    assert NpuSession.__module__ == "npu_rag.embedding.providers.vitisai"
    assert "provider_options" in source()
    assert re.search(r"config_file", source()) is not None
    # It is not the CPU backend's factory, widened.
    assert "from npu_rag.embedding.providers.cpu import" not in source()
    assert "providers.cpu" not in source()
