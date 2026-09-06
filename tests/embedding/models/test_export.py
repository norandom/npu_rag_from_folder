"""ONNX export at a fixed sequence length (task 3.2).

**Nothing here downloads or exports a real model.** The transformer is reached
through an injected `TrunkExporter`, exactly as task 3.1 reached the Hub through
an injected repository client, because a real export means a gigabyte of weights
and minutes of compilation. The live half - one real export, proving the shape
contract against ONNX Runtime - lives in ``test_export_live.py``.

The ONNX graphs the doubles write are **real** graphs, built with ``onnx.helper``
and loaded by a real ONNX Runtime session. That matters: the thing under test is
a shape-and-dtype contract that only a real runtime enforces, and a mocked
session would let a graph with an unpinned ``sequence_length`` dimension pass a
test suite and then fail on the NPU compiler.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from npu_rag.embedding.errors import (
    EmbeddingRuntimeError,
    LicenseAcceptanceRequired,
    PreparationError,
)
from npu_rag.embedding.models.acquire import AcquiredModel
from npu_rag.embedding.models.export import (
    ACTIVATION_PREFIX,
    ATTENTION_MASK,
    BIAS_PREFIX,
    DENSE_FILENAME,
    DENSE_ORDER_KEY,
    EXPORT_STAGE,
    INPUT_IDS,
    ONNX_FILENAME,
    TOKEN_EMBEDDINGS,
    WEIGHT_PREFIX,
    DenseStage,
    ExportedModel,
    TorchTrunkExporter,
    export_model,
)
from npu_rag.embedding.profiles import ModelProfile, profile_for
from npu_rag.embedding.reporting import ProgressUpdate

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "npu_rag"
    / "embedding"
    / "models"
    / "export.py"
)
#: The *package* the module under test lives in, not the module: that is what a
#: relative import's level counts up from.
MODULE_PACKAGE = "npu_rag.embedding.models"

# design.md -> Architecture -> dependency direction:
# types, errors -> reporting -> profiles -> environment -> models -> providers
# -> service -> bench
LAYERS_RIGHT_OF_MODELS = ("providers", "service", "bench")

GEMMA = profile_for("embeddinggemma-300m")
BGE = profile_for("bge-large-en-v1.5")

REVISION = "0" * 40


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
    """The guard above is only worth having if a different import spelling
    cannot side-step it."""
    imported = absolute_imports_of(statement, MODULE_PACKAGE)

    assert any(
        name.startswith(f"npu_rag.embedding.{layer}")
        for name in imported
        for layer in LAYERS_RIGHT_OF_MODELS
    )


# --------------------------------------------------------------------------
# Test doubles: real ONNX graphs, written to order
# --------------------------------------------------------------------------


def write_graph(
    destination: Path,
    *,
    batch: int | str = 1,
    sequence: int | str = 512,
    hidden: int = 8,
    table_type: int = TensorProto.FLOAT,
    output_type: int = TensorProto.FLOAT,
    include_mask: bool = True,
    bad_op: bool = False,
    extra_input: str | None = None,
    output_name: str = TOKEN_EMBEDDINGS,
    extra_output: str | None = None,
) -> None:
    """A genuine, runnable ONNX graph with the trunk's input and output shape.

    ``Gather`` over a small embedding table gives the right rank and dtype for a
    few kilobytes, and the attention mask is genuinely multiplied in rather than
    declared and ignored, so ONNX Runtime really does enforce its shape - which
    is the whole point of the fixed-shape contract.

    Every knob exists to build a graph that is wrong in exactly one way:
    ``batch``/``sequence`` accept a string to leave a dimension symbolic,
    ``table_type`` plants half-precision weights, ``include_mask`` drops an
    expected input, ``bad_op`` names an operator that parses as protobuf and
    then has no kernel, which is how a graph gets past ``onnx.load`` and dies
    at session construction. ``output_name`` renames the single output and
    ``extra_output`` adds a second one beside it - between them they build the
    two ways a graph can carry more or other than the trunk contract, which is
    what an export that folded a pooler back into the graph would look like.
    """
    vocab = 4
    table = numpy_helper.from_array(
        np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden),
        name="table_raw",
    )
    initializers = [table]
    nodes = []

    if table_type != TensorProto.FLOAT:
        # Half-precision weights that are cast up before use: what a graph whose
        # weights were quantised or downcast actually looks like from the
        # outside, since its output stays float32.
        initializers = [
            numpy_helper.from_array(
                np.arange(vocab * hidden)
                .reshape(vocab, hidden)
                .astype(onnx.helper.tensor_dtype_to_np_dtype(table_type)),
                name="table_raw",
            )
        ]
        nodes.append(
            helper.make_node(
                "Cast", ["table_raw"], ["table"], to=TensorProto.FLOAT
            )
        )
    else:
        nodes.append(helper.make_node("Identity", ["table_raw"], ["table"]))

    nodes.append(
        helper.make_node(
            "NoSuchOperator" if bad_op else "Gather",
            ["table", INPUT_IDS],
            ["gathered"],
        )
    )

    inputs = [
        helper.make_tensor_value_info(
            INPUT_IDS, TensorProto.INT64, [batch, sequence]
        )
    ]
    if include_mask:
        inputs.append(
            helper.make_tensor_value_info(
                ATTENTION_MASK, TensorProto.INT64, [batch, sequence]
            )
        )
        initializers.append(
            numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes")
        )
        nodes.append(
            helper.make_node(
                "Cast", [ATTENTION_MASK], ["mask_f"], to=TensorProto.FLOAT
            )
        )
        nodes.append(
            helper.make_node("Unsqueeze", ["mask_f", "axes"], ["mask_3d"])
        )
        nodes.append(helper.make_node("Mul", ["gathered", "mask_3d"], ["scaled"]))
        last = "scaled"
    else:
        last = "gathered"

    if output_type != TensorProto.FLOAT:
        nodes.append(
            helper.make_node("Cast", [last], [output_name], to=output_type)
        )
    else:
        nodes.append(helper.make_node("Identity", [last], [output_name]))

    outputs = [
        helper.make_tensor_value_info(
            output_name, output_type, [batch, sequence, hidden]
        )
    ]
    if extra_output is not None:
        nodes.append(helper.make_node("Identity", [last], [extra_output]))
        outputs.append(
            helper.make_tensor_value_info(
                extra_output, output_type, [batch, sequence, hidden]
            )
        )

    if extra_input is not None:
        inputs.append(
            helper.make_tensor_value_info(
                extra_input, TensorProto.INT64, [batch, sequence]
            )
        )

    graph = helper.make_graph(
        nodes,
        "trunk",
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))


def dense_stage(
    name: str, in_features: int, out_features: int, *, bias: bool = False
) -> DenseStage:
    weight = np.full((out_features, in_features), 0.5, dtype=np.float32)
    return DenseStage(
        name=name,
        weight=weight,
        bias=np.zeros(out_features, dtype=np.float32) if bias else None,
        activation="torch.nn.modules.linear.Identity",
    )


class FakeExporter:
    """A `TrunkExporter` that writes a graph to order and never uses torch."""

    def __init__(
        self,
        *,
        stages: Sequence[DenseStage] = (),
        graph: dict[str, Any] | None = None,
        graph_error: BaseException | None = None,
        dense_error: BaseException | None = None,
        partial_bytes: bool = False,
    ) -> None:
        self.stages = tuple(stages)
        self.graph = graph or {}
        self.graph_error = graph_error
        self.dense_error = dense_error
        self.partial_bytes = partial_bytes
        self.graph_calls: list[dict[str, Any]] = []
        self.dense_calls: list[Path] = []

    def export_graph(
        self,
        *,
        source: Path,
        destination: Path,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        self.graph_calls.append(
            {
                "source": source,
                "destination": destination,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
            }
        )
        if self.partial_bytes:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"half a graph")
        if self.graph_error is not None:
            raise self.graph_error
        write_graph(
            destination,
            **{
                "batch": batch_size,
                "sequence": sequence_length,
                **self.graph,
            },
        )

    def read_dense_stages(self, *, source: Path) -> Sequence[DenseStage]:
        self.dense_calls.append(source)
        if self.dense_error is not None:
            raise self.dense_error
        return self.stages


def acquired_for(profile: ModelProfile, path: Path) -> AcquiredModel:
    path.mkdir(parents=True, exist_ok=True)
    return AcquiredModel(
        model_id=profile.model_id, revision=REVISION, local_path=path
    )


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    return directory


@pytest.fixture
def destination(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


# --------------------------------------------------------------------------
# The happy path, and the shape the profile declares
# --------------------------------------------------------------------------


def test_export_writes_the_graph_and_reports_where_it_landed(
    source_dir: Path, destination: Path
) -> None:
    exporter = FakeExporter()

    result = export_model(
        BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
    )

    assert isinstance(result, ExportedModel)
    assert result.onnx_path == destination / ONNX_FILENAME
    assert result.onnx_path.is_file()
    assert result.model_id == BGE.model_id
    assert result.revision == REVISION
    assert result.batch_size == BGE.batch_size
    assert result.compiled_seq_len == BGE.compiled_seq_len


def test_the_graph_is_exported_at_the_profiles_compiled_length_and_batch(
    source_dir: Path, destination: Path
) -> None:
    """The compiled length is the profile's, not the architectural limit.

    ``bge-large-en-v1.5`` is the one candidate whose two numbers coincide, so
    the assertion is made against EmbeddingGemma, where 512 and 2048 differ and
    reading the wrong field is a live possibility.
    """
    exporter = FakeExporter(
        stages=[dense_stage("2_Dense", 768, 3072), dense_stage("3_Dense", 3072, 768)],
        graph={"hidden": 768},
    )

    export_model(
        GEMMA, acquired_for(GEMMA, source_dir), destination, exporter=exporter
    )

    assert len(exporter.graph_calls) == 1
    call = exporter.graph_calls[0]
    assert call["sequence_length"] == GEMMA.compiled_seq_len == 512
    assert call["sequence_length"] != GEMMA.architectural_context_limit
    assert call["batch_size"] == GEMMA.batch_size
    assert call["source"] == source_dir


def test_the_published_graph_declares_exactly_the_profiles_shape(
    source_dir: Path, destination: Path
) -> None:
    import onnxruntime as ort  # type: ignore[import-untyped]

    export_model(
        BGE, acquired_for(BGE, source_dir), destination, exporter=FakeExporter()
    )

    session = ort.InferenceSession(
        str(destination / ONNX_FILENAME), providers=["CPUExecutionProvider"]
    )
    shapes = {i.name: i.shape for i in session.get_inputs()}
    assert shapes == {
        INPUT_IDS: [BGE.batch_size, BGE.compiled_seq_len],
        ATTENTION_MASK: [BGE.batch_size, BGE.compiled_seq_len],
    }


def test_the_published_graph_rejects_any_other_shape(
    source_dir: Path, destination: Path
) -> None:
    """Task 3.2's Observable, first half. Verified on this machine during task
    1.3's spike: a fixed-shape graph answers a mismatched input with
    ``InvalidArgument`` rather than quietly reshaping it."""
    import onnxruntime as ort

    export_model(
        BGE, acquired_for(BGE, source_dir), destination, exporter=FakeExporter()
    )
    session = ort.InferenceSession(
        str(destination / ONNX_FILENAME), providers=["CPUExecutionProvider"]
    )
    wrong = BGE.compiled_seq_len // 2

    with pytest.raises(Exception) as caught:
        session.run(
            None,
            {
                INPUT_IDS: np.zeros((BGE.batch_size, wrong), dtype=np.int64),
                ATTENTION_MASK: np.ones((BGE.batch_size, wrong), dtype=np.int64),
            },
        )

    assert "INVALID_ARGUMENT" in str(caught.value).upper()


# --------------------------------------------------------------------------
# Rejecting a graph that is not actually pinned
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("graph", "expected"),
    [
        ({"sequence": "sequence_length"}, "sequence_length"),
        ({"batch": "batch_size"}, "batch_size"),
        ({"sequence": 256}, "256"),
        ({"batch": 2}, "2"),
    ],
)
def test_a_graph_whose_shape_is_not_the_profiles_is_refused(
    source_dir: Path,
    destination: Path,
    graph: dict[str, Any],
    expected: str,
) -> None:
    """A symbolic dimension is the dangerous one: it loads, it runs, and the NPU
    compiler needs a concrete value it will never find."""
    exporter = FakeExporter(graph=graph)

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert caught.value.model_id == BGE.model_id
    assert expected in str(caught.value)
    # The *input* is what was wrong, and the report has to say so. Without this
    # the output-shape check downstream would mask a deleted input check: both
    # branches refuse the graph, but only one of them tells the operator which
    # tensor to look at.
    assert INPUT_IDS in str(caught.value)
    assert not (destination / ONNX_FILENAME).exists()


def test_a_graph_missing_an_expected_input_is_refused(
    source_dir: Path, destination: Path
) -> None:
    """design.md's `TransformerBackend.run` takes token ids and an attention
    mask, so a trunk that does not accept both is not the trunk the backends
    will call."""
    exporter = FakeExporter(graph={"include_mask": False})

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert ATTENTION_MASK in str(caught.value)
    assert INPUT_IDS in str(caught.value)
    # Diagnosed, not merely survived: without the signature check the missing
    # input surfaces as a bare KeyError from the shape loop, which names the
    # tensor by accident and explains nothing.
    assert "KeyError" not in str(caught.value)


def test_a_graph_taking_an_input_the_backend_will_not_supply_is_refused(
    source_dir: Path, destination: Path
) -> None:
    """The BERT-derived candidates' ``forward`` also accepts ``token_type_ids``.
    Letting it become a third graph input would compile a model the backends
    cannot feed, so the signature is fixed in both directions."""
    exporter = FakeExporter(graph={"extra_input": "token_type_ids"})

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert "token_type_ids" in str(caught.value)


def test_a_graph_carrying_reduced_precision_weights_is_refused(
    source_dir: Path, destination: Path
) -> None:
    """design.md, TransformerBackend: "Precision is a property of the backend,
    not the export." The exported graph is FP32; BF16 targeting happens later,
    at session construction, where the EP performs the cast. A graph arriving
    here with half-precision weights has been quantised somewhere it should not
    have been, and its output dtype does not reveal that."""
    exporter = FakeExporter(graph={"table_type": TensorProto.FLOAT16})

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert "FLOAT16" in str(caught.value).upper()
    assert not (destination / ONNX_FILENAME).exists()


def test_a_graph_that_does_not_emit_float32_token_embeddings_is_refused(
    source_dir: Path, destination: Path
) -> None:
    exporter = FakeExporter(graph={"output_type": TensorProto.FLOAT16})

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert TOKEN_EMBEDDINGS in str(caught.value)


def test_a_graph_emitting_more_than_the_trunks_one_output_is_refused(
    source_dir: Path, destination: Path
) -> None:
    """The trunk contract is exactly one tensor, and the count is load-bearing.

    design.md's decision "Post-process pooling, Dense, and normalization outside
    the ONNX graph" puts those three stages in NumPy so requirement 3.5 holds by
    construction for every backend. A graph that also emits a pooled vector is a
    graph that has folded one of them back in, and accepting it would let a
    backend return the graph's pooling for one provider and NumPy's for another
    - the equivalence surface 5.4 exists to keep to the transformer alone.
    """
    exporter = FakeExporter(graph={"extra_output": "pooled"})

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert "pooled" in str(caught.value)
    assert not (destination / ONNX_FILENAME).exists()


def test_a_graph_whose_single_output_is_misnamed_is_refused(
    source_dir: Path, destination: Path
) -> None:
    """One output of the right shape and the wrong name is still not the
    contract: task 5.2 and every backend fetch this tensor by name."""
    exporter = FakeExporter(graph={"output_name": "last_hidden_state"})

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert "last_hidden_state" in str(caught.value)
    assert TOKEN_EMBEDDINGS in str(caught.value)
    assert not (destination / ONNX_FILENAME).exists()


def test_a_graph_onnx_runtime_cannot_load_is_refused(
    source_dir: Path, destination: Path
) -> None:
    """A graph can be well-formed protobuf and still have no kernel. The
    verification builds a real session precisely because a session is what the
    backends and the NPU compiler will build."""
    exporter = FakeExporter(graph={"bad_op": True})

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert caught.value.model_id == BGE.model_id
    assert "ONNX Runtime" in str(caught.value)
    assert not (destination / ONNX_FILENAME).exists()


def test_the_hidden_size_is_read_from_the_exported_graph(
    source_dir: Path, destination: Path
) -> None:
    exporter = FakeExporter(graph={"hidden": 17})

    result = export_model(
        BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
    )

    assert result.hidden_size == 17


# --------------------------------------------------------------------------
# The Dense stage - the silent-wrong-vectors failure
# --------------------------------------------------------------------------


def test_dense_weights_are_persisted_for_a_profile_that_declares_the_stage(
    source_dir: Path, destination: Path
) -> None:
    """Task 3.2's Observable, second half.

    research.md: the ONNX export covers only the transformer trunk, and
    EmbeddingGemma's sentence pipeline is Pooling -> Dense -> Normalize. A
    missing Dense stage produces correctly shaped, correctly normalised,
    semantically wrong vectors that no shape check catches.
    """
    stages = [dense_stage("2_Dense", 768, 3072), dense_stage("3_Dense", 3072, 768)]
    exporter = FakeExporter(stages=stages, graph={"hidden": 768})

    result = export_model(
        GEMMA, acquired_for(GEMMA, source_dir), destination, exporter=exporter
    )

    assert result.dense_path == destination / DENSE_FILENAME
    assert result.dense_path is not None
    assert result.dense_path.is_file()

    with np.load(result.dense_path) as loaded:
        order = [str(name) for name in loaded[DENSE_ORDER_KEY]]
        assert order == ["2_Dense", "3_Dense"]
        first = loaded[f"{WEIGHT_PREFIX}{order[0]}"]
        second = loaded[f"{WEIGHT_PREFIX}{order[1]}"]

    assert first.shape == (3072, 768)
    assert second.shape == (768, 3072)
    assert first.dtype == np.float32


def test_the_persisted_dense_order_is_the_pipeline_order_not_a_set(
    source_dir: Path, destination: Path
) -> None:
    """Reversing the two projections keeps every shape individually plausible
    and destroys the vectors, so the order is recorded rather than inferred.

    The stage names are deliberately in **non-alphabetical** pipeline order:
    ``z_first`` runs before ``a_second``. Every real pipeline seen so far -
    ``2_Dense`` then ``3_Dense`` - happens to be in sorted order, so a
    ``sorted()`` slipped into `_write_dense` would be indistinguishable from
    pipeline order in any test that used realistic names, and would survive.
    It must not: task 5.2 iterates `DENSE_ORDER_KEY` to decide the sequence the
    projections are applied in, so a sorted ``order`` array beside correctly
    named weights applies 768 -> 3072 -> 768 backwards. The result keeps its
    shape and keeps its unit norm and means nothing - the exact failure this
    module exists to prevent.
    """
    stages = [dense_stage("z_first", 4, 6), dense_stage("a_second", 6, 4)]
    exporter = FakeExporter(stages=stages, graph={"hidden": 4})
    profile = _profile_with(BGE, has_dense_stage=True, dimension=4)

    result = export_model(
        profile, acquired_for(profile, source_dir), destination, exporter=exporter
    )

    assert result.dense_path is not None
    with np.load(result.dense_path) as loaded:
        order = [str(name) for name in loaded[DENSE_ORDER_KEY]]

        # Pinned as a sequence, not a set: sorting these reverses them.
        assert order == ["z_first", "a_second"]
        assert order != sorted(order)

        # And the weights, looked up *through* that order, must chain. This is
        # what task 5.2 will actually do, so it is what the test does: a
        # mis-ordered order array makes the widths stop meeting.
        widths = [loaded[f"{WEIGHT_PREFIX}{name}"].shape for name in order]
        assert widths == [(6, 4), (4, 6)]
        assert widths[0][1] == 4, "the first stage consumes the trunk's width"
        assert widths[0][0] == widths[1][1], "the stages must chain in order"
        assert widths[1][0] == profile.dimension


def test_a_bias_and_activation_survive_the_round_trip(
    source_dir: Path, destination: Path
) -> None:
    """The activation is persisted because assuming Identity for a stage that
    has one would be the same silent, shape-preserving error as skipping the
    Dense stage entirely."""
    stage = DenseStage(
        name="2_Dense",
        weight=np.full((4, 4), 0.25, dtype=np.float32),
        bias=np.arange(4, dtype=np.float32),
        activation="torch.nn.modules.activation.Tanh",
    )
    profile = _profile_with(BGE, has_dense_stage=True, dimension=4)

    result = export_model(
        profile,
        acquired_for(profile, source_dir),
        destination,
        exporter=FakeExporter(stages=[stage], graph={"hidden": 4}),
    )

    assert result.dense_path is not None
    with np.load(result.dense_path) as loaded:
        assert np.array_equal(loaded[f"{BIAS_PREFIX}2_Dense"], np.arange(4, dtype=np.float32))
        assert str(loaded[f"{ACTIVATION_PREFIX}2_Dense"]) == (
            "torch.nn.modules.activation.Tanh"
        )


def test_a_profile_declaring_a_dense_stage_whose_model_has_none_fails(
    source_dir: Path, destination: Path
) -> None:
    """The Observable's second half, stated as a refusal: this is the one
    failure that would otherwise succeed silently."""
    exporter = FakeExporter(stages=[])

    with pytest.raises(PreparationError) as caught:
        export_model(
            GEMMA, acquired_for(GEMMA, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert caught.value.model_id == GEMMA.model_id
    assert "dense" in str(caught.value).lower()


def test_a_dense_failure_leaves_no_graph_behind(
    source_dir: Path, destination: Path
) -> None:
    """A ``model.onnx`` published without its Dense weights is exactly the
    artifact a later run must never find, so nothing is published until every
    part of the export has succeeded (8.5)."""
    with pytest.raises(PreparationError):
        export_model(
            GEMMA,
            acquired_for(GEMMA, source_dir),
            destination,
            exporter=FakeExporter(stages=[]),
        )

    assert not (destination / ONNX_FILENAME).exists()
    assert not (destination / DENSE_FILENAME).exists()


def test_a_profile_declaring_no_dense_stage_whose_model_has_one_fails(
    source_dir: Path, destination: Path
) -> None:
    """The mirror image, and just as silent: dropping a Dense stage the model
    really has would leave the profile and the artifact disagreeing about what
    the vectors mean."""
    # The trunk's width matches the projection deliberately, so the stage is
    # rejected for being *undeclared* rather than for not fitting - otherwise
    # the width check downstream would mask the absence of this one.
    exporter = FakeExporter(
        stages=[dense_stage("2_Dense", 1024, 1024)], graph={"hidden": 1024}
    )

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert "2_Dense" in str(caught.value)
    assert "declares no dense stage" in str(caught.value)


def test_no_dense_file_is_written_for_a_profile_without_the_stage(
    source_dir: Path, destination: Path
) -> None:
    result = export_model(
        BGE, acquired_for(BGE, source_dir), destination, exporter=FakeExporter()
    )

    assert result.dense_path is None
    assert not (destination / DENSE_FILENAME).exists()


def test_a_stale_dense_file_left_by_an_earlier_run_is_not_claimed(
    source_dir: Path, destination: Path
) -> None:
    """``dense_path`` reports what *this* export produced, not what it found.

    An earlier preparation of a different model into the same directory - or a
    profile whose dense stage was removed - leaves a ``dense.npz`` behind. If
    the result derived its ``dense_path`` from the file's presence rather than
    from the stages this run actually extracted, it would hand task 5.2 another
    model's projections to apply, which is the silent, shape-preserving
    corruption this module exists to prevent, arriving by a different route.
    """
    destination.mkdir(parents=True, exist_ok=True)
    stale = destination / DENSE_FILENAME
    stale.write_bytes(b"weights from some earlier run")

    result = export_model(
        BGE, acquired_for(BGE, source_dir), destination, exporter=FakeExporter()
    )

    # BGE declares no dense stage, so this export produced none - whatever the
    # directory happened to contain already.
    assert result.dense_path is None
    # The stale file is not this task's to delete (task 3.3 owns artifact
    # lifecycle), but it must not be adopted either.
    assert stale.read_bytes() == b"weights from some earlier run"


def test_dense_weights_that_do_not_fit_the_exported_trunk_are_refused(
    source_dir: Path, destination: Path
) -> None:
    """Pooling does not change the width, so the first projection consumes
    exactly the trunk's hidden size. Weights that do not fit belong to some
    other model, and applying them anyway is the substitution 4.6 forbids
    arriving by the back door."""
    exporter = FakeExporter(
        stages=[dense_stage("2_Dense", 768, 3072), dense_stage("3_Dense", 3072, 768)],
        graph={"hidden": 384},
    )

    with pytest.raises(PreparationError) as caught:
        export_model(
            GEMMA, acquired_for(GEMMA, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert "384" in str(caught.value) and "768" in str(caught.value)


def test_dense_stages_that_do_not_chain_are_refused(
    source_dir: Path, destination: Path
) -> None:
    exporter = FakeExporter(
        stages=[dense_stage("2_Dense", 768, 3072), dense_stage("3_Dense", 999, 768)],
        graph={"hidden": 768},
    )

    with pytest.raises(PreparationError) as caught:
        export_model(
            GEMMA, acquired_for(GEMMA, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert "3072" in str(caught.value) and "999" in str(caught.value)


def test_a_dense_chain_not_ending_at_the_profiles_dimension_is_refused(
    source_dir: Path, destination: Path
) -> None:
    """The last projection's width *is* the published vector dimension (3.6),
    so a disagreement here means the profile and the artifact describe different
    models."""
    exporter = FakeExporter(
        stages=[dense_stage("2_Dense", 768, 3072), dense_stage("3_Dense", 3072, 512)],
        graph={"hidden": 768},
    )

    with pytest.raises(PreparationError) as caught:
        export_model(
            GEMMA, acquired_for(GEMMA, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert str(GEMMA.dimension) in str(caught.value)
    assert "512" in str(caught.value)


# --------------------------------------------------------------------------
# Requirement 4.6: name the stage, never substitute a model
# --------------------------------------------------------------------------


def test_exporting_a_model_other_than_the_one_requested_is_refused(
    source_dir: Path, destination: Path
) -> None:
    """Requirement 4.6's second half. The quietest possible substitution is one
    made here, where the profile says one model and the acquired files are
    another's."""
    mismatched = AcquiredModel(
        model_id=BGE.model_id, revision=REVISION, local_path=source_dir
    )
    exporter = FakeExporter()

    with pytest.raises(PreparationError) as caught:
        export_model(GEMMA, mismatched, destination, exporter=exporter)

    assert caught.value.stage == EXPORT_STAGE
    assert GEMMA.model_id in str(caught.value)
    assert BGE.model_id in str(caught.value)
    assert exporter.graph_calls == []
    assert not destination.exists() or list(destination.iterdir()) == []


@pytest.mark.parametrize(
    "acquired_id",
    [
        # The dangerous one: a real, separately published variant of the same
        # model. It shares a prefix, a vendor and a family with the profile's
        # id, and it is a genuinely different set of weights - a quantisation
        # aware training checkpoint whose vectors are not the profile's. A guard
        # comparing prefixes rather than the whole string accepts it, and
        # requirement 4.6's "shall not substitute a different model" is broken
        # by something that looks almost right in a log line.
        "google/embeddinggemma-300m-qat",
        # A truncation of the same id: also a prefix, also not this model.
        "google/embeddinggemma-300",
        # Case variants. Hugging Face repository ids are case sensitive, so
        # these name repositories that are not the profile's; a guard that
        # case-folds before comparing accepts them.
        "google/EmbeddingGemma-300m",
        "GOOGLE/EMBEDDINGGEMMA-300M",
        # Surrounding whitespace: the same string to a human, a different
        # repository to the Hub.
        " google/embeddinggemma-300m",
    ],
)
def test_only_the_exact_requested_model_id_may_be_exported(
    acquired_id: str, source_dir: Path, destination: Path
) -> None:
    """Requirement 4.6's identity check is exact-string, not approximate.

    The neighbouring test pairs two wholly dissimilar ids, which any weakened
    comparison still separates and which therefore proves only that *a* check
    exists. These are the near misses that decide what kind of check it is.
    Every one of them must be refused, so neither a prefix comparison nor a
    case-insensitive one can pass for the real thing.
    """
    mismatched = AcquiredModel(
        model_id=acquired_id, revision=REVISION, local_path=source_dir
    )
    exporter = FakeExporter()

    with pytest.raises(PreparationError) as caught:
        export_model(GEMMA, mismatched, destination, exporter=exporter)

    assert caught.value.stage == EXPORT_STAGE
    assert GEMMA.model_id in str(caught.value)
    assert acquired_id.strip() in str(caught.value)
    # Refused before a single byte of graph was written, so there is no
    # half-exported substitute left behind either.
    assert exporter.graph_calls == []
    assert not destination.exists() or list(destination.iterdir()) == []


def test_an_arbitrary_exporter_failure_is_reported_as_the_export_stage(
    source_dir: Path, destination: Path
) -> None:
    exporter = FakeExporter(graph_error=RuntimeError("no kernel for gelu"))

    with pytest.raises(PreparationError) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value.stage == EXPORT_STAGE
    assert caught.value.model_id == BGE.model_id
    assert "no kernel for gelu" in str(caught.value)
    assert "RuntimeError" in str(caught.value)
    # ``from None``: a chained cause would print the raw traceback beneath the
    # diagnosed one, which is what task 3.1 established for acquisition.
    assert caught.value.__cause__ is None


def test_a_failure_names_which_step_of_the_export_it_was(
    source_dir: Path, destination: Path
) -> None:
    graph_failure = FakeExporter(graph_error=RuntimeError("boom"))
    dense_failure = FakeExporter(dense_error=RuntimeError("boom"))

    with pytest.raises(PreparationError) as first:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=graph_failure
        )
    with pytest.raises(PreparationError) as second:
        export_model(
            GEMMA,
            acquired_for(GEMMA, source_dir),
            destination,
            exporter=dense_failure,
        )

    assert str(first.value) != str(second.value)
    assert "graph" in str(first.value).lower()
    assert "dense" in str(second.value).lower()


def test_an_already_diagnosed_failure_passes_through_unchanged(
    source_dir: Path, destination: Path
) -> None:
    """Re-diagnosing would bury a specific stage under a generic one; task 3.1
    made the same choice for acquisition."""
    original = LicenseAcceptanceRequired(
        "terms not accepted",
        acceptance_url="https://example.invalid",
        model_id=BGE.model_id,
    )
    exporter = FakeExporter(graph_error=original)

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert caught.value is original


@pytest.mark.parametrize(
    "error", [ValueError("bad"), OSError("disk"), MemoryError("oom")]
)
def test_every_ordinary_failure_is_one_of_this_features_errors(
    source_dir: Path, destination: Path, error: Exception
) -> None:
    """8.2 is a type question: ``except PreparationError`` must cover the export
    stage entirely, so nothing raw may escape."""
    exporter = FakeExporter(graph_error=error)

    with pytest.raises(EmbeddingRuntimeError):
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )


def test_an_interrupt_is_not_dressed_up_as_a_preparation_failure(
    source_dir: Path, destination: Path
) -> None:
    """An operator stopping a multi-minute export has not hit a preparation
    failure. The interrupt propagates - but 8.5 still holds, because the cleanup
    that hides a half-written graph runs in a ``finally``."""
    exporter = FakeExporter(
        partial_bytes=True, graph_error=KeyboardInterrupt()
    )

    with pytest.raises(KeyboardInterrupt):
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert list(destination.iterdir()) == []


def test_a_partially_written_graph_is_never_published(
    source_dir: Path, destination: Path
) -> None:
    """8.5: an interrupted export must not leave an artifact a later run would
    treat as valid."""
    exporter = FakeExporter(
        partial_bytes=True, graph_error=RuntimeError("died mid-write")
    )

    with pytest.raises(PreparationError):
        export_model(
            BGE, acquired_for(BGE, source_dir), destination, exporter=exporter
        )

    assert not (destination / ONNX_FILENAME).exists()


# --------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------


def test_progress_is_reported_as_a_callback(
    source_dir: Path, destination: Path
) -> None:
    """design.md, Monitoring: progress is a callback, not logging, so callers
    choose presentation. Task 3.1 used a bare callback rather than
    ``RunTracker`` because export has no provider or execution mode to
    attribute, and the same holds here."""
    seen: list[ProgressUpdate] = []

    export_model(
        BGE,
        acquired_for(BGE, source_dir),
        destination,
        exporter=FakeExporter(),
        progress=seen.append,
    )

    assert seen != []
    assert all(update.operation.endswith(BGE.name) for update in seen)
    assert [u.completed for u in seen] == sorted(u.completed for u in seen)
    assert seen[0].completed == 0
    assert seen[-1].completed == seen[-1].total


# --------------------------------------------------------------------------
# Value-object invariants
# --------------------------------------------------------------------------


def test_a_dense_stage_weight_must_be_a_two_dimensional_float32_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        DenseStage(
            name="d",
            weight=np.zeros(4, dtype=np.float32),
            bias=None,
            activation="Identity",
        )
    with pytest.raises(ValueError, match="float32"):
        DenseStage(
            name="d",
            weight=np.zeros((4, 4), dtype=np.float64),
            bias=None,
            activation="Identity",
        )


def test_a_dense_stage_bias_must_match_the_projections_width() -> None:
    with pytest.raises(ValueError, match="bias"):
        DenseStage(
            name="d",
            weight=np.zeros((4, 8), dtype=np.float32),
            bias=np.zeros(8, dtype=np.float32),
            activation="Identity",
        )


def test_a_dense_stage_reports_its_widths_in_pipeline_terms() -> None:
    stage = dense_stage("d", 8, 4)

    assert stage.in_features == 8
    assert stage.out_features == 4


def test_an_exported_model_must_point_at_files_that_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model.onnx"):
        ExportedModel(
            model_id="m",
            revision=REVISION,
            onnx_path=tmp_path / "model.onnx",
            dense_path=None,
            batch_size=1,
            compiled_seq_len=512,
            hidden_size=8,
        )


# --------------------------------------------------------------------------
# The real Dense reader, against a fabricated on-disk layout (no network)
# --------------------------------------------------------------------------


def write_sentence_transformers_layout(
    root: Path,
    *,
    dense: Sequence[tuple[str, int, int, bool]] = (),
    include_modules_json: bool = True,
    omit_weights_for: str | None = None,
    weight_shape_override: tuple[int, int] | None = None,
) -> None:
    """A sentence-transformers repository as it actually sits on disk.

    Layout taken from the real ``google/embeddinggemma-300m`` repository read on
    2026-09-05: ``modules.json`` lists the pipeline in order, and each Dense
    module is a directory holding ``config.json`` and ``model.safetensors`` with
    a ``linear.weight`` key of shape ``(out_features, in_features)``.
    """
    from safetensors.numpy import save_file

    root.mkdir(parents=True, exist_ok=True)
    modules: list[dict[str, object]] = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]
    for index, (name, in_features, out_features, bias) in enumerate(dense):
        modules.append(
            {
                "idx": index + 2,
                "name": str(index + 2),
                "path": name,
                "type": "sentence_transformers.models.Dense",
            }
        )
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "in_features": in_features,
                    "out_features": out_features,
                    "bias": bias,
                    "activation_function": "torch.nn.modules.linear.Identity",
                }
            ),
            encoding="utf-8",
        )
        if name == omit_weights_for:
            continue
        shape = weight_shape_override or (out_features, in_features)
        tensors: dict[str, npt.NDArray[np.float32]] = {
            "linear.weight": np.full(shape, 0.5, dtype=np.float32)
        }
        if bias:
            tensors["linear.bias"] = np.zeros(out_features, dtype=np.float32)
        save_file(tensors, str(directory / "model.safetensors"))

    modules.append(
        {
            "idx": len(modules),
            "name": str(len(modules)),
            "path": f"{len(modules)}_Normalize",
            "type": "sentence_transformers.models.Normalize",
        }
    )
    if include_modules_json:
        (root / "modules.json").write_text(
            json.dumps(modules, indent=2), encoding="utf-8"
        )


def test_the_real_reader_finds_every_dense_module_in_pipeline_order(
    tmp_path: Path,
) -> None:
    write_sentence_transformers_layout(
        tmp_path, dense=[("2_Dense", 768, 3072, False), ("3_Dense", 3072, 768, False)]
    )

    stages = TorchTrunkExporter().read_dense_stages(source=tmp_path)

    assert [stage.name for stage in stages] == ["2_Dense", "3_Dense"]
    assert [(s.in_features, s.out_features) for s in stages] == [
        (768, 3072),
        (3072, 768),
    ]
    assert all(stage.bias is None for stage in stages)


def test_the_real_reader_loads_a_bias_when_the_module_declares_one(
    tmp_path: Path,
) -> None:
    write_sentence_transformers_layout(tmp_path, dense=[("2_Dense", 4, 8, True)])

    stages = TorchTrunkExporter().read_dense_stages(source=tmp_path)

    assert stages[0].bias is not None
    assert stages[0].bias.shape == (8,)


def test_the_real_reader_reports_no_stages_for_a_model_without_any(
    tmp_path: Path,
) -> None:
    write_sentence_transformers_layout(tmp_path)

    assert TorchTrunkExporter().read_dense_stages(source=tmp_path) == ()


def test_the_real_reader_reports_no_stages_when_there_is_no_modules_json(
    tmp_path: Path,
) -> None:
    """A plain ``transformers`` repository has no sentence-transformers pipeline
    at all. Reporting nothing is right for a profile that declares no Dense
    stage, and ``export_model`` refuses on behalf of one that does."""
    write_sentence_transformers_layout(tmp_path, include_modules_json=False)

    assert TorchTrunkExporter().read_dense_stages(source=tmp_path) == ()


def test_the_real_reader_refuses_a_dense_module_with_no_weights(
    tmp_path: Path,
) -> None:
    write_sentence_transformers_layout(
        tmp_path,
        dense=[("2_Dense", 4, 8, False)],
        omit_weights_for="2_Dense",
    )

    with pytest.raises(PreparationError) as caught:
        TorchTrunkExporter().read_dense_stages(source=tmp_path)

    assert caught.value.stage == EXPORT_STAGE
    assert "2_Dense" in str(caught.value)


def test_the_real_reader_refuses_weights_that_contradict_their_config(
    tmp_path: Path,
) -> None:
    """The config is the model's own statement of the projection's width; a
    weight matrix disagreeing with it means one of the two is not what it
    claims, and guessing which would be the silent choice."""
    write_sentence_transformers_layout(
        tmp_path,
        dense=[("2_Dense", 4, 8, False)],
        weight_shape_override=(9, 4),
    )

    with pytest.raises(PreparationError) as caught:
        TorchTrunkExporter().read_dense_stages(source=tmp_path)

    assert caught.value.stage == EXPORT_STAGE
    assert "(8, 4)" in str(caught.value) and "(9, 4)" in str(caught.value)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _profile_with(base: ModelProfile, **changes: Any) -> ModelProfile:
    """A throwaway profile for a shape the three real candidates do not cover.

    Constructed here rather than added to ``PROFILES``: requirement 4.1 names
    exactly three candidates and a test pins that set.
    """
    import dataclasses

    return dataclasses.replace(base, **changes)
