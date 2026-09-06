"""Real ONNX exports, proving task 3.2's Observable on real weights.

Two models are exported here, and the split is deliberate.

**`all-MiniLM-L6-v2` (~90 MB), for the shape contract.** Not one of requirement
4.1's three candidates; it is the model task 1.3's third probe compiled and ran
on this machine's NPU at 5.24x CPU throughput, so it is known to survive the
whole downstream path, and it is an order of magnitude cheaper to fetch than the
candidates. The profile it is exported under is constructed here and deliberately
**not** added to ``PROFILES``: requirement 4.1 names exactly three candidates and
``test_profiles.py`` pins that set. It carries the cheap, repeatable half of the
Observable - a graph ONNX Runtime holds to exactly one input shape.

**`google/embeddinggemma-300m` (~1.2 GB), for the dense contract.** The cheap
model cannot stand in for this one, because it has no Dense stage: its pipeline
is Transformer -> Pooling -> Normalize. EmbeddingGemma is requirement 4.2's
initial default candidate *and* the only candidate whose pipeline is
Transformer -> Pooling -> Dense(768->3072) -> Dense(3072->768) -> Normalize, so
it is the only model on which the extraction this task exists to perform is
load-bearing at all. research.md's follow-up - "Confirm the Dense stage's weights
are exported alongside the graph and loaded correctly; a missing Dense layer
produces plausible-looking but wrong vectors, which no shape check would catch" -
is answered here or nowhere. It is exported at its real profile (batch 1 x 512),
not a reduced one.

The unit tests in ``test_export.py`` cover the same logic against an injected
exporter and never touch the network or a byte of real weights. This module
exists for the two things they cannot establish: that a real transformer comes
back pinned to one shape, and that the dense weights written beside it are the
model's own.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from huggingface_hub import HfApi

from npu_rag.embedding.models.acquire import acquire_model, discover_credential
from npu_rag.embedding.models.export import (
    ATTENTION_MASK,
    DENSE_FILENAME,
    DENSE_ORDER_KEY,
    INPUT_IDS,
    ONNX_FILENAME,
    WEIGHT_PREFIX,
    ExportedModel,
    export_model,
)
from npu_rag.embedding.profiles import ModelProfile, profile_for

#: Small, ungated, and already proven on this machine's NPU (research.md, third
#: probe). Not a candidate model - see the module docstring.
CONTROL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Shorter than the 512 the real profiles compile at, for the same reason the
#: model is smaller: this test proves the *pinning*, and 128 pins just as
#: conclusively at a quarter of the export cost. It is also the length task
#: 1.3's probe compiled this model at.
CONTROL_SEQ_LEN = 128

CONTROL_PROFILE = ModelProfile(
    model_id=CONTROL_MODEL,
    dimension=384,
    compiled_seq_len=CONTROL_SEQ_LEN,
    architectural_context_limit=256,
    batch_size=1,
    pooling="mean",
    # Its pipeline is Transformer -> Pooling -> Normalize; read from the real
    # repository's modules.json on 2026-09-05.
    has_dense_stage=False,
    document_template="{content}",
    query_template="{content}",
    license_gated=False,
)


def _hub_reachable() -> bool:
    try:
        HfApi().model_info(CONTROL_MODEL, token=False, timeout=10)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hub_reachable(), reason="the Hugging Face Hub is not reachable"
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The one real export in this suite, shared by every assertion below."""
    destination = tmp_path_factory.mktemp("exported")
    acquired = acquire_model(
        CONTROL_PROFILE,
        allow_patterns=[
            "*.json",
            "*.txt",
            "model.safetensors",
            "1_Pooling/*",
        ],
    )
    started = time.perf_counter()
    result = export_model(CONTROL_PROFILE, acquired, destination)
    elapsed = time.perf_counter() - started
    assert elapsed >= 0.0
    return result.onnx_path


def test_the_exported_graph_declares_exactly_the_profiles_input_shape(
    exported: Path,
) -> None:
    """The Observable, first half, on a real transformer."""
    session = ort.InferenceSession(
        str(exported), providers=["CPUExecutionProvider"]
    )

    shapes = {i.name: (i.shape, i.type) for i in session.get_inputs()}
    assert shapes == {
        INPUT_IDS: ([1, CONTROL_SEQ_LEN], "tensor(int64)"),
        ATTENTION_MASK: ([1, CONTROL_SEQ_LEN], "tensor(int64)"),
    }
    assert all(isinstance(dim, int) for i in session.get_inputs() for dim in i.shape)


def test_the_exported_graph_runs_at_that_shape_and_returns_token_embeddings(
    exported: Path,
) -> None:
    session = ort.InferenceSession(
        str(exported), providers=["CPUExecutionProvider"]
    )

    (output,) = session.run(
        None,
        {
            INPUT_IDS: np.zeros((1, CONTROL_SEQ_LEN), dtype=np.int64),
            ATTENTION_MASK: np.ones((1, CONTROL_SEQ_LEN), dtype=np.int64),
        },
    )

    assert output.shape == (1, CONTROL_SEQ_LEN, CONTROL_PROFILE.dimension)
    assert output.dtype == np.float32


@pytest.mark.parametrize("wrong_length", [CONTROL_SEQ_LEN // 2, CONTROL_SEQ_LEN + 1])
def test_the_exported_graph_rejects_any_other_shape(
    exported: Path, wrong_length: int
) -> None:
    """The Observable, second half. Confirmed during task 1.3's spike: a graph
    with concrete dimensions answers a mismatched input with ``InvalidArgument``
    instead of silently reshaping it, which is what makes the compiled-length
    contract enforceable at runtime rather than merely documented."""
    session = ort.InferenceSession(
        str(exported), providers=["CPUExecutionProvider"]
    )

    with pytest.raises(Exception) as caught:
        session.run(
            None,
            {
                INPUT_IDS: np.zeros((1, wrong_length), dtype=np.int64),
                ATTENTION_MASK: np.ones((1, wrong_length), dtype=np.int64),
            },
        )

    assert "INVALID_ARGUMENT" in str(caught.value).upper()


def test_the_exported_graph_is_full_precision(exported: Path) -> None:
    """design.md, TransformerBackend: "Precision is a property of the backend,
    not the export." BF16 targeting happens later, at session construction,
    through the Vitis AI ``config_file`` option."""
    graph = onnx.load(str(exported)).graph

    floating = {
        onnx.TensorProto.DataType.Name(initializer.data_type)
        for initializer in graph.initializer
        if initializer.data_type
        in {
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.BFLOAT16,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.INT8,
            onnx.TensorProto.UINT8,
        }
    }
    assert floating == {"FLOAT"}


def test_no_dense_file_is_written_for_a_model_without_a_dense_stage(
    exported: Path,
) -> None:
    assert not (exported.parent / DENSE_FILENAME).exists()
    assert (exported.parent / ONNX_FILENAME).is_file()


# --------------------------------------------------------------------------
# The candidate that actually has a Dense stage
# --------------------------------------------------------------------------

#: Requirement 4.2's initial default candidate, and the only one of the three
#: whose sentence pipeline carries a Dense stage. Gated behind the Gemma Terms,
#: so these tests skip rather than fail where the licence has not been accepted
#: or no credential is configured - the same distinction requirement 4.5 makes.
GEMMA_PROFILE = profile_for("embeddinggemma-300m")

#: The files a trunk export plus a dense extraction actually needs. Named rather
#: than fetching the whole repository so the download stays at the weights and
#: the two Dense modules, and skips the duplicate formats the repo also carries.
GEMMA_PATTERNS = ("*.json", "*.txt", "model.safetensors", "*_Dense/*", "1_Pooling/*")


def _gemma_reachable() -> bool:
    """True only when the gated repository can actually be read.

    A gated repository's *metadata* is public (task 3.1's note), so a metadata
    call succeeding proves nothing about the gate. The credential is what
    decides, and its absence is a skip rather than a failure: this machine's
    licence acceptance is not a property of the code under test.
    """
    credential = discover_credential()
    if credential is None:
        return False
    try:
        HfApi().model_info(
            GEMMA_PROFILE.model_id, token=credential.reveal(), timeout=10
        )
    except Exception:
        return False
    return True


gemma_only = pytest.mark.skipif(
    not _gemma_reachable(),
    reason="google/embeddinggemma-300m needs an accepted licence and a credential",
)


@pytest.fixture(scope="module")
def exported_gemma(tmp_path_factory: pytest.TempPathFactory) -> ExportedModel:
    """One real export of the default candidate, at its real compiled shape.

    Module-scoped: the export is ~60 s and the graph ~1.2 GB, and every
    assertion below reads the same artifact. Acquisition is served from the
    Hugging Face cache on any run after the first, because task 3.1 pins the
    download to a resolved commit precisely so a second call re-fetches nothing.
    """
    destination = tmp_path_factory.mktemp("exported_gemma")
    acquired = acquire_model(GEMMA_PROFILE, allow_patterns=list(GEMMA_PATTERNS))
    return export_model(GEMMA_PROFILE, acquired, destination)


@gemma_only
def test_the_default_candidate_exports_at_its_declared_shape(
    exported_gemma: ExportedModel,
) -> None:
    """The Observable's first half on the candidate that will actually be
    served, at the real profile - batch 1 x 512 - rather than a reduced one."""
    session = ort.InferenceSession(
        str(exported_gemma.onnx_path), providers=["CPUExecutionProvider"]
    )

    expected = [GEMMA_PROFILE.batch_size, GEMMA_PROFILE.compiled_seq_len]
    assert {i.name: (i.shape, i.type) for i in session.get_inputs()} == {
        INPUT_IDS: (expected, "tensor(int64)"),
        ATTENTION_MASK: (expected, "tensor(int64)"),
    }
    # Every dimension a concrete integer, not a symbol: the NPU compiler is
    # given a number for each one or it has nothing to compile.
    assert all(
        isinstance(dim, int) for i in session.get_inputs() for dim in i.shape
    )


@gemma_only
def test_the_default_candidate_rejects_any_other_shape(
    exported_gemma: ExportedModel,
) -> None:
    """The Observable's second half. 511 and 513 rather than a round number:
    the graph is pinned to exactly 512, so its neighbours must be refused too."""
    session = ort.InferenceSession(
        str(exported_gemma.onnx_path), providers=["CPUExecutionProvider"]
    )

    for wrong in (GEMMA_PROFILE.compiled_seq_len - 1, GEMMA_PROFILE.compiled_seq_len + 1):
        with pytest.raises(Exception) as caught:
            session.run(
                None,
                {
                    INPUT_IDS: np.zeros((1, wrong), dtype=np.int64),
                    ATTENTION_MASK: np.ones((1, wrong), dtype=np.int64),
                },
            )
        assert "INVALID_ARGUMENT" in str(caught.value).upper()


@gemma_only
def test_the_default_candidates_dense_weights_are_the_models_own(
    exported_gemma: ExportedModel,
) -> None:
    """The Observable's dense half, and the reason this model is exported at all.

    The assertions are on the *values* the real repository carries - two
    projections, 768 -> 3072 -> 768, in that order - because the failure this
    guards against produces a file of the right shape containing the wrong
    numbers. Read from google/embeddinggemma-300m at commit 57c266a on
    2026-09-05; its modules.json declares 2_Dense then 3_Dense.
    """
    assert exported_gemma.dense_path is not None
    assert exported_gemma.dense_path.name == DENSE_FILENAME

    stored = np.load(exported_gemma.dense_path)

    # Order is stored as data, and it is the pipeline's order.
    assert list(stored[DENSE_ORDER_KEY]) == ["2_Dense", "3_Dense"]

    first = stored[f"{WEIGHT_PREFIX}2_Dense"]
    second = stored[f"{WEIGHT_PREFIX}3_Dense"]

    # sentence-transformers orientation, (out_features, in_features).
    assert first.shape == (3072, 768)
    assert second.shape == (768, 3072)
    assert first.dtype == np.float32
    assert second.dtype == np.float32

    # The chain the trunk feeds: hidden -> 3072 -> the published dimension.
    assert first.shape[1] == exported_gemma.hidden_size
    assert first.shape[0] == second.shape[1]
    assert second.shape[0] == GEMMA_PROFILE.dimension

    # Real trained weights, not zeros or a placeholder. A file of the right
    # shape full of zeros would satisfy every assertion above and annihilate
    # every vector the pipeline produces.
    assert np.any(first != 0.0)
    assert np.any(second != 0.0)
    assert np.isfinite(first).all()
    assert np.isfinite(second).all()


@gemma_only
def test_the_default_candidates_dense_weights_match_the_repository(
    exported_gemma: ExportedModel,
) -> None:
    """The persisted weights are byte-for-byte the repository's own tensors.

    Every other assertion in this module checks a property the weights have.
    This one checks their identity, which is the only thing that rules out a
    plausible substitute - the precise defect research.md's follow-up names.
    """
    from safetensors.numpy import load_file

    assert exported_gemma.dense_path is not None
    acquired = acquire_model(GEMMA_PROFILE, allow_patterns=list(GEMMA_PATTERNS))
    stored = np.load(exported_gemma.dense_path)

    for module in ("2_Dense", "3_Dense"):
        upstream = load_file(str(acquired.local_path / module / "model.safetensors"))
        np.testing.assert_array_equal(
            stored[f"{WEIGHT_PREFIX}{module}"],
            upstream["linear.weight"].astype(np.float32),
        )


@gemma_only
def test_the_default_candidates_graph_is_full_precision(
    exported_gemma: ExportedModel,
) -> None:
    """design.md: "Precision is a property of the backend, not the export." The
    graph stays FP32 so the Vitis AI EP performs the BF16 cast at session
    construction and `CpuBackend` can serve as requirement 6.3's reference."""
    graph = onnx.load(str(exported_gemma.onnx_path)).graph

    observed = {
        onnx.TensorProto.DataType.Name(initializer.data_type)
        for initializer in graph.initializer
        if initializer.data_type
        in {
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.BFLOAT16,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.INT8,
            onnx.TensorProto.UINT8,
        }
    }
    assert observed == {"FLOAT"}
