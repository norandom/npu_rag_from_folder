"""The NPU backend against a real compiled snapshot on real hardware (task 4.3).

``test_vitisai.py`` injects a session and fabricates ``context.onnx`` graphs,
which is the only way to reach the guards, the provider-choice policy and the
node-mix metric on a machine with no NPU. What it cannot establish is that a real
Vitis AI session runs the compiled snapshot, comes back NPU-first, returns token
embeddings, and reads the live node mix - and those are exactly what this file
measures.

**This costs a real NPU compilation** (task 3.3 measured ~160-310 s for
``all-MiniLM-L6-v2`` cold), so it is opt-in behind ``NPU_RAG_LIVE_COMPILE=1`` -
the same gate ``test_artifacts_live.py`` uses, so a single environment variable
runs both and the compiled artifact is reused within this module. It also skips
where the provider is not registered or the Hub is unreachable.

The model is ``all-MiniLM-L6-v2`` at batch 1 x sequence 128 - not one of
requirement 4.1's three candidates, for the reason tasks 3.2/3.3 gave: it is the
model this machine's NPU is proven on (task 1.3's third probe, 5.24x CPU) and an
order of magnitude cheaper than EmbeddingGemma, which is deliberately not
compiled here.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import pytest
from huggingface_hub import HfApi

from npu_rag.embedding.environment.xrt import XrtSmiWrapper
from npu_rag.embedding.models.artifacts import (
    OrtContextCompiler,
    PreparedArtifact,
    ensure_prepared,
)
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.providers.vitisai import (
    MINIMUM_PARTITION_SHARE,
    VITISAI_PROVIDER,
    VitisAIBackend,
    read_node_partition_share,
)
from npu_rag.embedding.types import ProviderChoice

CONTROL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONTROL_SEQ_LEN = 128

CONTROL_PROFILE = ModelProfile(
    model_id=CONTROL_MODEL,
    dimension=384,
    compiled_seq_len=CONTROL_SEQ_LEN,
    architectural_context_limit=256,
    batch_size=1,
    pooling="mean",
    has_dense_stage=False,
    document_template="{content}",
    query_template="{content}",
    license_gated=False,
)

OPT_IN = "NPU_RAG_LIVE_COMPILE"

#: The op the compiler leaves as a residue in the published snapshot beside the
#: single EPContext node. A live MiniLM artifact reads exactly these (design.md,
#: 2026-09-06 amendment): one EPContext plus Cast, Gather, GatherND.
EXPECTED_RESIDUE = {"Cast", "Gather", "GatherND"}


def _provider_registers() -> bool:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        return VITISAI_PROVIDER in ort.get_available_providers()
    except Exception:
        return False


def _hub_reachable() -> bool:
    try:
        HfApi().model_info(CONTROL_MODEL, token=False, timeout=10)
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get(OPT_IN),
        reason=f"set {OPT_IN}=1 to run the ~5 minute real NPU compilation",
    ),
    pytest.mark.skipif(
        not _provider_registers(),
        reason=f"{VITISAI_PROVIDER} is not registered in this interpreter",
    ),
    pytest.mark.skipif(
        not _hub_reachable(), reason="the Hugging Face Hub is not reachable"
    ),
]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> PreparedArtifact:
    """The one real compilation in this module, shared by every assertion."""
    root = tmp_path_factory.mktemp("artifacts")
    return ensure_prepared(
        CONTROL_PROFILE,
        ProviderChoice.NPU,
        root,
        compiler=OrtContextCompiler(),
    )


def _ids() -> npt.NDArray[np.int64]:
    row = np.zeros(CONTROL_SEQ_LEN, dtype=np.int64)
    row[:8] = np.array([101, 2023, 2003, 1037, 3231, 6251, 1012, 102], np.int64)
    return row.reshape(1, CONTROL_SEQ_LEN)


def _mask() -> npt.NDArray[np.int64]:
    row = np.zeros(CONTROL_SEQ_LEN, dtype=np.int64)
    row[:8] = 1
    return row.reshape(1, CONTROL_SEQ_LEN)


def test_the_node_mix_reads_the_live_minilm_residue(
    prepared: PreparedArtifact,
) -> None:
    """The metric's ground truth: one EPContext node plus the small cheap
    residue, against the real trunk. Measured 2026-09-06: 251 trunk nodes,
    residue {Cast, Gather, GatherND}, share 0.988."""
    assert prepared.context_path is not None
    graph = onnx.load(str(prepared.context_path), load_external_data=False).graph
    residue = {node.op_type for node in graph.node if node.op_type != "EPContext"}
    assert sum(node.op_type == "EPContext" for node in graph.node) == 1
    assert residue == EXPECTED_RESIDUE, residue

    share = read_node_partition_share(
        context_path=prepared.context_path, model_path=prepared.onnx_path
    )

    assert share is not None
    assert share >= MINIMUM_PARTITION_SHARE
    assert share == pytest.approx(0.988, abs=0.005)


def test_explicit_npu_constructs_and_reports_a_verified_share(
    prepared: PreparedArtifact,
) -> None:
    """The Observable, positive limb on real hardware: an explicit ``npu`` run
    reports verified partitioning above threshold rather than failing."""
    backend = VitisAIBackend(prepared, CONTROL_PROFILE, ProviderChoice.NPU)

    assert backend.provider is ProviderChoice.NPU
    assert backend.npu_partition_share is not None
    assert backend.npu_partition_share >= MINIMUM_PARTITION_SHARE


def test_guard_two_passes_the_real_session_is_npu_first(
    prepared: PreparedArtifact,
) -> None:
    """A CPU-first session would have raised in construction; reaching here means
    ``session.get_providers()[0]`` was the Vitis AI provider."""
    backend = VitisAIBackend(prepared, CONTROL_PROFILE, ProviderChoice.NPU)

    result = backend.run(_ids(), _mask())

    assert result.shape == (1, CONTROL_SEQ_LEN, CONTROL_PROFILE.dimension)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert float(result.std()) > 0.0


def test_the_same_input_twice_gives_identical_vectors(
    prepared: PreparedArtifact,
) -> None:
    """Requirement 3.10 at the NPU layer."""
    backend = VitisAIBackend(prepared, CONTROL_PROFILE, ProviderChoice.NPU)

    first = backend.run(_ids(), _mask())
    second = backend.run(_ids(), _mask())

    np.testing.assert_array_equal(first, second)


def test_a_live_hardware_context_is_visible_during_a_sustained_run(
    prepared: PreparedArtifact,
) -> None:
    """Runtime proof of execution, independent of the node mix: while the NPU
    embeds, ``xrt-smi examine --report aie-partitions`` shows a live hardware
    context (research.md, third probe; task 1.4's wrapper). Skips gracefully
    where the wrapper cannot read the report, since the graph-level evidence
    above already stands."""
    backend = VitisAIBackend(prepared, CONTROL_PROFILE, ProviderChoice.NPU)
    wrapper = XrtSmiWrapper()

    live = False
    deadline = time.perf_counter() + 20.0
    while time.perf_counter() < deadline and not live:
        for _ in range(50):
            backend.run(_ids(), _mask())
        occupancy = wrapper.read_partitions()
        if occupancy.context_live is None:
            pytest.skip(f"aie-partitions unreadable: {occupancy.unavailable_reason}")
        live = occupancy.context_live

    assert live, "no live hardware context observed during a sustained NPU run"
