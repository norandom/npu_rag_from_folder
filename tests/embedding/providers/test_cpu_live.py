"""The CPU backend against a real graph and a real session (task 4.2).

``test_cpu.py`` injects a session, which is the only way to reach the refusal
branches - a real ONNX Runtime session cannot be made to hand back the wrong
provider on demand. What it cannot establish is the two things this task exists
to deliver, and both are measured here:

- **The vectors are the graph's own, at full precision.** The backend's output
  is compared bit for bit against an independent FP32 ONNX Runtime session over
  the identical inputs. An adapter that cast, quantised, pooled, or fabricated
  anything at all fails this by construction, and requirement 6.3's reference
  role depends on exactly this equality.
- **It is selectable while this machine's NPU is present and healthy** (2.3).
  This machine's NPU works: task 1.3 ran ``all-MiniLM-L6-v2`` on it at 5.24x
  CPU. A test that only passed because the hardware was missing would prove
  nothing here, so the NPU-selection limb *asserts* the capability report says
  the NPU is reachable before it asserts the CPU is served anyway.

The model is ``all-MiniLM-L6-v2`` at batch 1 x sequence 128 - not one of
requirement 4.1's three candidates, for the reason tasks 3.2 and 3.3 gave: it is
the model this machine's NPU is proven on, and it is an order of magnitude
cheaper than EmbeddingGemma's 1.22 GB. Nothing about this task's contract is
model-specific, and the profile is deliberately not added to ``PROFILES``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from huggingface_hub import HfApi

from npu_rag.embedding.environment.capability import (
    VITISAI_PROVIDER,
    check_capability,
    probe_onnx_runtime,
)
from npu_rag.embedding.errors import ExecutionError
from npu_rag.embedding.models.acquire import (
    DEFAULT_REVISION,
    AcquiredModel,
    acquire_model,
)
from npu_rag.embedding.models.artifacts import (
    PreparedArtifact,
    Toolchain,
    ensure_prepared,
)
from npu_rag.embedding.models.export import ATTENTION_MASK, INPUT_IDS
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.providers.base import (
    BackendFactories,
    TransformerBackend,
    resolve_backend,
)
from npu_rag.embedding.providers.cpu import CPU_PROVIDER, CpuBackend
from npu_rag.embedding.reporting import ProgressCallback
from npu_rag.embedding.types import (
    CONDITION_PROVIDER_REGISTERED,
    CapabilityReport,
    ExecutionMode,
    ProviderChoice,
)

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

#: Only what a trunk export reads. The repository also carries ONNX, OpenVINO
#: and TensorFlow copies of the same weights, and fetching them would cost
#: several hundred megabytes for nothing.
PATTERNS = ("*.json", "*.txt", "model.safetensors", "1_Pooling/*")


def _hub_reachable() -> bool:
    try:
        HfApi().model_info(CONTROL_MODEL, token=False, timeout=10)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hub_reachable(), reason="the Hugging Face Hub is not reachable"
)

_probe = probe_onnx_runtime()
_providers = _probe.available_providers or ()

npu_present = pytest.mark.skipif(
    VITISAI_PROVIDER not in _providers,
    reason=(
        f"{VITISAI_PROVIDER} is not registered in this interpreter "
        f"(available: {list(_providers)}); the 2.3 limb this marks needs a "
        "machine whose NPU actually works"
    ),
)


def _acquirer(
    profile: ModelProfile,
    *,
    revision: str | None,
    progress: ProgressCallback | None,
) -> AcquiredModel:
    return acquire_model(
        profile,
        revision=revision or DEFAULT_REVISION,
        allow_patterns=list(PATTERNS),
        progress=progress,
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> PreparedArtifact:
    """One real CPU preparation - acquire, export, publish. No compilation:
    ``ensure_prepared`` compiles only for the NPU, which is exactly why the CPU
    path stays available on a machine with no NPU at all."""
    return ensure_prepared(
        CONTROL_PROFILE,
        ProviderChoice.CPU,
        tmp_path_factory.mktemp("artifacts"),
        toolchain=Toolchain(
            onnxruntime_version=str(ort.__version__),
            ryzen_ai_version=None,
            driver_version=None,
        ),
        acquirer=_acquirer,
    )


def _ids() -> npt.NDArray[np.int64]:
    """A padded batch: real token ids followed by padding, as the service will
    hand one over. Not zeros throughout - a graph fed a constant can return a
    constant, and every assertion below would still pass."""
    row = np.zeros(CONTROL_SEQ_LEN, dtype=np.int64)
    row[: 8] = np.array([101, 2023, 2003, 1037, 3231, 6251, 1012, 102], np.int64)
    return row.reshape(1, CONTROL_SEQ_LEN)


def _mask() -> npt.NDArray[np.int64]:
    row = np.zeros(CONTROL_SEQ_LEN, dtype=np.int64)
    row[: 8] = 1
    return row.reshape(1, CONTROL_SEQ_LEN)


def _reference_session(prepared: PreparedArtifact) -> ort.InferenceSession:
    """An independent full-precision session over the same published graph."""
    return ort.InferenceSession(
        str(prepared.onnx_path), providers=[CPU_PROVIDER]
    )


def test_it_runs_the_prepared_graph_and_returns_token_embeddings(
    prepared: PreparedArtifact,
) -> None:
    """The Observable's shape half, on a real transformer: the batch and the
    compiled length the NPU backend will also be handed, and the trunk's own
    hidden width."""
    backend = CpuBackend(prepared, CONTROL_PROFILE)

    result = backend.run(_ids(), _mask())

    assert result.shape == (1, CONTROL_SEQ_LEN, CONTROL_PROFILE.dimension)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    # Real activations, not a fabricated block: a constant array would satisfy
    # every shape and dtype assertion above.
    assert float(result.std()) > 0.0


def test_its_vectors_are_bitwise_the_graphs_own_full_precision_output(
    prepared: PreparedArtifact,
) -> None:
    """Requirement 6.3's reference role, measured. The benchmark compares NPU
    vectors against *full-precision CPU reference* vectors, so this backend must
    return what an untouched FP32 session returns - not something close to it."""
    backend = CpuBackend(prepared, CONTROL_PROFILE)
    token_ids, attention = _ids(), _mask()

    (expected,) = _reference_session(prepared).run(
        None, {INPUT_IDS: token_ids, ATTENTION_MASK: attention}
    )
    observed = backend.run(token_ids, attention)

    np.testing.assert_array_equal(observed, expected)
    assert observed.dtype == np.float32 == expected.dtype


def test_the_same_input_twice_gives_identical_vectors(
    prepared: PreparedArtifact,
) -> None:
    """Requirement 3.10, at the layer that could break it: the same text under
    the same model and provider returns identical vectors."""
    backend = CpuBackend(prepared, CONTROL_PROFILE)

    first = backend.run(_ids(), _mask())
    second = backend.run(_ids(), _mask())

    np.testing.assert_array_equal(first, second)


def test_the_mask_changes_the_vectors_so_both_inputs_really_reach_the_graph(
    prepared: PreparedArtifact,
) -> None:
    """A backend that fed only ``input_ids`` and passed a fabricated mask would
    return plausible vectors and be invisible to every other assertion here."""
    backend = CpuBackend(prepared, CONTROL_PROFILE)

    attended = backend.run(_ids(), _mask())
    everything = backend.run(_ids(), np.ones((1, CONTROL_SEQ_LEN), np.int64))

    assert not np.array_equal(attended, everything)


def test_a_batch_of_the_wrong_length_is_refused_by_the_real_backend(
    prepared: PreparedArtifact,
) -> None:
    """The compiled length is a contract, and the graph itself answers a
    mismatch with ``InvalidArgument`` (task 3.2). The backend refuses first, and
    either way nothing is reshaped into something that would silently run."""
    backend = CpuBackend(prepared, CONTROL_PROFILE)
    wrong = CONTROL_SEQ_LEN // 2

    with pytest.raises(ExecutionError) as caught:
        backend.run(
            np.zeros((1, wrong), np.int64), np.ones((1, wrong), np.int64)
        )

    assert caught.value.provider is ProviderChoice.CPU
    assert caught.value.model_id == CONTROL_MODEL


def test_it_reports_the_cpu_and_no_partition_share(
    prepared: PreparedArtifact,
) -> None:
    backend = CpuBackend(prepared, CONTROL_PROFILE)

    assert backend.provider is ProviderChoice.CPU
    assert backend.execution_mode is ExecutionMode.IN_PROCESS
    assert backend.npu_partition_share is None


# --------------------------------------------------------------------------
# Requirement 2.3 on a machine whose NPU is present and healthy
# --------------------------------------------------------------------------


@npu_present
def test_this_machines_npu_really_is_usable() -> None:
    """The premise the next test rests on. Without it, "the CPU was served
    anyway" would be indistinguishable from "there was nothing else to serve
    it with", and 2.3 would be untested on this machine."""
    report = check_capability()

    assert report.execution_mode is ExecutionMode.IN_PROCESS
    assert report.condition(CONDITION_PROVIDER_REGISTERED).satisfied is True


@npu_present
def test_explicit_cpu_is_served_by_the_cpu_while_the_npu_is_healthy(
    prepared: PreparedArtifact,
) -> None:
    """Requirement 2.3, end to end on this hardware: a measured report saying
    the NPU is fully usable, an explicit ``cpu`` selection, and real vectors off
    the CPU provider. The NPU factory raises, so a resolution that built one -
    or that preferred it - fails rather than quietly succeeding."""
    report = check_capability()
    assert report.execution_mode is ExecutionMode.IN_PROCESS

    def cpu_factory(
        profile: ModelProfile, capability: CapabilityReport, /
    ) -> TransformerBackend:
        return CpuBackend(prepared, profile)

    def npu_factory(
        profile: ModelProfile, capability: CapabilityReport, /
    ) -> TransformerBackend:
        raise AssertionError("explicit cpu selection must not build an NPU backend")

    backend, reason = resolve_backend(
        ProviderChoice.CPU,
        CONTROL_PROFILE,
        report,
        factories=BackendFactories(npu=npu_factory, cpu=cpu_factory),
    )
    result = backend.run(_ids(), _mask())

    assert backend.provider is ProviderChoice.CPU
    assert reason is None
    assert result.shape == (1, CONTROL_SEQ_LEN, CONTROL_PROFILE.dimension)

    (expected,) = _reference_session(prepared).run(
        None, {INPUT_IDS: _ids(), ATTENTION_MASK: _mask()}
    )
    np.testing.assert_array_equal(result, expected)
