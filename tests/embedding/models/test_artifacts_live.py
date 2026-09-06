"""One real NPU compilation, proving task 3.3's Observable on real hardware.

**This costs about five minutes of NPU compilation** and is therefore opt-in:
set ``NPU_RAG_LIVE_COMPILE=1`` to run it. Measured on this machine on
2026-09-06, compiling ``all-MiniLM-L6-v2`` pinned to batch 1 x sequence 128
through the Vitis AI EP took 310 seconds. Leaving it in the default suite would
quadruple the suite's runtime for a result that changes only when the vendor
toolchain does, and the requirements sanction ad hoc verification precisely so
that expensive evidence is gathered deliberately rather than continuously.

The model is ``all-MiniLM-L6-v2``, not one of requirement 4.1's three
candidates, for the same reason task 3.2's live export used it: it is the model
task 1.3's third probe compiled and ran on this machine's NPU at 5.24x CPU
throughput, so it is known to survive the whole path, and it is an order of
magnitude cheaper than EmbeddingGemma. The profile is constructed here and
deliberately **not** added to ``PROFILES``, which ``test_profiles.py`` pins.

What only a real compilation can establish, and what is asserted below:

- The snapshot is real - the published ``context.onnx`` carries an ``EPContext``
  node written by ``VitisAIExecutionProvider``, which a session that quietly ran
  on the CPU would not produce.
- The snapshot is **two files**, and the manifest knows about both. With
  ``ep.context_embed_mode`` at ``0`` the EP writes a sidecar carrying the
  compiled AIE binary; publishing only the ``.onnx`` would produce an artifact
  that reuse accepts and nothing can load.
- The published artifact survives the atomic rename. The sidecar is referenced
  by bare filename, so it must - but that is a fact about the vendor's format,
  and it is checked rather than assumed.
- A second preparation reuses it, says so, and costs a rounding error next to
  the five minutes the first one took (4.4).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import onnx
import pytest
from huggingface_hub import HfApi

from npu_rag.embedding.models.artifacts import (
    CONTEXT_FILENAME,
    MANIFEST_FILENAME,
    VITISAI_PROVIDER,
    OrtContextCompiler,
    PreparedArtifact,
    ensure_prepared,
)
from npu_rag.embedding.models.export import ONNX_FILENAME
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.types import ProviderChoice

#: Small, ungated, and already proven on this machine's NPU (research.md, third
#: probe). Not a candidate model - see the module docstring.
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

#: The opt-in switch. Named rather than inferred: a test this expensive should
#: never start because some unrelated condition happened to be true.
OPT_IN = "NPU_RAG_LIVE_COMPILE"

#: The EP context node ONNX Runtime writes into a compiled snapshot.
EP_CONTEXT_OP = "EPContext"


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
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, PreparedArtifact, float]:
    """The one real compilation in this suite, shared by every assertion."""
    root = tmp_path_factory.mktemp("artifacts")
    started = time.perf_counter()
    result = ensure_prepared(
        CONTROL_PROFILE,
        ProviderChoice.NPU,
        root,
        compiler=OrtContextCompiler(),
    )
    return root, result, time.perf_counter() - started


def test_a_real_compilation_publishes_a_context_snapshot(
    prepared: tuple[Path, PreparedArtifact, float],
) -> None:
    _root, result, elapsed = prepared

    assert result.reused is False
    assert result.context_path is not None
    assert result.context_path.is_file()
    assert result.context_path.stat().st_size > 0
    assert (result.directory / ONNX_FILENAME).is_file()
    assert (result.directory / MANIFEST_FILENAME).is_file()
    assert elapsed > 1.0


def test_the_snapshot_carries_a_vitis_ai_context_node(
    prepared: tuple[Path, PreparedArtifact, float],
) -> None:
    """A session that quietly ran on the CPU writes no such node. This is the
    difference between "a file exists" and "something was compiled"."""
    _root, result, _elapsed = prepared
    assert result.context_path is not None

    graph = onnx.load(str(result.context_path), load_external_data=False).graph
    contexts = [node for node in graph.node if node.op_type == EP_CONTEXT_OP]

    assert contexts, [node.op_type for node in graph.node]
    sources = {
        attribute.s.decode()
        for node in contexts
        for attribute in node.attribute
        if attribute.name == "source"
    }
    assert sources == {VITISAI_PROVIDER}


def test_the_sidecar_binary_is_published_alongside_and_referenced_relatively(
    prepared: tuple[Path, PreparedArtifact, float],
) -> None:
    """``ep.context_embed_mode`` 0 splits the snapshot in two. If the reference
    were absolute the atomic rename would break every artifact silently, so the
    reference is read back and checked rather than trusted."""
    _root, result, _elapsed = prepared
    assert result.context_path is not None

    graph = onnx.load(str(result.context_path), load_external_data=False).graph
    referenced = {
        attribute.s.decode()
        for node in graph.node
        if node.op_type == EP_CONTEXT_OP
        for attribute in node.attribute
        if attribute.name == "ep_cache_context"
    }

    assert referenced, "the context node names no compiled payload"
    for name in referenced:
        assert not Path(name).is_absolute(), name
        assert (result.directory / name).is_file()
        assert name in result.manifest.files


def test_the_manifest_records_what_this_machine_actually_prepared(
    prepared: tuple[Path, PreparedArtifact, float],
) -> None:
    _root, result, _elapsed = prepared
    identity = result.manifest.identity

    assert identity.model_id == CONTROL_MODEL
    assert len(identity.revision) == 40
    assert identity.provider == ProviderChoice.NPU.value
    assert identity.compiled_seq_len == CONTROL_SEQ_LEN
    assert identity.batch_size == 1
    assert identity.onnxruntime_version
    assert identity.ryzen_ai_version
    assert CONTEXT_FILENAME in result.manifest.files
    assert ONNX_FILENAME in result.manifest.files


def test_the_published_snapshot_loads_as_a_session(
    prepared: tuple[Path, PreparedArtifact, float],
) -> None:
    """The artifact is reusable in the sense that matters: ONNX Runtime accepts
    it. Executing it and verifying its partition share belong to task 4.3."""
    import onnxruntime as ort

    _root, result, _elapsed = prepared
    assert result.context_path is not None

    config_file = Path(ort.__file__).parent / "capi" / "vaip_config.json"
    session = ort.InferenceSession(
        str(result.context_path),
        providers=[VITISAI_PROVIDER],
        provider_options=[{"config_file": str(config_file)}],
    )

    inputs = {meta.name: list(meta.shape) for meta in session.get_inputs()}
    assert inputs
    assert all(shape[:2] == [1, CONTROL_SEQ_LEN] for shape in inputs.values())


def test_a_second_preparation_reuses_it_and_says_so(
    prepared: tuple[Path, PreparedArtifact, float],
) -> None:
    """Requirement 4.4's Observable, against a five-minute cold run."""
    root, first, cold = prepared

    started = time.perf_counter()
    second = ensure_prepared(
        CONTROL_PROFILE, ProviderChoice.NPU, root, compiler=OrtContextCompiler()
    )
    warm = time.perf_counter() - started

    assert second.reused is True
    assert "reus" in second.reason.lower()
    assert second.directory == first.directory
    assert second.manifest == first.manifest
    assert warm < cold / 10
