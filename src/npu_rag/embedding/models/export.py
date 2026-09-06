"""ONNX export at a fixed sequence length and batch size (task 3.2).

This is the second of the three preparation stages design.md names - acquisition,
**export**, compile, verify - and the one that turns a set of downloaded weights
into a graph the NPU compiler can accept. Two facts shape everything below.

**The export is only the trunk.** research.md, "ONNX export scope and the
sentence-embedding pipeline": an ONNX export converts the transformer and emits
*token* embeddings. EmbeddingGemma's sentence pipeline is Pooling -> Dense ->
Normalize, and design.md's decision "Post-process pooling, Dense, and
normalization outside the ONNX graph" keeps those three stages in NumPy, on the
service side of the port, so requirement 3.5 is true by construction for every
backend. That decision hands this module a second job: the Dense stage's weights
are not in the graph, so they must be **extracted and persisted beside it**, or
they are simply lost. A missing Dense stage yields vectors of the right shape
with the right norm that are semantically wrong - the one failure in this feature
that no shape check and no norm check can see, and that only the retrieval
quality measurement (6.4) would eventually notice. It is therefore refused here,
loudly, for any profile that declares the stage.

**The shape must be concrete.** NPU compilation fixes the input shape, so the
graph is exported at the profile's ``batch_size`` and ``compiled_seq_len`` and
every dimension is verified to be an integer before the artifact is published. A
Hugging Face export ordinarily carries symbolic ``batch_size``/``sequence_length``
dimensions (research.md, third probe: "Static shapes must be pinned before
compilation"), and a symbolic dimension is the dangerous kind of wrong: the graph
loads, it runs on CPU, and only the compiler later discovers it has no number to
work with. Measured during task 1.3's spike, a graph with concrete dimensions
answers a mismatched input with ``InvalidArgument``, which is what makes
requirement 3.6's published length enforceable rather than merely documented.

**The graph stays full precision.** design.md, TransformerBackend: "Precision is
a property of the backend, not the export." BF16 targeting happens later, at
session construction, through the Vitis AI ``config_file`` provider option, where
the execution provider performs the cast - which is also what lets `CpuBackend`
run the *same* graph at full precision and serve as requirement 6.3's reference.
Nothing here quantises, and a graph arriving with reduced-precision weights is
refused rather than passed on.

## Why this does not use `optimum`

design.md's Technology Stack names ``optimum`` for the export, and task 1.1
flagged that it had resolved to 2.x. Re-verified on 2026-09-05 against the
installed versions, as task 1.1 asked: **neither** ``optimum`` branch can do
this job here.

- ``optimum`` 2.3.0 contains no ONNX exporter at all - ``optimum.exporters.onnx``
  was split into a separate ``optimum-onnx`` distribution, which resolves to
  0.1.0 and pulls ``transformers`` back to 4.57 and ``huggingface_hub`` back to
  0.36, downgrading the stack `models/acquire.py` is already built on.
- ``optimum<2`` resolves to 1.27.0, which cannot be imported under
  ``transformers`` 5.x at all: ``optimum/exporters/tasks.py`` imports
  ``is_tf_available``, a name ``transformers`` 5 removed.

So the export is driven directly by ``torch.onnx.export``. That is not a
reduction in ambition. ``optimum``'s value is its per-task graph configurations,
whose main product is *symbolic* axes - precisely what would then have to be
rewritten to concrete values. Exporting from concrete example inputs produces the
pinned graph in one step, and the trunk's signature here is two tensors in, one
tensor out. The deviation is recorded rather than hidden; see the note in
``pyproject.toml``.

This module sits in ``models`` in design.md's dependency direction - ``types,
errors -> reporting -> profiles -> environment -> models -> providers -> service
-> bench`` - so it reads errors, reporting, profiles and ``models/acquire``, and
nothing to its right.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import onnx

from npu_rag.embedding.errors import EmbeddingRuntimeError, PreparationError
from npu_rag.embedding.models.acquire import AcquiredModel
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.reporting import ProgressCallback, ProgressUpdate

__all__ = [
    "ACTIVATION_PREFIX",
    "ATTENTION_MASK",
    "BIAS_PREFIX",
    "DENSE_FILENAME",
    "DENSE_ORDER_KEY",
    "EXPORT_STAGE",
    "INPUT_IDS",
    "ONNX_FILENAME",
    "OPSET_VERSION",
    "TOKEN_EMBEDDINGS",
    "WEIGHT_PREFIX",
    "DenseStage",
    "ExportedModel",
    "TorchTrunkExporter",
    "TrunkExporter",
    "export_model",
]

#: The stage every failure here reports (8.1, 4.6). Requirement 4.6 lists
#: acquisition, export, compile and verify as the four preparation stages an
#: operator must be able to tell apart; this module owns the second.
EXPORT_STAGE: Final = "export"

#: design.md, Physical Data Model: the exported trunk and the Dense weights sit
#: beside the manifest task 3.3 writes.
ONNX_FILENAME: Final = "model.onnx"
DENSE_FILENAME: Final = "dense.npz"

#: What a half-finished write is called. Every artifact is built under one of
#: these names and renamed into place only once the whole export has succeeded,
#: so an interruption leaves nothing a later run would trust (8.5).
_PARTIAL_SUFFIX: Final = ".partial"

#: The trunk's signature, and the one design.md's `TransformerBackend.run`
#: declares: ``run(token_ids, attention_mask) -> token embeddings``. Models whose
#: forward pass also accepts ``token_type_ids`` (the BERT-derived candidates) get
#: the all-zeros default the sentence-transformers pipeline uses for single
#: sentences, which is why it is not an input here.
INPUT_IDS: Final = "input_ids"
ATTENTION_MASK: Final = "attention_mask"
TOKEN_EMBEDDINGS: Final = "token_embeddings"

#: Opset 18 rather than 17: torch's exporter reports that it has no
#: implementations below 18 and would silently down-convert, "which may not be
#: successful". Measured 2026-09-05.
OPSET_VERSION: Final = 18

#: Keys inside ``dense.npz``. ``order`` is the pipeline order, stored rather than
#: inferred: two projections whose widths happen to chain in both directions
#: would each look plausible alone, and applying them the wrong way round is the
#: same silent, shape-preserving corruption as skipping them.
DENSE_ORDER_KEY: Final = "order"
WEIGHT_PREFIX: Final = "weight__"
BIAS_PREFIX: Final = "bias__"
ACTIVATION_PREFIX: Final = "activation__"

#: Export's steps, for progress reporting: write the graph, verify it, take the
#: Dense weights, publish.
_STEPS: Final = 4

#: The sentence-transformers module type that carries a projection, as it appears
#: in a repository's ``modules.json``. Read from the real
#: ``google/embeddinggemma-300m`` repository on 2026-09-05, whose pipeline is
#: Transformer -> Pooling -> Dense(768->3072) -> Dense(3072->768) -> Normalize.
_DENSE_MODULE_SUFFIX: Final = ".Dense"

#: The tensor names sentence-transformers writes inside a Dense module.
_DENSE_WEIGHT_KEY: Final = "linear.weight"
_DENSE_BIAS_KEY: Final = "linear.bias"

_MODULES_MANIFEST: Final = "modules.json"
_MODULE_CONFIG: Final = "config.json"
_MODULE_WEIGHTS: Final = "model.safetensors"

#: Initializer element types that would carry model *weights*. Structural
#: initializers - the int64 axis vectors and shape constants every graph has -
#: are deliberately absent, so the check below asks "what precision are the
#: weights in" rather than "what types appear anywhere".
_WEIGHT_ELEMENT_TYPES: Final = frozenset(
    {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.INT8,
        onnx.TensorProto.UINT8,
    }
)

#: The one element type a full-precision export may carry.
_FULL_PRECISION: Final = onnx.TensorProto.DataType.Name(onnx.TensorProto.FLOAT)

#: How ONNX Runtime spells the two dtypes this contract fixes.
_INT64_TENSOR: Final = "tensor(int64)"
_FLOAT32_TENSOR: Final = "tensor(float)"


# --------------------------------------------------------------------------
# What the export produces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseStage:
    """One projection from the sentence pipeline, lifted out of the graph.

    ``weight`` is stored in the orientation sentence-transformers writes it,
    ``(out_features, in_features)``, so the file is a faithful copy of the
    model's own parameters rather than a transposed convenience that a later
    reader would have to guess the convention of.

    ``activation`` is carried even though all three candidates use ``Identity``.
    Task 5.2 applies these stages, and a stage whose activation were assumed to
    be the identity when it is not would produce the same class of defect this
    whole module exists to prevent: right shape, right norm, wrong meaning.
    """

    name: str
    weight: npt.NDArray[np.float32]
    bias: npt.NDArray[np.float32] | None
    activation: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "a dense stage must be named: the name is what records its "
                "place in the pipeline"
            )
        if self.weight.ndim != 2:
            raise ValueError(
                f"dense stage {self.name!r} has a {self.weight.ndim}-dimensional "
                "weight; a projection is a two-dimensional matrix"
            )
        if self.weight.dtype != np.float32:
            raise ValueError(
                f"dense stage {self.name!r} has {self.weight.dtype} weights, "
                "not float32; the export is full precision, and a different "
                "precision here would apply a cast the backend is supposed to "
                "own"
            )
        if self.bias is not None:
            if self.bias.dtype != np.float32:
                raise ValueError(
                    f"dense stage {self.name!r} has a {self.bias.dtype} bias; "
                    "the export is full precision"
                )
            if self.bias.shape != (self.out_features,):
                raise ValueError(
                    f"dense stage {self.name!r} has a bias of shape "
                    f"{self.bias.shape}, which does not match its "
                    f"{self.out_features} outputs"
                )

    @property
    def in_features(self) -> int:
        """The width this stage consumes."""
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        """The width this stage produces."""
        return int(self.weight.shape[0])


@dataclass(frozen=True)
class ExportedModel:
    """Where the exported trunk and its Dense weights landed.

    ``revision`` travels through from acquisition untouched: task 3.3's manifest
    records it so a silently changed upstream model is detectable (design.md,
    Security Considerations; requirement 4.7), and it can only be that recorded
    fact if nothing between the download and the manifest invents its own.
    """

    model_id: str
    revision: str
    onnx_path: Path
    dense_path: Path | None
    batch_size: int
    compiled_seq_len: int
    hidden_size: int

    def __post_init__(self) -> None:
        if not self.onnx_path.is_file():
            raise ValueError(
                f"{self.onnx_path} does not exist: an export reports where the "
                f"graph is, so a missing {ONNX_FILENAME} is a failed export "
                "wearing a success"
            )
        if self.dense_path is not None and not self.dense_path.is_file():
            raise ValueError(
                f"{self.dense_path} does not exist, but this export claims to "
                "have persisted dense weights"
            )


# --------------------------------------------------------------------------
# The transformer seam
# --------------------------------------------------------------------------


class TrunkExporter(Protocol):
    """The two things the export needs from a transformer on disk.

    Kept to two methods for the same reason acquisition's repository client was:
    this is the boundary at which a real export is stood in for, and a real
    export means a gigabyte of weights and minutes of graph capture. Every unit
    test drives the logic through here; exactly one live test uses the real
    implementation.
    """

    def export_graph(
        self,
        *,
        source: Path,
        destination: Path,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        """Write an FP32 ONNX trunk pinned to exactly this shape."""
        ...

    def read_dense_stages(self, *, source: Path) -> Sequence[DenseStage]:
        """The sentence pipeline's projections, in pipeline order."""
        ...


@contextlib.contextmanager
def _captured_output() -> Iterator[io.StringIO]:
    """Run a block with its console output collected instead of printed.

    Two reasons, and the second is not cosmetic. design.md's Monitoring decision
    makes progress a callback "so callers choose presentation", and a library
    that printed - or that let ``transformers`` draw a weight-loading bar on its
    behalf - would take that choice away. Task 3.1 silenced the Hub's download
    bar for exactly this reason, and did it *per call* rather than through a
    process-global switch; redirecting the streams is the same discipline
    applied to a library that offers no per-call option at all.

    The second reason: measured on this machine on 2026-09-05,
    ``torch.onnx.export`` *crashes* when it prints. Its progress line contains
    U+2705, and a Windows console under cp1252 raises ``UnicodeEncodeError``
    from inside the exporter, failing an export that had otherwise succeeded.
    Collecting the text into a buffer removes the console encoding from the path
    entirely, and the buffer is quoted back in the diagnostic if the export does
    fail, so nothing is lost by capturing it.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


class TorchTrunkExporter:
    """`TrunkExporter` backed by ``transformers`` and ``torch.onnx.export``."""

    def export_graph(
        self,
        *,
        source: Path,
        destination: Path,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        # Imported here rather than at module scope: ``torch`` costs seconds to
        # import, and every unit test reaches this class's sibling method or
        # replaces it entirely. Nothing else in the package needs torch loaded
        # to talk about an export.
        import torch
        from transformers import AutoModel

        class Trunk(torch.nn.Module):
            """Reduces a transformer's rich output to the one exported tensor.

            ``AutoModel`` returns a structured output object; ONNX needs a
            tensor. Fixing the signature here also decides the exported graph's
            interface deliberately - the two inputs design.md's
            `TransformerBackend.run` declares - rather than inheriting whatever
            the model's own ``forward`` happens to accept. The BERT-derived
            candidates additionally accept ``token_type_ids``; leaving it out
            gives it the all-zeros default the sentence-transformers pipeline
            uses for single sentences, which is the correct value here.
            """

            def __init__(self, wrapped: torch.nn.Module) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(
                self, input_ids: torch.Tensor, attention_mask: torch.Tensor
            ) -> torch.Tensor:
                output = self.wrapped(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                hidden: torch.Tensor = output.last_hidden_state
                return hidden

        example = torch.ones((batch_size, sequence_length), dtype=torch.int64)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with _captured_output() as captured:
            try:
                # ``dtype=torch.float32`` rather than the checkpoint's own dtype:
                # the export is full precision by design, and a checkpoint
                # stored in bfloat16 would otherwise export a bfloat16 graph and
                # quietly move the precision decision out of the backend's hands.
                model = AutoModel.from_pretrained(source, dtype=torch.float32)
                trunk = Trunk(model)
                trunk.eval()
                torch.onnx.export(
                    trunk,
                    (example, example),
                    str(destination),
                    input_names=[INPUT_IDS, ATTENTION_MASK],
                    output_names=[TOKEN_EMBEDDINGS],
                    opset_version=OPSET_VERSION,
                    dynamo=True,
                    # The load-bearing argument. Left to its own devices the
                    # exporter infers symbolic dimensions from the traced
                    # program; declaring no dynamic shapes freezes every
                    # dimension at the example's concrete value, which is the
                    # whole contract this task delivers.
                    dynamic_shapes=None,
                    external_data=False,
                    optimize=True,
                    verbose=False,
                )
            except Exception as error:
                noise = captured.getvalue().strip()
                raise PreparationError(
                    f"exporting {source} to ONNX at batch "
                    f"{batch_size} x sequence {sequence_length} failed: "
                    f"{type(error).__name__}: {error}"
                    + (f" [exporter output: {noise[-800:]}]" if noise else ""),
                    stage=EXPORT_STAGE,
                ) from None

    def read_dense_stages(self, *, source: Path) -> Sequence[DenseStage]:
        """Read the Dense modules a sentence-transformers repository declares.

        A repository with no ``modules.json`` is a plain ``transformers`` model
        with no sentence pipeline, and reports no stages. That is the right
        answer for a profile that declares no Dense stage, and `export_model`
        refuses on behalf of one that does - so the ambiguity is resolved
        against the profile's declaration rather than silently here.
        """
        manifest = source / _MODULES_MANIFEST
        if not manifest.is_file():
            return ()

        try:
            declared = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"{manifest} could not be read, so the sentence pipeline's "
                f"shape is unknown: {type(error).__name__}: {error}",
                stage=EXPORT_STAGE,
            ) from None

        if not isinstance(declared, list):
            raise PreparationError(
                f"{manifest} does not contain a list of pipeline modules",
                stage=EXPORT_STAGE,
            )

        stages: list[DenseStage] = []
        for entry in declared:
            if not isinstance(entry, Mapping):
                continue
            kind = entry.get("type")
            path = entry.get("path")
            if not isinstance(kind, str) or not kind.endswith(
                _DENSE_MODULE_SUFFIX
            ):
                continue
            if not isinstance(path, str) or not path:
                raise PreparationError(
                    f"{manifest} declares a {kind} module with no path, so its "
                    "weights cannot be located",
                    stage=EXPORT_STAGE,
                )
            stages.append(self._read_one_stage(source / path, name=path))
        return tuple(stages)

    def _read_one_stage(self, directory: Path, *, name: str) -> DenseStage:
        from safetensors.numpy import load_file

        config_path = directory / _MODULE_CONFIG
        weights_path = directory / _MODULE_WEIGHTS
        for required in (config_path, weights_path):
            if not required.is_file():
                raise PreparationError(
                    f"the dense module {name!r} is declared by "
                    f"{_MODULES_MANIFEST} but {required} is missing; without it "
                    "the sentence pipeline cannot be reproduced, and vectors "
                    "computed without it would be the right shape and the wrong "
                    "meaning",
                    stage=EXPORT_STAGE,
                )

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            tensors = load_file(str(weights_path))
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"the dense module {name!r} could not be read: "
                f"{type(error).__name__}: {error}",
                stage=EXPORT_STAGE,
            ) from None

        weight = tensors.get(_DENSE_WEIGHT_KEY)
        if weight is None:
            raise PreparationError(
                f"the dense module {name!r} has no {_DENSE_WEIGHT_KEY} tensor; "
                f"it carries {sorted(tensors)}",
                stage=EXPORT_STAGE,
            )

        in_features = config.get("in_features")
        out_features = config.get("out_features")
        expected = (out_features, in_features)
        if weight.shape != expected:
            raise PreparationError(
                f"the dense module {name!r} declares a projection of shape "
                f"{expected} but its {_DENSE_WEIGHT_KEY} has shape "
                f"{tuple(weight.shape)}; one of the two is not what it claims, "
                "and guessing which would be the silent choice",
                stage=EXPORT_STAGE,
            )

        bias = tensors.get(_DENSE_BIAS_KEY)
        if bool(config.get("bias")) != (bias is not None):
            raise PreparationError(
                f"the dense module {name!r} declares bias="
                f"{bool(config.get('bias'))} but "
                f"{'carries' if bias is not None else 'has no'} "
                f"{_DENSE_BIAS_KEY}",
                stage=EXPORT_STAGE,
            )

        activation = config.get("activation_function")
        try:
            return DenseStage(
                name=name,
                weight=np.ascontiguousarray(weight, dtype=np.float32),
                bias=(
                    None
                    if bias is None
                    else np.ascontiguousarray(bias, dtype=np.float32)
                ),
                activation=(
                    activation
                    if isinstance(activation, str)
                    else "torch.nn.modules.linear.Identity"
                ),
            )
        except ValueError as error:
            raise PreparationError(
                f"the dense module {name!r} is not usable: {error}",
                stage=EXPORT_STAGE,
            ) from None


# --------------------------------------------------------------------------
# Verifying the graph really is pinned, and really is full precision
# --------------------------------------------------------------------------


def _verify_precision(path: Path, *, model_id: str) -> None:
    """Refuse a graph whose weights are not float32.

    Reduced-precision weights do not show up in the output dtype - a graph that
    casts float16 parameters up before use still emits float32 - so this reads
    the initializers rather than trusting the signature.
    """
    graph = onnx.load(str(path)).graph
    observed = {
        onnx.TensorProto.DataType.Name(initializer.data_type)
        for initializer in graph.initializer
        if initializer.data_type in _WEIGHT_ELEMENT_TYPES
    }
    if observed != {_FULL_PRECISION}:
        raise PreparationError(
            f"the exported graph carries weights of type "
            f"{sorted(observed) or 'no numeric type at all'}, not "
            f"{_FULL_PRECISION}. The export is full precision by design: BF16 "
            "targeting happens at session construction through the Vitis AI "
            "config_file option, where the execution provider performs the "
            "cast, and a graph quantised here would also stop being the "
            "full-precision reference requirement 6.3 compares against",
            model_id=model_id,
            stage=EXPORT_STAGE,
        )


def _verify_shape(path: Path, profile: ModelProfile) -> int:
    """Refuse a graph that is not pinned to exactly the profile's shape.

    Returns the trunk's hidden width, read from the graph rather than declared,
    because the graph is the thing that will actually run.

    The check goes through a real ONNX Runtime session rather than the protobuf,
    because a session is what the backends and the NPU compiler will build, and
    a session is where a symbolic dimension survives as a *string* where an
    integer was required.
    """
    # ``onnxruntime`` ships no annotations, so strict mode sees an untyped
    # import. The ignore is scoped to this one line rather than widened to a
    # per-module override, which would also hide real type errors in this file -
    # the same choice task 3.1 made for ``tqdm``.
    import onnxruntime as ort  # type: ignore[import-untyped]

    expected = [profile.batch_size, profile.compiled_seq_len]
    try:
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        raise PreparationError(
            f"the exported graph could not be loaded by ONNX Runtime: "
            f"{type(error).__name__}: {error}",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        ) from None

    inputs = {meta.name: meta for meta in session.get_inputs()}
    if set(inputs) != {INPUT_IDS, ATTENTION_MASK}:
        raise PreparationError(
            f"the exported graph takes {sorted(inputs)}, but the transformer "
            f"backend contract is exactly {sorted((INPUT_IDS, ATTENTION_MASK))}",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )

    for name in (INPUT_IDS, ATTENTION_MASK):
        meta = inputs[name]
        if meta.type != _INT64_TENSOR:
            raise PreparationError(
                f"the exported graph declares {name} as {meta.type}, not "
                f"{_INT64_TENSOR}",
                model_id=profile.model_id,
                stage=EXPORT_STAGE,
            )
        if list(meta.shape) != expected:
            raise PreparationError(
                f"the exported graph declares {name} with shape "
                f"{list(meta.shape)}, but {profile.name} is compiled at "
                f"{expected} (batch x sequence). A dimension left symbolic "
                "loads and runs on CPU and gives the NPU compiler no number to "
                "work with, so it is refused here rather than at compile time",
                model_id=profile.model_id,
                stage=EXPORT_STAGE,
            )

    outputs = session.get_outputs()
    if len(outputs) != 1 or outputs[0].name != TOKEN_EMBEDDINGS:
        raise PreparationError(
            f"the exported graph returns {[o.name for o in outputs]}, but the "
            f"trunk contract is a single {TOKEN_EMBEDDINGS} tensor; pooling, "
            "the dense stage and normalisation run outside the graph",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )

    output = outputs[0]
    if output.type != _FLOAT32_TENSOR:
        raise PreparationError(
            f"the exported graph declares {TOKEN_EMBEDDINGS} as {output.type}, "
            f"not {_FLOAT32_TENSOR}",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )

    shape = list(output.shape)
    if len(shape) != 3 or shape[:2] != expected:
        raise PreparationError(
            f"the exported graph declares {TOKEN_EMBEDDINGS} with shape "
            f"{shape}; token embeddings are (batch, sequence, hidden) at "
            f"{expected}",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )
    if not isinstance(shape[2], int):
        raise PreparationError(
            f"the exported graph leaves the hidden width of "
            f"{TOKEN_EMBEDDINGS} symbolic ({shape[2]!r})",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )
    return shape[2]


# --------------------------------------------------------------------------
# The dense stage
# --------------------------------------------------------------------------


def _verified_dense(
    stages: Sequence[DenseStage], profile: ModelProfile, *, hidden_size: int
) -> tuple[DenseStage, ...]:
    """Check the projections against what the profile says the model is.

    This is the Observable's second half, and every branch below exists because
    the corresponding mistake is invisible downstream: the vectors keep their
    shape, they keep their unit norm after normalisation, and only a retrieval
    quality measurement would eventually notice they mean nothing.
    """
    found = tuple(stages)

    if profile.has_dense_stage and not found:
        raise PreparationError(
            f"{profile.name} declares a dense stage but none was found beside "
            f"its weights. Its sentence pipeline is Pooling -> Dense -> "
            "Normalize and the ONNX export covers only the trunk, so the dense "
            "projection would simply be absent - producing vectors of the right "
            "shape and the right norm that are semantically wrong, which no "
            "shape check downstream can detect",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )

    if not profile.has_dense_stage and found:
        raise PreparationError(
            f"{profile.name} declares no dense stage, but its weights carry "
            f"{[stage.name for stage in found]}. Dropping a projection the "
            "model really has is the same silent corruption as omitting one it "
            "declares, so the disagreement is reported rather than resolved",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )

    if not found:
        return ()

    if found[0].in_features != hidden_size:
        raise PreparationError(
            f"the first dense stage {found[0].name!r} consumes "
            f"{found[0].in_features} features, but the exported trunk emits "
            f"{hidden_size}; pooling does not change the width, so these "
            "weights do not belong to this graph",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )

    for earlier, later in zip(found, found[1:], strict=False):
        if earlier.out_features != later.in_features:
            raise PreparationError(
                f"dense stage {earlier.name!r} produces "
                f"{earlier.out_features} features but {later.name!r} consumes "
                f"{later.in_features}; the projections do not chain, so they "
                "are not the pipeline they claim to be",
                model_id=profile.model_id,
                stage=EXPORT_STAGE,
            )

    if found[-1].out_features != profile.dimension:
        raise PreparationError(
            f"the dense stages end at {found[-1].out_features} features but "
            f"{profile.name} publishes a vector dimension of "
            f"{profile.dimension} (3.6); the profile and these weights describe "
            "different models",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )

    return found


def _write_dense(stages: Sequence[DenseStage], destination: Path) -> None:
    """Persist the projections, order included.

    One file, as design.md's Physical Data Model specifies. The order is written
    as data rather than left to key ordering, because a mapping's iteration
    order is not a contract anyone downstream should have to rely on for
    something this consequential.
    """
    payload: dict[str, npt.NDArray[Any]] = {
        DENSE_ORDER_KEY: np.array([stage.name for stage in stages])
    }
    for stage in stages:
        payload[f"{WEIGHT_PREFIX}{stage.name}"] = stage.weight
        payload[f"{ACTIVATION_PREFIX}{stage.name}"] = np.array(stage.activation)
        if stage.bias is not None:
            payload[f"{BIAS_PREFIX}{stage.name}"] = stage.bias
    with destination.open("wb") as handle:
        np.savez(handle, **payload)


# --------------------------------------------------------------------------
# Failure mapping
# --------------------------------------------------------------------------


def _guarded[T](
    action: Callable[[], T], *, step: str, profile: ModelProfile
) -> T:
    """Run one step of the export, translating any failure it raises.

    Every ordinary failure out of here is one of this feature's errors naming
    the export stage (4.6, 8.1, 8.2), and ``from None`` matches task 3.1's
    choice for acquisition: the cause's own text is already quoted in the
    message, and a chained traceback would print it a second time beneath the
    diagnosis.

    ``Exception``, not ``BaseException``: an operator interrupting a
    multi-minute export has not encountered a preparation failure, and dressing
    their ``KeyboardInterrupt`` up as one would be a worse report than the
    interrupt itself. Requirement 8.5's guarantee does not depend on catching it
    - `export_model` cleans up its partial writes in a ``finally``, which runs
    for an interrupt too.
    """
    try:
        return action()
    except EmbeddingRuntimeError:
        # Already diagnosed in this feature's vocabulary. Re-diagnosing would
        # bury a specific stage under a generic one.
        raise
    except Exception as error:
        raise PreparationError(
            f"could not export {profile.model_id} at the {step} step: "
            f"{type(error).__name__}: {error}",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        ) from None


def _emit(
    progress: ProgressCallback | None, operation: str, completed: int
) -> None:
    if progress is None:
        return
    progress(
        ProgressUpdate(operation=operation, completed=completed, total=_STEPS)
    )


def _discard(path: Path) -> None:
    """Remove a half-built artifact, and never fail while doing it."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def export_model(
    profile: ModelProfile,
    acquired: AcquiredModel,
    destination: Path,
    *,
    exporter: TrunkExporter | None = None,
    progress: ProgressCallback | None = None,
) -> ExportedModel:
    """Export ``profile``'s trunk to ONNX and persist its dense weights.

    The graph is pinned to the profile's ``batch_size`` and ``compiled_seq_len``
    and verified to declare exactly that shape before it is published, so the
    artifact a caller receives is one ONNX Runtime will hold to a single input
    shape and the NPU compiler can accept. It is exported at full precision;
    BF16 targeting is the backend's business.

    Every profile declaring ``has_dense_stage`` gets a ``dense.npz`` beside the
    graph, and a profile that declares one whose weights carry none fails here
    rather than producing plausible, meaningless vectors later.

    Nothing is written under its final name until every part has succeeded, so an
    interrupted export leaves no artifact a later run would treat as valid (8.5).
    Task 3.3 owns the manifest that decides reuse and staleness (4.3, 4.4, 4.7);
    this function only produces the two files it describes.

    Raises `PreparationError` naming the export stage for every failure (4.6,
    8.1), and never substitutes a different model - including when ``acquired``
    is not the model ``profile`` asked for.
    """
    if acquired.model_id != profile.model_id:
        raise PreparationError(
            f"asked to export {profile.model_id} but the acquired files are "
            f"{acquired.model_id}. Requirement 4.6 forbids substituting a "
            "different model, and this is the quietest place such a "
            "substitution could happen",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        )

    tool = exporter if exporter is not None else TorchTrunkExporter()
    operation = f"export:{profile.name}"
    onnx_final = destination / ONNX_FILENAME
    dense_final = destination / DENSE_FILENAME
    onnx_partial = destination / f"{ONNX_FILENAME}{_PARTIAL_SUFFIX}"
    dense_partial = destination / f"{DENSE_FILENAME}{_PARTIAL_SUFFIX}"

    _emit(progress, operation, 0)
    try:
        destination.mkdir(parents=True, exist_ok=True)

        _guarded(
            lambda: tool.export_graph(
                source=acquired.local_path,
                destination=onnx_partial,
                batch_size=profile.batch_size,
                sequence_length=profile.compiled_seq_len,
            ),
            step="graph export",
            profile=profile,
        )
        _emit(progress, operation, 1)

        _guarded(
            lambda: _verify_precision(onnx_partial, model_id=profile.model_id),
            step="graph verification",
            profile=profile,
        )
        hidden_size = _guarded(
            lambda: _verify_shape(onnx_partial, profile),
            step="graph verification",
            profile=profile,
        )
        _emit(progress, operation, 2)

        stages = _verified_dense(
            _guarded(
                lambda: tool.read_dense_stages(source=acquired.local_path),
                step="dense extraction",
                profile=profile,
            ),
            profile,
            hidden_size=hidden_size,
        )
        if stages:
            _guarded(
                lambda: _write_dense(stages, dense_partial),
                step="dense extraction",
                profile=profile,
            )
        _emit(progress, operation, 3)

        # Publication. Both renames happen only once everything above has
        # succeeded, which is what makes a half-finished export invisible.
        _guarded(
            lambda: os.replace(onnx_partial, onnx_final),
            step="publication",
            profile=profile,
        )
        if stages:
            _guarded(
                lambda: os.replace(dense_partial, dense_final),
                step="publication",
                profile=profile,
            )
    finally:
        # Whether the export succeeded or not: after a success the partials have
        # been renamed away and this is a no-op, and after a failure it is what
        # makes a half-written graph invisible to the next run (8.5).
        _discard(onnx_partial)
        _discard(dense_partial)

    try:
        exported = ExportedModel(
            model_id=profile.model_id,
            revision=acquired.revision,
            onnx_path=onnx_final,
            dense_path=dense_final if stages else None,
            batch_size=profile.batch_size,
            compiled_seq_len=profile.compiled_seq_len,
            hidden_size=hidden_size,
        )
    except ValueError as error:
        raise PreparationError(
            f"the export of {profile.model_id} did not produce a usable "
            f"result: {error}",
            model_id=profile.model_id,
            stage=EXPORT_STAGE,
        ) from None

    _emit(progress, operation, _STEPS)
    return exported
