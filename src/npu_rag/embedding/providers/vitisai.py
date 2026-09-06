"""The NPU adapter, with partition verification (task 4.3).

This is the `TransformerBackend` that runs the compiled snapshot on the Vitis AI
execution provider. It runs ``context.onnx`` - the EP-context snapshot task 3.3
publishes - never ``model.onnx``, which is the CPU backend's FP32 trunk (task
4.2). Reduced precision is a property of *this* backend: the ``config_file``
provider option switches the device data type to bfloat16 at session
construction, so the EP performs the cast (design.md, "Precision is a property of
the backend, not the export").

**Three checks, in order, because session creation proves nothing.** Measured on
this machine and recorded in design.md: constructing a session with
``providers=["VitisAIExecutionProvider"]`` when the EP is absent *succeeds*,
warns once, and silently runs the graph on the CPU - ``provider_options`` making
no difference. So:

1. **Guard one** - the provider must be in ``get_available_providers()`` *before*
   anything is constructed. Reused from ``environment/capability.py`` rather than
   re-implemented.
2. **Guard two** - ``session.get_providers()[0]`` must equal the provider
   *after* construction. A session that came back CPU-first is refused, not
   assumed, whatever the selection: a session claiming the NPU while running on
   the CPU is dishonest.
3. **Partition verification** - the third and finest check, and the one this
   task is really about.

**The partition signal, per design.md's 2026-09-06 amendment.** The compiler's
diagnostics (``preliminary-vaiml-pass-summary.txt``) do *not* survive the
EP-context flow: the cache directory is empty by the time the session returns, so
every published manifest carries ``observed_partition_share = null``. Applied
literally, the original "unverifiable ⇒ fail under explicit npu" policy would
fire on *every* NPU run. The durable evidence is the published ``context.onnx``'s
own node mix: after EP-context compilation the graph carries one ``EPContext``
node standing in for everything the compiler offloaded, plus any ops it did
*not* offload as ordinary nodes beside it. A live MiniLM artifact reads
``{EPContext: 1, Cast: 1, Gather: 1, GatherND: 1}``.

**Turning a node mix into a share.** A naive ``offloaded / total`` node ratio is
meaningless: the single ``EPContext`` node stands in for hundreds of original
ops, so that ratio reads a near-total offload as ~25% CPU-bound. Instead the
share is measured against the *original* ``model.onnx`` in the same artifact
directory::

    share = clamp01((N_original_nodes - R_residue_nodes) / N_original_nodes)

where ``R`` is the count of non-``EPContext`` nodes left in ``context.onnx`` -
the CPU residue - and ``N`` is the original trunk's node count. This is the
complement of the CPU residue fraction: "how much of the original graph is *not*
sitting loose on the CPU". It is deliberately conservative - compiler glue in the
residue (Cast/Gather) is not a subset of the original graph, so it counts against
the NPU share and can only *lower* the reported number, never inflate it.
Measured on this machine (2026-09-06): a live MiniLM artifact has a 251-node
trunk and a residue of ``{Cast: 1, Gather: 1, GatherND: 1}``, so it reads
``(251 - 3) / 251 = 0.988``; a graph the compiler largely rejected reads near 0.

**Unverifiable** now means *the node mix could not be read* - a missing or
corrupt ``context.onnx``, an unreadable ``model.onnx`` (no denominator), or
**zero ``EPContext`` nodes** (nothing was compiled to the NPU). It is `None`,
never zero: `errors.PartitionShareTooLow` keeps measured-low (a number below the
threshold) apart from unverifiable (`None`) because design.md requires both to
fail under explicit ``npu`` while staying distinguishable in the report.

**The provider-choice policy this backend implements.**

- Under explicit **``npu``**: unverifiable partitioning is a failure
  (`PartitionShareTooLow` with an ``observed_share`` of `None`); below-threshold
  is a failure (a measured ``observed_share``). The caller demanded the NPU;
  proceeding without evidence would reintroduce the silent degradation
  requirement 2 exists to prevent (2.2).
- Under **``auto``**: unverifiable or below-threshold partitioning is *recorded*
  and execution proceeds, since the caller already accepted substitution. The
  weakness is carried as data - ``npu_partition_share`` stays `None` or the low
  number - never printed.

**Reporting whether verification happened (2.6).** ``npu_partition_share`` is the
surface: a float at or above `MINIMUM_PARTITION_SHARE` is a *verified* offload; a
value below it, or `None`, is an *assumed* run that only ``auto`` reaches. The
service (task 5.3) maps this onto ``EmbedResult.partition_verified`` - `True`
when ``share is not None and share >= MINIMUM_PARTITION_SHARE``, `False`
otherwise - which is why the threshold is exported. The backend keeps exactly the
protocol's four members: `BoundBackend` delegates only those, so a fifth would be
invisible to the service anyway.

This module sits in the ``providers`` layer of design.md's dependency direction -
``types, errors -> reporting -> profiles -> environment -> models -> providers ->
service -> bench`` - so it reads ``models``, ``environment`` and everything
further left, and never reaches ``service`` or ``bench``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from npu_rag.embedding.environment.capability import (
    VITISAI_PROVIDER,
    probe_onnx_runtime,
)
from npu_rag.embedding.errors import (
    EnvironmentError_,
    ExecutionError,
    PartitionShareTooLow,
)
from npu_rag.embedding.models.artifacts import CONTEXT_FILENAME, PreparedArtifact
from npu_rag.embedding.models.export import ATTENTION_MASK, INPUT_IDS
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.types import ExecutionMode, ProviderChoice

__all__ = [
    "EP_CONTEXT_OP",
    "MINIMUM_PARTITION_SHARE",
    "SESSION_STAGE",
    "VITISAI_PROVIDER",
    "NpuSession",
    "NpuSessionFactory",
    "PartitionReader",
    "VitisAIBackend",
    "build_npu_session",
    "npu_provider_registered",
    "read_node_partition_share",
]

#: The op type ONNX Runtime writes into an EP-context snapshot for the subgraph
#: the compiler offloaded. Its presence - at least one such node - is the
#: minimum evidence that anything was compiled to the NPU at all.
EP_CONTEXT_OP: Final = "EPContext"

#: The fraction of the original graph that must be offloaded, below which the
#: graph is "mostly on the CPU" and verification fails under explicit ``npu``.
#:
#: ``0.5`` is the boundary of "mostly": below it, the majority of the original
#: trunk's nodes are left loose on the CPU rather than folded into the NPU
#: context. It is named rather than inlined so the service can compute
#: ``partition_verified`` against the same number, and it sits strictly inside
#: ``(0, 1)`` with headroom on both sides: a live MiniLM offload measures 0.988
#: (251 trunk nodes, 3 residue), well clear, and a graph the compiler could not
#: offload measures near 0. A
#: threshold of 1.0 would reject every real offload (glue in the residue keeps
#: the honest number just under 1.0); a threshold of 0.0 would accept a graph
#: wholly on the CPU.
MINIMUM_PARTITION_SHARE: Final = 0.5

#: The stage a failure to obtain a usable NPU session belongs to (8.1). The
#: artifact is already prepared, so this is on the execution side of the line;
#: partition verification proper reports errors.py's ``verify`` stage through
#: `PartitionShareTooLow`.
SESSION_STAGE: Final = "session"

#: ONNX Runtime's built-in provider, the one a session silently falls back to
#: when the requested provider is absent. Named so guard two can tell "the NPU
#: served" from "the CPU served while the session claimed the NPU".
_CPU_PROVIDER: Final = "CPUExecutionProvider"

#: The Vitis AI ``config_file`` provider option resolves here on a provisioned
#: machine (Implementation Note 1.2). It is what switches the device data type to
#: bfloat16, so a session built without it does not engage the NLP flow this
#: feature needs.
_VAIP_CONFIG_RELATIVE: Final = ("capi", "vaip_config.json")


# --------------------------------------------------------------------------
# The session seam
# --------------------------------------------------------------------------


class NpuSession(Protocol):
    """An ONNX Runtime session, narrowed to what this adapter uses.

    Two members, the same pair the CPU adapter's `Session` exposes. The
    difference is entirely in the *factory* below, which - unlike the CPU one -
    can express ``provider_options``, because that is where reduced-precision
    targeting is supplied.
    """

    def get_providers(self) -> Sequence[str]:
        """The providers this session actually holds, best first."""
        ...

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
        /,
    ) -> Sequence[Any]:
        """Execute the graph over one feed."""
        ...


class NpuSessionFactory(Protocol):
    """Builds one Vitis AI session for one compiled snapshot.

    This is the seam task 4.2 deliberately could not offer: its CPU
    ``SessionFactory`` cannot express ``SessionOptions`` or ``provider_options``,
    which is what makes "the CPU reduces precision nowhere" structural. The NPU
    path needs both - ``config_file`` engages bfloat16 - so this backend defines
    its own factory rather than widening the CPU one.
    """

    def __call__(
        self, model_path: Path, *, config_file: Path | None
    ) -> NpuSession: ...


def npu_provider_registered() -> bool:
    """Guard one, reusing ``environment/capability.py``'s probe.

    A module-level function so a test can substitute it - the only way to reach
    the guard's failure branch on a machine whose runtime is intact. The probe
    imports ONNX Runtime lazily and never raises, so this module stays importable
    where the vendor wheels are absent.
    """
    return probe_onnx_runtime().registers


def build_npu_session(
    model_path: Path, *, config_file: Path | None
) -> NpuSession:
    """The real `NpuSessionFactory`: a Vitis AI session over a compiled snapshot.

    ``config_file`` supplies the bfloat16 targeting, and it is the **only**
    provider option passed. Task 3.3 passes ``cache_dir``/``cache_key`` too, but
    at *compile* time, to produce the snapshot; loading an EP-context snapshot is
    a different operation the EP treats differently. Measured on this machine
    (2026-09-06, review of this task): supplying a ``cache_key`` when loading a
    context model whose baked-in key differs makes the EP call ``abort()`` -
    not raise - with "your cache key '128' is different from the one in the EP
    context model 'artifact'. This is not allowed. Please remove the cache key
    from provider options". A ``try``/``except`` cannot catch an abort; the
    interpreter dies. The compiled binary is not in any cache anyway: the
    ``EPContext`` node references its sidecar (``context.onnx_VITISAI.bin``) by
    bare filename, which ONNX Runtime resolves relative to the model file - so
    the snapshot is opened by its real path in its artifact directory.
    """
    # ``onnxruntime`` ships no annotations, so strict mode sees an untyped
    # import; the ignore is scoped to this line. The import is inside the
    # function so this module - and the package - imports where the vendor wheels
    # have never been installed.
    import onnxruntime as ort  # type: ignore[import-untyped]

    resolved = (
        config_file
        if config_file is not None
        else Path(ort.__file__).parent.joinpath(*_VAIP_CONFIG_RELATIVE)
    )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"the Vitis AI configuration file {resolved} is missing; it is the "
            "config_file provider option that switches the device data type to "
            "bfloat16"
        )
    session: NpuSession = ort.InferenceSession(
        str(model_path),
        providers=[VITISAI_PROVIDER],
        provider_options=[{"config_file": str(resolved)}],
    )
    return session


# --------------------------------------------------------------------------
# The node-mix partition metric
# --------------------------------------------------------------------------


class PartitionReader(Protocol):
    """Reads the offload share from a compiled snapshot and its trunk."""

    def __call__(
        self, *, context_path: Path, model_path: Path
    ) -> float | None: ...


def read_node_partition_share(
    *, context_path: Path, model_path: Path
) -> float | None:
    """The fraction of the graph the compiler offloaded, from the node mix.

    Reads ``context_path`` (the compiled snapshot) and ``model_path`` (the
    original FP32 trunk) and returns::

        clamp01((N_original_nodes - R_residue_nodes) / N_original_nodes)

    where ``R`` counts the non-``EPContext`` nodes left in the snapshot. Returns
    `None` - *unverifiable* - when the snapshot cannot be read, when it carries
    no ``EPContext`` node (nothing was compiled to the NPU), or when the trunk
    cannot be read (no denominator). Never raises: an unreadable graph is data,
    and the caller's provider-choice policy decides what to do with `None`.
    """
    context_ops = _op_counts(context_path)
    if context_ops is None:
        return None
    if context_ops.get(EP_CONTEXT_OP, 0) < 1:
        # No EPContext node means the compiler offloaded nothing - the snapshot
        # is not a verified NPU build, whatever else it contains. This is the
        # unverifiable case, never "100% offloaded".
        return None
    residue = sum(
        count for op, count in context_ops.items() if op != EP_CONTEXT_OP
    )
    original = _node_total(model_path)
    if original is None or original <= 0:
        return None
    share = (original - residue) / original
    if share < 0.0:
        return 0.0
    if share > 1.0:
        return 1.0
    return share


def _op_counts(path: Path) -> dict[str, int] | None:
    """Every op type in the graph at ``path`` and how often it occurs, or `None`."""
    graph = _load_graph(path)
    if graph is None:
        return None
    counts: dict[str, int] = {}
    for node in graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _node_total(path: Path) -> int | None:
    """The number of nodes in the graph at ``path``, or `None` if unreadable."""
    graph = _load_graph(path)
    return None if graph is None else len(graph.node)


def _load_graph(path: Path) -> Any | None:
    """Deserialize the graph at ``path``, or `None` if it cannot be read.

    ``load_external_data=False`` skips sidecar weight files, which keeps the
    read cheap for graphs that store weights externally. It does **not** help
    EmbeddingGemma: its 1.22 GB trunk stores weights *inline* (Implementation
    Note 3.2), so counting its nodes means parsing the whole protobuf once. That
    cost is paid once per backend construction - once per operation, never per
    batch - and is small beside the session build it accompanies. A streaming
    node count that skipped initializer bytes would be cheaper but would mean a
    hand-rolled protobuf reader for a number that is only ever read once.
    """
    # ``onnx`` is a default dependency and safe at module scope; it is imported
    # here to keep the graph-reading concern in one place.
    import onnx

    try:
        return onnx.load(str(path), load_external_data=False).graph
    except Exception:  # noqa: BLE001 - an absent or corrupt graph is data, not a raise
        return None


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class VitisAIBackend:
    """Runs a compiled snapshot on the NPU, with partition verification (2.2, 2.6).

    Constructed once per operation with the selection that produced it: an
    explicit ``npu`` request makes below-threshold or unverifiable partitioning a
    failure, while ``auto`` records the weakness and proceeds. The two verdicts
    on the identical graph are what make the provider-choice policy observable.
    """

    def __init__(
        self,
        artifact: PreparedArtifact,
        profile: ModelProfile,
        requested: ProviderChoice,
        *,
        session_factory: NpuSessionFactory | None = None,
        config_file: Path | None = None,
        partition_reader: PartitionReader | None = None,
        provider_probe: Callable[[], bool] | None = None,
    ) -> None:
        self._profile = profile
        self._requested = _npu_or_auto(requested, profile)
        self._config_file = config_file
        self._context = _require_context(artifact, profile)
        self._model = artifact.onnx_path
        self._shape = (profile.batch_size, profile.compiled_seq_len)
        probe = provider_probe if provider_probe is not None else npu_provider_registered
        build = session_factory if session_factory is not None else build_npu_session
        read = (
            partition_reader
            if partition_reader is not None
            else read_node_partition_share
        )
        self._session = self._open(probe, build)
        self._share = self._verify(read)

    # -- the contract ------------------------------------------------------

    @property
    def provider(self) -> ProviderChoice:
        """The NPU - reported because the session came back on it (guard two)."""
        return ProviderChoice.NPU

    @property
    def execution_mode(self) -> ExecutionMode:
        """This interpreter. The isolated worker (task 4.4) reports ISOLATED."""
        return ExecutionMode.IN_PROCESS

    @property
    def npu_partition_share(self) -> float | None:
        """The verified offload fraction, or `None` when unverifiable.

        Under explicit ``npu`` this is always a float at or above
        `MINIMUM_PARTITION_SHARE` - construction raised otherwise. Under ``auto``
        it may be `None` or a low number: the recorded warning that the caller,
        or the benchmark, reads to tell "verified on NPU" from "assumed on NPU".
        """
        return self._share

    def run(
        self,
        token_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Token embeddings for one padded batch, off the compiled snapshot.

        No pooling, no Dense stage, no normalization: post-processing runs once,
        on the service side of the port, for every backend (task 5.2).
        """
        self._check(INPUT_IDS, token_ids)
        self._check(ATTENTION_MASK, attention_mask)
        feed: Mapping[str, Any] = {
            INPUT_IDS: token_ids,
            ATTENTION_MASK: attention_mask,
        }
        try:
            outputs = self._session.run(None, feed)
        except Exception as error:
            raise self._failed(
                f"running {self._profile.name} on the NPU over a batch of "
                f"{self._shape[0]} x {self._shape[1]} failed: "
                f"{type(error).__name__}: {error}"
            ) from None
        return self._only_tensor(outputs)

    # -- getting a session, and proving it is the right one -----------------

    def _open(
        self, probe: Callable[[], bool], build: NpuSessionFactory
    ) -> NpuSession:
        """Guard one, then build, then guard two - in that order, because a
        session built on an unregistered provider comes back looking healthy."""
        if not probe():
            raise EnvironmentError_(
                f"{VITISAI_PROVIDER} is not registered in this interpreter, so "
                f"nothing here can run {self._profile.model_id} on the NPU. "
                "Requesting an absent provider from ONNX Runtime succeeds and "
                "runs the graph on the CPU, so this is checked before a session "
                "is built rather than after",
                provider=ProviderChoice.NPU,
                model_id=self._profile.model_id,
            )

        try:
            session = build(self._context, config_file=self._config_file)
        except Exception as error:
            raise ExecutionError(
                f"opening {self._context.name} for {self._profile.model_id} on "
                f"{VITISAI_PROVIDER} failed: {type(error).__name__}: {error}",
                provider=ProviderChoice.NPU,
                model_id=self._profile.model_id,
                stage=SESSION_STAGE,
            ) from None

        served = tuple(str(name) for name in session.get_providers())
        if not served or served[0] != VITISAI_PROVIDER:
            raise ExecutionError(
                f"a session was requested on {VITISAI_PROVIDER} and came back on "
                f"{', '.join(served) or 'no provider at all'}. Requesting a "
                "provider the runtime does not have succeeds and runs the graph "
                "somewhere else, so a session that is not NPU-first ran on the "
                f"{_CPU_PROVIDER} and is not this backend's session",
                provider=ProviderChoice.NPU,
                model_id=self._profile.model_id,
                stage=SESSION_STAGE,
            )
        return session

    def _verify(self, read: PartitionReader) -> float | None:
        """Apply the provider-choice policy to the node-mix share.

        Under explicit ``npu`` an unverifiable or below-threshold share is a
        failure; under ``auto`` it is recorded and returned so execution
        proceeds.
        """
        share = read(context_path=self._context, model_path=self._model)
        if self._requested is not ProviderChoice.NPU:
            return share
        if share is None:
            raise PartitionShareTooLow(
                f"the NPU partition of {self._profile.model_id} could not be "
                f"verified: {self._context.name}'s node mix carries no "
                f"{EP_CONTEXT_OP} node, or could not be read. Under an explicit "
                "'npu' selection an unverified partition is a failure - "
                "proceeding without evidence is the silent CPU degradation "
                "requirement 2 exists to prevent",
                observed_share=None,
                minimum_share=MINIMUM_PARTITION_SHARE,
                provider=ProviderChoice.NPU,
                model_id=self._profile.model_id,
            )
        if share < MINIMUM_PARTITION_SHARE:
            raise PartitionShareTooLow(
                f"only {share:.1%} of {self._profile.model_id} was offloaded to "
                f"the NPU, below the {MINIMUM_PARTITION_SHARE:.0%} threshold: the "
                "graph is running mostly on the CPU. Under an explicit 'npu' "
                "selection that is a failure, not a silent fallback",
                observed_share=share,
                minimum_share=MINIMUM_PARTITION_SHARE,
                provider=ProviderChoice.NPU,
                model_id=self._profile.model_id,
            )
        return share

    # -- what goes in, and what comes back ----------------------------------

    def _check(self, label: str, array: npt.NDArray[np.int64]) -> None:
        """Refuse a batch the compiled graph was not built for.

        Refuse, rather than reshape or cast: the compiled shape is a contract the
        caller can measure (3.8), and repairing a mismatch here would embed
        something other than what was asked for while every shape stayed
        plausible.
        """
        if array.shape != self._shape:
            raise self._failed(
                f"{label} arrived with shape {array.shape}, but "
                f"{self._profile.name} is compiled at {self._shape} "
                "(batch x sequence). A partial batch is padded by the caller, "
                "because static shapes are what the NPU compilation fixed"
            )
        if array.dtype != np.int64:
            raise self._failed(
                f"{label} arrived as {array.dtype}, but the graph declares it as "
                "int64. Casting it here would hide a tokenizer producing the "
                "wrong type"
            )

    def _only_tensor(self, outputs: Sequence[Any]) -> npt.NDArray[np.float32]:
        """The graph's single token-embedding tensor, checked against the inputs.

        Returned by identity - the array the session produced is the array the
        caller receives, so nothing is fabricated between the NPU and the service.
        """
        if len(outputs) != 1:
            raise self._failed(
                f"the graph returned {len(outputs)} tensors; the trunk contract "
                "is one tensor of token embeddings"
            )
        produced = outputs[0]
        if not isinstance(produced, np.ndarray):
            raise self._failed(
                f"the graph returned {type(produced).__name__}, not an array"
            )
        if produced.dtype != np.float32:
            raise self._failed(
                f"the graph returned token embeddings as {produced.dtype}, not "
                "float32. The EP casts to bfloat16 internally for compute, but "
                "the output tensor is float32; a narrower type here is a defect"
            )
        if produced.ndim != 3 or produced.shape[:2] != self._shape:
            raise self._failed(
                f"the graph returned shape {produced.shape} for a batch of "
                f"{self._shape}; token embeddings are (batch, sequence, hidden)"
            )
        embeddings: npt.NDArray[np.float32] = produced
        return embeddings

    def _failed(self, message: str) -> ExecutionError:
        """One execution failure, carrying requirement 8.1's three facts."""
        return ExecutionError(
            message,
            provider=ProviderChoice.NPU,
            model_id=self._profile.model_id,
        )


def _npu_or_auto(requested: ProviderChoice, profile: ModelProfile) -> ProviderChoice:
    """Only ``npu`` or ``auto`` reach this backend; ``cpu`` is a category error.

    ``ProviderChoice(requested)`` first, because a raw ``"npu"`` string compares
    equal to the member but is not identical to it, and the policy branches on
    identity.
    """
    choice = ProviderChoice(requested)
    if choice not in (ProviderChoice.NPU, ProviderChoice.AUTO):
        raise ValueError(
            f"the NPU backend serves 'npu' and 'auto' selections; {choice.value!r} "
            f"belongs to another provider and resolution never builds this "
            f"backend for it (model {profile.model_id})"
        )
    return choice


def _require_context(artifact: PreparedArtifact, profile: ModelProfile) -> Path:
    """The compiled snapshot to run, or an execution error if there is none.

    A CPU artifact has no ``context.onnx``; the NPU backend cannot run one that
    is not there, and refuses rather than reaching for ``model.onnx`` - which
    would run the FP32 trunk on the NPU with no compiled binary behind it.
    """
    context = artifact.context_path
    if context is None or not context.is_file():
        raise ExecutionError(
            f"the artifact for {profile.model_id} carries no {CONTEXT_FILENAME} "
            "snapshot, so there is nothing compiled for the NPU to run. A CPU "
            "artifact has only the FP32 trunk; running that on the NPU is not "
            "what this backend does",
            provider=ProviderChoice.NPU,
            model_id=profile.model_id,
            stage=SESSION_STAGE,
        )
    return context
