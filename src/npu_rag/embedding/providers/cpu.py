"""The CPU adapter, and the full-precision reference path (task 4.2).

This is the simplest `TransformerBackend` there can be: it opens the prepared
``model.onnx`` on ONNX Runtime's default provider and runs it. Everything
interesting about it is what it refuses to do.

**It is the reference, so it must not touch the numbers.** design.md fixes the
precision decision at the backend rather than the export - "the exported
``model.onnx`` is FP32; BF16 targeting happens at session construction through
the Vitis AI ``config_file`` option, so the EP performs the cast. `CpuBackend`
therefore executes the same graph at full precision and is the natural reference
for 6.3 - the two backends differ in numeric precision alone, which is exactly
what that measurement isolates." So no session option is set here at all, no
array is cast, and the tensor the session produced is the tensor `run` returns,
by identity. A backend that rounded, re-typed or "cleaned up" its output would
make requirement 6.3 measure the adapter instead of the hardware.

**It never asks whether the NPU is there.** Requirement 2.3: a ``cpu`` selection
is served by the CPU *regardless of whether the NPU is available*. That is a
statement about both directions - an absent accelerator must not stop this
backend, and a present, healthy one must not divert it - so this module imports
nothing from ``environment`` and reads no capability report. Selection is
`resolve_backend`'s business and this adapter's business is to run.

**Session creation is not evidence.** Measured on this machine and recorded in
design.md: constructing a session with a provider ONNX Runtime does not have
**succeeds**, warns once, and silently runs the graph on the CPU. Here that
outcome would be the right one by accident, which is precisely why it is
checked: ``session.get_providers()[0]`` is read back, so this adapter reports
the CPU because the CPU served, not because the CPU was asked for. Guard one -
the provider is registered *before* anything is constructed - is kept too, and
its failure is an environment failure rather than an execution one: a runtime
that cannot offer its own built-in provider is a broken installation, not a
model that would not run (requirement 8.2).

**A wrong shape is refused, never repaired.** The graph is pinned to
``(batch_size, compiled_seq_len)`` and answers anything else with
``InvalidArgument`` (task 3.2). NPU execution requires static shapes, so a
partial batch is padded by the *caller*; an adapter that quietly reshaped one
would embed text nobody asked for, at the right shape and the right norm, and
nothing downstream would notice.

**What is deliberately not here.** No pooling, no Dense stage, no
normalization - design.md puts post-processing on the service side of the port
so it executes identically no matter which adapter ran the transformer (task
5.2). And no factory that prepares an artifact: the artifact is handed in,
because where artifacts are rooted is a service-level decision (task 5.3), and a
factory here would have to invent that root.

This module sits in the ``providers`` layer of design.md's dependency direction
- ``types, errors -> reporting -> profiles -> environment -> models ->
providers -> service -> bench`` - so it reads ``models`` and everything further
left, and never reaches ``service`` or ``bench``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from npu_rag.embedding.errors import EnvironmentError_, ExecutionError
from npu_rag.embedding.models.artifacts import PreparedArtifact
from npu_rag.embedding.models.export import ATTENTION_MASK, INPUT_IDS
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.types import ExecutionMode, ProviderChoice

__all__ = [
    "CPU_PROVIDER",
    "SESSION_STAGE",
    "CpuBackend",
    "Session",
    "SessionFactory",
    "available_providers",
    "build_session",
]

#: ONNX Runtime's built-in provider - the one every installation has, and the
#: one the runtime silently falls back to when a requested provider is absent.
CPU_PROVIDER: Final = "CPUExecutionProvider"

#: The stage a failure to obtain a usable session belongs to. errors.py
#: documents four preparation stages and leaves ``stage`` a free-form string;
#: this one is on the execution side of that line - the artifact is already
#: prepared - but it is not a failure *of the graph*, and telling an operator
#: which of the two happened is the whole point of requirement 8.1.
SESSION_STAGE: Final = "session"


# --------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------


class Session(Protocol):
    """An ONNX Runtime session, narrowed to what this adapter is allowed to use.

    Two members, and the absence of a third matters: there is no hook here for
    session options, so no code path through this module can set one. The
    default session is a full-precision session, and that is the property
    requirement 6.3 rests on.
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


class SessionFactory(Protocol):
    """Builds one session for one graph on one provider list.

    A seam for the same reason acquisition, export and compilation have one: the
    refusal branches below - a session that came back on another provider, a
    session that will not build at all - cannot be reached with a real runtime,
    because a real runtime cannot be asked to misbehave on demand.
    """

    def __call__(
        self, model_path: Path, providers: Sequence[str], /
    ) -> Session: ...


def available_providers() -> tuple[str, ...]:
    """Which execution providers this interpreter's runtime actually offers.

    design.md's guard one. It is a module-level function rather than an inline
    import so a test can substitute it, which is the only way to reach the
    branch below on a machine whose runtime is intact.
    """
    # ``onnxruntime`` ships no annotations, so strict mode sees an untyped
    # import; the ignore is scoped to this line. The import is inside the
    # function so this module - and therefore the package - imports in an
    # environment that has never seen the vendor wheels.
    import onnxruntime as ort  # type: ignore[import-untyped]

    return tuple(str(name) for name in ort.get_available_providers())


def build_session(model_path: Path, providers: Sequence[str], /) -> Session:
    """The real `SessionFactory`: a default, full-precision session.

    No ``SessionOptions`` is constructed and no session configuration entry is
    set. That is not an omission - it is the reference path, and every knob that
    exists here would change the numbers requirement 6.3 compares against.
    """
    # The untyped-import ignore sits on the first of these two imports, in
    # `available_providers`; mypy reports one per module, so repeating it here
    # would itself be an error under `warn_unused_ignores`.
    import onnxruntime as ort

    session: Session = ort.InferenceSession(
        str(model_path), providers=list(providers)
    )
    return session


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class CpuBackend:
    """Runs a prepared graph on the CPU at full precision (2.3, 6.3).

    The graph is the artifact's ``model.onnx`` - the FP32 trunk - never the
    compiled snapshot beside it in an NPU artifact, which is a different
    provider's build of a different thing.
    """

    def __init__(
        self,
        artifact: PreparedArtifact,
        profile: ModelProfile,
        *,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._profile = profile
        self._graph = artifact.onnx_path
        self._shape = (profile.batch_size, profile.compiled_seq_len)
        self._session = self._open(
            session_factory if session_factory is not None else build_session
        )

    # -- the contract ------------------------------------------------------

    @property
    def provider(self) -> ProviderChoice:
        """The CPU - and it is reported because the CPU served, not because it
        was requested; the constructor read the session back to find out."""
        return ProviderChoice.CPU

    @property
    def execution_mode(self) -> ExecutionMode:
        """This interpreter. The isolated worker exists for the vendor runtime
        (requirement 5.1); the CPU provider is here in every environment."""
        return ExecutionMode.IN_PROCESS

    @property
    def npu_partition_share(self) -> float | None:
        """Not applicable, which the protocol spells ``None``.

        Never a number. A share reads as a *verified offload*, and there is no
        offload here to verify - reporting one would let a CPU run be mistaken
        for a hardware-accelerated one in exactly the reports requirement 2.6
        exists to keep honest.
        """
        return None

    def run(
        self,
        token_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Token embeddings for one padded batch, at full precision.

        No pooling, no Dense stage, no normalization: the caller still holds the
        mask it passed, and post-processing runs once, on the service side of
        the port, for every backend (task 5.2).
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
                f"running {self._profile.name} over a batch of "
                f"{self._shape[0]} x {self._shape[1]} failed: "
                f"{type(error).__name__}: {error}"
            ) from None
        return self._only_tensor(outputs)

    # -- getting a session, and proving it is the right one -----------------

    def _open(self, build: SessionFactory) -> Session:
        """Guard one, then build, then guard two. In that order, for a reason:
        a session built on an unregistered provider comes back looking healthy,
        so the pre-check must happen while there is still nothing to inspect."""
        registered = available_providers()
        if CPU_PROVIDER not in registered:
            raise EnvironmentError_(
                f"{CPU_PROVIDER} is not registered in this interpreter, so "
                f"nothing here can execute {self._profile.model_id}. Available "
                f"providers: {', '.join(registered) or 'none'}. That provider "
                "is built into every ONNX Runtime build, so its absence means "
                "the installed runtime is incomplete rather than that this "
                "model cannot run",
                provider=ProviderChoice.CPU,
                model_id=self._profile.model_id,
            )

        try:
            session = build(self._graph, (CPU_PROVIDER,))
        except Exception as error:
            raise ExecutionError(
                f"opening {self._graph.name} for {self._profile.model_id} on "
                f"{CPU_PROVIDER} failed: {type(error).__name__}: {error}",
                provider=ProviderChoice.CPU,
                model_id=self._profile.model_id,
                stage=SESSION_STAGE,
            ) from None

        served = tuple(str(name) for name in session.get_providers())
        if not served or served[0] != CPU_PROVIDER:
            raise ExecutionError(
                f"a session was requested on {CPU_PROVIDER} and came back on "
                f"{', '.join(served) or 'no provider at all'}. Requesting a "
                "provider the runtime does not have succeeds and runs the graph "
                "somewhere else, so what came back is read rather than assumed: "
                "this backend reports the provider that ran",
                provider=ProviderChoice.CPU,
                model_id=self._profile.model_id,
                stage=SESSION_STAGE,
            )
        return session

    # -- what goes in, and what comes back ----------------------------------

    def _check(self, label: str, array: npt.NDArray[np.int64]) -> None:
        """Refuse a batch the compiled graph was not built for.

        Refuse, rather than reshape or cast. The compiled length is a contract
        the caller can measure for itself (requirement 3.8), and repairing a
        mismatch here would silently embed something other than what was asked
        for while every shape and norm downstream stayed plausible.
        """
        if array.shape != self._shape:
            raise self._failed(
                f"{label} arrived with shape {array.shape}, but "
                f"{self._profile.name} is compiled at {self._shape} "
                "(batch x sequence). A partial batch is padded by the caller, "
                "because static shapes are what make the same batch runnable on "
                "either provider"
            )
        if array.dtype != np.int64:
            raise self._failed(
                f"{label} arrived as {array.dtype}, but the graph declares it "
                "as int64. Casting it here would hide a tokenizer that is "
                "producing the wrong type"
            )

    def _only_tensor(self, outputs: Sequence[Any]) -> npt.NDArray[np.float32]:
        """The graph's single tensor, checked against what was fed in.

        Returned by identity - this is the reference path, so the array the
        session produced is the array the caller receives.
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
                "float32. This backend is the full-precision reference the "
                "benchmark compares reduced-precision execution against, so a "
                "narrower type here would leave that comparison measuring "
                "nothing"
            )
        if produced.ndim != 3 or produced.shape[:2] != self._shape:
            raise self._failed(
                f"the graph returned shape {produced.shape} for a batch of "
                f"{self._shape}; token embeddings are (batch, sequence, hidden)"
            )
        embeddings: npt.NDArray[np.float32] = produced
        return embeddings

    def _failed(self, message: str, *, stage: str | None = None) -> ExecutionError:
        """One execution failure, carrying requirement 8.1's three facts."""
        return ExecutionError(
            message,
            provider=ProviderChoice.CPU,
            model_id=self._profile.model_id,
            stage=stage,
        )
