"""The backend contract and the provider-resolution policy (task 4.1).

This module is the port half of design.md's ports-and-adapters seam. It declares
what an execution backend *is* and decides which one an operation gets; it
executes nothing. The adapters that do - `CpuBackend` (task 4.2),
`VitisAIBackend` (task 4.3), `IsolatedBackend` (task 4.4) - implement
`TransformerBackend` and are handed in, so this module never imports a session,
a provider, or a vendor DLL.

**The backend returns token embeddings and nothing more.** design.md fixes this
as a binding decision: "Post-processing sits on the service side of the port, so
pooling, the Dense stage, and normalization execute identically no matter which
adapter ran the transformer. This makes 3.5 structural and collapses 5.4's
equivalence surface to the transformer alone." So `TransformerBackend.run`
returns the ``(batch, seq, hidden)`` token embeddings for the mask its caller
already holds, and masked mean pooling, the Dense stage and L2 normalization all
belong to task 5.2. Nothing in this file pools, projects or normalizes anything,
and a test asserts that structurally rather than trusting this paragraph.

**Why the mask is not returned.** design.md is self-inconsistent here: the
TransformerBackend prose says the backend "returns token embeddings plus the
attention mask", while its own Service Interface sketch types ``run`` as
returning ``npt.NDArray[np.float32]`` alone. The sketch is followed. The mask is
the caller's own argument - it goes in and is still in scope on the way out - so
returning a copy of it would create a second, competing mask that could disagree
with the first after a padding change. The pair post-processing needs is
``(the mask you passed, the embeddings you got back)``, which is what this
signature yields.

**Why the resolution policy is separate from backend construction.** design.md
sketches ``resolve_backend(choice, profile, capability)``. Constructing the
concrete adapters here would mean importing them, and they in turn import this
module for the protocol - a cycle inside one layer. The construction step is
therefore injected as `BackendFactories`. That keeps the policy - the whole of
requirements 2.1, 2.2, 2.4, 2.5 and 2.7 - decidable and testable on a machine
with no NPU at all, which is the only way both branches of requirement 2 can be
reached from one machine.

**What is deliberately not here.** design.md's two mandatory provider guards
split across two tasks:

1. **Pre-check** - ``get_available_providers()`` must contain the requested
   provider *before* anything is constructed. That is already implemented, in
   ``environment/capability.py``: the ``execution_provider_registered``
   condition is exactly that probe, and this module consumes its verdict through
   `CapabilityReport` rather than repeating it.
2. **Post-check** - ``session.get_providers()[0]`` must equal the requested
   provider *after* construction. That belongs to the concrete NPU adapter (task
   4.3), because only the component that builds a session can interrogate it.

The reason both exist is measured, not theoretical: requesting an absent
execution provider from ONNX Runtime **succeeds**, emits only a ``UserWarning``,
and silently runs the graph on the CPU, with ``provider_options`` making no
difference. A ``try``/``except`` around session construction detects nothing.
`npu_partition_share` is the third and finest check's surface; task 4.3
populates it from the published ``context.onnx``'s node mix, per design.md's
2026-09-06 amendment.

This module sits in the ``providers`` layer of design.md's dependency direction
- ``types, errors -> reporting -> profiles -> environment -> models -> providers
-> service -> bench`` - so it may read everything to its left and must never
reach ``service`` or ``bench``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from npu_rag.embedding.errors import NpuUnavailableError
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.types import (
    CapabilityReport,
    Condition,
    ExecutionMode,
    ProviderChoice,
)

__all__ = [
    "BackendFactories",
    "BackendFactory",
    "BoundBackend",
    "TransformerBackend",
    "npu_unavailable_reason",
    "resolve_backend",
]


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


@runtime_checkable
class TransformerBackend(Protocol):
    """Executes a prepared graph and returns token embeddings, nothing more.

    Four members, and the absence of any fifth is the point. There is no
    ``pool``, no ``normalize``, no ``embed``: an adapter that offered one would
    let two providers' post-processing drift apart, which is the silent
    divergence requirement 3.5 and design.md's port placement exist to prevent.

    Preconditions: ``token_ids`` and ``attention_mask`` are both shaped
    ``(batch, compiled_seq_len)`` and dtype ``int64``. NPU execution requires
    static shapes, so a partial batch is padded by the *caller* rather than
    reshaped here.

    Postconditions: the returned array is shaped
    ``(batch, compiled_seq_len, hidden)`` and dtype ``float32``.
    """

    @property
    def provider(self) -> ProviderChoice:
        """Which provider this adapter actually runs on: ``npu`` or ``cpu``.

        Never ``auto``. ``auto`` is a request, and this is an outcome - the
        distinction requirement 2.6 rests on.
        """
        ...

    @property
    def execution_mode(self) -> ExecutionMode:
        """Whether the work happens in this interpreter or in a separate one.

        Requirement 5.3: isolated execution is reported for every operation it
        serves, and the adapter is the only thing that knows.
        """
        ...

    @property
    def npu_partition_share(self) -> float | None:
        """How much of the graph the NPU took, or ``None`` if not established.

        ``None`` is *unverified*, never "assume it was fine": design.md requires
        an explicit ``npu`` selection to distinguish "verified on NPU" from
        "assumed on NPU". A CPU adapter reports ``None`` because the question
        does not apply to it. Task 4.3 fills this in.
        """
        ...

    def run(
        self,
        token_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Token embeddings for one padded batch. No pooling, no normalization."""
        ...


class BackendFactory(Protocol):
    """Builds one adapter for a profile in a known environment.

    The capability report is passed rather than re-derived so that the NPU
    factory can choose between the in-process and isolated adapters from the
    same verdict the policy used, instead of probing the environment a second
    time and possibly disagreeing with it.

    The parameters are positional-only, so any two-argument callable satisfies
    this - a function, a bound method, a class's constructor.
    """

    def __call__(
        self, profile: ModelProfile, capability: CapabilityReport, /
    ) -> TransformerBackend: ...


@dataclass(frozen=True)
class BackendFactories:
    """How `resolve_backend` builds whichever adapter the policy chose.

    Required rather than defaulted. A default pair would have to import the
    concrete adapters, which import this module back, and - worse - a caller who
    forgot to supply them would silently get whatever the default was, which is
    the shape of every substitution requirement 2 forbids.
    """

    npu: BackendFactory
    cpu: BackendFactory


# --------------------------------------------------------------------------
# The binding (requirement 2.7)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundBackend:
    """One backend, bound to one operation, with why it is the one chosen.

    Requirement 2.7 - "while an embedding operation is in progress, the
    Embedding Runtime shall not change provider before the operation completes"
    - is a statement about the whole run, so it is made structural here rather
    than promised in a docstring: the binding is frozen, so the resolved adapter
    cannot be swapped out from under a batch that is halfway through. The
    service builds one of these per ``embed()`` call (design.md, Requirements
    Traceability: "Backend bound once per ``embed()`` call").

    It satisfies `TransformerBackend` itself, delegating each member, so a
    caller holds the binding rather than the adapter and cannot accidentally
    keep a reference that outlives the operation's decision.

    The invariants are requirement 2's "exactly when", made unconstructible to
    violate:

    - the serving provider is never ``auto`` - that is the request, not the
      answer (2.6);
    - an explicit ``npu`` or ``cpu`` request is served by that provider or not
      at all, and carries no fallback reason because nothing was substituted
      (2.2, 2.3);
    - under ``auto``, a reason is present if and only if the CPU served (2.4,
      2.5). A reason on an NPU run explains something that did not happen; a
      CPU run with no reason is the silent degradation requirement 2 exists to
      make impossible.
    """

    backend: TransformerBackend
    requested: ProviderChoice
    fallback_reason: str | None

    def __post_init__(self) -> None:
        # Normalizing both choices is not decoration. ``ProviderChoice`` is a
        # ``StrEnum``, so a raw ``"npu"`` compares equal to the member but is
        # not identical to it, and every branch below is written with ``is``.
        # An unconverted string would slip past each one in turn and land on
        # whatever the last case happens to be.
        object.__setattr__(self, "requested", ProviderChoice(self.requested))
        served = self.provider
        if served is ProviderChoice.AUTO:
            raise ValueError(
                "a backend reports the provider that actually ran; 'auto' is a "
                "request, not an outcome (requirement 2.6) - report npu or cpu"
            )
        if self.fallback_reason is not None and not self.fallback_reason.strip():
            raise ValueError(
                "fallback_reason must say specifically why the NPU was not "
                f"used, got {self.fallback_reason!r}: a blank reason reports a "
                "fallback and explains nothing (requirement 2.5)"
            )
        if self.requested is ProviderChoice.AUTO:
            self._check_auto(served)
            return
        if served is not self.requested:
            raise ValueError(
                f"provider {self.requested.value!r} was requested explicitly "
                f"and a {served.value!r} backend was produced; an explicit "
                "choice is honoured or it fails, never substituted "
                "(requirement 2.2)"
            )
        if self.fallback_reason is not None:
            raise ValueError(
                f"provider {self.requested.value!r} was requested explicitly "
                "and served, so nothing fell back; a fallback reason belongs "
                "to an 'auto' selection only (requirement 2.5)"
            )

    def _check_auto(self, served: ProviderChoice) -> None:
        """Under ``auto``: a reason exactly when the CPU served instead."""
        if served is ProviderChoice.CPU and self.fallback_reason is None:
            raise ValueError(
                "automatic selection used the CPU without saying why the NPU "
                "was not used; requirement 2.5 requires the specific reason to "
                "be reported before embedding on the CPU"
            )
        if served is ProviderChoice.NPU and self.fallback_reason is not None:
            raise ValueError(
                "automatic selection used the NPU, so nothing fell back, but a "
                f"fallback reason was recorded: {self.fallback_reason!r} "
                "(requirement 2.4)"
            )

    @property
    def provider(self) -> ProviderChoice:
        """The provider that will serve this operation (2.6)."""
        return ProviderChoice(self.backend.provider)

    @property
    def execution_mode(self) -> ExecutionMode:
        return self.backend.execution_mode

    @property
    def npu_partition_share(self) -> float | None:
        return self.backend.npu_partition_share

    @property
    def fell_back(self) -> bool:
        """Whether the CPU served because the NPU could not (2.5)."""
        return self.fallback_reason is not None

    def run(
        self,
        token_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Delegate to the bound adapter, unchanged in both directions."""
        return self.backend.run(token_ids, attention_mask)


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------


def npu_unavailable_reason(capability: CapabilityReport) -> str | None:
    """Why the NPU cannot serve, or ``None`` where it can.

    "Can" means either execution mode that reaches the hardware: ``IN_PROCESS``
    runs it in this interpreter, and ``ISOLATED`` runs it in a separate one,
    which requirement 5.1 provides precisely so that NPU embedding stays
    available when the application's own environment cannot host it. Only
    ``UNAVAILABLE`` means no route exists.

    The string is built from the report's *unsatisfied conditions*, each of
    which carries its observed value, its requirement and its remediation by
    construction (`Condition`'s own invariant, requirement 1.3). That is what
    makes requirement 2.5's "specific reason" specific: "the NPU is unavailable"
    would only repeat the verdict the caller can already see.

    It is never blank and never ``None`` for an unavailable NPU, including in
    the odd case of an ``UNAVAILABLE`` verdict with every condition satisfied -
    only ``IN_PROCESS`` is invariant-bound to a condition, so that report is
    constructible, and requirement 2.5 has no "sometimes".
    """
    if capability.execution_mode is not ExecutionMode.UNAVAILABLE:
        return None
    unmet = [c for c in capability.conditions if not c.satisfied]
    if not unmet:
        return (
            f"NPU execution reported as {ExecutionMode.UNAVAILABLE.value} with "
            "no unsatisfied condition to explain it; the capability report "
            "names no route to the hardware"
        )
    return " | ".join(_describe(condition) for condition in unmet)


def _describe(condition: Condition) -> str:
    """One unmet condition as a line an operator can act on (1.3, 8.1)."""
    detail = f"{condition.name}: {condition.observed}"
    extras = [
        f"{label}: {value}"
        for label, value in (
            ("required", condition.required),
            ("remediation", condition.remediation),
        )
        if value
    ]
    return f"{detail} ({'; '.join(extras)})" if extras else detail


def resolve_backend(
    choice: ProviderChoice,
    profile: ModelProfile,
    capability: CapabilityReport,
    *,
    factories: BackendFactories,
) -> tuple[TransformerBackend, str | None]:
    """Bind one backend to one operation, or fail (2.1, 2.2, 2.4, 2.5, 2.7).

    Returns the bound backend and the fallback reason, which is non-``None``
    only under ``auto`` and only when the CPU served instead of the NPU.

    - ``npu`` with an unusable NPU raises `NpuUnavailableError` naming the unmet
      condition, and builds nothing at all - not even the CPU adapter it must
      not return (2.2).
    - ``cpu`` is served by the CPU whatever the NPU is doing (2.3).
    - ``auto`` prefers the NPU where it is reachable (2.4) and otherwise reports
      the specific reason before falling back (2.5).

    The result is frozen, so the provider cannot change part-way through the
    operation it was resolved for (2.7).
    """
    # A ``StrEnum`` member is equal to its value but not identical to it, and
    # the branches below compare by identity. Converting first means a caller
    # handing over a plain ``"npu"`` - from a CLI flag, a config file, a JSON
    # frame - gets the NPU or a clear rejection, never a silent slide into the
    # ``auto`` branch. Requirement 2.1 accepts the three spellings; nothing else.
    choice = ProviderChoice(choice)
    reason = npu_unavailable_reason(capability)

    if choice is ProviderChoice.CPU:
        return _bind(factories.cpu, profile, capability, choice, None), None

    if choice is ProviderChoice.NPU:
        if reason is not None:
            raise NpuUnavailableError(
                f"provider 'npu' was requested explicitly and the NPU is not "
                f"usable: {reason}. No other provider will be substituted",
                provider=ProviderChoice.NPU,
                model_id=profile.model_id,
            )
        return _bind(factories.npu, profile, capability, choice, None), None

    if reason is None:
        return _bind(factories.npu, profile, capability, choice, None), None
    return _bind(factories.cpu, profile, capability, choice, reason), reason


def _bind(
    factory: BackendFactory,
    profile: ModelProfile,
    capability: CapabilityReport,
    requested: ProviderChoice,
    fallback_reason: str | None,
) -> BoundBackend:
    """Build the adapter and freeze the decision around it.

    The check on the way out is the last place a substitution could enter. The
    environment said the NPU was usable and resolution chose it; if what came
    back runs somewhere else, the caller who asked for ``npu`` explicitly must
    hear about it as an NPU failure rather than receive CPU vectors - the
    outcome requirement 2.2 exists to make impossible, reached from inside the
    process instead of from the environment.
    """
    backend = factory(profile, capability)
    served = ProviderChoice(backend.provider)
    if requested is ProviderChoice.NPU and served is not ProviderChoice.NPU:
        raise NpuUnavailableError(
            f"provider 'npu' was requested explicitly and resolution produced a "
            f"{served.value!r} backend; an explicit NPU request is never served "
            "by another provider",
            provider=ProviderChoice.NPU,
            model_id=profile.model_id,
        )
    return BoundBackend(
        backend=backend, requested=requested, fallback_reason=fallback_reason
    )
