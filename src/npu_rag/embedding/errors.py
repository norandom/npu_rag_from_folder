"""The failure taxonomy (task 2.1), structured so 8.2 is a type question.

Requirement 8.2 requires environment failures, model-preparation failures, and
execution failures to be distinguishable from one another. design.md's Error
Strategy fixes *how*: "make the category structural so 8.2 is satisfied by the
type rather than by message text". So the three categories are three sibling
classes under one root, and a caller separates them with ``except`` rather than
by matching strings:

    EmbeddingRuntimeError
    |- EnvironmentError_          NPU absent, runtime missing, driver too old
    |  `- NpuUnavailableError     raised under an explicit `npu` selection
    |- PreparationError           acquisition, export, compile, verify
    |  |- LicenseAcceptanceRequired   gated repository, carries acceptance URL
    |  `- PartitionShareTooLow        graph largely - or unverifiably - on CPU
    `- ExecutionError             session run failures
       `- IsolatedWorkerError     worker absent, failed to start, died

Requirement 8.1 requires every failure to report the provider in use, the model
in use, and the stage at which it occurred. Here that is a property of the root
class, so it cannot be forgotten by a subclass:

- **provider** and **model_id** are ``None`` when genuinely unknown. They really
  can be: the capability check fails before any model is chosen, and provider
  resolution fails before any provider is bound. ``None`` is the single
  representation of that absence - a blank string is normalized to it rather
  than kept, so a reader never has to tell "" apart from ``None``.
- **stage** is never absent. Whoever raises always knows what it was doing, and
  where it does not say, its type does: every class below declares the stage its
  failures belong to. This is the asymmetry to remember - two of the three facts
  are optional, the third is not.

Nothing here validates its way into raising from a constructor. An error class
that can fail to be built would replace a diagnosed failure with a confusing
one, at exactly the moment the diagnosis matters most.

This module sits in the leftmost layer of design.md's dependency direction
(``types, errors -> reporting -> profiles -> environment -> models -> providers
-> service -> bench``); it imports the provider vocabulary from ``types`` and
nothing else from this project.
"""

from __future__ import annotations

from typing import ClassVar

from npu_rag.embedding.types import ProviderChoice

__all__ = [
    "EmbeddingRuntimeError",
    "EnvironmentError_",
    "ExecutionError",
    "IsolatedWorkerError",
    "LicenseAcceptanceRequired",
    "NpuUnavailableError",
    "PartitionShareTooLow",
    "PreparationError",
]


class EmbeddingRuntimeError(Exception):
    """Root of the taxonomy: carries provider, model, and stage (8.1).

    Catching this catches every failure this feature raises, which is what a
    caller that does not care about the category should do. A caller that *does*
    care catches one of the three sibling categories below instead - never by
    inspecting the message, which is written for a human and may change.
    """

    #: The stage reported when the raiser does not name one. The root's is
    #: deliberately vague because the root is the catch-all; every subclass
    #: replaces it with the point in the flow where its failures happen.
    default_stage: ClassVar[str] = "unspecified"

    def __init__(
        self,
        message: str,
        *,
        provider: ProviderChoice | None = None,
        model_id: str | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider: ProviderChoice | None = provider
        self.model_id: str | None = _cleaned(model_id)
        self.stage: str = _cleaned(stage) or self.default_stage

    def __str__(self) -> str:
        """The message with the three required facts appended.

        Requirement 8.1 is a reporting requirement, and the report an operator
        actually reads is the one in the traceback. The attributes carry the
        same facts for a caller that wants to act on them, and the *type* - not
        this text - remains the discriminator for 8.2.
        """
        return (
            f"{self.message} [provider={self.provider or 'unknown'}, "
            f"model={self.model_id or 'unknown'}, stage={self.stage}]"
        )


class EnvironmentError_(EmbeddingRuntimeError):
    """The machine or this interpreter cannot do NPU work at all.

    NPU absent, vendor runtime missing or hollow, driver inadequate. design.md's
    Error Categories: report observed versus required with remediation, and
    never attempt repair (1.3, 1.4) - the capability check reports environment
    state as *data*, and this exception is for the moment a caller demanded
    something that state cannot support.

    The trailing underscore keeps this out of the way of ``builtins.
    EnvironmentError``, which is an alias of ``OSError``; the name is design.md's.
    """

    default_stage: ClassVar[str] = "environment"


class NpuUnavailableError(EnvironmentError_):
    """Requirement 2.2: ``npu`` was named explicitly and the NPU is unusable.

    Raised by provider resolution *before* any preparation happens, so the
    caller gets the unmet condition rather than a late failure inside a session.
    Nothing may substitute another provider after this - that silent degradation
    is the exact failure requirement 2 exists to prevent.
    """

    default_stage: ClassVar[str] = "provider_resolution"


class PreparationError(EmbeddingRuntimeError):
    """A model could not be made ready for the requested provider.

    Requirement 4.6: name the failing stage - acquisition, export, compile, or
    verify - and never substitute a different model or provider.
    """

    default_stage: ClassVar[str] = "preparation"


class LicenseAcceptanceRequired(PreparationError):
    """Requirement 4.5: the model is gated behind terms not yet accepted.

    Raised specifically rather than surfacing as a generic download or
    authorization failure, because the remedy is a licence acceptance the
    operator performs once, not a retry. ``acceptance_url`` has no default: this
    error cannot be raised without saying where to go.
    """

    default_stage: ClassVar[str] = "acquisition"

    def __init__(
        self,
        message: str,
        *,
        acceptance_url: str,
        provider: ProviderChoice | None = None,
        model_id: str | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__(
            message, provider=provider, model_id=model_id, stage=stage
        )
        self.acceptance_url = acceptance_url

    def __str__(self) -> str:
        return f"{super().__str__()} Accept the terms at {self.acceptance_url}"


class PartitionShareTooLow(PreparationError):
    """The prepared graph is not actually running on the NPU.

    Two different facts share this type, and design.md (TransformerBackend,
    Risks) requires both to fail under an explicit ``npu`` choice while staying
    distinguishable in the report:

    - ``observed_share`` is a number: the compiler assigned that fraction of the
      graph to the NPU and it is below ``minimum_share``. Measured zero is a
      measurement, not an absence.
    - ``observed_share`` is ``None``: the partition diagnostics could not be
      read at all, so nothing was verified. Proceeding on the strength of an
      unread file would reintroduce the silent degradation of 2.2.
    """

    default_stage: ClassVar[str] = "verify"

    def __init__(
        self,
        message: str,
        *,
        observed_share: float | None = None,
        minimum_share: float | None = None,
        provider: ProviderChoice | None = None,
        model_id: str | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__(
            message, provider=provider, model_id=model_id, stage=stage
        )
        self.observed_share = observed_share
        self.minimum_share = minimum_share

    @property
    def unverifiable(self) -> bool:
        """Whether the share was never established, as opposed to measured low."""
        return self.observed_share is None


class ExecutionError(EmbeddingRuntimeError):
    """A prepared model failed while running.

    design.md's Error Categories: report the completed count on interruption
    (8.4), and never let a failed run silently promote a different provider
    (5.5).
    """

    default_stage: ClassVar[str] = "execution"


class IsolatedWorkerError(ExecutionError):
    """The separate-environment worker is absent, would not start, or died.

    Requirement 5.5: this is reported, and execution does not continue under
    another provider unless that provider was explicitly selected.
    """

    default_stage: ClassVar[str] = "isolated_worker"


def _cleaned(value: str | None) -> str | None:
    """A non-blank string, or ``None``. Absence gets one representation."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
