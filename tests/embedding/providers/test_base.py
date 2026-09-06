"""The backend protocol and the provider-resolution policy (task 4.1).

Every test here runs on a machine with **no NPU**. The capability report is
fabricated rather than measured, because the policy under test is precisely the
mapping from an environment verdict to a provider decision, and a test that
called ``check_capability()`` would only ever exercise whichever branch this
particular machine happens to be in. The one measured assertion - that this
machine's real report resolves ``AUTO`` to the NPU - lives in
``test_base_live.py`` and skips where the hardware is absent.

The Observable this file has to pin has two limbs, and both directions of each
matter:

- explicit ``npu`` on a machine without a usable NPU **raises**, and produces no
  backend at all - not a CPU one, not a lying NPU one;
- automatic selection returns a non-empty reason **exactly when** it falls back:
  a reason present on a non-fallback run is as wrong as a missing one on a
  fallback.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from npu_rag.embedding.errors import NpuUnavailableError
from npu_rag.embedding.profiles import ModelProfile, initial_default_profile
from npu_rag.embedding.providers.base import (
    BackendFactories,
    BoundBackend,
    TransformerBackend,
    npu_unavailable_reason,
    resolve_backend,
)
from npu_rag.embedding.types import (
    CONDITION_PROVIDER_REGISTERED,
    CapabilityReport,
    Condition,
    ExecutionMode,
    ProviderChoice,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "npu_rag"
    / "embedding"
    / "providers"
    / "base.py"
)
PACKAGE_ROOT = MODULE_PATH.parents[1]
MODULE_PACKAGE = "npu_rag.embedding"


# --------------------------------------------------------------------------
# Fabricated environments and fake backends
# --------------------------------------------------------------------------


def condition(name: str, *, satisfied: bool) -> Condition:
    """One condition. An unsatisfied one carries its three required fields."""
    if satisfied:
        return Condition(
            name=name,
            satisfied=True,
            observed="present",
            required="present",
            remediation=None,
        )
    return Condition(
        name=name,
        satisfied=False,
        observed=f"{name} is absent",
        required=f"{name} must be present",
        remediation=f"run the provisioner to fix {name}",
    )


def capability(
    mode: ExecutionMode,
    *,
    unsatisfied: tuple[str, ...] = (),
    conditions: tuple[Condition, ...] | None = None,
) -> CapabilityReport:
    """A fabricated report. Nothing here touches this machine's hardware."""
    if conditions is None:
        names = ("npu_device_present", CONDITION_PROVIDER_REGISTERED)
        conditions = tuple(
            condition(name, satisfied=name not in unsatisfied) for name in names
        )
    return CapabilityReport(
        conditions=conditions,
        execution_mode=mode,
        driver_version="32.0.20102.3930",
        runtime_version="1.23.2",
        device_name="NPU Strix",
        power_reporting_supported=True,
    )


IN_PROCESS = capability(ExecutionMode.IN_PROCESS)
ISOLATED = capability(
    ExecutionMode.ISOLATED, unsatisfied=(CONDITION_PROVIDER_REGISTERED,)
)
UNAVAILABLE = capability(
    ExecutionMode.UNAVAILABLE,
    unsatisfied=("npu_device_present", CONDITION_PROVIDER_REGISTERED),
)

EVERY_MODE = (ExecutionMode.IN_PROCESS, ExecutionMode.ISOLATED, ExecutionMode.UNAVAILABLE)
REPORTS = {
    ExecutionMode.IN_PROCESS: IN_PROCESS,
    ExecutionMode.ISOLATED: ISOLATED,
    ExecutionMode.UNAVAILABLE: UNAVAILABLE,
}


class FakeBackend:
    """A backend that satisfies the protocol without any ONNX session."""

    def __init__(
        self,
        provider: ProviderChoice,
        *,
        execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS,
        npu_partition_share: float | None = None,
        hidden: int = 3,
    ) -> None:
        self._provider = provider
        self._execution_mode = execution_mode
        self._share = npu_partition_share
        self._hidden = hidden
        self.calls: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = []

    @property
    def provider(self) -> ProviderChoice:
        return self._provider

    @property
    def execution_mode(self) -> ExecutionMode:
        return self._execution_mode

    @property
    def npu_partition_share(self) -> float | None:
        return self._share

    def run(
        self,
        token_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        self.calls.append((token_ids, attention_mask))
        batch, seq = token_ids.shape
        return np.full((batch, seq, self._hidden), 0.5, dtype=np.float32)


class RecordingFactory:
    """A backend factory that records what it was asked to build."""

    def __init__(self, backend: TransformerBackend) -> None:
        self._backend = backend
        self.calls: list[tuple[ModelProfile, CapabilityReport]] = []

    def __call__(
        self, profile: ModelProfile, report: CapabilityReport, /
    ) -> TransformerBackend:
        self.calls.append((profile, report))
        return self._backend


def factories(
    *,
    npu: TransformerBackend | None = None,
    cpu: TransformerBackend | None = None,
) -> tuple[BackendFactories, RecordingFactory, RecordingFactory]:
    npu_factory = RecordingFactory(npu or FakeBackend(ProviderChoice.NPU))
    cpu_factory = RecordingFactory(cpu or FakeBackend(ProviderChoice.CPU))
    return BackendFactories(npu=npu_factory, cpu=cpu_factory), npu_factory, cpu_factory


PROFILE = initial_default_profile()


# --------------------------------------------------------------------------
# The protocol: token embeddings and nothing more (design.md, TransformerBackend)
# --------------------------------------------------------------------------


def test_a_backend_returning_token_embeddings_satisfies_the_protocol() -> None:
    assert isinstance(FakeBackend(ProviderChoice.CPU), TransformerBackend)


def test_an_object_without_run_does_not_satisfy_the_protocol() -> None:
    class NotABackend:
        provider = ProviderChoice.CPU
        execution_mode = ExecutionMode.IN_PROCESS
        npu_partition_share = None

    assert not isinstance(NotABackend(), TransformerBackend)


def test_the_protocol_declares_no_pooling_or_normalization_member() -> None:
    """design.md: the backend "never pools, never normalizes - that belongs to
    the service so it cannot differ per backend".

    The absence of a fifth member is the assertion. A protocol that also offered
    ``pool`` or ``normalize`` would let each adapter carry its own, which is the
    per-backend divergence design.md's port placement exists to rule out.
    """
    body = next(
        node
        for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.ClassDef) and node.name == "TransformerBackend"
    )
    members = {
        node.name
        for node in body.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    assert members == {"provider", "execution_mode", "npu_partition_share", "run"}


def test_run_returns_token_embeddings_shaped_by_the_inputs() -> None:
    """One vector per *token*, not one per input: pooling has not happened."""
    backend = FakeBackend(ProviderChoice.CPU, hidden=7)
    token_ids = np.ones((2, 5), dtype=np.int64)
    mask = np.ones((2, 5), dtype=np.int64)

    embeddings = backend.run(token_ids, mask)

    assert embeddings.shape == (2, 5, 7)


# --------------------------------------------------------------------------
# Requirement 2.1: npu, cpu and auto are all accepted
# --------------------------------------------------------------------------


@pytest.mark.parametrize("choice", list(ProviderChoice))
def test_every_declared_choice_is_accepted(choice: ProviderChoice) -> None:
    bundle, _, _ = factories()

    backend, _reason = resolve_backend(
        choice, PROFILE, IN_PROCESS, factories=bundle
    )

    assert backend.provider in {ProviderChoice.NPU, ProviderChoice.CPU}


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("npu", ProviderChoice.NPU),
        ("cpu", ProviderChoice.CPU),
        ("auto", ProviderChoice.NPU),
    ],
)
def test_the_plain_string_spelling_is_accepted_too(
    spelling: str, expected: ProviderChoice
) -> None:
    """A CLI or a JSON payload hands over the value, not the member.

    Identity comparison against a ``StrEnum`` member is ``False`` for an equal
    string, so an unconverted ``"npu"`` would silently fall through to the
    ``auto`` branch - a *substitution*, arrived at by a type slip.
    """
    bundle, _, _ = factories()

    backend, _reason = resolve_backend(
        spelling,  # type: ignore[arg-type]
        PROFILE,
        IN_PROCESS,
        factories=bundle,
    )

    assert backend.provider is expected


def test_the_string_npu_still_fails_rather_than_falling_back() -> None:
    """The sharp edge of the spelling above, and the reason it is not cosmetic.

    An unconverted ``"npu"`` reaches the ``auto`` branch, and on a machine where
    the NPU happens to work that branch picks the NPU too - so the bug is
    invisible exactly until the NPU is missing, at which point the caller who
    demanded it silently receives CPU vectors.
    """
    bundle, _, cpu_factory = factories()

    with pytest.raises(NpuUnavailableError):
        resolve_backend(
            "npu",  # type: ignore[arg-type]
            PROFILE,
            UNAVAILABLE,
            factories=bundle,
        )

    assert cpu_factory.calls == []


def test_an_undeclared_choice_is_rejected_rather_than_defaulted() -> None:
    bundle, _, _ = factories()

    with pytest.raises(ValueError):
        resolve_backend(
            "gpu",  # type: ignore[arg-type]
            PROFILE,
            IN_PROCESS,
            factories=bundle,
        )


# --------------------------------------------------------------------------
# Requirement 2.2: explicit npu fails rather than substituting
# --------------------------------------------------------------------------


def test_explicit_npu_on_an_unusable_machine_raises() -> None:
    """The task's Observable, first limb."""
    bundle, _, _ = factories()

    with pytest.raises(NpuUnavailableError):
        resolve_backend(ProviderChoice.NPU, PROFILE, UNAVAILABLE, factories=bundle)


def test_explicit_npu_on_an_unusable_machine_returns_no_cpu_backend() -> None:
    """Raising is only half of it: no CPU backend may be built either.

    A resolution that constructed the CPU adapter and *then* raised would leave
    a loaded session behind and, more importantly, would mean the substitute
    exists and only an exception stands between it and the caller.
    """
    bundle, npu_factory, cpu_factory = factories()

    with pytest.raises(NpuUnavailableError):
        resolve_backend(ProviderChoice.NPU, PROFILE, UNAVAILABLE, factories=bundle)

    assert cpu_factory.calls == []
    assert npu_factory.calls == []


def test_the_npu_failure_names_the_unmet_condition() -> None:
    """Requirement 2.2: "fail with a diagnostic naming the unmet condition"."""
    bundle, _, _ = factories()

    with pytest.raises(NpuUnavailableError) as caught:
        resolve_backend(ProviderChoice.NPU, PROFILE, UNAVAILABLE, factories=bundle)

    message = str(caught.value)
    assert CONDITION_PROVIDER_REGISTERED in message
    assert "npu_device_present" in message
    assert "run the provisioner to fix npu_device_present" in message


def test_the_npu_failure_carries_provider_model_and_stage() -> None:
    """Requirement 8.1, through the taxonomy's own fields."""
    bundle, _, _ = factories()

    with pytest.raises(NpuUnavailableError) as caught:
        resolve_backend(ProviderChoice.NPU, PROFILE, UNAVAILABLE, factories=bundle)

    error = caught.value
    assert error.provider is ProviderChoice.NPU
    assert error.model_id == PROFILE.model_id
    assert error.stage == "provider_resolution"


@pytest.mark.parametrize("mode", [ExecutionMode.IN_PROCESS, ExecutionMode.ISOLATED])
def test_explicit_npu_succeeds_wherever_the_npu_is_reachable(
    mode: ExecutionMode,
) -> None:
    """Isolated execution is still the NPU serving (requirement 5.1)."""
    bundle, npu_factory, _ = factories()

    backend, reason = resolve_backend(
        ProviderChoice.NPU, PROFILE, REPORTS[mode], factories=bundle
    )

    assert backend.provider is ProviderChoice.NPU
    assert reason is None
    assert len(npu_factory.calls) == 1


def test_a_factory_that_hands_back_the_wrong_provider_is_refused() -> None:
    """The last place a substitution could still creep in.

    Resolution decided the NPU, and something built a CPU adapter instead.
    Returning it would be the silent degradation requirement 2 exists to
    prevent, arrived at from inside rather than from the environment.
    """
    bundle, _, _ = factories(npu=FakeBackend(ProviderChoice.CPU))

    with pytest.raises(NpuUnavailableError, match="cpu"):
        resolve_backend(ProviderChoice.NPU, PROFILE, IN_PROCESS, factories=bundle)


# --------------------------------------------------------------------------
# Requirement 2.3 (adjacent): explicit cpu is served regardless
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", EVERY_MODE)
def test_explicit_cpu_is_served_whatever_the_npu_is_doing(
    mode: ExecutionMode,
) -> None:
    bundle, npu_factory, cpu_factory = factories()

    backend, reason = resolve_backend(
        ProviderChoice.CPU, PROFILE, REPORTS[mode], factories=bundle
    )

    assert backend.provider is ProviderChoice.CPU
    assert reason is None, "a forced CPU run did not fall back to anything"
    assert npu_factory.calls == []
    assert len(cpu_factory.calls) == 1


# --------------------------------------------------------------------------
# Requirements 2.4 and 2.5: auto prefers the NPU and says why when it does not
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [ExecutionMode.IN_PROCESS, ExecutionMode.ISOLATED])
def test_auto_prefers_the_npu_where_it_is_available(mode: ExecutionMode) -> None:
    """Requirement 2.4."""
    bundle, npu_factory, cpu_factory = factories()

    backend, reason = resolve_backend(
        ProviderChoice.AUTO, PROFILE, REPORTS[mode], factories=bundle
    )

    assert backend.provider is ProviderChoice.NPU
    assert reason is None
    assert len(npu_factory.calls) == 1
    assert cpu_factory.calls == []


def test_auto_falls_back_to_the_cpu_with_a_reason() -> None:
    """Requirement 2.5, and the Observable's second limb."""
    bundle, npu_factory, cpu_factory = factories()

    backend, reason = resolve_backend(
        ProviderChoice.AUTO, PROFILE, UNAVAILABLE, factories=bundle
    )

    assert backend.provider is ProviderChoice.CPU
    assert reason is not None
    assert reason.strip() != ""
    assert npu_factory.calls == []
    assert len(cpu_factory.calls) == 1


def test_the_fallback_reason_is_specific_rather_than_generic() -> None:
    """Requirement 2.5 asks for "the specific reason", which a bare
    "NPU unavailable" is not: it repeats the verdict instead of explaining it."""
    bundle, _, _ = factories()

    _backend, reason = resolve_backend(
        ProviderChoice.AUTO, PROFILE, UNAVAILABLE, factories=bundle
    )

    assert reason is not None
    assert CONDITION_PROVIDER_REGISTERED in reason
    assert "npu_device_present is absent" in reason
    assert "run the provisioner to fix npu_device_present" in reason


@pytest.mark.parametrize("choice", list(ProviderChoice))
@pytest.mark.parametrize("mode", EVERY_MODE)
def test_a_reason_is_returned_exactly_when_the_run_falls_back(
    choice: ProviderChoice, mode: ExecutionMode
) -> None:
    """Both directions of "exactly when", over every choice and environment.

    A reason on a run that did *not* fall back is as wrong as a missing one on
    a run that did, so this sweeps the whole grid rather than the happy corner.
    """
    bundle, _, _ = factories()
    falls_back = (
        choice is ProviderChoice.AUTO and mode is ExecutionMode.UNAVAILABLE
    )
    raises = choice is ProviderChoice.NPU and mode is ExecutionMode.UNAVAILABLE

    if raises:
        with pytest.raises(NpuUnavailableError):
            resolve_backend(choice, PROFILE, REPORTS[mode], factories=bundle)
        return

    backend, reason = resolve_backend(
        choice, PROFILE, REPORTS[mode], factories=bundle
    )

    if falls_back:
        assert reason is not None and reason.strip() != ""
        assert backend.provider is ProviderChoice.CPU
    else:
        assert reason is None


def test_the_profile_and_report_reach_the_factory_unchanged() -> None:
    bundle, npu_factory, _ = factories()

    resolve_backend(ProviderChoice.AUTO, PROFILE, IN_PROCESS, factories=bundle)

    assert npu_factory.calls == [(PROFILE, IN_PROCESS)]


# --------------------------------------------------------------------------
# npu_unavailable_reason: the policy's own explanation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [ExecutionMode.IN_PROCESS, ExecutionMode.ISOLATED])
def test_a_reachable_npu_has_no_unavailability_reason(mode: ExecutionMode) -> None:
    assert npu_unavailable_reason(REPORTS[mode]) is None


def test_an_unreachable_npu_reports_every_unmet_condition() -> None:
    reason = npu_unavailable_reason(UNAVAILABLE)

    assert reason is not None
    assert "npu_device_present" in reason
    assert CONDITION_PROVIDER_REGISTERED in reason


def test_the_reason_is_never_blank_even_with_no_unmet_condition() -> None:
    """An ``UNAVAILABLE`` verdict with every condition satisfied is odd, not
    impossible - only ``IN_PROCESS`` is invariant-bound to a condition. The
    reason must still say something, because requirement 2.5 has no
    "sometimes"."""
    report = capability(
        ExecutionMode.UNAVAILABLE,
        conditions=(condition("npu_device_present", satisfied=True),),
    )

    reason = npu_unavailable_reason(report)

    assert reason is not None
    assert reason.strip() != ""
    assert ExecutionMode.UNAVAILABLE.value in reason


def test_the_reason_is_never_blank_for_a_report_with_no_conditions() -> None:
    report = capability(ExecutionMode.UNAVAILABLE, conditions=())

    reason = npu_unavailable_reason(report)

    assert reason is not None
    assert reason.strip() != ""


# --------------------------------------------------------------------------
# Requirement 2.7: the resolved backend is bound for the whole operation
# --------------------------------------------------------------------------


def test_the_binding_is_frozen_so_the_provider_cannot_change_mid_run() -> None:
    """Requirement 2.7 made structural rather than promised in a docstring."""
    bundle, _, _ = factories()
    backend, _reason = resolve_backend(
        ProviderChoice.AUTO, PROFILE, IN_PROCESS, factories=bundle
    )

    assert isinstance(backend, BoundBackend)
    with pytest.raises(dataclasses.FrozenInstanceError):
        backend.backend = FakeBackend(ProviderChoice.CPU)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        backend.fallback_reason = "swapped"  # type: ignore[misc]


def test_the_binding_is_itself_a_backend_and_delegates_every_member() -> None:
    inner = FakeBackend(
        ProviderChoice.NPU,
        execution_mode=ExecutionMode.ISOLATED,
        npu_partition_share=0.97,
    )
    bundle, _, _ = factories(npu=inner)

    backend, _reason = resolve_backend(
        ProviderChoice.NPU, PROFILE, ISOLATED, factories=bundle
    )

    assert isinstance(backend, TransformerBackend)
    assert backend.provider is ProviderChoice.NPU
    assert backend.execution_mode is ExecutionMode.ISOLATED
    assert backend.npu_partition_share == 0.97


def test_the_binding_passes_the_arrays_through_untouched() -> None:
    inner = FakeBackend(ProviderChoice.CPU, hidden=4)
    bundle, _, _ = factories(cpu=inner)
    backend, _reason = resolve_backend(
        ProviderChoice.CPU, PROFILE, IN_PROCESS, factories=bundle
    )
    token_ids = np.arange(6, dtype=np.int64).reshape(2, 3)
    mask = np.ones((2, 3), dtype=np.int64)

    embeddings = backend.run(token_ids, mask)

    assert embeddings.shape == (2, 3, 4)
    assert len(inner.calls) == 1
    assert np.array_equal(inner.calls[0][0], token_ids)
    assert np.array_equal(inner.calls[0][1], mask)


def test_each_resolution_yields_its_own_binding() -> None:
    """Two operations get two bindings; neither can disturb the other."""
    bundle, _, _ = factories()

    first, _ = resolve_backend(
        ProviderChoice.AUTO, PROFILE, IN_PROCESS, factories=bundle
    )
    second, _ = resolve_backend(
        ProviderChoice.CPU, PROFILE, IN_PROCESS, factories=bundle
    )

    assert first is not second
    assert first.provider is ProviderChoice.NPU
    assert second.provider is ProviderChoice.CPU


# --------------------------------------------------------------------------
# BoundBackend's own invariants - the contract a future backend is held to
# --------------------------------------------------------------------------


def test_a_binding_never_reports_auto_as_the_serving_provider() -> None:
    """``AUTO`` is a request, never an outcome (types.py, requirement 2.6)."""
    with pytest.raises(ValueError, match="auto"):
        BoundBackend(
            backend=FakeBackend(ProviderChoice.AUTO),
            requested=ProviderChoice.AUTO,
            fallback_reason=None,
        )


@pytest.mark.parametrize(
    ("requested", "served"),
    [
        (ProviderChoice.NPU, ProviderChoice.CPU),
        (ProviderChoice.CPU, ProviderChoice.NPU),
    ],
)
def test_an_explicit_choice_must_be_honoured_by_the_binding(
    requested: ProviderChoice, served: ProviderChoice
) -> None:
    with pytest.raises(ValueError):
        BoundBackend(
            backend=FakeBackend(served),
            requested=requested,
            fallback_reason=None,
        )


@pytest.mark.parametrize("requested", [ProviderChoice.NPU, ProviderChoice.CPU])
def test_an_explicit_choice_never_carries_a_fallback_reason(
    requested: ProviderChoice,
) -> None:
    """Nothing was substituted, so there is nothing to explain (2.5)."""
    with pytest.raises(ValueError):
        BoundBackend(
            backend=FakeBackend(requested),
            requested=requested,
            fallback_reason="the NPU was busy",
        )


def test_auto_on_the_npu_may_not_carry_a_reason() -> None:
    with pytest.raises(ValueError):
        BoundBackend(
            backend=FakeBackend(ProviderChoice.NPU),
            requested=ProviderChoice.AUTO,
            fallback_reason="fell back",
        )


def test_auto_on_the_cpu_must_carry_a_reason() -> None:
    """The missing half of "exactly when", enforced at the binding."""
    with pytest.raises(ValueError):
        BoundBackend(
            backend=FakeBackend(ProviderChoice.CPU),
            requested=ProviderChoice.AUTO,
            fallback_reason=None,
        )


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_a_blank_fallback_reason_is_not_a_reason(blank: str) -> None:
    """An empty string is falsy in one place and truthy-shaped in another; it
    would render as a fallback with no explanation."""
    with pytest.raises(ValueError):
        BoundBackend(
            backend=FakeBackend(ProviderChoice.CPU),
            requested=ProviderChoice.AUTO,
            fallback_reason=blank,
        )


def test_a_well_formed_fallback_binding_is_accepted() -> None:
    bound = BoundBackend(
        backend=FakeBackend(ProviderChoice.CPU),
        requested=ProviderChoice.AUTO,
        fallback_reason="the execution provider did not register",
    )

    assert bound.provider is ProviderChoice.CPU
    assert bound.fell_back is True


def test_a_non_fallback_binding_reports_that_it_did_not_fall_back() -> None:
    bound = BoundBackend(
        backend=FakeBackend(ProviderChoice.NPU),
        requested=ProviderChoice.AUTO,
        fallback_reason=None,
    )

    assert bound.fell_back is False


# --------------------------------------------------------------------------
# This module is the protocol and the policy - not a session, not a pipeline
# --------------------------------------------------------------------------


def source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def imported_names(source_text: str, package: str) -> list[str]:
    """Every imported name as an absolute dotted path, relatives resolved."""
    parts = package.split(".")
    names: list[str] = []
    for node in ast.walk(ast.parse(source_text)):
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


def test_the_policy_module_constructs_no_inference_session() -> None:
    """Guard two - ``session.get_providers()[0]`` after construction - belongs
    to the concrete NPU adapter (task 4.3). This module decides *which* backend
    to build, and building one is somebody else's job."""
    text = source()

    assert [
        name
        for name in imported_names(text, MODULE_PACKAGE)
        if name == "onnxruntime" or name.startswith("onnxruntime.")
    ] == []

    called = {
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
    }
    assert [c for c in called if "InferenceSession" in c or "SessionOptions" in c] == []


FORBIDDEN_NUMERIC_CALLS = (
    "np.mean",
    "numpy.mean",
    "np.linalg",
    "numpy.linalg",
    "np.sum",
    "numpy.sum",
)


def test_the_policy_module_neither_pools_nor_normalizes() -> None:
    """design.md: post-processing "sits on the service side of the port, so
    pooling, the Dense stage, and normalization execute identically no matter
    which adapter ran the transformer" (task 5.2 owns it)."""
    text = source()
    tree = ast.parse(text)

    called = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert [c for c in called if c.startswith(FORBIDDEN_NUMERIC_CALLS)] == []

    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    offenders = sorted(
        name
        for name in defined
        for banned in ("pool", "normal", "dense")
        if banned in name.lower()
    )
    assert offenders == []


def test_the_policy_module_neither_prints_nor_logs() -> None:
    text = source()

    forbidden_imports = ("logging", "sys", "warnings")
    assert [
        name
        for name in imported_names(text, MODULE_PACKAGE)
        for banned in forbidden_imports
        if name == banned or name.startswith(f"{banned}.")
    ] == []

    called = {
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
    }
    assert [
        c
        for c in called
        if c == "print" or c.startswith(("logging.", "warnings.", "sys.std"))
    ] == []


def test_the_policy_module_does_not_redefine_the_shared_vocabulary() -> None:
    defined = {
        node.name
        for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.ClassDef | ast.FunctionDef)
    }

    assert defined.isdisjoint(
        {"ProviderChoice", "ExecutionMode", "CapabilityReport", "Condition"}
    )


# --------------------------------------------------------------------------
# Dependency direction (design.md, Architecture)
#
# Implementation Notes 1.1 and 1.4 both defer a package-wide layer guard to
# "when providers/base.py lands". It lands here.
# --------------------------------------------------------------------------

#: "types, errors -> reporting -> profiles -> environment -> models ->
#: providers -> service -> bench". Index is the layer's position; a module in
#: one layer may import only from strictly lower ones.
LAYER_ORDER: tuple[tuple[str, ...], ...] = (
    ("types", "errors"),
    ("reporting",),
    ("profiles",),
    ("environment",),
    ("models",),
    # `tokenize` shares a rank with `providers`: both sit above `models` and
    # below `service`, and neither may import the other. Added 2026-09-06 when
    # task 5.1's review found `tokenize.py` was invisible to this guard.
    ("providers", "tokenize"),
    ("service",),
    ("bench",),
)

LAYER_OF = {
    name: rank for rank, names in enumerate(LAYER_ORDER) for name in names
}


def test_the_providers_layer_imports_nothing_from_a_later_one() -> None:
    later = ("service", "bench")

    forbidden = sorted(
        {
            name
            for name in imported_names(source(), MODULE_PACKAGE)
            for layer in later
            if name == f"{MODULE_PACKAGE}.{layer}"
            or name.startswith(f"{MODULE_PACKAGE}.{layer}.")
        }
    )
    assert forbidden == []


def test_every_module_in_the_package_respects_the_layer_order() -> None:
    """The package-wide guard Implementation Notes 1.1 and 1.4 deferred to this
    task. The per-module guards each police one file; nothing until now checked
    that ``models`` never reaches into ``providers``, which design.md calls out
    by name as an error rather than a style issue."""
    violations: list[str] = []
    unplaced: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        own = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        if own == "__init__":
            continue
        if own not in LAYER_OF:
            # Do NOT skip. Until 2026-09-06 this was a silent `continue`, which
            # made a module absent from LAYER_ORDER invisible to this guard both
            # as importer and as target - so "package-wide" was a misnomer and
            # every new module was unpoliced until someone remembered the table.
            # Task 5.1's `tokenize.py` was the first to fall through it. Failing
            # loudly forces a deliberate placement decision for each new module.
            unplaced.append(str(relative))
            continue
        # Relative imports resolve against the *containing* package, which for
        # `providers/base.py` is `npu_rag.embedding.providers`, not the package
        # root - one dot fewer would silently mis-resolve every `from . import`.
        containing = ".".join((MODULE_PACKAGE, *relative.parts[:-1]))
        for name in imported_names(path.read_text(encoding="utf-8"), containing):
            if not name.startswith(f"{MODULE_PACKAGE}."):
                continue
            target = name[len(MODULE_PACKAGE) + 1 :].split(".")[0]
            rank = LAYER_OF.get(target)
            if rank is not None and rank > LAYER_OF[own]:
                violations.append(f"{relative}: imports {name}")
    assert unplaced == [], (
        f"these modules have no entry in LAYER_ORDER and are therefore invisible "
        f"to this guard, as importer and as target: {unplaced}. Add each at its "
        f"correct rank in the dependency direction rather than deleting this check."
    )
    assert violations == []


@pytest.mark.parametrize(
    "statement",
    [
        "import npu_rag.embedding.service",
        "from npu_rag.embedding.service import EmbeddingService",
        "from npu_rag.embedding import bench",
        "from .. import service",
        "from ..bench import harness",
        "from ..service import EmbeddingService",
    ],
)
def test_the_layer_guard_recognises_every_import_spelling(statement: str) -> None:
    """Note 1.1: the earlier guard "skips relative imports"; this one does not.

    The relative spellings are written as ``providers/base.py`` would have to
    write them: one dot is the ``providers`` package itself, so reaching
    ``service`` from there takes two.
    """
    resolved = imported_names(statement, f"{MODULE_PACKAGE}.providers")

    assert any(
        name.startswith(f"{MODULE_PACKAGE}.{layer}")
        for name in resolved
        for layer in ("service", "bench")
    ), f"{statement!r} resolved to {resolved}"


def test_the_package_wide_layer_guard_is_not_vacuous() -> None:
    """Note 1.1 warned that the earlier boundary guard "passes vacuously if the
    package is deleted". This one walks real files, so prove it sees them and
    that a planted inversion would be caught."""
    scanned = [
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if (
            path.relative_to(PACKAGE_ROOT).parts[0]
            if len(path.relative_to(PACKAGE_ROOT).parts) > 1
            else path.relative_to(PACKAGE_ROOT).stem
        )
        in LAYER_OF
    ]
    assert len(scanned) >= 8

    inverted = imported_names(
        "from npu_rag.embedding.providers import base", MODULE_PACKAGE
    )
    target = inverted[0][len(MODULE_PACKAGE) + 1 :].split(".")[0]
    assert LAYER_OF[target] > LAYER_OF["models"]


def test_resolve_backend_is_reachable_from_the_providers_package() -> None:
    module: Any = __import__(
        "npu_rag.embedding.providers.base", fromlist=["resolve_backend"]
    )

    assert module.resolve_backend is resolve_backend
