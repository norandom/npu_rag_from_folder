"""Provider resolution against this machine's real capability report (4.1).

The unit tests fabricate every environment, which is the only way to reach all
three of them from one machine. What they cannot show is that the policy reads a
*measured* report correctly - that the field it consults is the one the checker
actually fills, and that this hardware therefore resolves ``AUTO`` to the NPU
with no fallback reason at all.

No backend is constructed here: the concrete adapters are tasks 4.2 and 4.3, so
the factories hand back inert stand-ins. What is live is the `CapabilityReport`.

Everything skips wholesale where the execution provider is not registered in
this interpreter - the state of any machine without the vendor runtime, and also
the state this repository falls into after a bare ``uv sync`` (repair with
``uv run python -m tools.provision_npu``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from npu_rag.embedding.environment.capability import (
    VITISAI_PROVIDER,
    check_capability,
    probe_onnx_runtime,
)
from npu_rag.embedding.profiles import ModelProfile, initial_default_profile
from npu_rag.embedding.providers.base import (
    BackendFactories,
    TransformerBackend,
    npu_unavailable_reason,
    resolve_backend,
)
from npu_rag.embedding.types import (
    CapabilityReport,
    ExecutionMode,
    ProviderChoice,
)

_probe = probe_onnx_runtime()
_providers = _probe.available_providers or ()

provider_registered = pytest.mark.skipif(
    VITISAI_PROVIDER not in _providers,
    reason=(
        f"{VITISAI_PROVIDER} is not registered in this interpreter "
        f"(available: {list(_providers)})"
    ),
)


class InertBackend:
    """A stand-in for the adapters tasks 4.2 and 4.3 will build."""

    def __init__(self, provider: ProviderChoice, mode: ExecutionMode) -> None:
        self._provider = provider
        self._mode = mode

    @property
    def provider(self) -> ProviderChoice:
        return self._provider

    @property
    def execution_mode(self) -> ExecutionMode:
        return self._mode

    @property
    def npu_partition_share(self) -> float | None:
        return None

    def run(
        self,
        token_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        raise AssertionError("the live policy test never executes a graph")


def _factories() -> BackendFactories:
    def npu(
        profile: ModelProfile, capability: CapabilityReport, /
    ) -> TransformerBackend:
        return InertBackend(ProviderChoice.NPU, capability.execution_mode)

    def cpu(
        profile: ModelProfile, capability: CapabilityReport, /
    ) -> TransformerBackend:
        return InertBackend(ProviderChoice.CPU, ExecutionMode.IN_PROCESS)

    return BackendFactories(npu=npu, cpu=cpu)


@provider_registered
def test_this_machine_resolves_auto_to_the_npu_with_no_reason() -> None:
    """Requirement 2.4 against the environment task 1.3 proved end to end."""
    report = check_capability()

    backend, reason = resolve_backend(
        ProviderChoice.AUTO,
        initial_default_profile(),
        report,
        factories=_factories(),
    )

    assert backend.provider is ProviderChoice.NPU
    assert reason is None, f"this machine reported a fallback reason: {reason}"


@provider_registered
def test_this_machine_reports_no_npu_unavailability_reason() -> None:
    assert npu_unavailable_reason(check_capability()) is None


@provider_registered
def test_explicit_npu_resolves_rather_than_raising_here() -> None:
    """Requirement 2.2's other side: where the NPU *is* usable, asking for it
    must not fail. A policy that raised unconditionally would pass every
    unavailable-machine test in the unit file."""
    report = check_capability()

    backend, reason = resolve_backend(
        ProviderChoice.NPU,
        initial_default_profile(),
        report,
        factories=_factories(),
    )

    assert backend.provider is ProviderChoice.NPU
    assert backend.execution_mode is ExecutionMode.IN_PROCESS
    assert reason is None
