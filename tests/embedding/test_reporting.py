"""Progress reporting and run summaries (task 2.3).

Covers requirements 2.6, 5.6, 8.3, 8.4 and 8.6, plus the two design.md
commitments that shape the module: "Progress is a callback, not logging, so
callers choose presentation" (Error Handling -> Monitoring) and the dependency
direction ``types, errors -> reporting -> profiles -> ...`` (Architecture).
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from npu_rag.embedding.errors import IsolatedWorkerError
from npu_rag.embedding.reporting import (
    ProgressCallback,
    ProgressUpdate,
    RunSummary,
    RunTracker,
    SummaryCallback,
)
from npu_rag.embedding.types import ExecutionMode, ProviderChoice

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "npu_rag"
    / "embedding"
    / "reporting.py"
)

#: Exactly the types ``json.dumps`` writes without a custom encoder. Checked by
#: identity rather than ``isinstance`` on purpose: ``StrEnum`` members and other
#: ``str`` subclasses pass an ``isinstance`` check while still being live
#: objects, and the isolated backend's framing carries plain JSON only
#: (design.md, "Isolated adapter framing").
JSON_SCALARS = (str, int, float, bool, type(None))


class Clock:
    """A deterministic ``perf_counter`` substitute."""

    def __init__(self, *readings: float) -> None:
        self._readings = list(readings)
        self._last = self._readings[0] if self._readings else 0.0

    def __call__(self) -> float:
        if self._readings:
            self._last = self._readings.pop(0)
        return self._last


class Recorder:
    """A progress callback shaped as a class, to prove the protocol is
    structural rather than a function-only convention."""

    def __init__(self) -> None:
        self.updates: list[ProgressUpdate] = []

    def __call__(self, update: ProgressUpdate) -> None:
        self.updates.append(update)


def make_tracker(
    *,
    input_count: int = 3,
    operation: str = "embed_documents",
    provider_served: ProviderChoice = ProviderChoice.NPU,
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS,
    progress: ProgressCallback | None = None,
    on_finish: SummaryCallback | None = None,
    clock: Callable[[], float] | None = None,
) -> RunTracker:
    return RunTracker(
        operation=operation,
        provider_served=provider_served,
        execution_mode=execution_mode,
        input_count=input_count,
        progress=progress,
        on_finish=on_finish,
        clock=clock if clock is not None else Clock(0.0),
    )


# --------------------------------------------------------------------------
# ProgressUpdate: the payload requirement 8.3 describes
# --------------------------------------------------------------------------


def test_update_carries_completed_and_remaining() -> None:
    """8.3: progress reports inputs completed and inputs remaining."""
    update = ProgressUpdate(operation="embed_documents", completed=2, total=7)

    assert update.completed == 2
    assert update.remaining == 5
    assert update.total == 7


def test_remaining_is_derived_so_it_cannot_disagree_with_completed() -> None:
    """Storing ``remaining`` alongside ``completed`` would let a caller be sent
    a payload whose two halves contradict each other."""
    update = ProgressUpdate(operation="embed_documents", completed=7, total=7)

    assert update.remaining == 0
    assert "remaining" not in type(update).__dataclass_fields__
    assert "remaining" not in update.as_mapping()


@pytest.mark.parametrize(
    ("completed", "total"),
    [(-1, 3), (0, -1), (4, 3)],
)
def test_update_rejects_impossible_counts(completed: int, total: int) -> None:
    with pytest.raises(ValueError):
        ProgressUpdate(
            operation="embed_documents", completed=completed, total=total
        )


@pytest.mark.parametrize("operation", ["", "   "])
def test_update_rejects_a_blank_operation_label(operation: str) -> None:
    with pytest.raises(ValueError):
        ProgressUpdate(operation=operation, completed=0, total=1)


def test_update_is_immutable() -> None:
    update = ProgressUpdate(operation="embed_documents", completed=0, total=1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        update.completed = 5  # type: ignore[misc]


# --------------------------------------------------------------------------
# Serialisability: the payload must cross a process boundary (5.6)
# --------------------------------------------------------------------------


def test_update_mapping_holds_only_json_scalars() -> None:
    """design.md, Isolated adapter framing: the wire carries JSON, no pickle.

    A payload field holding a live object - a tracker, a backend, a callback -
    would still render fine in-process and fail only under isolated execution.
    """
    mapping = ProgressUpdate(
        operation="embed_documents", completed=2, total=7
    ).as_mapping()

    offenders = {
        key: type(value)
        for key, value in mapping.items()
        if type(value) not in JSON_SCALARS
    }
    assert offenders == {}


def test_update_survives_a_json_round_trip_unchanged() -> None:
    update = ProgressUpdate(operation="embed_queries", completed=3, total=9)

    revived = ProgressUpdate.from_mapping(json.loads(json.dumps(update.as_mapping())))

    assert revived == update
    assert revived.remaining == update.remaining


#: A well-formed update payload, to be damaged one key at a time.
GOOD_UPDATE: dict[str, object] = {
    "operation": "embed_documents",
    "completed": 1,
    "total": 4,
}

#: A well-formed summary payload, likewise.
GOOD_SUMMARY: dict[str, object] = {
    "operation": "embed_documents",
    "provider_served": "npu",
    "execution_mode": "isolated",
    "input_count": 4,
    "completed_count": 2,
    "truncated_count": 1,
    "elapsed_seconds": 0.5,
    "interruption": "KeyboardInterrupt",
    "fallback_reason": None,
}


@pytest.mark.parametrize("missing", sorted(GOOD_UPDATE))
def test_update_from_mapping_names_every_missing_key(missing: str) -> None:
    """A truncated frame must fail as a named validation error.

    Reading a key straight out of the mapping would raise a bare ``KeyError``
    instead, which tells a caller debugging a process boundary nothing about
    which half of the payload arrived.
    """
    damaged = {key: value for key, value in GOOD_UPDATE.items() if key != missing}

    with pytest.raises(ValueError, match=missing):
        ProgressUpdate.from_mapping(damaged)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("operation", 7),
        ("operation", None),
        ("completed", "1"),
        ("completed", 1.5),
        ("completed", None),
        ("total", "4"),
        ("total", True),
    ],
)
def test_update_from_mapping_rejects_a_wrongly_typed_field(
    key: str, value: object
) -> None:
    with pytest.raises(ValueError, match=key):
        ProgressUpdate.from_mapping({**GOOD_UPDATE, key: value})


@pytest.mark.parametrize("missing", sorted(GOOD_SUMMARY))
def test_summary_from_mapping_names_every_missing_key(missing: str) -> None:
    damaged = {key: value for key, value in GOOD_SUMMARY.items() if key != missing}

    with pytest.raises(ValueError, match=missing):
        RunSummary.from_mapping(damaged)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("operation", 7),
        ("provider_served", 7),
        ("provider_served", None),
        ("execution_mode", 7),
        ("execution_mode", None),
        ("input_count", "4"),
        ("completed_count", "2"),
        ("truncated_count", 1.5),
        ("elapsed_seconds", "0.5"),
        ("elapsed_seconds", None),
        ("elapsed_seconds", True),
        ("interruption", 7),
        ("interruption", 1.5),
        ("fallback_reason", 7),
        ("fallback_reason", 1.5),
    ],
)
def test_summary_from_mapping_rejects_a_wrongly_typed_field(
    key: str, value: object
) -> None:
    with pytest.raises(ValueError, match=key):
        RunSummary.from_mapping({**GOOD_SUMMARY, key: value})


def test_summary_from_mapping_accepts_a_null_interruption() -> None:
    """``None`` is the success case, and must not be confused with absence."""
    revived = RunSummary.from_mapping(
        {**GOOD_SUMMARY, "completed_count": 4, "interruption": None}
    )

    assert revived.interrupted is False
    assert revived.completed_count == 4


def test_summary_from_mapping_accepts_an_integer_elapsed_time() -> None:
    """JSON writes ``0.0`` back as ``0``; that is a number, not a type error."""
    revived = RunSummary.from_mapping({**GOOD_SUMMARY, "elapsed_seconds": 0})

    assert revived.elapsed_seconds == 0.0


@pytest.mark.parametrize("key", ["provider_served", "execution_mode"])
def test_summary_from_mapping_rejects_an_unknown_enum_member(key: str) -> None:
    with pytest.raises(ValueError):
        RunSummary.from_mapping({**GOOD_SUMMARY, key: "teleporter"})


def test_summary_from_mapping_still_enforces_the_invariants() -> None:
    """Validation on the way in does not replace validation of the value."""
    with pytest.raises(ValueError, match="auto"):
        RunSummary.from_mapping({**GOOD_SUMMARY, "provider_served": "auto"})


def test_summary_mapping_holds_only_json_scalars() -> None:
    summary = RunSummary(
        operation="embed_documents",
        provider_served=ProviderChoice.CPU,
        execution_mode=ExecutionMode.IN_PROCESS,
        input_count=4,
        completed_count=4,
        truncated_count=1,
        elapsed_seconds=0.5,
        interruption=None,
    )

    mapping = summary.as_mapping()

    offenders = {
        key: type(value)
        for key, value in mapping.items()
        if type(value) not in JSON_SCALARS
    }
    assert offenders == {}
    assert RunSummary.from_mapping(json.loads(json.dumps(mapping))) == summary


# --------------------------------------------------------------------------
# fallback_reason: why the NPU was not used (2.5)
#
# Added by task 4.1, which owns requirements 2.4 and 2.5. design.md's
# Requirements Traceability maps them to ``RunSummary.fallback_reason``, and
# `resolve_backend` produces the string this field carries to the caller.
# --------------------------------------------------------------------------


def test_a_run_that_did_not_fall_back_carries_no_reason() -> None:
    """The default, and the common case: nothing was substituted."""
    summary = RunSummary(
        operation="embed_documents",
        provider_served=ProviderChoice.NPU,
        execution_mode=ExecutionMode.IN_PROCESS,
        input_count=2,
        completed_count=2,
        truncated_count=0,
        elapsed_seconds=0.25,
        interruption=None,
    )

    assert summary.fallback_reason is None


def test_a_fallback_run_reports_why_the_npu_was_not_used() -> None:
    summary = RunSummary(
        operation="embed_documents",
        provider_served=ProviderChoice.CPU,
        execution_mode=ExecutionMode.IN_PROCESS,
        input_count=2,
        completed_count=2,
        truncated_count=0,
        elapsed_seconds=0.25,
        interruption=None,
        fallback_reason="execution_provider_registered: not registered here",
    )

    assert summary.fallback_reason == (
        "execution_provider_registered: not registered here"
    )


def test_a_fallback_reason_on_an_npu_run_is_a_contradiction() -> None:
    """Falling back means the CPU served instead (2.5). A summary claiming the
    NPU served *and* explaining why it did not would report both answers."""
    with pytest.raises(ValueError, match="fallback_reason"):
        RunSummary(
            operation="embed_documents",
            provider_served=ProviderChoice.NPU,
            execution_mode=ExecutionMode.IN_PROCESS,
            input_count=2,
            completed_count=2,
            truncated_count=0,
            elapsed_seconds=0.25,
            interruption=None,
            fallback_reason="the NPU was not used",
        )


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_a_blank_fallback_reason_is_rejected(blank: str) -> None:
    with pytest.raises(ValueError, match="fallback_reason"):
        RunSummary(
            operation="embed_documents",
            provider_served=ProviderChoice.CPU,
            execution_mode=ExecutionMode.IN_PROCESS,
            input_count=2,
            completed_count=2,
            truncated_count=0,
            elapsed_seconds=0.25,
            interruption=None,
            fallback_reason=blank,
        )


def test_the_fallback_reason_survives_the_json_round_trip() -> None:
    """5.6's process boundary: the reason must reach an isolated caller too."""
    summary = RunSummary(
        operation="embed_documents",
        provider_served=ProviderChoice.CPU,
        execution_mode=ExecutionMode.ISOLATED,
        input_count=1,
        completed_count=1,
        truncated_count=0,
        elapsed_seconds=0.5,
        interruption=None,
        fallback_reason="npu_device_present: absent",
    )

    mapping = summary.as_mapping()

    assert mapping["fallback_reason"] == "npu_device_present: absent"
    offenders = {
        key: type(value)
        for key, value in mapping.items()
        if type(value) not in JSON_SCALARS
    }
    assert offenders == {}
    assert RunSummary.from_mapping(json.loads(json.dumps(mapping))) == summary


def test_from_mapping_accepts_a_stated_fallback_reason() -> None:
    revived = RunSummary.from_mapping(
        {
            **GOOD_SUMMARY,
            "provider_served": "cpu",
            "fallback_reason": "vendor_runtime_installed: absent",
        }
    )

    assert revived.fallback_reason == "vendor_runtime_installed: absent"


# --------------------------------------------------------------------------
# RunSummary: what a completed run reports (8.6, 2.6)
# --------------------------------------------------------------------------


def test_summary_reports_elapsed_time_and_input_count() -> None:
    """8.6: successful completion reports elapsed time and input count."""
    summary = RunSummary(
        operation="embed_documents",
        provider_served=ProviderChoice.NPU,
        execution_mode=ExecutionMode.IN_PROCESS,
        input_count=12,
        completed_count=12,
        truncated_count=0,
        elapsed_seconds=1.25,
        interruption=None,
    )

    assert summary.elapsed_seconds == 1.25
    assert summary.input_count == 12
    assert summary.interrupted is False


def test_summary_is_immutable() -> None:
    """Held to the same contract as ``ProgressUpdate``: a summary that could be
    edited after the fact is a record of nothing."""
    summary = RunSummary(
        operation="embed_documents",
        provider_served=ProviderChoice.NPU,
        execution_mode=ExecutionMode.IN_PROCESS,
        input_count=2,
        completed_count=2,
        truncated_count=0,
        elapsed_seconds=0.5,
        interruption=None,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.completed_count = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.provider_served = ProviderChoice.CPU  # type: ignore[misc]


def test_summary_rejects_auto_as_the_provider_that_served() -> None:
    """2.6 reports which provider *actually* served; ``AUTO`` is a request."""
    with pytest.raises(ValueError, match="auto"):
        RunSummary(
            operation="embed_documents",
            provider_served=ProviderChoice.AUTO,
            execution_mode=ExecutionMode.IN_PROCESS,
            input_count=1,
            completed_count=1,
            truncated_count=0,
            elapsed_seconds=0.0,
            interruption=None,
        )


def test_summary_rejects_success_that_did_not_finish_every_input() -> None:
    with pytest.raises(ValueError):
        RunSummary(
            operation="embed_documents",
            provider_served=ProviderChoice.CPU,
            execution_mode=ExecutionMode.IN_PROCESS,
            input_count=5,
            completed_count=4,
            truncated_count=0,
            elapsed_seconds=0.0,
            interruption=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_count", -1),
        ("input_count", -1),
        ("truncated_count", -1),
        ("elapsed_seconds", -0.001),
    ],
)
def test_summary_rejects_negative_quantities(field: str, value: float) -> None:
    kwargs: dict[str, Any] = {
        "operation": "embed_documents",
        "provider_served": ProviderChoice.CPU,
        "execution_mode": ExecutionMode.IN_PROCESS,
        "input_count": 2,
        "completed_count": 2,
        "truncated_count": 0,
        "elapsed_seconds": 0.0,
        "interruption": None,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        RunSummary(**kwargs)


def test_summary_rejects_more_completed_than_inputs() -> None:
    with pytest.raises(ValueError):
        RunSummary(
            operation="embed_documents",
            provider_served=ProviderChoice.CPU,
            execution_mode=ExecutionMode.IN_PROCESS,
            input_count=2,
            completed_count=3,
            truncated_count=0,
            elapsed_seconds=0.0,
            interruption="KeyboardInterrupt",
        )


def test_summary_rejects_more_truncated_than_inputs() -> None:
    with pytest.raises(ValueError):
        RunSummary(
            operation="embed_documents",
            provider_served=ProviderChoice.CPU,
            execution_mode=ExecutionMode.IN_PROCESS,
            input_count=2,
            completed_count=2,
            truncated_count=3,
            elapsed_seconds=0.0,
            interruption=None,
        )


@pytest.mark.parametrize("interruption", ["", "   "])
def test_summary_rejects_a_blank_interruption_reason(interruption: str) -> None:
    """A blank reason is the ambiguous third state between the two outcomes."""
    with pytest.raises(ValueError):
        RunSummary(
            operation="embed_documents",
            provider_served=ProviderChoice.CPU,
            execution_mode=ExecutionMode.IN_PROCESS,
            input_count=2,
            completed_count=1,
            truncated_count=0,
            elapsed_seconds=0.0,
            interruption=interruption,
        )


def test_summary_rejects_a_non_finite_elapsed_time() -> None:
    with pytest.raises(ValueError):
        RunSummary(
            operation="embed_documents",
            provider_served=ProviderChoice.CPU,
            execution_mode=ExecutionMode.IN_PROCESS,
            input_count=1,
            completed_count=1,
            truncated_count=0,
            elapsed_seconds=math.inf,
            interruption=None,
        )


# --------------------------------------------------------------------------
# RunTracker: progress emission (8.3) and monotonicity
# --------------------------------------------------------------------------


def test_entering_announces_the_total_before_any_work() -> None:
    recorder = Recorder()

    with make_tracker(input_count=4, progress=recorder):
        pass

    assert recorder.updates[0] == ProgressUpdate(
        operation="embed_documents", completed=0, total=4
    )


def test_a_long_batch_emits_monotonically_increasing_progress() -> None:
    """The task's Observable, and 8.3."""
    total = 500
    recorder = Recorder()

    with make_tracker(input_count=total, progress=recorder) as tracker:
        for _ in range(total):
            tracker.advance()

    completed = [update.completed for update in recorder.updates]
    assert completed == list(range(total + 1))
    assert all(
        later > earlier for earlier, later in zip(completed, completed[1:])
    )
    assert [update.remaining for update in recorder.updates][-1] == 0


def test_progress_never_moves_backwards_even_across_batched_advances() -> None:
    recorder = Recorder()

    with make_tracker(input_count=10, progress=recorder) as tracker:
        tracker.advance(4)
        tracker.advance(1)
        tracker.advance(5)

    completed = [update.completed for update in recorder.updates]
    assert completed == [0, 4, 5, 10]


@pytest.mark.parametrize("count", [0, -1, -7])
def test_advance_rejects_a_non_positive_step(count: int) -> None:
    with make_tracker(input_count=5) as tracker:
        with pytest.raises(ValueError):
            tracker.advance(count)
        assert tracker.completed == 0


def test_advance_rejects_completing_more_inputs_than_exist() -> None:
    with make_tracker(input_count=3) as tracker:
        tracker.advance(2)
        with pytest.raises(ValueError):
            tracker.advance(2)
        assert tracker.completed == 2


def test_a_rejected_advance_emits_no_progress() -> None:
    recorder = Recorder()

    with make_tracker(input_count=2, progress=recorder) as tracker:
        with pytest.raises(ValueError):
            tracker.advance(9)

    assert [update.completed for update in recorder.updates] == [0]


def test_a_run_without_a_progress_callback_still_works() -> None:
    with make_tracker(input_count=2) as tracker:
        tracker.advance(2)

    assert tracker.completed == 2


def test_a_plain_function_satisfies_the_progress_callback() -> None:
    seen: list[int] = []

    with make_tracker(input_count=2, progress=lambda u: seen.append(u.completed)):
        pass

    assert seen == [0]


def test_advance_returns_the_update_it_emitted() -> None:
    with make_tracker(input_count=2) as tracker:
        update = tracker.advance()

    assert update == ProgressUpdate(
        operation="embed_documents", completed=1, total=2
    )


# --------------------------------------------------------------------------
# RunTracker: the same terms for every backend (5.6)
# --------------------------------------------------------------------------


def test_isolated_and_in_process_progress_are_identical_over_the_wire() -> None:
    """5.6: isolated execution reports progress on the same terms.

    The isolated backend can only send bytes, so its updates are reconstructed
    from JSON here. Identical sequences prove the callback shape is usable by
    both without a second progress vocabulary.
    """
    total = 20

    in_process: list[ProgressUpdate] = []
    with RunTracker(
        operation="embed_documents",
        provider_served=ProviderChoice.NPU,
        execution_mode=ExecutionMode.IN_PROCESS,
        input_count=total,
        progress=in_process.append,
        clock=Clock(0.0),
    ) as tracker:
        for _ in range(total):
            tracker.advance()

    wire: list[str] = []
    with RunTracker(
        operation="embed_documents",
        provider_served=ProviderChoice.NPU,
        execution_mode=ExecutionMode.ISOLATED,
        input_count=total,
        progress=lambda u: wire.append(json.dumps(u.as_mapping())),
        clock=Clock(0.0),
    ) as tracker:
        for _ in range(total):
            tracker.advance()

    isolated = [ProgressUpdate.from_mapping(json.loads(frame)) for frame in wire]
    assert isolated == in_process


# --------------------------------------------------------------------------
# RunTracker: successful completion (8.6, 2.6)
# --------------------------------------------------------------------------


def test_a_completed_run_summarises_time_count_provider_mode_truncation() -> None:
    """8.6 and 2.6, together: everything design.md's Monitoring note lists."""
    with make_tracker(
        input_count=3,
        provider_served=ProviderChoice.CPU,
        execution_mode=ExecutionMode.ISOLATED,
        clock=Clock(10.0, 10.0, 12.5),
    ) as tracker:
        tracker.record_truncated(2)
        tracker.advance(3)

    summary = tracker.result
    assert summary is not None
    assert summary.elapsed_seconds == pytest.approx(2.5)
    assert summary.input_count == 3
    assert summary.completed_count == 3
    assert summary.provider_served is ProviderChoice.CPU
    assert summary.execution_mode is ExecutionMode.ISOLATED
    assert summary.truncated_count == 2
    assert summary.interrupted is False
    assert summary.interruption is None


def test_the_summary_is_delivered_to_the_finish_callback_on_success() -> None:
    received: list[RunSummary] = []

    with make_tracker(input_count=2, on_finish=received.append) as tracker:
        tracker.advance(2)

    assert len(received) == 1
    assert received[0] is tracker.result


def test_no_summary_exists_before_the_run_ends() -> None:
    tracker = make_tracker(input_count=2)

    assert tracker.result is None
    with tracker:
        assert tracker.result is None
    assert tracker.result is not None


def test_record_truncated_accumulates() -> None:
    with make_tracker(input_count=5) as tracker:
        tracker.record_truncated(1)
        tracker.record_truncated(2)
        tracker.advance(5)

    assert tracker.result is not None
    assert tracker.result.truncated_count == 3


@pytest.mark.parametrize("count", [-1, 6])
def test_record_truncated_rejects_impossible_counts(count: int) -> None:
    with make_tracker(input_count=5) as tracker:
        with pytest.raises(ValueError):
            tracker.record_truncated(count)
        tracker.advance(5)


# --------------------------------------------------------------------------
# RunTracker: interruption (8.4)
# --------------------------------------------------------------------------


def embed_and_be_interrupted(
    sink: SummaryCallback, total: int, die_at: int
) -> None:
    """A run whose tracker the caller never sees - the realistic case.

    ``EmbeddingService`` creates the tracker internally, so a caller that
    catches ``KeyboardInterrupt`` has no object to interrogate. Requirement 8.4
    is only satisfied if the completed count reaches them anyway.
    """
    with RunTracker(
        operation="embed_documents",
        provider_served=ProviderChoice.NPU,
        execution_mode=ExecutionMode.IN_PROCESS,
        input_count=total,
        on_finish=sink,
        clock=Clock(0.0, 0.0, 4.0),
    ) as tracker:
        for index in range(total):
            if index == die_at:
                raise KeyboardInterrupt
            tracker.advance()


def test_a_keyboard_interrupt_still_reports_how_many_inputs_completed() -> None:
    """8.4, under an abrupt stop the run never chose to make."""
    received: list[RunSummary] = []

    with pytest.raises(KeyboardInterrupt):
        embed_and_be_interrupted(received.append, total=100, die_at=37)

    assert len(received) == 1
    summary = received[0]
    assert summary.completed_count == 37
    assert summary.input_count == 100
    assert summary.interrupted is True
    assert "KeyboardInterrupt" in (summary.interruption or "")
    assert summary.elapsed_seconds == pytest.approx(4.0)


def test_the_interruption_never_swallows_the_exception() -> None:
    with pytest.raises(IsolatedWorkerError):
        with make_tracker(input_count=4) as tracker:
            tracker.advance()
            raise IsolatedWorkerError("worker died")


def test_a_dead_worker_reports_the_completed_count_and_the_reason() -> None:
    """8.4 with 5.5: an execution failure mid-batch is an interruption."""
    received: list[RunSummary] = []

    with pytest.raises(IsolatedWorkerError):
        with make_tracker(input_count=6, on_finish=received.append) as tracker:
            tracker.advance(2)
            raise IsolatedWorkerError("worker died")

    reason = received[0].interruption or ""
    assert received[0].completed_count == 2
    assert "IsolatedWorkerError" in reason
    assert "worker died" in reason


def test_an_interrupted_run_records_progress_emitted_before_the_stop() -> None:
    recorder = Recorder()

    with pytest.raises(ZeroDivisionError):
        with make_tracker(input_count=5, progress=recorder) as tracker:
            tracker.advance(3)
            raise ZeroDivisionError

    assert [update.completed for update in recorder.updates] == [0, 3]
    assert tracker.result is not None
    assert tracker.result.completed_count == 3


def test_a_clean_early_exit_is_not_reported_as_a_successful_run() -> None:
    """8.6 reports *completion*; a run that stopped early did not complete.

    A ``break``, an early ``return``, or a backend that quietly produced fewer
    vectors than inputs raises nothing at all. Treating "no exception" as
    "success" would report a partial batch as a finished one - a silent
    shortfall with no error attached to notice it by.
    """
    received: list[RunSummary] = []

    with make_tracker(input_count=10, on_finish=received.append) as tracker:
        for _ in range(4):
            tracker.advance()

    summary = received[0]
    assert summary.completed_count == 4
    assert summary.input_count == 10
    assert summary.interrupted is True
    assert "4 of 10" in (summary.interruption or "")


def test_an_exception_raised_after_the_last_input_still_reports_completion(
) -> None:
    received: list[RunSummary] = []

    with pytest.raises(RuntimeError):
        with make_tracker(input_count=2, on_finish=received.append) as tracker:
            tracker.advance(2)
            raise RuntimeError("post-batch bookkeeping failed")

    assert received[0].completed_count == 2
    assert received[0].interrupted is True


def test_the_summary_is_stored_even_if_the_finish_callback_raises() -> None:
    def hostile(summary: RunSummary) -> None:
        raise RuntimeError("presentation code failed")

    tracker = make_tracker(input_count=2, on_finish=hostile)
    with pytest.raises(RuntimeError):
        with tracker:
            tracker.advance(2)

    assert tracker.result is not None
    assert tracker.result.completed_count == 2


# --------------------------------------------------------------------------
# Construction guards
# --------------------------------------------------------------------------


def test_the_tracker_rejects_auto_as_the_serving_provider() -> None:
    with pytest.raises(ValueError, match="auto"):
        make_tracker(provider_served=ProviderChoice.AUTO)


@pytest.mark.parametrize("operation", ["", "  "])
def test_the_tracker_rejects_a_blank_operation_label(operation: str) -> None:
    with pytest.raises(ValueError):
        make_tracker(operation=operation)


def test_the_tracker_rejects_a_negative_input_count() -> None:
    with pytest.raises(ValueError):
        make_tracker(input_count=-1)


def test_an_empty_batch_completes_without_advancing() -> None:
    recorder = Recorder()

    with make_tracker(input_count=0, progress=recorder) as tracker:
        pass

    assert tracker.result is not None
    assert tracker.result.completed_count == 0
    assert tracker.result.interrupted is False
    assert [update.remaining for update in recorder.updates] == [0]


# --------------------------------------------------------------------------
# A tracker records one run (8.4, 8.6)
# --------------------------------------------------------------------------


def test_a_second_run_cannot_report_success_over_a_partial_batch() -> None:
    """The count must not survive into a run that did not do the work.

    Ten inputs, four finished, then four more: a tracker that carried its count
    forward would sum them to ten and report ``interruption=None`` - a
    *successful* run over eight inputs. That is precisely the false success the
    RunSummary completeness invariant exists to forbid, reached by the side
    door rather than by constructing a bad summary.
    """
    received: list[RunSummary] = []
    tracker = make_tracker(input_count=10, on_finish=received.append)

    with tracker:
        tracker.advance(4)

    with pytest.raises(RuntimeError, match="already"):
        with tracker:
            tracker.advance(4)

    assert len(received) == 1
    assert received[0].completed_count == 4
    assert received[0].interrupted is True
    assert tracker.completed == 4


def test_a_tracker_cannot_be_entered_twice() -> None:
    """A summary is a record of one run, so a tracker serves one run.

    Reusing it would overwrite ``result`` and fire ``on_finish`` twice for what
    the caller holds as a single object. The service builds a tracker per
    operation, so re-entry is a programming error rather than a use case.
    """
    tracker = make_tracker(input_count=2)

    with tracker:
        tracker.advance(2)

    with pytest.raises(RuntimeError, match="already"):
        with tracker:
            pass


def test_re_entering_does_not_disturb_the_completed_run_s_record() -> None:
    tracker = make_tracker(input_count=4)
    with tracker:
        tracker.advance(4)
    first = tracker.result

    with pytest.raises(RuntimeError):
        with tracker:
            pass

    assert tracker.result is first
    assert tracker.result is not None
    assert tracker.result.interrupted is False


def test_re_entry_is_refused_even_while_the_first_run_is_open() -> None:
    tracker = make_tracker(input_count=4)

    with tracker:
        with pytest.raises(RuntimeError, match="already"):
            with tracker:
                pass
        tracker.advance(4)

    assert tracker.result is not None
    assert tracker.result.completed_count == 4


def test_re_entry_is_refused_after_an_interrupted_run() -> None:
    tracker = make_tracker(input_count=4)

    with pytest.raises(ZeroDivisionError):
        with tracker:
            tracker.advance(1)
            raise ZeroDivisionError

    with pytest.raises(RuntimeError, match="already"):
        with tracker:
            pass


# --------------------------------------------------------------------------
# Elapsed time comes from a monotonic source (8.6, and requirement 6 timings)
# --------------------------------------------------------------------------


def test_the_default_clock_is_monotonic() -> None:
    """The docstring promises a clock "unaffected by wall clock adjustments".

    Every other test injects a fake clock, so nothing else would notice this
    silently becoming ``time.time`` - and a backwards NTP correction mid-run
    would then land on the clamp below and report zero elapsed seconds for a
    long batch. Requirement 8.6 and the benchmark's timings share this path.
    """
    default = inspect.signature(RunTracker.__init__).parameters["clock"].default

    assert default is time.perf_counter
    readings = [default() for _ in range(64)]
    assert all(later >= earlier for earlier, later in zip(readings, readings[1:]))


def test_a_clock_that_runs_backwards_yields_no_negative_elapsed_time() -> None:
    """Binds the clamp: without it, RunSummary would reject the run outright."""
    with make_tracker(input_count=1, clock=Clock(100.0, 100.0, 40.0)) as tracker:
        tracker.advance()

    assert tracker.result is not None
    assert tracker.result.elapsed_seconds == 0.0


# --------------------------------------------------------------------------
# The module writes nothing to a terminal (design.md, Monitoring)
# --------------------------------------------------------------------------


def test_reporting_neither_prints_nor_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """design.md: "Progress is a callback, not logging, so callers choose
    presentation." A library that printed would decide presentation for them."""
    with make_tracker(input_count=3, progress=Recorder()) as tracker:
        tracker.advance(3)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


#: Call targets that would choose presentation for the caller. Matched against
#: the *unparsed* callee, so an attribute call such as ``sys.stderr.write(...)``
#: is caught as readily as a bare ``print(...)`` - the earlier guard inspected
#: only ``ast.Name`` targets and would have missed every one of these.
FORBIDDEN_OUTPUT_CALLS = (
    "sys.stdout",
    "sys.stderr",
    "stdout.",
    "stderr.",
    "logging.",
    "warnings.",
    "os.write",
)

#: Modules whose whole purpose is emitting somewhere this library does not own.
FORBIDDEN_OUTPUT_IMPORTS = ("logging", "sys", "warnings")


def test_reporting_source_calls_no_output_or_logging_api() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    imported = absolute_imports_of(source, MODULE_PACKAGE)

    assert [
        name
        for name in imported
        for forbidden in FORBIDDEN_OUTPUT_IMPORTS
        if name == forbidden or name.startswith(f"{forbidden}.")
    ] == []

    called = {
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    }
    offenders = sorted(
        callee
        for callee in called
        if callee == "print" or callee.startswith(FORBIDDEN_OUTPUT_CALLS)
    )
    assert offenders == []


@pytest.mark.parametrize(
    "statement",
    [
        "print(update)",
        "sys.stderr.write(str(update))",
        "sys.stdout.write(str(update))",
        "logging.getLogger(__name__).info(update)",
        "warnings.warn('progress')",
    ],
)
def test_the_output_guard_recognises_attribute_calls_too(statement: str) -> None:
    called = {
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(statement))
        if isinstance(node, ast.Call)
    }

    assert any(
        callee == "print" or callee.startswith(FORBIDDEN_OUTPUT_CALLS)
        for callee in called
    ), f"{statement!r} unparsed to {sorted(called)}"


# --------------------------------------------------------------------------
# Dependency direction (design.md, Architecture)
# --------------------------------------------------------------------------

MODULE_PACKAGE = "npu_rag.embedding"

#: design.md, Architecture: "types, errors -> reporting -> profiles ->
#: environment -> models -> providers -> service -> bench". ``reporting`` may
#: read ``types`` and ``errors`` and nothing else in this package.
LAYERS_RIGHT_OF_REPORTING = (
    "profiles",
    "environment",
    "models",
    "providers",
    "service",
    "bench",
)


def absolute_imports_of(source: str, package: str) -> list[str]:
    """Every name ``source`` imports, as an absolute dotted path.

    Relative imports are resolved rather than skipped, so ``from . import
    profiles`` cannot evade a guard that only recognises the absolute spelling.
    """
    parts = package.split(".")
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
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


def test_reporting_imports_nothing_from_a_later_layer() -> None:
    imported = absolute_imports_of(
        MODULE_PATH.read_text(encoding="utf-8"), MODULE_PACKAGE
    )

    forbidden = sorted(
        {
            name
            for name in imported
            for layer in LAYERS_RIGHT_OF_REPORTING
            if name == f"npu_rag.embedding.{layer}"
            or name.startswith(f"npu_rag.embedding.{layer}.")
        }
    )
    assert forbidden == []


@pytest.mark.parametrize(
    "statement",
    [
        "import npu_rag.embedding.profiles",
        "from npu_rag.embedding.providers import base",
        "from npu_rag.embedding import environment",
        "from . import profiles",
        "from .providers import base",
        "from ..embedding.service import EmbeddingService",
    ],
)
def test_the_layer_guard_recognises_every_import_spelling(statement: str) -> None:
    imported = absolute_imports_of(statement, MODULE_PACKAGE)

    assert any(
        name.startswith(f"npu_rag.embedding.{layer}")
        for name in imported
        for layer in LAYERS_RIGHT_OF_REPORTING
    ), f"{statement!r} resolved to {imported}"


def test_reporting_does_not_redefine_the_shared_vocabulary() -> None:
    """``ProviderChoice`` and ``ExecutionMode`` belong to ``types`` (task 2.1).

    A local re-definition would type-check and then compare unequal against
    every other module's copy.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    defined = {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef | ast.FunctionDef)
    }

    assert defined.isdisjoint({"ProviderChoice", "ExecutionMode"})


def test_mapping_helpers_accept_any_mapping() -> None:
    from types import MappingProxyType

    source: Mapping[str, object] = MappingProxyType(
        {"operation": "embed_queries", "completed": 1, "total": 4}
    )

    assert ProgressUpdate.from_mapping(source).remaining == 3
