"""Progress callbacks and run summaries (task 2.3).

design.md's Error Handling section states the shape of this module in one
sentence: "Progress is a callback, not logging, so callers choose presentation
(5.6, 8.3). Every completed operation emits elapsed time, input count, provider
served, execution mode, and truncation count (2.6, 8.6)." Nothing here writes to
a stream, configures a logger, or assumes a terminal exists - a library that
printed would be choosing presentation on the caller's behalf, and the
benchmark harness explicitly "must not assume a terminal".

**Why the payloads are plain data.** Requirement 5.6 says isolated execution
reports progress "on the same terms" as in-process execution, and the isolated
adapter's framing carries a JSON header and a raw numeric payload with no
pickle (design.md, "Isolated adapter framing"). A progress payload holding a
live object - the tracker, a session, the callback itself - would work in-process
and fail only across the process boundary, which is the worst place to discover
it. ``ProgressUpdate`` and ``RunSummary`` therefore convert to and from plain
mappings of JSON scalars, and hold nothing else.

**Why the summary is built by a context manager.** Requirement 8.6 wants a
report on success; requirement 8.4 wants a completed count when the batch is
interrupted *before* completion. An interruption is not something the run
chooses - a ``KeyboardInterrupt``, an ONNX Runtime failure mid-batch, or an
isolated worker dying (5.5) all unwind the stack without asking. Building the
summary only at the bottom of a successful loop would satisfy 8.6 and lose 8.4
entirely. ``RunTracker`` instead accumulates the count as the run proceeds and
finishes the summary in ``__exit__``, which Python runs during exception
unwinding, so both requirements are served by one mechanism.

**Why the summary has a delivery callback.** The service creates its tracker
internally, so a caller that catches ``KeyboardInterrupt`` at the top level
holds no object to interrogate. ``on_finish`` hands the summary out through the
same callback discipline as progress, which is what makes 8.4 reachable rather
than merely recorded.

This module sits immediately right of ``types`` and ``errors`` in design.md's
dependency direction - ``types, errors -> reporting -> profiles -> ...`` - so it
reads those two and nothing else in this package.

**Not here:** ``EmbedResult``. Requirement 3.9's ``truncated_indices`` and the
partition-verification verdict are claims only the service and the backends can
make, and they land with the service (task 5.3). ``RunSummary`` carries the
run-level facts 8.6 and 2.6 name; 5.3 composes it with the vectors rather than
duplicating it.

``RunSummary.fallback_reason`` was added by **task 4.1**, which owns
requirements 2.4 and 2.5 and produces the string in
``providers/base.resolve_backend``. Task 2.3 left it out deliberately - it had
no producer then - and recorded the obligation in tasks.md's Implementation
Notes so the carrier design.md's traceability table names could not go missing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

from npu_rag.embedding.types import ExecutionMode, ProviderChoice

__all__ = [
    "ProgressCallback",
    "ProgressUpdate",
    "RunSummary",
    "RunTracker",
    "SummaryCallback",
]


# --------------------------------------------------------------------------
# Shared validation
#
# The tracker validates eagerly at construction and the summary validates at
# construction too, so a misconfigured run fails at its own call site rather
# than several hundred inputs later. Sharing these helpers is what keeps the two
# checks from drifting apart.
# --------------------------------------------------------------------------


def _checked_label(value: str, field: str) -> str:
    """A non-blank label. A blank one renders as an anonymous progress bar."""
    if not value.strip():
        raise ValueError(f"{field} must be a non-blank label, got {value!r}")
    return value


def _checked_count(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{field} cannot be negative, got {value}")
    return value


def _checked_served(provider: ProviderChoice) -> ProviderChoice:
    """The provider that *actually* served, which is never ``auto``.

    Requirement 2.6 reports the provider that served an operation, and
    ``ProviderChoice.AUTO`` is a request rather than an outcome (types.py). A
    summary saying ``auto`` would report the question back instead of the answer
    - exactly the ambiguity requirement 2 exists to remove.
    """
    if provider is ProviderChoice.AUTO:
        raise ValueError(
            "provider_served names the provider that actually ran; 'auto' is a "
            "request, not an outcome (requirement 2.6) - report npu or cpu"
        )
    return provider


def _require_keys(mapping: Mapping[str, object], *keys: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")


def _int_from(mapping: Mapping[str, object], key: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an int, got {value!r}")
    return value


def _str_from(mapping: Mapping[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a str, got {value!r}")
    return value


def _optional_str_from(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a str or null, got {value!r}")
    return value


def _float_from(mapping: Mapping[str, object], key: str) -> float:
    value = mapping[key]
    # JSON writes 0.0 back as an int, so an int is a valid float here.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number, got {value!r}")
    return float(value)


# --------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressUpdate:
    """One progress observation: what is done, and what is left (8.3).

    ``remaining`` is derived from ``total`` rather than stored. Two independent
    fields could be sent out disagreeing with each other, and a caller rendering
    a bar from the wrong half would show a plausible, wrong number - the class
    of defect that produces no error.

    ``operation`` is the only identity carried, deliberately. A caller with two
    concurrent runs needs to tell their bars apart; anything more - provider,
    model, partition share - is run-level fact that belongs on ``RunSummary``,
    and repeating it on every one of thirty thousand updates would make the
    payload a second, competing summary.
    """

    operation: str
    completed: int
    total: int

    def __post_init__(self) -> None:
        _checked_label(self.operation, "operation")
        _checked_count(self.completed, "completed")
        _checked_count(self.total, "total")
        if self.completed > self.total:
            raise ValueError(
                f"completed ({self.completed}) cannot exceed total "
                f"({self.total}): progress past the end of the batch is a "
                "miscount, not a milestone"
            )

    @property
    def remaining(self) -> int:
        """Inputs still to do (requirement 8.3's second half)."""
        return self.total - self.completed

    def as_mapping(self) -> dict[str, object]:
        """A JSON-serialisable view, for crossing a process boundary (5.6)."""
        return {
            "operation": str(self.operation),
            "completed": int(self.completed),
            "total": int(self.total),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Rebuild an update received as plain data. Inverse of `as_mapping`."""
        _require_keys(mapping, "operation", "completed", "total")
        return cls(
            operation=_str_from(mapping, "operation"),
            completed=_int_from(mapping, "completed"),
            total=_int_from(mapping, "total"),
        )


class ProgressCallback(Protocol):
    """How a caller learns a batch is advancing (5.6, 8.3).

    The parameter is positional-only so any single-argument callable satisfies
    it - ``list.append``, a lambda, a bound method, a class with ``__call__`` -
    without having to match a parameter name.

    A callback that raises is not caught. Swallowing it would need somewhere to
    record the loss, and this module has deliberately nowhere to record
    anything; the run is interrupted instead, which requirement 8.4 already
    covers.
    """

    def __call__(self, update: ProgressUpdate, /) -> None: ...


# --------------------------------------------------------------------------
# Run summary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    """What happened over one embedding run (2.6, 8.4, 8.6).

    ``interruption`` follows design.md's ``Measurement`` precedent of a single
    nullable reason rather than a flag plus a reason that can disagree: ``None``
    means the run finished, and any other value is why it did not. The invariant
    that a finished run completed every input is enforced here, so "success"
    cannot be reported over a partial batch.

    ``completed_count`` and ``input_count`` are both carried because they answer
    two different requirements: 8.6 wants how many inputs the successful run
    processed, and 8.4 wants how many of them finished when it did not.

    ``fallback_reason`` is requirement 2.5's carrier, added by task 4.1 which
    owns that requirement (design.md's Requirements Traceability maps 2.4/2.5 to
    this field). It follows ``interruption``'s single-nullable-reason shape:
    ``None`` means nothing was substituted, and any other value is the specific
    reason the NPU was not used. It is meaningful only under an ``auto``
    selection - an explicit choice is either honoured or fails - and only ever
    accompanies a CPU-served run, which is the invariant enforced below.
    """

    operation: str
    provider_served: ProviderChoice
    execution_mode: ExecutionMode
    input_count: int
    completed_count: int
    truncated_count: int
    elapsed_seconds: float
    interruption: str | None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        _checked_label(self.operation, "operation")
        _checked_served(self.provider_served)
        _checked_count(self.input_count, "input_count")
        _checked_count(self.completed_count, "completed_count")
        _checked_count(self.truncated_count, "truncated_count")
        if self.elapsed_seconds < 0 or not math.isfinite(self.elapsed_seconds):
            raise ValueError(
                f"elapsed_seconds must be a finite, non-negative number of "
                f"seconds, got {self.elapsed_seconds!r}"
            )
        if self.completed_count > self.input_count:
            raise ValueError(
                f"completed_count ({self.completed_count}) cannot exceed "
                f"input_count ({self.input_count})"
            )
        if self.truncated_count > self.input_count:
            raise ValueError(
                f"truncated_count ({self.truncated_count}) cannot exceed "
                f"input_count ({self.input_count})"
            )
        if self.interruption is None:
            if self.completed_count != self.input_count:
                raise ValueError(
                    f"a run reported as complete must have finished every "
                    f"input, but {self.completed_count} of {self.input_count} "
                    "did: set interruption to say why it stopped (8.4)"
                )
        else:
            _checked_label(self.interruption, "interruption")
        if self.fallback_reason is not None:
            _checked_label(self.fallback_reason, "fallback_reason")
            if self.provider_served is not ProviderChoice.CPU:
                raise ValueError(
                    "fallback_reason explains why the NPU was not used, so the "
                    f"CPU must be what served; got provider_served="
                    f"{self.provider_served.value!r} (requirement 2.5)"
                )

    @property
    def interrupted(self) -> bool:
        """Whether the run stopped before finishing (8.4)."""
        return self.interruption is not None

    def as_mapping(self) -> dict[str, object]:
        """A JSON-serialisable view, symmetric with `ProgressUpdate`."""
        return {
            "operation": str(self.operation),
            "provider_served": self.provider_served.value,
            "execution_mode": self.execution_mode.value,
            "input_count": int(self.input_count),
            "completed_count": int(self.completed_count),
            "truncated_count": int(self.truncated_count),
            "elapsed_seconds": float(self.elapsed_seconds),
            "interruption": self.interruption,
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Rebuild a summary received as plain data. Inverse of `as_mapping`."""
        _require_keys(
            mapping,
            "operation",
            "provider_served",
            "execution_mode",
            "input_count",
            "completed_count",
            "truncated_count",
            "elapsed_seconds",
            "interruption",
            "fallback_reason",
        )
        return cls(
            operation=_str_from(mapping, "operation"),
            provider_served=ProviderChoice(_str_from(mapping, "provider_served")),
            execution_mode=ExecutionMode(_str_from(mapping, "execution_mode")),
            input_count=_int_from(mapping, "input_count"),
            completed_count=_int_from(mapping, "completed_count"),
            truncated_count=_int_from(mapping, "truncated_count"),
            elapsed_seconds=_float_from(mapping, "elapsed_seconds"),
            interruption=_optional_str_from(mapping, "interruption"),
            fallback_reason=_optional_str_from(mapping, "fallback_reason"),
        )


class SummaryCallback(Protocol):
    """How a caller receives the run summary, however the run ended.

    This is what makes requirement 8.4 reachable: an interrupted run's summary
    has to reach someone who never held the tracker, and the callback the caller
    supplied is the only reference that survives the stack unwinding.
    """

    def __call__(self, summary: RunSummary, /) -> None: ...


# --------------------------------------------------------------------------
# The tracker
# --------------------------------------------------------------------------


def _describe(exc_type: type[BaseException], exc: BaseException | None) -> str:
    """Why the run stopped, in one line, never blank.

    The exception type leads because 8.2's category distinction is carried by
    the type; the message follows when there is one. ``KeyboardInterrupt``
    usually has none, which is why the type alone must already be a valid
    reason.
    """
    detail = str(exc) if exc is not None else ""
    if detail.strip():
        return f"{exc_type.__qualname__}: {detail}"
    return exc_type.__qualname__


class RunTracker:
    """Counts a run's progress and finishes its summary however it ends.

    Used as a context manager, always::

        with RunTracker(...) as tracker:
            for batch in batches:
                ...
                tracker.advance(len(batch))

    On the way out - normally or through an exception - it builds a
    `RunSummary` and hands it to ``on_finish``. The exception is never
    swallowed: ``__exit__`` returns ``None``, so an interrupted batch reports
    its completed count *and* still fails (8.4).

    The count is accumulated as work happens rather than computed at the end,
    which is the whole reason 8.4 can be satisfied under an abrupt stop. That
    also makes a tracker **single-use**: entering it twice is refused, because
    the accumulated count would otherwise carry into a run that did not do the
    work. See `__enter__`.

    ``clock`` is injectable so elapsed time is testable without sleeping; it
    defaults to ``time.perf_counter``, which is monotonic and unaffected by wall
    clock adjustments during a long run.
    """

    def __init__(
        self,
        *,
        operation: str,
        provider_served: ProviderChoice,
        execution_mode: ExecutionMode,
        input_count: int,
        progress: ProgressCallback | None = None,
        on_finish: SummaryCallback | None = None,
        fallback_reason: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._operation = _checked_label(operation, "operation")
        self._provider_served = _checked_served(provider_served)
        self._execution_mode = execution_mode
        self._input_count = _checked_count(input_count, "input_count")
        # Validated here rather than only at ``__exit__``. The summary is built
        # when the run ends, which for an interrupted run is while an exception
        # is already propagating; a `RunSummary` rejecting the pair there would
        # replace the caller's real failure with a reporting one. Task 5.3 added
        # this parameter: until then `RunSummary.fallback_reason` had no
        # producer, so requirement 2.5's carrier could never be filled by the
        # component that knows the reason.
        if fallback_reason is not None:
            _checked_label(fallback_reason, "fallback_reason")
            if self._provider_served is not ProviderChoice.CPU:
                raise ValueError(
                    "fallback_reason explains why the NPU was not used, so the "
                    f"CPU must be what served; got provider_served="
                    f"{self._provider_served.value!r} (requirement 2.5)"
                )
        self._fallback_reason = fallback_reason
        self._progress = progress
        self._on_finish = on_finish
        self._clock = clock
        self._completed = 0
        self._truncated = 0
        self._started = clock()
        self._entered = False
        self._result: RunSummary | None = None

    @property
    def completed(self) -> int:
        """Inputs finished so far. Never decreases (the task's Observable)."""
        return self._completed

    @property
    def input_count(self) -> int:
        return self._input_count

    @property
    def result(self) -> RunSummary | None:
        """The summary, once the run has ended; ``None`` while it is running.

        Set before ``on_finish`` is called, so a hostile presentation callback
        cannot cost the caller the record.
        """
        return self._result

    def __enter__(self) -> Self:
        """Begin the run. A tracker records exactly one.

        Re-entry is refused rather than reset. The completed count is
        accumulated across the block, so a second run over the same tracker
        would add its inputs to the first run's total: ten inputs, four
        finished, then four more, and the summary reports ten of ten with no
        interruption - a *successful* run over eight inputs. That is the false
        success `RunSummary`'s completeness invariant exists to forbid, reached
        by carrying state between runs instead of by building a bad summary.

        Resetting the counters would close that hole too, but silently: it would
        discard the first run's `result` and fire ``on_finish`` twice for what
        the caller holds as one object. A tracker is built per operation, so
        re-entry is a caller's mistake, and saying so is better than papering
        over it.
        """
        if self._entered:
            raise RuntimeError(
                "this RunTracker has already been entered and records a single "
                "run; build a new one per operation rather than reusing it"
            )
        self._entered = True
        self._started = self._clock()
        self._emit()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finish the summary and deliver it. Never suppresses the exception.

        ``exc_type`` is ``BaseException``-wide on purpose: ``KeyboardInterrupt``
        and ``SystemExit`` are exactly the abrupt stops requirement 8.4 is
        about, and a handler that only saw ``Exception`` would miss them.

        A raising ``on_finish`` propagates, with the original exception kept as
        its ``__context__``; the summary is already stored either way.

        A block that exits *cleanly* having advanced past fewer inputs than it
        was given - a ``break``, an early ``return``, a backend that quietly
        produced short output - is not a success either. Requirement 8.6 reports
        completion, and this run did not complete; it is recorded with the
        no-error reason so 8.4's count still reaches the caller.
        """
        self._result = RunSummary(
            operation=self._operation,
            provider_served=self._provider_served,
            execution_mode=self._execution_mode,
            input_count=self._input_count,
            completed_count=self._completed,
            truncated_count=self._truncated,
            elapsed_seconds=max(0.0, self._clock() - self._started),
            interruption=self._interruption(exc_type, exc),
            fallback_reason=self._fallback_reason,
        )
        if self._on_finish is not None:
            self._on_finish(self._result)
        return None

    def _interruption(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
    ) -> str | None:
        """``None`` only when the run both raised nothing and finished."""
        if exc_type is not None:
            return _describe(exc_type, exc)
        if self._completed != self._input_count:
            return (
                f"ended without an error after {self._completed} of "
                f"{self._input_count} inputs"
            )
        return None

    def advance(self, count: int = 1) -> ProgressUpdate:
        """Record ``count`` more finished inputs and emit progress (8.3).

        Rejects a non-positive step and any step that would carry the count past
        ``input_count``. Both are ways of reporting progress that did not
        happen: the first would emit a repeated or falling count, the second a
        count larger than the work. The counter is left untouched when the step
        is rejected, so a caught error cannot leave progress inconsistent.
        """
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"count must be an int, got {count!r}")
        if count < 1:
            raise ValueError(
                f"advance must move forward by at least one input, got {count}: "
                "progress never stalls or reverses (requirement 8.3)"
            )
        if self._completed + count > self._input_count:
            raise ValueError(
                f"advancing by {count} would report "
                f"{self._completed + count} of {self._input_count} inputs "
                "completed; a run cannot finish more inputs than it was given"
            )
        self._completed += count
        return self._emit()

    def record_truncated(self, count: int = 1) -> None:
        """Note that ``count`` more inputs were shortened to fit (3.9, 8.6).

        The count is the run-level fact design.md's Monitoring note lists.
        *Which* inputs were truncated is `EmbedResult.truncated_indices`, owned
        by the service (task 5.3), because only the service knows the ordering.
        """
        _checked_count(count, "count")
        if self._truncated + count > self._input_count:
            raise ValueError(
                f"truncating {count} more would report "
                f"{self._truncated + count} of {self._input_count} inputs "
                "truncated; a run cannot truncate more inputs than it was given"
            )
        self._truncated += count

    def _emit(self) -> ProgressUpdate:
        update = ProgressUpdate(
            operation=self._operation,
            completed=self._completed,
            total=self._input_count,
        )
        if self._progress is not None:
            self._progress(update)
        return update
