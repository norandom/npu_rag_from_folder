"""The vocabulary every other module in this feature speaks (task 2.1).

This is the leftmost layer of design.md's dependency direction - ``types,
errors -> reporting -> profiles -> environment -> models -> providers ->
service -> bench`` - so it imports nothing from this project at all. Everything
here is a value: an enumeration or a frozen dataclass, with no behaviour beyond
the invariants that make an illegal value unconstructible.

Three of these types spent task 1.5 living in ``environment/capability.py``,
because design.md's File Structure Plan assigns ``CapabilityReport`` to this
module while the CapabilityChecker section sketches it inline, and 1.5 could not
create a module this task owns. They are moved here now and re-exported from
``capability.py``, so there is one class per concept rather than one on each
side of the boundary.

**What is deliberately not here yet.** design.md's File Structure Plan also
lists ``EmbedResult`` and ``ModelProfile``. ``ModelProfile`` belongs to task
2.2. ``EmbedResult`` and ``EmbeddingContract`` land with the service (task 5.3):
their fields are claims *about a completed operation* - which provider actually
served it, whether the partition share was verified rather than assumed, which
inputs were truncated - and every one of those invariants is enforced by the
component that produces them. Declaring the shape here first would add a record
no code can fill and no test can falsify.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "CONDITION_PROVIDER_REGISTERED",
    "CapabilityReport",
    "Condition",
    "DocumentText",
    "ExecutionMode",
    "ProviderChoice",
    "TextKind",
]


class TextKind(StrEnum):
    """Whether a text is corpus content or a search query (requirement 3.2).

    There is no third member and no default: requirement 3.3 requires a request
    that omits the kind to be *rejected* rather than assumed, and an
    ``UNSPECIFIED`` member would be exactly the assumption it forbids. The
    embedding surface enforces this structurally - ``embed_documents`` and
    ``embed_queries`` are separate calls (design.md, EmbeddingService) - so this
    enumeration exists to name the two conventions, not to be passed around as
    an optional flag.
    """

    DOCUMENT = "document"
    QUERY = "query"


class ProviderChoice(StrEnum):
    """The compute provider a caller asks for (requirement 2.1).

    ``AUTO`` is a *request*, never an outcome: requirement 2.6 makes the runtime
    report which provider actually served an operation, and that report names
    ``NPU`` or ``CPU``. Requirement 2.2 forbids ``NPU`` from ever degrading to
    another provider silently, which is why the choice is carried explicitly
    rather than inferred from what happens to be available.
    """

    NPU = "npu"
    CPU = "cpu"
    AUTO = "auto"


class ExecutionMode(StrEnum):
    """Where NPU execution is reachable from, if anywhere (requirement 1.5)."""

    IN_PROCESS = "in_process"
    ISOLATED = "isolated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DocumentText:
    """A corpus text together with the title its template has a slot for.

    Documents are not bare strings because the document-side convention is a
    template with two slots, not a prefix: EmbeddingGemma's is
    ``title: {title} | text: {content}``, and an absent title is rendered as the
    literal sentinel ``none`` (requirement 3.4, design.md EmbeddingService).

    Invariant: a title is either a real title or explicitly ``None``. A blank
    string is the ambiguous third state - it would render as an empty slot,
    which is neither the title nor the sentinel, and would silently change the
    text that gets embedded without changing any shape or norm a test would
    notice.

    Content is deliberately unvalidated here. What counts as usable input is a
    service-level judgement bound to a model's compiled length, and requirement
    3.9 requires over-long input to be *reported and shortened* rather than
    rejected; a value object that second-guessed that would move the decision to
    the wrong place.
    """

    content: str
    title: str | None = None

    def __post_init__(self) -> None:
        if self.title is not None and not self.title.strip():
            raise ValueError(
                "a blank title is not a title: pass None so the document "
                "template renders its absence sentinel, or pass the real title"
            )


#: The condition whose satisfaction *is* the in-process verdict. It lives beside
#: ``CapabilityReport`` because that report's invariant reads it (design.md,
#: Domain Model: ``execution_mode`` is ``IN_PROCESS`` only when EP registration
#: succeeded in this interpreter). The remaining condition names stay with the
#: checker that produces them, in ``environment/capability.py``.
CONDITION_PROVIDER_REGISTERED = "execution_provider_registered"


@dataclass(frozen=True)
class Condition:
    """One environment condition, evaluated on its own (requirement 1.2).

    Invariant: an unsatisfied condition carries ``observed``, ``required`` and
    ``remediation``, which is requirement 1.3 made structural. A satisfied
    condition may carry them and usually does - the driver condition reports
    both versions whether or not it passes, because those numbers are the whole
    point of looking.
    """

    name: str
    satisfied: bool
    observed: str | None
    required: str | None
    remediation: str | None

    def __post_init__(self) -> None:
        if self.satisfied:
            return
        missing = [
            field
            for field in ("observed", "required", "remediation")
            if not getattr(self, field)
        ]
        if missing:
            raise ValueError(
                f"unsatisfied condition {self.name!r} must carry "
                f"{', '.join(missing)}: an unmet condition without a "
                "remediation is a dead end for the operator (requirement 1.3)"
            )


@dataclass(frozen=True)
class CapabilityReport:
    """Every condition, the execution-mode verdict, and run provenance.

    ``driver_version`` and ``runtime_version`` are carried here because
    requirement 6.6 builds the benchmark's run context from this report, so
    version provenance has exactly one source.
    """

    conditions: tuple[Condition, ...]
    execution_mode: ExecutionMode
    driver_version: str | None
    runtime_version: str | None
    device_name: str | None
    #: Whether *this platform* reports estimated power at all (Strix and later
    #: do; PHX/HPT and Linux do not). Not whether the most recent sample carried
    #: a number - ``N/A`` occurs intermittently here on a part that does support
    #: it, and requirement 6.8 needs those two omissions to stay distinct.
    power_reporting_supported: bool

    def __post_init__(self) -> None:
        names = [condition.name for condition in self.conditions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate condition names in report: {names}")
        if self.execution_mode is not ExecutionMode.IN_PROCESS:
            return
        registered = {
            condition.name: condition.satisfied for condition in self.conditions
        }.get(CONDITION_PROVIDER_REGISTERED)
        if registered is not True:
            raise ValueError(
                "execution_mode IN_PROCESS requires a satisfied "
                f"{CONDITION_PROVIDER_REGISTERED!r} condition: the provider "
                "registering in this interpreter is the only evidence that "
                "in-process NPU execution is reachable"
            )

    def condition(self, name: str) -> Condition:
        """The named condition. An unknown name is a programming error."""
        for condition in self.conditions:
            if condition.name == name:
                return condition
        raise KeyError(name)
