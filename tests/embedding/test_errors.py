"""Unit tests for the error taxonomy (task 2.1).

Requirement 8.2 asks for environment, preparation, and execution failures to be
*distinguishable*, and design.md's Error Strategy fixes how: "make the category
structural so 8.2 is satisfied by the type rather than by message text". The
central tests here are therefore written as real ``try``/``except`` blocks
rather than as ``issubclass`` assertions - a hierarchy that had been flattened
would still satisfy an ``issubclass`` matrix written the lazy way, but it cannot
satisfy an ``except`` clause that must not fire.

Requirement 8.1 asks that a failure report name the provider, the model, and the
stage. That is enforced here over every exported error type, including the ones
constructed knowing none of them, because the moment an operation fails is
exactly the moment those three facts are hardest to reconstruct.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from npu_rag.embedding import errors
from npu_rag.embedding.errors import (
    EmbeddingRuntimeError,
    EnvironmentError_,
    ExecutionError,
    IsolatedWorkerError,
    LicenseAcceptanceRequired,
    NpuUnavailableError,
    PartitionShareTooLow,
    PreparationError,
)
from npu_rag.embedding.types import ProviderChoice

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "npu_rag"
    / "embedding"
    / "errors.py"
)

#: design.md, Error Strategy: the three top categories that map one-to-one onto
#: requirement 8.2's required distinction.
CATEGORIES: tuple[type[EmbeddingRuntimeError], ...] = (
    EnvironmentError_,
    PreparationError,
    ExecutionError,
)

#: Every named subclass design.md specifies, against the category it belongs to.
SUBCLASS_CATEGORY: tuple[tuple[type[EmbeddingRuntimeError], type[EmbeddingRuntimeError]], ...] = (
    (NpuUnavailableError, EnvironmentError_),
    (LicenseAcceptanceRequired, PreparationError),
    (PartitionShareTooLow, PreparationError),
    (IsolatedWorkerError, ExecutionError),
)

#: Constructor arguments a type demands beyond the message. Kept as a table so
#: that ``test_every_exported_error_type_is_covered_here`` fails loudly when a
#: type is added to the taxonomy without being exercised below.
REQUIRED_ARGUMENTS: dict[str, dict[str, object]] = {
    "EmbeddingRuntimeError": {},
    "EnvironmentError_": {},
    "NpuUnavailableError": {},
    "PreparationError": {},
    "LicenseAcceptanceRequired": {
        "acceptance_url": "https://huggingface.co/google/embeddinggemma-300m"
    },
    "PartitionShareTooLow": {},
    "ExecutionError": {},
    "IsolatedWorkerError": {},
}


def error_type(name: str) -> type[EmbeddingRuntimeError]:
    exported = getattr(errors, name)
    assert isinstance(exported, type) and issubclass(exported, EmbeddingRuntimeError)
    return exported


def make(
    error_class: type[EmbeddingRuntimeError], **overrides: object
) -> EmbeddingRuntimeError:
    """Construct ``error_class`` knowing nothing but its message, unless the
    caller says otherwise."""
    arguments = dict(REQUIRED_ARGUMENTS[error_class.__name__])
    arguments.update(overrides)
    return error_class("something went wrong", **arguments)  # type: ignore[arg-type]


ALL_NAMES = tuple(errors.__all__)


def test_every_exported_error_type_is_covered_here() -> None:
    assert set(ALL_NAMES) == set(REQUIRED_ARGUMENTS)


# --------------------------------------------------------------------------
# Requirement 8.2: the categories are separable by type
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raised", CATEGORIES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("caught_by", CATEGORIES, ids=lambda cls: cls.__name__)
def test_a_category_catches_itself_and_no_other_category(
    raised: type[EmbeddingRuntimeError], caught_by: type[EmbeddingRuntimeError]
) -> None:
    """The Observable, stated directly: catching the environment category
    succeeds without catching preparation or execution failures."""
    caught = False
    try:
        raise make(raised)
    except caught_by:
        caught = True
    except EmbeddingRuntimeError:
        pass

    assert caught is (raised is caught_by)


@pytest.mark.parametrize(
    "subclass, category",
    SUBCLASS_CATEGORY,
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_a_named_subclass_is_caught_only_by_its_own_category(
    subclass: type[EmbeddingRuntimeError], category: type[EmbeddingRuntimeError]
) -> None:
    for candidate in CATEGORIES:
        caught = False
        try:
            raise make(subclass)
        except candidate:
            caught = True
        except EmbeddingRuntimeError:
            pass
        assert caught is (candidate is category), (
            f"{subclass.__name__} must be caught by {category.__name__} alone, "
            f"but `except {candidate.__name__}` {'did' if caught else 'did not'} fire"
        )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_error_is_caught_by_the_taxonomy_root(name: str) -> None:
    """One ``except`` clause still covers everything this feature raises, so a
    caller that does not care about the category need not enumerate them."""
    with pytest.raises(EmbeddingRuntimeError):
        raise make(error_type(name))


def test_the_environment_category_is_not_the_builtin_of_that_name() -> None:
    """``builtins.EnvironmentError`` is an alias of ``OSError``. Inheriting from
    it - an easy slip given the name design.md chose - would make every failed
    file operation in the process look like an environment failure of this
    feature."""
    assert not issubclass(EnvironmentError_, OSError)
    assert issubclass(EnvironmentError_, EmbeddingRuntimeError)


# --------------------------------------------------------------------------
# Requirement 8.1: provider, model, and stage on every instance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_error_exposes_provider_model_and_stage_when_nothing_is_known(
    name: str,
) -> None:
    error = make(error_type(name))

    assert error.provider is None
    assert error.model_id is None
    assert isinstance(error.stage, str) and error.stage.strip()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_error_carries_the_context_it_was_given(name: str) -> None:
    error = make(
        error_type(name),
        provider=ProviderChoice.NPU,
        model_id="embeddinggemma-300m",
        stage="export",
    )

    assert error.provider is ProviderChoice.NPU
    assert error.model_id == "embeddinggemma-300m"
    assert error.stage == "export"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_error_reports_its_context_in_its_message(name: str) -> None:
    """8.1 is a *reporting* requirement: the operator reading a traceback must
    see the three facts without a debugger. The type remains the discriminator
    (8.2); this text is for the human."""
    rendered = str(
        make(
            error_type(name),
            provider=ProviderChoice.CPU,
            model_id="bge-large-en-v1.5",
            stage="session_run",
        )
    )

    assert "something went wrong" in rendered
    assert "cpu" in rendered
    assert "bge-large-en-v1.5" in rendered
    assert "session_run" in rendered


@pytest.mark.parametrize("name", ALL_NAMES)
def test_an_unknown_provider_and_model_are_reported_as_unknown(name: str) -> None:
    """Absence has one representation - ``None`` on the attribute, the word
    ``unknown`` in the text - so a reader never has to guess whether a blank
    means "not applicable" or "we forgot to say"."""
    rendered = str(make(error_type(name)))

    assert rendered.count("unknown") == 2


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_model_id_is_normalized_to_unknown(name: str, blank: str) -> None:
    assert make(error_type(name), model_id=blank).model_id is None


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_stage_falls_back_to_the_type_s_own_stage(
    name: str, blank: str
) -> None:
    """A stage is never absent: a raiser always knows what it was doing, and
    where it does not say, the type does."""
    error_class = error_type(name)
    assert make(error_class, stage=blank).stage == error_class.default_stage


@pytest.mark.parametrize("name", ALL_NAMES)
def test_no_type_inherits_a_stage_that_names_nothing(name: str) -> None:
    """Each type states its own stage rather than borrowing its parent's, so
    ``LicenseAcceptanceRequired`` does not report itself as generic
    "preparation" when the raiser omits the detail (4.6)."""
    error_class = error_type(name)
    stage = error_class.default_stage

    assert stage.strip()
    if error_class is not EmbeddingRuntimeError:
        assert "default_stage" in vars(error_class), (
            f"{name} does not declare its own default stage"
        )


@pytest.mark.parametrize(
    "name, stage",
    [
        ("EnvironmentError_", "environment"),
        ("NpuUnavailableError", "provider_resolution"),
        ("PreparationError", "preparation"),
        ("LicenseAcceptanceRequired", "acquisition"),
        ("PartitionShareTooLow", "verify"),
        ("ExecutionError", "execution"),
        ("IsolatedWorkerError", "isolated_worker"),
    ],
)
def test_each_type_names_the_stage_its_failure_belongs_to(
    name: str, stage: str
) -> None:
    """design.md, Error Strategy, names the preparation stages as acquisition,
    export, compile, and verify; the remaining defaults follow the point in the
    flow where design.md says each failure is raised."""
    assert error_type(name).default_stage == stage


def test_the_raising_error_is_an_ordinary_exception() -> None:
    """Nothing in this taxonomy validates its way into raising from a
    constructor: an error class that can fail to be built would replace a
    diagnosed failure with a confusing one."""
    error = EmbeddingRuntimeError("plain")
    assert isinstance(error, Exception)
    assert error.args == ("plain",)


# --------------------------------------------------------------------------
# What the named subclasses carry beyond the three common facts
# --------------------------------------------------------------------------


def test_license_acceptance_required_carries_the_acceptance_step() -> None:
    """Requirement 4.5: report the licensing requirement and the acceptance
    step rather than a generic retrieval error. The acceptance URL is a
    constructor requirement, so the specific error cannot be raised without
    telling the operator where to go."""
    error = LicenseAcceptanceRequired(
        "embeddinggemma-300m is gated",
        acceptance_url="https://huggingface.co/google/embeddinggemma-300m",
        model_id="embeddinggemma-300m",
    )

    assert error.acceptance_url == "https://huggingface.co/google/embeddinggemma-300m"
    assert error.acceptance_url in str(error)


def test_license_acceptance_required_cannot_be_raised_without_the_step() -> None:
    with pytest.raises(TypeError):
        LicenseAcceptanceRequired("gated")  # type: ignore[call-arg]


def test_partition_share_too_low_distinguishes_measured_from_unverifiable() -> None:
    """design.md, TransformerBackend Risks: under an explicit ``npu`` choice an
    *unverifiable* partition share fails exactly as a measured-too-low one
    does, but the two are different facts and the report must not conflate
    them."""
    measured = PartitionShareTooLow(
        "the graph ran mostly on the CPU",
        observed_share=0.0,
        minimum_share=0.9,
    )
    unverifiable = PartitionShareTooLow(
        "the partition summary could not be read",
        minimum_share=0.9,
    )

    assert measured.observed_share == 0.0
    assert measured.unverifiable is False
    assert unverifiable.observed_share is None
    assert unverifiable.unverifiable is True
    assert measured.minimum_share == unverifiable.minimum_share == 0.9


def test_partition_share_zero_is_a_measurement_not_an_absence() -> None:
    """The authoritative summary on this machine reported ``Number of operators
    supported by VAIML: 0(0.000%)``. A falsy-rather-than-``None`` test would
    read that measured zero as "never measured"."""
    assert PartitionShareTooLow("none offloaded", observed_share=0.0).unverifiable is False


# --------------------------------------------------------------------------
# Boundary guard
# --------------------------------------------------------------------------


def test_errors_imports_nothing_from_this_project_but_types() -> None:
    """design.md, Architecture: ``types`` and ``errors`` are the leftmost
    layer. ``errors`` may name the provider vocabulary and nothing else."""
    imported: list[str] = []
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            assert node.level == 0, "a relative import is still a project import"

    project_imports: Callable[[str], bool] = lambda name: name.startswith("npu_rag")
    assert [name for name in imported if project_imports(name)] == [
        "npu_rag.embedding.types"
    ]
