"""Unit tests for the ingest error taxonomy (task 1.2).

design.md's Error Strategy makes the category structural so a caller
separates failures with ``except`` rather than by matching strings. The
central tests here are therefore written as real ``try``/``except`` blocks
rather than as ``issubclass`` assertions.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from npu_rag.ingest import errors
from npu_rag.ingest.errors import (
    DiscoveryError,
    ExtractionError,
    IngestError,
    StateError,
    VisionError,
    VisionUnavailable,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "ingest" / "errors.py"
)

#: design.md, Error Strategy: the named categories under the root.
CATEGORIES: tuple[type[IngestError], ...] = (
    DiscoveryError,
    ExtractionError,
    VisionError,
    VisionUnavailable,
    StateError,
)

REQUIRED_ARGUMENTS: dict[str, dict[str, object]] = {
    "IngestError": {},
    "DiscoveryError": {},
    "ExtractionError": {},
    "VisionError": {},
    "VisionUnavailable": {},
    "StateError": {},
}


def error_type(name: str) -> type[IngestError]:
    exported = getattr(errors, name)
    assert isinstance(exported, type) and issubclass(exported, IngestError)
    return exported


def make(error_class: type[IngestError], **overrides: object) -> IngestError:
    arguments = dict(REQUIRED_ARGUMENTS[error_class.__name__])
    arguments.update(overrides)
    return error_class("something went wrong", **arguments)  # type: ignore[arg-type]


ALL_NAMES = tuple(errors.__all__)


def test_every_exported_error_type_is_covered_here() -> None:
    assert set(ALL_NAMES) == set(REQUIRED_ARGUMENTS)


# --------------------------------------------------------------------------
# Categories are separable by type
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raised", CATEGORIES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("caught_by", CATEGORIES, ids=lambda cls: cls.__name__)
def test_a_category_catches_itself_and_no_other_category(
    raised: type[IngestError], caught_by: type[IngestError]
) -> None:
    caught = False
    try:
        raise make(raised)
    except caught_by:
        caught = True
    except IngestError:
        pass

    assert caught is (raised is caught_by)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_error_is_caught_by_the_taxonomy_root(name: str) -> None:
    with pytest.raises(IngestError):
        raise make(error_type(name))


def test_vision_unavailable_is_not_a_vision_error() -> None:
    """401/403 latch as VisionUnavailable; other vision failures are
    VisionError. Catching one must not swallow the other."""
    assert not issubclass(VisionUnavailable, VisionError)
    assert not issubclass(VisionError, VisionUnavailable)


# --------------------------------------------------------------------------
# Stage and path on every instance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_error_exposes_stage_and_path_when_nothing_is_known(name: str) -> None:
    error = make(error_type(name))

    assert error.path is None
    assert isinstance(error.stage, str) and error.stage.strip()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_error_carries_the_stage_and_path_it_was_given(name: str) -> None:
    path = Path("archive/alice/broken.pdf")
    error = make(error_type(name), stage="extraction", path=path)

    assert error.stage == "extraction"
    assert error.path == path


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_error_reports_stage_and_path_in_its_message(name: str) -> None:
    rendered = str(
        make(
            error_type(name),
            stage="extraction",
            path=Path("archive/alice/broken.pdf"),
        )
    )

    assert "something went wrong" in rendered
    assert "extraction" in rendered
    assert "broken.pdf" in rendered


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_stage_falls_back_to_the_type_s_own_stage(
    name: str, blank: str
) -> None:
    error_class = error_type(name)
    assert make(error_class, stage=blank).stage == error_class.default_stage


@pytest.mark.parametrize("name", ALL_NAMES)
def test_no_type_inherits_a_stage_that_names_nothing(name: str) -> None:
    error_class = error_type(name)
    stage = error_class.default_stage

    assert stage.strip()
    if error_class is not IngestError:
        assert "default_stage" in vars(error_class), (
            f"{name} does not declare its own default stage"
        )


@pytest.mark.parametrize(
    "name, stage",
    [
        ("DiscoveryError", "discovery"),
        ("ExtractionError", "extraction"),
        ("VisionError", "vision"),
        ("VisionUnavailable", "vision_unavailable"),
        ("StateError", "state"),
    ],
)
def test_each_type_names_the_stage_its_failure_belongs_to(
    name: str, stage: str
) -> None:
    assert error_type(name).default_stage == stage


def test_the_raising_error_is_an_ordinary_exception() -> None:
    """Nothing in this taxonomy validates its way into raising from a
    constructor: an error class that can fail to be built would replace a
    diagnosed failure with a confusing one."""
    error = IngestError("plain")
    assert isinstance(error, Exception)
    assert error.args == ("plain",)
    assert error.path is None
    assert error.stage == IngestError.default_stage


# --------------------------------------------------------------------------
# Boundary guard
# --------------------------------------------------------------------------


def test_errors_imports_nothing_from_this_project_but_types() -> None:
    """design.md, Architecture: types and errors are the leftmost layer.
    errors may import types and nothing else from this project."""
    imported: list[str] = []
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            assert node.level == 0, "a relative import is still a project import"

    project_imports: Callable[[str], bool] = lambda name: name.startswith("npu_rag")
    assert all(
        name == "npu_rag.ingest.types" for name in imported if project_imports(name)
    )
