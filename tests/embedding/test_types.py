"""Unit tests for the shared domain vocabulary (task 2.1).

``types.py`` is the root of design.md's dependency direction ("types, errors ->
reporting -> profiles -> environment -> models -> providers -> service ->
bench"), so two things are asserted here that no later test can assert for it:
that it imports nothing from this project at all, and that the environment layer
now takes ``ExecutionMode``, ``Condition`` and ``CapabilityReport`` *from* it
rather than defining its own.

That relocation closes the drift recorded against task 1.5: design.md's File
Structure Plan lists ``CapabilityReport`` under ``types.py`` while the
CapabilityChecker section sketches it inline, and 1.5 could not create
``types.py`` because this task owns it.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from npu_rag.embedding import types
from npu_rag.embedding.environment import capability
from npu_rag.embedding.types import (
    CONDITION_PROVIDER_REGISTERED,
    CapabilityReport,
    Condition,
    DocumentText,
    ExecutionMode,
    ProviderChoice,
    TextKind,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "npu_rag"
    / "embedding"
    / "types.py"
)


# --------------------------------------------------------------------------
# The declared vocabularies
# --------------------------------------------------------------------------


def test_text_kind_offers_exactly_document_and_query() -> None:
    """Requirement 3.2: every request declares corpus content or search query,
    and requirement 3.3 forbids a default - so there is no third member and no
    "unspecified" escape hatch to fall back on."""
    assert {kind.value for kind in TextKind} == {"document", "query"}


def test_provider_choice_offers_exactly_npu_cpu_and_auto() -> None:
    """Requirement 2.1 names these three and only these three."""
    assert {choice.value for choice in ProviderChoice} == {"npu", "cpu", "auto"}


def test_execution_mode_offers_exactly_the_three_verdicts() -> None:
    """Requirement 1.5: reachable in-process, reachable only in isolation, or
    not reachable at all."""
    assert {mode.value for mode in ExecutionMode} == {
        "in_process",
        "isolated",
        "unavailable",
    }


@pytest.mark.parametrize(
    "member, text",
    [
        (TextKind.DOCUMENT, "document"),
        (TextKind.QUERY, "query"),
        (ProviderChoice.NPU, "npu"),
        (ProviderChoice.CPU, "cpu"),
        (ProviderChoice.AUTO, "auto"),
        (ExecutionMode.IN_PROCESS, "in_process"),
        (ExecutionMode.ISOLATED, "isolated"),
        (ExecutionMode.UNAVAILABLE, "unavailable"),
    ],
)
def test_every_vocabulary_member_is_its_own_wire_text(member: str, text: str) -> None:
    """These values are written into manifests and benchmark records, so the
    member and its serialized form must not be able to drift apart."""
    assert member == text
    assert f"{member}" == text


@pytest.mark.parametrize(
    "vocabulary, text",
    [(TextKind, "query"), (ProviderChoice, "auto"), (ExecutionMode, "isolated")],
)
def test_every_vocabulary_reconstructs_from_its_text(
    vocabulary: type[TextKind] | type[ProviderChoice] | type[ExecutionMode],
    text: str,
) -> None:
    assert vocabulary(text).value == text


# --------------------------------------------------------------------------
# DocumentText: the title slot a bare string cannot carry
# --------------------------------------------------------------------------


def test_document_text_carries_content_and_an_optional_title() -> None:
    """design.md, EmbeddingService: the document template has a real title slot
    (``title: {title} | text: {content}``), which is why documents are not
    passed as bare strings (3.4)."""
    titled = DocumentText(content="body text", title="A Title")
    assert (titled.content, titled.title) == ("body text", "A Title")

    untitled = DocumentText(content="body text")
    assert untitled.title is None


def test_document_text_is_a_frozen_value_object() -> None:
    document = DocumentText(content="body text")
    with pytest.raises(dataclasses.FrozenInstanceError):
        document.content = "something else"  # type: ignore[misc]


def test_document_text_compares_by_value() -> None:
    assert DocumentText("body", "Title") == DocumentText("body", "Title")
    assert DocumentText("body", "Title") != DocumentText("body", None)


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n  "])
def test_document_text_rejects_a_blank_title(blank: str) -> None:
    """A blank title is the ambiguous third state. The document template
    renders an absent title as the literal ``none`` sentinel, so "no title"
    must be spelled ``None``; a blank string would render as an empty slot and
    silently change the embedded text (3.4)."""
    with pytest.raises(ValueError, match="title"):
        DocumentText(content="body text", title=blank)


def test_document_text_keeps_a_real_title_verbatim() -> None:
    """Titles are not trimmed, cased, or otherwise rewritten here: the template
    substitution belongs to the profile (task 2.2), not to the value object."""
    assert DocumentText("body", " Spaced Title ").title == " Spaced Title "


# --------------------------------------------------------------------------
# The relocation from environment/capability.py
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["ExecutionMode", "Condition", "CapabilityReport"]
)
def test_relocated_type_is_defined_in_types(name: str) -> None:
    """design.md, File Structure Plan: ``types.py`` holds ``CapabilityReport``;
    it and its two companions were parked in ``capability.py`` only because
    task 1.5 could not create this module."""
    assert getattr(types, name).__module__ == "npu_rag.embedding.types"


@pytest.mark.parametrize(
    "name", ["ExecutionMode", "Condition", "CapabilityReport"]
)
def test_capability_exposes_the_same_object_it_used_to_define(name: str) -> None:
    """The environment layer keeps re-exporting these, so every existing
    importer keeps working and there is exactly one class per concept - not a
    copy on each side of the move."""
    assert getattr(capability, name) is getattr(types, name)


def test_the_registration_condition_name_travels_with_the_report() -> None:
    """``CapabilityReport``'s invariant keys on this name (design.md, Domain
    Model: ``execution_mode`` is ``IN_PROCESS`` only when EP registration
    succeeded), so the constant has to live beside the invariant that reads it.
    The remaining four condition names stay with the checker that produces
    them."""
    assert CONDITION_PROVIDER_REGISTERED == "execution_provider_registered"
    assert capability.CONDITION_PROVIDER_REGISTERED is CONDITION_PROVIDER_REGISTERED


def test_capability_defines_none_of_the_relocated_names_itself() -> None:
    """A re-export is one class seen from two places; a second definition with
    the same name is two classes that will diverge. Identity assertions cannot
    tell those apart for a plain string constant - CPython interns
    identifier-like literals - so the guard is structural."""
    capability_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "npu_rag"
        / "embedding"
        / "environment"
        / "capability.py"
    )
    relocated = {"ExecutionMode", "Condition", "CapabilityReport"}
    relocated_constant = "CONDITION_PROVIDER_REGISTERED"

    defined: list[str] = []
    for node in ast.parse(capability_path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name in relocated:
            defined.append(node.name)
        elif isinstance(node, ast.Assign):
            defined.extend(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id == relocated_constant
            )

    assert defined == []


def test_relocated_capability_report_still_enforces_its_invariant() -> None:
    """The move must not have cost the invariant. An ``IN_PROCESS`` verdict
    without a satisfied registration condition is still a programming error."""
    with pytest.raises(ValueError, match=CONDITION_PROVIDER_REGISTERED):
        CapabilityReport(
            conditions=(
                Condition(
                    name=CONDITION_PROVIDER_REGISTERED,
                    satisfied=False,
                    observed="absent",
                    required="present",
                    remediation="provision the vendor runtime",
                ),
            ),
            execution_mode=ExecutionMode.IN_PROCESS,
            driver_version=None,
            runtime_version=None,
            device_name=None,
            power_reporting_supported=False,
        )


# --------------------------------------------------------------------------
# Boundary guard
# --------------------------------------------------------------------------


def test_types_imports_nothing_from_this_project() -> None:
    """design.md, Architecture: ``types`` is the leftmost layer. Anything it
    imported from ``npu_rag`` would be a cycle by construction."""
    imported: list[str] = []
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            assert node.level == 0, "a relative import is still a project import"

    assert [name for name in imported if name.startswith("npu_rag")] == []
