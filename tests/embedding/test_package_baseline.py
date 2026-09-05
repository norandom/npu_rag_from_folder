"""Baseline guards for the embedding package skeleton.

These assert the structural commitments made in design.md's "File Structure
Plan" and the dependency-direction rule from "Allowed Dependencies": the
embedding package sits at the outward end of the dependency chain and must not
import from any sibling ``npu_rag`` sub-package.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "embedding"

# design.md -> File Structure Plan -> Directory Structure
EXPECTED_SUBPACKAGES = ("environment", "models", "providers", "bench")


def test_embedding_package_is_importable() -> None:
    module = importlib.import_module("npu_rag.embedding")
    assert module.__name__ == "npu_rag.embedding"


@pytest.mark.parametrize("subpackage", EXPECTED_SUBPACKAGES)
def test_expected_subpackage_is_importable(subpackage: str) -> None:
    module = importlib.import_module(f"npu_rag.embedding.{subpackage}")
    assert module.__name__ == f"npu_rag.embedding.{subpackage}"


def test_package_is_marked_as_typed() -> None:
    assert (PACKAGE_ROOT.parent / "py.typed").is_file()


def _imported_module_names(source: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_embedding_package_imports_no_sibling_npu_rag_package() -> None:
    """design.md, Allowed Dependencies: this package must not import from any
    other ``npu_rag`` sub-package."""
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for name in _imported_module_names(path.read_text(encoding="utf-8")):
            if name == "npu_rag" or (
                name.startswith("npu_rag.")
                and not name.startswith("npu_rag.embedding")
            ):
                violations.append(f"{path}: {name}")
    assert violations == []
