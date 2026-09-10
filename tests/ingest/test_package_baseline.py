"""Baseline guards for the ingest package skeleton.

These assert the structural commitments made in design.md's "File Structure
Plan" -> Directory Structure and "Modified Files": the ingest package exists as
a sibling of ``npu_rag.embedding`` with an empty public surface, its third-party
dependencies are declared in the default set, and the default state-file path
is gitignored.

The package-wide layer guard is task 1.3 and is not seeded here.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
GITIGNORE = REPO_ROOT / ".gitignore"

# design.md -> Allowed Dependencies / Technology Stack / Modified Files
REQUIRED_DEFAULT_PACKAGES = (
    "openpyxl",
    "pypdfium2",
    "pillow",
    "httpx",
    "markdown-it-py",
)

# design.md -> Technology Stack (pillow has no version there)
TECHNOLOGY_STACK_VERSIONS = {
    "markdown-it-py": "4.2",
    "httpx": "0.28",
    "openpyxl": "3.1.5",
    "pypdfium2": "5.13",
}

DEFAULT_STATE_PATH = ".npu_rag/ingest.sqlite"


def _requirement_name(spec: str) -> str:
    token = spec.strip()
    for marker in (";", "["):
        token = token.split(marker, 1)[0]
    for separator in (">=", "<=", "==", "~=", "!=", ">", "<", "="):
        token = token.split(separator, 1)[0]
    return token.strip().lower()


def _project_dependencies() -> list[str]:
    manifest = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return list(manifest["project"]["dependencies"])


def test_ingest_package_is_importable() -> None:
    module = importlib.import_module("npu_rag.ingest")
    assert module.__name__ == "npu_rag.ingest"


def test_ingest_public_surface_is_empty() -> None:
    module = importlib.import_module("npu_rag.ingest")
    assert module.__all__ == []


def test_ingest_third_party_dependencies_are_declared_in_the_default_set() -> None:
    """design.md, Allowed Dependencies and Modified Files: five packages in
    ``[project] dependencies``, no new group."""
    manifest = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    declared_names = {_requirement_name(spec) for spec in manifest["project"]["dependencies"]}
    missing = set(REQUIRED_DEFAULT_PACKAGES) - declared_names
    assert missing == set()
    assert "ingest" not in manifest.get("dependency-groups", {})
    assert "ingest" not in manifest.get("project", {}).get("optional-dependencies", {})


def test_ingest_dependency_versions_match_technology_stack() -> None:
    """design.md, Technology Stack: declared specifiers carry the named versions."""
    specs_by_name = {
        _requirement_name(spec): spec for spec in _project_dependencies()
    }
    for package, version in TECHNOLOGY_STACK_VERSIONS.items():
        assert package in specs_by_name, f"{package} is not in [project] dependencies"
        assert version in specs_by_name[package], (
            f"{package} specifier {specs_by_name[package]!r} does not include {version}"
        )


def test_default_state_path_is_gitignored() -> None:
    """design.md, Modified Files: default state path ``.npu_rag/ingest.sqlite``."""
    lines = [
        line.strip()
        for line in GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert DEFAULT_STATE_PATH in lines
