"""Baseline guards for the vector-index package skeleton.

These assert the structural commitments made in design.md:
- The index package exists under ``npu_rag.index``
- ``lancedb>=0.38.0`` is declared in project dependencies
- Upstream packages (``npu_rag.embedding`` and ``npu_rag.ingest``) do not import ``npu_rag.index``
- Layer-order within ``npu_rag.index`` is respected
"""

from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC_ROOT = REPO_ROOT / "src" / "npu_rag"
INDEX_ROOT = SRC_ROOT / "index"
EMBEDDING_ROOT = SRC_ROOT / "embedding"
INGEST_ROOT = SRC_ROOT / "ingest"

# Layer order within npu_rag.index:
# Leftward only: higher rank can import from lower rank, not vice versa.
LAYER_ORDER: tuple[tuple[str, ...], ...] = (
    ("types", "errors"),
    ("schema",),
    ("store",),
    ("retriever", "fusion", "diversity", "expansion", "rerank", "compat"),
    ("sync",),
    ("service",),
)

LAYER_OF: dict[str, int] = {}
for rank, group in enumerate(LAYER_ORDER):
    for module_name in group:
        LAYER_OF[module_name] = rank


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


def _imported_module_names(source: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_index_package_is_importable() -> None:
    module = importlib.import_module("npu_rag.index")
    assert module.__name__ == "npu_rag.index"


def test_lancedb_is_declared_in_dependencies() -> None:
    deps = _project_dependencies()
    dep_names = {_requirement_name(d) for d in deps}
    assert "lancedb" in dep_names
    lancedb_spec = next(d for d in deps if _requirement_name(d) == "lancedb")
    assert ">=0.38" in lancedb_spec or ">=0.38.0" in lancedb_spec


def test_embedding_package_does_not_import_index() -> None:
    violations: list[str] = []
    for path in sorted(EMBEDDING_ROOT.rglob("*.py")):
        for name in _imported_module_names(path.read_text(encoding="utf-8")):
            if name == "npu_rag.index" or name.startswith("npu_rag.index."):
                violations.append(f"{path.name}: {name}")
    assert violations == [], f"Embedding package must not import index: {violations}"


def test_ingest_package_does_not_import_index() -> None:
    violations: list[str] = []
    for path in sorted(INGEST_ROOT.rglob("*.py")):
        for name in _imported_module_names(path.read_text(encoding="utf-8")):
            if name == "npu_rag.index" or name.startswith("npu_rag.index."):
                violations.append(f"{path.name}: {name}")
    assert violations == [], f"Ingest package must not import index: {violations}"


def _get_module_rank(mod_name: str) -> int | None:
    base = mod_name.split(".")[-1]
    return LAYER_OF.get(base)


def test_index_layer_order_is_consistent() -> None:
    """Check that layer order within npu_rag.index is acyclic and respects ranks."""
    violations: list[str] = []
    for path in sorted(INDEX_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        mod_stem = path.stem
        importer_rank = LAYER_OF.get(mod_stem)
        if importer_rank is None:
            continue
        for name in _imported_module_names(path.read_text(encoding="utf-8")):
            if name.startswith("npu_rag.index."):
                imported_stem = name.split(".")[-1]
                imported_rank = LAYER_OF.get(imported_stem)
                if imported_rank is not None and imported_rank > importer_rank:
                    violations.append(
                        f"{path.name} (rank {importer_rank}) imports {name} (rank {imported_rank})"
                    )
    assert violations == []
