"""Baseline guards for the ingest package skeleton.

These assert the structural commitments made in design.md's "File Structure
Plan" -> Directory Structure and "Modified Files": the ingest package exists as
a sibling of ``npu_rag.embedding`` with an empty public surface, its third-party
dependencies are declared in the default set, and the default state-file path
is gitignored.

The package-wide layer guard (task 1.3) walks every on-disk module, enforces
leftward-only imports against the seeded rank table, and asserts by name that
only ``vision.py`` imports ``httpx`` and that ``extract/*`` never imports
``vision`` or ``state``.
"""

from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

import pytest

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


# --------------------------------------------------------------------------
# Dependency direction (design.md, Architecture)
#
# Seeded with the full order before the later modules exist, so subsequent
# tasks add files without editing this table. Rank of a file is
# relative.parts[0] when it lives in a subpackage, else relative.stem.
# --------------------------------------------------------------------------

PACKAGE_ROOT = REPO_ROOT / "src" / "npu_rag" / "ingest"
MODULE_PACKAGE = "npu_rag.ingest"

#: "types, errors -> config -> credential -> identity -> state -> discover ->
#: route -> extract -> vision -> chunk, report -> pipeline". Index is the
#: layer's position; a module in one layer may import only from the same
#: rank or a strictly lower one.
LAYER_ORDER: tuple[tuple[str, ...], ...] = (
    ("types", "errors"),
    ("config",),
    ("credential",),
    ("identity",),
    ("state",),
    ("discover",),
    ("route",),
    ("extract",),
    ("vision",),
    ("chunk", "report"),
    ("pipeline",),
)

LAYER_OF = {
    name: rank for rank, names in enumerate(LAYER_ORDER) for name in names
}

_DECOY_TOP_LEVEL = PACKAGE_ROOT / "decoy.py"
_DECOY_EXTRACT = PACKAGE_ROOT / "extract" / "decoy.py"


def imported_names(source_text: str, package: str) -> list[str]:
    """Every imported name as an absolute dotted path, relatives resolved."""
    parts = package.split(".")
    names: list[str] = []
    for node in ast.walk(ast.parse(source_text)):
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


def _relative(path: Path) -> Path:
    return path.relative_to(PACKAGE_ROOT)


def _own_rank_name(relative: Path) -> str:
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


def _iter_package_modules() -> list[Path]:
    return [
        path
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if "__pycache__" not in _relative(path).parts
    ]


def _containing_package(relative: Path) -> str:
    return ".".join((MODULE_PACKAGE, *relative.parts[:-1]))


def _imports_named(names: list[str], qualified: str) -> bool:
    return any(name == qualified or name.startswith(f"{qualified}.") for name in names)


def collect_rank_issues(
    layer_of: dict[str, int] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(unplaced, rank_violations)`` for every on-disk module.

    An on-disk module absent from the rank table is a failure, not a skip.
    Relative imports resolve against the containing package, not the root.
    """
    mapping = LAYER_OF if layer_of is None else layer_of
    unplaced: list[str] = []
    violations: list[str] = []
    for path in _iter_package_modules():
        relative = _relative(path)
        own = _own_rank_name(relative)
        if own == "__init__":
            continue
        posix = relative.as_posix()
        if own not in mapping:
            unplaced.append(posix)
            continue
        containing = _containing_package(relative)
        for name in imported_names(path.read_text(encoding="utf-8"), containing):
            if not name.startswith(f"{MODULE_PACKAGE}."):
                continue
            rest = name[len(MODULE_PACKAGE) + 1 :]
            if rest == "":
                continue
            target = rest.split(".")[0]
            rank = mapping.get(target)
            if rank is not None and rank > mapping[own]:
                violations.append(f"{posix}: imports {name}")
    return unplaced, violations


def httpx_import_offenders() -> list[str]:
    """Modules other than ``vision.py`` that import ``httpx``."""
    offenders: list[str] = []
    for path in _iter_package_modules():
        relative = _relative(path)
        containing = _containing_package(relative)
        names = imported_names(path.read_text(encoding="utf-8"), containing)
        if not _imports_named(names, "httpx"):
            continue
        posix = relative.as_posix()
        if posix != "vision.py":
            offenders.append(f"{posix}: imports httpx")
    return offenders


def extract_forbidden_imports() -> list[str]:
    """``extract/*`` files that import the vision or state modules."""
    hits: list[str] = []
    extract_root = PACKAGE_ROOT / "extract"
    if not extract_root.is_dir():
        return hits
    for path in sorted(extract_root.rglob("*.py")):
        relative = _relative(path)
        if "__pycache__" in relative.parts:
            continue
        containing = _containing_package(relative)
        names = imported_names(path.read_text(encoding="utf-8"), containing)
        posix = relative.as_posix()
        for target in ("vision", "state"):
            qualified = f"{MODULE_PACKAGE}.{target}"
            if _imports_named(names, qualified):
                imported = [
                    name
                    for name in names
                    if name == qualified or name.startswith(f"{qualified}.")
                ]
                for name in imported:
                    hits.append(f"{posix}: imports {name}")
    return hits


def _remove_decoy(path: Path) -> None:
    path.unlink(missing_ok=True)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for leftover in cache.glob(f"{path.stem}*"):
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
        try:
            next(cache.iterdir())
        except StopIteration:
            cache.rmdir()
        except OSError:
            pass
    parent = path.parent
    while parent != PACKAGE_ROOT:
        cache = parent / "__pycache__"
        if cache.is_dir():
            try:
                cache.rmdir()
            except OSError:
                break
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def test_layer_order_is_seeded_with_the_design_dependency_direction() -> None:
    """design.md, Architecture: the full left-to-right order, seeded up front."""
    assert LAYER_ORDER == (
        ("types", "errors"),
        ("config",),
        ("credential",),
        ("identity",),
        ("state",),
        ("discover",),
        ("route",),
        ("extract",),
        ("vision",),
        ("chunk", "report"),
        ("pipeline",),
    )
    names = [name for group in LAYER_ORDER for name in group]
    assert len(names) == len(set(names))


def test_every_module_in_the_package_respects_the_layer_order() -> None:
    """Walk every on-disk module. An unplaced file is a failure, not a skip."""
    unplaced, violations = collect_rank_issues()
    assert unplaced == [], (
        f"these modules have no entry in LAYER_ORDER and are therefore invisible "
        f"to this guard, as importer and as target: {unplaced}. Add each at its "
        f"correct rank in the dependency direction rather than deleting this check."
    )
    assert violations == []


def test_the_package_wide_layer_guard_is_not_vacuous() -> None:
    """The walk sees the modules that already exist at the first rank."""
    scanned = [
        _relative(path).as_posix()
        for path in _iter_package_modules()
        if _own_rank_name(_relative(path)) in LAYER_OF
    ]
    assert "types.py" in scanned
    assert "errors.py" in scanned


def test_only_the_vision_module_imports_httpx() -> None:
    """design.md, Allowed Dependencies: no module other than vision.py imports httpx."""
    assert httpx_import_offenders() == []


def test_extract_modules_never_import_vision_or_state() -> None:
    """design.md, Testing Strategy Guard: extract/* never imports vision or state.

    Rank inheritance would allow extract -> state (state sits to the left) and
    would miss a same-rank intra-subpackage alias, so this is a name check.
    """
    assert extract_forbidden_imports() == []


@pytest.mark.parametrize(
    "statement",
    [
        "import npu_rag.ingest.pipeline",
        "from npu_rag.ingest.pipeline import IngestPipeline",
        "from npu_rag.ingest import pipeline",
        "from .. import pipeline",
        "from ..pipeline import IngestPipeline",
    ],
)
def test_the_layer_guard_recognises_every_import_spelling(statement: str) -> None:
    """Relative imports resolve against the containing package, not the root.

    Spellings are written as ``extract/*.py`` would have to write them: one
    dot is the extract subpackage, so reaching ``pipeline`` takes two.
    """
    resolved = imported_names(statement, f"{MODULE_PACKAGE}.extract")

    assert any(
        name == f"{MODULE_PACKAGE}.pipeline"
        or name.startswith(f"{MODULE_PACKAGE}.pipeline.")
        for name in resolved
    ), f"{statement!r} resolved to {resolved}"


def test_a_top_level_decoy_with_an_upward_import_fails_the_guard_by_name() -> None:
    """Observable 1: a planted top-level upward import is named in the failure."""
    try:
        _DECOY_TOP_LEVEL.write_text(
            "from npu_rag.ingest.pipeline import run_ingest\nimport httpx\n",
            encoding="utf-8",
        )
        unplaced_absent, _ = collect_rank_issues()
        assert "decoy.py" in unplaced_absent, (
            f"on-disk module absent from LAYER_ORDER was not named: {unplaced_absent!r}"
        )
        placed = {**LAYER_OF, "decoy": LAYER_OF["types"]}
        unplaced, violations = collect_rank_issues(placed)
        named = [item for item in unplaced + violations if "decoy.py" in item]
        assert named, (
            f"top-level decoy path was not named in the guard failure: "
            f"unplaced={unplaced!r} violations={violations!r}"
        )
        assert any("pipeline" in item for item in violations), (
            f"upward import of pipeline was not reported: {violations!r}"
        )
        httpx_hits = httpx_import_offenders()
        assert any("decoy.py" in item for item in httpx_hits), (
            f"non-vision httpx import was not named: {httpx_hits!r}"
        )
    finally:
        _remove_decoy(_DECOY_TOP_LEVEL)


def test_an_extract_decoy_importing_vision_fails_the_name_based_assertion() -> None:
    """Observable 2: extract/* importing vision fails the name-based assertion."""
    try:
        _DECOY_EXTRACT.parent.mkdir(parents=True, exist_ok=True)
        _DECOY_EXTRACT.write_text(
            "from npu_rag.ingest import vision\nfrom npu_rag.ingest import state\n",
            encoding="utf-8",
        )
        hits = extract_forbidden_imports()
        named = [item for item in hits if "decoy.py" in item]
        assert named, (
            f"extract decoy path was not named in the name-based assertion: {hits!r}"
        )
        assert any("vision" in item for item in named), (
            f"extract decoy importing vision was not reported: {hits!r}"
        )
        assert any("state" in item for item in named), (
            f"extract decoy importing state was not reported: {hits!r}"
        )
        _, rank_violations = collect_rank_issues()
        assert any("vision" in item for item in rank_violations)
        assert not any("state" in item for item in rank_violations), (
            f"rank check cannot see extract->state (state is leftward); "
            f"name-based assertion must carry that case: {rank_violations!r}"
        )
    finally:
        _remove_decoy(_DECOY_EXTRACT)


def test_removing_both_decoys_restores_green() -> None:
    """Observable 3: both decoys fail while planted; removing them restores green."""
    try:
        _DECOY_EXTRACT.parent.mkdir(parents=True, exist_ok=True)
        _DECOY_TOP_LEVEL.write_text(
            "from npu_rag.ingest.pipeline import run_ingest\nimport httpx\n",
            encoding="utf-8",
        )
        _DECOY_EXTRACT.write_text(
            "from npu_rag.ingest import vision\nfrom npu_rag.ingest import state\n",
            encoding="utf-8",
        )
        placed = {**LAYER_OF, "decoy": LAYER_OF["types"]}
        unplaced, violations = collect_rank_issues(placed)
        assert any(
            item == "decoy.py" or item.startswith("decoy.py:")
            for item in unplaced + violations
        ), (
            f"top-level decoy was not named: unplaced={unplaced!r} "
            f"violations={violations!r}"
        )
        extract_hits = extract_forbidden_imports()
        assert any("decoy.py" in item and "vision" in item for item in extract_hits), (
            f"extract decoy was not named by the name-based assertion: {extract_hits!r}"
        )
        assert any("decoy.py" in item for item in httpx_import_offenders())
    finally:
        _remove_decoy(_DECOY_TOP_LEVEL)
        _remove_decoy(_DECOY_EXTRACT)

    unplaced, violations = collect_rank_issues()
    assert unplaced == []
    assert violations == []
    assert extract_forbidden_imports() == []
    assert httpx_import_offenders() == []
    assert not _DECOY_TOP_LEVEL.exists()
    assert not _DECOY_EXTRACT.exists()
