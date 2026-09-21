"""Baseline guards for the `npu-rag` console entry point.

These pin the structural commitments of `search-cli` requirement 1 and of
design.md §1 (principle 5, Fast Startup), §2 (Component Layout), §3.5
(Subcommands) and §4 (Error Taxonomy & Exit Codes):

- `[project.scripts]` declares `npu-rag = "npu_rag.cli.main:main"` (req 1.1).
- `--help`, `--version` and an invalid invocation perform **zero heavy
  imports** - `onnxruntime`, `torch`, `lancedb`, `transformers` (req 1.2).
- Exactly the five subcommands `search`, `index`, `status`, `tags`,
  `benchmark` are registered (req 1.3).
- A subcommand's handler module is imported only when that subcommand runs.
- Exit codes are 0 success / 1 user-or-config error / 2 runtime error, and a
  user error never reaches the terminal as a traceback (design §4).

**On the "under 50 milliseconds" figure in requirement 1.2.** The load-bearing
assertion in this file is the *structural* one: no heavy module in
`sys.modules` after the entry point has run. A stopwatch assertion is an
assertion about the machine, not about this code - see the "Standing lesson"
section of `.kiro/specs/npu-embedding-runtime/tasks.md`, rules 6-8, where task
6.2 shipped a live timing test that held at idle and failed roughly three runs
in four under load. `test_help_wall_clock_is_only_a_smoke_bound` keeps a
deliberately generous timing check for reference and names itself as such;
a bound tight enough to police 50 ms would fail whenever someone else's build
is running, which teaches people to ignore the suite.

The two categories are kept apart on purpose:
- properties of the code -> everything except the wall-clock test;
- claims about the machine -> the wall-clock test alone, which no other
  assertion depends on.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
import tomllib
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
MAIN_SOURCE = REPO_ROOT / "src" / "npu_rag" / "cli" / "main.py"
CLI_INIT_SOURCE = REPO_ROOT / "src" / "npu_rag" / "cli" / "__init__.py"

SCRIPT_NAME = "npu-rag"
ENTRY_POINT = "npu_rag.cli.main:main"

# Requirement 1.2 names exactly these four as the imports that must stay out of
# the startup path. Spelled as the top-level module each distribution provides,
# which is what `name in sys.modules` sees.
HEAVY_MODULES: tuple[str, ...] = ("lancedb", "onnxruntime", "torch", "transformers")

# Requirement 1.3, and design.md §2 for the handler filenames. Written out as
# literals rather than derived from the code under test, so that the equality
# assertions below are a real comparison and not a restatement.
EXPECTED_SUBCOMMANDS: tuple[str, ...] = ("search", "index", "status", "tags", "benchmark")
EXPECTED_COMMAND_MODULES: dict[str, str] = {
    "search": "npu_rag.cli.cmd_search",
    "index": "npu_rag.cli.cmd_index",
    "status": "npu_rag.cli.cmd_status",
    "tags": "npu_rag.cli.cmd_tags",
    "benchmark": "npu_rag.cli.cmd_benchmark",
}

# A probe runs in a *fresh* interpreter, does something, and records which of the
# guarded modules ended up in `sys.modules`. The result travels through a file
# rather than stdout so that a chatty import cannot corrupt it.
_PROBE_TEMPLATE = '''\
import json
import sys
import time
from pathlib import Path

HEAVY = {heavy!r}
RESULT = Path({result!r})
payload = {{}}

{body}

payload["heavy"] = sorted(name for name in HEAVY if name in sys.modules)
payload["cli_modules"] = sorted(n for n in sys.modules if n.startswith("npu_rag.cli"))
RESULT.write_text(json.dumps(payload), encoding="utf-8")
'''


def _run_probe(tmp_path: Path, body: str, name: str = "probe") -> dict[str, Any]:
    """Execute `body` in a fresh interpreter and return its recorded payload."""
    result_path = tmp_path / f"{name}.json"
    script_path = tmp_path / f"{name}.py"
    script_path.write_text(
        _PROBE_TEMPLATE.format(
            heavy=HEAVY_MODULES,
            result=str(result_path),
            body=body,
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"probe exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    payload: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    payload["stdout"] = completed.stdout
    payload["stderr"] = completed.stderr
    return payload


def _run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI the way a user would, through `python -m`."""
    return subprocess.run(
        [sys.executable, "-m", "npu_rag.cli.main", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )


def _manifest() -> dict[str, Any]:
    data: dict[str, Any] = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data


def _module_scope_imports(source_path: Path) -> list[str]:
    """Names imported at module scope (any nesting depth is excluded)."""
    import ast

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` blocks are still module scope at runtime for
            # the purposes of this guard, so look inside them too.
            for inner in node.body:
                if isinstance(inner, ast.Import):
                    names.extend(alias.name for alias in inner.names)
                elif isinstance(inner, ast.ImportFrom) and inner.level == 0 and inner.module:
                    names.append(inner.module)
    return names


def _registered_subcommands(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    """The subcommand names argparse will actually accept, in registration order."""
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public accessor
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            choices = action.choices
            assert choices is not None, "subparsers action carries no choices"
            return tuple(str(choice) for choice in choices)
    raise AssertionError("the top-level parser registers no subparsers at all")


# --------------------------------------------------------------------------- #
# Requirement 1.1 - packaging and entry point
# --------------------------------------------------------------------------- #


def test_console_script_is_declared_in_pyproject() -> None:
    scripts = _manifest()["project"]["scripts"]
    assert scripts[SCRIPT_NAME] == ENTRY_POINT
    assert isinstance(scripts, dict)


def test_declared_entry_point_resolves_to_a_callable() -> None:
    """Bind the manifest string to real code, not to this file's constant."""
    declared: str = _manifest()["project"]["scripts"][SCRIPT_NAME]
    module_name, separator, attribute = declared.partition(":")
    assert separator == ":", f"{declared!r} is not a module:attribute entry point"
    module = importlib.import_module(module_name)
    target = getattr(module, attribute)
    assert callable(target)


def test_cli_package_does_not_shadow_the_main_submodule() -> None:
    """`npu_rag.cli.main` must mean the module, not a function of the same name.

    A `from npu_rag.cli.main import main` in the package `__init__` would bind
    `main` on the package to the function and shadow the submodule, so
    `from npu_rag.cli import main` would hand back a function - and which of
    the two wins would depend on import order. This pins the module.
    """
    package = importlib.import_module("npu_rag.cli")
    main_module = importlib.import_module("npu_rag.cli.main")
    assert isinstance(main_module, types.ModuleType)
    assert getattr(package, "main") is main_module
    assert callable(main_module.main)


# --------------------------------------------------------------------------- #
# Requirement 1.3 - the five subcommands
# --------------------------------------------------------------------------- #


def test_parser_registers_exactly_the_five_subcommands() -> None:
    from npu_rag.cli.main import build_parser

    registered = _registered_subcommands(build_parser())
    assert registered == EXPECTED_SUBCOMMANDS, (
        "argparse must accept exactly the five subcommands of requirement 1.3, "
        f"in that order; got {registered}"
    )


@pytest.mark.parametrize("name", EXPECTED_SUBCOMMANDS)
def test_each_subcommand_parses_and_is_recorded(name: str) -> None:
    from npu_rag.cli.main import build_parser

    args = build_parser().parse_args([name])
    assert args.command == name


def test_command_module_map_matches_the_designed_filenames() -> None:
    from npu_rag.cli.main import COMMAND_MODULES

    assert dict(COMMAND_MODULES) == EXPECTED_COMMAND_MODULES


def test_command_module_map_and_parser_agree() -> None:
    """The relation, not just the parts: dispatch can only reach what parses."""
    from npu_rag.cli.main import COMMAND_MODULES, build_parser

    assert tuple(COMMAND_MODULES) == _registered_subcommands(build_parser())


def test_help_names_every_subcommand() -> None:
    from npu_rag.cli.main import build_parser

    rendered = build_parser().format_help()
    missing = [name for name in EXPECTED_SUBCOMMANDS if name not in rendered]
    assert missing == []
    assert SCRIPT_NAME in rendered


def test_help_through_the_module_entry_exits_zero_on_stdout() -> None:
    completed = _run_cli("--help")
    assert completed.returncode == 0
    assert f"usage: {SCRIPT_NAME}" in completed.stdout
    for name in EXPECTED_SUBCOMMANDS:
        assert name in completed.stdout
    assert completed.stderr == ""


def test_version_reports_the_packaged_version_on_stdout() -> None:
    expected: str = _manifest()["project"]["version"]
    completed = _run_cli("--version")
    assert completed.returncode == 0
    assert completed.stdout.strip() == f"{SCRIPT_NAME} {expected}"
    assert completed.stderr == ""


# --------------------------------------------------------------------------- #
# Requirement 1.2 / design principle 5 - no heavy imports on the fast paths
# --------------------------------------------------------------------------- #


def test_fast_paths_import_no_heavy_module(tmp_path: Path) -> None:
    """The contract: `--help`, `--version`, bare and invalid invocations cost
    no more than argparse. This is the structural form of requirement 1.2."""
    payload = _run_probe(
        tmp_path,
        body=(
            "from npu_rag.cli.main import main\n"
            'payload["help"] = main(["--help"])\n'
            'payload["version"] = main(["--version"])\n'
            'payload["bare"] = main([])\n'
            'payload["invalid"] = main(["no-such-command"])\n'
        ),
        name="fast_paths",
    )
    assert payload["heavy"] == [], (
        "the startup path imported a module requirement 1.2 forbids: "
        f"{payload['heavy']}"
    )
    # No handler module may be imported by a path that did not invoke it. Today
    # no `cmd_*.py` exists, so this is future-proofing: once task 4.1 lands
    # `cmd_search.py`, an eager import in `build_parser()` would otherwise be
    # silent - it would cost startup time without failing anything.
    eager = [name for name in payload["cli_modules"] if name.startswith("npu_rag.cli.cmd_")]
    assert eager == [], f"a handler module was imported without being invoked: {eager}"
    # Non-vacuity control for the line above: the probe really does observe the
    # cli namespace, so an empty list means "none imported", not "never looked".
    assert "npu_rag.cli.main" in payload["cli_modules"]
    assert payload["help"] == 0
    assert payload["version"] == 0
    assert payload["bare"] == 1
    assert payload["invalid"] == 1


def test_importing_the_entry_module_alone_imports_no_heavy_module(tmp_path: Path) -> None:
    payload = _run_probe(
        tmp_path,
        body="import npu_rag.cli.main\n",
        name="import_only",
    )
    assert payload["heavy"] == []


def test_the_heavy_import_probe_can_see_a_real_heavy_import(tmp_path: Path) -> None:
    """Non-vacuity control for the two tests above.

    Without this, a typo in `HEAVY_MODULES` or a probe that never looked at
    `sys.modules` would make `payload["heavy"] == []` pass forever. `find_spec`
    proves the guarded names are the ones these distributions actually provide
    (without paying to import them); the real `import lancedb` proves the
    `sys.modules` check registers a heavy import when one happens.
    """
    payload = _run_probe(
        tmp_path,
        body=(
            "import importlib.util\n"
            'payload["resolvable"] = sorted(\n'
            "    name for name in HEAVY if importlib.util.find_spec(name) is not None\n"
            ")\n"
            "import lancedb  # noqa: F401 - deliberately heavy\n"
        ),
        name="control",
    )
    # lancedb, onnxruntime and transformers are declared project dependencies,
    # so they are always resolvable here; torch lives in the optional `export`
    # extra and may legitimately be absent.
    assert {"lancedb", "onnxruntime", "transformers"} <= set(payload["resolvable"])
    # The spelling is pinned against a literal, which looks redundant and is not.
    # A misspelled guard name is *unresolvable*, and unresolvable is exactly what
    # a legitimately absent optional dependency looks like - so no `find_spec`
    # check can tell "torhc" from "torch is not installed on this machine". An
    # attempt to do so was written, planted against, and failed: review's "torhc"
    # mutant passed all 31 tests, silently retiring one of the four modules
    # requirement 1.2 names. Comparing to a literal is the only check that
    # survives both a typo and a machine without the optional `export` extra.
    assert HEAVY_MODULES == ("lancedb", "onnxruntime", "torch", "transformers")
    assert payload["heavy"] == ["lancedb"]


def test_main_module_has_no_forbidden_module_scope_import() -> None:
    """Read the source, not the runtime: catches an eager import that happens
    to be unreachable in the paths the probes exercise."""
    imported = _module_scope_imports(MAIN_SOURCE) + _module_scope_imports(CLI_INIT_SOURCE)
    forbidden = [
        name
        for name in imported
        if name.split(".")[0] in HEAVY_MODULES or name.startswith("npu_rag.cli.cmd_")
    ]
    assert forbidden == [], f"module-scope imports must stay light: {forbidden}"


def test_help_wall_clock_is_only_a_smoke_bound() -> None:
    """A SMOKE BOUND, NOT THE CONTRACT.

    Requirement 1.2 says "under 50 milliseconds", but a subprocess pays CPython
    interpreter startup before any of this code runs, so a subprocess stopwatch
    cannot measure the 50 ms claim even in principle - and an in-process
    stopwatch measures the machine's current load at least as much as it
    measures this module. The bound below is deliberately loose: it exists to
    catch a catastrophic regression (something dragging in a multi-second
    import) and nothing finer. The real guard is
    `test_fast_paths_import_no_heavy_module`.
    """
    started = time.perf_counter()
    completed = _run_cli("--help")
    elapsed = time.perf_counter() - started
    assert completed.returncode == 0
    assert elapsed < 10.0, f"`--help` took {elapsed:.3f}s, which is pathological"


# --------------------------------------------------------------------------- #
# Lazy dispatch
# --------------------------------------------------------------------------- #


def test_handler_module_is_imported_only_when_its_subcommand_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the real dispatch path with a stand-in handler module.

    `import_module` consults `sys.modules` first, so a module planted there is
    exactly what the shipped code would receive from a real `cmd_tags.py`. The
    `--help` leg is the discriminator: it proves the handler is not reached
    when the subcommand was not asked for.
    """
    from npu_rag.cli import main as main_module

    seen: list[argparse.Namespace] = []

    def run(args: argparse.Namespace) -> int:
        seen.append(args)
        return 0

    stub = types.ModuleType("npu_rag.cli.cmd_tags")
    setattr(stub, "run", run)
    monkeypatch.setitem(sys.modules, "npu_rag.cli.cmd_tags", stub)

    assert main_module.main(["--help"]) == 0
    assert seen == [], "--help must not reach a command handler"

    assert main_module.main(["tags"]) == 0
    assert len(seen) == 1
    assert seen[0].command == "tags"


def test_missing_handler_module_is_a_runtime_error_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from npu_rag.cli import main as main_module

    monkeypatch.setattr(
        main_module,
        "COMMAND_MODULES",
        {"tags": "npu_rag.cli.cmd_absent_by_design"},
    )
    assert main_module.main(["tags"]) == main_module.EXIT_RUNTIME
    captured = capsys.readouterr()
    assert "npu_rag.cli.cmd_absent_by_design" in captured.err
    assert captured.out == ""


def test_handler_without_a_run_callable_is_a_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from npu_rag.cli import main as main_module

    stub = types.ModuleType("npu_rag.cli.cmd_status")
    monkeypatch.setitem(sys.modules, "npu_rag.cli.cmd_status", stub)
    assert main_module.main(["status"]) == main_module.EXIT_RUNTIME
    assert "run" in capsys.readouterr().err


def test_handler_whose_run_is_not_callable_is_a_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`run` present but not callable must be diagnosed, not called.

    This is a different case from `run` being absent, and the difference is
    load-bearing: `getattr` succeeds here, so dispatch code that only checks
    for `None` would call an integer and hand the user a `TypeError`
    traceback. Found by mutation - dropping the `callable()` check survived
    every other test in this file, because the missing-attribute case reaches
    the same branch either way.
    """
    from npu_rag.cli import main as main_module

    stub = types.ModuleType("npu_rag.cli.cmd_search")
    setattr(stub, "run", 42)
    monkeypatch.setitem(sys.modules, "npu_rag.cli.cmd_search", stub)

    assert main_module.main(["search"]) == main_module.EXIT_RUNTIME
    captured = capsys.readouterr()
    assert "run" in captured.err
    assert captured.out == ""


def test_a_genuine_import_error_inside_a_handler_is_not_disguised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "not available" branch must key on the handler module itself.

    Swallowing every `ModuleNotFoundError` would turn a real missing dependency
    deep inside a command into the misleading message "this command is not
    available".
    """
    from npu_rag.cli import main as main_module

    def exploding_import(name: str) -> types.ModuleType:
        raise ModuleNotFoundError("No module named 'some_absent_dependency'", name="some_absent_dependency")

    monkeypatch.setattr(main_module, "import_module", exploding_import)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        main_module.main(["search"])
    assert excinfo.value.name == "some_absent_dependency"


# --------------------------------------------------------------------------- #
# Design §4 - error taxonomy and exit codes
# --------------------------------------------------------------------------- #


def test_exit_code_constants_match_the_design_taxonomy() -> None:
    from npu_rag.cli.main import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE

    assert (EXIT_OK, EXIT_USAGE, EXIT_RUNTIME) == (0, 1, 2)


def test_bare_invocation_prints_help_to_stderr_and_exits_one() -> None:
    completed = _run_cli()
    assert completed.returncode == 1
    assert f"usage: {SCRIPT_NAME}" in completed.stderr
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ("no-such-command",),
        ("--no-such-flag",),
        ("search", "--no-such-flag"),
    ],
)
def test_user_errors_exit_one_with_a_diagnostic_and_no_traceback(argv: tuple[str, ...]) -> None:
    """Design §4 reserves exit 2 for hardware and runtime failure, so a typo
    must not land there - which is argparse's own default and the mistake this
    test exists to catch.

    The diagnostic is asserted by shape, not merely by being non-empty: an
    absent package also exits 1 with something on stderr, and during this
    task's RED phase that is exactly how these three cases passed vacuously.
    """
    completed = _run_cli(*argv)
    assert completed.returncode == 1, f"{argv} exited {completed.returncode}"
    assert f"usage: {SCRIPT_NAME}" in completed.stderr
    assert "error:" in completed.stderr
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr


def test_keyboard_interrupt_in_a_handler_is_a_runtime_exit_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from npu_rag.cli import main as main_module

    def run(args: argparse.Namespace) -> int:
        raise KeyboardInterrupt

    stub = types.ModuleType("npu_rag.cli.cmd_index")
    setattr(stub, "run", run)
    monkeypatch.setitem(sys.modules, "npu_rag.cli.cmd_index", stub)

    assert main_module.main(["index"]) == main_module.EXIT_RUNTIME
    assert capsys.readouterr().err.strip() != ""


def test_handler_return_code_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-vacuity control for the dispatch tests: the code the handler returns
    is the code `main` returns, so a `return 0` hardcoded in dispatch dies."""
    from npu_rag.cli import main as main_module

    def run(args: argparse.Namespace) -> int:
        return 7

    stub = types.ModuleType("npu_rag.cli.cmd_benchmark")
    setattr(stub, "run", run)
    monkeypatch.setitem(sys.modules, "npu_rag.cli.cmd_benchmark", stub)

    assert main_module.main(["benchmark"]) == 7
