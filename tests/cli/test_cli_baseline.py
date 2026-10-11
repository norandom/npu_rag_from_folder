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
    # No handler module may be imported by a path that did not invoke it. Since
    # task 4.1 this is load-bearing rather than future-proofing: `cmd_search.py`
    # exists and `build_parser` consults it for its flags, so an eager import -
    # registering every command's arguments instead of the one in `argv` - would
    # otherwise be silent, costing startup time without failing anything.
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


# --------------------------------------------------------------------------- #
# Lazy argument registration (task 4.1's amended boundary)
# --------------------------------------------------------------------------- #
#
# `build_parser` asks the handler module of the command named in `argv` - and of
# no other - to register its own flags. The property that has to hold is a
# conjunction: the flags must actually appear, AND the four modules that were
# not asked for must stay unimported, AND requirement 1.2's fast paths must
# still import none at all. Each half is pinned separately below, because
# either one alone is satisfiable by code that breaks the other: registering
# everything eagerly satisfies the first, and registering nothing satisfies the
# second.


def _stub_command_module(
    monkeypatch: pytest.MonkeyPatch, command: str, calls: list[str]
) -> None:
    """Plant a handler module for `command` that records its registration."""
    from npu_rag.cli.main import COMMAND_MODULES

    module_name = COMMAND_MODULES[command]
    stub = types.ModuleType(module_name)

    def add_arguments(parser: argparse.ArgumentParser) -> None:
        calls.append(command)
        parser.add_argument(f"--{command}-only", action="store_true")

    def run(args: argparse.Namespace) -> int:
        return 0

    setattr(stub, "add_arguments", add_arguments)
    setattr(stub, "run", run)
    monkeypatch.setitem(sys.modules, module_name, stub)


def test_arguments_hook_attribute_is_the_documented_name() -> None:
    """Pinned against a literal: `cmd_*` modules are written against this name."""
    from npu_rag.cli.main import ARGUMENTS_ATTR

    assert ARGUMENTS_ATTR == "add_arguments"


@pytest.mark.parametrize("command", EXPECTED_SUBCOMMANDS)
def test_only_the_command_named_in_argv_registers_its_arguments(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    from npu_rag.cli.main import build_parser

    calls: list[str] = []
    for name in EXPECTED_SUBCOMMANDS:
        _stub_command_module(monkeypatch, name, calls)

    build_parser([command, "whatever"])
    assert calls == [command]


def test_a_positional_that_names_another_command_is_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`npu-rag search index` searches for "index"; it does not run `index`."""
    from npu_rag.cli.main import build_parser

    calls: list[str] = []
    for name in EXPECTED_SUBCOMMANDS:
        _stub_command_module(monkeypatch, name, calls)

    build_parser(["search", "index", "--json"])
    assert calls == ["search"]


@pytest.mark.parametrize("argv", [[], ["--help"], ["--version"], ["no-such-command"]])
def test_the_fast_paths_register_no_arguments_at_all(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    from npu_rag.cli.main import build_parser

    calls: list[str] = []
    for name in EXPECTED_SUBCOMMANDS:
        _stub_command_module(monkeypatch, name, calls)

    build_parser(argv)
    assert calls == []


def test_build_parser_without_argv_registers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-argv form is the shape every structural test above uses."""
    from npu_rag.cli.main import build_parser

    calls: list[str] = []
    for name in EXPECTED_SUBCOMMANDS:
        _stub_command_module(monkeypatch, name, calls)

    build_parser()
    assert calls == []


def test_a_registered_flag_actually_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-vacuity control: the hook's effect is observable through parsing."""
    from npu_rag.cli.main import build_parser

    calls: list[str] = []
    _stub_command_module(monkeypatch, "status", calls)

    argv = ["status", "--status-only"]
    args = build_parser(argv).parse_args(argv)
    assert args.status_only is True
    assert args.command == "status"


def test_a_handler_without_the_hook_keeps_a_bare_subparser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 5.3 depends on this: `benchmark` must stay as it was."""
    from npu_rag.cli import main as main_module

    stub = types.ModuleType(main_module.COMMAND_MODULES["status"])
    setattr(stub, "run", lambda args: 0)
    monkeypatch.setitem(sys.modules, main_module.COMMAND_MODULES["status"], stub)

    argv = ["status"]
    assert main_module.build_parser(argv).parse_args(argv).command == "status"
    assert main_module.main(["status", "--not-a-flag"]) == main_module.EXIT_USAGE


def test_a_non_callable_arguments_hook_is_ignored_rather_than_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`add_arguments` present but not callable must not be invoked.

    The same shape was found by mutation against `HANDLER_ATTR` in task 1.1:
    `getattr` succeeds, so a check for `None` alone would call an integer and
    hand the user a `TypeError` traceback - here before parsing has even begun,
    which is the worst place for one.
    """
    from npu_rag.cli import main as main_module

    stub = types.ModuleType(main_module.COMMAND_MODULES["tags"])
    setattr(stub, "add_arguments", 42)
    setattr(stub, "run", lambda args: 0)
    monkeypatch.setitem(sys.modules, main_module.COMMAND_MODULES["tags"], stub)

    argv = ["tags"]
    assert main_module.build_parser(argv).parse_args(argv).command == "tags"
    assert main_module.main(argv) == 0


def test_a_genuine_import_error_during_registration_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration must not hide a missing dependency any more than dispatch does.

    Asserted against `build_parser` specifically, and not through `main`: with
    the guard removed, `dispatch` would raise the same error a moment later, so
    only the earlier call site can tell the two apart.
    """
    from npu_rag.cli import main as main_module

    def exploding_import(name: str) -> types.ModuleType:
        raise ModuleNotFoundError(
            "No module named 'some_absent_dependency'", name="some_absent_dependency"
        )

    monkeypatch.setattr(main_module, "import_module", exploding_import)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        main_module.build_parser(["search", "a query"])
    assert excinfo.value.name == "some_absent_dependency"


def test_an_absent_handler_module_during_registration_is_tolerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control for the test above: the handler's *own* absence is not an error
    here - `dispatch` owns that diagnostic, after parsing."""
    from npu_rag.cli import main as main_module

    def missing_import(name: str) -> types.ModuleType:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(main_module, "import_module", missing_import)
    argv = ["status"]
    parser = main_module.build_parser(argv)
    assert parser.parse_args(argv).command == "status"
    assert _registered_subcommands(parser) == EXPECTED_SUBCOMMANDS


def test_an_absent_handler_module_still_parses_and_then_exits_two() -> None:
    """`benchmark` has no module at all, so the hook must be a no-op for it.

    Task 5.3 is blocked and this is the honest behaviour it is blocked at: the
    diagnostic comes from `dispatch`, after parsing, and is a runtime error.
    """
    completed = _run_cli("benchmark")
    assert completed.returncode == 2
    assert "npu_rag.cli.cmd_benchmark" in completed.stderr
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr


def test_search_help_lists_the_commands_own_flags() -> None:
    """The user-visible consequence of the hook (design §3.5)."""
    completed = _run_cli("search", "--help")
    assert completed.returncode == 0
    for flag in ("--folder", "--tag", "--group-by-folder", "--top-k", "--provider"):
        assert flag in completed.stdout, f"{flag} missing from `search --help`"
    assert completed.stderr == ""


def test_search_help_imports_the_handler_but_nothing_heavy(tmp_path: Path) -> None:
    """Requirement 1.2 covers `npu-rag search --help` too.

    The handler module has to be imported for its flags to exist, so the
    guarantee for this path is narrower than for the fast paths and is stated
    separately: `cmd_search` is present, and `lancedb`, `onnxruntime`, `torch`
    and `transformers` are not.
    """
    payload = _run_probe(
        tmp_path,
        body=(
            "from npu_rag.cli.main import main\n"
            'payload["code"] = main(["search", "--help"])\n'
        ),
        name="search_help",
    )
    assert payload["code"] == 0
    assert payload["heavy"] == [], (
        f"`search --help` imported a forbidden module: {payload['heavy']}"
    )
    assert "npu_rag.cli.cmd_search" in payload["cli_modules"], (
        "the flags cannot have been registered without importing the handler"
    )
    assert "--top-k" in payload["stdout"]


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
