"""`npu-rag` console entry point: argument parsing and lazy command dispatch.

Design anchors: `.kiro/specs/search-cli/design.md` §1 principle 5 (Fast
Startup), §2 (Component Layout), §3.5 (Subcommands) and §4 (Error Taxonomy &
Exit Codes); requirements 1.1, 1.2 and 1.3.

Two rules govern this module.

**Nothing heavy at module scope.** `onnxruntime`, `torch`, `transformers` and
`lancedb` must not be imported - directly or transitively - by `--help`,
`--version`, a bare invocation or a mistyped one. A handler module is resolved
by name inside `dispatch()`, so a subcommand's dependencies are paid for only
by the user who asked for that subcommand. `tests/cli/test_cli_baseline.py`
enforces this structurally, in a fresh interpreter, rather than with a
stopwatch.

**A user error is exit 1, never a traceback.** argparse's own `error()` exits
2, which design §4 reserves for hardware and runtime failure; `_CliParser`
overrides it so that "you typed the wrong flag" and "the NPU is unavailable"
stay distinguishable to a caller.

Command handlers are not implemented here. Each `cmd_<name>.py` module is
expected to expose `run(args: argparse.Namespace) -> int`; until the module
exists, invoking its subcommand reports a runtime error rather than crashing.

**A command owns its own flags, and pays for them only when it is used.** A
subcommand's options are registered by its handler module through
`ARGUMENTS_ATTR` (`add_arguments(parser)`), which `build_parser` calls for the
single command named in `argv` and for no other. Design §3.5 gives `search`
eighteen options; registering all five commands' surfaces here would both
contradict that ownership and import every handler module on every
invocation, which is precisely what requirement 1.2 forbids. A module that
exposes no `add_arguments` is unaffected and its subparser stays bare.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Final, NoReturn, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

__all__ = [
    "ARGUMENTS_ATTR",
    "COMMAND_MODULES",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_USAGE",
    "build_parser",
    "dispatch",
    "main",
    "resolve_version",
]

PROGRAM_NAME: Final = "npu-rag"
DISTRIBUTION_NAME: Final = "npu-rag"
DESCRIPTION: Final = "NPU-accelerated local retrieval over a personal document archive."

# design.md §4 - Error Taxonomy & Exit Codes.
EXIT_OK: Final = 0
"""Success: clean output rendered to stdout."""
EXIT_USAGE: Final = 1
"""Configuration or user error: unknown tag, malformed config, bad argument."""
EXIT_RUNTIME: Final = 2
"""Hardware or runtime error: provider unavailable, corrupted index."""

#: Subcommand name -> handler module. Requirement 1.3 fixes the five names and
#: design.md §2 fixes the filenames, so both are spelled out rather than
#: generated: a test compares this mapping against its own literal copy, and a
#: derived mapping would make that comparison a restatement instead of a check.
#: Registration order here is the order argparse offers them in `--help`.
COMMAND_MODULES: Final[dict[str, str]] = {
    "search": "npu_rag.cli.cmd_search",
    "index": "npu_rag.cli.cmd_index",
    "status": "npu_rag.cli.cmd_status",
    "tags": "npu_rag.cli.cmd_tags",
    "benchmark": "npu_rag.cli.cmd_benchmark",
}

#: One-line summary per subcommand, taken from requirement 1.3.
COMMAND_SUMMARIES: Final[dict[str, str]] = {
    "search": "Query retrieval across the archive with filtering and ranking controls.",
    "index": "Ingest, chunk, embed, and synchronize local documents into LanceDB.",
    "status": "Display index health, chunk counts, model architecture, and freshness.",
    "tags": "Inspect and validate the configured tag-to-folder taxonomy.",
    "benchmark": "Execute the embedding throughput and latency benchmark harness.",
}

#: The callable every `cmd_<name>.py` module must expose.
HANDLER_ATTR: Final = "run"

#: The optional callable a `cmd_<name>.py` module may expose to register its
#: own flags: `add_arguments(parser) -> None`. Consulted by `build_parser` for
#: the ONE command present in `argv` and never for the others - see
#: `_registered_command` for why that restriction is requirement 1.2 and not an
#: optimisation. A module without it keeps working exactly as before: its
#: subparser takes no arguments, which is what `benchmark` (blocked, task 5.3)
#: still relies on.
ARGUMENTS_ATTR: Final = "add_arguments"


def resolve_version() -> str:
    """Return the installed distribution version.

    `importlib.metadata` is imported here and not at module scope: it drags in
    `email`, `zipfile` and `csv`, none of which the `--help` path needs.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:  # pragma: no cover - requires an uninstalled tree
        return "unknown"


class _CliParser(argparse.ArgumentParser):
    """An `ArgumentParser` whose usage errors exit 1 rather than argparse's 2.

    `add_subparsers` defaults `parser_class` to `type(self)`, so every
    subcommand parser inherits this behaviour without further wiring.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


class _VersionAction(argparse.Action):
    """`--version`, printed to stdout and resolved only when asked for.

    `argparse`'s built-in version action needs the version string at
    parser-construction time, which would make every `--help` and every
    mistyped command pay for `importlib.metadata`. It also writes to stderr in
    some paths; a version banner belongs on stdout so `npu-rag --version` can
    be captured by a script.
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        help: str | None = None,  # noqa: A002 - argparse's own parameter name
    ) -> None:
        super().__init__(
            option_strings=list(option_strings),
            dest=dest,
            default=argparse.SUPPRESS,
            nargs=0,
            help=help,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        sys.stdout.write(f"{PROGRAM_NAME} {resolve_version()}\n")
        raise SystemExit(EXIT_OK)


def _registered_command(argv: Sequence[str]) -> str | None:
    """The subcommand `argv` names, if it names one of ours.

    The first token that is not an option is the candidate: the top-level
    parser takes only `--version`, which consumes no value, so nothing else can
    occupy that position. A token that is not one of the five commands yields
    `None`, which is what keeps requirement 1.2's fast paths free: `--help`,
    `--version`, a bare invocation and a mistyped command all resolve to
    `None`, and no `cmd_*` module is imported for any of them.

    Everything after the command belongs to the command, so a query that
    happens to read `index` cannot be mistaken for one - the first non-option
    token has already been claimed.
    """
    for token in argv:
        if token.startswith("-"):
            continue
        return token if token in COMMAND_MODULES else None
    return None


def _register_arguments(parser: argparse.ArgumentParser, command: str) -> None:
    """Let `command`'s handler module register its own flags, if it can.

    Three outcomes, all deliberate. The module exposes `ARGUMENTS_ATTR` and it
    is called. The module exists without it - the subparser stays bare, which
    is exactly the behaviour every command had before this hook. The module
    does not exist at all - nothing happens here, and `dispatch()` reports the
    missing handler as a runtime error after parsing, which is where that
    diagnostic has always come from.

    A `ModuleNotFoundError` naming something *other* than the handler module is
    re-raised, for the same reason `dispatch()` re-raises it: a genuine missing
    dependency inside a command must not be disguised as an absent command.
    """
    module_name = COMMAND_MODULES[command]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        return
    register = getattr(module, ARGUMENTS_ATTR, None)
    if callable(register):
        register(parser)


def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    """Build the top-level parser and register the five subcommands.

    Subcommand *flags* are not written here: each `cmd_<name>.py` owns its own
    arguments and registers them through `ARGUMENTS_ATTR`. What is fixed here
    is the set of commands, and the fact that at most one handler module is
    imported to build a parser.

    :param argv: The arguments about to be parsed. Given, the one command they
        name has its flags registered; omitted, every subparser is bare - which
        is what `--help` for the top-level parser and the structural tests want,
        and is the shape this function had before the hook existed.
    """
    parser = _CliParser(prog=PROGRAM_NAME, description=DESCRIPTION)
    parser.add_argument(
        "--version",
        action=_VersionAction,
        help="Show the installed npu-rag version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    requested = None if argv is None else _registered_command(argv)
    for name in COMMAND_MODULES:
        summary = COMMAND_SUMMARIES[name]
        subparser = subparsers.add_parser(name, help=summary, description=summary)
        if name == requested:
            _register_arguments(subparser, name)
    return parser


def _fail(message: str, code: int) -> int:
    """Write a diagnostic to stderr and return `code`. Never raises."""
    sys.stderr.write(f"{PROGRAM_NAME}: error: {message}\n")
    return code


def _resolve_handler(module: ModuleType) -> Callable[[argparse.Namespace], int] | None:
    candidate = getattr(module, HANDLER_ATTR, None)
    if not callable(candidate):
        return None
    return cast("Callable[[argparse.Namespace], int]", candidate)


def dispatch(args: argparse.Namespace) -> int:
    """Import the handler module for `args.command` and run it.

    This is the only place a subcommand's module is named at run time, and it
    is reached only after parsing has already succeeded.
    """
    command = str(args.command)
    module_name = COMMAND_MODULES[command]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only the handler module's own absence means "not available yet". A
        # `ModuleNotFoundError` for anything else is a genuine missing
        # dependency inside the command and must not be disguised as one.
        if exc.name != module_name:
            raise
        return _fail(
            f"the '{command}' command is not available: its handler module "
            f"'{module_name}' is not installed.",
            EXIT_RUNTIME,
        )

    handler = _resolve_handler(module)
    if handler is None:
        return _fail(
            f"the '{command}' handler module '{module_name}' exposes no callable "
            f"'{HANDLER_ATTR}(args)'.",
            EXIT_RUNTIME,
        )
    return handler(args)


def _exit_status(code: int | str | None) -> int:
    """Normalise a `SystemExit` payload into a process exit code."""
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    sys.stderr.write(f"{code}\n")
    return EXIT_USAGE


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point declared as `npu-rag` under `[project.scripts]`.

    Returns a process exit code; it does not call `sys.exit` itself, so it can
    be driven directly from a test.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(tokens)
    try:
        args = parser.parse_args(tokens)
    except SystemExit as exc:
        # `--help`, `--version` and every parse error land here.
        return _exit_status(exc.code)

    command: str | None = getattr(args, "command", None)
    if command is None:
        # No command is a user error, so the help goes to stderr and stdout
        # stays clean for whatever the user was actually piping into.
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    try:
        return dispatch(args)
    except KeyboardInterrupt:
        return _fail("interrupted.", EXIT_RUNTIME)


if __name__ == "__main__":
    raise SystemExit(main())
