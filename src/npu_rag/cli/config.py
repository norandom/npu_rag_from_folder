"""Configuration resolution for the `npu-rag` CLI (task 1.2).

Design anchors: `.kiro/specs/search-cli/design.md` §3.1 (`CliConfig`,
`load_config()`) and §4 (Error Taxonomy & Exit Codes); requirements 2.1, 2.2
and 2.3.

Four layers, highest priority first (requirement 2.1)::

    command-line flags  >  NPU_RAG_*  >  npu-rag.toml  >  built-in defaults

Three rules govern this module.

**A layer only wins where it actually supplied something.** `argparse` fills
every option it was not given with `None`, so a call site can hand the whole
namespace over as `cli_args` and `None` is read as absence, not as a value. An
environment variable that is empty or whitespace-only is likewise absence,
which is the rule that decides what ``NPU_RAG_ANSI_COLORS=`` means; it is
stated here rather than left to whatever `os.environ` happens to contain.

**Environment variables are strings; `CliConfig` is typed.** Every value is
coerced by the same per-field function regardless of which layer produced it,
so ``NPU_RAG_DEFAULT_LIMIT=10`` becomes the integer ``10`` and
``NPU_RAG_ANSI_COLORS=false`` becomes ``False`` - `bool("false")` is `True`,
and that is the defect this arrangement exists to prevent.

**A bad value is a user error, never a traceback.** Anything the user can fix
by editing a file, a variable or a flag raises `ConfigError` with a message
naming *where* the value came from; `emit_config_error()` renders it to stderr
and returns exit code 1, matching design §4 and the pattern task 1.1
established in `main.py`. `tomllib.TOMLDecodeError` never escapes.

Nothing heavy is imported here: this module is on the path of every
subcommand, so requirement 1.2's startup budget applies to it as well.

Not in scope, deliberately: the `NO_COLOR` / non-TTY detection of requirement
8.2. That is a property of the *stream at run time*, not a configuration
layer, and it belongs to `render.py` (task 3.1). `ansi_colors` here is the
user's stated preference, which rendering may still override downwards.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import TextIO

__all__ = [
    "CONFIG_FILENAME",
    "ENV_PREFIX",
    "EXIT_CONFIG_ERROR",
    "MODE_CHOICES",
    "PROVIDER_CHOICES",
    "USER_CONFIG_RELATIVE",
    "CliConfig",
    "ConfigError",
    "emit_config_error",
    "env_var_names",
    "find_config_file",
    "load_config",
]

PROGRAM_NAME: Final = "npu-rag"

#: Requirement 2.1 / design §3.1: the project file and the user-level fallback.
CONFIG_FILENAME: Final = "npu-rag.toml"
USER_CONFIG_RELATIVE: Final = Path(".config") / "npu-rag" / "config.toml"

#: Requirement 2.1 fixes the prefix; a field's variable is the prefix plus its
#: upper-cased name, which is what makes `NPU_RAG_ARCHIVE_PATH`,
#: `NPU_RAG_PROVIDER` and `NPU_RAG_INDEX_PATH` come out as the requirement
#: spells them.
ENV_PREFIX: Final = "NPU_RAG_"

#: design §4 assigns 1 to "Configuration or User Error". `main.EXIT_USAGE` is
#: the same number; it is not imported, because `main` may one day import this
#: module and the two must not become a cycle. A test compares them.
EXIT_CONFIG_ERROR: Final = 1

#: Requirement 6.1.
PROVIDER_CHOICES: Final[tuple[str, ...]] = ("npu", "cpu", "auto")
#: Requirement 5.1.
MODE_CHOICES: Final[tuple[str, ...]] = ("dense", "fts", "hybrid")

_TRUE_LITERALS: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE_LITERALS: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})

# Built-in defaults (the lowest layer of requirement 2.1).
#
# `archive_path` and `index_path` stay *relative*: resolving them here would
# bake the process's working directory into a value the user may have written
# as a relative path on purpose, and `Path(".")` would stop meaning "here".
DEFAULT_ARCHIVE_PATH: Final = Path(".")
DEFAULT_INDEX_PATH: Final = Path(".npu_rag") / "lancedb"
#: Spelled as a literal rather than imported from `npu_rag.embedding.profiles`,
#: which would put the whole embedding package on the CLI's startup path. A
#: test asserts the literal is still a declared profile.
DEFAULT_MODEL_NAME: Final = "embeddinggemma-300m"
DEFAULT_PROVIDER: Final = "auto"
DEFAULT_MODE: Final = "hybrid"
DEFAULT_LIMIT: Final = 10
DEFAULT_ANSI_COLORS: Final = True

_T = TypeVar("_T")


class ConfigError(Exception):
    """A configuration problem the user can fix (design §4, exit code 1).

    Carries a message that names the offending value *and where it came from* -
    a file and key, an environment variable, or a command-line option - because
    "invalid literal for int()" tells the user nothing about which of the four
    layers to go and edit.
    """


@dataclass(frozen=True)
class CliConfig:
    """Resolved CLI configuration (design §3.1).

    Built by `load_config()`, which is the only place the four-layer precedence
    of requirement 2.1 and the value validation live.

    **Validation is deliberately NOT in `__post_init__`**, against this repo's
    other 13 frozen-config sites (`IngestConfig` among them). Ruled on
    2026-09-21 at review of task 1.2: requirement 2.3 demands an *actionable*
    message, which means naming where the bad value came from - a file and key,
    an environment variable, a flag - and that provenance is irrecoverably gone
    by the time `__post_init__` runs. `IngestConfig.__post_init__` can only say
    "token_budget must be positive", which is precisely the unactionable
    message 2.3 forbids. Those other sites validate objects assembled in code
    from trusted values; this one terminates an untrusted four-layer pipeline of
    user input.

    **The cost, recorded so it is not rediscovered:** constructing `CliConfig`
    directly is unvalidated. A `provider` of `"gpu"` or a negative
    `default_limit` will be accepted and will fail later, further from the
    user. That is tolerable only while `load_config()` is the sole production
    construction path. Any future code that builds a `CliConfig` some other way
    - a `from_mapping`, an MCP server, a test helper promoted into `src/` - owns
    validating its own inputs, and if a second path appears this decision should
    be revisited rather than quietly extended.
    """

    archive_path: Path
    index_path: Path
    model_name: str
    provider: str  # "npu", "cpu", "auto"
    tags: dict[str, tuple[str, ...]]
    default_mode: str
    default_limit: int
    ansi_colors: bool


# --------------------------------------------------------------------------- #
# Coercion - one function per field, applied to every layer alike
# --------------------------------------------------------------------------- #


def _kind(value: object) -> str:
    return type(value).__name__


def _listed(choices: tuple[str, ...]) -> str:
    return ", ".join(repr(choice) for choice in choices)


def _coerce_path(raw: object, origin: str) -> Path:
    if isinstance(raw, Path):
        candidate = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ConfigError(f"{origin} is empty; expected a filesystem path.")
        candidate = Path(text)
    else:
        raise ConfigError(f"{origin} must be a filesystem path, not {_kind(raw)}.")
    # `~/Archive` in a config file must not create a literal `./~` directory.
    return candidate.expanduser()


def _coerce_text(raw: object, origin: str) -> str:
    if not isinstance(raw, str):
        raise ConfigError(f"{origin} must be text, not {_kind(raw)}.")
    text = raw.strip()
    if not text:
        raise ConfigError(f"{origin} is empty; expected a name.")
    return text


def _choice_coercer(choices: tuple[str, ...]) -> Callable[[object, str], str]:
    def coerce(raw: object, origin: str) -> str:
        if not isinstance(raw, str):
            raise ConfigError(
                f"{origin} must be one of {_listed(choices)}, not {_kind(raw)}."
            )
        value = raw.strip().lower()
        if value not in choices:
            raise ConfigError(f"{origin} must be one of {_listed(choices)}; got {raw!r}.")
        return value

    return coerce


def _coerce_positive_int(raw: object, origin: str) -> int:
    # `bool` is a subclass of `int`, so `ansi_colors`-shaped input would
    # otherwise silently become 0 or 1 here.
    if isinstance(raw, bool):
        raise ConfigError(f"{origin} must be a positive integer, not a boolean.")
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        try:
            value = int(raw.strip())
        except ValueError:
            raise ConfigError(
                f"{origin} must be a positive integer; got {raw!r}."
            ) from None
    else:
        raise ConfigError(f"{origin} must be a positive integer, not {_kind(raw)}.")
    if value <= 0:
        raise ConfigError(f"{origin} must be greater than zero; got {value}.")
    return value


def _coerce_bool(raw: object, origin: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in _TRUE_LITERALS:
            return True
        if text in _FALSE_LITERALS:
            return False
        accepted = _listed(tuple(sorted(_TRUE_LITERALS | _FALSE_LITERALS)))
        raise ConfigError(f"{origin} must be a boolean; got {raw!r}. Accepted: {accepted}.")
    raise ConfigError(f"{origin} must be true or false, not {_kind(raw)}.")


# --------------------------------------------------------------------------- #
# Field table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _FieldSpec(Generic[_T]):
    """One scalar `CliConfig` field and how each layer names it."""

    name: str
    default: _T
    coerce: Callable[[object, str], _T]

    @property
    def env_var(self) -> str:
        return f"{ENV_PREFIX}{self.name.upper()}"

    @property
    def option(self) -> str:
        return f"--{self.name.replace('_', '-')}"


_ARCHIVE_PATH: Final = _FieldSpec("archive_path", DEFAULT_ARCHIVE_PATH, _coerce_path)
_INDEX_PATH: Final = _FieldSpec("index_path", DEFAULT_INDEX_PATH, _coerce_path)
_MODEL_NAME: Final = _FieldSpec("model_name", DEFAULT_MODEL_NAME, _coerce_text)
_PROVIDER: Final = _FieldSpec("provider", DEFAULT_PROVIDER, _choice_coercer(PROVIDER_CHOICES))
_DEFAULT_MODE: Final = _FieldSpec("default_mode", DEFAULT_MODE, _choice_coercer(MODE_CHOICES))
_DEFAULT_LIMIT: Final = _FieldSpec("default_limit", DEFAULT_LIMIT, _coerce_positive_int)
_ANSI_COLORS: Final = _FieldSpec("ansi_colors", DEFAULT_ANSI_COLORS, _coerce_bool)

#: Every scalar field, in the order design §3.1 declares them. `tags` is absent
#: on purpose: a many-to-many table has no sensible flat string spelling, so it
#: is a TOML-or-default field with no environment variable and no flag.
_SCALAR_SPECS: Final[tuple[_FieldSpec[Any], ...]] = (
    _ARCHIVE_PATH,
    _INDEX_PATH,
    _MODEL_NAME,
    _PROVIDER,
    _DEFAULT_MODE,
    _DEFAULT_LIMIT,
    _ANSI_COLORS,
)

_TAGS_KEY: Final = "tags"


def env_var_names() -> dict[str, str]:
    """Map each configurable field to the `NPU_RAG_*` variable that sets it."""
    return {spec.name: spec.env_var for spec in _SCALAR_SPECS}


# --------------------------------------------------------------------------- #
# Discovery and parsing
# --------------------------------------------------------------------------- #


def find_config_file(
    *, start: Path | None = None, home: Path | None = None, stop: Path | None = None
) -> Path | None:
    """The configuration file that applies at `start`, or `None`.

    Requirement 2.1 and design §3.1: the current directory, then each parent in
    turn, then `~/.config/npu-rag/config.toml`. The nearest file wins, and the
    user-level file is consulted only when the upward walk found nothing.

    The upward walk is deliberately the same shape as
    `npu_rag.embedding.models.acquire.find_dotenv` - `(origin, *origin.parents)`
    over a resolved path - so the two behave identically from a nested working
    directory. It diverges from design §3.1 in one respect: §3.1 says "up to git
    root", and this walks the whole chain by default. The CLI has no git
    dependency and is meant to run over archives that are not repositories at
    all, so a git-root stop would make a config file unreachable exactly where
    the archive is not under version control.

    `stop` bounds that walk: it is examined and then the walk ends, so `stop` is
    *inclusive*. It exists because an unbounded walk is not testable in
    isolation - `tmp_path` sits under the user's `%TEMP%`, so a single
    `npu-rag.toml` anywhere above it (and `~/npu-rag.toml` is the obvious thing
    for a user of this tool to create) silently changes what a test with no
    config file of its own discovers. A `stop` that is not an ancestor of
    `start` never matches, and the walk then runs to the filesystem root as
    usual. The user-level fallback is unaffected by `stop`.
    """
    origin = (start if start is not None else Path.cwd()).resolve()
    boundary = stop.resolve() if stop is not None else None
    for directory in (origin, *origin.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        if boundary is not None and directory == boundary:
            break

    base = home if home is not None else Path.home()
    user_config = base / USER_CONFIG_RELATIVE
    if user_config.is_file():
        return user_config
    return None


def _parse(path: Path) -> dict[str, object]:
    """Read one TOML file (requirement 2.2), turning any failure into a
    `ConfigError` (requirement 2.3)."""
    try:
        with path.open("rb") as handle:
            data: dict[str, object] = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"{path} is not valid TOML: {exc}. "
            f"Fix the file, or remove it to fall back to the built-in defaults."
        ) from exc
    except OSError as exc:
        raise ConfigError(f"{path} could not be read: {exc}.") from exc
    return data


def _load_table(
    *, start: Path | None, home: Path | None, stop: Path | None, config_path: Path | None
) -> tuple[Path | None, dict[str, object]]:
    if config_path is not None:
        candidate = Path(config_path)
        if not candidate.is_file():
            raise ConfigError(
                f"configuration file not found: {candidate}. "
                f"Name an existing file, or omit it to search for {CONFIG_FILENAME}."
            )
    else:
        found = find_config_file(start=start, home=home, stop=stop)
        if found is None:
            return None, {}
        candidate = found
    return candidate, _parse(candidate)


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #


def _resolve(
    spec: _FieldSpec[_T],
    *,
    cli_args: Mapping[str, object],
    env: Mapping[str, str],
    table: Mapping[str, object],
    source: str,
) -> _T:
    """The four layers of requirement 2.1, consulted highest first."""
    supplied = cli_args.get(spec.name)
    if supplied is not None:
        return spec.coerce(supplied, f"command-line option {spec.option}")

    from_env = env.get(spec.env_var)
    if from_env is not None and from_env.strip():
        return spec.coerce(from_env, f"environment variable {spec.env_var}")

    if spec.name in table:
        return spec.coerce(table[spec.name], f"{source}: key '{spec.name}'")

    return spec.default


def _resolve_tags(table: Mapping[str, object], *, source: str) -> dict[str, tuple[str, ...]]:
    """The `[tags]` table (requirement 3.2), TOML or nothing.

    Author order inside a tag is preserved: task 2.1 compiles it into an SQL
    predicate and requirement 4 cares about sequence, so this is a tuple rather
    than a set.
    """
    raw = table.get(_TAGS_KEY)
    if raw is None:
        return {}
    origin = f"{source}: [{_TAGS_KEY}]"
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{origin} must be a table mapping tag names to folder names, not {_kind(raw)}."
        )

    resolved: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        name = str(key)
        entry = f"{origin} entry '{name}'"
        if isinstance(value, str):
            resolved[name] = (value,)
            continue
        if not isinstance(value, list):
            raise ConfigError(
                f"{entry} must be a folder name or a list of folder names, not {_kind(value)}."
            )
        folders: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(
                    f"{entry} must list folder names as strings; got {_kind(item)} {item!r}."
                )
            folders.append(item)
        resolved[name] = tuple(folders)
    return resolved


def load_config(
    *,
    cli_args: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    start: Path | None = None,
    home: Path | None = None,
    stop: Path | None = None,
    config_path: Path | None = None,
) -> CliConfig:
    """Resolve the effective configuration (requirement 2.1).

    :param cli_args: Parsed command-line values keyed by `CliConfig` field name;
        `vars(namespace)` is the expected call. A value of `None`, and any key
        that is not a configurable field, are ignored.
    :param env: The environment to read; defaults to `os.environ`. An empty or
        whitespace-only variable counts as unset.
    :param start: Where the upward search for `npu-rag.toml` begins; defaults to
        the process's working directory.
    :param home: The home directory holding `.config/npu-rag/config.toml`;
        defaults to `Path.home()`. Injectable so tests never read a real home.
    :param stop: Inclusive upper bound for the upward search; see
        `find_config_file`. Unbounded by default.
    :param config_path: An explicit file, used verbatim instead of discovery.
        Unlike a discovered file, its absence is an error (requirement 2.3
        speaks of "a specified configuration file").

    :raises ConfigError: for any user-correctable problem - a malformed or
        unreadable file, a value of the wrong shape, an unknown provider or
        mode. Callers render it with `emit_config_error()`.
    """
    resolved_cli: Mapping[str, object] = {} if cli_args is None else cli_args
    resolved_env: Mapping[str, str] = os.environ if env is None else env

    path, table = _load_table(
        start=start, home=home, stop=stop, config_path=config_path
    )
    source = str(path) if path is not None else "<no configuration file>"

    def field(spec: _FieldSpec[_T]) -> _T:
        return _resolve(
            spec, cli_args=resolved_cli, env=resolved_env, table=table, source=source
        )

    return CliConfig(
        archive_path=field(_ARCHIVE_PATH),
        index_path=field(_INDEX_PATH),
        model_name=field(_MODEL_NAME),
        provider=field(_PROVIDER),
        tags=_resolve_tags(table, source=source),
        default_mode=field(_DEFAULT_MODE),
        default_limit=field(_DEFAULT_LIMIT),
        ansi_colors=field(_ANSI_COLORS),
    )


def emit_config_error(
    error: ConfigError, *, program: str = PROGRAM_NAME, stream: TextIO | None = None
) -> int:
    """Render a `ConfigError` to stderr and return the exit code (design §4).

    The counterpart of `main._fail`: a diagnostic on stderr, nothing on stdout,
    no traceback, exit 1. `stream` is resolved at call time so `capsys` and a
    redirected stderr both see the message.
    """
    target = sys.stderr if stream is None else stream
    target.write(f"{program}: error: {error}\n")
    return EXIT_CONFIG_ERROR
