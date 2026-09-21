"""Configuration precedence and resolution for `npu-rag` (task 1.2).

Pins requirement 2.1 (CLI flags > `NPU_RAG_*` > `npu-rag.toml` > defaults),
requirement 2.2 (`tomllib`), requirement 2.3 (a malformed file exits 1 with an
actionable message and no traceback) and design.md §3.1 / §4.

**On the shape of the precedence tests.** The obvious way to test a four-layer
precedence chain is one test per layer, each setting only that layer - and it
is worthless: every such test passes against a completely *reversed* chain,
because no two layers ever disagree. `KEYS` below therefore gives every field
four mutually distinguishable values, one per layer, and
`test_precedence_resolves_to_the_highest_layer_present` writes *all layers from
the winner downward* before asserting the winner. Every lower layer is actively
contradicting the expected answer, so the assertion can only hold if the
ordering is the one requirement 2.1 specifies.

`test_every_ladder_step_is_distinguishable` is the non-vacuity guard for that
fixture (standing lesson rule 3, `.kiro/specs/npu-embedding-runtime/tasks.md`):
it fails loudly if someone later "simplifies" a ladder so that two adjacent
layers agree and silently disarms the assertions above it.

`ansi_colors` is the one ladder that cannot be four distinct values - a bool has
two - so it alternates `True/False/True/False` down the layers. That catches
every single-layer swap, but a *total* reversal of the chain is invisible to it
in the env-wins case; the five non-boolean ladders catch that, which is why the
precedence test asserts across all seven fields rather than one.

**On hermeticity.** The upward walk runs to the filesystem root in production,
and `tmp_path` lives under the user's `%TEMP%`, so any `npu-rag.toml` in an
ancestor - `~/npu-rag.toml` is the obvious thing for a user of this tool to
create - would be picked up by every test that expects to find no config file
at all. Review planted exactly that and **7 tests failed**. Every test here
therefore goes through `_load`, which bounds the walk at `tmp_path`; the
production semantics stay pinned by
`test_the_walk_is_unbounded_unless_a_stop_is_given` rather than merely worked
around. This is standing lesson rules 5-8: a test that passes because the
machine happens to be clean is not a test of this code.

Nothing here reads the developer's real home directory, real working directory
or real environment except the three tests that exist to prove the defaults for
those inputs are wired at all, and those inject a fake home, chdir into
`tmp_path`, and clear `NPU_RAG_*` first.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from npu_rag.cli import config as config_module
from npu_rag.cli.config import CliConfig, ConfigError, load_config

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

CONFIG_SOURCE = Path(config_module.__file__)

# Requirement 1.2's forbidden startup imports. `config.py` is imported by every
# subcommand, so it is held to the same standard as `main.py`.
HEAVY_MODULES: tuple[str, ...] = ("lancedb", "onnxruntime", "torch", "transformers")

# Requirement 2.1 names the `NPU_RAG_` prefix and three of the variables by
# example. Written as literals, not derived from the module under test, so the
# comparison below is a check rather than a restatement.
EXPECTED_ENV_VARS: dict[str, str] = {
    "archive_path": "NPU_RAG_ARCHIVE_PATH",
    "index_path": "NPU_RAG_INDEX_PATH",
    "model_name": "NPU_RAG_MODEL_NAME",
    "provider": "NPU_RAG_PROVIDER",
    "default_mode": "NPU_RAG_DEFAULT_MODE",
    "default_limit": "NPU_RAG_DEFAULT_LIMIT",
    "ansi_colors": "NPU_RAG_ANSI_COLORS",
}

#: Precedence layers, highest first (requirement 2.1).
LAYERS: tuple[str, ...] = ("cli", "env", "toml", "default")


@dataclasses.dataclass(frozen=True)
class _Key:
    """One `CliConfig` field, with a distinguishable value per layer.

    `*_raw` is what the layer carries (a TOML scalar, an environment string, a
    parsed CLI value); `*_value` is what `CliConfig` must end up holding.
    """

    name: str
    default: object
    toml_raw: object
    toml_value: object
    env_raw: str
    env_value: object
    cli_raw: object
    cli_value: object

    def raw(self, layer: str) -> object:
        return {"cli": self.cli_raw, "env": self.env_raw, "toml": self.toml_raw}[layer]

    def value(self, layer: str) -> object:
        return {
            "cli": self.cli_value,
            "env": self.env_value,
            "toml": self.toml_value,
            "default": self.default,
        }[layer]


KEYS: tuple[_Key, ...] = (
    _Key(
        name="archive_path",
        default=Path("."),
        toml_raw="archive-from-toml",
        toml_value=Path("archive-from-toml"),
        env_raw="archive-from-env",
        env_value=Path("archive-from-env"),
        cli_raw=Path("archive-from-cli"),
        cli_value=Path("archive-from-cli"),
    ),
    _Key(
        name="index_path",
        default=Path(".npu_rag") / "lancedb",
        toml_raw="index-from-toml",
        toml_value=Path("index-from-toml"),
        env_raw="index-from-env",
        env_value=Path("index-from-env"),
        cli_raw=Path("index-from-cli"),
        cli_value=Path("index-from-cli"),
    ),
    _Key(
        name="model_name",
        default="embeddinggemma-300m",
        toml_raw="model-from-toml",
        toml_value="model-from-toml",
        env_raw="model-from-env",
        env_value="model-from-env",
        cli_raw="model-from-cli",
        cli_value="model-from-cli",
    ),
    # Only three legal providers exist, so this ladder alternates rather than
    # taking four distinct values; every adjacent pair still disagrees.
    _Key(
        name="provider",
        default="auto",
        toml_raw="npu",
        toml_value="npu",
        env_raw="cpu",
        env_value="cpu",
        cli_raw="npu",
        cli_value="npu",
    ),
    _Key(
        name="default_mode",
        default="hybrid",
        toml_raw="dense",
        toml_value="dense",
        env_raw="fts",
        env_value="fts",
        cli_raw="dense",
        cli_value="dense",
    ),
    # The env rung is the string "12": a resolver that forwarded environment
    # values untouched would put `"12"` into an `int` field and fail here.
    _Key(
        name="default_limit",
        default=10,
        toml_raw=11,
        toml_value=11,
        env_raw="12",
        env_value=12,
        cli_raw=13,
        cli_value=13,
    ),
    # `"false"` is a non-empty string: `bool("false")` is `True`, so a resolver
    # that coerced with the builtin would return the wrong answer here.
    _Key(
        name="ansi_colors",
        default=True,
        toml_raw=False,
        toml_value=False,
        env_raw="true",
        env_value=True,
        cli_raw=False,
        cli_value=False,
    ),
)


def _toml_scalar(value: object) -> str:
    """Render a Python scalar as its TOML spelling."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return '"{}"'.format(str(value).replace("\\", "\\\\"))


def _write_toml(directory: Path, body: str, *, name: str = "npu-rag.toml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def _nowhere(tmp_path: Path) -> Path:
    """A home directory that exists but holds no user config."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    return home


def _empty_start(tmp_path: Path) -> Path:
    """A directory whose whole ancestor chain is inside `tmp_path`."""
    start = tmp_path / "empty" / "cwd"
    start.mkdir(parents=True, exist_ok=True)
    return start


def _load(
    tmp_path: Path,
    *,
    cli_args: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    start: Path | None = None,
    home: Path | None = None,
    config_path: Path | None = None,
) -> CliConfig:
    """`load_config()` bounded to `tmp_path`, with no real home and no real env.

    The `stop=tmp_path` is what makes this file hermetic: in production the
    upward search runs to the filesystem root, and `tmp_path` sits under the
    user's `%TEMP%`, so one `npu-rag.toml` above it changes the answer for every
    test that expects to find nothing. The unbounded behaviour is asserted
    separately by `test_the_walk_is_unbounded_unless_a_stop_is_given`, so this
    bound hides nothing.
    """
    return load_config(
        cli_args={} if cli_args is None else cli_args,
        env={} if env is None else env,
        start=_empty_start(tmp_path) if start is None else start,
        home=_nowhere(tmp_path) if home is None else home,
        stop=tmp_path,
        config_path=config_path,
    )


def _find(tmp_path: Path, start: Path, *, home: Path | None = None) -> Path | None:
    """`find_config_file()` bounded to `tmp_path`, for the same reason as `_load`."""
    return config_module.find_config_file(
        start=start, home=_nowhere(tmp_path) if home is None else home, stop=tmp_path
    )


def _materialise(
    tmp_path: Path, present: tuple[str, ...]
) -> tuple[dict[str, object], dict[str, str], Path]:
    """Populate the requested layers and return `(cli_args, env, start)`.

    The TOML file is written one directory *above* `start` so that the upward
    walk of design §3.1 is exercised by the precedence tests too.
    """
    project = tmp_path / "project"
    start = project / "nested" / "deeper"
    start.mkdir(parents=True, exist_ok=True)

    cli_args: dict[str, object] = {}
    env: dict[str, str] = {}
    if "toml" in present:
        lines = [f"{key.name} = {_toml_scalar(key.toml_raw)}" for key in KEYS]
        _write_toml(project, "\n".join(lines) + "\n")
    if "env" in present:
        env = {EXPECTED_ENV_VARS[key.name]: key.env_raw for key in KEYS}
    if "cli" in present:
        cli_args = {key.name: key.cli_raw for key in KEYS}
    return cli_args, env, start


@pytest.fixture
def clean_environ(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every `NPU_RAG_*` variable from the real process environment."""
    for name in list(os.environ):
        if name.startswith("NPU_RAG_"):
            monkeypatch.delenv(name, raising=False)
    yield


def _module_scope_imports(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
        elif isinstance(node, ast.If):
            for inner in node.body:
                if isinstance(inner, ast.Import):
                    names.extend(alias.name for alias in inner.names)
                elif isinstance(inner, ast.ImportFrom) and inner.level == 0 and inner.module:
                    names.append(inner.module)
    return names


# --------------------------------------------------------------------------- #
# Design §3.1 - the model
# --------------------------------------------------------------------------- #


def test_cli_config_carries_exactly_the_designed_fields() -> None:
    """design.md §3.1 fixes the dataclass; this pins the contract 2.1-5.2 use."""
    fields = {field.name: field.type for field in dataclasses.fields(CliConfig)}
    assert list(fields) == [
        "archive_path",
        "index_path",
        "model_name",
        "provider",
        "tags",
        "default_mode",
        "default_limit",
        "ansi_colors",
    ]


def test_cli_config_is_frozen(tmp_path: Path) -> None:
    resolved = _load(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.provider = "cpu"  # type: ignore[misc]


def test_env_variable_names_match_the_documented_prefix() -> None:
    """Requirement 2.1 names the prefix and three of the variables by example."""
    assert config_module.ENV_PREFIX == "NPU_RAG_"
    assert config_module.env_var_names() == EXPECTED_ENV_VARS
    for name in EXPECTED_ENV_VARS.values():
        assert name.startswith("NPU_RAG_")


def test_config_module_imports_nothing_heavy() -> None:
    """Every subcommand will import this module, so it stays on the fast path."""
    imported = _module_scope_imports(CONFIG_SOURCE)
    forbidden = [name for name in imported if name.split(".")[0] in HEAVY_MODULES]
    assert forbidden == [], f"config.py must import nothing heavy: {forbidden}"
    # Requirement 2.2 names the parser explicitly.
    assert "tomllib" in imported


# --------------------------------------------------------------------------- #
# Requirement 2.1 - precedence
# --------------------------------------------------------------------------- #


def test_every_ladder_step_is_distinguishable() -> None:
    """Non-vacuity guard for `KEYS`.

    If two adjacent layers of a ladder ever hold the same value, the precedence
    assertion between them stops being able to tell a correct resolver from one
    that consults the layers in the wrong order - and it would still pass. This
    fails instead.
    """
    for key in KEYS:
        values = [key.value(layer) for layer in LAYERS]
        for higher, lower in zip(values, values[1:], strict=False):
            assert higher != lower, (
                f"{key.name}: adjacent precedence layers must disagree, "
                f"otherwise the ordering assertion is vacuous (got {higher!r} twice)"
            )
    # The env rung must arrive as a string for every key: that is what makes the
    # coercion assertions in the precedence test load-bearing.
    for key in KEYS:
        assert isinstance(key.env_raw, str)


@pytest.mark.parametrize("winner", LAYERS)
def test_precedence_resolves_to_the_highest_layer_present(tmp_path: Path, winner: str) -> None:
    """Requirement 2.1, with every lower layer contradicting the expected answer.

    `present` is the winning layer *and everything below it*, so for
    `winner="env"` the TOML file on disk says something else entirely; an
    implementation that consulted TOML before the environment returns the TOML
    value and fails. For `winner="default"` nothing is supplied anywhere, which
    is the only case where the built-in defaults may appear.
    """
    present = LAYERS[LAYERS.index(winner) :]
    cli_args, env, start = _materialise(tmp_path, present)

    resolved = _load(tmp_path, cli_args=cli_args, env=env, start=start)

    for key in KEYS:
        actual = getattr(resolved, key.name)
        assert actual == key.value(winner), (
            f"{key.name}: with layers {present} present, the {winner} layer must win; "
            f"got {actual!r}, expected {key.value(winner)!r}"
        )
        assert type(actual) is type(key.value(winner)), (
            f"{key.name}: resolved to the right value with the wrong type {type(actual)!r}"
        )


def test_a_cli_value_of_none_does_not_mask_a_lower_layer(tmp_path: Path) -> None:
    """`argparse` fills unsupplied options with `None`; that is *absence*.

    Without this, `load_config(cli_args=vars(namespace))` would overwrite every
    configured value with `None` the moment a flag was omitted.
    """
    _, env, start = _materialise(tmp_path, ("env", "toml"))
    cli_args: dict[str, object] = {key.name: None for key in KEYS}

    resolved = _load(tmp_path, cli_args=cli_args, env=env, start=start)

    for key in KEYS:
        assert getattr(resolved, key.name) == key.env_value


def test_unknown_cli_keys_are_ignored(tmp_path: Path) -> None:
    """Call sites pass a whole `argparse.Namespace`, which carries `command`."""
    resolved = _load(
        tmp_path, cli_args={"command": "search", "top_k": 5, "model_name": "from-cli"}
    )
    assert resolved.model_name == "from-cli"


def test_an_empty_environment_variable_is_treated_as_unset(tmp_path: Path) -> None:
    """`NPU_RAG_PROVIDER=` must fall through, not blank out the TOML value.

    This is the rule that decides what `ansi_colors=""` means, so it is pinned
    rather than left to whatever `os.environ` happens to hand over.
    """
    _, _, start = _materialise(tmp_path, ("toml",))
    env = {EXPECTED_ENV_VARS[key.name]: "   " for key in KEYS}
    env[EXPECTED_ENV_VARS["model_name"]] = ""

    resolved = _load(tmp_path, env=env, start=start)

    for key in KEYS:
        assert getattr(resolved, key.name) == key.toml_value


def test_environment_defaults_to_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_environ: None
) -> None:
    """Non-vacuity control: without this every env test could pass against a
    resolver that never looks at the real process environment at all."""
    _, _, start = _materialise(tmp_path, ("toml",))
    monkeypatch.setenv("NPU_RAG_MODEL_NAME", "model-from-os-environ")

    resolved = load_config(start=start, home=_nowhere(tmp_path), stop=tmp_path)

    assert resolved.model_name == "model-from-os-environ"
    # The TOML layer is still consulted for everything the environment omits.
    assert resolved.provider == "npu"


# --------------------------------------------------------------------------- #
# Requirement 2.1 / design §3.1 - discovery
# --------------------------------------------------------------------------- #


def test_config_is_discovered_in_the_current_directory(tmp_path: Path) -> None:
    start = _empty_start(tmp_path)
    _write_toml(start, 'model_name = "from-cwd"\n')

    assert _load(tmp_path, start=start).model_name == "from-cwd"


def test_config_is_discovered_in_a_distant_parent_directory(tmp_path: Path) -> None:
    """The walk must not stop at the starting directory or at its parent."""
    root = tmp_path / "root"
    start = root / "a" / "b" / "c" / "d"
    start.mkdir(parents=True)
    _write_toml(root, 'model_name = "from-a-distant-parent"\n')

    assert _load(tmp_path, start=start).model_name == "from-a-distant-parent"


def test_the_nearest_config_file_wins(tmp_path: Path) -> None:
    """Two files up the chain: the closer one, not the first one found going up
    from the filesystem root, and not the last."""
    root = tmp_path / "root"
    start = root / "inner" / "leaf"
    start.mkdir(parents=True)
    _write_toml(root, 'model_name = "from-outer"\n')
    _write_toml(root / "inner", 'model_name = "from-inner"\n')

    assert _load(tmp_path, start=start).model_name == "from-inner"


def test_user_config_is_the_fallback_when_no_local_file_exists(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_toml(
        home / ".config" / "npu-rag", 'model_name = "from-user-config"\n', name="config.toml"
    )

    assert _load(tmp_path, home=home).model_name == "from-user-config"


def test_a_local_file_beats_the_user_config(tmp_path: Path) -> None:
    """Design §3.1 lists `~/.config/npu-rag/config.toml` last.

    Consulting it first would be invisible to every other discovery test here,
    because none of them has both files.
    """
    home = tmp_path / "home"
    _write_toml(
        home / ".config" / "npu-rag", 'model_name = "from-user-config"\n', name="config.toml"
    )
    start = _empty_start(tmp_path)
    _write_toml(start, 'model_name = "from-project"\n')

    assert _load(tmp_path, start=start, home=home).model_name == "from-project"


def test_no_config_file_anywhere_yields_the_built_in_defaults(tmp_path: Path) -> None:
    start = _empty_start(tmp_path)

    assert _find(tmp_path, start) is None
    resolved = _load(tmp_path, start=start)
    for key in KEYS:
        assert getattr(resolved, key.name) == key.default
    assert resolved.tags == {}


def test_discovery_uses_the_expected_filenames(tmp_path: Path) -> None:
    """Requirement 2.1 and design §3.1 fix both names."""
    assert config_module.CONFIG_FILENAME == "npu-rag.toml"
    assert config_module.USER_CONFIG_RELATIVE == Path(".config") / "npu-rag" / "config.toml"
    start = _empty_start(tmp_path)
    _write_toml(start, 'model_name = "wrong-name"\n', name="npu_rag.toml")
    assert _find(tmp_path, start) is None


def test_the_walk_is_unbounded_unless_a_stop_is_given(tmp_path: Path) -> None:
    """The production semantics of the walk, pinned rather than worked around.

    Every other test in this file bounds the search at `tmp_path` so that no
    `npu-rag.toml` on the developer's machine can change the answer. That bound
    must not leak into shipped behaviour: discovering a config file from a
    nested working directory is the entire point of design §3.1's walk. So this
    asserts both halves against the same fixture - unbounded finds the file two
    directories up, `stop` at the intermediate directory does not.
    """
    outer = tmp_path / "outer"
    middle = outer / "middle"
    start = middle / "leaf"
    start.mkdir(parents=True)
    planted = _write_toml(outer, 'model_name = "from-above-the-stop"\n')
    home = _nowhere(tmp_path)

    # Unbounded (the shipped default): reached, however far up it is.
    assert config_module.find_config_file(start=start, home=home) == planted
    assert load_config(cli_args={}, env={}, start=start, home=home).model_name == (
        "from-above-the-stop"
    )

    # Bounded inclusively at `middle`: `start` and `middle` are examined, then
    # the walk ends, so the file in `outer` is out of reach.
    assert config_module.find_config_file(start=start, home=home, stop=middle) is None
    assert load_config(
        cli_args={}, env={}, start=start, home=home, stop=middle
    ).model_name == "embeddinggemma-300m"


def test_the_stop_directory_is_itself_searched(tmp_path: Path) -> None:
    """`stop` is inclusive: it is examined and *then* the walk ends.

    An exclusive bound would break the walk one directory early, which is
    invisible to the test above (its planted file is two levels up).
    """
    root = tmp_path / "bounded"
    start = root / "a" / "b"
    start.mkdir(parents=True)
    planted = _write_toml(root, 'model_name = "from-the-stop-directory"\n')

    found = config_module.find_config_file(start=start, home=_nowhere(tmp_path), stop=root)

    assert found == planted


def test_a_stop_that_is_not_an_ancestor_does_not_truncate_the_walk(tmp_path: Path) -> None:
    """A `stop` off the path is simply never matched; it must not stop anything."""
    root = tmp_path / "unrelated-stop"
    start = root / "a" / "b"
    start.mkdir(parents=True)
    planted = _write_toml(root, 'model_name = "still-found"\n')
    elsewhere = tmp_path / "some" / "other" / "branch"
    elsewhere.mkdir(parents=True)

    found = config_module.find_config_file(
        start=start, home=_nowhere(tmp_path), stop=elsewhere
    )

    assert found == planted


def test_start_defaults_to_the_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity control for the discovery tests: they all inject `start`."""
    start = _empty_start(tmp_path)
    _write_toml(start, 'model_name = "from-the-cwd"\n')
    monkeypatch.chdir(start)

    resolved = load_config(cli_args={}, env={}, home=_nowhere(tmp_path), stop=tmp_path)

    assert resolved.model_name == "from-the-cwd"


def test_home_defaults_to_path_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-vacuity control for the user-config fallback, without reading the
    developer's real home directory."""
    home = tmp_path / "pretend-home"
    _write_toml(
        home / ".config" / "npu-rag", 'model_name = "from-fake-home"\n', name="config.toml"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    resolved = load_config(
        cli_args={}, env={}, start=_empty_start(tmp_path), stop=tmp_path
    )

    assert resolved.model_name == "from-fake-home"


def test_an_explicitly_named_config_file_is_used_verbatim(tmp_path: Path) -> None:
    """Requirement 2.3 speaks of "a specified configuration file"."""
    explicit = _write_toml(
        tmp_path / "elsewhere", 'model_name = "from-explicit"\n', name="custom.toml"
    )
    start = _empty_start(tmp_path)
    _write_toml(start, 'model_name = "from-discovery"\n')

    resolved = _load(tmp_path, start=start, config_path=explicit)

    assert resolved.model_name == "from-explicit"


def test_an_explicitly_named_missing_config_file_is_an_error(tmp_path: Path) -> None:
    """Absence is silent only for *discovered* files; a named one must complain."""
    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, config_path=tmp_path / "absent.toml")
    assert "absent.toml" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Requirement 2.2 - tomllib, and the [tags] table
# --------------------------------------------------------------------------- #


def test_tags_are_parsed_into_folder_tuples(tmp_path: Path) -> None:
    """Requirement 3.2's `[tags]` table, many-to-many, N folders per tag."""
    start = _empty_start(tmp_path)
    _write_toml(
        start,
        "[tags]\n"
        'siddha = ["Bhagwati Ugra Tara Sadhana", "Bhagwan Garuda*"]\n'
        'finance = ["Bhagwan Garuda*", "Macro Notes"]\n'
        'solo = "Single Folder"\n',
    )

    resolved = _load(tmp_path, start=start)

    assert resolved.tags == {
        "siddha": ("Bhagwati Ugra Tara Sadhana", "Bhagwan Garuda*"),
        "finance": ("Bhagwan Garuda*", "Macro Notes"),
        "solo": ("Single Folder",),
    }
    # Order within a tag is the author's order, not a set or a sort: task 2.1
    # builds an SQL predicate from it and requirement 4 cares about sequence.
    assert resolved.tags["siddha"][0] == "Bhagwati Ugra Tara Sadhana"


def test_tags_default_to_empty_without_a_tags_table(tmp_path: Path) -> None:
    start = _empty_start(tmp_path)
    _write_toml(start, 'model_name = "no-tags-here"\n')

    resolved = _load(tmp_path, start=start)

    assert resolved.tags == {}
    assert resolved.model_name == "no-tags-here"


def test_a_non_string_folder_in_a_tag_is_rejected(tmp_path: Path) -> None:
    start = _empty_start(tmp_path)
    _write_toml(start, '[tags]\nsiddha = ["Folder A", 17]\n')

    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, start=start)

    assert "siddha" in str(excinfo.value)


@pytest.mark.parametrize(
    "table",
    [
        "[tags]\nalpha = 3\n",
        "[tags]\nalpha = true\n",
        "[tags]\nalpha = 1.5\n",
        "[tags]\nalpha = {x = 1}\n",
    ],
    ids=["int", "bool", "float", "inline-table"],
)
def test_a_tag_value_that_is_neither_a_string_nor_a_list_is_rejected(
    tmp_path: Path, table: str
) -> None:
    """The value-type guard, distinct from the item-type guard above.

    Review found this unasserted: deleting the `isinstance(value, list)` check
    passed all 123 tests, yet without it `alpha = 3` escapes as a bare
    `TypeError: 'int' object is not iterable` - a requirement 2.3 traceback -
    and `alpha = {x = 1}` is silently accepted as `('x',)`, which is a tag
    quietly resolving to a folder nobody named. The item-type case above cannot
    see either, because it only ever passes a list.
    """
    start = _empty_start(tmp_path)
    _write_toml(start, table)

    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, start=start)

    assert "alpha" in str(excinfo.value)


def test_the_parser_is_tomllib(tmp_path: Path) -> None:
    """Requirement 2.2. Proven by behaviour: a document only `tomllib` accepts.

    TOML 1.0 heterogeneous arrays and dotted keys are not INI or JSON, so a
    hand-rolled parser or a JSON loader fails this.
    """
    start = _empty_start(tmp_path)
    body = 'model_name = "tomllib-parsed"\ndefault_limit = 4\ntags.alpha = ["one", "two"]\n'
    assert tomllib.loads(body)["model_name"] == "tomllib-parsed"

    resolved = _load(tmp_path, start=start, config_path=_write_toml(start, body))

    assert resolved.model_name == "tomllib-parsed"
    assert resolved.default_limit == 4
    assert resolved.tags == {"alpha": ("one", "two")}


# --------------------------------------------------------------------------- #
# Environment coercion - strings in, typed values out
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("  false  ", False),
    ],
)
def test_boolean_environment_values_are_coerced_not_truth_tested(
    tmp_path: Path, raw: str, expected: bool
) -> None:
    """`bool("false")` is `True`. This is the bug this test exists to catch.

    The TOML layer underneath deliberately says the *opposite* of the default,
    so a resolver that dropped the variable entirely also fails.
    """
    start = _empty_start(tmp_path)
    _write_toml(start, "ansi_colors = false\n")

    resolved = _load(tmp_path, env={"NPU_RAG_ANSI_COLORS": raw}, start=start)

    assert resolved.ansi_colors is expected


def test_integer_environment_values_are_coerced(tmp_path: Path) -> None:
    resolved = _load(tmp_path, env={"NPU_RAG_DEFAULT_LIMIT": " 25 "})
    assert resolved.default_limit == 25
    assert isinstance(resolved.default_limit, int)
    assert not isinstance(resolved.default_limit, bool)


def test_path_environment_values_become_paths(tmp_path: Path) -> None:
    resolved = _load(tmp_path, env={"NPU_RAG_ARCHIVE_PATH": str(tmp_path / "Archive")})
    assert resolved.archive_path == tmp_path / "Archive"
    assert isinstance(resolved.archive_path, Path)


def test_a_tilde_path_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`archive_path = "~/Archive"` must not create a literal `./~` directory."""
    fake_home = tmp_path / "expansion-home"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    resolved = _load(tmp_path, env={"NPU_RAG_ARCHIVE_PATH": "~/Archive"})

    assert resolved.archive_path == fake_home / "Archive"


@pytest.mark.parametrize(
    ("body", "field", "expected"),
    [
        ('archive_path = "  spaced-archive  "\n', "archive_path", Path("spaced-archive")),
        ('index_path = "  spaced-index  "\n', "index_path", Path("spaced-index")),
        ('model_name = "  spaced-model  "\n', "model_name", "spaced-model"),
        ('provider = "  cpu  "\n', "provider", "cpu"),
        ('default_mode = "  dense  "\n', "default_mode", "dense"),
        ('default_limit = "  7  "\n', "default_limit", 7),
    ],
)
def test_surrounding_whitespace_is_stripped_from_scalar_values(
    tmp_path: Path, body: str, field: str, expected: object
) -> None:
    """A value is normalised, not taken literally.

    Found by review's mutation sweep: dropping `.strip()` from the path and text
    coercers survived the whole file, because nothing asserted the *normalised*
    form - only that a well-formed value came through. `Path("  a  ")` is a real
    directory name on Windows and a silently wrong one.
    """
    start = _empty_start(tmp_path)
    _write_toml(start, body)

    resolved = _load(tmp_path, start=start)

    actual = getattr(resolved, field)
    assert actual == expected
    assert type(actual) is type(expected)


@pytest.mark.parametrize(
    ("variable", "raw", "expected_fragments"),
    [
        ("NPU_RAG_DEFAULT_LIMIT", "abc", ("NPU_RAG_DEFAULT_LIMIT", "abc")),
        ("NPU_RAG_DEFAULT_LIMIT", "0", ("NPU_RAG_DEFAULT_LIMIT",)),
        ("NPU_RAG_DEFAULT_LIMIT", "-3", ("NPU_RAG_DEFAULT_LIMIT",)),
        ("NPU_RAG_DEFAULT_LIMIT", "1.5", ("NPU_RAG_DEFAULT_LIMIT", "1.5")),
        ("NPU_RAG_ANSI_COLORS", "maybe", ("NPU_RAG_ANSI_COLORS", "maybe")),
        ("NPU_RAG_PROVIDER", "gpu", ("NPU_RAG_PROVIDER", "npu", "cpu", "auto")),
        ("NPU_RAG_DEFAULT_MODE", "fuzzy", ("NPU_RAG_DEFAULT_MODE", "dense", "fts", "hybrid")),
    ],
)
def test_invalid_environment_values_name_the_variable(
    tmp_path: Path, variable: str, raw: str, expected_fragments: tuple[str, ...]
) -> None:
    """Requirement 2.3's class of failure: a user error, reported actionably.

    A bare `ValueError` from `int()` names neither the variable nor the legal
    values, so the message content is asserted, not merely its existence.
    """
    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, env={variable: raw})

    message = str(excinfo.value)
    for fragment in expected_fragments:
        assert fragment in message, f"{message!r} does not mention {fragment!r}"


@pytest.mark.parametrize(
    ("body", "expected_fragment"),
    [
        ('provider = "gpu"\n', "provider"),
        ('default_mode = "fuzzy"\n', "default_mode"),
        ("default_limit = 0\n", "default_limit"),
        # `bool` is a subclass of `int`, so a TOML boolean would otherwise be
        # accepted here and land in the field as `True`. Found by mutation:
        # dropping the bool guard in the integer coercer survived every other
        # test in this file.
        ("default_limit = true\n", "default_limit"),
        # TOML has a float scalar, so this is reachable user input, not a
        # defensive impossibility. Without the `else` arm it truncates to 1.
        ("default_limit = 1.5\n", "default_limit"),
        ("ansi_colors = 1\n", "ansi_colors"),
        ("archive_path = 7\n", "archive_path"),
        # An empty or whitespace-only scalar is a mistake, not a value. With the
        # guard gone `archive_path = ""` becomes `Path(".")` and an index run
        # walks the entire working tree instead of reporting the error.
        ('archive_path = ""\n', "archive_path"),
        ('archive_path = "   "\n', "archive_path"),
        ('index_path = ""\n', "index_path"),
        ("model_name = 7\n", "model_name"),
        ('model_name = ""\n', "model_name"),
        ('model_name = "   "\n', "model_name"),
        ('provider = ""\n', "provider"),
        ("tags = 3\n", "tags"),
    ],
)
def test_invalid_toml_values_name_the_key_and_the_file(
    tmp_path: Path, body: str, expected_fragment: str
) -> None:
    start = _empty_start(tmp_path)
    path = _write_toml(start, body)

    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, start=start)

    message = str(excinfo.value)
    assert expected_fragment in message
    assert str(path) in message, "the diagnostic must say which file to fix"


def test_an_invalid_cli_value_names_the_option(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, cli_args={"provider": "gpu"})
    assert "--provider" in str(excinfo.value)


def test_an_empty_cli_value_names_the_option(tmp_path: Path) -> None:
    """An empty string from a flag is a mistake at the highest layer too."""
    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, cli_args={"archive_path": "   "})
    assert "--archive-path" in str(excinfo.value)


def test_case_is_normalised_for_choice_values(tmp_path: Path) -> None:
    resolved = _load(
        tmp_path, env={"NPU_RAG_PROVIDER": "NPU", "NPU_RAG_DEFAULT_MODE": "Hybrid"}
    )
    assert resolved.provider == "npu"
    assert resolved.default_mode == "hybrid"


def test_choice_vocabularies_match_the_requirements() -> None:
    """Requirement 6.1 fixes the providers; requirement 5.1 fixes the modes."""
    assert config_module.PROVIDER_CHOICES == ("npu", "cpu", "auto")
    assert config_module.MODE_CHOICES == ("dense", "fts", "hybrid")


# --------------------------------------------------------------------------- #
# Requirement 2.3 - a malformed file exits 1 with no traceback
# --------------------------------------------------------------------------- #


def test_malformed_toml_raises_a_config_error_naming_the_file(tmp_path: Path) -> None:
    start = _empty_start(tmp_path)
    path = _write_toml(start, 'model_name = "unterminated\nprovider = npu\n')

    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, start=start)

    message = str(excinfo.value)
    assert str(path) in message
    assert "TOML" in message or "toml" in message
    # The parser's own positional detail must survive into the diagnostic,
    # otherwise "your file is broken" is all the user gets.
    assert "line" in message.lower()


def test_a_toml_decode_error_does_not_escape_as_itself(tmp_path: Path) -> None:
    """`tomllib.TOMLDecodeError` reaching a caller *is* the traceback of
    requirement 2.3, however informative its message happens to be."""
    start = _empty_start(tmp_path)
    _write_toml(start, "this is not toml at all\n")

    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, start=start)

    assert not isinstance(excinfo.value, tomllib.TOMLDecodeError)
    assert isinstance(excinfo.value.__cause__, tomllib.TOMLDecodeError)


def test_emit_config_error_exits_one_on_stderr_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 2.3 end to end, in the shape a `cmd_*.py` handler will use.

    Task 1.1 established the pattern: diagnostics on stderr, stdout untouched,
    never a traceback on a user-error path. The `startswith` is deliberate - a
    diagnostic that does not say which program produced it is not actionable
    inside a pipeline, and merely checking for `"error:"` left that unasserted.
    """
    start = _empty_start(tmp_path)
    path = _write_toml(start, "model_name = [unclosed\n")

    try:
        _load(tmp_path, start=start)
    except ConfigError as error:
        code = config_module.emit_config_error(error)
    else:  # pragma: no cover - the fixture is malformed by construction
        raise AssertionError("a malformed config file must raise ConfigError")

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert captured.err.startswith("npu-rag: error: ")
    assert captured.err.endswith("\n")
    assert str(path) in captured.err


def test_emit_config_error_honours_an_explicit_program_and_stream() -> None:
    """The whole rendered line, pinned exactly once."""
    buffer = io.StringIO()

    code = config_module.emit_config_error(
        ConfigError("archive_path is empty"), program="npu-rag-probe", stream=buffer
    )

    assert code == 1
    assert buffer.getvalue() == "npu-rag-probe: error: archive_path is empty\n"


def test_the_config_error_exit_code_is_the_cli_usage_code() -> None:
    """Design §4 assigns 1 to configuration and user error; `main.py` owns the
    taxonomy, so the two constants are compared rather than both asserted to 1."""
    from npu_rag.cli.main import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE

    assert config_module.EXIT_CONFIG_ERROR == EXIT_USAGE
    # The taxonomy must still have three distinct codes, so "equals EXIT_USAGE"
    # cannot be satisfied by collapsing them all onto one number.
    distinct: set[int] = {EXIT_OK, EXIT_USAGE, EXIT_RUNTIME}
    assert len(distinct) == 3


def test_a_directory_named_like_the_config_file_is_walked_past(tmp_path: Path) -> None:
    """Discovery tests for a *file*, so a directory of that name is not one.

    Named for what it proves. It was previously called
    `test_an_unreadable_config_file_is_a_config_error_not_an_oserror`, which was
    wrong twice over: nothing here is unreadable, and the `OSError` arm of
    `_parse` is never reached - `is_file()` is already False for a directory, so
    the walk continues past it. The real unreadable case is the test below.
    """
    start = _empty_start(tmp_path)
    (start / "npu-rag.toml").mkdir()
    _write_toml(start.parent, 'model_name = "reached-past-the-directory"\n')

    # Nothing raises, *and* the walk genuinely continued rather than stopping or
    # returning early: the real file one directory up is still found. Asserting
    # only "the defaults came back" would also pass if discovery had aborted.
    assert _load(tmp_path, start=start).model_name == "reached-past-the-directory"


def test_a_config_file_that_cannot_be_read_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `OSError` arm of `_parse`, which nothing else exercises.

    A config file the process may stat but not open is ordinary on Windows
    (an ACL, another process holding it exclusively), and design §4 puts it in
    the same class as a malformed one: exit 1 with a diagnostic naming the file,
    not a `PermissionError` traceback. The arm is kept rather than deleted as
    unreachable, so it is asserted here.
    """
    start = _empty_start(tmp_path)
    path = _write_toml(start, 'model_name = "never-read"\n')
    real_open = Path.open

    def refuse(self: Path, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if self.name == config_module.CONFIG_FILENAME:
            raise PermissionError(13, "Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse)

    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, start=start)

    assert not isinstance(excinfo.value, OSError)
    assert isinstance(excinfo.value.__cause__, PermissionError)
    assert str(path) in str(excinfo.value)


def test_config_error_is_an_exception_carrying_its_message() -> None:
    error = ConfigError("archive_path must be a filesystem path")
    assert isinstance(error, Exception)
    assert str(error) == "archive_path must be a filesystem path"


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_built_in_defaults_match_the_documented_values(tmp_path: Path) -> None:
    """Requirement 5.1 fixes `hybrid` and 10; requirement 6.1's `auto` is the
    only provider that is not a demand for specific hardware."""
    resolved = _load(tmp_path)
    assert resolved.archive_path == Path(".")
    assert resolved.index_path == Path(".npu_rag") / "lancedb"
    assert resolved.model_name == "embeddinggemma-300m"
    assert resolved.provider == "auto"
    assert resolved.default_mode == "hybrid"
    assert resolved.default_limit == 10
    assert resolved.ansi_colors is True
    assert resolved.tags == {}


def test_the_default_model_is_a_declared_profile() -> None:
    """The default is spelled as a literal in `config.py` to keep the module off
    the heavy import path; this is the edge that would otherwise rot silently."""
    from npu_rag.embedding.profiles import PROFILES

    resolved_default = next(key for key in KEYS if key.name == "model_name").default
    assert resolved_default in PROFILES


def test_defaults_are_not_shared_between_calls(tmp_path: Path) -> None:
    """`tags` is a mutable dict on a frozen dataclass; two loads must not alias."""
    start = _empty_start(tmp_path)
    _write_toml(start, '[tags]\nalpha = ["one"]\n')

    first = _load(tmp_path, start=start)
    second = _load(tmp_path, start=start)
    first.tags["alpha"] = ("mutated",)

    assert second.tags == {"alpha": ("one",)}


def test_load_config_does_not_mutate_the_inputs_it_is_given(tmp_path: Path) -> None:
    env: dict[str, str] = {"NPU_RAG_PROVIDER": "cpu"}
    cli_args: dict[str, object] = {"model_name": "from-cli"}
    env_before = dict(env)
    cli_before = dict(cli_args)

    _load(tmp_path, cli_args=cli_args, env=env)

    assert env == env_before
    assert cli_args == cli_before


def test_a_mapping_that_is_not_a_dict_is_accepted(tmp_path: Path) -> None:
    """The signature promises `Mapping`, so `os.environ` and a `MappingProxy`
    both work; this pins that nothing calls a `dict`-only method."""
    from types import MappingProxyType

    env: Mapping[str, str] = MappingProxyType({"NPU_RAG_PROVIDER": "cpu"})

    resolved = _load(
        tmp_path, cli_args=MappingProxyType({"model_name": "from-cli"}), env=env
    )

    assert resolved.provider == "cpu"
    assert resolved.model_name == "from-cli"
