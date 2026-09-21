"""Tests for the tag taxonomy and pushdown filter builder (task 2.1).

Requirements 3.1 (`--folder`, name / relative subfolder / glob), 3.2 (`[tags]`,
N folders per tag and N folders sharing tags), 3.3 (`--tag` expansion into a
combined pushdown filter) and 3.4 (actionable unknown-tag diagnostic); design
§3.2 and §4.

**These tests assert meaning, not text.** A generated SQL predicate is checked
by *running it against a real LanceDB table* holding rows whose folder names
carry every character that breaks naive quoting - an apostrophe, a literal
`%`, a literal `_`, a path separator. A string-equality assertion on generated
SQL breaks on harmless reformatting and still cannot say whether the logic is
right; asking the database which rows come back can.

Only three textual assertions survive, each pinning a property no row-level
check can observe: that an apostrophe is doubled rather than backslashed, that
`LIKE` carries its `ESCAPE` clause, and that "no predicates" is `None` rather
than the empty string (which would be a falsy value some call site would read
as "no filter" and another would push down as a syntax error).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from npu_rag.cli.config import ConfigError, emit_config_error
from npu_rag.cli.tags import (
    AUTHOR_COLUMN,
    PATH_COLUMN,
    EmptyTagError,
    TagRegistry,
    UnknownTagError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from npu_rag.index.store import LanceIndexStore


# --------------------------------------------------------------------------- #
# The corpus. `author` is the first directory component beneath the archive
# root (`npu_rag.ingest.discover._author`), so it is exactly the folder name;
# `file_path` is `str(Path)`, which on Windows is backslash-separated.
#
# Every row exists to break something:
#   apo    - an apostrophe, which terminates a naively interpolated literal.
#   und    - a literal `_`, a LIKE single-character wildcard. `BX` is its foil:
#            an unescaped `B_ Restab` matches both, which is the live defect in
#            design §3.2's snippet.
#   pct    - a literal `%`, a LIKE any-run wildcard. `100 Returns` is its foil.
#   gar1/2 - two folders a glob must reach together.
#   sub    - a file in a nested subfolder, for requirement 3.1's "relative
#            subfolder" on a platform whose separator is `\`.
# --------------------------------------------------------------------------- #

_ROOT: Final = "D:\\Arch"

#: chunk_id, author (folder), path relative to the root, chunk_index.
_ROWS: Final[tuple[tuple[str, str, str, int], ...]] = (
    ("plain-0", "Plain Folder", "Plain Folder\\a.md", 0),
    ("plain-1", "Plain Folder", "Plain Folder\\b.md", 1),
    ("apo-0", "O'Brien's Notes", "O'Brien's Notes\\a.md", 0),
    ("und-0", "B_ Restabilization", "B_ Restabilization\\a.md", 0),
    ("undx-0", "BX Restabilization", "BX Restabilization\\a.md", 0),
    ("pct-0", "100% Returns", "100% Returns\\a.md", 0),
    ("pctless-0", "100 Returns", "100 Returns\\a.md", 0),
    ("gar1-0", "Bhagwan Garuda Vol 1", "Bhagwan Garuda Vol 1\\a.md", 0),
    ("gar2-0", "Bhagwan Garuda Vol 2", "Bhagwan Garuda Vol 2\\a.md", 1),
    # `Vol 10` is `Vol ?`'s foil: it is what tells `?` (exactly one character)
    # apart from `*` (any run), which every shorter-name fixture absorbs.
    ("gar10-0", "Bhagwan Garuda Vol 10", "Bhagwan Garuda Vol 10\\a.md", 0),
    ("tara-0", "Bhagwati Ugra Tara Sadhana", "Bhagwati Ugra Tara Sadhana\\a.md", 0),
    ("axia-0", "Axia_ Market Profile Masterclass", "Axia_ Market Profile Masterclass\\a.md", 0),
    ("sub-0", "Parent", "Parent\\Child\\a.md", 0),
    # Requirement 8.4's Devanagari, in a folder name rather than a snippet: the
    # escaping walks characters, and a byte-oriented one would split a
    # combining mark.
    ("dev-0", "श्री गुरु साधना", "श्री गुरु साधना\\a.md", 0),
)

#: Requirement 3.2's many-to-many shape, deliberately non-degenerate:
#: `siddha` has three folders, and `shared` shares one of them with `siddha`.
_TAGS: Final[dict[str, tuple[str, ...]]] = {
    "siddha": ("Bhagwati Ugra Tara Sadhana", "Bhagwan Garuda*", "O'Brien's Notes"),
    "markets": ("Axia_ Market Profile Masterclass", "100% Returns"),
    "shared": ("Plain Folder", "Bhagwati Ugra Tara Sadhana"),
}


@pytest.fixture
def registry() -> TagRegistry:
    return TagRegistry(_TAGS)


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> LanceIndexStore:
    """A real LanceDB table holding `_ROWS`, built once for the module."""
    from npu_rag.index.schema import VECTOR_DIMENSION, ChunkSchema
    from npu_rag.index.store import LanceIndexStore

    db_path: Path = tmp_path_factory.mktemp("tags-db") / "lancedb"
    index_store = LanceIndexStore(db_path)
    vector = [0.0] * VECTOR_DIMENSION
    index_store.add(
        [
            ChunkSchema(
                chunk_id=chunk_id,
                file_path=f"{_ROOT}\\{relative}",
                file_hash="h",
                chunk_index=chunk_index,
                chunk_kind="prose",
                title=None,
                headings=[],
                locator_type="markdown",
                locator_json="{}",
                text="body",
                vector=vector,
                token_count=1,
                created_at=0.0,
                author=author,
            )
            for chunk_id, author, relative, chunk_index in _ROWS
        ]
    )
    return index_store


def _matching(store: LanceIndexStore, predicate: str | None) -> set[str]:
    """The chunk_ids a compiled predicate actually selects."""
    assert predicate is not None
    rows = store.table.search().where(predicate).limit(len(_ROWS) * 2).to_list()
    return {str(row["chunk_id"]) for row in rows}


# --------------------------------------------------------------------------- #
# Requirement 3.2 - the taxonomy itself
# --------------------------------------------------------------------------- #


def test_tag_resolves_to_all_its_folders_in_declaration_order(registry: TagRegistry) -> None:
    """N folders per tag, in the order the author wrote them (requirement 3.2).

    Order is asserted, not just membership: `config._resolve_tags` keeps a tuple
    rather than a set precisely so this survives, and a mutant returning only
    the first folder would pass a membership check on a one-folder fixture.
    """
    assert registry.resolve_tag("siddha") == (
        "Bhagwati Ugra Tara Sadhana",
        "Bhagwan Garuda*",
        "O'Brien's Notes",
    )
    assert len(registry.resolve_tag("siddha")) == 3


def test_one_folder_may_belong_to_several_tags(registry: TagRegistry) -> None:
    """N folders sharing tags - the other half of requirement 3.2's many-to-many."""
    shared = "Bhagwati Ugra Tara Sadhana"
    assert shared in registry.resolve_tag("siddha")
    assert shared in registry.resolve_tag("shared")


def test_resolving_several_tags_dedupes_and_keeps_first_seen_order(
    registry: TagRegistry,
) -> None:
    """A folder reached through two tags is expanded once, not twice."""
    assert registry.resolve_tags(["siddha", "shared"]) == (
        "Bhagwati Ugra Tara Sadhana",
        "Bhagwan Garuda*",
        "O'Brien's Notes",
        "Plain Folder",
    )


def test_registry_exposes_its_table_for_the_tags_subcommand(registry: TagRegistry) -> None:
    """Task 4.2's `npu-rag tags` needs to list everything, valid or not."""
    assert registry.tags == _TAGS
    assert registry.tag_names() == ("markets", "shared", "siddha")


def test_registry_copies_its_input(registry: TagRegistry) -> None:
    """Mutating the caller's dict afterwards must not change the registry."""
    mutable = {"alpha": ("A",)}
    built = TagRegistry(mutable)
    mutable["alpha"] = ("B",)
    assert built.resolve_tag("alpha") == ("A",)


def test_folder_values_are_stripped(registry: TagRegistry) -> None:
    """`alpha = [" A "]` scopes to `A`; the padding is not part of the name."""
    assert TagRegistry({"alpha": ("  Plain Folder  ",)}).resolve_tag("alpha") == (
        "Plain Folder",
    )


# --------------------------------------------------------------------------- #
# Requirement 3.4 - the unknown-tag diagnostic
# --------------------------------------------------------------------------- #


def test_unknown_tag_lists_every_available_tag(registry: TagRegistry) -> None:
    """Requirement 3.4 verbatim: "an actionable error listing all available tags"."""
    with pytest.raises(UnknownTagError) as excinfo:
        registry.resolve_tag("siddah")
    message = str(excinfo.value)
    assert "siddah" in message
    for name in _TAGS:
        assert f"'{name}'" in message


def test_unknown_tag_names_where_the_taxonomy_lives(registry: TagRegistry) -> None:
    """Origin-aware, like every other diagnostic in `config.py`."""
    with pytest.raises(UnknownTagError, match="npu-rag.toml"):
        registry.resolve_tag("nope")


def test_unknown_tag_origin_is_injectable() -> None:
    """The real discovered path, when a caller knows it."""
    built = TagRegistry(_TAGS, source="C:\\cfg\\npu-rag.toml")
    with pytest.raises(UnknownTagError, match=r"C:\\cfg\\npu-rag\.toml"):
        built.resolve_tag("nope")


def test_unknown_tag_with_no_tags_configured_says_so() -> None:
    """An empty taxonomy must not render as `Available tags: ` with nothing."""
    with pytest.raises(UnknownTagError) as excinfo:
        TagRegistry({}).resolve_tag("siddha")
    message = str(excinfo.value)
    assert "Available tags:" not in message
    assert "[tags]" in message


def test_unknown_tag_is_a_config_error_and_exits_one(
    registry: TagRegistry, capsys: pytest.CaptureFixture[str]
) -> None:
    """Design §4 puts "unknown tag" and "malformed config" in the same bucket.

    Subclassing `ConfigError` is what lets task 4.1 catch one exception type and
    render both through `emit_config_error`, rather than maintaining two
    near-identical `except` arms that will drift apart.
    """
    assert issubclass(UnknownTagError, ConfigError)
    assert issubclass(EmptyTagError, ConfigError)
    error = pytest.raises(UnknownTagError, registry.resolve_tag, "nope").value
    assert emit_config_error(error) == 1
    assert "npu-rag: error:" in capsys.readouterr().err


def test_blank_tag_name_is_rejected(registry: TagRegistry) -> None:
    """`--tag ""`, or the empty element of `--tag a,,b`."""
    with pytest.raises(ConfigError, match="empty"):
        registry.resolve_tag("   ")


def test_tag_name_is_stripped(registry: TagRegistry) -> None:
    """`--tag " siddha"` from a quoted shell argument still resolves."""
    assert registry.resolve_tag("  siddha  ") == registry.resolve_tag("siddha")


# --------------------------------------------------------------------------- #
# Degenerate `[tags]` entries (Implementation Note 1.2, binding on this task)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "label"),
    [
        ((), "alpha = []"),
        (("",), 'alpha = ""'),
        (("", "  "), 'alpha = ["", "  "]'),
        (("Plain Folder", ""), 'alpha = ["Plain Folder", ""]'),
    ],
)
def test_degenerate_tag_entry_is_an_error(value: tuple[str, ...], label: str) -> None:
    """A tag that expands to nothing, or to a blank name, is a config error.

    `config.py` accepts all four of these silently, so the ruling is made here.
    Both alternatives are worse than an error: a blank folder name compiles to
    `file_path LIKE '%%'`, which matches *everything* - a scoping flag that
    widens scope - and an empty expansion compiles to a predicate matching
    nothing, which is indistinguishable in the output from a query that simply
    had no hits. Only an error tells the user their `npu-rag.toml` is wrong.
    """
    with pytest.raises(EmptyTagError) as excinfo:
        TagRegistry({"alpha": value}).resolve_tag("alpha")
    message = str(excinfo.value)
    assert "alpha" in message
    assert "npu-rag.toml" in message
    assert label  # the case label documents the TOML that produces `value`


def test_a_degenerate_entry_does_not_poison_the_other_tags() -> None:
    """Validation is lazy, so one broken tag does not break an unrelated search.

    `npu-rag search --folder X` must not fail because some tag the user did not
    ask for is malformed, and `npu-rag tags` (task 4.2) must still be able to
    list a broken entry so the user can see what to fix.
    """
    built = TagRegistry({"alpha": (), "beta": ("Plain Folder",)})
    assert built.resolve_tag("beta") == ("Plain Folder",)
    assert built.tag_names() == ("alpha", "beta")
    with pytest.raises(EmptyTagError):
        built.resolve_tag("alpha")


# --------------------------------------------------------------------------- #
# SQL safety - the defect design §3.2 ships
# --------------------------------------------------------------------------- #


def test_apostrophe_in_a_folder_name_is_doubled_not_backslashed(
    registry: TagRegistry,
) -> None:
    """LanceDB's parser rejects `\\'`; `''` is the only escape it accepts.

    Pinned textually because no row-level check can distinguish "escaped
    correctly" from "escaped some other way that also happens to work".
    """
    sql = registry.build_filter_sql(folder="O'Brien's Notes")
    assert sql is not None
    assert "O''Brien''s Notes" in sql
    # Not `"\\'" not in sql`: the `ESCAPE '\'` clause legitimately contains
    # that pair. What must be absent is the backslash-escaped *name*, which is
    # what a C-style escaper would emit and what LanceDB's parser rejects.
    assert "O\\'Brien" not in sql


def test_apostrophe_folder_selects_exactly_its_rows(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """The escaping actually works against a real table (requirement 3.1)."""
    sql = registry.build_filter_sql(folder="O'Brien's Notes")
    assert _matching(store, sql) == {"apo-0"}


def test_apostrophe_folder_reached_through_a_tag(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """The `[tags]` table is user-controlled too, and takes the same path."""
    sql = registry.build_filter_sql(tags=["siddha"])
    assert "apo-0" in _matching(store, sql)


@pytest.mark.parametrize(
    "hostile",
    [
        "x' OR 1=1 --",
        "x'; DROP TABLE chunks; --",
        "x' OR author LIKE '%",
    ],
)
def test_crafted_folder_name_cannot_terminate_the_literal(
    registry: TagRegistry, store: LanceIndexStore, hostile: str
) -> None:
    """A value that tries to close the quote stays one opaque folder name.

    The predicate must *parse and run* - a value that made it a syntax error
    would be a different bug, not safety - and must select nothing, because no
    folder in the corpus is called that.
    """
    sql = registry.build_filter_sql(folder=hostile)
    assert _matching(store, sql) == set()


def test_crafted_tag_value_cannot_terminate_the_literal(store: LanceIndexStore) -> None:
    """`npu-rag.toml` is a file a user may have copied from somewhere else."""
    built = TagRegistry({"evil": ("x' OR 1=1 --",)})
    assert _matching(store, built.build_filter_sql(tags=["evil"])) == set()


# --------------------------------------------------------------------------- #
# LIKE metacharacters - `%` and `_` in a folder name are literal
# --------------------------------------------------------------------------- #


def test_underscore_in_a_folder_name_is_literal_not_a_wildcard(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """`B_ Restabilization` must not drag in `BX Restabilization`.

    This is the concrete defect in design §3.2: `file_path LIKE '%B_ Restab%'`
    matches both, because `_` is LIKE's single-character wildcard.
    """
    sql = registry.build_filter_sql(folder="B_ Restabilization")
    assert _matching(store, sql) == {"und-0"}


def test_percent_in_a_folder_name_is_literal_not_a_wildcard(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """`100% Returns` must not drag in `100 Returns`."""
    sql = registry.build_filter_sql(folder="100% Returns")
    assert _matching(store, sql) == {"pct-0"}


def test_like_clauses_declare_their_escape_character(registry: TagRegistry) -> None:
    """Escaping `%` and `_` is inert without it, and LanceDB accepts only `\\`.

    Probed against lancedb 0.38.0: `ESCAPE '!'` raises "LIKE does not support
    escape_char other than the backslash".
    """
    sql = registry.build_filter_sql(folder="B_ Restabilization")
    assert sql is not None
    assert sql.count("LIKE") == sql.count("ESCAPE '\\'")
    assert sql.count("ESCAPE '\\'") >= 1


def test_metacharacter_folder_reached_through_a_tag(store: LanceIndexStore) -> None:
    """`markets` names both `Axia_ ...` and `100% ...`; neither may over-match."""
    built = TagRegistry(_TAGS)
    assert _matching(store, built.build_filter_sql(tags=["markets"])) == {
        "axia-0",
        "pct-0",
    }


# --------------------------------------------------------------------------- #
# Requirement 3.1 - name, relative subfolder, glob
# --------------------------------------------------------------------------- #


def test_plain_folder_name_selects_the_whole_folder(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    assert _matching(store, registry.build_filter_sql(folder="Plain Folder")) == {
        "plain-0",
        "plain-1",
    }


def test_folder_name_selects_nested_files_too(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """`Parent` reaches `Parent\\Child\\a.md`."""
    assert _matching(store, registry.build_filter_sql(folder="Parent")) == {"sub-0"}


def test_a_folder_name_does_not_match_a_longer_sibling(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """`Vol 1` must not drag in `Vol 10`.

    Design §3.2's `file_path LIKE '%{folder}%'` returns both, because a folder
    name is a prefix of every longer name beside it. This is the same class of
    silent over-match as the unescaped `_`, and it is why the `file_path` term
    requires a path separator immediately after the pattern.
    """
    assert _matching(
        store, registry.build_filter_sql(folder="Bhagwan Garuda Vol 1")
    ) == {"gar1-0"}


def test_a_file_name_is_not_a_folder(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """`--folder` scopes to folders; a bare substring would also match files.

    Every row's path ends in `a.md` or `b.md`, and a file name is never
    followed by a separator, so the component-boundary anchor excludes them.
    """
    assert _matching(store, registry.build_filter_sql(folder="a.md")) == set()


@pytest.mark.parametrize("spelling", ["Parent/Child", "Parent\\Child"])
def test_relative_subfolder_matches_whichever_separator_the_user_typed(
    registry: TagRegistry, store: LanceIndexStore, spelling: str
) -> None:
    """Requirement 3.1's "relative subfolder", on a platform that stores `\\`.

    `file_path` is `str(Path)`, so it is backslash-separated on Windows. A user
    who types the forward slash their shell and their spec both use would
    otherwise get a silent empty result.
    """
    assert _matching(store, registry.build_filter_sql(folder=spelling)) == {"sub-0"}


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        # `*` is any run, including an empty one.
        ("Bhagwan Garuda*", {"gar1-0", "gar2-0", "gar10-0"}),
        # `?` is exactly one character, so `Vol 10` is excluded. Translating
        # `?` to `%` instead would silently include it.
        ("Bhagwan Garuda Vol ?", {"gar1-0", "gar2-0"}),
        # `*` in the middle must span eight characters (" Garuda "), which a
        # translation to `_` cannot.
        ("Bhagwan*Vol 1", {"gar1-0"}),
        ("*Ugra Tara*", {"tara-0"}),
        ("Bhagwan Garuda Vol 1", {"gar1-0"}),
        ("*गुरु*", {"dev-0"}),
    ],
)
def test_glob_patterns(
    registry: TagRegistry, store: LanceIndexStore, pattern: str, expected: set[str]
) -> None:
    """Requirement 3.1's glob, and requirement 3.2's `"Bhagwan Garuda*"` example."""
    assert _matching(store, registry.build_filter_sql(folder=pattern)) == expected


def test_glob_expansion_inside_a_tag(store: LanceIndexStore) -> None:
    """`siddha` reaches all three Garuda volumes through one glob entry."""
    built = TagRegistry(_TAGS)
    assert _matching(store, built.build_filter_sql(tags=["siddha"])) == {
        "tara-0",
        "gar1-0",
        "gar2-0",
        "gar10-0",
        "apo-0",
    }


def test_non_ascii_folder_name_round_trips(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """Requirement 8.4's scripts appear in folder names, not only in snippets."""
    assert _matching(store, registry.build_filter_sql(folder="श्री गुरु साधना")) == {
        "dev-0"
    }


def test_folder_matching_is_case_sensitive(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """Pinned because it is a decision, not an accident.

    LanceDB's `=` and `LIKE` are both case-sensitive, and nothing here folds
    case. A user who types `plain folder` gets nothing rather than a surprise
    match, which is the same contract the filesystem listing they copied the
    name from would give them. If task 4.1 ever wants case-insensitive scoping
    it is a deliberate change here, and this test is what will notice.
    """
    assert _matching(store, registry.build_filter_sql(folder="plain folder")) == set()


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_folder_is_rejected_rather_than_silently_ignored(
    registry: TagRegistry, blank: str
) -> None:
    """`if folder:` in design §3.2 drops `--folder ""` and searches everything.

    A scoping flag that silently does not scope is the failure this refuses.
    The empty string is the case that matters: whitespace is truthy, so a
    truthiness guard would still reach the check and the defect would hide.
    """
    with pytest.raises(ConfigError, match="--folder"):
        registry.build_filter_sql(folder=blank)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_custom_filter_is_rejected(registry: TagRegistry, blank: str) -> None:
    with pytest.raises(ConfigError, match="--filter"):
        registry.build_filter_sql(custom_filter=blank)


# --------------------------------------------------------------------------- #
# Requirement 3.3 - combination and precedence
# --------------------------------------------------------------------------- #

_FOLDER: Final = "Plain Folder"
_TAG: Final = "shared"  # -> Plain Folder, Bhagwati Ugra Tara Sadhana
_FILTER: Final = "chunk_index = 1"  # -> plain-1, gar2-0


@pytest.mark.parametrize(
    ("folder", "tags", "custom", "expected"),
    [
        (_FOLDER, None, None, {"plain-0", "plain-1"}),
        (None, [_TAG], None, {"plain-0", "plain-1", "tara-0"}),
        (None, None, _FILTER, {"plain-1", "gar2-0"}),
        (_FOLDER, [_TAG], None, {"plain-0", "plain-1"}),
        (_FOLDER, None, _FILTER, {"plain-1"}),
        (None, [_TAG], _FILTER, {"plain-1"}),
        (_FOLDER, [_TAG], _FILTER, {"plain-1"}),
    ],
    ids=["folder", "tag", "filter", "folder+tag", "folder+filter", "tag+filter", "all"],
)
def test_all_seven_combinations_are_conjunctive(
    registry: TagRegistry,
    store: LanceIndexStore,
    folder: str | None,
    tags: list[str] | None,
    custom: str | None,
    expected: set[str],
) -> None:
    """`AND` between clauses; `OR` only inside a tag's expansion.

    Every row here distinguishes that shape from a flattened `OR` across
    everything: `folder+tag` would yield `tara-0` as well, and every
    combination with `--filter` would yield the union instead of the
    intersection.
    """
    sql = registry.build_filter_sql(folder=folder, tags=tags, custom_filter=custom)
    assert _matching(store, sql) == expected


def test_folder_and_a_disjoint_tag_select_nothing(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """A conjunction of two folder scopes no row can satisfy.

    Under a flattened `OR` this returns eight rows instead of none.
    """
    sql = registry.build_filter_sql(folder=_FOLDER, tags=["siddha"])
    assert _matching(store, sql) == set()


def test_several_tags_union_their_folders(
    registry: TagRegistry, store: LanceIndexStore
) -> None:
    """Requirement 3.3's "multiple comma-separated tags" - `OR` between them."""
    sql = registry.build_filter_sql(tags=["markets", "shared"])
    assert _matching(store, sql) == {"axia-0", "pct-0", "plain-0", "plain-1", "tara-0"}


def test_custom_filter_survives_a_tag(registry: TagRegistry) -> None:
    """A dropped `--filter` is invisible whenever the tag alone narrows enough."""
    sql = registry.build_filter_sql(tags=["shared"], custom_filter=_FILTER)
    assert sql is not None
    assert _FILTER in sql


def test_custom_filter_is_parenthesised(registry: TagRegistry, store: LanceIndexStore) -> None:
    """An unbracketed `OR` inside `--filter` would otherwise swallow the scope."""
    sql = registry.build_filter_sql(
        folder=_FOLDER, custom_filter="chunk_index = 1 OR chunk_index = 0"
    )
    assert _matching(store, sql) == {"plain-0", "plain-1"}


def test_unknown_tag_surfaces_from_build_filter_sql(registry: TagRegistry) -> None:
    """The diagnostic must reach the user through the real entry point too."""
    with pytest.raises(UnknownTagError):
        registry.build_filter_sql(folder=_FOLDER, tags=["siddah"])


# --------------------------------------------------------------------------- #
# The empty case
# --------------------------------------------------------------------------- #


def test_no_inputs_compile_to_none_not_empty_string(registry: TagRegistry) -> None:
    """`""` is falsy *and* an SQL syntax error; only one of those is safe.

    `DenseRetriever.retrieve` guards with `if filter_sql:`, so `""` happens to
    work there - and `LanceIndexStore.delete_by_predicate` would push it
    straight down. `None` is unambiguous at every call site.
    """
    assert registry.build_filter_sql() is None
    assert registry.build_filter_sql(folder=None, tags=None, custom_filter=None) is None


def test_an_empty_tag_sequence_is_absence_not_an_error(registry: TagRegistry) -> None:
    """argparse hands `[]` for an `append` option that was never given."""
    assert registry.build_filter_sql(tags=[]) is None
    assert registry.build_filter_sql(folder=_FOLDER, tags=[]) is not None


def test_both_columns_design_names_are_present_in_the_clause(
    registry: TagRegistry,
) -> None:
    """Design §3.2 matches on `author` *and* `file_path`; both must be emitted.

    This is the one property no row-level assertion can see, and it is worth
    stating why. Against today's schema the `author` term is *behaviourally*
    redundant: `author` is the first directory component beneath the archive
    root and `file_path` is the full path containing it, so every row the exact
    `author` term selects, the `file_path` substring term already selects. The
    `OR` therefore never widens. It is kept because design §3.2 specifies it
    and because it is the term that stays correct if `file_path` is ever stored
    relative to a root below the folder - at which point the substring term
    stops covering it, silently. Deleting it would pass every other test here,
    so this is the test that would notice.
    """
    literal = registry.build_filter_sql(folder="Plain Folder")
    assert literal is not None
    assert f"{AUTHOR_COLUMN} = " in literal
    assert f"{PATH_COLUMN} LIKE " in literal

    # A glob cannot be an equality; the term becomes a LIKE rather than
    # disappearing, which is what keeps `--folder 'Bhagwan Garuda*'` an exact
    # folder-name match rather than only a path substring.
    globbed = registry.build_filter_sql(folder="Bhagwan Garuda*")
    assert globbed is not None
    assert f"{AUTHOR_COLUMN} LIKE " in globbed


def test_column_literals_are_still_fields_of_the_index_schema() -> None:
    """`tags.py` spells `author`/`file_path` rather than importing `ChunkSchema`.

    Importing `npu_rag.index.schema` would put `lancedb` and `pydantic` on the
    startup path of every `--help` (requirement 1.2), so the columns are
    literals - and a literal can silently stop matching the schema. `config.py`
    pins `DEFAULT_MODEL_NAME` against the profile registry for the same reason.
    """
    from npu_rag.index.schema import ChunkSchema

    assert AUTHOR_COLUMN == "author"
    assert PATH_COLUMN == "file_path"
    assert AUTHOR_COLUMN in ChunkSchema.model_fields
    assert PATH_COLUMN in ChunkSchema.model_fields


def test_an_empty_registry_still_filters_by_folder(store: LanceIndexStore) -> None:
    """`--folder` does not depend on a `[tags]` table existing."""
    sql = TagRegistry({}).build_filter_sql(folder="Plain Folder")
    assert _matching(store, sql) == {"plain-0", "plain-1"}
