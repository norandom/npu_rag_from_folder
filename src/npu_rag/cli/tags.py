"""Tag taxonomy and LanceDB pushdown-filter compilation (task 2.1).

Design anchors: `.kiro/specs/search-cli/design.md` §3.2 (`tags.py`) and §4
(Error Taxonomy & Exit Codes); requirements 3.1, 3.2, 3.3 and 3.4.

`TagRegistry` turns three user inputs - `--folder`, `--tag` and `--filter` -
into one SQL `WHERE` expression that `VectorIndex` pushes down into LanceDB, so
scoping a query to a folder or a subject area costs nothing at index time
(design §1 principle 2).

Four rules govern this module.

**Every value here is user-controlled, so every value is escaped.** `--folder`
and `--tag` come from the command line and the folder names come from the
`[tags]` table of a `npu-rag.toml` the user may have copied from somewhere
else. Design §3.2's snippet interpolates all three directly::

    f"(author = '{folder}' OR file_path LIKE '%{folder}%')"   # do not copy

which breaks outright on the first folder name containing an apostrophe - and
this corpus has `Bhagwati Ugra Tara Sadhana`, `Axia_ Market Profile
Masterclass`, `B_ Restabilization of German Statutory Pension Insurance` - and
lets a crafted value close the literal and append arbitrary SQL. Nothing is
interpolated raw except `--filter`, which requirement 5.1 defines as an
"arbitrary SQL WHERE predicate": that is the user deliberately writing SQL for
their own database, so it is passed through, parenthesised so its operators
cannot escape their clause.

**`%` and `_` are literal in a folder name; `*` and `?` are the wildcards.**
Requirement 3.1 calls `--folder` a "name_or_glob" and requirement 3.2's example
is `"Bhagwan Garuda*"`, so the user's wildcard vocabulary is the shell's, not
SQL's. `B_ Restabilization` must therefore not also match `BX
Restabilization`, and `100% Returns` must not also match `100 Returns`, which
is what design §3.2's unescaped `LIKE '%{folder}%'` does today. Escaping `%`
and `_` requires an `ESCAPE` clause, and lancedb 0.38.0 accepts exactly one
escape character: probed 2026-09-21, `ESCAPE '!'` raises "LIKE does not support
escape_char other than the backslash (\\)".

**A degenerate tag is an error, not a guess.** See `resolve_tag`.

**Validation is lazy.** A malformed `[tags]` entry is diagnosed when that tag
is used, not when the registry is built, so an unrelated `npu-rag search
--folder X` still works and task 4.2's `npu-rag tags` can still list a broken
entry for the user to see. `tags` and `tag_names()` therefore report the table
as written.

Nothing heavy is imported: `lancedb` is not needed to *write* SQL, and this
module is on the startup path of `search` and `tags` (requirement 1.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from npu_rag.cli.config import CONFIG_FILENAME, ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "AUTHOR_COLUMN",
    "GLOB_CHARS",
    "LIKE_ESCAPE_CHAR",
    "PATH_COLUMN",
    "PATH_SEPARATORS",
    "EmptyTagError",
    "TagRegistry",
    "UnknownTagError",
]

#: The indexed folder name. `npu_rag.ingest.discover._author` derives it as the
#: first directory component beneath the archive root, so for this folder-first
#: product `author` *is* the folder (design §1 principle 1).
AUTHOR_COLUMN: Final = "author"

#: The absolute source path, stored as `str(Path)` - backslash-separated on
#: Windows. Matching it as a substring is what reaches files in nested
#: subfolders, which an `author` equality cannot.
PATH_COLUMN: Final = "file_path"

# Both column names are spelled as literals rather than imported from
# `npu_rag.index.schema`, which would put `lancedb` and `pydantic` on the
# startup path of every `--help`. A test pins both against `ChunkSchema`.

#: The user's wildcards, translated to SQL's `%` and `_`.
GLOB_CHARS: Final[frozenset[str]] = frozenset("*?")

#: lancedb 0.38.0 supports no other escape character.
LIKE_ESCAPE_CHAR: Final = "\\"

#: Both spellings of a path separator. A user types `Sub/Folder`; Windows
#: stored `Sub\Folder`.
PATH_SEPARATORS: Final[tuple[str, str]] = ("/", "\\")

_DEFAULT_SOURCE: Final = CONFIG_FILENAME


class UnknownTagError(ConfigError):
    """`--tag` named a tag the `[tags]` table does not define (requirement 3.4).

    A subclass of `ConfigError` rather than a sibling, for two reasons. Design
    §4 puts "unknown tag" and "malformed config" in the same row - exit code 1,
    "Configuration or User Error" - so task 4.1 should be able to catch one
    exception type and render both through `emit_config_error()`; two sibling
    types would mean two `except` arms that will eventually disagree about
    format or exit code. And `ConfigError` is already the right shape: its
    contract is a message naming the offending value *and where it came from*,
    and `config._resolve` already raises it for command-line values, so it was
    never a file-only error to begin with.
    """


class EmptyTagError(ConfigError):
    """A `[tags]` entry that names no folder, or a blank one.

    `config._resolve_tags` accepts `alpha = []` as `()` and `alpha = ""` as
    `('',)` without complaint; see `TagRegistry.resolve_tag` for why this is an
    error here rather than a silently-compiled filter.
    """


# --------------------------------------------------------------------------- #
# SQL emission
# --------------------------------------------------------------------------- #


def _sql_literal(text: str) -> str:
    """Quote `text` as an SQL string literal.

    Doubling the quote is not a stylistic choice: probed against lancedb
    0.38.0, `author = 'O\\'Brien'` is rejected by the parser with "Unterminated
    string literal", while `author = 'O''Brien'` selects the row. This is the
    same escape `LanceIndexStore.delete_by_chunk_ids` already uses.
    """
    return "'" + text.replace("'", "''") + "'"


def _like_pattern(spec: str) -> str:
    """Translate a user glob into a SQL `LIKE` pattern.

    One pass, because the two halves interact: `*` and `?` become the SQL
    wildcards, while a `%`, `_` or `\\` the user actually typed becomes a
    literal. Escaping after translating would re-escape the wildcards just
    produced; escaping before would hide the glob characters.
    """
    out: list[str] = []
    for char in spec:
        if char == "*":
            out.append("%")
        elif char == "?":
            out.append("_")
        elif char in ("%", "_", LIKE_ESCAPE_CHAR):
            out.append(LIKE_ESCAPE_CHAR + char)
        else:
            out.append(char)
    return "".join(out)


def _like_clause(column: str, pattern: str) -> str:
    return (
        f"{column} LIKE {_sql_literal(pattern)} "
        f"ESCAPE {_sql_literal(LIKE_ESCAPE_CHAR)}"
    )


def _has_separator(spec: str) -> bool:
    return any(separator in spec for separator in PATH_SEPARATORS)


def _path_clauses(spec: str) -> list[str]:
    """`file_path` terms for `spec`, one per separator spelling.

    Two things are going on, and both are departures from design §3.2's bare
    `file_path LIKE '%{folder}%'`.

    **Both separator spellings are emitted.** Requirement 3.1 accepts a
    "relative subfolder", and `file_path` is `str(Path)` - backslash-separated
    on Windows. A user typing the forward slash their shell and this spec both
    use would otherwise get a silent empty result, which for a scoping flag is
    indistinguishable from a query that genuinely had no hits.

    **The match is anchored at a path-component boundary on the right.** A bare
    substring makes a folder name match any longer sibling that starts with it:
    `--folder "Bhagwan Garuda Vol 1"` also returns `Bhagwan Garuda Vol 10`,
    and `--folder "Parent"` would match a *file* called `Parent.md`. Requiring
    a separator immediately after the pattern removes both: a component that
    is followed by a separator is a directory, and one whose name merely starts
    with the pattern is not followed by a separator at that point. The left
    side is deliberately left open, which is what still lets a partial trailing
    path - `Child`, or `Parent/Child` - select the folder it names.
    """
    clauses: list[str] = []
    for separator in PATH_SEPARATORS:
        normalized = spec.replace("\\", separator).replace("/", separator)
        pattern = f"%{_like_pattern(normalized)}{_like_pattern(separator)}%"
        clauses.append(_like_clause(PATH_COLUMN, pattern))
    return clauses


def _folder_clause(spec: str) -> str:
    """The predicate selecting one folder name, subfolder path, or glob.

    Two deviations from design §3.2, both forced by requirement 3.1's "glob
    pattern" and by what `author` actually holds:

    * `author = '...'` becomes `author LIKE ... ESCAPE ...` **only** when the
      spec contains a glob character; an equality can never match `Bhagwan
      Garuda*`. Literal names keep the equality, which prunes better than a
      `LIKE` and keeps the common case identical to the design.
    * The `author` term is omitted entirely for a spec containing a path
      separator. `author` is a single path component, so such a term is dead
      SQL that can only confuse a reader of the compiled filter.
    """
    terms: list[str] = []
    if not _has_separator(spec):
        if GLOB_CHARS & set(spec):
            terms.append(_like_clause(AUTHOR_COLUMN, _like_pattern(spec)))
        else:
            terms.append(f"{AUTHOR_COLUMN} = {_sql_literal(spec)}")
    terms.extend(_path_clauses(spec))
    return "(" + " OR ".join(terms) + ")"


def _require_value(raw: str, *, origin: str, expected: str) -> str:
    """Reject a blank value instead of silently dropping the flag.

    Design §3.2 guards with `if folder:`, which makes `--folder ""` a no-op:
    the user asked for a scope, got none, and nothing said so. `config.py`'s
    coercers already take the opposite line ("is empty; expected a filesystem
    path"), and this follows them.
    """
    text = raw.strip()
    if not text:
        raise ConfigError(f"{origin} is empty; expected {expected}.")
    return text


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


class TagRegistry:
    """The `[tags]` taxonomy, and the compiler for its pushdown filters.

    :param tags: `CliConfig.tags` - tag name to folder names or globs. Copied,
        so a later mutation of the caller's mapping cannot change the registry.
    :param source: Where the taxonomy came from, for diagnostics. Defaults to
        the config filename; a caller holding the discovered path should pass
        it, matching `config.load_config`'s origin-aware messages.
    """

    def __init__(
        self, tags: Mapping[str, Sequence[str]], *, source: str = _DEFAULT_SOURCE
    ) -> None:
        self._tags: dict[str, tuple[str, ...]] = {
            str(name): tuple(folders) for name, folders in tags.items()
        }
        self._source = source

    @property
    def tags(self) -> dict[str, tuple[str, ...]]:
        """The taxonomy as written, degenerate entries included.

        Requirement 3.5 / task 4.2's `npu-rag tags` must be able to show the
        user a broken entry; a view that hid one would hide the thing they need
        to fix.
        """
        return dict(self._tags)

    def tag_names(self) -> tuple[str, ...]:
        """Every configured tag, sorted.

        Sorted rather than in TOML order: the order of keys in a hand-written
        table carries no meaning, and a listing the user has to scan should be
        scannable. Folder order *inside* a tag is the author's and is preserved.
        """
        return tuple(sorted(self._tags))

    def resolve_tag(self, tag_name: str) -> tuple[str, ...]:
        """The folder names and globs a tag expands to (requirements 3.2, 3.3).

        :raises UnknownTagError: the tag is not defined. Requirement 3.4: the
            message lists every available tag.
        :raises EmptyTagError: the tag expands to nothing, or to a blank name.

        **Why a degenerate entry is an error** (Implementation Note 1.2 leaves
        the ruling to this task). `config.py` accepts `alpha = []`, `alpha =
        ""` and `alpha = ["", ""]` silently, and requirements 3.2 and 3.4 do
        not say what they mean. The two behaviours available without an error
        are both wrong, and wrong in ways the user cannot see:

        * A blank folder name compiles to `file_path LIKE '%%'`, which matches
          **every document in the archive**. A flag whose whole purpose is to
          narrow the search would silently widen it to everything, and the
          results would look plausible.
        * An empty expansion compiles to a predicate matching nothing, and
          "your taxonomy is broken" then reaches the user as "no results for
          your query" - indistinguishable from a genuine miss.

        Requirement 3.3 says `--tag` "expands the tag into its constituent
        folders"; a tag with no constituent folders cannot satisfy it. An error
        naming the file, the entry and the position is the only outcome that
        sends the user to the line they need to edit.
        """
        name = tag_name.strip()
        if not name:
            raise ConfigError(
                f"command-line option --tag is empty; expected a tag name. "
                f"{self._available_sentence()}"
            )
        if name not in self._tags:
            raise UnknownTagError(
                f"{self._source}: unknown tag {name!r}. {self._available_sentence()}"
            )

        folders = self._tags[name]
        if not folders:
            raise EmptyTagError(
                f"{self._source}: [tags] entry {name!r} lists no folders; a tag "
                f"must name at least one folder, or be removed."
            )
        for position, folder in enumerate(folders, start=1):
            if not folder.strip():
                raise EmptyTagError(
                    f"{self._source}: [tags] entry {name!r} has a blank folder "
                    f"name at position {position}; every entry must name a "
                    f"folder. A blank name would scope the search to every "
                    f"document instead of narrowing it."
                )
        return tuple(folder.strip() for folder in folders)

    def resolve_tags(self, tag_names: Sequence[str]) -> tuple[str, ...]:
        """The union of several tags' folders, de-duplicated.

        First-seen order is kept, so the expansion of `--tag a,b` reads in the
        order the user asked for. De-duplication matters because requirement
        3.2 is explicitly many-to-many: two tags sharing a folder would
        otherwise emit that folder's clause twice.
        """
        seen: dict[str, None] = {}
        for name in tag_names:
            for folder in self.resolve_tag(name):
                seen.setdefault(folder, None)
        return tuple(seen)

    def build_filter_sql(
        self,
        *,
        folder: str | None = None,
        tags: Sequence[str] | None = None,
        custom_filter: str | None = None,
    ) -> str | None:
        """Compile `--folder`, `--tag` and `--filter` into one SQL predicate.

        The shape is `AND` between the three clauses and `OR` *inside* a tag's
        expansion: naming a folder and a tag narrows to their intersection,
        while a tag naming three folders matches any of them. A flattened `OR`
        across everything would be indistinguishable from this whenever only
        one clause is supplied, which is why the tests exercise all seven
        non-empty combinations.

        :param folder: A folder name, relative subfolder, or glob
            (requirement 3.1). Blank is an error, not an absence.
        :param tags: Tag names to expand (requirement 3.3). An empty sequence
            is absence - that is what argparse hands over for an option never
            given.
        :param custom_filter: A raw SQL `WHERE` predicate (requirement 5.1),
            passed through unescaped by design and parenthesised so that an
            `OR` inside it cannot swallow the surrounding scope.

        :returns: The predicate, or `None` when nothing was supplied.
            Deliberately `None` and never `""`: an empty string is falsy at one
            call site and a syntax error at the next, so "no filter" must not
            be spelled as a value.
        """
        predicates: list[str] = []

        if folder is not None:
            predicates.append(
                _folder_clause(
                    _require_value(
                        folder,
                        origin="command-line option --folder",
                        expected="a folder name, relative subfolder, or glob pattern",
                    )
                )
            )

        if tags:
            expansion = " OR ".join(_folder_clause(spec) for spec in self.resolve_tags(tags))
            predicates.append(f"({expansion})")

        if custom_filter is not None:
            predicates.append(
                "("
                + _require_value(
                    custom_filter,
                    origin="command-line option --filter",
                    expected="an SQL WHERE expression",
                )
                + ")"
            )

        if not predicates:
            return None
        return " AND ".join(predicates)

    def _available_sentence(self) -> str:
        """Requirement 3.4's "listing all available tags", or why there are none."""
        if not self._tags:
            return (
                f"No tags are configured: add a [tags] table to {self._source}, "
                f'for example  research = ["Folder A", "Folder B"].'
            )
        listed = ", ".join(repr(name) for name in self.tag_names())
        return f"Available tags: {listed}."
