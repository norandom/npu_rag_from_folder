"""`npu-rag search` - the query command (task 4.1).

Design anchors: `.kiro/specs/search-cli/design.md` §3.5 (`npu-rag search
<query>`) and §4 (Error Taxonomy & Exit Codes); requirements 3.1-3.4 (folder
scoping and tag pushdown), 4.1-4.4 (grouping, context expansion,
breadcrumbs), 5.1-5.2 (retrieval controls and parse-time validation), 6.1-6.3
(provider accountability) and 8.1-8.4 (terminal and JSON output).

This module owns two things design §3.5 asks of it: the *registration* of the
search flags (`add_arguments`, called by `main.build_parser` only when `search`
is the command actually being run) and their *execution* (`run`). Everything
below the CLI is borrowed: `config.load_config` resolves the four
configuration layers, `tags.TagRegistry` compiles the pushdown filter,
`order.folder_of` names folders, `render` formats, and
`index.VectorIndex.search` does the retrieval.

Seven rulings, each with the reason, because several of them are departures.

----------------------------------------------------------------------------
Ruling 1 - `--fetch-k` is NOT offered, and that is a deviation
----------------------------------------------------------------------------

Requirement 5.1 and design §3.5 both list `--fetch-k <int>` (default 20,
"candidate pool per retriever"). `SearchOptions` has **no `fetch_k` field**:
`index/service.py:142` computes the pool itself as ``max(opts.limit * 4, 30)``
and nothing can influence it from outside. `npu_rag.index` is a closed upstream
spec (`vector-index`) and outside this task's boundary, so the three available
courses were: accept the flag and ignore it, reach into `index/` to add the
field, or omit the flag. The first is a flag that lies to the user - the worst
of the three and the only one that cannot be noticed - and the second is a
downstream patch where the fix belongs upstream. So the flag is **omitted**
and the gap is routed to the `vector-index` spec, which owns `SearchOptions`.

`--top-k` reaches the pool indirectly (the pool is four times the limit), which
is why nothing here is silently degraded: the depth still scales with what the
user asked for, it just cannot be set independently.

----------------------------------------------------------------------------
Ruling 2 - `--mode` keeps requirement 5.1's vocabulary, not the enum's
----------------------------------------------------------------------------

Requirement 5.1 spells the modes ``dense|fts|hybrid`` and `config.MODE_CHOICES`
already validates `default_mode` against exactly those three, so that is the
user-facing vocabulary and :data:`SEARCH_MODES` translates it to
`SearchMode`'s own spelling (``dense_only``/``lexical_only``/``hybrid``). The
two differ, so the mapping is written out as a literal and a test pins each
target against the real enum - a mapping derived from the enum would accept
``--mode lexical_only`` and reject the ``fts`` the requirement promises.

`--metric` and `--fusion` need no such translation: their spellings and the
enum values coincide. They are still spelled here as literals, for the reason
given under ruling 7 - importing `npu_rag.index.types` to read them would put
lancedb on the `--help` path.

----------------------------------------------------------------------------
Ruling 3 - numeric bounds are enforced here, at parse time
----------------------------------------------------------------------------

Requirement 5.2 says "strictly validated at parse time", and implementation
note 1.2 explains why this cannot be delegated: `CliConfig` validates only
inside `load_config()`, so `replace(cfg, default_limit=args.top_k)` would
validate nothing. The bounds are therefore `argparse` `type=` callables, which
route through `_CliParser.error()` and so exit **1** per design §4.

What is rejected, and why that boundary:

* ``--top-k``, ``--rrf-k``, ``--max-chunks-per-doc``: integers ``>= 1``.
  A limit of zero is refused by `VectorIndex.search` itself (`QueryError`),
  which would reach the user as exit 2 for what is plainly a bad argument. For
  ``--rrf-k`` the boundary is also the safe one: `fusion.py:110` divides by
  ``k + rank + 1`` with ``rank`` from 0, so a negative ``k`` is a
  `ZeroDivisionError` waiting for the first result, and ``k = 0`` removes the
  rank smoothing that is the whole of what RRF's constant does. Requirement
  5.1's own default is 60.
* ``--context``: integer ``>= 0``. Zero is the documented default ("attach no
  neighbours") so it must be accepted; a negative window has no meaning and
  `ContextExpander` would silently treat it as none.
* ``--mmr`` and ``--linear-alpha``: floats in ``[0.0, 1.0]``. Both ends are
  meaningful and both are accepted. Out of range is rejected rather than
  clamped because the engine's two halves disagree about it:
  `service.py:217` *clamps* ``1.0 - mmr`` into range, so ``--mmr 1.5`` would
  silently behave as ``--mmr 1.0``, while `LinearCombination.__init__` *raises
  a bare `ValueError`* for an alpha outside the interval, which would reach the
  user as a traceback. ``nan`` fails the range comparison and is rejected with
  the same message.

----------------------------------------------------------------------------
Ruling 4 - the provider reported is the one that SERVED, and a fallback is said
----------------------------------------------------------------------------

Requirement 6.3 wants the *active* execution provider, and implementation note
3.1 is emphatic that `render` reports verbatim and will never upgrade a request
into a claim: ``[Provider: npu]`` in the output is the signal this wiring is
broken. So nothing here ever passes the ``npu|cpu|auto`` alias on. What is
passed is :data:`NPU_ORT_PROVIDER`/:data:`CPU_ORT_PROVIDER` - the ONNX Runtime
names - chosen by the `ProviderChoice` that `EmbedResult.provider_served`
reports after the query vector has actually been computed.

That is also why the embedder is a thin adapter (:class:`_ServiceQueryEmbedder`)
rather than the `EmbeddingService` itself. `index/service.py:40` calls
``embed_queries([query], provider=ProviderChoice.AUTO)`` with the choice
**hardcoded**, so handing the service over directly would quietly discard
``--provider``; the adapter exposes `embed_query`, which that same function
prefers, and so keeps both the user's selection and the served provider
reachable.

Requirement 6.2's "never silently fall back" is discharged in three places:

* ``--provider npu`` on a machine where the NPU is unusable **fails** with
  `NpuUnavailableError`, exit 2, naming the unmet condition. It fails *before*
  `build_service`, because the NPU artifact path begins with a compile measured
  in tens of minutes and failing after that would be indistinguishable from a
  hang.
* ``--provider auto`` that lands on the CPU still succeeds - that is what
  `auto` means - but prints :data:`FALLBACK_NOTICE_PREFIX` and the specific
  reason to **stderr** first, so a fallback is never silent and the reason
  never pollutes a `--json` stdout.
* a completed dense or hybrid query that recorded **no** provider is a runtime
  error (:class:`ProviderAttributionError`, exit 2) rather than output with a
  guessed attribution. Requirement 6.3 has no "sometimes".

`--mode fts` is the one execution with no provider to report: no ONNX Runtime
session is created at all, because lexical retrieval never embeds. Reporting
``CPUExecutionProvider`` there would be a fabrication, so it reports
:data:`NO_EMBEDDING_PROVIDER`, which is honest and still non-blank (`render`
refuses a blank attribution).

----------------------------------------------------------------------------
Ruling 5 - the archive root is resolved, the index is checked before it is opened
----------------------------------------------------------------------------

Implementation note 2.2 binds this task: `CliConfig.archive_path` is
`Path(".")` by default, and a relative root cannot match an absolute
`file_path`, so every folder label would silently degrade to a full parent
path. It is passed `.expanduser().resolve()`.

The index is checked for existence *before* anything opens it, by
:func:`missing_index_message`, because `LanceIndexStore.__init__`
(`index/store.py:26-37`) **creates** the directory and an empty ``chunks``
table. A first-run ``npu-rag search`` would otherwise report "No results." for
every query, having silently created an empty database - a success message for
a state the user has to be told about. The diagnostic names the path and the
command that fixes it, and exits 2: design §4 files index-database problems
under "Hardware / Runtime Error", and nothing the user typed was wrong.

----------------------------------------------------------------------------
Ruling 6 - `folder_tags` is built here, and keyed exactly as `folder_of` labels
----------------------------------------------------------------------------

`TagRegistry` has no reverse lookup and `tags.py` is outside this boundary, so
:func:`folder_tag_map` owns folder -> tags. Note 3.1: the keys must be
precisely what `order.folder_of` produced for the results being rendered, or
the ``(Tag: siddha)`` annotation silently does nothing. The matching vocabulary
is `tags.py`'s - ``*`` and ``?`` are wildcards, ``%``, ``_`` and ``[`` are
literal - rather than `fnmatch`'s, which would read ``A[1]`` as a character
class. A spec naming a subfolder (``Parent/Child``) matches on its first
component, since that is the component `folder_of` returns. Degenerate `[tags]`
entries are skipped rather than raised on, because an annotation is not worth
breaking an unrelated search for (note 2.1's lazy-validation rule).

----------------------------------------------------------------------------
Ruling 7 - startup cost: nothing heavy at module scope
----------------------------------------------------------------------------

Requirement 1.2's fast paths include ``npu-rag search --help``, which imports
this module (that is how the flags get registered). `lancedb`, `onnxruntime`,
`torch` and `transformers` are therefore imported **inside** `run()` and its
helpers only, and the names needed for annotations are confined to
`TYPE_CHECKING`. `npu_rag.cli.render`, `order`, `tags`, `config` and `main` are
all import-light and are imported normally. A probe test pins this.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from npu_rag.cli import render
from npu_rag.cli.config import (
    MODE_CHOICES,
    PROVIDER_CHOICES,
    CliConfig,
    ConfigError,
    emit_config_error,
    load_config,
)
from npu_rag.cli.main import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, PROGRAM_NAME
from npu_rag.cli.order import folder_of
from npu_rag.cli.tags import PATH_SEPARATORS, TagRegistry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from npu_rag.embedding.service import EmbeddingService
    from npu_rag.embedding.types import ProviderChoice
    from npu_rag.index.types import SearchOptions, SearchResponse

__all__ = [
    "ARTIFACTS_DIRNAME",
    "CPU_ORT_PROVIDER",
    "EXIT_CODES",
    "FALLBACK_NOTICE_PREFIX",
    "FUSION_CHOICES",
    "LEXICAL_MODE",
    "METRIC_CHOICES",
    "NO_EMBEDDING_PROVIDER",
    "NPU_ORT_PROVIDER",
    "SEARCH_MODES",
    "ProviderAttributionError",
    "add_arguments",
    "artifacts_root",
    "folder_tag_map",
    "missing_index_message",
    "open_index_at",
    "open_query_embedder",
    "run",
]

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

#: Requirement 5.1's `--mode` spelling -> the `SearchMode` value it selects.
#: See ruling 2; the key set is `config.MODE_CHOICES` and a test pins both
#: sides against literals and against the enum.
SEARCH_MODES: Final[dict[str, str]] = {
    "dense": "dense_only",
    "fts": "lexical_only",
    "hybrid": "hybrid",
}

#: The one mode that embeds nothing, in the user's spelling.
LEXICAL_MODE: Final = "fts"

#: Requirement 5.1's `--metric` and `--fusion` vocabularies, which coincide
#: with `DistanceMetric` and `FusionMethod`. Spelled as literals so that
#: `search --help` does not import lancedb (ruling 7); a test compares them
#: with the enums.
METRIC_CHOICES: Final[tuple[str, ...]] = ("cosine", "dot", "l2")
FUSION_CHOICES: Final[tuple[str, ...]] = ("rrf", "linear")

#: The ONNX Runtime provider names requirement 6.3 reports. Taken from
#: `render`, which already cross-checks both spellings against the modules that
#: own them, so there is one spelling in the CLI and not three.
NPU_ORT_PROVIDER: Final = render.NPU_PROVIDER
CPU_ORT_PROVIDER: Final = render.CPU_PROVIDER

#: What a lexical-only search reports (ruling 4). Non-blank, because `render`
#: refuses a blank attribution, and deliberately not an ONNX Runtime name,
#: because no session ran.
NO_EMBEDDING_PROVIDER: Final = "none (lexical-only search ran no embedding model)"

#: Requirement 6.2's audible half of an `auto` fallback.
FALLBACK_NOTICE_PREFIX: Final = (
    "the NPU was not used; this query ran on the CPU. Reason: "
)

#: Where prepared model artifacts live, relative to the index's parent. The
#: index is the only user-configured location the search command has, and
#: `CliConfig` carries no artifacts field, so the two sit side by side:
#: `.npu_rag/lancedb` and `.npu_rag/artifacts` under the default configuration.
#: Tasks 5.1 and 5.2 must use `artifacts_root()` rather than re-deriving this.
ARTIFACTS_DIRNAME: Final = "artifacts"

#: Requirement 5.1's defaults, as design §3.5 and the requirement spell them.
DEFAULT_METRIC: Final = "cosine"
DEFAULT_FUSION: Final = "rrf"
DEFAULT_RRF_K: Final = 60
DEFAULT_LINEAR_ALPHA: Final = 0.5
DEFAULT_CONTEXT: Final = 0

#: LanceDB writes one `<table>.lance` directory per table inside the database
#: directory, so the presence of any of them is what distinguishes a real index
#: from an empty folder. Matched as a glob rather than by table name, so this
#: check does not silently depend on `LanceIndexStore`'s default table.
_TABLE_GLOB: Final = "*.lance"

#: `tags.py`'s glob vocabulary (`GLOB_CHARS`), as regular-expression fragments.
_GLOB_REGEX: Final[dict[str, str]] = {"*": ".*", "?": "."}


class ProviderAttributionError(Exception):
    """A query completed without recording which provider served it.

    Requirement 6.3 reports the active provider for *every* execution, so an
    unattributable run is a runtime failure (exit 2) and not output with a
    plausible guess. It is its own type so that the diagnostic cannot be
    confused with an embedding or storage failure.
    """


# --------------------------------------------------------------------------- #
# Seams
# --------------------------------------------------------------------------- #


class QueryEmbedder(Protocol):
    """What `run` needs of an embedder: a query vector, and attribution.

    `index/service.py:33-44` prefers `embed_query` over `embed_queries`, which
    is what lets the adapter below keep the user's `--provider` selection (see
    ruling 4). `provider_served` and `fallback_reason` are filled in by the
    call, so both are `None` until a query has actually been embedded.
    """

    provider_served: str | None
    fallback_reason: str | None

    def embed_query(self, query: str) -> list[float]: ...


class IndexSearcher(Protocol):
    """The one method this command uses on `VectorIndex`."""

    def search(
        self,
        query: str,
        options: SearchOptions | None = ...,
        embed_service: object | None = ...,
    ) -> SearchResponse: ...


if TYPE_CHECKING:
    ConfigLoader = Callable[..., CliConfig]
    IndexOpener = Callable[[Path], IndexSearcher]
    EmbedderOpener = Callable[[CliConfig], QueryEmbedder]


# --------------------------------------------------------------------------- #
# Requirement 5.2 - parse-time numeric validation (ruling 3)
# --------------------------------------------------------------------------- #


def _positive_int(raw: str) -> int:
    """An integer of 1 or more."""
    return _bounded_int(raw, minimum=1, described="a positive integer (1 or greater)")


def _non_negative_int(raw: str) -> int:
    """An integer of 0 or more."""
    return _bounded_int(raw, minimum=0, described="a whole number (0 or greater)")


def _bounded_int(raw: str, *, minimum: int, described: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be {described}; got {raw!r}") from None
    if value < minimum:
        raise argparse.ArgumentTypeError(f"must be {described}; got {value}")
    return value


def _unit_interval(raw: str) -> float:
    """A float in `[0.0, 1.0]`, both ends included.

    `nan` parses as a float and then fails the comparison, which is the
    intended outcome: a diversity factor that compares false against every
    bound would otherwise flow into the engine and silently disable MMR.
    """
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a number between 0.0 and 1.0; got {raw!r}"
        ) from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"must be between 0.0 and 1.0 inclusive; got {raw}"
        )
    return value


# --------------------------------------------------------------------------- #
# Registration (design §3.5)
# --------------------------------------------------------------------------- #


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register `search`'s flags on its subparser.

    Called by `main.build_parser` through `main.ARGUMENTS_ATTR`, and only for
    the command actually present in `argv` - see that module for why the hook
    exists and what it must not cost.

    `--top-k`, `--mode` and `--provider` default to `None` rather than to their
    documented values. That is requirement 2.1: `config.load_config` reads
    `None` as "this layer supplied nothing", so an environment variable or
    `npu-rag.toml` can still decide them. A literal default here would make the
    flag always win and silently retire the two lower layers.
    """
    parser.add_argument(
        "query",
        help="The text to search for. Quote it if it contains spaces.",
    )

    scope = parser.add_argument_group("folder scoping and tag taxonomy")
    scope.add_argument(
        "--folder",
        metavar="<name|glob>",
        help=(
            "Restrict the search to one folder, relative subfolder, or glob "
            "(for example 'Bhagwan Garuda*'). Requirement 3.1."
        ),
    )
    scope.add_argument(
        "--tag",
        action="append",
        metavar="<name[,name]>",
        help=(
            "Restrict the search to the folders a configured tag names. "
            "Repeatable, and accepts a comma-separated list. Requirement 3.3."
        ),
    )
    scope.add_argument(
        "--filter",
        metavar="<sql>",
        help="An additional SQL WHERE predicate, ANDed with the scope above.",
    )

    presentation = parser.add_argument_group("presentation")
    presentation.add_argument(
        "--group-by-folder",
        action="store_true",
        help=(
            "Group hits under their folder and restore each folder's narrative "
            "order (session numbers, dates) instead of score order."
        ),
    )
    presentation.add_argument(
        "--context",
        type=_non_negative_int,
        default=DEFAULT_CONTEXT,
        metavar="<n>",
        help=(
            "Attach N preceding and N succeeding chunks to each hit "
            f"(default {DEFAULT_CONTEXT})."
        ),
    )
    presentation.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON on stdout instead of formatted text.",
    )
    presentation.add_argument(
        "--no-color",
        action="store_true",
        help="Never emit ANSI styling, even on a terminal that supports it.",
    )

    retrieval = parser.add_argument_group("retrieval controls")
    retrieval.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        help=(
            "Retrieval mode: 'dense' vectors only, 'fts' full-text only, or "
            "'hybrid' (the default, and whatever the configuration says)."
        ),
    )
    retrieval.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default=DEFAULT_METRIC,
        help=f"Dense distance metric (default {DEFAULT_METRIC}).",
    )
    retrieval.add_argument(
        "--fusion",
        choices=FUSION_CHOICES,
        default=DEFAULT_FUSION,
        help=f"How hybrid rankings are combined (default {DEFAULT_FUSION}).",
    )
    retrieval.add_argument(
        "--rrf-k",
        type=_positive_int,
        default=DEFAULT_RRF_K,
        metavar="<k>",
        help=f"Reciprocal-rank-fusion smoothing constant (default {DEFAULT_RRF_K}).",
    )
    retrieval.add_argument(
        "--linear-alpha",
        type=_unit_interval,
        default=DEFAULT_LINEAR_ALPHA,
        metavar="<alpha>",
        help=(
            "Dense weight for linear fusion, between 0.0 and 1.0 "
            f"(default {DEFAULT_LINEAR_ALPHA}). Only used with --fusion linear."
        ),
    )
    retrieval.add_argument(
        "--mmr",
        type=_unit_interval,
        metavar="<lambda>",
        help=(
            "Maximal-marginal-relevance diversity between 0.0 and 1.0. "
            "Opt-in: omitted, no diversity re-ranking is applied."
        ),
    )
    retrieval.add_argument(
        "--max-chunks-per-doc",
        type=_positive_int,
        metavar="<n>",
        help="Keep at most N chunks from any one document.",
    )
    retrieval.add_argument(
        "--rollup",
        action="store_true",
        help="Aggregate chunk scores to document level before truncating.",
    )
    retrieval.add_argument(
        "--top-k",
        type=_positive_int,
        metavar="<n>",
        help=(
            "How many results to return. Defaults to the configured "
            "default_limit, which is 10 unless set."
        ),
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        help=(
            "Where the query is embedded: 'npu' (fail if the NPU is "
            "unavailable), 'cpu', or 'auto' (prefer the NPU, and say so when "
            "it cannot be used). Defaults to the configured provider."
        ),
    )


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def artifacts_root(config: CliConfig) -> Path:
    """Where prepared model artifacts live for this configuration.

    A sibling of the index directory, so one configured location puts the
    database and the compiled graphs together: `.npu_rag/lancedb` and
    `.npu_rag/artifacts` by default, and both move when `index_path` does.
    """
    return config.index_path.expanduser().resolve().parent / ARTIFACTS_DIRNAME


def missing_index_message(index_path: Path) -> str | None:
    """A diagnostic when `index_path` holds no index yet, else `None`.

    Ruling 5: this runs *before* anything opens the store, because opening one
    creates it. The message names the path and the command that fixes it, since
    "no results" is what the user would otherwise see on their first search.
    """
    if index_path.is_dir() and any(index_path.glob(_TABLE_GLOB)):
        return None
    return (
        f"no search index found at {index_path}. "
        f"Run `{PROGRAM_NAME} index` first to build one, or point "
        f"NPU_RAG_INDEX_PATH (or index_path in npu-rag.toml) at an existing index."
    )


# --------------------------------------------------------------------------- #
# Ruling 6 - folder -> tags, keyed as `order.folder_of` labels
# --------------------------------------------------------------------------- #


def _first_component(spec: str) -> str:
    """The first path component of a folder spec, either separator spelling."""
    text = spec
    for separator in PATH_SEPARATORS:
        text = text.replace(separator, "\x00")
    for part in text.split("\x00"):
        if part.strip():
            return part.strip()
    return ""


def _spec_pattern(spec: str) -> re.Pattern[str]:
    """A whole-string matcher for one folder spec, in `tags.py`'s vocabulary.

    `*` and `?` are the wildcards; every other character - `%`, `_`, `[` and
    the rest - is literal. `fnmatch` is deliberately not used: it would read
    `A[1]` as a character class, so a folder actually named `A[1]` would stop
    being annotated, and `tags.py`'s SQL filter would disagree with the label.
    """
    out = [_GLOB_REGEX.get(char, re.escape(char)) for char in os.path.normcase(spec)]
    return re.compile("".join(out) + r"\Z")


def folder_tag_map(
    folders: Iterable[str], tags: Mapping[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """Map each folder label to the tags that name it (requirement 3.2).

    :param folders: Labels exactly as `order.folder_of` produced them - the
        first path component beneath the resolved archive root. Note 3.1: a key
        that does not match a produced label annotates nothing, silently, which
        is why the caller passes the labels rather than this function guessing
        them.
    :param tags: `CliConfig.tags`, as written. Degenerate entries - a tag
        naming no folders, or a blank folder name - are skipped: `--tag` makes
        them an error through `TagRegistry.resolve_tag`, but an *annotation* is
        not worth failing an unrelated search for.
    :returns: Folder label to tag names, each folder's tags in sorted tag
        order, folders with no tag absent entirely.
    """
    labels = list(folders)
    mapping: dict[str, list[str]] = {}
    for tag in sorted(tags):
        for spec in tags[tag]:
            component = _first_component(spec)
            if not component:
                continue
            pattern = _spec_pattern(component)
            for label in labels:
                if pattern.match(os.path.normcase(label)):
                    carried = mapping.setdefault(label, [])
                    if tag not in carried:
                        carried.append(tag)
    return {label: tuple(names) for label, names in mapping.items()}


# --------------------------------------------------------------------------- #
# The default seams (ruling 4, ruling 7)
# --------------------------------------------------------------------------- #


class _ServiceQueryEmbedder:
    """Adapts `EmbeddingService` to `QueryEmbedder`, recording what served.

    Two jobs, both forced by `index/service.py`. It exposes `embed_query`,
    which `_get_query_vector` prefers, so the `--provider` selection is not
    replaced by the `ProviderChoice.AUTO` that function hardcodes on the
    `embed_queries` path. And it keeps `provider_served` and `fallback_reason`
    from the `EmbedResult`, which `SearchResponse` has nowhere to carry and
    requirements 6.2 and 6.3 both need.
    """

    __slots__ = ("_provider", "_service", "fallback_reason", "provider_served")

    def __init__(self, service: EmbeddingService, provider: ProviderChoice) -> None:
        self._service = service
        self._provider = provider
        self.provider_served: str | None = None
        self.fallback_reason: str | None = None

    def embed_query(self, query: str) -> list[float]:
        result = self._service.embed_queries([query], self._provider)
        self.provider_served = str(result.provider_served)
        self.fallback_reason = result.fallback_reason
        return [float(value) for value in result.vectors[0]]


def open_index_at(index_path: Path) -> IndexSearcher:
    """Open the vector index at `index_path` (lancedb enters the process here)."""
    from npu_rag.index.service import VectorIndex

    return VectorIndex(index_path)


def open_query_embedder(config: CliConfig) -> QueryEmbedder:
    """Build the query embedder for `config.provider` (requirements 6.1, 6.2).

    An explicit `--provider npu` is checked against the capability report
    *before* `build_service`, which would otherwise acquire, export and compile
    for the NPU - tens of minutes - only to be refused by `resolve_backend` at
    the first `embed_queries`. Under `auto` the same report decides which
    artifact to prepare, while the request handed to `embed_queries` stays
    `auto`, so `EmbedResult.fallback_reason` is still produced by the one
    component that owns it.

    :raises ConfigError: for a `model_name` no profile declares - user
        configuration, so exit 1 rather than a `KeyError` traceback.
    :raises NpuUnavailableError: for an explicit NPU request the machine cannot
        honour (exit 2, requirement 6.2).
    """
    from npu_rag.embedding.environment.capability import check_capability
    from npu_rag.embedding.errors import NpuUnavailableError
    from npu_rag.embedding.profiles import PROFILES, profile_for
    from npu_rag.embedding.providers.base import npu_unavailable_reason
    from npu_rag.embedding.service import build_service
    from npu_rag.embedding.types import ProviderChoice

    requested = ProviderChoice(config.provider)
    try:
        profile = profile_for(config.model_name)
    except KeyError:
        declared = ", ".join(repr(name) for name in sorted(PROFILES))
        raise ConfigError(
            f"unknown model {config.model_name!r}. Declared models: {declared}."
        ) from None

    capability = check_capability()
    reason = npu_unavailable_reason(capability)
    if requested is ProviderChoice.NPU and reason is not None:
        raise NpuUnavailableError(
            f"provider 'npu' was requested explicitly and the NPU is not usable: "
            f"{reason}. No other provider will be substituted; re-run with "
            f"--provider cpu to search on the CPU, or repair the NPU "
            f"provisioning (uv run python -m tools.provision_npu).",
            provider=requested,
            model_id=profile.model_id,
        )

    if requested is ProviderChoice.CPU or reason is not None:
        artifact_provider = ProviderChoice.CPU
    else:
        artifact_provider = ProviderChoice.NPU
    service = build_service(
        profile, capability, artifacts_root(config), provider=artifact_provider
    )
    return _ServiceQueryEmbedder(service, requested)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _fail(message: str, code: int) -> int:
    """A diagnostic on stderr and nothing on stdout - `main._fail`'s format.

    Spelled here rather than imported: `main._fail` is private to the
    dispatcher, and `config.emit_config_error` already duplicates the same
    single line for the same reason. A test pins all three to one shape.
    """
    sys.stderr.write(f"{PROGRAM_NAME}: error: {message}\n")
    return code


def _notice(message: str) -> None:
    """A non-error remark on stderr, so `--json` stdout stays parseable."""
    sys.stderr.write(f"{PROGRAM_NAME}: notice: {message}\n")


def _tag_names(raw: Sequence[str] | None) -> list[str]:
    """`--tag a,b --tag c` as one list, splitting on commas (requirement 3.3).

    Blanks are **kept**, not dropped: `--tag ""` and a trailing comma both
    reach `TagRegistry.resolve_tag`, which diagnoses them. Dropping them here
    would make a flag whose only job is to narrow do nothing at all, which is
    the failure mode `tags._require_value` exists to prevent.
    """
    if not raw:
        return []
    return [name for entry in raw for name in entry.split(",")]


def _resolved_mode(args: argparse.Namespace, config: CliConfig) -> str:
    """The effective `--mode`, in requirement 5.1's spelling.

    Note 1.2: a `CliConfig` built outside `load_config()` is unvalidated, so
    `default_mode` is checked here rather than assumed - the message names
    where the value came from, as requirement 2.3 asks.
    """
    mode: str | None = args.mode
    if mode is None:
        mode = config.default_mode
    if mode not in SEARCH_MODES:
        listed = ", ".join(repr(name) for name in MODE_CHOICES)
        raise ConfigError(
            f"configured default_mode must be one of {listed}; got {mode!r}."
        )
    return mode


def _build_options(
    args: argparse.Namespace, config: CliConfig, *, mode: str, filter_sql: str | None
) -> SearchOptions:
    """Every flag of requirement 5.1, on the `SearchOptions` field it controls."""
    from npu_rag.index.types import (
        DistanceMetric,
        FusionMethod,
        SearchMode,
        SearchOptions,
    )

    limit = config.default_limit if args.top_k is None else args.top_k
    return SearchOptions(
        limit=limit,
        mode=SearchMode(SEARCH_MODES[mode]),
        metric=DistanceMetric(args.metric),
        filter_sql=filter_sql,
        fusion_method=FusionMethod(args.fusion),
        rrf_k=args.rrf_k,
        linear_alpha=args.linear_alpha,
        mmr_diversity=args.mmr,
        max_chunks_per_document=args.max_chunks_per_doc,
        document_rollup=args.rollup,
        context_window=args.context,
    )


def _provider_label(embedder: QueryEmbedder | None) -> str:
    """The ONNX Runtime name of whatever served, or a refusal (ruling 4)."""
    if embedder is None:
        return NO_EMBEDDING_PROVIDER
    served = embedder.provider_served
    if served is None:
        raise ProviderAttributionError(
            "the query completed without recording which execution provider "
            "served it, so the result cannot be attributed (requirement 6.3)."
        )
    if served == "npu":
        return NPU_ORT_PROVIDER
    if served == "cpu":
        return CPU_ORT_PROVIDER
    raise ProviderAttributionError(
        f"the embedding runtime reported an unknown execution provider "
        f"{served!r}; expected 'npu' or 'cpu' (requirement 6.3)."
    )


def _render(
    response: SearchResponse,
    args: argparse.Namespace,
    config: CliConfig,
    *,
    provider: str,
    archive_root: Path,
) -> None:
    """Format and print, through `render.emit` (note 3.1, requirement 8.4)."""
    folders = {
        folder_of(result.file_path, archive_root=archive_root)
        for result in response.results
    }
    color = render.should_colorize(
        prefer_color=config.ansi_colors and not args.no_color
    )
    render.emit(
        render.render_search(
            response,
            provider=provider,
            as_json=args.json,
            archive_root=archive_root,
            group_by_folder=args.group_by_folder,
            folder_tags=folder_tag_map(folders, config.tags),
            color=color,
        )
    )


def run(
    args: argparse.Namespace,
    *,
    load: ConfigLoader = load_config,
    open_index: IndexOpener = open_index_at,
    open_embedder: EmbedderOpener = open_query_embedder,
) -> int:
    """Execute `npu-rag search` and return its exit code (design §3.5, §4).

    The order of operations is deliberate and is what keeps the exit codes
    honest: everything the user can get wrong is diagnosed before anything
    expensive or stateful happens.

    1. configuration, then the query text, then the tag expansion and pushdown
       filter - all `ConfigError`, all exit 1, all before the index is touched;
    2. the index's existence (ruling 5), exit 2, before a store is constructed
       and therefore before an empty one can be created;
    3. the embedder, which is where an explicit NPU request fails (exit 2) and
       where an `auto` fallback acquires its reason;
    4. the search, the attribution check, the fallback notice, the output.

    The three seams are injectable for testing only; their defaults are the
    production path. `main.dispatch` calls this with the namespace alone.
    """
    try:
        config = load(cli_args=vars(args))
        query = str(args.query)
        if not query.strip():
            raise ConfigError(
                "the search query is empty; name the text to search for."
            )
        mode = _resolved_mode(args, config)
        filter_sql = TagRegistry(config.tags).build_filter_sql(
            folder=args.folder,
            tags=_tag_names(args.tag),
            custom_filter=args.filter,
        )
        options = _build_options(args, config, mode=mode, filter_sql=filter_sql)
    except ConfigError as error:
        return emit_config_error(error)

    index_path = config.index_path.expanduser().resolve()
    absent = missing_index_message(index_path)
    if absent is not None:
        return _fail(absent, EXIT_RUNTIME)

    from npu_rag.embedding.errors import EmbeddingRuntimeError
    from npu_rag.index.errors import VectorIndexError

    try:
        embedder = None if mode == LEXICAL_MODE else open_embedder(config)
        index = open_index(index_path)
        response = index.search(query, options, embedder)
        provider = _provider_label(embedder)
        if embedder is not None and embedder.fallback_reason is not None:
            _notice(f"{FALLBACK_NOTICE_PREFIX}{embedder.fallback_reason}")
        _render(
            response,
            args,
            config,
            provider=provider,
            archive_root=config.archive_path.expanduser().resolve(),
        )
    except ConfigError as error:
        return emit_config_error(error)
    except ProviderAttributionError as error:
        return _fail(str(error), EXIT_RUNTIME)
    except (EmbeddingRuntimeError, VectorIndexError, OSError) as error:
        return _fail(str(error), EXIT_RUNTIME)
    return EXIT_OK


EXIT_CODES: Final[tuple[int, int, int]] = (EXIT_OK, EXIT_USAGE, EXIT_RUNTIME)
"""Design §4's taxonomy, re-exported so a test can pin this module to it."""
