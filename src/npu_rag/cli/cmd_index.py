"""`npu-rag index`, ingest half: configuration -> `RunReport` -> summary (task 5.1a).

Design anchors: `.kiro/specs/search-cli/design.md` §3.5 (`npu-rag index`) and
§4 (Error Taxonomy & Exit Codes); requirements 7.1-7.5, 8.3 and 2.1.

**Scope.** This module discovers, extracts and chunks; it does not embed and it
does not write an index. Task 5.1b adds `build_service`, `VectorIndex.sync`
and the live progress counter of requirement 7.3. Nothing here is a stub: the
summary states `index_written: false` in both renderings, so a user is never
left to infer from a clean exit that their index was updated.

Five rulings shape the module.

**1. `token_budget` is derived from the model profile and has no configuration
layer.** It must equal the compiled sequence length of the graph that will
embed these chunks, or the runtime silently truncates text the chunker thought
it had room for - the defect `npu-embedding-runtime` requirement 3.6 exists to
prevent. `build_ingest_config` therefore takes a `ModelProfile` and reads
`max_input_tokens` off it, and a test pins that *derivation* by passing a
profile with a different sequence length, never the number 512.

**2. The state store follows `index_path`, not the process's directory.**
`IngestConfig.state_path` defaults to a *relative* `.npu_rag/ingest.sqlite`,
so requirement 7.2's incremental skip would silently key off wherever the user
happened to `cd`. It is anchored beside the index instead - the same reasoning
and the same shape as `cmd_search.artifacts_root` (note 4.1).

**3. A completed run with omissions is a success.** `describer=None` makes
every image a `vision_unavailable` omission naming the capability it wanted,
and a corrupt file becomes a `failed` omission while its siblings still
commit. Hiding that behind a non-zero exit would make requirement 7.5's report
pointless, so the exit code says whether the *run* completed and the summary
says what was left out. Design §4's 1 and 2 keep their stated meanings: a
value the user can fix, and a failure of the machine.

**4. A missing archive root is exit 1, diagnosed before discovery.**
`ingest/discover.py` turns an absent root into a `ROOT_UNAVAILABLE` omission
and never raises, so without this check `npu-rag index --archive-path typo`
would exit 0 having indexed nothing - the silent-degradation failure mode.

**5. Printing goes through `render.emit`** (note 3.1), the only place where
requirement 8.4's UTF-8 enforcement and the write are inseparable. The summary
is deliberately uncoloured: design §3.5 gives `index` no `--no-color` flag,
and adding one would widen a flag surface the design fixes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING, Final, Protocol

from npu_rag.cli import render
from npu_rag.cli.config import (
    PROVIDER_CHOICES,
    CliConfig,
    ConfigError,
    emit_config_error,
    load_config,
)
from npu_rag.cli.main import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, PROGRAM_NAME

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from npu_rag.embedding.profiles import ModelProfile
    from npu_rag.embedding.tokenize import ModelTokenizer
    from npu_rag.ingest.config import IngestConfig
    from npu_rag.ingest.types import RunReport

__all__ = [
    "DURATION_DIGITS",
    "EXIT_CODES",
    "FAILURE_DETAIL_LIMIT",
    "NO_EMBEDDING_PROVIDER",
    "REBUILD_NOTICE",
    "STATE_FILENAME",
    "add_arguments",
    "build_ingest_config",
    "load_profile",
    "missing_archive_message",
    "open_tokenizer",
    "run",
    "run_pipeline",
    "state_path_for",
    "summary_payload",
    "render_summary",
]

#: The ingest state database, a sibling of the index directory (ruling 2). The
#: basename matches `IngestConfig`'s own default so an existing
#: `.npu_rag/ingest.sqlite` is still found under the default configuration.
STATE_FILENAME: Final = "ingest.sqlite"

#: Seconds are reported to milliseconds, mirroring `render`'s elapsed precision
#: for search. Pinned against a literal in the tests.
DURATION_DIGITS: Final = 3

#: What this phase reports for the active execution provider. Note 3.1 forbids
#: passing the `npu|cpu|auto` *alias* off as an attribution, and no embedding
#: session runs here, so the honest answer names the absence.
NO_EMBEDDING_PROVIDER: Final = "none (no embedding model ran in this phase)"

#: How many failing files the terminal summary names before it says how many
#: more there were. The JSON payload lists them all: a script can page through
#: 1343 of them, a terminal cannot.
FAILURE_DETAIL_LIMIT: Final = 10

#: Said once, on stderr, when `--full-rebuild` is given: the flag parses and is
#: carried into the summary, but nothing consumes it until 5.1b.
REBUILD_NOTICE: Final = (
    "--full-rebuild was accepted and recorded, but no index was written: "
    "embedding and index synchronisation are task 5.1b."
)


# --------------------------------------------------------------------------- #
# Seams
# --------------------------------------------------------------------------- #


class _Ingester(Protocol):
    def __call__(self, config: IngestConfig, tokenizer: ModelTokenizer) -> RunReport: ...


if TYPE_CHECKING:
    ConfigLoader = Callable[..., CliConfig]
    ProfileLoader = Callable[[str], ModelProfile]
    TokenizerLoader = Callable[[ModelProfile], ModelTokenizer]


# --------------------------------------------------------------------------- #
# Registration (design §3.5)
# --------------------------------------------------------------------------- #


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register `index`'s four flags on its subparser.

    Called by `main.build_parser` through `main.ARGUMENTS_ATTR`, for the one
    command present in `argv` and no other.

    `--archive-path` and `--provider` default to `None`, not to a value:
    `config.load_config` reads `None` as "this layer supplied nothing", so
    `NPU_RAG_ARCHIVE_PATH` and `npu-rag.toml` can still decide them. A literal
    default here would make the flag always win and retire requirement 2.1's
    two lower layers. `--full-rebuild` and `--json` have no configuration field
    behind them, so `store_true`'s `False` retires nothing.
    """
    parser.add_argument(
        "--archive-path",
        metavar="<dir>",
        help=(
            "The folder to index. Defaults to the configured archive_path "
            "(NPU_RAG_ARCHIVE_PATH, or archive_path in npu-rag.toml)."
        ),
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help=(
            "Discard the existing index and re-index every file instead of "
            "skipping unchanged ones (requirement 7.3). Recorded in this "
            "release's summary; the rebuild itself lands with index writing."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        help=(
            "Where embeddings will be computed: 'npu' (fail if the NPU is "
            "unavailable), 'cpu', or 'auto'. Defaults to the configured "
            "provider. No embedding runs in this release."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the run summary as JSON on stdout (requirement 8.3).",
    )


# --------------------------------------------------------------------------- #
# The config bridge (ruling 1)
# --------------------------------------------------------------------------- #


def load_profile(model_name: str) -> ModelProfile:
    """The `ModelProfile` for `model_name`, as a configuration error if unknown.

    `profiles.profile_for` raises `KeyError` listing the declared candidates;
    that is the right information and the wrong exception for a value that came
    out of a TOML file, so it is re-raised as `ConfigError` naming where the
    user can change it (requirement 2.3).
    """
    from npu_rag.embedding.profiles import profile_for

    try:
        return profile_for(model_name)
    except KeyError as error:
        raise ConfigError(
            f"configured model_name {model_name!r} is not a declared model: "
            f"{error.args[0]}. Set model_name in npu-rag.toml, or "
            f"NPU_RAG_MODEL_NAME, to one of those names."
        ) from None


def state_path_for(config: CliConfig) -> Path:
    """Where the ingest state database lives for this configuration (ruling 2)."""
    return config.index_path.expanduser().resolve().parent / STATE_FILENAME


def missing_archive_message(archive_root: Path) -> str | None:
    """A diagnostic when `archive_root` is not a directory, else `None` (ruling 4)."""
    if archive_root.is_dir():
        return None
    return (
        f"archive path {archive_root} is not an existing directory. Point "
        f"--archive-path (or NPU_RAG_ARCHIVE_PATH, or archive_path in "
        f"npu-rag.toml) at the folder holding the documents to index."
    )


def build_ingest_config(config: CliConfig, *, profile: ModelProfile) -> IngestConfig:
    """Turn a `CliConfig` plus the active model profile into an `IngestConfig`.

    Field by field, because the two objects share no field at all:

    - `roots`: the resolved, absolute `archive_path` as a one-element tuple.
      Resolved per note 2.2; one element because `CliConfig` offers exactly one
      and several would need a config amendment, not a guess.
    - `token_budget`: `profile.max_input_tokens` (ruling 1).
    - `state_path`: `state_path_for(config)` (ruling 2).
    - `include` / `exclude`: left empty, which `ingest/discover.py` reads as
      include-all and exclude-none. Requirement 7.1's extension set is
      `ingest/route.py`'s to decide, and a glob list here would override it.
    - everything else - `prose_overlap_tokens`, `min_page_chars`,
      `min_image_pixels`, the three `vision_*` fields - keeps the ingest
      package's default. `CliConfig` has no field for any of them, and the
      vision three are inert while `describer` is `None`.
    """
    from npu_rag.ingest.config import IngestConfig

    return IngestConfig(
        roots=(config.archive_path.expanduser().resolve(),),
        token_budget=profile.max_input_tokens,
        state_path=state_path_for(config),
    )


def open_tokenizer(profile: ModelProfile) -> ModelTokenizer:
    """The tokenizer the chunker measures with.

    A tokenizer acquisition, not a model export: nothing is compiled and the
    NPU is not touched. Imported inside the function because
    `npu_rag.embedding.tokenize` pulls in `transformers`, which requirement 1.2
    keeps off every startup path including `index --help`.
    """
    from npu_rag.embedding.tokenize import load_tokenizer

    return load_tokenizer(profile)


def run_pipeline(config: IngestConfig, tokenizer: ModelTokenizer) -> RunReport:
    """Run ingest with no vision describer.

    `describer=None` is the intended configuration, not a gap: `run_ingest`
    tries `LocalOcrDescriber()`, which raises `VisionUnavailable` when the OCR
    weights are absent, and every image then becomes a `vision_unavailable`
    omission naming the capability it wanted. Requirement 7.5's summary reports
    those, which is how the user learns what their archive's images cost them.
    """
    from npu_rag.ingest import run_ingest

    return run_ingest(config, tokenizer, describer=None)


# --------------------------------------------------------------------------- #
# Requirement 7.5 / 8.3 - the run summary
# --------------------------------------------------------------------------- #


def summary_payload(
    report: RunReport,
    *,
    ingest_config: IngestConfig,
    config: CliConfig,
    profile: ModelProfile,
    full_rebuild: bool,
    duration_seconds: float,
) -> dict[str, object]:
    """Requirement 7.5's summary as one snake_case mapping (requirement 8.3).

    Two places where `RunReport`'s surface and the requirement's wording do not
    coincide, resolved in the report's favour because inventing a field would
    be worse than naming the limit:

    - **added vs updated.** `RunReport` counts `files_processed`; the state
      store's `FileStatus` distinction between a new and a changed file is not
      carried out of the pipeline, so one number covers both and is labelled as
      such. `files_unchanged` is requirement 7.2's content-hash skip and
      `files_skipped` is discovery's and the router's - two different things
      that the single word "skipped" would merge.
    - **chunks created.** `report.records` is *every* current chunk record: the
      ones this run chunked and the ones re-emitted unchanged from the state
      store. Nothing distinguishes them, so the key is `chunk_records` and the
      count of ids this run retired is reported beside it.

    `failures` carries the path and reason of every `FAILED` omission, and only
    those. Category counts are what requirement 7.5 asks for, but "Failed: 1"
    with no reason leaves the user nothing to act on, and a failure is the one
    category that is always actionable - the others say what the archive
    contains or what capability is missing, which the counts and
    `missing_capabilities` already report.
    """
    from npu_rag.ingest.types import OmissionCategory

    counts = report.omission_counts
    capabilities = sorted(
        {
            omission.missing_capability
            for omission in report.omissions
            if omission.missing_capability
        }
    )
    return {
        "command": "index",
        "archive_path": str(ingest_config.roots[0]),
        "index_path": str(config.index_path.expanduser().resolve()),
        "state_path": str(ingest_config.state_path),
        "model_name": config.model_name,
        "model_id": profile.model_id,
        "token_budget": ingest_config.token_budget,
        "requested_provider": config.provider,
        "active_provider": NO_EMBEDDING_PROVIDER,
        "full_rebuild": full_rebuild,
        "embedding_performed": False,
        "index_written": False,
        "files": {
            "processed": report.files_processed,
            "unchanged": report.files_unchanged,
            "skipped": report.files_skipped,
            "failed": report.files_failed,
            "total": (
                report.files_processed
                + report.files_unchanged
                + report.files_skipped
                + report.files_failed
            ),
        },
        "chunk_records": len(report.records),
        "chunk_ids_removed": len(report.removed_chunk_ids),
        "no_work_required": report.no_work_required,
        "omissions_total": len(report.omissions),
        "omissions_by_category": {
            category.value: counts[category]
            for category in sorted(counts, key=lambda item: item.value)
        },
        "missing_capabilities": capabilities,
        "failures": [
            {"path": str(omission.path), "reason": omission.reason}
            for omission in report.omissions
            if omission.category is OmissionCategory.FAILED
        ],
        "duration_seconds": round(duration_seconds, DURATION_DIGITS),
    }


def _yes_no(flag: bool) -> str:  # noqa: FBT001 - a formatter, not a behaviour switch
    return "yes" if flag else "no"


def _summary_text(payload: Mapping[str, object]) -> str:
    """Render the payload as the terminal summary of requirement 7.5.

    Reads the payload rather than the `RunReport` so the two renderings cannot
    disagree about a count: there is one accounting, and `--json` and the text
    are two views of it.
    """
    files = payload["files"]
    assert isinstance(files, dict)  # noqa: S101 - narrows the payload's own shape
    categories = payload["omissions_by_category"]
    assert isinstance(categories, dict)  # noqa: S101
    capabilities = payload["missing_capabilities"]
    assert isinstance(capabilities, list)  # noqa: S101
    failures = payload["failures"]
    assert isinstance(failures, list)  # noqa: S101

    lines = [
        "Index Run Summary",
        "=" * len("Index Run Summary"),
        f"Archive:  {payload['archive_path']}",
        f"Index:    {payload['index_path']} (not written in this release)",
        f"State:    {payload['state_path']}",
        f"Model:    {payload['model_name']} (token budget {payload['token_budget']})",
        f"Provider: requested {payload['requested_provider']}, "
        f"active {payload['active_provider']}",
        f"Full rebuild requested: {_yes_no(bool(payload['full_rebuild']))}",
        "",
        "Files:",
        f"  Processed (added or updated): {files['processed']}",
        f"  Unchanged (content hash matched): {files['unchanged']}",
        f"  Skipped (unsupported or unreadable): {files['skipped']}",
        f"  Failed: {files['failed']}",
        f"  Total: {files['total']}",
        "",
        "Chunks:",
        f"  Chunk records: {payload['chunk_records']}",
        f"  Chunk ids removed: {payload['chunk_ids_removed']}",
        f"  No work required: {_yes_no(bool(payload['no_work_required']))}",
        "",
        f"Omissions by category ({payload['omissions_total']} total):",
    ]
    if categories:
        lines.extend(f"  {name}: {count}" for name, count in categories.items())
    else:
        lines.append("  (none)")
    if capabilities:
        lines.append(f"  Missing capabilities: {', '.join(str(c) for c in capabilities)}")
    if failures:
        lines.extend(["", "Failures:"])
        for entry in failures[:FAILURE_DETAIL_LIMIT]:
            assert isinstance(entry, dict)  # noqa: S101
            lines.append(f"  {entry['path']}: {entry['reason']}")
        remaining = len(failures) - FAILURE_DETAIL_LIMIT
        if remaining > 0:
            lines.append(f"  ... and {remaining} more (use --json for all of them)")
    lines.extend(["", f"Duration: {payload['duration_seconds']}s"])
    return "\n".join(lines)


def render_summary(payload: Mapping[str, object], *, as_json: bool) -> str:
    """The summary in whichever rendering requirement 8.3's flag selected.

    `ensure_ascii=False` matches `render.render_search_json`: a path carrying
    Devanagari stays literal for a human and for `jq`, which is why the write
    has to go through `render.emit` (requirement 8.4).
    """
    if as_json:
        return json.dumps(payload, ensure_ascii=False, indent=render.JSON_INDENT)
    return _summary_text(payload)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _fail(message: str, code: int) -> int:
    """A diagnostic on stderr and nothing on stdout - `main._fail`'s format."""
    render.emit(f"{PROGRAM_NAME}: error: {message}", stream=sys.stderr)
    return code


def _notice(message: str) -> None:
    """A non-error remark on stderr, so `--json` stdout stays parseable."""
    render.emit(f"{PROGRAM_NAME}: notice: {message}", stream=sys.stderr)


def run(
    args: argparse.Namespace,
    *,
    load: ConfigLoader = load_config,
    profiles: ProfileLoader = load_profile,
    tokenizers: TokenizerLoader = open_tokenizer,
    ingest: _Ingester = run_pipeline,
) -> int:
    """Execute `npu-rag index`'s ingest half and return its exit code.

    The order is what keeps design §4's codes honest: everything the user can
    get wrong is diagnosed before anything downloads, opens a database or walks
    a filesystem. Discovery treats an absent archive as zero files, so check
    the root first or a mistyped path would look like an empty archive.

    1. configuration, the model profile, and the archive root - all exit 1, all
       before the tokenizer is fetched (ruling 4);
    2. the tokenizer, where an acquisition or licence failure is exit 2;
    3. the run itself, where a state-store or filesystem failure is exit 2;
    4. the summary, and exit 0 - including when the run reported omissions or
       per-file failures, which are the report's subject and not its verdict
       (ruling 3).

    The four seams are injectable for testing; their defaults are the
    production path and `main.dispatch` calls this with the namespace alone.
    """
    try:
        config = load(cli_args=vars(args))
        profile = profiles(config.model_name)
        ingest_config = build_ingest_config(config, profile=profile)
    except ConfigError as error:
        return emit_config_error(error)
    except ValueError as error:
        # `IngestConfig.__post_init__`'s own guards. Reachable only through a
        # profile whose sequence length is not positive, which is a broken
        # declaration rather than a typo - but it is still a value, so it is
        # reported as one instead of as a traceback.
        return _fail(f"the ingest configuration is not usable: {error}", EXIT_USAGE)

    archive_error = missing_archive_message(ingest_config.roots[0])
    if archive_error is not None:
        return _fail(archive_error, EXIT_USAGE)

    full_rebuild = bool(getattr(args, "full_rebuild", False))
    if full_rebuild:
        _notice(REBUILD_NOTICE)

    from npu_rag.embedding.errors import EmbeddingRuntimeError
    from npu_rag.ingest.errors import IngestError

    started = time.perf_counter()
    try:
        tokenizer = tokenizers(profile)
        report = ingest(ingest_config, tokenizer)
    except EmbeddingRuntimeError as error:
        # Acquisition, licence acceptance and loading failures. Exit 2, as in
        # `cmd_search`: the user may be able to act on a licence prompt, but the
        # failure is of the machine's preparation and not of a configured value.
        return _fail(str(error), EXIT_RUNTIME)
    except (IngestError, OSError) as error:
        return _fail(str(error), EXIT_RUNTIME)
    duration = time.perf_counter() - started

    render.emit(
        render_summary(
            summary_payload(
                report,
                ingest_config=ingest_config,
                config=config,
                profile=profile,
                full_rebuild=full_rebuild,
                duration_seconds=duration,
            ),
            as_json=bool(getattr(args, "json", False)),
        )
    )
    return EXIT_OK


EXIT_CODES: Final[tuple[int, int, int]] = (EXIT_OK, EXIT_USAGE, EXIT_RUNTIME)
"""Design §4's taxonomy, re-exported so a test can pin this module to it."""
