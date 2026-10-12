"""Index an archive into LanceDB through the ingest and embedding runtimes.

Heavyweight imports stay inside execution seams so help and argument errors
stay fast. A normal run reuses ingest fingerprints and existing vectors. A full
rebuild resets both stores before scanning. Output goes through render.emit so
Unicode diagnostics work on legacy console encodings.
"""
from __future__ import annotations

import argparse
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
from npu_rag.cli.index_summary import (
    DURATION_DIGITS,
    FAILURE_DETAIL_LIMIT,
    NO_EMBEDDING_PROVIDER,
    render_summary,
    summary_payload,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from npu_rag.embedding.profiles import ModelProfile
    from npu_rag.embedding.service import EmbedResult
    from npu_rag.embedding.tokenize import ModelTokenizer
    from npu_rag.embedding.types import DocumentText, ProviderChoice
    from npu_rag.index.types import SyncSummary
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

REBUILD_NOTICE: Final = "full rebuild requested; discarding index and ingest state."
PROGRESS_INTERVAL_SECONDS: Final = 0.5


# --------------------------------------------------------------------------- #
# Seams
# --------------------------------------------------------------------------- #


class _Ingester(Protocol):
    def __call__(self, config: IngestConfig, tokenizer: ModelTokenizer) -> RunReport: ...


class _Index(Protocol):
    def verify_model_identity(self, model_identity: str) -> None: ...

    def sync(
        self, report: RunReport, embed_service: object, *, model_identity: str,
        provider: ProviderChoice, progress_callback: Callable[[int, int], None],
    ) -> SyncSummary: ...


class _Embedder(Protocol):
    provider_served: str | None
    fallback_reason: str | None

    def embed_documents(
        self, texts: Sequence[DocumentText], provider: ProviderChoice,
    ) -> EmbedResult: ...


if TYPE_CHECKING:
    ConfigLoader = Callable[..., CliConfig]
    ProfileLoader = Callable[[str], ModelProfile]
    TokenizerLoader = Callable[[ModelProfile], ModelTokenizer]
    IndexOpener = Callable[[Path], _Index]
    EmbedderOpener = Callable[[CliConfig, ModelProfile], _Embedder]
    Rebuilder = Callable[[Path, Path], None]


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
            "skipping unchanged ones (requirement 7.3)."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        help=(
            "Where embeddings will be computed: 'npu' (fail if the NPU is "
            "unavailable), 'cpu', or 'auto'. Defaults to the configured "
            "provider."
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


class _EmbeddingService(Protocol):
    def embed_documents(
        self, texts: Sequence[DocumentText], provider: ProviderChoice,
    ) -> EmbedResult: ...


class _TrackingEmbedder:
    """Keep the actual provider and fallback reason from every sync batch."""

    def __init__(self, service: _EmbeddingService) -> None:
        self._service = service
        self.provider_served: str | None = None
        self.fallback_reason: str | None = None

    def embed_documents(
        self, texts: Sequence[DocumentText], provider: ProviderChoice,
    ) -> EmbedResult:
        result = self._service.embed_documents(texts, provider)
        served = str(result.provider_served)
        if self.provider_served is not None and self.provider_served != served:
            raise RuntimeError("the active embedding provider changed during indexing")
        self.provider_served = served
        self.fallback_reason = result.fallback_reason
        return result


def open_embedder(config: CliConfig, profile: ModelProfile) -> _Embedder:
    """Prepare the chosen artifact after checking explicit NPU availability."""
    from npu_rag.cli.cmd_search import artifacts_root
    from npu_rag.embedding.environment.capability import check_capability
    from npu_rag.embedding.errors import NpuUnavailableError
    from npu_rag.embedding.providers.base import npu_unavailable_reason
    from npu_rag.embedding.service import build_service
    from npu_rag.embedding.types import ProviderChoice

    requested = ProviderChoice(config.provider)
    capability = check_capability()
    reason = npu_unavailable_reason(capability)
    if requested is ProviderChoice.NPU and reason is not None:
        raise NpuUnavailableError(
            f"provider 'npu' was requested explicitly and the NPU is not usable: "
            f"{reason}. Re-run with --provider cpu or repair NPU provisioning.",
            provider=requested,
            model_id=profile.model_id,
        )
    artifact_provider = (
        ProviderChoice.CPU
        if requested is ProviderChoice.CPU or reason is not None
        else ProviderChoice.NPU
    )
    return _TrackingEmbedder(
        build_service(profile, capability, artifacts_root(config), provider=artifact_provider)
    )


def open_index_at(index_path: Path) -> _Index:
    from npu_rag.index import VectorIndex

    return VectorIndex(index_path)


def rebuild_at(index_path: Path, state_path: Path) -> None:
    """Discard the chunk table and ingest fingerprints for a true rebuild."""
    import lancedb

    if index_path.is_dir():
        db = lancedb.connect(str(index_path))
        if "chunks" in db.list_tables().tables:
            db.drop_table("chunks")
    state_path.unlink(missing_ok=True)


class _Progress:
    """Throttle progress on elapsed wall time, always showing the final count."""

    def __init__(self, report: RunReport, started: float) -> None:
        self._files = report.files_processed + report.files_unchanged + report.files_skipped + report.files_failed
        self._processed = report.files_processed
        self._started = started
        self._last = started
        self._last_done = -1

    def __call__(self, done: int, total: int) -> None:
        now = time.perf_counter()
        if done == self._last_done:
            return
        if done < total and now - self._last < PROGRESS_INTERVAL_SECONDS:
            return
        elapsed = max(now - self._started, 0.001)
        render.emit(
            f"Progress: files {self._processed}/{self._files}; "
            f"chunks {done}/{total}; elapsed {elapsed:.1f}s; "
            f"throughput {done / elapsed:.1f} chunks/s",
            stream=sys.stderr,
        )
        self._last = now
        self._last_done = done


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
    open_index: IndexOpener = open_index_at,
    open_embedder: EmbedderOpener = open_embedder,
    rebuild: Rebuilder = rebuild_at,
) -> int:
    """Validate configuration, prepare, ingest, sync, and emit the run summary."""
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

    from npu_rag.embedding.errors import EmbeddingRuntimeError
    from npu_rag.ingest.errors import IngestError
    from npu_rag.embedding.types import ProviderChoice
    from npu_rag.index import format_model_identity
    from npu_rag.index.errors import VectorIndexError
    from npu_rag.cli.cmd_search import CPU_ORT_PROVIDER, NPU_ORT_PROVIDER

    started = time.perf_counter()
    embedder: _Embedder | None = None
    try:
        tokenizer = tokenizers(profile)
        embedder = open_embedder(config, profile)
        if full_rebuild:
            _notice(REBUILD_NOTICE)
            rebuild(config.index_path.expanduser().resolve(), ingest_config.state_path)
        report = ingest(ingest_config, tokenizer)
        index = open_index(config.index_path.expanduser().resolve())
        identity = format_model_identity(profile.model_id, profile.dimension)
        index.verify_model_identity(identity)
        progress = _Progress(report, started)
        sync_result = index.sync(
            report, embedder, model_identity=identity,
            provider=ProviderChoice(config.provider), progress_callback=progress,
        )
        embedded = sync_result.chunks_added + sync_result.chunks_updated
        progress(embedded, embedded)
        if embedder.fallback_reason is not None:
            _notice(
                "the NPU was not used; this indexing run used the CPU. "
                f"Reason: {embedder.fallback_reason}"
            )
        if embedder.provider_served == "npu":
            active_provider = NPU_ORT_PROVIDER
        elif embedder.provider_served == "cpu":
            active_provider = CPU_ORT_PROVIDER
        elif embedded == 0:
            active_provider = NO_EMBEDDING_PROVIDER
        else:
            raise RuntimeError("indexing completed without an active provider attribution")
    except EmbeddingRuntimeError as error:
        # Acquisition, licence acceptance and loading failures. Exit 2, as in
        # `cmd_search`: the user may be able to act on a licence prompt, but the
        # failure is of the machine's preparation and not of a configured value.
        if embedder is not None and embedder.fallback_reason is not None:
            _notice(f"the NPU was not used; indexing ran on the CPU. Reason: {embedder.fallback_reason}")
        return _fail(str(error), EXIT_RUNTIME)
    except (IngestError, VectorIndexError, OSError, RuntimeError) as error:
        if embedder is not None and embedder.fallback_reason is not None:
            _notice(f"the NPU was not used; indexing ran on the CPU. Reason: {embedder.fallback_reason}")
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
                active_provider=active_provider,
                embedding_performed=embedded > 0,
                index_written=True,
            ),
            as_json=bool(getattr(args, "json", False)),
        )
    )
    return EXIT_OK


EXIT_CODES: Final[tuple[int, int, int]] = (EXIT_OK, EXIT_USAGE, EXIT_RUNTIME)
"""Design §4's taxonomy, re-exported so a test can pin this module to it."""
