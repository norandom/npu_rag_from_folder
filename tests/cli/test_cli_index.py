"""Tests for `npu-rag index`, ingest half (task 5.1a).

Requirements under test: 7.1-7.5 (the indexing command and its summary), 8.3
(`--json`), 8.4 (UTF-8 stdout), 2.1 (configuration precedence) and design §3.5
/ §4 (the flag surface and the exit codes).

Four rules shape this file.

**The flag surface and the payload schema are pinned against hand-written
literals.** Implementation notes 1.1 and 2.2 record the same defect twice: a
constant tested only through a parameterization derived from itself is not
covered at all. `EXPECTED_OPTIONS` and `EXPECTED_PAYLOAD_KEYS` are therefore
written out here, and separate assertions tie those literals to what argparse
registers and to what the command emits.

**`token_budget` is pinned as a derivation, not as a number.** The one test
that would pass a hardcoded `512` is excluded by construction: the profile
handed to `build_ingest_config` carries a sequence length that is deliberately
*not* 512, so a literal in the implementation cannot satisfy it. A second test
anchors the derivation to all three real profiles and records that today's
answer happens to be 512.

**Every fixture's given order differs from its expected order** (notes 2.2 and
4.1). The hand-written omission table is built in an order that is neither
sorted by category nor sorted by count, and the file counts are five distinct
numbers, so a swapped or sorted-away field is observable.

**Requirement 8.4 is unobservable through `capsys`,** which is already UTF-8
(note 4.1). The test that distinguishes `render.emit` from a bare `print`
replaces `sys.stdout` with a real cp1252 `TextIOWrapper` and renders a path
containing Devanagari.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from npu_rag.cli import cmd_index
from npu_rag.cli import render as render_module
from npu_rag.cli.config import CliConfig, ConfigError
from npu_rag.cli.main import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, build_parser, main

if TYPE_CHECKING:
    from collections.abc import Sequence
    from npu_rag.embedding.tokenize import ModelTokenizer
    from npu_rag.ingest.config import IngestConfig
    from npu_rag.ingest.types import RunReport

# --------------------------------------------------------------------------- #
# Literals (design §3.5, requirements 7.5 and 8.3)
# --------------------------------------------------------------------------- #

#: Design §3.5's four options for `index`, plus argparse's own `-h/--help`.
EXPECTED_OPTIONS: tuple[str, ...] = (
    "--help",
    "--archive-path",
    "--full-rebuild",
    "--provider",
    "--json",
)

#: The flags whose `None` default is load-bearing: `config.load_config` reads
#: `None` as "this layer supplied nothing", so a literal default here would
#: retire requirement 2.1's environment and TOML layers.
CONFIG_BACKED_FLAGS: tuple[str, ...] = ("archive_path", "provider")

#: Requirement 8.3's payload, written out by hand. Order is irrelevant to a
#: `jq` consumer; the key *set* is the contract.
EXPECTED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "command",
        "archive_path",
        "index_path",
        "state_path",
        "model_name",
        "model_id",
        "token_budget",
        "requested_provider",
        "active_provider",
        "full_rebuild",
        "embedding_performed",
        "index_written",
        "files",
        "chunk_records",
        "chunk_ids_removed",
        "no_work_required",
        "omissions_total",
        "omissions_by_category",
        "missing_capabilities",
        "failures",
        "duration_seconds",
    }
)

EXPECTED_FILE_KEYS: frozenset[str] = frozenset(
    {"processed", "unchanged", "skipped", "failed", "total"}
)

#: `npu_rag.embedding.profiles` declares exactly these three (requirement 4.1
#: of `npu-embedding-runtime`), and every one compiles at 512 today.
EXPECTED_MODELS: tuple[str, ...] = (
    "embeddinggemma-300m",
    "gte-modernbert-base",
    "nomic-embed-text-v1.5",
)
TODAYS_COMPILED_SEQ_LEN = 512

#: Deliberately not 512, and not a multiple of it: a `token_budget` that came
#: from a literal cannot produce this.
PROBE_SEQ_LEN = 384

#: `IngestConfig`'s own defaults for the fields `CliConfig` cannot supply.
EXPECTED_PROSE_OVERLAP = 64
EXPECTED_MIN_PAGE_CHARS = 50
EXPECTED_MIN_IMAGE_PIXELS = 200


# --------------------------------------------------------------------------- #
# Doubles and builders
# --------------------------------------------------------------------------- #


def make_config(
    tmp_path: Path,
    *,
    archive: str = "corpus",
    index: str = "store/lancedb",
    model_name: str = "gte-modernbert-base",
    provider: str = "auto",
    create_archive: bool = True,
) -> CliConfig:
    """A `CliConfig` with no file, no environment and no upward walk.

    Note 1.2: `load_config()` walks the whole parent chain, so a `npu-rag.toml`
    anywhere above `%TEMP%` would change what these tests see. The `load` seam
    is injected instead, and the one test that drives the real loader bounds it
    with `start`/`home`/`stop` inside `tmp_path`.
    """
    archive_path = tmp_path / archive
    if create_archive:
        archive_path.mkdir(parents=True, exist_ok=True)
    return CliConfig(
        archive_path=archive_path,
        index_path=tmp_path / index,
        model_name=model_name,
        provider=provider,
        tags={},
        default_mode="hybrid",
        default_limit=10,
        ansi_colors=False,
    )


def probe_profile(*, compiled_seq_len: int = PROBE_SEQ_LEN) -> Any:
    """A real `ModelProfile` whose sequence length is chosen by the test.

    Derived from a declared profile with `dataclasses.replace` so every other
    field stays the genuine article and `__post_init__`'s invariants hold; only
    the one value this file needs to vary is varied.
    """
    from dataclasses import replace

    from npu_rag.embedding.profiles import profile_for

    return replace(
        profile_for("gte-modernbert-base"),
        model_id="test/probe-model",
        compiled_seq_len=compiled_seq_len,
    )


def make_record(chunk_id: str) -> Any:
    from npu_rag.ingest.types import ChunkKind, ChunkRecord, MarkdownLocator

    return ChunkRecord(
        chunk_id=chunk_id,
        source_path=Path("a.md"),
        root_id="root",
        author="Author",
        title="Title",
        kind=ChunkKind.PROSE,
        text="some text",
        locator=MarkdownLocator(line_range=(1, 5), ordinal=0),
        ordinal=0,
        token_count=9,
        truncated=False,
        provenance=None,
    )


def make_omission(category: str, *, path: str, capability: str | None = None) -> Any:
    from npu_rag.ingest.types import Omission, OmissionCategory

    return Omission(
        category=OmissionCategory(category),
        path=Path(path),
        reason=f"reason for {path}",
        missing_capability=capability,
    )


#: The hand-written omission table. Its given order is neither sorted by
#: category value (`failed` < `unsupported` < `vision_unavailable`) nor by
#: count, and the counts are 3 / 1 / 2 rather than equal, so a dropped `sorted`
#: and a miscount are both observable. `det.onnx` appears twice so the
#: de-duplication of `missing_capabilities` has something to do.
OMISSION_FIXTURE: tuple[tuple[str, str, str | None], ...] = (
    ("vision_unavailable", "img/a.png", "det.onnx"),
    ("unsupported", "weird.xyz", None),
    ("failed", "broken1.md", None),
    ("vision_unavailable", "img/b.png", "det.onnx"),
    ("failed", "broken2.md", None),
    ("vision_unavailable", "img/c.png", "rec.onnx"),
    ("failed", "broken3.md", None),
)

#: What the fixture above must produce, written out rather than computed.
EXPECTED_CATEGORY_COUNTS: dict[str, int] = {
    "failed": 3,
    "unsupported": 1,
    "vision_unavailable": 3,
}
EXPECTED_CAPABILITIES: list[str] = ["det.onnx", "rec.onnx"]
EXPECTED_FAILURE_PATHS: list[str] = ["broken1.md", "broken2.md", "broken3.md"]


def make_report(
    *,
    processed: int = 7,
    unchanged: int = 4,
    skipped: int = 2,
    failed: int = 3,
    records: int = 11,
    removed: Sequence[str] = ("gone-1", "gone-2"),
    omissions: Sequence[tuple[str, str, str | None]] = OMISSION_FIXTURE,
    no_work_required: bool = False,
) -> Any:
    """A `RunReport` whose five counts are five distinct numbers.

    Distinct on purpose: with equal counts, a payload that read
    `files_unchanged` where it meant `files_skipped` would pass.
    """
    from npu_rag.ingest.types import RunReport

    return RunReport(
        files_processed=processed,
        files_unchanged=unchanged,
        files_skipped=skipped,
        files_failed=failed,
        omissions=tuple(
            make_omission(category, path=path, capability=capability)
            for category, path, capability in omissions
        ),
        removed_chunk_ids=tuple(removed),
        no_work_required=no_work_required,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=tuple(make_record(f"c{n}") for n in range(records)),
    )


class RecordingIngester:
    """Stands in for `run_ingest`, recording the `IngestConfig` it was handed."""

    def __init__(self, report: Any | None = None, error: Exception | None = None) -> None:
        self.report = make_report() if report is None else report
        self.error = error
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, config: Any, tokenizer: Any) -> Any:
        self.calls.append((config, tokenizer))
        if self.error is not None:
            raise self.error
        return self.report


class RecordingTokenizers:
    """Stands in for `open_tokenizer`; counts whether it was reached at all."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[Any] = []

    def __call__(self, profile: Any) -> Any:
        self.calls.append(profile)
        if self.error is not None:
            raise self.error
        return object()


class RecordingEmbedder:
    fallback_reason: str | None = None

    def __init__(self, provider_served: str | None = "cpu") -> None:
        self.provider_served = provider_served

    def embed_documents(self, texts: Any, provider: Any) -> Any:
        raise AssertionError("the recording index must not call the embedder")


class RecordingIndex:
    def __init__(self) -> None:
        self.identity: str | None = None
        self.sync_kwargs: dict[str, Any] = {}
        self.events: list[str] = []

    def verify_model_identity(self, model_identity: str) -> None:
        self.events.append("verify")
        self.identity = model_identity

    def sync(self, report: Any, embed_service: Any, **kwargs: Any) -> Any:
        from npu_rag.index.types import SyncSummary

        self.sync_kwargs = kwargs
        self.events.append("sync")
        kwargs["progress_callback"](len(report.records), len(report.records))
        return SyncSummary(
            chunks_added=len(report.records), chunks_updated=0,
            chunks_deleted=len(report.removed_chunk_ids), total_chunks=len(report.records),
            files_indexed=report.files_processed, elapsed_seconds=0.1,
        )


def parse(*argv: str) -> argparse.Namespace:
    """Parse through the real top-level parser, lazy registration hook and all."""
    tokens = list(argv)
    return build_parser(tokens).parse_args(tokens)


def invoke(
    *argv: str,
    config: CliConfig,
    ingester: RecordingIngester | None = None,
    tokenizers: RecordingTokenizers | None = None,
    profile: Any | None = None,
    index: RecordingIndex | None = None,
    embedder: RecordingEmbedder | None = None,
) -> tuple[int, RecordingIngester]:
    """Run the real `run()` with its four seams injected."""
    used = RecordingIngester() if ingester is None else ingester
    resolved_profile = probe_profile() if profile is None else profile
    code = cmd_index.run(
        parse("index", *argv),
        load=lambda **_kwargs: config,
        profiles=lambda _name: resolved_profile,
        tokenizers=RecordingTokenizers() if tokenizers is None else tokenizers,
        ingest=used,
        open_index=lambda _path: RecordingIndex() if index is None else index,
        open_embedder=lambda _config, _profile: RecordingEmbedder() if embedder is None else embedder,
        rebuild=lambda _index, _state: None,
    )
    return code, used


def payload_of(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    captured = capsys.readouterr()
    parsed: dict[str, Any] = json.loads(captured.out)
    return parsed


def _run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "npu_rag.cli.main", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )


# --------------------------------------------------------------------------- #
# Design §3.5 - the flag surface
# --------------------------------------------------------------------------- #


def test_registered_options_are_exactly_the_designed_four() -> None:
    parser = build_parser(["index"])
    subparser = _index_subparser(parser)
    registered = tuple(
        option
        for action in subparser._actions  # noqa: SLF001 - argparse exposes no accessor
        for option in action.option_strings
        if option.startswith("--")
    )
    assert registered == EXPECTED_OPTIONS


def _index_subparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return cast("argparse.ArgumentParser", action.choices["index"])
    raise AssertionError("the top-level parser registers no subparsers")


@pytest.mark.parametrize("dest", CONFIG_BACKED_FLAGS)
def test_a_flag_a_config_field_also_supplies_defaults_to_none(dest: str) -> None:
    """Requirement 2.1: an unset flag must not out-rank the env and TOML layers."""
    args = parse("index")
    assert getattr(args, dest) is None


@pytest.mark.parametrize("dest", ["full_rebuild", "json"])
def test_a_flag_with_no_config_field_defaults_to_false(dest: str) -> None:
    assert getattr(parse("index"), dest) is False


def test_unknown_provider_is_a_usage_error_not_a_runtime_one() -> None:
    completed = _run_cli("index", "--provider", "gpu")
    assert completed.returncode == EXIT_USAGE
    assert "gpu" in completed.stderr
    assert completed.stdout == ""


def test_flag_values_reach_the_configuration_loader(tmp_path: Path) -> None:
    """The flags are wired through `load_config`, not read off the namespace."""
    seen: list[dict[str, object]] = []
    config = make_config(tmp_path)

    def recording_load(**kwargs: Any) -> CliConfig:
        seen.append(dict(kwargs["cli_args"]))
        return config

    cmd_index.run(
        parse("index", "--archive-path", str(tmp_path / "corpus"), "--provider", "cpu"),
        load=recording_load,
        profiles=lambda _name: probe_profile(),
        tokenizers=RecordingTokenizers(),
        ingest=RecordingIngester(),
        open_index=lambda _path: RecordingIndex(),
        open_embedder=lambda _config, _profile: RecordingEmbedder(),
    )
    assert seen[0]["archive_path"] == str(tmp_path / "corpus")
    assert seen[0]["provider"] == "cpu"


def test_an_environment_variable_still_decides_the_archive_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 2.1's second layer, through the real loader (note 1.2's bound)."""
    from npu_rag.cli.config import load_config

    corpus = tmp_path / "from-env"
    corpus.mkdir()
    monkeypatch.setenv("NPU_RAG_ARCHIVE_PATH", str(corpus))
    config = load_config(
        cli_args=vars(parse("index")),
        start=tmp_path,
        home=tmp_path,
        stop=tmp_path,
    )
    assert config.archive_path == corpus


# --------------------------------------------------------------------------- #
# The config bridge (requirements 7.1, 7.2)
# --------------------------------------------------------------------------- #


def test_roots_is_the_resolved_absolute_archive_path(tmp_path: Path) -> None:
    """Note 2.2: a relative or unnormalised root cannot match an index path."""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    config = CliConfig(
        archive_path=tmp_path / "a" / ".." / "a" / "b",
        index_path=tmp_path / "store" / "lancedb",
        model_name="gte-modernbert-base",
        provider="auto",
        tags={},
        default_mode="hybrid",
        default_limit=10,
        ansi_colors=False,
    )
    ingest_config = cmd_index.build_ingest_config(config, profile=probe_profile())
    assert ingest_config.roots == (nested.resolve(),)
    assert ingest_config.roots[0].is_absolute()
    assert ".." not in ingest_config.roots[0].parts


def test_a_relative_archive_path_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`CliConfig.archive_path` defaults to `Path(".")` (note 2.2)."""
    monkeypatch.chdir(tmp_path)
    config = CliConfig(
        archive_path=Path("."),
        index_path=Path(".npu_rag") / "lancedb",
        model_name="gte-modernbert-base",
        provider="auto",
        tags={},
        default_mode="hybrid",
        default_limit=10,
        ansi_colors=False,
    )
    ingest_config = cmd_index.build_ingest_config(config, profile=probe_profile())
    assert ingest_config.roots == (tmp_path.resolve(),)


def test_token_budget_comes_from_the_profile_and_not_from_a_literal(
    tmp_path: Path,
) -> None:
    """The one assertion a hardcoded `512` cannot satisfy (ruling 1)."""
    ingest_config = cmd_index.build_ingest_config(
        make_config(tmp_path), profile=probe_profile(compiled_seq_len=PROBE_SEQ_LEN)
    )
    assert ingest_config.token_budget == PROBE_SEQ_LEN
    assert ingest_config.token_budget != TODAYS_COMPILED_SEQ_LEN


@pytest.mark.parametrize("model_name", EXPECTED_MODELS)
def test_token_budget_equals_the_models_compiled_sequence_length(
    tmp_path: Path, model_name: str
) -> None:
    """The derivation anchored to the real profiles, and to today's number.

    `max_input_tokens` *is* `compiled_seq_len`; both are asserted, because the
    contract is the compiled graph's input width and a future profile that
    separated the two would have to be a deliberate change here.
    """
    from npu_rag.embedding.profiles import profile_for

    profile = profile_for(model_name)
    ingest_config = cmd_index.build_ingest_config(
        make_config(tmp_path, model_name=model_name), profile=profile
    )
    assert ingest_config.token_budget == profile.max_input_tokens
    assert ingest_config.token_budget == profile.compiled_seq_len
    assert ingest_config.token_budget == TODAYS_COMPILED_SEQ_LEN


def test_state_path_is_an_absolute_sibling_of_the_configured_index(
    tmp_path: Path,
) -> None:
    """Ruling 2: the incremental skip must not key off the working directory.

    The second half is the non-vacuity control for the first: a `state_path`
    left at `IngestConfig`'s relative default would satisfy neither, but one
    pinned to a constant directory would satisfy only the first.
    """
    first = cmd_index.build_ingest_config(
        make_config(tmp_path, index="one/lancedb"), profile=probe_profile()
    )
    second = cmd_index.build_ingest_config(
        make_config(tmp_path, index="two/lancedb"), profile=probe_profile()
    )
    assert cmd_index.STATE_FILENAME == "ingest.sqlite"
    assert first.state_path == (tmp_path / "one" / "ingest.sqlite").resolve()
    assert first.state_path.is_absolute()
    assert second.state_path.parent == (tmp_path / "two").resolve()


def test_the_fields_cli_config_cannot_supply_keep_the_ingest_defaults(
    tmp_path: Path,
) -> None:
    ingest_config = cmd_index.build_ingest_config(
        make_config(tmp_path), profile=probe_profile()
    )
    assert ingest_config.include == ()
    assert ingest_config.exclude == ()
    assert ingest_config.prose_overlap_tokens == EXPECTED_PROSE_OVERLAP
    assert ingest_config.min_page_chars == EXPECTED_MIN_PAGE_CHARS
    assert ingest_config.min_image_pixels == EXPECTED_MIN_IMAGE_PIXELS


def test_an_unknown_model_name_is_a_config_error_naming_the_candidates() -> None:
    with pytest.raises(ConfigError) as excinfo:
        cmd_index.load_profile("embeddinggemma-9000")
    message = str(excinfo.value)
    assert "embeddinggemma-9000" in message
    assert "model_name" in message
    for name in EXPECTED_MODELS:
        assert name in message


def test_load_profile_returns_the_declared_profile_for_every_model() -> None:
    """Non-vacuity control: the error path above is not the only path."""
    from npu_rag.embedding.profiles import profile_for

    assert [cmd_index.load_profile(name) for name in EXPECTED_MODELS] == [
        profile_for(name) for name in EXPECTED_MODELS
    ]


def test_the_pipeline_is_run_without_a_vision_describer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`describer=None` is the intended configuration (requirement 7.5)."""
    import npu_rag.ingest as ingest_package

    seen: list[object] = []

    def fake_run_ingest(config: Any, tokenizer: Any, *, describer: Any = None) -> Any:
        seen.append(describer)
        return make_report()

    monkeypatch.setattr(ingest_package, "run_ingest", fake_run_ingest)
    ingest_config = cmd_index.build_ingest_config(
        make_config(tmp_path), profile=probe_profile()
    )
    cmd_index.run_pipeline(ingest_config, cast("Any", object()))
    assert seen == [None]


# --------------------------------------------------------------------------- #
# Design §4 - exit codes and ordering
# --------------------------------------------------------------------------- #


def test_exit_code_constants_match_the_design_taxonomy() -> None:
    assert cmd_index.EXIT_CODES == (0, 1, 2)


def test_a_completed_run_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, ingester = invoke("--json", config=make_config(tmp_path))
    assert code == EXIT_OK
    assert len(ingester.calls) == 1
    assert payload_of(capsys)["command"] == "index"


def test_a_run_that_completed_with_omissions_and_failures_is_still_a_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ruling 3: omissions are the report's subject, not its verdict."""
    code, _ = invoke("--json", config=make_config(tmp_path))
    payload = payload_of(capsys)
    assert code == EXIT_OK
    assert payload["omissions_total"] > 0
    assert cast("dict[str, int]", payload["files"])["failed"] > 0


def test_a_configuration_error_exits_one_with_nothing_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_load(**_kwargs: Any) -> CliConfig:
        raise ConfigError("npu-rag.toml: key 'provider' must be one of 'npu', 'cpu'")

    code = cmd_index.run(
        parse("index"),
        load=failing_load,
        profiles=lambda _name: probe_profile(),
        tokenizers=RecordingTokenizers(),
        ingest=RecordingIngester(),
    )
    captured = capsys.readouterr()
    assert code == EXIT_USAGE
    assert captured.out == ""
    assert "npu-rag.toml" in captured.err
    assert "Traceback" not in captured.err


def test_an_unknown_configured_model_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = make_config(tmp_path, model_name="embeddinggemma-9000")
    ingester = RecordingIngester()
    code = cmd_index.run(
        parse("index"),
        load=lambda **_kwargs: config,
        profiles=cmd_index.load_profile,
        tokenizers=RecordingTokenizers(),
        ingest=ingester,
    )
    assert code == EXIT_USAGE
    assert ingester.calls == []
    assert "embeddinggemma-9000" in capsys.readouterr().err


@pytest.mark.parametrize("shape", ["absent", "a-file"])
def test_an_archive_path_that_is_not_a_directory_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], shape: str
) -> None:
    """Ruling 4: discovery would have made this a silent zero-file success."""
    config = make_config(tmp_path, archive="corpus", create_archive=False)
    if shape == "a-file":
        config.archive_path.parent.mkdir(parents=True, exist_ok=True)
        config.archive_path.write_text("not a directory", encoding="utf-8")

    ingester = RecordingIngester()
    tokenizers = RecordingTokenizers()
    code, _ = invoke(config=config, ingester=ingester, tokenizers=tokenizers)
    captured = capsys.readouterr()
    assert code == EXIT_USAGE
    assert captured.out == ""
    assert str(config.archive_path.resolve()) in captured.err
    for hint in ("--archive-path", "NPU_RAG_ARCHIVE_PATH", "npu-rag.toml"):
        assert hint in captured.err
    assert ingester.calls == [], "nothing may be ingested from a missing root"
    assert tokenizers.calls == [], "a missing root must be diagnosed before a download"


def test_an_existing_but_empty_archive_is_a_successful_zero_file_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The discriminator for the test above: empty is legitimate, absent is not."""
    empty = make_report(
        processed=0, unchanged=0, skipped=0, failed=0, records=0, removed=(), omissions=()
    )
    code, _ = invoke("--json", config=make_config(tmp_path), ingester=RecordingIngester(empty))
    assert code == EXIT_OK
    assert cast("dict[str, int]", payload_of(capsys)["files"])["total"] == 0


def test_a_tokenizer_failure_is_a_runtime_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from npu_rag.embedding.errors import PreparationError

    error = PreparationError(
        "the tokenizer could not be loaded", model_id="test/probe-model", stage="tokenizer"
    )
    code, ingester = invoke(
        config=make_config(tmp_path), tokenizers=RecordingTokenizers(error)
    )
    captured = capsys.readouterr()
    assert code == EXIT_RUNTIME
    assert captured.out == ""
    assert "tokenizer" in captured.err
    assert ingester.calls == []


@pytest.mark.parametrize("kind", ["state", "os"])
def test_a_pipeline_failure_is_a_runtime_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    from npu_rag.ingest.errors import StateError

    error: Exception = (
        StateError("database is locked", path=tmp_path / "ingest.sqlite")
        if kind == "state"
        else OSError("the archive volume went away")
    )
    code, _ = invoke(config=make_config(tmp_path), ingester=RecordingIngester(error=error))
    captured = capsys.readouterr()
    assert code == EXIT_RUNTIME
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert captured.err.startswith("npu-rag: error: ")


def test_full_rebuild_is_announced_on_stderr_and_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flag resets both stores and remains visible in JSON."""
    code, _ = invoke("--json", "--full-rebuild", config=make_config(tmp_path))
    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert cmd_index.REBUILD_NOTICE in captured.err
    assert json.loads(captured.out)["full_rebuild"] is True

    # Progress still goes to stderr; only the rebuild notice is flag specific.
    code, _ = invoke("--json", config=make_config(tmp_path))
    quiet = capsys.readouterr()
    assert code == EXIT_OK
    assert cmd_index.REBUILD_NOTICE not in quiet.err
    assert json.loads(quiet.out)["full_rebuild"] is False


def test_the_provider_flag_reaches_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = invoke("--json", config=make_config(tmp_path, provider="cpu"))
    assert code == EXIT_OK
    assert payload_of(capsys)["requested_provider"] == "cpu"


def test_the_ingest_config_the_pipeline_receives_is_the_bridged_one(
    tmp_path: Path,
) -> None:
    """The seam records the object, so the bridge is observed end to end."""
    config = make_config(tmp_path)
    profile = probe_profile()
    code, ingester = invoke(config=config, profile=profile)
    assert code == EXIT_OK
    received = ingester.calls[0][0]
    assert received.roots == (config.archive_path.resolve(),)
    assert received.token_budget == PROBE_SEQ_LEN
    assert received.state_path == (tmp_path / "store" / "ingest.sqlite").resolve()


# --------------------------------------------------------------------------- #
# Requirement 8.4 / note 3.1 - printing goes through `render.emit`
# --------------------------------------------------------------------------- #


def test_the_summary_is_written_through_render_emit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    emitted: list[str] = []
    monkeypatch.setattr(render_module, "emit", lambda text, **_k: emitted.append(text))
    code, _ = invoke(config=make_config(tmp_path))
    assert code == EXIT_OK
    assert len(emitted) == 2
    assert emitted[0].startswith("Progress:")
    assert emitted[1].startswith("Index Run Summary")
    assert capsys.readouterr().out == "", "nothing may bypass the emit seam"


def test_a_devanagari_path_survives_a_cp1252_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Note 4.1: `capsys` is UTF-8, so only a real cp1252 stream can see this.

    A bare `print` here raises `UnicodeEncodeError` - the charmap crash
    requirement 8.4 names - because nothing would have reconfigured the stream.
    """
    devanagari = "गरुड"
    config = make_config(tmp_path, archive=devanagari)
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", newline="")
    monkeypatch.setattr(sys, "stdout", stream)
    code, _ = invoke(config=config)
    stream.flush()
    assert code == EXIT_OK
    assert devanagari in buffer.getvalue().decode("utf-8")


def test_a_devanagari_archive_error_survives_a_cp1252_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    devanagari = "\u0917\u0930\u0941\u0921"
    config = make_config(tmp_path, archive=devanagari, create_archive=False)
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", newline="")
    monkeypatch.setattr(sys, "stderr", stream)
    code, _ = invoke(config=config)
    stream.flush()
    assert code == EXIT_USAGE
    assert devanagari in buffer.getvalue().decode("utf-8")


def test_sync_receives_model_identity_and_requested_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from npu_rag.embedding.types import ProviderChoice
    from npu_rag.index import format_model_identity

    index = RecordingIndex()
    profile = probe_profile()
    code, _ = invoke("--json", config=make_config(tmp_path, provider="cpu"),
                     profile=profile, index=index)
    payload = payload_of(capsys)
    expected = format_model_identity(profile.model_id, profile.dimension)
    assert code == EXIT_OK
    assert index.identity == expected
    assert index.events == ["verify", "sync"]
    assert index.sync_kwargs["model_identity"] == expected
    assert index.sync_kwargs["provider"] is ProviderChoice.CPU
    assert payload["index_written"] is True
    assert payload["embedding_performed"] is True
    assert payload["active_provider"] == "CPUExecutionProvider"


@pytest.mark.parametrize("served, records, expected", [
    ("npu", 2, "VitisAIExecutionProvider"),
    ("cpu", 2, "CPUExecutionProvider"),
    (None, 0, cmd_index.NO_EMBEDDING_PROVIDER),
    ("rocm", 2, None),
])
def test_actual_provider_attribution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    served: str | None, records: int, expected: str | None,
) -> None:
    code, _ = invoke(
        "--json", config=make_config(tmp_path), embedder=RecordingEmbedder(served),
        ingester=RecordingIngester(make_report(records=records)),
    )
    captured = capsys.readouterr()
    if expected is None:
        assert code == EXIT_RUNTIME
        assert "Traceback" not in captured.err
        assert "active provider attribution" in captured.err
        assert captured.out == ""
    else:
        assert code == EXIT_OK
        assert json.loads(captured.out)["active_provider"] == expected


@pytest.mark.parametrize("requested, usable, expected", [
    ("npu", False, None), ("auto", False, "cpu"),
    ("auto", True, "npu"), ("cpu", True, "cpu"),
])
def test_real_open_embedder_checks_capability_and_reuses_search_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    requested: str, usable: bool, expected: str | None,
) -> None:
    from npu_rag.cli import cmd_search
    from npu_rag.embedding.environment import capability as capability_module
    from npu_rag.embedding import service
    from npu_rag.embedding.errors import NpuUnavailableError
    from npu_rag.embedding.types import CapabilityReport, Condition, ExecutionMode, ProviderChoice

    capability = CapabilityReport(
        (Condition("execution_provider_registered", usable, "test provider absent",
                   "Vitis AI provider", "install vendor runtime"),),
        ExecutionMode.IN_PROCESS if usable else ExecutionMode.UNAVAILABLE,
        None, None, None, False,
    )
    config, profile = make_config(tmp_path, provider=requested), probe_profile()
    calls: list[tuple[Any, ...]] = []
    root = tmp_path / "shared-search-artifacts-sentinel"
    monkeypatch.setattr(cmd_search, "artifacts_root", lambda cfg: root if cfg == config else None)
    monkeypatch.setattr(capability_module, "check_capability", lambda: capability)

    def build(profile: Any, capability: Any, root: Path, *, provider: Any) -> Any:
        calls.append((profile, capability, root, provider))
        return RecordingEmbedder()

    monkeypatch.setattr(service, "build_service", build)
    if expected is None:
        with pytest.raises(NpuUnavailableError, match="test provider absent"):
            cmd_index.open_embedder(config, profile)
        code = cmd_index.run(
            parse("index", "--json"), load=lambda **_kw: config,
            profiles=lambda _name: profile, tokenizers=RecordingTokenizers(),
            ingest=RecordingIngester(), open_index=lambda _path: RecordingIndex(),
        )
        captured = capsys.readouterr()
        assert code == EXIT_RUNTIME
        assert "test provider absent" in captured.err and "--provider cpu" in captured.err
        assert captured.out == "" and "Traceback" not in captured.err
        assert calls == []
    else:
        assert isinstance(cmd_index.open_embedder(config, profile), cmd_index._TrackingEmbedder)
        assert calls == [(profile, capability, cmd_search.artifacts_root(config), ProviderChoice(expected))]


def test_tracking_embedder_records_results_and_rejects_provider_changes() -> None:
    from dataclasses import replace
    from unittest.mock import Mock

    import numpy as np
    from npu_rag.embedding.service import EmbedResult
    from npu_rag.embedding.types import DocumentText, ExecutionMode, ProviderChoice

    result = EmbedResult(
        np.ones((1, 2), dtype=np.float32), ProviderChoice.CPU, ExecutionMode.UNAVAILABLE,
        None, "test NPU unavailable", (), 0.1, 1,
    )
    service = Mock()
    service.embed_documents.side_effect = [result, result, replace(
        result, provider_served=ProviderChoice.NPU, execution_mode=ExecutionMode.IN_PROCESS,
        partition_verified=True, fallback_reason=None,
    )]
    tracker = cmd_index._TrackingEmbedder(service)
    texts = [DocumentText("test document")]
    for _ in range(2):
        assert tracker.embed_documents(texts, ProviderChoice.AUTO) is result
        assert tracker.provider_served == "cpu"
        assert tracker.fallback_reason == "test NPU unavailable"
    service.embed_documents.assert_called_with(texts, ProviderChoice.AUTO)
    with pytest.raises(RuntimeError, match="provider changed"):
        tracker.embed_documents(texts, ProviderChoice.AUTO)


def test_updates_emit_completion_even_without_sync_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npu_rag.index.types import SyncSummary

    class SilentIndex(RecordingIndex):
        def sync(self, report: Any, embed_service: Any, **kwargs: Any) -> Any:
            return SyncSummary(chunks_added=0, chunks_updated=3, chunks_deleted=0,
                               total_chunks=3, files_indexed=1, elapsed_seconds=0.1)

    emitted: list[str] = []
    monkeypatch.setattr(render_module, "emit", lambda text, **_kw: emitted.append(text))
    code, _ = invoke("--json", config=make_config(tmp_path), index=SilentIndex())
    assert code == EXIT_OK
    assert len(emitted) == 2
    assert emitted[0].startswith("Progress:") and "chunks 3/3" in emitted[0]
    assert json.loads(emitted[1])["embedding_performed"] is True


def test_open_index_at_passes_the_path_to_vector_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock
    import npu_rag.index

    constructor = Mock(return_value=RecordingIndex())
    monkeypatch.setattr(npu_rag.index, "VectorIndex", constructor)
    assert cmd_index.open_index_at(tmp_path) is constructor.return_value
    constructor.assert_called_once_with(tmp_path)


def test_auto_cpu_fallback_is_a_notice_on_stderr_and_json_stays_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    embedder = RecordingEmbedder()
    embedder.fallback_reason = "Vitis AI provider is unavailable"
    config = make_config(tmp_path)
    code = cmd_index.run(
        parse("index", "--json"), load=lambda **_kw: config,
        profiles=lambda _name: probe_profile(), tokenizers=RecordingTokenizers(),
        ingest=RecordingIngester(), open_index=lambda _path: RecordingIndex(),
        open_embedder=lambda _config, _profile: embedder,
    )
    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert "notice:" in captured.err
    assert "Vitis AI provider is unavailable" in captured.err
    assert json.loads(captured.out)["active_provider"] == "CPUExecutionProvider"


def test_full_rebuild_resets_before_ingest_and_default_does_not(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    config = make_config(tmp_path)

    def reset(_index: Path, _state: Path) -> None:
        events.append("reset")

    def ingest(config: IngestConfig, tokenizer: ModelTokenizer) -> RunReport:
        events.append("ingest")
        return cast("RunReport", make_report())

    for flags, expected in (((), ["ingest"]), (("--full-rebuild",), ["reset", "ingest"])):
        events.clear()
        code = cmd_index.run(
            parse("index", *flags), load=lambda **_kw: config,
            profiles=lambda _name: probe_profile(), tokenizers=RecordingTokenizers(),
            ingest=ingest, open_index=lambda _path: RecordingIndex(),
            open_embedder=lambda _config, _profile: RecordingEmbedder(), rebuild=reset,
        )
        assert code == EXIT_OK
        assert events == expected


def test_rebuild_discards_the_real_chunk_table_and_ingest_state(tmp_path: Path) -> None:
    import lancedb

    from npu_rag.index.schema import ChunkSchema

    index_path = tmp_path / "lancedb"
    state_path = tmp_path / "ingest.sqlite"
    state_path.write_bytes(b"stale state")
    db = lancedb.connect(str(index_path))
    db.create_table("chunks", schema=ChunkSchema)
    assert "chunks" in db.list_tables().tables

    cmd_index.rebuild_at(index_path, state_path)

    assert "chunks" not in db.list_tables().tables
    assert not state_path.exists()


def test_progress_throttles_on_wall_time_and_always_emits_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[str] = []
    times = iter((0.1, 0.2, 0.6, 0.61))
    monkeypatch.setattr(time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(render_module, "emit", lambda value, **_kw: emitted.append(value))
    progress = cmd_index._Progress(make_report(), 0.0)
    for done in (1, 2, 3, 4):
        progress(done, 4)
    assert len(emitted) == 2
    assert "chunks 3/4" in emitted[0]
    assert "chunks 4/4" in emitted[1]
    assert "files 7/16" in emitted[1]
    assert "throughput" in emitted[1]


# --------------------------------------------------------------------------- #
# The bridge, bridging: a real ingest run with no network and no compile
# --------------------------------------------------------------------------- #


def _real_corpus(root: Path) -> Path:
    """A two-document corpus with one image, built by hand."""
    session = root / "Day 1"
    session.mkdir(parents=True)
    (session / "notes.md").write_text(
        "# Garuda Rite\n\nThe practitioner forms the Garuda mudra.\n\n"
        "![plate](plate.png)\n",
        encoding="utf-8",
    )
    (session / "devanagari.md").write_text(
        "# गरुड\n\nगरुड "
        "मुद्रा।\n",
        encoding="utf-8",
    )
    _write_png(session / "plate.png")
    return root


def _write_png(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (256, 256), (240, 230, 200)).save(path)


def test_the_bridge_drives_a_real_ingest_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end through `run_ingest` itself, with the offline tokenizer.

    No network (the fixture blocks it), no embedding model and no compile: this
    is the test that proves the `IngestConfig` the bridge builds is one the
    pipeline accepts, which a mock-shaped test cannot establish.
    """
    from tests.ingest.conftest import load_offline_tokenizer

    config = make_config(
        tmp_path, archive="corpus", model_name="gte-modernbert-base", create_archive=False
    )
    _real_corpus(config.archive_path)

    code = cmd_index.run(
        parse("index", "--json"),
        load=lambda **_kwargs: config,
        profiles=cmd_index.load_profile,
        tokenizers=lambda _profile: load_offline_tokenizer(),
        ingest=cmd_index.run_pipeline,
        open_index=lambda _path: RecordingIndex(),
        open_embedder=lambda _config, _profile: RecordingEmbedder(),
    )
    payload = payload_of(capsys)
    files = cast("dict[str, int]", payload["files"])
    assert code == EXIT_OK
    assert files["processed"] == 3, payload
    assert files["failed"] == 0, payload
    assert payload["chunk_records"] >= 2
    assert payload["token_budget"] == TODAYS_COMPILED_SEQ_LEN
    assert payload["state_path"].endswith("ingest.sqlite")
    assert Path(cast("str", payload["state_path"])).is_file()
    # The designed degradation: no describer, so the image is reported, named.
    categories = cast("dict[str, int]", payload["omissions_by_category"])
    assert categories.get("vision_unavailable", 0) >= 1, payload
    assert payload["missing_capabilities"] != []


def test_a_second_real_run_skips_every_unchanged_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 7.2, and the proof that `state_path` is reused (ruling 2)."""
    from tests.ingest.conftest import load_offline_tokenizer

    config = make_config(
        tmp_path, archive="corpus", model_name="gte-modernbert-base", create_archive=False
    )
    _real_corpus(config.archive_path)
    tokenizer = load_offline_tokenizer()

    def once() -> dict[str, Any]:
        cmd_index.run(
            parse("index", "--json"),
            load=lambda **_kwargs: config,
            profiles=cmd_index.load_profile,
            tokenizers=lambda _profile: tokenizer,
            ingest=cmd_index.run_pipeline,
            open_index=lambda _path: RecordingIndex(),
            open_embedder=lambda _config, _profile: RecordingEmbedder(),
        )
        return payload_of(capsys)

    first = once()
    second = once()
    assert cast("dict[str, int]", first["files"])["processed"] == 3
    assert cast("dict[str, int]", second["files"])["unchanged"] == 3
    assert cast("dict[str, int]", second["files"])["processed"] == 0
    assert second["no_work_required"] is True
    assert second["chunk_records"] == first["chunk_records"]


# --------------------------------------------------------------------------- #
# Through the real entry point
# --------------------------------------------------------------------------- #


def test_index_help_exits_zero_and_lists_the_designed_flags() -> None:
    completed = _run_cli("index", "--help")
    assert completed.returncode == EXIT_OK
    for flag in EXPECTED_OPTIONS:
        assert flag in completed.stdout, f"{flag} missing from `index --help`"
    assert completed.stderr == ""


def test_the_real_entry_point_reaches_this_handlers_own_diagnostic(
    tmp_path: Path,
) -> None:
    """`main` dispatches to `run`, and ruling 4's message is what comes back.

    Driven through a subprocess rather than with the seams injected, because
    the seams are what the other tests replace: this one has to show that the
    production defaults are wired. `--archive-path` is the highest
    configuration layer, so no file on this machine can change the answer.
    """
    missing = tmp_path / "no-such-corpus"
    completed = _run_cli("index", "--archive-path", str(missing))
    assert completed.returncode == EXIT_USAGE
    assert "is not an existing directory" in completed.stderr
    assert str(missing) in completed.stderr
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr


def test_main_passes_this_handlers_exit_code_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher calls `run(args)` with the namespace alone (note 1.1)."""
    seen: list[argparse.Namespace] = []

    def fake_run(args: argparse.Namespace) -> int:
        seen.append(args)
        return EXIT_OK

    monkeypatch.setattr(cmd_index, "run", fake_run)
    assert main(["index", "--json", "--full-rebuild"]) == EXIT_OK
    assert len(seen) == 1
    assert seen[0].command == "index"
    assert seen[0].json is True
    assert seen[0].full_rebuild is True
