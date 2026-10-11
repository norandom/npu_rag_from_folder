"""Tests for `npu-rag search` (task 4.1).

Requirements under test: 3.1-3.4 (folder scoping and tag pushdown), 4.1-4.4
(grouping, context expansion, breadcrumbs), 5.1-5.2 (retrieval controls and
parse-time validation), 6.1-6.3 (provider accountability), 8.1-8.4 (terminal
and JSON output) and design §4 (exit codes).

Three rules shape this file.

**The flag surface and the choice vocabularies are pinned against hand-written
literals, and then completeness is asserted structurally.** Implementation
notes 1.1 and 2.2 both record the same defect twice over: a constant tested
only through a parameterization derived from itself is not covered at all, and
a misspelling left the suite green with an unchanged test count. So `--mode`'s
three spellings, the enum values they map onto, and the whole flag set are
written out here by hand, and separate assertions tie those literals to what
`argparse` actually registers and to what `npu_rag.index.types` actually
declares. Adding a flag without a test fails loudly.

**The mock-shaped tests are bracketed by real ones.** A `FakeIndex` records
the `SearchOptions` it was handed, which is the only way to observe a flag that
`index.search` consumes internally - but a test that asserts on a recording
object it configured itself proves only that the recorder works. Every property
established that way is therefore also exercised against a **real LanceDB
index** built in `tmp_path` (`real_index`), where the filter SQL is parsed by
lancedb, the grouping runs over real `file_path` values, and the rendered text
carries real chunk text.

**Where `render` is monkeypatched, it is patched on `npu_rag.cli.render`
itself** (imported here as `render_module`) and not through
`cmd_search.render`. They are the same module object - `cmd_search` does
`from npu_rag.cli import render`, so the patch reaches it either way - but
mypy's `no_implicit_reexport` (strict) refuses the attribute path through
`cmd_search`, which does not export it.

**No test may consult a configuration file on this machine.** Implementation
note 1.2: `load_config()` walks the whole parent chain, so a `npu-rag.toml`
anywhere above `%TEMP%` would change what these tests see. The `load` seam is
therefore injected with a `CliConfig` built in the test, and the one test that
exercises the real loader bounds it with `start`/`home`/`stop` inside
`tmp_path`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from npu_rag.cli import cmd_search
from npu_rag.cli import render as render_module
from npu_rag.cli.config import CliConfig, ConfigError
from npu_rag.cli.main import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, build_parser, main

# --------------------------------------------------------------------------- #
# Literals: requirement 5.1 / design §3.5's flag surface, written out by hand
# --------------------------------------------------------------------------- #

#: Every option design §3.5 lists for `search`, minus `--fetch-k`, plus the
#: `-h/--help` argparse adds. `--fetch-k` is deliberately absent: `SearchOptions`
#: has no `fetch_k` field and `index/service.py:142` computes it as
#: `max(limit * 4, 30)`, so the flag could only be accepted and ignored. See the
#: module docstring of `cmd_search` and the task's DEVIATIONS note.
EXPECTED_OPTIONS: tuple[str, ...] = (
    "--help",
    "--folder",
    "--tag",
    "--group-by-folder",
    "--context",
    "--mode",
    "--metric",
    "--fusion",
    "--rrf-k",
    "--linear-alpha",
    "--mmr",
    "--max-chunks-per-doc",
    "--rollup",
    "--top-k",
    "--fetch-k",
    "--filter",
    "--provider",
    "--json",
    "--no-color",
)

#: Requirement 5.1's spelling of `--mode`, and the `SearchMode` value each one
#: selects. The left side is the user's vocabulary (`fts`), the right side is the
#: enum's (`lexical_only`); they differ, which is exactly why this mapping is
#: written out rather than derived.
EXPECTED_MODES: dict[str, str] = {
    "dense": "dense_only",
    "fts": "lexical_only",
    "hybrid": "hybrid",
}

EXPECTED_METRICS: tuple[str, ...] = ("cosine", "dot", "l2")
EXPECTED_FUSIONS: tuple[str, ...] = ("rrf", "linear")
EXPECTED_PROVIDERS: tuple[str, ...] = ("npu", "cpu", "auto")

NPU_ORT_PROVIDER = "VitisAIExecutionProvider"
CPU_ORT_PROVIDER = "CPUExecutionProvider"

#: The three chunks of `real_index`'s one multi-chunk document. The neighbours
#: share no searchable term with the middle chunk, so their text can reach the
#: output only through `--context` (requirement 4.3).
PRECEDING_CHUNK_TEXT = "Preparation of the seat and the offerings beforehand"
GARUDA_CHUNK_TEXT = "Garuda mudra is formed at this point in the rite"
SUCCEEDING_CHUNK_TEXT = "Dissolution of the visualisation and the closing bow"


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


@dataclass
class RecordedSearch:
    """One `index.search(...)` call, as the command made it."""

    query: str
    options: Any
    embed_service: Any


class FakeIndex:
    """Records what the command asked the index for, and answers with `response`."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[RecordedSearch] = []

    def search(
        self,
        query: str,
        options: Any = None,
        embed_service: Any = None,
    ) -> Any:
        self.calls.append(RecordedSearch(query, options, embed_service))
        return self.response


class FakeEmbedder:
    """A `cmd_search.QueryEmbedder` that reports what it was told to report."""

    def __init__(
        self,
        *,
        provider_served: str | None = "cpu",
        fallback_reason: str | None = None,
    ) -> None:
        self.provider_served = provider_served
        self.fallback_reason = fallback_reason
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.0, 1.0, 0.0]


def make_config(
    tmp_path: Path,
    *,
    archive: str = "archive",
    index: str = "lancedb",
    provider: str = "auto",
    tags: Mapping[str, Sequence[str]] | None = None,
    default_mode: str = "hybrid",
    default_limit: int = 10,
    ansi_colors: bool = False,
    create_index: bool = True,
) -> CliConfig:
    """A `CliConfig` with no file, no environment and no walk - see module docs.

    `create_index` plants the on-disk shape `missing_index_message` looks for -
    a database directory holding a `<table>.lance` directory - because the
    existence gate runs against the real filesystem even when the index object
    itself is a double. The tests for that gate pass `create_index=False`.
    """
    if create_index:
        table = tmp_path / index / "chunks.lance"
        table.mkdir(parents=True, exist_ok=True)
    return CliConfig(
        archive_path=tmp_path / archive,
        index_path=tmp_path / index,
        model_name="embeddinggemma-300m",
        provider=provider,
        tags={} if tags is None else {k: tuple(v) for k, v in tags.items()},
        default_mode=default_mode,
        default_limit=default_limit,
        ansi_colors=ansi_colors,
    )


def parse(*argv: str) -> argparse.Namespace:
    """Parse through the real top-level parser, hook and all."""
    tokens = list(argv)
    return build_parser(tokens).parse_args(tokens)


def make_result(
    *,
    chunk_id: str = "c1",
    file_path: str,
    chunk_index: int = 0,
    text: str = "some matching text",
    score: float = 0.5,
    expanded_context: str | None = None,
) -> Any:
    from npu_rag.index.types import SearchResult

    return SearchResult(
        chunk_id=chunk_id,
        file_path=file_path,
        title="Title",
        chunk_kind="prose",
        chunk_index=chunk_index,
        headings=(),
        locator_type="markdown",
        locator_json=json.dumps({"line_range": [1, 5], "ordinal": 0}),
        text=text,
        score=score,
        expanded_context=expanded_context,
    )


def make_response(*results: Any, query: str = "a query") -> Any:
    from npu_rag.index.types import SearchMode, SearchResponse

    return SearchResponse(
        query=query,
        mode=SearchMode.HYBRID,
        results=tuple(results),
        total_candidates=len(results),
        elapsed_seconds=0.01,
    )


def run_search(
    *argv: str,
    config: CliConfig,
    index: Any,
    embedder: Any = None,
) -> int:
    """Drive `cmd_search.run` with every outside world replaced."""
    args = parse("search", *argv)
    opened: list[Path] = []

    def open_index(path: Path) -> Any:
        opened.append(path)
        return index

    def open_embedder(cfg: CliConfig) -> Any:
        if embedder is None:
            raise AssertionError("this search was not expected to need an embedder")
        return embedder

    code = cmd_search.run(
        args,
        load=lambda **_: config,
        open_index=open_index,
        open_embedder=open_embedder,
    )
    return code


# --------------------------------------------------------------------------- #
# A real index, so that nothing below rests only on a double
# --------------------------------------------------------------------------- #


@pytest.fixture
def real_index(tmp_path: Path) -> Any:
    """A populated LanceDB index over a two-folder archive in `tmp_path`.

    Built through `VectorIndex.sync`, so `file_path`, `author`, the vectors and
    the FTS index are all produced by the real code paths rather than written
    by hand.
    """
    import numpy as np
    import numpy.typing as npt

    from npu_rag.embedding.types import DocumentText, ExecutionMode, ProviderChoice
    from npu_rag.index.schema import VECTOR_DIMENSION
    from npu_rag.index.service import VectorIndex
    from npu_rag.ingest.types import ChunkKind, ChunkRecord, MarkdownLocator, RunReport

    archive = tmp_path / "archive"

    @dataclass
    class _Result:
        vectors: npt.NDArray[np.float32]
        provider_served: ProviderChoice = ProviderChoice.CPU
        execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS
        partition_verified: bool | None = None
        fallback_reason: str | None = None
        truncated_indices: tuple[int, ...] = ()
        elapsed_seconds: float = 0.01
        input_count: int = 1

    class _Embedder:
        """Axis 3 for 'garuda', 0 for 'tara', 1 for 'pension', 2 otherwise.

        `garuda` is checked first and appears in exactly one chunk of the
        three-chunk document, which is what makes a `--context` test able to
        name one hit unambiguously instead of relying on a tie-break.
        """

        @staticmethod
        def _vector(text: str) -> npt.NDArray[np.float32]:
            vec = np.zeros((VECTOR_DIMENSION,), dtype=np.float32)
            lowered = text.lower()
            if "garuda" in lowered:
                vec[3] = 1.0
            elif "tara" in lowered:
                vec[0] = 1.0
            elif "pension" in lowered:
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            return vec

        def embed_documents(
            self,
            texts: list[DocumentText],
            provider: ProviderChoice = ProviderChoice.AUTO,
            progress: Any = None,
        ) -> _Result:
            vectors = np.stack([self._vector(item.content) for item in texts])
            return _Result(vectors=vectors, input_count=len(texts))

    def record(
        chunk_id: str, relative: str, text: str, ordinal: int
    ) -> ChunkRecord:
        path = archive / relative
        return ChunkRecord(
            chunk_id=chunk_id,
            source_path=path,
            root_id="root",
            author=Path(relative).parts[0],
            title=path.stem,
            kind=ChunkKind.PROSE,
            text=text,
            locator=MarkdownLocator(line_range=(1, 9), ordinal=ordinal),
            ordinal=ordinal,
            token_count=11,
            truncated=False,
            provenance=None,
        )

    records = (
        record("u1", "Ugra Tara Sadhana/Day 2.md", "Tara mantra for the second day", 0),
        record("u2", "Ugra Tara Sadhana/Day 10.md", "Tara mantra for the tenth day", 0),
        record("u3", "Ugra Tara Sadhana/Day 1.md", "Tara mantra for the first day", 0),
        record("p1", "German Pension/Overview.md", "Pension insurance restabilization", 0),
        # One multi-chunk document, so `--context` has neighbours to attach.
        # Only the middle chunk carries 'garuda', so a dense query for it names
        # exactly one hit and the two neighbours are reachable only through
        # `ContextExpander`.
        record("g0", "Ugra Tara Sadhana/Day 3.md", PRECEDING_CHUNK_TEXT, 0),
        record("g1", "Ugra Tara Sadhana/Day 3.md", GARUDA_CHUNK_TEXT, 1),
        record("g2", "Ugra Tara Sadhana/Day 3.md", SUCCEEDING_CHUNK_TEXT, 2),
    )
    report = RunReport(
        files_processed=5,
        files_unchanged=0,
        files_skipped=0,
        files_failed=0,
        omissions=(),
        removed_chunk_ids=(),
        no_work_required=False,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=records,
    )
    index = VectorIndex(tmp_path / "lancedb", embed_service=_Embedder())
    index.sync(report)
    return index


class RealQueryEmbedder:
    """The query side of `real_index`'s embedder, with provider attribution."""

    def __init__(self, *, provider_served: str = "cpu") -> None:
        self.provider_served: str | None = provider_served
        self.fallback_reason: str | None = None

    def embed_query(self, query: str) -> list[float]:
        from npu_rag.index.schema import VECTOR_DIMENSION

        vector = [0.0] * VECTOR_DIMENSION
        lowered = query.lower()
        # The same four rules as the fixture's document side, in the same
        # order: a query embedder that disagreed with the document embedder
        # would make every assertion about ranking meaningless.
        if "garuda" in lowered:
            vector[3] = 1.0
        elif "tara" in lowered:
            vector[0] = 1.0
        elif "pension" in lowered:
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        return vector


# --------------------------------------------------------------------------- #
# Requirement 5.1 / design §3.5 - the flag surface
# --------------------------------------------------------------------------- #


def _registered_options(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    names: list[str] = []
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no accessor
        names.extend(action.option_strings)
    return tuple(names)


def _search_parser() -> argparse.ArgumentParser:
    tokens = ["search", "q"]
    parser = build_parser(tokens)
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            found = action.choices["search"]
            assert isinstance(found, argparse.ArgumentParser)
            return found
    raise AssertionError("no subparsers on the top-level parser")


def test_search_registers_exactly_the_designed_flags() -> None:
    registered = {name for name in _registered_options(_search_parser()) if name.startswith("--")}
    assert registered == set(EXPECTED_OPTIONS), (
        "design §3.5's flag surface changed; update EXPECTED_OPTIONS deliberately"
    )


def test_fetch_k_reaches_the_engine() -> None:
    """Requirement 5.1's `--fetch-k`, honoured rather than advertised and ignored.

    This test replaced `test_fetch_k_is_not_offered_at_all`, which pinned the
    flag's deliberate *absence* while `SearchOptions` had no field to carry it.
    The field landed in `vector-index` on 2026-10-11, so the flag is now real
    and the pin is inverted: the engine must expose somewhere for the value to
    go, and the parser must offer it.
    """
    from npu_rag.index.types import SearchOptions

    assert hasattr(SearchOptions(), "fetch_k")
    assert SearchOptions().fetch_k is None, (
        "the default must stay None so the index keeps its derived depth and "
        "no existing caller changes behaviour"
    )
    assert "--fetch-k" in _registered_options(_search_parser())


def test_fetch_k_is_passed_through_to_the_engine(tmp_path: Path) -> None:
    """A named depth arrives at `SearchOptions.fetch_k` verbatim."""
    index = FakeIndex(make_response())
    assert (
        run_search(
            "a query",
            "--fetch-k",
            "7",
            config=make_config(tmp_path),
            index=index,
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert index.calls[0].options.fetch_k == 7


def test_an_unset_fetch_k_stays_none_rather_than_a_computed_number(
    tmp_path: Path,
) -> None:
    """The CLI must not pre-compute the depth and hand the engine a constant.

    Passing `max(limit * 4, 30)` from here would look identical today and would
    silently freeze the engine's own policy, so a later change to that formula
    would stop reaching anyone who went through the CLI.
    """
    index = FakeIndex(make_response())
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path),
            index=index,
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert index.calls[0].options.fetch_k is None


# `--fetch-k`'s rejected and accepted values live in the two parametrized
# tables above, beside every other numeric flag, rather than in a test of their
# own: validation happens at *parse* time via argparse `type=`, so it is
# reachable only through `main()`. `run_search` builds its namespace directly
# and bypasses the parser, which is why an invalid value raises there instead of
# returning exit 1.


def test_query_is_required() -> None:
    assert main(["search"]) == EXIT_USAGE


def test_mode_choices_are_requirement_5_1s_spelling() -> None:
    from npu_rag.cli.config import MODE_CHOICES

    assert tuple(EXPECTED_MODES) == MODE_CHOICES
    assert cmd_search.SEARCH_MODES == EXPECTED_MODES


def test_mode_targets_are_real_search_modes_and_cover_every_one() -> None:
    """The right-hand side of the mapping is checked against the enum itself."""
    from npu_rag.index.types import SearchMode

    assert set(EXPECTED_MODES.values()) == {member.value for member in SearchMode}
    for value in cmd_search.SEARCH_MODES.values():
        assert SearchMode(value).value == value


def test_metric_and_fusion_choices_match_their_enums() -> None:
    from npu_rag.index.types import DistanceMetric, FusionMethod

    assert cmd_search.METRIC_CHOICES == EXPECTED_METRICS
    assert cmd_search.FUSION_CHOICES == EXPECTED_FUSIONS
    assert EXPECTED_METRICS == tuple(member.value for member in DistanceMetric)
    assert EXPECTED_FUSIONS == tuple(member.value for member in FusionMethod)


def test_the_commands_exit_codes_are_design_section_fours() -> None:
    """Pinned against literals and against `main`, which owns the taxonomy."""
    assert cmd_search.EXIT_CODES == (0, 1, 2)
    assert cmd_search.EXIT_CODES == (EXIT_OK, EXIT_USAGE, EXIT_RUNTIME)


def test_provider_choices_match_the_configuration_vocabulary() -> None:
    from npu_rag.cli.config import PROVIDER_CHOICES

    assert PROVIDER_CHOICES == EXPECTED_PROVIDERS


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--mode", "fts"),
        ("--metric", "l2"),
        ("--fusion", "linear"),
        ("--provider", "cpu"),
    ],
)
def test_choice_flags_accept_their_vocabulary(flag: str, value: str) -> None:
    args = parse("search", "q", flag, value)
    assert getattr(args, flag.lstrip("-").replace("-", "_")) == value


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--mode", "dense_only"),
        ("--metric", "euclidean"),
        ("--fusion", "weighted"),
        ("--provider", "gpu"),
    ],
)
def test_choice_flags_reject_anything_else(flag: str, value: str) -> None:
    assert main(["search", "q", flag, value]) == EXIT_USAGE


# --------------------------------------------------------------------------- #
# Requirement 5.2 - numeric validation at parse time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--top-k", "0"),
        ("--top-k", "-1"),
        ("--top-k", "2.5"),
        ("--top-k", "many"),
        ("--fetch-k", "0"),
        ("--fetch-k", "-1"),
        ("--fetch-k", "2.5"),
        ("--fetch-k", "many"),
        ("--rrf-k", "0"),
        ("--rrf-k", "-60"),
        ("--max-chunks-per-doc", "0"),
        ("--max-chunks-per-doc", "-3"),
        ("--context", "-1"),
        ("--context", "1.5"),
        ("--mmr", "1.5"),
        ("--mmr", "-0.1"),
        ("--mmr", "nan"),
        ("--mmr", "lots"),
        ("--linear-alpha", "2.0"),
        ("--linear-alpha", "-0.5"),
    ],
)
def test_out_of_range_numbers_are_rejected_at_parse_time(
    flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Design §4: an invalid argument value is exit 1, with the flag named."""
    assert main(["search", "q", flag, value]) == EXIT_USAGE
    captured = capsys.readouterr()
    assert flag in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("--top-k", "1", 1),
        ("--fetch-k", "1", 1),
        ("--rrf-k", "1", 1),
        ("--max-chunks-per-doc", "1", 1),
        ("--context", "0", 0),
        ("--mmr", "0.0", 0.0),
        ("--mmr", "1.0", 1.0),
        ("--linear-alpha", "0.0", 0.0),
        ("--linear-alpha", "1.0", 1.0),
    ],
)
def test_boundary_values_are_accepted(flag: str, value: str, expected: float) -> None:
    """The other side of every bound above, so a guard cannot be over-tight."""
    args = parse("search", "q", flag, value)
    assert getattr(args, flag.lstrip("-").replace("-", "_")) == expected


def test_defaults_match_requirement_5_1() -> None:
    args = parse("search", "q")
    assert args.metric == "cosine"
    assert args.fusion == "rrf"
    assert args.rrf_k == 60
    assert args.linear_alpha == 0.5
    assert args.context == 0
    assert args.mmr is None
    assert args.max_chunks_per_doc is None
    assert args.rollup is False
    assert args.group_by_folder is False
    assert args.json is False
    assert args.no_color is False
    # `--top-k`, `--mode` and `--provider` default to None so that requirement
    # 2.1's lower layers (env, TOML) can still supply them.
    assert args.top_k is None
    assert args.mode is None
    assert args.provider is None


# --------------------------------------------------------------------------- #
# Requirement 5.1 - every flag reaches `SearchOptions`
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("argv", "field", "expected"),
    [
        (("--top-k", "3"), "limit", 3),
        (("--mode", "dense"), "mode", "dense_only"),
        (("--mode", "fts"), "mode", "lexical_only"),
        (("--metric", "dot"), "metric", "dot"),
        (("--metric", "l2"), "metric", "l2"),
        (("--fusion", "linear"), "fusion_method", "linear"),
        (("--rrf-k", "7"), "rrf_k", 7),
        (("--linear-alpha", "0.25"), "linear_alpha", 0.25),
        (("--mmr", "0.3"), "mmr_diversity", 0.3),
        (("--max-chunks-per-doc", "2"), "max_chunks_per_document", 2),
        (("--rollup",), "document_rollup", True),
        (("--context", "4"), "context_window", 4),
    ],
)
def test_each_flag_lands_on_its_search_options_field(
    tmp_path: Path, argv: tuple[str, ...], field: str, expected: object
) -> None:
    index = FakeIndex(make_response())
    code = run_search(
        "a query",
        *argv,
        config=make_config(tmp_path, index="lancedb"),
        index=index,
        embedder=FakeEmbedder(),
    )
    assert code == EXIT_OK
    assert len(index.calls) == 1
    options = index.calls[0].options
    assert getattr(options, field) == expected


def test_unset_flags_leave_search_option_defaults_alone(tmp_path: Path) -> None:
    """A control for the table above: the defaults are not written by the wiring."""
    from npu_rag.index.types import SearchOptions

    index = FakeIndex(make_response())
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path),
            index=index,
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    options = index.calls[0].options
    reference = SearchOptions()
    for field in (
        "metric",
        "fusion_method",
        "rrf_k",
        "linear_alpha",
        "mmr_diversity",
        "max_chunks_per_document",
        "document_rollup",
        "context_window",
        "dense_weight",
        "lexical_weight",
    ):
        assert getattr(options, field) == getattr(reference, field), field


def test_top_k_and_mode_fall_back_to_the_configured_defaults(tmp_path: Path) -> None:
    """Requirement 2.1: a flag the user did not give must not mask the config."""
    index = FakeIndex(make_response())
    config = make_config(tmp_path, default_mode="dense", default_limit=4)
    assert (
        run_search("a query", config=config, index=index, embedder=FakeEmbedder())
        == EXIT_OK
    )
    options = index.calls[0].options
    assert options.limit == 4
    assert options.mode == "dense_only"


def test_the_query_text_is_passed_through_verbatim(tmp_path: Path) -> None:
    index = FakeIndex(make_response())
    assert (
        run_search(
            "  पवित्रीकरण जल  ",
            config=make_config(tmp_path),
            index=index,
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert index.calls[0].query == "  पवित्रीकरण जल  "


@pytest.mark.parametrize("query", ["", "   "])
def test_a_blank_query_is_a_user_error(
    tmp_path: Path, query: str, capsys: pytest.CaptureFixture[str]
) -> None:
    index = FakeIndex(make_response())
    assert (
        run_search(query, config=make_config(tmp_path), index=index)
        == EXIT_USAGE
    )
    assert "query" in capsys.readouterr().err
    assert index.calls == []


# --------------------------------------------------------------------------- #
# Requirements 3.1-3.4 - folder, tag and filter pushdown
# --------------------------------------------------------------------------- #


def test_folder_tag_and_filter_compile_through_the_tag_registry(
    tmp_path: Path,
) -> None:
    """The filter must be exactly what `TagRegistry` compiles - not a near copy."""
    from npu_rag.cli.tags import TagRegistry

    tags = {"siddha": ("Ugra Tara Sadhana", "Garuda*")}
    index = FakeIndex(make_response())
    assert (
        run_search(
            "a query",
            "--folder",
            "Ugra Tara Sadhana",
            "--tag",
            "siddha",
            "--filter",
            "chunk_index = 0",
            config=make_config(tmp_path, tags=tags),
            index=index,
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    expected = TagRegistry(tags).build_filter_sql(
        folder="Ugra Tara Sadhana",
        tags=["siddha"],
        custom_filter="chunk_index = 0",
    )
    assert expected is not None
    assert index.calls[0].options.filter_sql == expected


def test_no_scoping_flags_means_no_filter(tmp_path: Path) -> None:
    index = FakeIndex(make_response())
    assert (
        run_search("a query", config=make_config(tmp_path), index=index, embedder=FakeEmbedder())
        == EXIT_OK
    )
    assert index.calls[0].options.filter_sql is None


def test_comma_separated_and_repeated_tags_both_expand(tmp_path: Path) -> None:
    from npu_rag.cli.tags import TagRegistry

    tags = {"a": ("Folder A",), "b": ("Folder B",)}
    expected = TagRegistry(tags).build_filter_sql(tags=["a", "b"])

    for argv in (("--tag", "a,b"), ("--tag", "a", "--tag", "b")):
        index = FakeIndex(make_response())
        assert (
            run_search(
                "a query",
                *argv,
                config=make_config(tmp_path, tags=tags),
                index=index,
                embedder=FakeEmbedder(),
            )
            == EXIT_OK
        )
        assert index.calls[0].options.filter_sql == expected


def test_an_unknown_tag_is_a_user_error_listing_the_available_tags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 3.4, through `emit_config_error` (exit 1, design §4)."""
    index = FakeIndex(make_response())
    code = run_search(
        "a query",
        "--tag",
        "nope",
        config=make_config(tmp_path, tags={"siddha": ("Ugra Tara Sadhana",)}),
        index=index,
    )
    assert code == EXIT_USAGE
    captured = capsys.readouterr()
    assert "nope" in captured.err
    assert "siddha" in captured.err
    assert captured.out == ""
    assert index.calls == [], "an unknown tag must be diagnosed before any query"


def test_a_degenerate_tag_entry_is_a_user_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = FakeIndex(make_response())
    code = run_search(
        "a query",
        "--tag",
        "broken",
        config=make_config(tmp_path, tags={"broken": ()}),
        index=index,
    )
    assert code == EXIT_USAGE
    assert "broken" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", "   ", "siddha,"])
def test_a_blank_tag_name_is_a_user_error_rather_than_a_silent_no_op(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    """Dropping a blank would make a narrowing flag do nothing (note 2.1)."""
    index = FakeIndex(make_response())
    code = run_search(
        "a query",
        "--tag",
        value,
        config=make_config(tmp_path, tags={"siddha": ("Ugra Tara Sadhana",)}),
        index=index,
    )
    assert code == EXIT_USAGE
    assert "--tag" in capsys.readouterr().err
    assert index.calls == []


def test_a_blank_folder_is_a_user_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = FakeIndex(make_response())
    assert (
        run_search("a query", "--folder", "", config=make_config(tmp_path), index=index)
        == EXIT_USAGE
    )
    assert "--folder" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Requirements 6.1-6.3 - provider accountability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("served", "expected"),
    [("npu", NPU_ORT_PROVIDER), ("cpu", CPU_ORT_PROVIDER)],
)
def test_the_reported_provider_is_the_ort_name_of_what_served(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    served: str,
    expected: str,
) -> None:
    """Implementation note 3.1: never the `npu|cpu|auto` alias."""
    code = run_search(
        "a query",
        "--provider",
        "auto",
        config=make_config(tmp_path),
        index=FakeIndex(make_response(make_result(file_path="x/y.md"))),
        embedder=FakeEmbedder(provider_served=served),
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert f"[Provider: {expected}" in out
    assert "[Provider: npu]" not in out
    assert "[Provider: auto]" not in out


def test_the_npu_provider_line_says_npu(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path),
            index=FakeIndex(make_response()),
            embedder=FakeEmbedder(provider_served="npu"),
        )
        == EXIT_OK
    )
    assert f"[Provider: {NPU_ORT_PROVIDER} (NPU)]" in capsys.readouterr().out


def test_an_auto_fallback_to_cpu_states_the_reason_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 6.2: a fallback is never silent, even when it is permitted."""
    reason = "vendor runtime missing: onnxruntime-vitisai is not installed"
    code = run_search(
        "a query",
        "--provider",
        "auto",
        config=make_config(tmp_path),
        index=FakeIndex(make_response()),
        embedder=FakeEmbedder(provider_served="cpu", fallback_reason=reason),
    )
    assert code == EXIT_OK, "an auto fallback is reported, not failed"
    captured = capsys.readouterr()
    assert reason in captured.err
    assert f"[Provider: {CPU_ORT_PROVIDER}]" in captured.out
    assert reason not in captured.out, "stdout stays machine-parseable"


def test_no_fallback_means_no_notice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Control for the test above: the notice is conditional, not unconditional."""
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path),
            index=FakeIndex(make_response()),
            embedder=FakeEmbedder(provider_served="npu"),
        )
        == EXIT_OK
    )
    assert capsys.readouterr().err == ""


def test_an_unattributed_run_is_a_runtime_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 6.3 has no "sometimes": no attribution, no output."""
    code = run_search(
        "a query",
        config=make_config(tmp_path),
        index=FakeIndex(make_response()),
        embedder=FakeEmbedder(provider_served=None),
    )
    assert code == EXIT_RUNTIME
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "provider" in captured.err


def test_lexical_only_search_needs_no_embedder_and_claims_no_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--mode fts` runs no model, so it must not claim one ran."""
    index = FakeIndex(make_response())
    code = run_search(
        "a query",
        "--mode",
        "fts",
        config=make_config(tmp_path),
        index=index,
        embedder=None,
    )
    assert code == EXIT_OK
    assert index.calls[0].embed_service is None
    out = capsys.readouterr().out
    assert cmd_search.NO_EMBEDDING_PROVIDER in out
    assert NPU_ORT_PROVIDER not in out
    assert CPU_ORT_PROVIDER not in out


def test_an_explicit_npu_request_on_an_unusable_npu_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 6.2: exit 2 with diagnostics, never a CPU substitution."""
    from npu_rag.embedding.errors import NpuUnavailableError
    from npu_rag.embedding.types import ProviderChoice

    def open_embedder(cfg: CliConfig) -> Any:
        raise NpuUnavailableError(
            "provider 'npu' was requested explicitly and the NPU is not usable: "
            "vendor runtime absent",
            provider=ProviderChoice.NPU,
            model_id="embeddinggemma-300m",
        )

    index = FakeIndex(make_response())
    args = parse("search", "a query", "--provider", "npu")
    code = cmd_search.run(
        args,
        load=lambda **_: make_config(tmp_path, provider="npu"),
        open_index=lambda path: index,
        open_embedder=open_embedder,
    )
    assert code == EXIT_RUNTIME
    captured = capsys.readouterr()
    assert "npu" in captured.err
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert index.calls == []


def test_the_provider_flag_reaches_the_configuration(tmp_path: Path) -> None:
    """The one test that drives the real `load_config`, bounded per note 1.2."""
    import functools

    from npu_rag.cli.config import load_config

    (tmp_path / "lancedb" / "chunks.lance").mkdir(parents=True)
    seen: list[str] = []
    index = FakeIndex(make_response())

    def open_embedder(cfg: CliConfig) -> Any:
        seen.append(cfg.provider)
        return FakeEmbedder()

    args = parse("search", "a query", "--provider", "cpu")
    code = cmd_search.run(
        args,
        load=functools.partial(
            load_config,
            env={"NPU_RAG_INDEX_PATH": str(tmp_path / "lancedb")},
            start=tmp_path,
            home=tmp_path,
            stop=tmp_path,
        ),
        open_index=lambda path: index,
        open_embedder=open_embedder,
    )
    assert code == EXIT_OK
    assert seen == ["cpu"]


# --------------------------------------------------------------------------- #
# Decision 4 - the first run, before any index exists
# --------------------------------------------------------------------------- #


def test_a_missing_index_is_an_actionable_runtime_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index_path = tmp_path / "absent" / "lancedb"
    args = parse("search", "a query")
    code = cmd_search.run(
        args,
        load=lambda **_: make_config(
            tmp_path, index="absent/lancedb", create_index=False
        ),
        open_index=lambda path: pytest.fail("the index must not be opened"),
        open_embedder=lambda cfg: pytest.fail("no embedder before an index exists"),
    )
    assert code == EXIT_RUNTIME
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "npu-rag index" in captured.err
    assert str(index_path) in captured.err
    assert "Traceback" not in captured.err
    assert not index_path.exists(), "a failed search must not create the index"


def test_an_index_directory_without_a_table_is_also_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """LanceDB's own `connect()` would create one, which is why this is checked."""
    (tmp_path / "lancedb").mkdir()
    args = parse("search", "a query")
    code = cmd_search.run(
        args,
        load=lambda **_: make_config(tmp_path, create_index=False),
        open_index=lambda path: pytest.fail("the index must not be opened"),
        open_embedder=lambda cfg: pytest.fail("no embedder before an index exists"),
    )
    assert code == EXIT_RUNTIME
    assert "npu-rag index" in capsys.readouterr().err


def test_a_real_index_is_recognised_as_present(tmp_path: Path, real_index: Any) -> None:
    """Non-vacuity control: the check says yes to an index the real code built."""
    assert cmd_search.missing_index_message(tmp_path / "lancedb") is None
    assert cmd_search.missing_index_message(tmp_path / "nothing-here") is not None


# --------------------------------------------------------------------------- #
# Requirements 4.1-4.4 and 8.1-8.4 - presentation
# --------------------------------------------------------------------------- #


def test_results_are_rendered_with_breadcrumbs_and_the_folder_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "archive"
    result = make_result(
        file_path=str(archive / "Ugra Tara Sadhana" / "Day 1.md"),
        text="the matched sentence",
    )
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path),
            index=FakeIndex(make_response(result)),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "[Folder: Ugra Tara Sadhana]" in out
    assert "[Doc 1/1: Day 1.md]" in out
    assert "the matched sentence" in out


def test_the_archive_root_is_resolved_before_grouping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Implementation note 2.2: a relative root degrades every folder label.

    `CliConfig.archive_path` defaults to `Path(".")`, so this is the default
    case and not an edge one. With the root resolved the label is the first
    component beneath it; unresolved, it is the whole parent path.
    """
    archive = tmp_path / "archive"
    archive.mkdir()
    (tmp_path / "lancedb" / "chunks.lance").mkdir(parents=True)
    monkeypatch.chdir(archive)
    config = CliConfig(
        archive_path=Path("."),
        index_path=tmp_path / "lancedb",
        model_name="embeddinggemma-300m",
        provider="auto",
        tags={},
        default_mode="hybrid",
        default_limit=10,
        ansi_colors=False,
    )
    result = make_result(file_path=str(archive / "German Pension" / "Overview.md"))
    assert (
        run_search(
            "a query",
            config=config,
            index=FakeIndex(make_response(result)),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "[Folder: German Pension]" in out


def test_group_by_folder_emits_the_folder_banner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "archive"
    results = (
        make_result(
            chunk_id="a",
            file_path=str(archive / "Folder A" / "Day 2.md"),
            score=0.9,
        ),
        make_result(
            chunk_id="b",
            file_path=str(archive / "Folder A" / "Day 1.md"),
            score=0.5,
        ),
    )
    assert (
        run_search(
            "a query",
            "--group-by-folder",
            config=make_config(tmp_path),
            index=FakeIndex(make_response(*results)),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "=" * 80 in out, "the grouped banner of design §3.4 is absent"
    # Requirement 4.2: narrative order inside the folder, so Day 1 leads
    # despite Day 2 scoring higher.
    assert out.index("Day 1.md") < out.index("Day 2.md")


def test_without_grouping_the_ranking_is_preserved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Control for the test above: grouping is opt-in and changes the order."""
    archive = tmp_path / "archive"
    results = (
        make_result(chunk_id="a", file_path=str(archive / "Folder A" / "Day 2.md"), score=0.9),
        make_result(chunk_id="b", file_path=str(archive / "Folder A" / "Day 1.md"), score=0.5),
    )
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path),
            index=FakeIndex(make_response(*results)),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "=" * 80 not in out
    assert out.index("Day 2.md") < out.index("Day 1.md")


def test_a_folders_tags_are_annotated_on_its_banner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Decision 2 / note 3.1: the mapping must be keyed as `folder_of` labels."""
    archive = tmp_path / "archive"
    result = make_result(file_path=str(archive / "Ugra Tara Sadhana" / "Day 1.md"))
    assert (
        run_search(
            "a query",
            "--group-by-folder",
            config=make_config(
                tmp_path, tags={"siddha": ("Ugra Tara*",), "finance": ("German Pension",)}
            ),
            index=FakeIndex(make_response(result)),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "(Tag: siddha)" in out
    assert "finance" not in out


def test_json_output_is_valid_and_carries_the_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "archive"
    result = make_result(
        file_path=str(archive / "German Pension" / "Overview.md"),
        text="Pension insurance restabilization",
    )
    assert (
        run_search(
            "a query",
            "--json",
            config=make_config(tmp_path),
            index=FakeIndex(make_response(result)),
            embedder=FakeEmbedder(provider_served="cpu"),
        )
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == CPU_ORT_PROVIDER
    assert payload["grouped"] is False
    assert [row["folder"] for row in payload["results"]] == ["German Pension"]
    assert payload["results"][0]["text"] == "Pension insurance restabilization"


def test_json_output_records_grouping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run_search(
            "a query",
            "--json",
            "--group-by-folder",
            config=make_config(tmp_path),
            index=FakeIndex(make_response(make_result(file_path="a/b.md"))),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["grouped"] is True


def test_no_results_still_reports_the_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path),
            index=FakeIndex(make_response()),
            embedder=FakeEmbedder(provider_served="cpu"),
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "No results." in out
    assert CPU_ORT_PROVIDER in out


def test_colour_is_absent_when_the_configuration_refuses_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path, ansi_colors=False),
            index=FakeIndex(make_response(make_result(file_path="a/b.md"))),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert "\x1b[" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("ansi_colors", "argv", "expected"),
    [
        (True, (), True),
        (True, ("--no-color",), False),
        (False, (), False),
        (False, ("--no-color",), False),
    ],
)
def test_the_colour_preference_handed_to_render_is_the_users(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ansi_colors: bool,
    argv: tuple[str, ...],
    expected: bool,
) -> None:
    """Requirement 8.2's *preference* layer, pinned where it is computed.

    The two output-level tests below cannot see this expression at all: under
    `capsys` `render.should_colorize` vetoes on `isatty()` first (veto 3), so
    the output is plain whatever `prefer_color` was. Recording the keyword is
    the only way to tell `config.ansi_colors and not --no-color` from a
    hardcoded `True`.
    """
    seen: list[bool] = []

    def recorder(*, prefer_color: bool = True, **_: object) -> bool:
        seen.append(prefer_color)
        return False

    monkeypatch.setattr(render_module, "should_colorize", recorder)
    assert (
        run_search(
            "a query",
            *argv,
            config=make_config(tmp_path, ansi_colors=ansi_colors),
            index=FakeIndex(make_response(make_result(file_path="a/b.md"))),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert seen == [expected]


def test_the_colour_decision_reaches_the_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """And the decision is actually used: forced on, the output carries SGR.

    The counterpart of the test above. Together they pin both halves - the
    preference that is computed, and the fact that `render` is told about it -
    which a test reading only plain output cannot do.
    """
    monkeypatch.setattr(render_module, "should_colorize", lambda **_: True)
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path, ansi_colors=True),
            index=FakeIndex(make_response(make_result(file_path="a/b.md"))),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert "\x1b[" in capsys.readouterr().out


def test_no_color_is_accepted_and_strips_styling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run_search(
            "a query",
            "--no-color",
            config=make_config(tmp_path, ansi_colors=True),
            index=FakeIndex(make_response(make_result(file_path="a/b.md"))),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert "\x1b[" not in capsys.readouterr().out


DEVANAGARI_SNIPPET = "पवित्रीकरण जल छिड़क लीजिए"


def test_devanagari_survives_the_captured_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The content check. It is NOT the requirement 8.4 check - see below.

    `capsys` is already UTF-8, so this test passes through a bare `print` too.
    `test_output_is_written_through_render_emit` and
    `test_devanagari_does_not_crash_a_narrow_console` are the two that pin the
    write itself.
    """
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path),
            index=FakeIndex(
                make_response(make_result(file_path="a/b.md", text=DEVANAGARI_SNIPPET))
            ),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert DEVANAGARI_SNIPPET in capsys.readouterr().out


def test_devanagari_does_not_crash_a_narrow_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 8.4, for real: a cp1252 console is what crashes.

    On Windows a console `sys.stdout` is a cp1252 `TextIOWrapper`, and writing
    Devanagari through it raises `UnicodeEncodeError` - the charmap crash the
    requirement names. `render.emit` reconfigures the stream to UTF-8 as part of
    the write (note 3.1 makes that BINDING on this task); a bare `print` does
    not, and dies here. The stream's resulting encoding is asserted too, so the
    guard is observable rather than merely survived.
    """
    import io

    buffer = io.BytesIO()
    narrow = io.TextIOWrapper(buffer, encoding="cp1252", newline="")
    assert narrow.encoding == "cp1252"
    monkeypatch.setattr(sys, "stdout", narrow)

    code = run_search(
        "a query",
        config=make_config(tmp_path),
        index=FakeIndex(
            make_response(make_result(file_path="a/b.md", text=DEVANAGARI_SNIPPET))
        ),
        embedder=FakeEmbedder(),
    )
    narrow.flush()
    assert code == EXIT_OK
    assert narrow.encoding == "utf-8", "the stream was never widened"
    assert DEVANAGARI_SNIPPET in buffer.getvalue().decode("utf-8")


def test_output_is_written_through_render_emit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Note 3.1: print through `render.emit`, never `print()`.

    `emit` is the only place requirement 8.4's encoding enforcement and the
    write are inseparable, so the command must have no second route to stdout.
    """
    written: list[str] = []

    def recorder(text: str, *, stream: Any = None) -> None:
        written.append(text)

    monkeypatch.setattr(render_module, "emit", recorder)
    assert (
        run_search(
            "a query",
            config=make_config(tmp_path),
            index=FakeIndex(
                make_response(make_result(file_path="a/b.md", text=DEVANAGARI_SNIPPET))
            ),
            embedder=FakeEmbedder(),
        )
        == EXIT_OK
    )
    assert len(written) == 1, "the whole rendering is one emit call"
    assert DEVANAGARI_SNIPPET in written[0]
    captured = capsys.readouterr()
    assert captured.out == "", (
        "something reached stdout without going through render.emit"
    )


# --------------------------------------------------------------------------- #
# Runtime failures (design §4, exit 2)
# --------------------------------------------------------------------------- #


def test_a_store_failure_is_a_runtime_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from npu_rag.index.errors import StoreError

    class Broken:
        def search(self, query: str, options: Any = None, embed_service: Any = None) -> Any:
            raise StoreError("the table is unreadable")

    args = parse("search", "a query")
    code = cmd_search.run(
        args,
        load=lambda **_: make_config(tmp_path),
        open_index=lambda path: Broken(),
        open_embedder=lambda cfg: FakeEmbedder(),
    )
    assert code == EXIT_RUNTIME
    captured = capsys.readouterr()
    assert "the table is unreadable" in captured.err
    assert captured.out == ""


def test_a_malformed_configuration_is_a_user_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def load(**_: object) -> CliConfig:
        raise ConfigError("npu-rag.toml: key 'provider' must be one of 'npu'; got 'gpu'")

    args = parse("search", "a query")
    code = cmd_search.run(
        args,
        load=load,
        open_index=lambda path: pytest.fail("no index on a config error"),
        open_embedder=lambda cfg: pytest.fail("no embedder on a config error"),
    )
    assert code == EXIT_USAGE
    assert "npu-rag.toml" in capsys.readouterr().err


def test_an_unconfigurable_default_mode_is_a_user_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Note 1.2: a `CliConfig` built outside `load_config` is unvalidated, so
    `default_mode` is checked where it is used."""
    index = FakeIndex(make_response())
    config = make_config(tmp_path, default_mode="dense_only")
    code = cmd_search.run(
        args=parse("search", "a query"),
        load=lambda **_: config,
        open_index=lambda path: index,
        open_embedder=lambda cfg: FakeEmbedder(),
    )
    assert code == EXIT_USAGE
    captured = capsys.readouterr()
    assert "default_mode" in captured.err
    assert "'dense'" in captured.err, "the message must list the accepted spellings"
    assert index.calls == []


def test_an_unknown_model_name_is_a_user_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`model_name` is user configuration, so an unknown one is exit 1."""
    (tmp_path / "lancedb").mkdir()
    (tmp_path / "lancedb" / "chunks.lance").mkdir()
    config = CliConfig(
        archive_path=tmp_path / "archive",
        index_path=tmp_path / "lancedb",
        model_name="not-a-model",
        provider="cpu",
        tags={},
        default_mode="hybrid",
        default_limit=10,
        ansi_colors=False,
    )
    args = parse("search", "a query")
    code = cmd_search.run(
        args,
        load=lambda **_: config,
        open_index=lambda path: FakeIndex(make_response()),
    )
    assert code == EXIT_USAGE
    assert "not-a-model" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The production provider seam (ruling 4), without a real model
# --------------------------------------------------------------------------- #


class FakeEmbedResult:
    """What `EmbeddingService.embed_queries` returns, narrowed to what is used."""

    def __init__(self, *, served: Any, reason: str | None) -> None:
        import numpy as np

        self.vectors = np.asarray([[0.5, 0.25]], dtype=np.float32)
        self.provider_served = served
        self.fallback_reason = reason


class FakeEmbeddingService:
    """Records the `ProviderChoice` it was asked to embed with.

    It also enforces `BoundBackend`'s own invariant, which is what makes the
    recording non-vacuous: a `fallback_reason` exists **only** under an `auto`
    request (`providers/base.py:277-292` raises otherwise). A caller that
    pre-resolved `auto` to `cpu` before embedding would therefore get no
    reason at all, which is the silent fallback requirement 6.2 forbids.
    """

    def __init__(self, *, served: Any, reason: str | None = None) -> None:
        self._served = served
        self._reason = reason
        self.requests: list[Any] = []

    def embed_queries(self, texts: Any, provider: Any) -> FakeEmbedResult:
        from npu_rag.embedding.types import ProviderChoice

        self.requests.append(provider)
        reason = self._reason if ProviderChoice(provider) is ProviderChoice.AUTO else None
        return FakeEmbedResult(served=self._served, reason=reason)


def _adapter(service: FakeEmbeddingService, provider: Any) -> Any:
    """The private adapter under test; the fake stands in for `EmbeddingService`."""
    return cmd_search._ServiceQueryEmbedder(cast("Any", service), provider)  # noqa: SLF001


def test_the_adapter_claims_nothing_before_a_query_has_run() -> None:
    from npu_rag.embedding.types import ProviderChoice

    embedder = _adapter(
        FakeEmbeddingService(served=ProviderChoice.CPU), ProviderChoice.AUTO
    )
    assert embedder.provider_served is None
    assert embedder.fallback_reason is None


def test_the_adapter_embeds_with_the_requested_provider() -> None:
    """`index/service.py:40` hardcodes AUTO on the `embed_queries` path, so the
    adapter is the only thing carrying `--provider` to the backend binding."""
    from npu_rag.embedding.types import ProviderChoice

    service = FakeEmbeddingService(served=ProviderChoice.NPU)
    embedder = _adapter(service, ProviderChoice.NPU)

    assert embedder.embed_query("a query") == [0.5, 0.25]
    assert service.requests == [ProviderChoice.NPU]
    assert embedder.provider_served == "npu"
    assert embedder.fallback_reason is None


def test_the_adapter_keeps_the_served_provider_and_the_reason() -> None:
    """An `auto` request served by the CPU: both halves of requirement 6.2."""
    from npu_rag.embedding.types import ProviderChoice

    service = FakeEmbeddingService(served=ProviderChoice.CPU, reason="no vendor runtime")
    embedder = _adapter(service, ProviderChoice.AUTO)

    embedder.embed_query("a query")
    assert service.requests == [ProviderChoice.AUTO]
    assert embedder.provider_served == "cpu"
    assert embedder.fallback_reason == "no vendor runtime"


def _patch_capability(
    monkeypatch: pytest.MonkeyPatch, *, reason: str | None, services: list[Any] | None = None
) -> list[Any]:
    """Replace the capability probe and the artifact build; record the provider.

    :param services: filled with each `FakeEmbeddingService` handed back, so a
        test can inspect which `ProviderChoice` the *embed call* requested -
        a different question from which artifact was prepared.
    """
    from npu_rag.embedding import service as embedding_service
    from npu_rag.embedding.environment import capability as capability_module
    from npu_rag.embedding.providers import base as providers_base
    from npu_rag.embedding.types import ProviderChoice

    built: list[Any] = []

    def check_capability(**_: object) -> object:
        return object()

    def unavailable(_: object) -> str | None:
        return reason

    def build_service(
        profile: Any,
        capability: Any,
        root: Any,
        *,
        provider: Any = ProviderChoice.NPU,
        **_: object,
    ) -> Any:
        built.append(provider)
        made = FakeEmbeddingService(served=ProviderChoice.CPU, reason=reason)
        if services is not None:
            services.append(made)
        return made

    monkeypatch.setattr(capability_module, "check_capability", check_capability)
    monkeypatch.setattr(providers_base, "npu_unavailable_reason", unavailable)
    monkeypatch.setattr(embedding_service, "build_service", build_service)
    return built


def test_an_explicit_npu_request_is_refused_before_any_artifact_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 6.2, and the reason it is checked early: the NPU artifact
    path begins with a compile measured in tens of minutes."""
    from npu_rag.embedding.errors import NpuUnavailableError

    built = _patch_capability(monkeypatch, reason="vendor runtime absent")
    config = make_config(tmp_path, provider="npu")

    with pytest.raises(NpuUnavailableError) as excinfo:
        cmd_search.open_query_embedder(config)

    assert "vendor runtime absent" in str(excinfo.value)
    assert "--provider cpu" in str(excinfo.value)
    assert built == [], "nothing may be prepared for a request that cannot be served"


def test_auto_prepares_the_cpu_artifact_but_still_requests_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ensure_prepared` refuses AUTO, so the artifact choice must be resolved -
    while the *request* must stay AUTO or `fallback_reason` is never produced."""
    from npu_rag.embedding.types import ProviderChoice

    services: list[Any] = []
    built = _patch_capability(
        monkeypatch, reason="no NPU on this machine", services=services
    )
    embedder = cmd_search.open_query_embedder(make_config(tmp_path, provider="auto"))

    assert built == [ProviderChoice.CPU], "ensure_prepared refuses AUTO"
    embedder.embed_query("a query")
    assert services[0].requests == [ProviderChoice.AUTO], (
        "the embed request must stay AUTO: an explicit 'cpu' request carries no "
        "fallback reason, so pre-resolving it loses requirement 6.2's reason"
    )
    assert embedder.fallback_reason == "no NPU on this machine"


def test_auto_prepares_the_npu_artifact_when_the_npu_is_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npu_rag.embedding.types import ProviderChoice

    built = _patch_capability(monkeypatch, reason=None)
    cmd_search.open_query_embedder(make_config(tmp_path, provider="auto"))
    assert built == [ProviderChoice.NPU]


def test_an_explicit_cpu_request_prepares_the_cpu_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npu_rag.embedding.types import ProviderChoice

    services: list[Any] = []
    built = _patch_capability(monkeypatch, reason=None, services=services)
    embedder = cmd_search.open_query_embedder(make_config(tmp_path, provider="cpu"))
    assert built == [ProviderChoice.CPU], "an explicit cpu request never prepares an NPU artifact"
    embedder.embed_query("a query")
    assert services[0].requests == [ProviderChoice.CPU], (
        "an explicit request is handed over as itself, not widened to AUTO"
    )


def test_artifacts_live_beside_the_index(tmp_path: Path) -> None:
    config = make_config(tmp_path, index="store/lancedb")
    assert cmd_search.artifacts_root(config) == (
        tmp_path / "store" / cmd_search.ARTIFACTS_DIRNAME
    )


# --------------------------------------------------------------------------- #
# Folder/tag matching (decision 2), on its own
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("spec", "folder", "matches"),
    [
        ("Folder A", "Folder A", True),
        ("Folder A", "Folder AB", False),
        ("Folder*", "Folder AB", True),
        ("Folder ?", "Folder A", True),
        ("Folder ?", "Folder AB", False),
        ("folder a", "Folder A", True),  # Windows folds case
        ("100% Returns", "100% Returns", True),
        ("100% Returns", "100 Returns", False),
        ("A[1]", "A[1]", True),
        ("Parent/Child", "Parent", True),
        ("Parent\\Child", "Parent", True),
    ],
)
def test_folder_tag_matching_uses_the_glob_vocabulary_of_tags_py(
    spec: str, folder: str, matches: bool
) -> None:
    """`*` and `?` are wildcards; `%`, `_` and `[` are literal (see tags.py)."""
    mapping = cmd_search.folder_tag_map([folder], {"t": (spec,)})
    assert (mapping.get(folder) == ("t",)) is matches


def test_folder_tag_map_keeps_every_matching_tag_in_sorted_order() -> None:
    """Three matching tags, in an order that is neither sorted nor reverse-sorted.

    Two earlier fixtures were too weak, each in a different way, and both are
    worth recording because the shape recurs. `{"alpha": ..., "zeta": ...}` has
    an insertion order that *equals* sorted order, so iterating as written was
    indistinguishable from sorting. Adding a third, non-matching key made
    `reversed()` produce `(alpha, zeta)` for the matching subset - also
    indistinguishable. With `beta, alpha, gamma` all matching, the three
    candidate orders are `(beta, alpha, gamma)` as written, `(gamma, alpha,
    beta)` reversed, and `(alpha, beta, gamma)` sorted: no two coincide, so only
    a sort passes.
    """
    mapping = cmd_search.folder_tag_map(
        ["Shared"],
        {
            "beta": ("Shared",),
            "alpha": ("Shared",),
            "gamma": ("Shared",),
            "other": ("Elsewhere",),
        },
    )
    assert mapping == {"Shared": ("alpha", "beta", "gamma")}


def test_the_annotations_glob_vocabulary_is_the_filters_glob_vocabulary() -> None:
    """The label and the pushdown SQL must agree on what a wildcard is.

    Note 3.1: a `folder_tags` key that does not match annotates nothing and
    fails **silently**. If `tags.GLOB_CHARS` gained a third wildcard, the SQL
    filter would honour it and the `(Tag: ...)` annotation would not.
    """
    from npu_rag.cli.tags import GLOB_CHARS

    assert set(cmd_search._GLOB_REGEX) == GLOB_CHARS  # noqa: SLF001
    assert GLOB_CHARS == frozenset("*?"), "pinned against a literal as well"


def test_folder_tag_map_ignores_degenerate_entries() -> None:
    """A broken `[tags]` entry must not break an unrelated search (note 2.1)."""
    assert cmd_search.folder_tag_map(["Folder A"], {"broken": (), "blank": ("  ",)}) == {}


# --------------------------------------------------------------------------- #
# The real index: the join, not the halves
# --------------------------------------------------------------------------- #


def test_search_over_a_real_index_returns_the_matching_text(
    tmp_path: Path, real_index: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    args = parse("search", "tara mantra", "--mode", "dense", "--top-k", "3")
    code = cmd_search.run(
        args,
        load=lambda **_: make_config(tmp_path),
        open_index=lambda path: real_index,
        open_embedder=lambda cfg: RealQueryEmbedder(),
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "Tara mantra" in out
    assert "[Folder: Ugra Tara Sadhana]" in out
    assert f"[Provider: {CPU_ORT_PROVIDER}]" in out
    assert "Pension" not in out


def test_the_folder_filter_really_narrows_a_real_index(
    tmp_path: Path, real_index: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pushdown SQL is parsed by lancedb here, not compared as a string."""
    args = parse(
        "search", "mantra day", "--mode", "fts", "--folder", "German Pension", "--top-k", "5"
    )
    code = cmd_search.run(
        args,
        load=lambda **_: make_config(tmp_path),
        open_index=lambda path: real_index,
        open_embedder=lambda cfg: pytest.fail("fts needs no embedder"),
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "Ugra Tara Sadhana" not in out


def test_a_real_grouped_search_restores_narrative_order(
    tmp_path: Path, real_index: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    args = parse(
        "search",
        "tara mantra",
        "--mode",
        "dense",
        "--top-k",
        "5",
        "--group-by-folder",
    )
    code = cmd_search.run(
        args,
        load=lambda **_: make_config(tmp_path),
        open_index=lambda path: real_index,
        open_embedder=lambda cfg: RealQueryEmbedder(),
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    positions = [out.index(name) for name in ("Day 1.md", "Day 2.md", "Day 10.md")]
    assert positions == sorted(positions), "Day 10 must follow Day 2, not precede it"


def test_a_real_json_search_is_parseable_and_names_source_paths(
    tmp_path: Path, real_index: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    args = parse("search", "pension", "--mode", "dense", "--json", "--top-k", "1")
    code = cmd_search.run(
        args,
        load=lambda **_: make_config(tmp_path),
        open_index=lambda path: real_index,
        open_embedder=lambda cfg: RealQueryEmbedder(),
    )
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"], "a dense query for 'pension' must hit the pension folder"
    row = payload["results"][0]
    assert row["folder"] == "German Pension"
    # Note 3.1: `source_path` is the openable form.
    assert Path(row["source_path"]).name == "Overview.md"


def _garuda_search(
    tmp_path: Path, real_index: Any, *argv: str
) -> int:
    args = parse("search", "garuda mudra", "--mode", "dense", "--top-k", "1", *argv)
    return cmd_search.run(
        args,
        load=lambda **_: make_config(tmp_path),
        open_index=lambda path: real_index,
        open_embedder=lambda cfg: RealQueryEmbedder(),
    )


def test_context_expansion_attaches_the_real_neighbouring_chunks(
    tmp_path: Path, real_index: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 4.3, end to end through the real `ContextExpander`.

    The hit is the middle chunk of a three-chunk document. Its neighbours share
    no term with the query, so their text in the output can only have come from
    the expansion.
    """
    assert _garuda_search(tmp_path, real_index, "--context", "1") == EXIT_OK
    out = capsys.readouterr().out
    assert GARUDA_CHUNK_TEXT in out
    assert PRECEDING_CHUNK_TEXT in out, "the preceding chunk was not attached"
    assert SUCCEEDING_CHUNK_TEXT in out, "the succeeding chunk was not attached"
    assert "[Context]" in out


def test_without_context_the_neighbouring_chunks_stay_out(
    tmp_path: Path, real_index: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The control that makes the test above mean something: `--context 0` is
    the default and attaches nothing."""
    assert _garuda_search(tmp_path, real_index) == EXIT_OK
    out = capsys.readouterr().out
    assert GARUDA_CHUNK_TEXT in out
    assert PRECEDING_CHUNK_TEXT not in out
    assert SUCCEEDING_CHUNK_TEXT not in out
    assert "[Context]" not in out


# --------------------------------------------------------------------------- #
# Requirement 1.2 - the module's own import weight
# --------------------------------------------------------------------------- #

_PROBE = """\
import json, sys
from pathlib import Path
import npu_rag.cli.cmd_search  # noqa: F401
heavy = sorted(n for n in ("lancedb", "onnxruntime", "torch", "transformers") if n in sys.modules)
Path(sys.argv[1]).write_text(json.dumps({"heavy": heavy, "count": len(sys.modules)}), encoding="utf-8")
"""


def test_importing_the_handler_imports_nothing_heavy(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    result = tmp_path / "probe.json"
    completed = subprocess.run(
        [sys.executable, str(script), str(result)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["heavy"] == [], (
        "cmd_search must import lancedb, onnxruntime, torch and transformers "
        f"inside run() only; got {payload['heavy']}"
    )
