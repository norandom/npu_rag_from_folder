"""Tests for the embedding-model identity stamp and its compatibility policy.

The failure under test is not "the configured model changed" - that is merely a
configuration edit. It is **one table holding vectors from two models**, which
produces no error, no distinguishable rows, and similarity scores that are
meaningless but entirely plausible. Every test here is written so that a
comparison which is ``True`` by construction cannot pass it: wherever two
identities appear they are genuinely different strings, and the expected
rendering of a stamp is pinned against a hand-written literal rather than
against another call to the function that produces it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from npu_rag.index.compat import UNSTAMPED_LABEL, verify_model_identity
from npu_rag.index.errors import ConfigurationError, SchemaError
from npu_rag.index.schema import (
    MODEL_IDENTITY_SEPARATOR,
    UNSTAMPED_MODEL_IDENTITY,
    VECTOR_DIMENSION,
    ChunkSchema,
    chunk_record_to_schema,
    format_model_identity,
    validate_model_identity,
)
from npu_rag.index.service import VectorIndex
from npu_rag.index.store import LanceIndexStore
from npu_rag.index.types import IndexStats
from npu_rag.ingest.types import ChunkKind, ChunkRecord, MarkdownLocator

# Two REAL, different identities. Hand-written literals, not derived from
# `format_model_identity`, so a mutant that changes how a stamp is spelled is
# visible here and a mutant that makes the comparison vacuous cannot hide: the
# two strings differ in the model part while agreeing on the width, which is
# precisely the same-dimension-different-model case that fails nowhere else.
GEMMA = "google/embeddinggemma-300m:768"
GEMMA_2 = "google/embeddinggemma-2-300m:768"
# A third, differing only in the width, for the Matryoshka case.
GEMMA_NARROW = "google/embeddinggemma-300m:256"


def _record(chunk_id: str, text: str = "chunk text") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source_path=Path("docs/doc.md"),
        root_id="root",
        author="Author",
        title="Doc",
        kind=ChunkKind.PROSE,
        text=text,
        locator=MarkdownLocator(line_range=(1, 5), ordinal=0),
        ordinal=0,
        token_count=5,
        truncated=False,
        provenance=None,
    )


def _vector() -> list[float]:
    return [0.01 * (i % 50) for i in range(VECTOR_DIMENSION)]


def _row(chunk_id: str, model_identity: str) -> ChunkSchema:
    return chunk_record_to_schema(
        _record(chunk_id, text=f"text for {chunk_id}"),
        _vector(),
        model_identity=model_identity,
    )


def _store(tmp_path: Path, name: str = "ldb") -> LanceIndexStore:
    return LanceIndexStore(tmp_path / name)


def _legacy_table(db_path: Path, rows: list[dict[str, str]]) -> Any:
    """A ``chunks`` table with NO ``model_identity`` column.

    Stands in for an index written before task 8.2. The schema is inferred from
    the rows, so the column is genuinely absent rather than present-and-empty -
    which is the distinction that made ``stats()`` raise.
    """
    import lancedb

    db = lancedb.connect(str(db_path))
    return db.create_table("chunks", data=rows, mode="create")


# --------------------------------------------------------------------------
# What the identity string contains
# --------------------------------------------------------------------------


def test_format_model_identity_renders_the_pinned_literal() -> None:
    """Pinned against a literal written by hand, not against the formatter."""
    assert format_model_identity("google/embeddinggemma-300m", 768) == GEMMA
    assert format_model_identity("google/embeddinggemma-2-300m", 768) == GEMMA_2
    assert format_model_identity("google/embeddinggemma-300m", 256) == GEMMA_NARROW


def test_identity_defaults_to_the_stored_vector_width() -> None:
    assert format_model_identity("google/embeddinggemma-300m") == GEMMA
    assert str(VECTOR_DIMENSION) in GEMMA


def test_identity_separates_width_so_two_widths_are_two_identities() -> None:
    """One repository at two Matryoshka widths is two vector spaces."""
    wide = format_model_identity("google/embeddinggemma-300m", 768)
    narrow = format_model_identity("google/embeddinggemma-300m", 256)
    assert wide != narrow
    assert MODEL_IDENTITY_SEPARATOR in wide


def test_identity_rejects_values_that_would_stamp_nothing() -> None:
    with pytest.raises(SchemaError, match="non-empty model id"):
        format_model_identity("")
    with pytest.raises(SchemaError, match="non-empty model id"):
        format_model_identity("   ")
    with pytest.raises(SchemaError, match="must not contain"):
        format_model_identity("google/embeddinggemma-300m:768")
    with pytest.raises(SchemaError, match="positive vector dimension"):
        format_model_identity("google/embeddinggemma-300m", 0)


@pytest.mark.parametrize(
    "bad",
    [
        "   ",
        "google/embeddinggemma-300m",
        "google/embeddinggemma-300m:",
        "google/embeddinggemma-300m:wide",
        ":768",
    ],
)
def test_validate_model_identity_rejects_malformed_stamps(bad: str) -> None:
    """A stamp that names no model is worse than none: it reads as evidence."""
    with pytest.raises(SchemaError):
        validate_model_identity(bad)


def test_blank_identity_is_rejected_with_its_own_diagnostic() -> None:
    """The blank guard is reachable only through its message.

    An all-whitespace string is *also* malformed - it carries no separator -
    so the generic branch would raise anyway and a test asserting only
    ``SchemaError`` cannot see whether the blank guard exists. The message is
    the behaviour: it names ``UNSTAMPED_MODEL_IDENTITY`` as the way to say
    "unknown", which the malformed message does not.
    """
    for blank in ("   ", "\t", "\n", " "):
        with pytest.raises(SchemaError, match="blank model identity"):
            validate_model_identity(blank)

    # Whitespace *around* a separator is a malformed stamp, not a blank one:
    # something was written there, it just is not a model id.
    with pytest.raises(SchemaError, match="Malformed model identity"):
        validate_model_identity("  :768")


def test_validate_model_identity_accepts_the_unstamped_sentinel() -> None:
    assert validate_model_identity(UNSTAMPED_MODEL_IDENTITY) == ""
    assert validate_model_identity(GEMMA) == GEMMA


# --------------------------------------------------------------------------
# The column: per-row, so a mixed table is detectable
# --------------------------------------------------------------------------


def test_chunk_schema_built_by_hand_is_unstamped_not_plausible() -> None:
    """The column's own default, which the converter never exercises.

    Rows are also built directly from ``ChunkSchema`` outside this package
    (``tests/cli/test_cli_tags.py`` does), and such a row never passes through
    ``validate_model_identity``. If the class default were a real-looking
    string the compatibility check would treat it as a recorded model and
    report a mismatch naming a model that does not exist.
    """
    schema = ChunkSchema(
        chunk_id="hand_built",
        file_path="a.md",
        file_hash="h",
        chunk_index=0,
        chunk_kind="prose",
        title=None,
        headings=[],
        locator_type="markdown",
        locator_json="{}",
        text="text",
        vector=_vector(),
        token_count=1,
        created_at=1.0,
    )
    assert schema.model_identity == UNSTAMPED_MODEL_IDENTITY
    assert schema.model_identity == ""


def test_chunk_record_to_schema_stamps_the_supplied_identity() -> None:
    schema = chunk_record_to_schema(_record("c1"), _vector(), model_identity=GEMMA)
    assert schema.model_identity == GEMMA


def test_chunk_record_to_schema_is_unstamped_without_an_identity() -> None:
    schema = chunk_record_to_schema(_record("c1"), _vector())
    assert schema.model_identity == UNSTAMPED_MODEL_IDENTITY


def test_chunk_record_to_schema_refuses_a_malformed_identity() -> None:
    with pytest.raises(SchemaError, match="Malformed model identity"):
        chunk_record_to_schema(
            _record("c1"), _vector(), model_identity="google/embeddinggemma-300m"
        )


def test_identity_is_recorded_per_row_and_survives_a_reopen(tmp_path: Path) -> None:
    """A per-row column, not a sidecar: the table itself says which rows."""
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", GEMMA), _row("c", GEMMA_2)])

    reopened = _store(tmp_path)
    rows = reopened.table.search().select(["chunk_id", "model_identity"]).limit(10).to_list()
    by_id = {row["chunk_id"]: row["model_identity"] for row in rows}
    assert by_id == {"a": GEMMA, "b": GEMMA, "c": GEMMA_2}


def test_model_identity_counts_counts_each_identity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", GEMMA_2), _row("c", GEMMA_2)])
    assert store.model_identity_counts() == {GEMMA: 1, GEMMA_2: 2}


def test_model_identity_counts_keeps_unstamped_rows_visible(tmp_path: Path) -> None:
    """Dropping the unstamped bucket would hide the partially stamped case."""
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", UNSTAMPED_MODEL_IDENTITY)])
    assert store.model_identity_counts() == {GEMMA: 1, UNSTAMPED_MODEL_IDENTITY: 1}


def test_model_identity_counts_is_empty_for_an_empty_index(tmp_path: Path) -> None:
    assert _store(tmp_path).model_identity_counts() == {}


def test_model_identity_counts_reads_every_row_not_a_page(tmp_path: Path) -> None:
    """Any truncation of the scan can hide the second model entirely.

    The scan is **unlimited by design**, not limited to a measured row count:
    a plain non-vector query in lancedb 0.38.0 already returns every row, so a
    limit would add nothing, and computing one from ``len(table)`` first would
    open a check-then-read window in which a table that grew between the two
    calls is scanned short - letting the gate pass a mixed index because the
    second model's rows fell outside the page. So this test does not probe a
    default limit (there is none on this path); it probes that whatever the
    scan does, it sees every row. 42 rows with a single odd one out is enough
    for any page-sized truncation to be caught.
    """
    store = _store(tmp_path)
    store.add([_row(f"c{i}", GEMMA) for i in range(41)] + [_row("odd", GEMMA_2)])
    counts = store.model_identity_counts()
    assert counts == {GEMMA: 41, GEMMA_2: 1}
    assert sum(counts.values()) == store.count() == 42


def test_identity_counts_of_a_column_less_table_report_unstamped(
    tmp_path: Path,
) -> None:
    """A table predating the column must not look like corruption.

    Before this case was handled, ``model_identity_counts`` raised a raw Arrow
    "No field named model_identity" through ``StoreError`` - so both ``sync``
    and ``status`` showed the operator something that reads like a damaged
    index, and the one message that tells them what to do (rebuild) was never
    reached. Reporting the rows as wholly unstamped routes them into the
    unstamped branch, whose text is written for exactly this index.
    """
    table = _legacy_table(
        tmp_path / "legacy",
        [
            {"chunk_id": "a", "file_path": "a.md", "text": "one"},
            {"chunk_id": "b", "file_path": "b.md", "text": "two"},
        ],
    )
    assert "model_identity" not in table.schema.names

    store = LanceIndexStore(tmp_path / "legacy")
    assert store.model_identity_counts() == {UNSTAMPED_MODEL_IDENTITY: 2}

    stats = store.stats()
    assert stats.model_identities == ()
    assert stats.unstamped_chunks == 2
    assert stats.is_mixed_model is False

    # And the policy says what to do about it, rather than reporting a fault.
    with pytest.raises(ConfigurationError) as excinfo:
        verify_model_identity(store, GEMMA)
    message = str(excinfo.value)
    assert "no recorded" in message
    assert "rebuild" in message.lower()
    assert "model_identity" not in message


def test_empty_column_less_table_is_accepted(tmp_path: Path) -> None:
    """Zero rows is an empty index, not N unstamped ones."""
    table = _legacy_table(
        tmp_path / "legacy_empty",
        [{"chunk_id": "a", "file_path": "a.md", "text": "one"}],
    )
    table.delete("chunk_id = 'a'")
    assert "model_identity" not in table.schema.names
    assert len(table) == 0

    store = LanceIndexStore(tmp_path / "legacy_empty")
    assert store.model_identity_counts() == {}
    verify_model_identity(store, GEMMA)


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------


def test_empty_index_accepts_any_identity(tmp_path: Path) -> None:
    """The first sync has to be able to happen."""
    store = _store(tmp_path)
    verify_model_identity(store, GEMMA)
    verify_model_identity(store, UNSTAMPED_MODEL_IDENTITY)


def test_matching_identity_is_accepted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", GEMMA)])
    verify_model_identity(store, GEMMA)


def test_same_width_different_model_is_rejected(tmp_path: Path) -> None:
    """The whole point: both are 768-d, so nothing else catches this."""
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", GEMMA)])
    with pytest.raises(ConfigurationError) as excinfo:
        verify_model_identity(store, GEMMA_2)
    message = str(excinfo.value)
    # Actionable, and each fact has to be *attributed* rather than merely
    # mentioned: a message naming both strings without saying which is which is
    # not usable. Both identities appear twice in the text, so a bare presence
    # check survives dropping either attribution - verified by planting it.
    assert f"was built with {GEMMA}" in message
    assert f"the configured model is {GEMMA_2}" in message
    assert "2 chunks" in message
    assert "rebuild" in message.lower()
    assert str(tmp_path) in message


def test_same_model_different_width_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_row("a", GEMMA)])
    with pytest.raises(ConfigurationError, match="mismatch"):
        verify_model_identity(store, GEMMA_NARROW)


def test_mismatch_message_counts_one_chunk_in_the_singular(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_row("a", GEMMA)])
    with pytest.raises(ConfigurationError) as excinfo:
        verify_model_identity(store, GEMMA_2)
    assert "1 chunk)" in str(excinfo.value)


@pytest.mark.parametrize(
    "configured",
    [
        # Differs only in case. Hugging Face repository ids are case-sensitive,
        # so these are two different repositories and two vector spaces; a
        # case-insensitive comparison would silently accept the swap, which is
        # the exact class of silent acceptance this task exists to prevent.
        "Google/EmbeddingGemma-300m:768",
        "GOOGLE/EMBEDDINGGEMMA-300M:768",
        "google/embeddinggemma-300M:768",
    ],
)
def test_identity_comparison_is_case_sensitive(
    tmp_path: Path, configured: str
) -> None:
    store = _store(tmp_path)
    store.add([_row("a", GEMMA)])
    assert configured.lower() == GEMMA.lower()
    assert configured != GEMMA
    with pytest.raises(ConfigurationError, match="mismatch"):
        verify_model_identity(store, configured)


def test_mixed_index_is_rejected_for_every_identity(tmp_path: Path) -> None:
    """Including the identity of one of the two models it already holds."""
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", GEMMA_2)])

    for configured in (GEMMA, GEMMA_2, GEMMA_NARROW, UNSTAMPED_MODEL_IDENTITY):
        with pytest.raises(ConfigurationError) as excinfo:
            verify_model_identity(store, configured)
        message = str(excinfo.value)
        assert "more than one vector space" in message
        # The inventory names both, so the operator can see what to delete.
        assert GEMMA in message
        assert GEMMA_2 in message


def test_mixed_inventory_is_rendered_in_a_stable_sorted_order(
    tmp_path: Path,
) -> None:
    """Insertion order is deliberately NOT sorted order.

    ``GEMMA_2`` sorts before ``GEMMA`` ('2' < '3' at the differing position)
    while being inserted after it, so preserving insertion order, reversing it,
    or dropping the ``sorted()`` are each distinguishable from sorting. The
    configured identity is a third value, so the first occurrence of each of
    the two stored identities is necessarily inside the inventory.
    """
    assert sorted((GEMMA, GEMMA_2)) == [GEMMA_2, GEMMA]

    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", GEMMA_2)])

    with pytest.raises(ConfigurationError) as excinfo:
        verify_model_identity(store, GEMMA_NARROW)
    message = str(excinfo.value)
    assert message.index(GEMMA_2) < message.index(GEMMA)


def test_partially_stamped_inventory_puts_the_absent_identity_first(
    tmp_path: Path,
) -> None:
    """``""`` sorts before every real identity, and is inserted after one."""
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", UNSTAMPED_MODEL_IDENTITY)])

    with pytest.raises(ConfigurationError) as excinfo:
        verify_model_identity(store, GEMMA_NARROW)
    message = str(excinfo.value)
    assert message.index(UNSTAMPED_LABEL) < message.index(GEMMA)


def test_partially_stamped_index_is_reported_as_mixed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", UNSTAMPED_MODEL_IDENTITY)])
    with pytest.raises(ConfigurationError) as excinfo:
        verify_model_identity(store, GEMMA)
    message = str(excinfo.value)
    assert "more than one vector space" in message
    # The absent identity is rendered as words, never as an empty gap.
    assert UNSTAMPED_LABEL in message


def test_fully_unstamped_index_is_rejected_for_a_stamped_caller(
    tmp_path: Path,
) -> None:
    """Unstamped fails rather than passing as 'unknown'.

    A pass would be an unverifiable claim about rows whose vector space nobody
    recorded. No index exists in the field yet, so every unstamped index that
    will ever be met is a caller that forgot to stamp.
    """
    store = _store(tmp_path)
    store.add([_row("a", UNSTAMPED_MODEL_IDENTITY)])
    with pytest.raises(ConfigurationError) as excinfo:
        verify_model_identity(store, GEMMA)
    message = str(excinfo.value)
    assert "no recorded" in message
    assert GEMMA in message
    assert "rebuild" in message.lower()


def test_fully_unstamped_index_accepts_an_unstamped_caller(tmp_path: Path) -> None:
    """Both sides state 'unknown', so nothing becomes mixed by proceeding."""
    store = _store(tmp_path)
    store.add([_row("a", UNSTAMPED_MODEL_IDENTITY)])
    verify_model_identity(store, UNSTAMPED_MODEL_IDENTITY)


def test_stamped_index_rejects_a_caller_with_no_identity(tmp_path: Path) -> None:
    """Otherwise the default argument on `sync` would create the mixed case."""
    store = _store(tmp_path)
    store.add([_row("a", GEMMA)])
    with pytest.raises(ConfigurationError) as excinfo:
        verify_model_identity(store, UNSTAMPED_MODEL_IDENTITY)
    message = str(excinfo.value)
    assert "supplied no embedding-model identity" in message
    assert GEMMA in message


def test_configuration_error_is_not_a_store_or_query_error() -> None:
    """Pinned so the taxonomy choice cannot drift into an I/O-looking error."""
    from npu_rag.index.errors import QueryError, StoreError, SyncError

    assert not issubclass(ConfigurationError, (StoreError, QueryError, SyncError))


# --------------------------------------------------------------------------
# IndexStats
# --------------------------------------------------------------------------


def test_stats_reports_the_identity_the_index_was_built_with(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", GEMMA)])
    stats = store.stats()
    assert stats.model_identities == (GEMMA,)
    assert stats.unstamped_chunks == 0
    assert stats.is_mixed_model is False


def test_stats_reports_both_identities_of_a_mixed_index(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_row("a", GEMMA), _row("b", GEMMA_2), _row("c", GEMMA_2)])
    stats = store.stats()
    # Sorted, so the tuple is stable for a reporter. 'google/embeddinggemma-2'
    # sorts before 'google/embeddinggemma-3' because '2' < '3'.
    assert stats.model_identities == (GEMMA_2, GEMMA)
    assert stats.unstamped_chunks == 0
    assert stats.is_mixed_model is True


def test_stats_counts_unstamped_rows_separately(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(
        [
            _row("a", GEMMA),
            _row("b", UNSTAMPED_MODEL_IDENTITY),
            _row("c", UNSTAMPED_MODEL_IDENTITY),
        ]
    )
    stats = store.stats()
    assert stats.model_identities == (GEMMA,)
    assert stats.unstamped_chunks == 2
    assert stats.is_mixed_model is True


def test_stats_of_an_entirely_unstamped_index_is_unknown_not_mixed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.add([_row("a", UNSTAMPED_MODEL_IDENTITY), _row("b", UNSTAMPED_MODEL_IDENTITY)])
    stats = store.stats()
    assert stats.model_identities == ()
    assert stats.unstamped_chunks == 2
    assert stats.is_mixed_model is False


def test_stats_of_an_empty_index_reports_no_identity(tmp_path: Path) -> None:
    stats = _store(tmp_path).stats()
    assert stats.total_chunks == 0
    assert stats.model_identities == ()
    assert stats.unstamped_chunks == 0
    assert stats.is_mixed_model is False


def test_stats_tracks_the_table_rather_than_caching(tmp_path: Path) -> None:
    """A stale value is the failure mode a `status` report cannot notice."""
    store = _store(tmp_path)
    store.add([_row("a", GEMMA)])
    assert store.stats().model_identities == (GEMMA,)

    store.delete_by_chunk_ids(["a"])
    store.add([_row("b", GEMMA_2)])
    after = store.stats()
    assert after.model_identities == (GEMMA_2,)
    assert after.total_chunks == 1


def test_index_stats_new_fields_are_backward_compatible_defaults() -> None:
    """The five original positional fields still construct an IndexStats."""
    stats = IndexStats(
        total_chunks=1000,
        total_documents=120,
        vector_dimension=768,
        has_fts_index=True,
        index_size_bytes=4096000,
    )
    assert stats.model_identities == ()
    assert stats.unstamped_chunks == 0
    assert stats.is_mixed_model is False


def test_is_mixed_model_requires_a_recorded_identity_alongside_unstamped() -> None:
    """Hand-built, so each branch of the property is probed independently."""
    def stats(identities: tuple[str, ...], unstamped: int) -> IndexStats:
        return IndexStats(
            total_chunks=3,
            total_documents=1,
            vector_dimension=768,
            has_fts_index=False,
            index_size_bytes=0,
            model_identities=identities,
            unstamped_chunks=unstamped,
        )

    assert stats((), 3).is_mixed_model is False
    assert stats((GEMMA,), 0).is_mixed_model is False
    assert stats((GEMMA,), 1).is_mixed_model is True
    assert stats((GEMMA, GEMMA_2), 0).is_mixed_model is True
    assert stats((GEMMA, GEMMA_2), 2).is_mixed_model is True


# --------------------------------------------------------------------------
# The facade seam the CLI must use
# --------------------------------------------------------------------------


def test_vector_index_exposes_the_same_policy(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_row("a", GEMMA)])
    index = VectorIndex(store)

    index.verify_model_identity(GEMMA)
    with pytest.raises(ConfigurationError, match="mismatch"):
        index.verify_model_identity(GEMMA_2)


def test_vector_index_sync_forwards_the_identity_to_the_stamp(tmp_path: Path) -> None:
    """A facade that dropped the argument would build an unstamped index."""
    import numpy as np

    from npu_rag.embedding.types import DocumentText, ProviderChoice
    from npu_rag.ingest.types import RunReport

    class _Embedder:
        def embed_documents(
            self,
            texts: list[DocumentText],
            provider: ProviderChoice = ProviderChoice.AUTO,
            progress: object = None,
        ) -> object:
            vectors = np.zeros((len(texts), VECTOR_DIMENSION), dtype=np.float32)
            return type("_Result", (), {"vectors": vectors})()

    report = RunReport(
        files_processed=1,
        files_unchanged=0,
        files_skipped=0,
        files_failed=0,
        omissions=(),
        removed_chunk_ids=(),
        no_work_required=False,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=(_record("c1"),),
    )
    store = _store(tmp_path)
    index = VectorIndex(store, embed_service=_Embedder())
    index.sync(report, model_identity=GEMMA)

    assert store.model_identity_counts() == {GEMMA: 1}
    assert index.stats().model_identities == (GEMMA,)


def test_public_package_exports_the_identity_surface() -> None:
    import npu_rag.index as pkg

    assert pkg.format_model_identity("google/embeddinggemma-300m", 768) == GEMMA
    assert pkg.UNSTAMPED_MODEL_IDENTITY == ""
    assert pkg.verify_model_identity is verify_model_identity
    for name in (
        "UNSTAMPED_MODEL_IDENTITY",
        "format_model_identity",
        "verify_model_identity",
    ):
        assert name in pkg.__all__
