"""Unit tests for file state and the chunk registry (task 2.2).

Requirements 8.1, 8.3, 8.4: classify a file as new, changed or unchanged
from persisted content and parameter fingerprints before any extraction;
reuse an unchanged file's records; report files absent since the previous
run, with their chunk ids, by comparing run-id stamps.

``state.py`` sits to the right of identity and to the left of vision.
These tests never import vision, and they open a temporary SQLite path
rather than the default ``.npu_rag/ingest.sqlite``.
"""

from __future__ import annotations

import ast
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from npu_rag.ingest.errors import StateError
from npu_rag.ingest.identity import ParamsFingerprint, content_hash
from npu_rag.ingest.state import FileStatus, StateStore
from npu_rag.ingest.types import (
    ChunkKind,
    ChunkRecord,
    MarkdownLocator,
    PageLocator,
    Provenance,
    SourceFile,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "ingest" / "state.py"
)

PARAMS = ParamsFingerprint(digest="a" * 64)
OTHER_PARAMS = ParamsFingerprint(digest="b" * 64)
LOCATOR = MarkdownLocator(line_range=(1, 4), ordinal=0)
PAGE_LOCATOR = PageLocator(page=2)
PROVENANCE = Provenance(vision_model="test-vision", prompt_version="1")


def _imported_names(source_text: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source_text)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            assert node.level == 0, "a relative import still reaches a rightward module"
    return names


def make_source(root: Path, relative: str, *, author: str | None = None) -> SourceFile:
    relative_path = Path(relative)
    return SourceFile(
        path=root / relative_path,
        root=root,
        relative_path=relative_path,
        author=author if author is not None else relative_path.parts[0],
    )


def write_source(root: Path, relative: str, data: bytes) -> SourceFile:
    source = make_source(root, relative)
    source.path.parent.mkdir(parents=True, exist_ok=True)
    source.path.write_bytes(data)
    return source


def make_record(
    source: SourceFile,
    *,
    chunk_id: str,
    text: str = "hello",
    kind: ChunkKind = ChunkKind.PROSE,
    ordinal: int = 0,
    locator: MarkdownLocator | PageLocator = LOCATOR,
    provenance: Provenance | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source_path=source.path,
        root_id=source.root.as_posix(),
        author=source.author,
        title="Notes",
        kind=kind,
        text=text,
        locator=locator,
        ordinal=ordinal,
        token_count=8,
        truncated=False,
        provenance=provenance,
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "ingest.sqlite"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "archive"


@pytest.fixture
def store(db_path: Path) -> Iterator[StateStore]:
    opened = StateStore(db_path)
    try:
        yield opened
    finally:
        opened.close()


# --------------------------------------------------------------------------
# FileStatus
# --------------------------------------------------------------------------


def test_file_status_is_new_changed_or_unchanged_not_deleted() -> None:
    """Requirement 8.1: classify as new, changed or unchanged. Deleted is a
    report from ``deleted_since``, not a classify result."""
    assert {member.value for member in FileStatus} == {"new", "changed", "unchanged"}
    assert not hasattr(FileStatus, "DELETED")
    assert FileStatus.NEW is FileStatus("new")
    assert FileStatus.CHANGED is FileStatus("changed")
    assert FileStatus.UNCHANGED is FileStatus("unchanged")


# --------------------------------------------------------------------------
# classify before extract (requirement 8.1)
# --------------------------------------------------------------------------


def test_an_unseen_file_classifies_as_new(
    store: StateStore, root: Path
) -> None:
    source = make_source(root, "alice/notes.md")
    assert store.classify(source, "hash-a", PARAMS) is FileStatus.NEW
    assert store.classify(source, "hash-a", PARAMS) is FileStatus.NEW
    assert store.records_for(source) == ()


def test_classify_does_not_write_file_state(
    store: StateStore, root: Path, db_path: Path
) -> None:
    source = make_source(root, "alice/notes.md")
    store.classify(source, "hash-a", PARAMS)
    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM file_state").fetchone()
    assert count is not None
    assert count[0] == 0


def test_matching_content_and_params_classify_as_unchanged(
    store: StateStore, root: Path
) -> None:
    source = make_source(root, "alice/notes.md")
    record = make_record(source, chunk_id="c1")
    store.commit_file(source, "hash-a", PARAMS, (record,), "run-1")
    assert store.classify(source, "hash-a", PARAMS) is FileStatus.UNCHANGED


def test_a_content_hash_change_classifies_as_changed(
    store: StateStore, root: Path
) -> None:
    source = make_source(root, "alice/notes.md")
    store.commit_file(
        source, "hash-a", PARAMS, (make_record(source, chunk_id="c1"),), "run-1"
    )
    assert store.classify(source, "hash-b", PARAMS) is FileStatus.CHANGED


def test_a_params_fingerprint_change_classifies_as_changed(
    store: StateStore, root: Path
) -> None:
    """Requirement 8.1 / 8.5: output-affecting parameters enter classification."""
    source = make_source(root, "alice/notes.md")
    store.commit_file(
        source, "hash-a", PARAMS, (make_record(source, chunk_id="c1"),), "run-1"
    )
    assert store.classify(source, "hash-a", OTHER_PARAMS) is FileStatus.CHANGED


# --------------------------------------------------------------------------
# Crash between two files (requirement 8.1 recovery)
# --------------------------------------------------------------------------


def test_a_crash_between_two_files_leaves_only_the_committed_file(
    db_path: Path, root: Path
) -> None:
    """Observable: a simulated crash between two files leaves the first
    committed and the second classified as new on reopen."""
    file_a = make_source(root, "alice/a.md")
    file_b = make_source(root, "bob/b.md")
    records_a = (make_record(file_a, chunk_id="a1", text="alpha"),)

    store = StateStore(db_path)
    try:
        store.commit_file(file_a, "hash-a", PARAMS, records_a, "run-1")
    finally:
        store.close()

    reopened = StateStore(db_path)
    try:
        assert reopened.classify(file_a, "hash-a", PARAMS) is FileStatus.UNCHANGED
        assert reopened.records_for(file_a) == records_a
        assert reopened.classify(file_b, "hash-b", PARAMS) is FileStatus.NEW
        assert reopened.records_for(file_b) == ()
    finally:
        reopened.close()


# --------------------------------------------------------------------------
# Second identical run (requirement 8.3)
# --------------------------------------------------------------------------


def test_a_second_identical_run_classifies_every_file_unchanged_and_returns_records(
    db_path: Path, root: Path
) -> None:
    """Observable: a second identical run classifies every file unchanged
    and returns its records without extraction. JSON round-trip holds."""
    file_a = write_source(root, "alice/a.md", b"alpha")
    file_b = write_source(root, "bob/b.md", b"beta")
    records_a = (make_record(file_a, chunk_id="a1", text="alpha"),)
    records_b = (
        make_record(file_b, chunk_id="b1", text="beta one"),
        make_record(
            file_b,
            chunk_id="b2",
            text="a described chart",
            kind=ChunkKind.FIGURE,
            ordinal=1,
            locator=PAGE_LOCATOR,
            provenance=PROVENANCE,
        ),
    )

    first = StateStore(db_path)
    try:
        first.commit_file(
            file_a, content_hash(file_a.path), PARAMS, records_a, "run-1"
        )
        first.commit_file(
            file_b, content_hash(file_b.path), PARAMS, records_b, "run-1"
        )
    finally:
        first.close()

    second = StateStore(db_path)
    try:
        for source, expected in ((file_a, records_a), (file_b, records_b)):
            digest = content_hash(source.path)
            assert second.classify(source, digest, PARAMS) is FileStatus.UNCHANGED
            restored = second.records_for(source)
            assert restored == expected
            assert all(
                ChunkRecord.from_json(record.to_json()) == record
                for record in restored
            )
    finally:
        second.close()


# --------------------------------------------------------------------------
# deleted_since (requirement 8.4)
# --------------------------------------------------------------------------


def test_deleted_since_reports_files_not_stamped_with_the_current_run(
    store: StateStore, root: Path
) -> None:
    """Requirement 8.4: files whose last_seen_run is older than the current
    run are reported with their chunk ids."""
    file_a = make_source(root, "alice/a.md")
    file_b = make_source(root, "bob/b.md")
    records_a = (make_record(file_a, chunk_id="a1"),)
    records_b = (
        make_record(file_b, chunk_id="b1"),
        make_record(file_b, chunk_id="b2", ordinal=1, text="second"),
    )
    store.commit_file(file_a, "hash-a", PARAMS, records_a, "run-1")
    store.commit_file(file_b, "hash-b", PARAMS, records_b, "run-1")
    assert store.deleted_since("run-1") == ()

    store.commit_file(file_a, "hash-a", PARAMS, records_a, "run-2")
    deleted = store.deleted_since("run-2")
    assert len(deleted) == 1
    missing, chunk_ids = deleted[0]
    assert missing.root == file_b.root
    assert missing.relative_path == file_b.relative_path
    assert chunk_ids == ("b1", "b2")


def test_classify_does_not_stamp_a_run_id(
    store: StateStore, root: Path
) -> None:
    """last_seen_run is written by ``commit_file``, not by classify."""
    file_a = make_source(root, "alice/a.md")
    store.commit_file(
        file_a, "hash-a", PARAMS, (make_record(file_a, chunk_id="a1"),), "run-1"
    )
    store.classify(file_a, "hash-a", PARAMS)
    missing = store.deleted_since("run-2")
    assert len(missing) == 1
    assert missing[0][0].relative_path == file_a.relative_path
    assert missing[0][1] == ("a1",)


# --------------------------------------------------------------------------
# One transaction per file
# --------------------------------------------------------------------------


def test_commit_writes_state_and_registry_rows_together(
    store: StateStore, root: Path, db_path: Path
) -> None:
    source = make_source(root, "alice/notes.md")
    records = (
        make_record(source, chunk_id="c1", text="one"),
        make_record(source, chunk_id="c2", text="two", ordinal=1),
    )
    store.commit_file(source, "hash-a", PARAMS, records, "run-1")

    with sqlite3.connect(db_path) as connection:
        state = connection.execute(
            "SELECT content_hash, params_fingerprint, status, last_seen_run "
            "FROM file_state WHERE root_id = ? AND relative_path = ?",
            (source.root.as_posix(), source.relative_path.as_posix()),
        ).fetchone()
        registry = connection.execute(
            "SELECT chunk_id, record_json FROM chunk_registry "
            "WHERE root_id = ? AND relative_path = ? ORDER BY rowid",
            (source.root.as_posix(), source.relative_path.as_posix()),
        ).fetchall()

    assert state is not None
    assert state[0] == "hash-a"
    assert state[1] == PARAMS.digest
    assert state[3] == "run-1"
    assert [row[0] for row in registry] == ["c1", "c2"]
    assert [ChunkRecord.from_json(row[1]) for row in registry] == list(records)


def test_a_failed_commit_does_not_leave_a_partial_file(
    store: StateStore, root: Path
) -> None:
    """file_state and chunk_registry commit in one transaction: a second
    insert that collides on chunk_id rolls the whole file back."""
    source = make_source(root, "alice/notes.md")
    original = (make_record(source, chunk_id="c1", text="kept"),)
    store.commit_file(source, "hash-a", PARAMS, original, "run-1")

    colliding = (
        make_record(source, chunk_id="dup", text="first"),
        make_record(source, chunk_id="dup", text="second", ordinal=1),
    )
    with pytest.raises(StateError):
        store.commit_file(source, "hash-b", PARAMS, colliding, "run-2")

    assert store.classify(source, "hash-a", PARAMS) is FileStatus.UNCHANGED
    assert store.records_for(source) == original


def test_recommitting_a_changed_file_replaces_its_registry_rows(
    store: StateStore, root: Path
) -> None:
    source = make_source(root, "alice/notes.md")
    store.commit_file(
        source,
        "hash-a",
        PARAMS,
        (make_record(source, chunk_id="old-1"), make_record(source, chunk_id="old-2", ordinal=1)),
        "run-1",
    )
    replacement = (make_record(source, chunk_id="new-1", text="replaced"),)
    store.commit_file(source, "hash-b", PARAMS, replacement, "run-2")
    assert store.classify(source, "hash-b", PARAMS) is FileStatus.UNCHANGED
    assert store.records_for(source) == replacement


def test_the_same_relative_path_under_two_roots_is_independent(
    store: StateStore, tmp_path: Path
) -> None:
    first = make_source(tmp_path / "root-a", "alice/notes.md")
    second = make_source(tmp_path / "root-b", "alice/notes.md")
    store.commit_file(
        first, "hash-a", PARAMS, (make_record(first, chunk_id="a1"),), "run-1"
    )
    assert store.classify(second, "hash-a", PARAMS) is FileStatus.NEW
    store.commit_file(
        second, "hash-b", PARAMS, (make_record(second, chunk_id="b1"),), "run-1"
    )
    assert store.records_for(first)[0].chunk_id == "a1"
    assert store.records_for(second)[0].chunk_id == "b1"


# --------------------------------------------------------------------------
# Physical schema
# --------------------------------------------------------------------------


def test_schema_matches_the_physical_data_model(db_path: Path) -> None:
    """design.md, Physical Data Model: file_state and chunk_registry."""
    store = StateStore(db_path)
    store.close()

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"
            )
        }
        file_state_cols = [
            row[1]
            for row in connection.execute("PRAGMA table_info(file_state)")
        ]
        registry_cols = [
            row[1]
            for row in connection.execute("PRAGMA table_info(chunk_registry)")
        ]

    assert "file_state" in tables
    assert "chunk_registry" in tables
    assert file_state_cols == [
        "root_id",
        "relative_path",
        "content_hash",
        "params_fingerprint",
        "status",
        "last_seen_run",
    ]
    assert registry_cols == ["chunk_id", "root_id", "relative_path", "record_json"]
    assert "chunk_registry_file" in indexes


def test_state_store_does_not_use_the_default_path(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        assert db_path.is_file()
        assert db_path.name == "ingest.sqlite"
        assert ".npu_rag" not in db_path.parts
    finally:
        store.close()


# --------------------------------------------------------------------------
# Boundary guard
# --------------------------------------------------------------------------


def test_state_does_not_import_rightward_or_vision_modules() -> None:
    """design.md, Architecture: state may import types, errors, config,
    credential, identity. It must not import vision (task 2.3 lives there
    as cache lookups) or anything to its right."""
    imported = _imported_names(MODULE_PATH.read_text(encoding="utf-8"))
    project = [name for name in imported if name.startswith("npu_rag")]
    allowed_prefixes = (
        "npu_rag.ingest.types",
        "npu_rag.ingest.errors",
        "npu_rag.ingest.config",
        "npu_rag.ingest.credential",
        "npu_rag.ingest.identity",
    )
    forbidden = [
        name
        for name in project
        if not any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in allowed_prefixes
        )
    ]
    assert forbidden == []
    joined = "\n".join(imported)
    assert "npu_rag.ingest.vision" not in joined
    assert "npu_rag.ingest.discover" not in joined
    assert "npu_rag.ingest.pipeline" not in joined
    assert "npu_rag.ingest.identity" in joined
    assert "npu_rag.ingest.types" in joined
