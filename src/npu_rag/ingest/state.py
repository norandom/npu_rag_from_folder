"""File state and the chunk registry (task 2.2).

One SQLite file holds per-file fingerprints and each serialised
``ChunkRecord``. Classification is a read: a file is new, changed or
unchanged before any extraction. A file's state row and its registry
rows commit together, stamped with the run id, so a crash between files
leaves the first durable and the second unseen. Unchanged files re-emit
from the registry; files whose run-id stamp is older than the current
run are the deleted set.

Vision cache lookups are task 2.3; this module creates the two tables
the file-level contract needs and does not import ``vision``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Self

from npu_rag.ingest.errors import StateError
from npu_rag.ingest.identity import ParamsFingerprint
from npu_rag.ingest.types import ChunkRecord, SourceFile

__all__ = [
    "FileStatus",
    "StateStore",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_state (
  root_id TEXT NOT NULL, relative_path TEXT NOT NULL,
  content_hash TEXT NOT NULL, params_fingerprint TEXT NOT NULL,
  status TEXT NOT NULL, last_seen_run TEXT NOT NULL,
  PRIMARY KEY (root_id, relative_path));
CREATE TABLE IF NOT EXISTS chunk_registry (
  chunk_id TEXT PRIMARY KEY, root_id TEXT NOT NULL, relative_path TEXT NOT NULL,
  record_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunk_registry_file ON chunk_registry (root_id, relative_path);
"""


class FileStatus(StrEnum):
    """How a present file compares to persisted state (requirement 8.1).

    Deleted is not a member: absence is reported by ``deleted_since``.
    """

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class StateStore:
    """SQLite-backed file state and chunk registry at ``path``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(path)
            self._connection.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise StateError(str(exc), path=path) from exc

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def classify(
        self,
        file: SourceFile,
        content_hash: str,
        params: ParamsFingerprint,
    ) -> FileStatus:
        """Classify ``file`` from persisted fingerprints; does not write."""
        row = self._state_row(file)
        if row is None:
            return FileStatus.NEW
        stored_hash, stored_params = row
        if stored_hash == content_hash and stored_params == params.digest:
            return FileStatus.UNCHANGED
        return FileStatus.CHANGED

    def commit_file(
        self,
        file: SourceFile,
        content_hash: str,
        params: ParamsFingerprint,
        records: Sequence[ChunkRecord],
        run_id: str,
    ) -> None:
        """Write one file's state and records in a single transaction."""
        root_id, relative = _file_key(file)
        status = self.classify(file, content_hash, params)
        payloads = tuple(
            (record.chunk_id, record.to_json()) for record in records
        )
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO file_state (
                      root_id, relative_path, content_hash, params_fingerprint,
                      status, last_seen_run)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (root_id, relative_path) DO UPDATE SET
                      content_hash = excluded.content_hash,
                      params_fingerprint = excluded.params_fingerprint,
                      status = excluded.status,
                      last_seen_run = excluded.last_seen_run
                    """,
                    (
                        root_id,
                        relative,
                        content_hash,
                        params.digest,
                        status.value,
                        run_id,
                    ),
                )
                self._connection.execute(
                    """
                    DELETE FROM chunk_registry
                    WHERE root_id = ? AND relative_path = ?
                    """,
                    (root_id, relative),
                )
                self._connection.executemany(
                    """
                    INSERT INTO chunk_registry (
                      chunk_id, root_id, relative_path, record_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (chunk_id, root_id, relative, payload)
                        for chunk_id, payload in payloads
                    ],
                )
        except sqlite3.Error as exc:
            raise StateError(str(exc), path=file.path) from exc

    def records_for(self, file: SourceFile) -> tuple[ChunkRecord, ...]:
        """An unchanged file's retained records, re-emitted without extraction."""
        root_id, relative = _file_key(file)
        try:
            rows = self._connection.execute(
                """
                SELECT record_json FROM chunk_registry
                WHERE root_id = ? AND relative_path = ?
                ORDER BY rowid
                """,
                (root_id, relative),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StateError(str(exc), path=file.path) from exc
        return tuple(ChunkRecord.from_json(payload) for (payload,) in rows)

    def deleted_since(
        self, run_id: str
    ) -> tuple[tuple[SourceFile, tuple[str, ...]], ...]:
        """Files whose last_seen_run is not ``run_id``, with their chunk ids."""
        try:
            files = self._connection.execute(
                """
                SELECT root_id, relative_path FROM file_state
                WHERE last_seen_run != ?
                ORDER BY root_id, relative_path
                """,
                (run_id,),
            ).fetchall()
            result: list[tuple[SourceFile, tuple[str, ...]]] = []
            for root_id, relative in files:
                ids = self._connection.execute(
                    """
                    SELECT chunk_id FROM chunk_registry
                    WHERE root_id = ? AND relative_path = ?
                    ORDER BY rowid
                    """,
                    (root_id, relative),
                ).fetchall()
                result.append(
                    (
                        _source_file(root_id, relative),
                        tuple(chunk_id for (chunk_id,) in ids),
                    )
                )
        except sqlite3.Error as exc:
            raise StateError(str(exc), path=self._path) from exc
        return tuple(result)

    def _state_row(self, file: SourceFile) -> tuple[str, str] | None:
        root_id, relative = _file_key(file)
        try:
            row = self._connection.execute(
                """
                SELECT content_hash, params_fingerprint FROM file_state
                WHERE root_id = ? AND relative_path = ?
                """,
                (root_id, relative),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StateError(str(exc), path=file.path) from exc
        if row is None:
            return None
        content, fingerprint = row
        if not isinstance(content, str) or not isinstance(fingerprint, str):
            raise StateError(
                "file_state row is not two text columns",
                path=file.path,
            )
        return content, fingerprint


def _file_key(file: SourceFile) -> tuple[str, str]:
    return file.root.as_posix(), file.relative_path.as_posix()


def _source_file(root_id: str, relative: str) -> SourceFile:
    root = Path(root_id)
    relative_path = Path(relative)
    author = relative_path.parts[0] if relative_path.parts else ""
    return SourceFile(
        path=root / relative_path,
        root=root,
        relative_path=relative_path,
        author=author,
    )
