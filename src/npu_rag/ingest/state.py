"""File state, the chunk registry, and the vision cache.

One SQLite file holds per-file fingerprints, each serialised
``ChunkRecord``, and image-to-text results keyed by image hash, model id
and prompt version. Classification is a read: a file is new, changed or
unchanged before any extraction. A file's state row and its registry
rows commit together, stamped with the run id, so a crash between files
leaves the first durable and the second unseen. Unchanged files re-emit
from the registry; files whose run-id stamp is older than the current
run are the deleted set.

The vision cache is read and written under a lock from worker threads
sharing this connection. A hit returns the stored text and does nothing
else. This module does not import ``vision``.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
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
CREATE TABLE IF NOT EXISTS vision_cache (
  image_sha256 TEXT NOT NULL, model_id TEXT NOT NULL, prompt_version TEXT NOT NULL,
  text TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (image_sha256, model_id, prompt_version));
"""


class FileStatus(StrEnum):
    """How a present file compares to persisted state (requirement 8.1).

    Deleted is not a member: absence is reported by ``deleted_since``.
    """

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class StateStore:
    """SQLite-backed file state, chunk registry and vision cache at ``path``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(path, check_same_thread=False)
            self._connection.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise StateError(str(exc), path=path) from exc

    def close(self) -> None:
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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

    def cached_description(
        self, image_sha256: str, model_id: str, prompt_version: str
    ) -> str | None:
        """Return a retained image-to-text result, or ``None`` on a miss.

        A hit is a read: it does not rewrite the cache row or touch
        ``file_state``.
        """
        with self._lock:
            try:
                row = self._connection.execute(
                    """
                    SELECT text FROM vision_cache
                    WHERE image_sha256 = ? AND model_id = ? AND prompt_version = ?
                    """,
                    (image_sha256, model_id, prompt_version),
                ).fetchone()
            except sqlite3.Error as exc:
                raise StateError(str(exc), path=self._path) from exc
        if row is None:
            return None
        text = row[0]
        if not isinstance(text, str):
            raise StateError(
                "vision_cache text is not a text column",
                path=self._path,
            )
        return text

    def store_description(
        self,
        image_sha256: str,
        model_id: str,
        prompt_version: str,
        text: str,
    ) -> None:
        """Retain an image-to-text result keyed by image, model and prompt."""
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO vision_cache (
                          image_sha256, model_id, prompt_version, text, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (image_sha256, model_id, prompt_version)
                        DO UPDATE SET
                          text = excluded.text,
                          created_at = excluded.created_at
                        """,
                        (
                            image_sha256,
                            model_id,
                            prompt_version,
                            text,
                            created_at,
                        ),
                    )
            except sqlite3.Error as exc:
                raise StateError(str(exc), path=self._path) from exc

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
