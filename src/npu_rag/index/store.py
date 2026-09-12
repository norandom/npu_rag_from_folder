"""Embedded LanceDB storage implementation for vector-index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import lancedb
from lancedb.table import Table

from npu_rag.index.errors import StoreError
from npu_rag.index.schema import VECTOR_DIMENSION, ChunkSchema
from npu_rag.index.types import IndexStats


class LanceIndexStore:
    """Manages the embedded LanceDB database and chunks table."""

    def __init__(
        self,
        db_path: Path | str,
        table_name: str = "chunks",
    ) -> None:
        self._db_path = Path(db_path).resolve()
        self._table_name = table_name
        self._db_path.mkdir(parents=True, exist_ok=True)
        try:
            self._db = lancedb.connect(str(self._db_path))
            table_names = list(self._db.list_tables().tables)
            if self._table_name in table_names:
                self._table = self._db.open_table(self._table_name)
            else:
                self._table = self._db.create_table(
                    self._table_name,
                    schema=ChunkSchema,
                    mode="create",
                )
        except Exception as exc:
            raise StoreError(f"Failed to initialize LanceDB store at {self._db_path}: {exc}") from exc

    @property
    def table(self) -> Table:
        """The underlying LanceDB table."""
        return self._table

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def db_path(self) -> Path:
        return self._db_path

    def count(self) -> int:
        """Return total number of chunks stored."""
        try:
            return len(self._table)
        except Exception as exc:
            raise StoreError(f"Failed to count records: {exc}") from exc

    def add(self, records: Sequence[ChunkSchema | dict[str, Any]]) -> int:
        """Insert records into the table. Returns number of records added."""
        if not records:
            return 0
        try:
            self._table.add(records)
            return len(records)
        except Exception as exc:
            raise StoreError(f"Failed to add records to LanceDB: {exc}") from exc

    def upsert(self, records: Sequence[ChunkSchema | dict[str, Any]]) -> int:
        """Upsert records keyed on chunk_id. Existing chunk_ids are updated; new ones inserted."""
        if not records:
            return 0
        try:
            (
                self._table.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records)
            )
            return len(records)
        except Exception as exc:
            raise StoreError(f"Failed to upsert records into LanceDB: {exc}") from exc

    def delete_by_chunk_ids(self, chunk_ids: Sequence[str]) -> int:
        """Delete chunks matching any of the given chunk_ids."""
        if not chunk_ids:
            return 0
        try:
            escaped = [cid.replace("'", "''") for cid in chunk_ids]
            in_clause = ", ".join(f"'{cid}'" for cid in escaped)
            self._table.delete(f"chunk_id IN ({in_clause})")
            return len(chunk_ids)
        except Exception as exc:
            raise StoreError(f"Failed to delete chunks by IDs: {exc}") from exc

    def delete_by_file_path(self, file_path: str | Path) -> None:
        """Delete all chunks belonging to a given source file path."""
        try:
            clean = str(file_path).replace("'", "''")
            self._table.delete(f"file_path = '{clean}'")
        except Exception as exc:
            raise StoreError(f"Failed to delete chunks for file_path '{file_path}': {exc}") from exc

    def delete_by_predicate(self, predicate: str) -> None:
        """Delete chunks satisfying the given SQL predicate."""
        try:
            self._table.delete(predicate)
        except Exception as exc:
            raise StoreError(f"Failed to delete chunks with predicate '{predicate}': {exc}") from exc

    def compact(self) -> None:
        """Run table optimization (compaction and cleanup)."""
        try:
            if hasattr(self._table, "optimize"):
                self._table.optimize()
        except Exception as exc:
            raise StoreError(f"Failed to compact LanceDB table: {exc}") from exc

    def create_fts_index(self, field: str = "text", *, replace: bool = True) -> None:
        """Create or rebuild a native FTS index on the specified text column."""
        try:
            import lancedb.index

            self._table.create_index(field, config=lancedb.index.FTS(), replace=replace)
        except Exception as exc:
            raise StoreError(f"Failed to create FTS index on '{field}': {exc}") from exc

    def stats(self) -> IndexStats:
        """Gather index statistics."""
        try:
            total_chunks = len(self._table)
            distinct_docs = 0
            if total_chunks > 0:
                try:
                    df = self._table.search().select(["file_path"]).limit(total_chunks).to_arrow()
                    distinct_docs = len(df["file_path"].unique())
                except Exception:
                    distinct_docs = total_chunks

            size_bytes = 0
            candidate_paths = [self._db_path]
            table_uri = getattr(self._table, "uri", None)
            if table_uri:
                candidate_paths.append(Path(str(table_uri)))
            for base in list(candidate_paths):
                if base.drive:
                    for d in ("C:", "D:"):
                        if d.lower() != base.drive.lower():
                            candidate_paths.append(Path(d + str(base)[len(base.drive) :]))

            for path in candidate_paths:
                if path.exists():
                    if path.is_file():
                        size_bytes += path.stat().st_size
                    else:
                        for f in path.rglob("*"):
                            if f.is_file():
                                size_bytes += f.stat().st_size
                if size_bytes > 0:
                    break

            has_fts = False
            try:
                indices = self._table.list_indices()
                has_fts = any("fts" in str(idx).lower() for idx in indices)
            except Exception:
                has_fts = False

            return IndexStats(
                total_chunks=total_chunks,
                total_documents=distinct_docs,
                vector_dimension=VECTOR_DIMENSION,
                has_fts_index=has_fts,
                index_size_bytes=size_bytes,
            )
        except Exception as exc:
            raise StoreError(f"Failed to gather index stats: {exc}") from exc
