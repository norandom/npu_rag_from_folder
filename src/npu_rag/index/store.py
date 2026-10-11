"""Embedded LanceDB storage implementation for vector-index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import lancedb
from lancedb.table import Table

from npu_rag.index.errors import StoreError
from npu_rag.index.schema import (
    UNSTAMPED_MODEL_IDENTITY,
    VECTOR_DIMENSION,
    ChunkSchema,
)
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

    def model_identity_counts(self) -> dict[str, int]:
        """Row count per recorded embedding-model identity.

        Keys are the ``model_identity`` column verbatim, so rows that carry no
        identity appear under ``UNSTAMPED_MODEL_IDENTITY`` (``""``) instead of
        being dropped: a partially stamped table is exactly one of the mixed
        cases this method exists to make visible. An empty table yields ``{}``.

        Only the one column is read, so this stays cheap enough to call before
        a sync or a search rather than being folded into :meth:`stats`, which
        also walks the database directory to size it.

        The scan carries **no limit, deliberately**. A plain (non-vector) query
        in lancedb 0.38.0 returns every row - measured, including after FTS and
        IVF_PQ indexes exist - so a limit would add nothing, and the obvious way
        to write one (``len(table)`` then ``.limit(total)``) is a check-then-read
        window: a table that grew between the two calls would be scanned short,
        and the gate could then pass a *mixed* index because the second model's
        rows fell outside the page. The unlimited scan cannot have that bug.

        A table predating the ``model_identity`` column is reported as wholly
        unstamped rather than raised on, so it reaches
        ``compat.verify_model_identity``'s unstamped branch - whose message is
        written for exactly this case and tells the operator to rebuild - and so
        that :meth:`stats` keeps answering instead of failing a ``status``
        report with what looks like corruption.
        """
        try:
            if "model_identity" not in self._table.schema.names:
                total = len(self._table)
                return {UNSTAMPED_MODEL_IDENTITY: total} if total else {}
            arrow = self._table.search().select(["model_identity"]).to_arrow()
            counts: dict[str, int] = {}
            for value in arrow["model_identity"].to_pylist():
                key = value if isinstance(value, str) else UNSTAMPED_MODEL_IDENTITY
                counts[key] = counts.get(key, 0) + 1
            return counts
        except Exception as exc:
            raise StoreError(f"Failed to read embedding model identities: {exc}") from exc

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

            identity_counts = self.model_identity_counts()
            model_identities = tuple(
                sorted(
                    identity
                    for identity in identity_counts
                    if identity != UNSTAMPED_MODEL_IDENTITY
                )
            )
            unstamped_chunks = identity_counts.get(UNSTAMPED_MODEL_IDENTITY, 0)

            return IndexStats(
                total_chunks=total_chunks,
                total_documents=distinct_docs,
                vector_dimension=VECTOR_DIMENSION,
                has_fts_index=has_fts,
                index_size_bytes=size_bytes,
                model_identities=model_identities,
                unstamped_chunks=unstamped_chunks,
            )
        except Exception as exc:
            raise StoreError(f"Failed to gather index stats: {exc}") from exc
