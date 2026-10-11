"""Synchronization bridge between document ingest and vector index."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from npu_rag.embedding.types import DocumentText, ProviderChoice
from npu_rag.index.compat import verify_model_identity
from npu_rag.index.errors import SyncError
from npu_rag.index.schema import (
    UNSTAMPED_MODEL_IDENTITY,
    ChunkSchema,
    chunk_record_to_schema,
)
from npu_rag.index.store import LanceIndexStore
from npu_rag.index.types import SyncSummary
from npu_rag.ingest.types import ChunkRecord, RunReport

IngestReport = RunReport


class IndexSynchronizer:
    """Synchronizes document ingest outputs into embedded LanceDB vector storage."""

    def __init__(self, store: LanceIndexStore) -> None:
        self._store = store

    @property
    def store(self) -> LanceIndexStore:
        return self._store

    def sync(
        self,
        report: RunReport,
        embed_service: Any,
        *,
        model_identity: str = UNSTAMPED_MODEL_IDENTITY,
        provider: ProviderChoice | str = ProviderChoice.AUTO,
        batch_size: int = 64,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SyncSummary:
        """Synchronize ingest report with vector store.

        0. Refuses to write if the index disagrees with ``model_identity``.
        1. Purges removed chunk IDs from index.
        2. Detects new and modified chunks vs already indexed chunks.
        3. Generates embeddings for new/changed chunks only.
        4. Upserts new/changed records and updates FTS index.
        5. Returns a detailed SyncSummary.

        ``model_identity`` identifies the embedding model whose vectors this run
        will store, as ``schema.format_model_identity`` builds it. It is
        supplied by the caller and never introspected off ``embed_service``:
        ``EmbeddingService`` keeps its profile private and ``npu_rag.index``
        must not reach into it.

        It carries a default so that the one existing caller that predates
        stamping keeps working, but the default cannot produce a mixed index:
        step 0 rejects unstamped writes into a stamped table, and stamped
        writes into an unstamped one, before a single row is touched. A caller
        that never passes an identity therefore builds an index that is
        *uniformly* unstamped and visibly so (``IndexStats.unstamped_chunks``),
        never one that is quietly half of each.
        """
        t0 = time.perf_counter()
        if report is None:
            raise SyncError("Ingest report must not be None")
        if batch_size <= 0:
            raise SyncError("Batch size must be greater than zero")

        # Step 0, deliberately outside the try below: a ConfigurationError here
        # is a policy refusal and must reach the caller as itself, not wrapped
        # into a SyncError that reads like an I/O failure. It runs before the
        # purge as well as before the write - an incompatible run must not be
        # able to delete rows it is not allowed to replace.
        verify_model_identity(self._store, model_identity)

        if isinstance(provider, ProviderChoice):
            resolved_provider = provider
        else:
            try:
                resolved_provider = ProviderChoice(provider)
            except ValueError:
                resolved_provider = ProviderChoice.AUTO

        try:
            # 1. Purge removed chunks
            chunks_deleted = 0
            if report.removed_chunk_ids:
                chunks_deleted = self._store.delete_by_chunk_ids(report.removed_chunk_ids)

            # 2. Inspect existing records in store to identify new vs updated vs unchanged
            existing_chunks: dict[str, str] = {}
            if self._store.count() > 0:
                tbl_data = self._store.table.search().select(["chunk_id", "text"]).to_arrow()
                cids = tbl_data["chunk_id"].to_pylist()
                texts = tbl_data["text"].to_pylist()
                existing_chunks = dict(zip(cids, texts))

            chunks_to_embed: list[ChunkRecord] = []
            chunks_added = 0
            chunks_updated = 0

            for record in report.records:
                if record.chunk_id not in existing_chunks:
                    # Brand new chunk
                    chunks_to_embed.append(record)
                    chunks_added += 1
                elif existing_chunks[record.chunk_id] != record.text:
                    # Existing chunk with updated text
                    chunks_to_embed.append(record)
                    chunks_updated += 1
                else:
                    # Unchanged chunk: zero re-embedding
                    pass

            # 3. Batch embed new and changed chunks
            if chunks_to_embed:
                all_vectors: list[Sequence[float]] = []
                total_embed = len(chunks_to_embed)

                for i in range(0, total_embed, batch_size):
                    batch = chunks_to_embed[i : i + batch_size]
                    doc_texts = [
                        DocumentText(
                            content=c.text,
                            title=c.title if (c.title and c.title.strip()) else None,
                        )
                        for c in batch
                    ]
                    embed_result = embed_service.embed_documents(
                        doc_texts,
                        provider=resolved_provider,
                    )
                    for vec in embed_result.vectors:
                        all_vectors.append(vec)

                    if progress_callback is not None:
                        progress_callback(min(i + batch_size, total_embed), total_embed)

                # 4. Convert and upsert to LanceDB
                schema_records: list[ChunkSchema] = []
                for record, vec in zip(chunks_to_embed, all_vectors):
                    schema = chunk_record_to_schema(
                        record, vec, model_identity=model_identity
                    )
                    schema_records.append(schema)

                self._store.upsert(schema_records)

            # 5. Rebuild or refresh FTS index if database was modified
            if chunks_added > 0 or chunks_updated > 0 or chunks_deleted > 0:
                if self._store.count() > 0:
                    self._store.create_fts_index(field="text", replace=True)

            # 6. Compute totals
            total_chunks = self._store.count()
            files_indexed = 0
            if total_chunks > 0:
                file_paths = set(
                    self._store.table.search().select(["file_path"]).to_arrow()["file_path"].to_pylist()
                )
                files_indexed = len(file_paths)

            elapsed = time.perf_counter() - t0
            return SyncSummary(
                chunks_added=chunks_added,
                chunks_updated=chunks_updated,
                chunks_deleted=chunks_deleted,
                total_chunks=total_chunks,
                files_indexed=files_indexed,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            if isinstance(exc, SyncError):
                raise
            raise SyncError(f"Synchronization failed: {exc}") from exc
