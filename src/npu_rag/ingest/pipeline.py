"""Ingest orchestration pipeline and single entry point (task 7.2).

Orchestrates the entire ingest process:
  1. Validates configuration and cross-checks budget against tokenizer limit.
  2. Initializes StateStore (aborts before processing any file on store failure).
  3. Discovers source files across configured roots, isolating unavailable roots.
  4. Routes files by format and classifies them before extraction.
  5. Re-emits unchanged files' retained records from StateStore without re-extracting.
  6. For new/changed files: extracts segments, resolves images with bounded concurrency,
     chunks content against the token budget, and commits to StateStore.
  7. Isolates per-file failures into FAILED omissions while allowing KeyboardInterrupt through.
  8. Tracks vision requests issued and cache hits.
  9. Identifies deleted chunk IDs since the previous run.
 10. Returns RunReport carrying all current chunk records.

Requirements: 8.3, 8.6, 9.1, 9.5, 10.1.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.ingest.chunk import Chunker
from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.discover import Discoverer
from npu_rag.ingest.errors import IngestError, StateError
from npu_rag.ingest.extract.base import Extractor
from npu_rag.ingest.extract.excel import ExcelExtractor
from npu_rag.ingest.extract.image import ImageExtractor
from npu_rag.ingest.extract.markdown import MarkdownExtractor
from npu_rag.ingest.extract.pdf import PdfExtractor
from npu_rag.ingest.extract.text import TextExtractor
from npu_rag.ingest.identity import content_hash, params_fingerprint
from npu_rag.ingest.report import RunReport
from npu_rag.ingest.route import ExtractionPath, Router
from npu_rag.ingest.state import FileStatus, StateStore
from npu_rag.ingest.types import (
    ChunkRecord,
    Extracted,
    FigureSegment,
    ImageRef,
    Omission,
    OmissionCategory,
    Segment,
)
from npu_rag.ingest.vision import (
    PIPELINE_VERSION,
    LocalOcrDescriber,
    VisionDescriber,
    resolve_all,
)

__all__ = ["run_ingest"]

_EXTRACTOR_VERSIONS: Mapping[str, str] = {
    "excel": "1",
    "image": "1",
    "markdown": "1",
    "pdf": "1",
    "text": "1",
}

_ADAPTERS: dict[ExtractionPath, Extractor] = {
    ExtractionPath.MARKDOWN: MarkdownExtractor(),
    ExtractionPath.TEXT: TextExtractor(),
    ExtractionPath.PDF: PdfExtractor(),
    ExtractionPath.EXCEL: ExcelExtractor(),
    ExtractionPath.IMAGE: ImageExtractor(),
}


class _TrackingDescriber:
    """Wrapper around VisionDescriber that counts calls to describe()."""

    def __init__(self, describer: VisionDescriber) -> None:
        self._describer = describer
        self.requests = 0

    @property
    def recognizer_id(self) -> str:
        return getattr(self._describer, "recognizer_id", "local_ocr")

    def describe(self, image: ImageRef) -> str:
        self.requests += 1
        return self._describer.describe(image)


class _TrackingStore:
    """Wrapper around StateStore that counts vision cache hits."""

    def __init__(self, store: StateStore) -> None:
        self._store = store
        self.hits = 0

    def cached_description(
        self, image_sha256: str, model_id: str, prompt_version: str
    ) -> str | None:
        res = self._store.cached_description(image_sha256, model_id, prompt_version)
        if res is not None:
            self.hits += 1
        return res

    def store_description(
        self, image_sha256: str, model_id: str, prompt_version: str, text: str
    ) -> None:
        self._store.store_description(image_sha256, model_id, prompt_version, text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


def run_ingest(
    config: IngestConfig,
    tokenizer: ModelTokenizer,
    *,
    describer: VisionDescriber | None = None,
) -> RunReport:
    """Execute the ingest pipeline from source roots to ChunkRecords and RunReport."""
    # Cross-check budget against tokenizer limit
    if tokenizer.max_input_tokens < config.token_budget:
        raise ValueError(
            f"token_budget ({config.token_budget}) exceeds tokenizer max_input_tokens "
            f"({tokenizer.max_input_tokens})"
        )

    # Initialize StateStore (fails fast before processing files if store cannot open)
    with StateStore(config.state_path) as store:
        tracking_store = _TrackingStore(store)

        # Setup vision describer
        active_describer = describer
        if active_describer is None:
            try:
                active_describer = LocalOcrDescriber()
            except Exception:
                active_describer = None

        tracking_describer = (
            _TrackingDescriber(active_describer) if active_describer is not None else None
        )

        # Discover source files
        discoverer = Discoverer()
        discovery = discoverer.discover(config)
        files, root_omissions = discovery.files, discovery.omissions

        # Compute parameter fingerprint
        tokenizer_id = getattr(
            tokenizer, "model_id", getattr(tokenizer, "name_or_path", "default-tokenizer")
        )
        params = params_fingerprint(
            config=config,
            tokenizer_id=str(tokenizer_id),
            extractor_versions=_EXTRACTOR_VERSIONS,
            prompt_version=PIPELINE_VERSION,
        )

        run_id = uuid.uuid4().hex
        router = Router()
        chunker = Chunker(
            tokenizer,
            budget=config.token_budget,
            prose_overlap_tokens=config.prose_overlap_tokens,
        )

        files_processed = 0
        files_unchanged = 0
        files_skipped = len(root_omissions)
        files_failed = 0
        omissions: list[Omission] = list(root_omissions)
        all_records: list[ChunkRecord] = []

        for file in files:
            try:
                # 1. Route
                route_res = router.route(file)
                if isinstance(route_res, Omission):
                    omissions.append(route_res)
                    files_skipped += 1
                    continue
                extraction_path = route_res

                # 2. Content hash & Classify
                file_hash = content_hash(file.path)
                status = store.classify(file, file_hash, params)

                if status is FileStatus.UNCHANGED:
                    # Re-emit unchanged file's records from store (requirement 8.3)
                    retained = store.records_for(file)
                    # Stamp with current run_id so file is not reported as deleted
                    store.commit_file(file, file_hash, params, retained, run_id)
                    all_records.extend(retained)
                    files_unchanged += 1
                    continue

                # 3. Extract
                adapter = _ADAPTERS[extraction_path]
                extracted = adapter.extract(file, config)
                omissions.extend(extracted.omissions)

                # 4. Resolve image references
                image_refs = [
                    s for s in extracted.segments if isinstance(s, ImageRef)
                ]
                if image_refs:
                    resolved_images = resolve_all(
                        image_refs,
                        store=cast(StateStore, tracking_store),
                        describer=cast(VisionDescriber, tracking_describer),
                        config=config,
                        source_path=file.path,
                    )
                    resolved_iter = iter(resolved_images)
                    final_segments: list[Segment | FigureSegment] = []
                    for seg in extracted.segments:
                        if isinstance(seg, ImageRef):
                            res = next(resolved_iter)
                            if res.figure is not None:
                                final_segments.append(res.figure)
                            if res.omission is not None:
                                omissions.append(res.omission)
                        else:
                            final_segments.append(seg)
                    extracted = Extracted(
                        title=extracted.title,
                        segments=tuple(final_segments),
                        omissions=extracted.omissions,
                    )

                # 5. Chunk
                file_records = chunker.chunk(file, extracted, params=params)

                # 6. Commit to StateStore
                store.commit_file(file, file_hash, params, file_records, run_id)
                all_records.extend(file_records)
                files_processed += 1

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                files_failed += 1
                msg = getattr(exc, "message", None) or str(exc) or type(exc).__name__
                omissions.append(
                    Omission(
                        category=OmissionCategory.FAILED,
                        path=file.path,
                        reason=msg,
                    )
                )

        # Compute deleted chunk IDs at the end of the run (requirement 8.4)
        deleted_files_and_chunks = store.deleted_since(run_id)
        removed_chunk_ids = tuple(
            cid for _f, cids in deleted_files_and_chunks for cid in cids
        )
        # Purge files deleted from source roots so they don't persist indefinitely
        store.purge_deleted(run_id)

        no_work_required = (
            files_processed == 0 and not removed_chunk_ids and files_failed == 0
        )

        return RunReport(
            files_processed=files_processed,
            files_unchanged=files_unchanged,
            files_skipped=files_skipped,
            files_failed=files_failed,
            omissions=tuple(omissions),
            removed_chunk_ids=removed_chunk_ids,
            no_work_required=no_work_required,
            vision_requests_issued=(
                tracking_describer.requests if tracking_describer is not None else 0
            ),
            vision_cache_hits=tracking_store.hits,
            records=tuple(all_records),
        )
