"""Validation 8.1: Incremental behaviour end to end.

Validates the full incremental lifecycle across four scenarios:
  1. Double run over a temporary root: the second run extracts nothing, issues
     no describer call, reports no work required, and its report carries every
     retained record from the store.
  2. File deletion: delete a file and its chunk ids are reported removed.
  3. Parameter change: change one output-affecting parameter and affected files
     are reclassified changed and re-extracted; OCR result is reused from cache.
  4. File addition: add one file and only it is processed.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import npu_rag.ingest.pipeline as pipeline_mod
from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.extract.base import Extractor
from npu_rag.ingest.pipeline import run_ingest
from npu_rag.ingest.route import ExtractionPath
from npu_rag.ingest.types import Extracted, ImageRef, SourceFile


class _SpyExtractor:
    """Wraps an Extractor to record every SourceFile passed to extract()."""

    def __init__(self, inner: Extractor) -> None:
        self._inner = inner
        self.calls: list[SourceFile] = []

    def extract(self, source: SourceFile, config: IngestConfig) -> Extracted:
        self.calls.append(source)
        return self._inner.extract(source, config)


class _SpyDescriber:
    """Fake VisionDescriber that records every ImageRef passed to describe()."""

    def __init__(self, recognizer_id: str = "mock-rec-v1") -> None:
        self.recognizer_id = recognizer_id
        self.calls: list[ImageRef] = []

    def describe(self, image: ImageRef) -> str:
        self.calls.append(image)
        return f"Description of image {image.width}x{image.height}"


def _create_png(path: Path, width: int = 250, height: int = 250) -> None:
    img = Image.new("RGB", (width, height), color=(180, 180, 180))
    img.save(path, format="PNG")


def test_incremental_behaviour_end_to_end(
    tmp_path: Path,
    runtime_tokenizer: ModelTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observable 8.1: Across the four scenarios the extractor and describer fakes
    record exactly the calls the scenario permits and no others, and each report's
    counts and removed ids match the literal expected values.
    """
    root = tmp_path / "archive"
    root.mkdir()
    state_path = tmp_path / "state.sqlite"

    # Install spies on the extractors
    markdown_spy = _SpyExtractor(pipeline_mod._ADAPTERS[ExtractionPath.MARKDOWN])
    image_spy = _SpyExtractor(pipeline_mod._ADAPTERS[ExtractionPath.IMAGE])
    text_spy = _SpyExtractor(pipeline_mod._ADAPTERS[ExtractionPath.TEXT])

    monkeypatch.setitem(pipeline_mod._ADAPTERS, ExtractionPath.MARKDOWN, markdown_spy)
    monkeypatch.setitem(pipeline_mod._ADAPTERS, ExtractionPath.IMAGE, image_spy)
    monkeypatch.setitem(pipeline_mod._ADAPTERS, ExtractionPath.TEXT, text_spy)

    describer_spy = _SpyDescriber()

    # Plant initial files: 1 markdown, 1 image, 1 text
    doc_path = root / "article.md"
    doc_path.write_text("# Article\n\nSome interesting prose content.", encoding="utf-8")

    img_path = root / "figure.png"
    _create_png(img_path, width=250, height=250)

    txt_path = root / "notes.txt"
    txt_path.write_text("Simple plain text notes for indexing.", encoding="utf-8")

    base_config = IngestConfig(
        roots=(root,),
        token_budget=512,
        prose_overlap_tokens=32,
        min_image_pixels=200,
        state_path=state_path,
    )

    # -------------------------------------------------------------------------
    # Scenario 1: Initial run & immediate second run (Requirements 8.1, 8.3, 8.6)
    # -------------------------------------------------------------------------

    # Run 1: Initial ingest
    report1 = run_ingest(base_config, runtime_tokenizer, describer=describer_spy)

    assert report1.files_processed == 3
    assert report1.files_unchanged == 0
    assert report1.files_skipped == 0
    assert report1.files_failed == 0
    assert report1.no_work_required is False
    assert report1.removed_chunk_ids == ()
    assert len(report1.records) >= 3
    assert report1.vision_requests_issued == 1
    assert report1.vision_cache_hits == 0

    assert len(markdown_spy.calls) == 1
    assert len(image_spy.calls) == 1
    assert len(text_spy.calls) == 1
    assert len(describer_spy.calls) == 1

    # Record initial chunk IDs
    records1_by_id = {r.chunk_id: r for r in report1.records}
    txt_chunk_ids = {
        r.chunk_id for r in report1.records if r.source_path.name == "notes.txt"
    }
    assert len(txt_chunk_ids) > 0

    # Run 2: Exact re-run with no changes
    report2 = run_ingest(base_config, runtime_tokenizer, describer=describer_spy)

    assert report2.files_processed == 0
    assert report2.files_unchanged == 3
    assert report2.files_skipped == 0
    assert report2.files_failed == 0
    assert report2.no_work_required is True
    assert report2.removed_chunk_ids == ()
    assert report2.vision_requests_issued == 0
    assert report2.vision_cache_hits == 0

    # Zero extra extractor or describer calls
    assert len(markdown_spy.calls) == 1
    assert len(image_spy.calls) == 1
    assert len(text_spy.calls) == 1
    assert len(describer_spy.calls) == 1

    # All records re-emitted identically from store
    records2_by_id = {r.chunk_id: r for r in report2.records}
    assert records2_by_id == records1_by_id

    # -------------------------------------------------------------------------
    # Scenario 2: File deletion (Requirement 8.4)
    # -------------------------------------------------------------------------
    txt_path.unlink()

    # Run 3: File deleted
    report3 = run_ingest(base_config, runtime_tokenizer, describer=describer_spy)

    assert report3.files_processed == 0
    assert report3.files_unchanged == 2
    assert report3.files_skipped == 0
    assert report3.files_failed == 0
    assert report3.no_work_required is False
    assert set(report3.removed_chunk_ids) == txt_chunk_ids
    assert report3.vision_requests_issued == 0
    assert report3.vision_cache_hits == 0

    # No extractor or describer calls
    assert len(markdown_spy.calls) == 1
    assert len(image_spy.calls) == 1
    assert len(text_spy.calls) == 1
    assert len(describer_spy.calls) == 1

    # Records include only the retained files
    remaining_ids = {r.chunk_id for r in report3.records}
    assert txt_chunk_ids.isdisjoint(remaining_ids)

    # -------------------------------------------------------------------------
    # Scenario 3: Parameter change (Requirement 8.2, 8.5)
    # -------------------------------------------------------------------------
    # Changing prose_overlap_tokens changes params_fingerprint
    changed_config = IngestConfig(
        roots=(root,),
        token_budget=512,
        prose_overlap_tokens=48,  # changed from 32
        min_image_pixels=200,
        state_path=state_path,
    )

    # Run 4: Parameter changed -> remaining files reclassified as changed
    report4 = run_ingest(changed_config, runtime_tokenizer, describer=describer_spy)

    assert report4.files_processed == 2
    assert report4.files_unchanged == 0
    assert report4.files_skipped == 0
    assert report4.files_failed == 0
    assert report4.no_work_required is False
    assert report4.removed_chunk_ids == ()

    # Both remaining files were re-extracted
    assert len(markdown_spy.calls) == 2  # 1 new call
    assert len(image_spy.calls) == 2     # 1 new call
    assert len(text_spy.calls) == 1      # unchanged

    # But OCR describer was NOT called because result was served from cache!
    assert len(describer_spy.calls) == 1
    assert report4.vision_requests_issued == 0
    assert report4.vision_cache_hits == 1

    # -------------------------------------------------------------------------
    # Scenario 4: Add one file (Requirements 8.1, 8.3)
    # -------------------------------------------------------------------------
    extra_path = root / "extra.md"
    extra_path.write_text("# Extra\n\nNewly added document.", encoding="utf-8")

    # Run 5: One file added
    report5 = run_ingest(changed_config, runtime_tokenizer, describer=describer_spy)

    assert report5.files_processed == 1
    assert report5.files_unchanged == 2
    assert report5.files_skipped == 0
    assert report5.files_failed == 0
    assert report5.no_work_required is False
    assert report5.removed_chunk_ids == ()

    # Only extra.md extracted
    assert len(markdown_spy.calls) == 3  # 1 new call for extra.md
    assert markdown_spy.calls[-1].path.name == "extra.md"
    assert len(image_spy.calls) == 2     # 0 new calls
    assert len(text_spy.calls) == 1      # 0 new calls
    assert len(describer_spy.calls) == 1 # 0 new calls

    # Emitted records contain all 3 active documents
    report5_filenames = {r.source_path.name for r in report5.records}
    assert report5_filenames == {"article.md", "figure.png", "extra.md"}
