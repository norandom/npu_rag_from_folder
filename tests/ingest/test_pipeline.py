"""Tests for the ingest pipeline orchestration and public surface (task 7.2).

Boundary: pipeline.py and npu_rag.ingest.__init__.py.
Requirements: 8.3, 8.6, 9.1, 9.5, 10.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import npu_rag.ingest
from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import StateError
from npu_rag.ingest.pipeline import run_ingest
from npu_rag.ingest.types import (
    ChunkKind,
    ChunkRecord,
    ImageRef,
    OmissionCategory,
    RunReport,
)


class _FakeDescriber:
    def __init__(self, description: str = "A mock chart") -> None:
        self.recognizer_id = "mock-rec"
        self.description = description
        self.call_count = 0

    def describe(self, ref: ImageRef) -> str:
        self.call_count += 1
        return self.description


def _write_minimal_png(path: Path, width: int = 250, height: int = 250) -> None:
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    img.save(path, format="PNG")


def _norm_path(path: Path | str) -> str:
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    return text.replace("\\", "/").lower()


def test_public_surface_exports_contracts() -> None:
    """Task 7.2: export the public contracts from the package root."""
    assert hasattr(npu_rag.ingest, "run_ingest")
    assert hasattr(npu_rag.ingest, "IngestConfig")
    assert hasattr(npu_rag.ingest, "RunReport")
    assert hasattr(npu_rag.ingest, "ChunkRecord")
    assert hasattr(npu_rag.ingest, "ChunkKind")
    assert "run_ingest" in npu_rag.ingest.__all__
    assert "IngestConfig" in npu_rag.ingest.__all__
    assert "RunReport" in npu_rag.ingest.__all__
    assert "ChunkRecord" in npu_rag.ingest.__all__


def test_run_ingest_single_file_success(
    tmp_path: Path, runtime_tokenizer: ModelTokenizer
) -> None:
    """Single file run successfully processes and returns RunReport with records."""
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "guide.md"
    doc.write_text("# Guide\n\nThis is a simple guide document.", encoding="utf-8")

    state_path = tmp_path / "state.sqlite"
    config = IngestConfig(roots=(root,), token_budget=512, state_path=state_path)

    report = run_ingest(config, runtime_tokenizer)
    assert report.files_processed == 1
    assert report.files_unchanged == 0
    assert report.files_skipped == 0
    assert report.files_failed == 0
    assert report.no_work_required is False
    assert len(report.records) > 0
    assert _norm_path(report.records[0].source_path) == _norm_path(doc)


def test_run_ingest_unchanged_file_reemits_records(
    tmp_path: Path, runtime_tokenizer: ModelTokenizer
) -> None:
    """Requirement 8.3 & 8.6: unchanged files re-emit records from store; second run is no work required."""
    root = tmp_path / "docs"
    root.mkdir()
    doc1 = root / "one.md"
    doc1.write_text("# Doc One\n\nFirst document content.", encoding="utf-8")
    doc2 = root / "two.md"
    doc2.write_text("# Doc Two\n\nSecond document content.", encoding="utf-8")

    state_path = tmp_path / "state.sqlite"
    config = IngestConfig(roots=(root,), token_budget=512, state_path=state_path)

    # First run: both processed
    report1 = run_ingest(config, runtime_tokenizer)
    assert report1.files_processed == 2
    assert report1.files_unchanged == 0
    assert report1.no_work_required is False
    records1 = report1.records
    assert len(records1) >= 2

    # Second run: 0 processed, 2 unchanged, retained records re-emitted
    report2 = run_ingest(config, runtime_tokenizer)
    assert report2.files_processed == 0
    assert report2.files_unchanged == 2
    assert report2.files_failed == 0
    assert report2.no_work_required is True
    # Re-emitted records match
    assert len(report2.records) == len(records1)
    assert {r.chunk_id for r in report2.records} == {r.chunk_id for r in records1}


def test_run_ingest_file_deletion_reports_removed_chunk_ids(
    tmp_path: Path, runtime_tokenizer: ModelTokenizer
) -> None:
    """Requirement 8.4: deleted file reports its chunk ids as removed."""
    root = tmp_path / "docs"
    root.mkdir()
    doc1 = root / "keep.md"
    doc1.write_text("# Keep\n\nKeeping this file.", encoding="utf-8")
    doc2 = root / "delete_me.md"
    doc2.write_text("# Delete\n\nDeleting this file.", encoding="utf-8")

    state_path = tmp_path / "state.sqlite"
    config = IngestConfig(roots=(root,), token_budget=512, state_path=state_path)

    report1 = run_ingest(config, runtime_tokenizer)
    assert report1.files_processed == 2
    deleted_cids = {r.chunk_id for r in report1.records if _norm_path(r.source_path) == _norm_path(doc2)}
    assert len(deleted_cids) > 0

    # Delete the file
    doc2.unlink()

    report2 = run_ingest(config, runtime_tokenizer)
    assert report2.files_processed == 0
    assert report2.files_unchanged == 1
    assert set(report2.removed_chunk_ids) == deleted_cids
    assert report2.no_work_required is False


def test_run_ingest_failure_isolation_with_corrupt_file(
    tmp_path: Path, runtime_tokenizer: ModelTokenizer
) -> None:
    """Observable 1 (Requirement 9.1): corrupt file is a failed omission, good file committed."""
    root = tmp_path / "docs"
    root.mkdir()
    good = root / "good.md"
    good.write_text("# Good Document\n\nValid text.", encoding="utf-8")

    corrupt = root / "corrupt.pdf"
    corrupt.write_bytes(b"NOT A REAL PDF HEADER")

    state_path = tmp_path / "state.sqlite"
    config = IngestConfig(roots=(root,), token_budget=512, state_path=state_path)

    report = run_ingest(config, runtime_tokenizer)
    assert report.files_processed == 1
    assert report.files_failed == 1
    assert report.no_work_required is False

    failed_omissions = [o for o in report.omissions if o.category == OmissionCategory.FAILED]
    assert len(failed_omissions) == 1
    assert _norm_path(failed_omissions[0].path) == _norm_path(corrupt)

    # Good file was committed and its records returned
    assert len(report.records) > 0
    assert any(_norm_path(r.source_path) == _norm_path(good) for r in report.records)


def test_run_ingest_all_files_fail_still_returns_report(
    tmp_path: Path, runtime_tokenizer: ModelTokenizer
) -> None:
    """Observable 2 (Requirement 9.5): run where every file fails still returns a report."""
    root = tmp_path / "docs"
    root.mkdir()
    corrupt1 = root / "bad1.pdf"
    corrupt1.write_bytes(b"GARBAGE")
    corrupt2 = root / "bad2.xlsx"
    corrupt2.write_bytes(b"NOT_A_ZIP")

    state_path = tmp_path / "state.sqlite"
    config = IngestConfig(roots=(root,), token_budget=512, state_path=state_path)

    report = run_ingest(config, runtime_tokenizer)
    assert report.files_processed == 0
    assert report.files_failed == 2
    assert report.no_work_required is False
    assert len(report.records) == 0
    assert len(report.omissions) == 2


def test_run_ingest_unsupported_file_is_skipped(
    tmp_path: Path, runtime_tokenizer: ModelTokenizer
) -> None:
    """Requirement 2.4: unsupported file is an unsupported omission and counted as skipped."""
    root = tmp_path / "docs"
    root.mkdir()
    unsupported = root / "archive.bin"
    unsupported.write_bytes(b"\x00\x01\x02\x03\x04")

    state_path = tmp_path / "state.sqlite"
    config = IngestConfig(roots=(root,), token_budget=512, state_path=state_path)

    report = run_ingest(config, runtime_tokenizer)
    assert report.files_skipped == 1
    assert report.files_processed == 0
    assert any(o.category == OmissionCategory.UNSUPPORTED for o in report.omissions)


def test_run_ingest_keyboard_interrupt_propagates(
    tmp_path: Path, runtime_tokenizer: ModelTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 9.1: keyboard interrupt propagates and is not swallowed."""
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "doc.md"
    doc.write_text("# Doc\n\nContent", encoding="utf-8")

    state_path = tmp_path / "state.sqlite"
    config = IngestConfig(roots=(root,), token_budget=512, state_path=state_path)

    # Patch MarkdownExtractor to raise KeyboardInterrupt
    from npu_rag.ingest.extract.markdown import MarkdownExtractor

    def _interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt()

    monkeypatch.setattr(MarkdownExtractor, "extract", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_ingest(config, runtime_tokenizer)


def test_run_ingest_state_store_failure_aborts_before_files(
    tmp_path: Path, runtime_tokenizer: ModelTokenizer
) -> None:
    """Requirement: abort before any file on a state-store failure."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "doc.md").write_text("# Doc", encoding="utf-8")

    # Pass an invalid state_path (e.g. an existing directory)
    bad_state = tmp_path / "bad_db_dir"
    bad_state.mkdir()

    config = IngestConfig(roots=(root,), token_budget=512, state_path=bad_state)
    with pytest.raises(StateError):
        run_ingest(config, runtime_tokenizer)


def test_run_ingest_vision_describer_and_cache(
    tmp_path: Path, runtime_tokenizer: ModelTokenizer
) -> None:
    """Images are resolved via describer and cached in state store."""
    root = tmp_path / "docs"
    root.mkdir()
    img_path = root / "chart.png"
    _write_minimal_png(img_path, 250, 250)

    state_path = tmp_path / "state.sqlite"
    config = IngestConfig(roots=(root,), token_budget=512, state_path=state_path)

    describer = _FakeDescriber("Custom chart description")

    # Run 1: image described
    report1 = run_ingest(config, runtime_tokenizer, describer=describer)
    assert report1.files_processed == 1
    assert describer.call_count == 1
    assert report1.vision_requests_issued == 1
    assert report1.vision_cache_hits == 0

    # Run 2: Change a param so file is re-extracted, but image is served from vision_cache
    config2 = IngestConfig(roots=(root,), token_budget=512, min_page_chars=60, state_path=state_path)
    report2 = run_ingest(config2, runtime_tokenizer, describer=describer)
    assert report2.files_processed == 1
    # describer was NOT called again
    assert describer.call_count == 1
    assert report2.vision_requests_issued == 0
    assert report2.vision_cache_hits == 1
