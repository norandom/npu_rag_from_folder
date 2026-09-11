"""Tests for the ingest run report and human-readable rendering (task 7.1).

Boundary: report.py.
Requirements: 1.4, 2.4, 5.2, 5.3, 8.4, 8.6, 9.2, 9.3, 9.4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npu_rag.ingest.report import omission_counts, render
from npu_rag.ingest.types import (
    ChunkKind,
    ChunkRecord,
    MarkdownLocator,
    Omission,
    OmissionCategory,
    RunReport,
)


def _make_record(chunk_id: str, path: Path) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source_path=path,
        root_id="root-1",
        author="alice",
        title="Test Doc",
        kind=ChunkKind.PROSE,
        text="Sample prose content for record.",
        locator=MarkdownLocator(line_range=(1, 10), ordinal=0),
        ordinal=0,
        token_count=15,
        truncated=False,
        provenance=None,
    )


def test_omission_counts_groups_by_category() -> None:
    """Requirement 9.2: counts by omission category."""
    omissions = (
        Omission(
            category=OmissionCategory.UNSUPPORTED,
            path=Path("docs/data.xyz"),
            reason="unsupported file format",
        ),
        Omission(
            category=OmissionCategory.UNSUPPORTED,
            path=Path("docs/binary.dat"),
            reason="unknown content sniff",
        ),
        Omission(
            category=OmissionCategory.VISION_UNAVAILABLE,
            path=Path("images/fig1.png"),
            reason="detector model missing",
            missing_capability="ocr_det",
        ),
        Omission(
            category=OmissionCategory.FAILED,
            path=Path("broken.pdf"),
            reason="syntax error in trailer",
        ),
        Omission(
            category=OmissionCategory.BELOW_THRESHOLD,
            path=Path("icon.png"),
            reason="shorter side below 200px",
        ),
    )
    report = RunReport(
        files_processed=5,
        files_unchanged=2,
        files_skipped=4,
        files_failed=1,
        omissions=omissions,
        removed_chunk_ids=(),
        no_work_required=False,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=(),
    )

    counts = omission_counts(report)
    assert counts == {
        OmissionCategory.UNSUPPORTED: 2,
        OmissionCategory.VISION_UNAVAILABLE: 1,
        OmissionCategory.FAILED: 1,
        OmissionCategory.BELOW_THRESHOLD: 1,
    }
    # Property on RunReport matches helper function
    assert report.omission_counts == counts


def test_omission_counts_empty_omissions() -> None:
    report = RunReport(
        files_processed=3,
        files_unchanged=0,
        files_skipped=0,
        files_failed=0,
        omissions=(),
        removed_chunk_ids=(),
        no_work_required=False,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=(),
    )
    assert omission_counts(report) == {}
    assert report.omission_counts == {}


def test_render_no_work_required_run() -> None:
    """Requirement 8.6: report that no work was required on an unchanged run."""
    report = RunReport(
        files_processed=0,
        files_unchanged=4,
        files_skipped=0,
        files_failed=0,
        omissions=(),
        removed_chunk_ids=(),
        no_work_required=True,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=(_make_record("c1", Path("a.md")),),
    )

    text = render(report)
    assert "No work required" in text
    assert "Processed: 0" in text
    assert "Unchanged: 4" in text
    assert "Skipped: 0" in text
    assert "Failed: 0" in text
    assert "Total files: 4" in text
    assert "Records retained: 1" in text
    # Also callable as report.render()
    assert report.render() == text


def test_render_mixed_run_observable() -> None:
    """Observable: a rendered report for a mixed run lists every skipped and failed file
    with its reason and states the totals (requirements 1.4, 2.4, 5.2, 5.3, 8.4, 9.2, 9.3, 9.4)."""
    p_unsupported = Path("archive/note.unknown")
    p_ocr = Path("archive/chart.png")
    p_fail = Path("archive/corrupt.pdf")
    p_below = Path("archive/thumb.jpg")
    p_root = Path("missing_root")

    omissions = (
        Omission(
            category=OmissionCategory.ROOT_UNAVAILABLE,
            path=p_root,
            reason="directory does not exist or unreadable",
        ),
        Omission(
            category=OmissionCategory.UNSUPPORTED,
            path=p_unsupported,
            reason="unrecognized file type",
        ),
        Omission(
            category=OmissionCategory.VISION_UNAVAILABLE,
            path=p_ocr,
            reason="ONNX recognizer model absent",
            missing_capability="ocr_rec",
        ),
        Omission(
            category=OmissionCategory.BELOW_THRESHOLD,
            path=p_below,
            reason="image 64x64 below 200px minimum",
        ),
        Omission(
            category=OmissionCategory.FAILED,
            path=p_fail,
            reason="PdfError: damaged startxref header",
        ),
    )

    rec1 = _make_record("c1", Path("archive/good1.md"))
    rec2 = _make_record("c2", Path("archive/good2.md"))

    report = RunReport(
        files_processed=2,
        files_unchanged=3,
        files_skipped=4,
        files_failed=1,
        omissions=omissions,
        removed_chunk_ids=("old_c9", "old_c10"),
        no_work_required=False,
        vision_requests_issued=3,
        vision_cache_hits=5,
        records=(rec1, rec2),
    )

    rendered = render(report)

    # Totals are stated
    assert "Processed: 2" in rendered
    assert "Unchanged: 3" in rendered
    assert "Skipped: 4" in rendered
    assert "Failed: 1" in rendered
    assert "Total files: 10" in rendered

    # Omission categories are counted
    assert "root_unavailable: 1" in rendered
    assert "unsupported: 1" in rendered
    assert "vision_unavailable: 1" in rendered
    assert "below_threshold: 1" in rendered
    assert "failed: 1" in rendered

    # Every skipped and failed input is listed by path and reason
    assert str(p_root) in rendered
    assert "directory does not exist or unreadable" in rendered

    assert str(p_unsupported) in rendered
    assert "unrecognized file type" in rendered

    assert str(p_fail) in rendered
    assert "PdfError: damaged startxref header" in rendered

    assert str(p_below) in rendered
    assert "image 64x64 below 200px minimum" in rendered

    # Requirement 9.4: missing capability named where applicable
    assert str(p_ocr) in rendered
    assert "ONNX recognizer model absent" in rendered
    assert "missing capability: ocr_rec" in rendered

    # Vision requests vs cache hits
    assert "Vision requests issued: 3" in rendered
    assert "Vision cache hits: 5" in rendered

    # Removed chunk IDs reported (requirement 8.4)
    assert "Removed chunk IDs: 2" in rendered
    assert "old_c9" in rendered
    assert "old_c10" in rendered

    # Active records
    assert "Total chunk records: 2" in rendered


def test_render_all_failed_run() -> None:
    """Requirement 9.5: complete a run and produce its report even when every file fails."""
    p1 = Path("bad1.pdf")
    p2 = Path("bad2.xlsx")
    omissions = (
        Omission(category=OmissionCategory.FAILED, path=p1, reason="bad pdf format"),
        Omission(category=OmissionCategory.FAILED, path=p2, reason="corrupt zip archive"),
    )
    report = RunReport(
        files_processed=0,
        files_unchanged=0,
        files_skipped=0,
        files_failed=2,
        omissions=omissions,
        removed_chunk_ids=(),
        no_work_required=False,
        vision_requests_issued=0,
        vision_cache_hits=0,
        records=(),
    )

    rendered = render(report)
    assert "Processed: 0" in rendered
    assert "Failed: 2" in rendered
    assert "Total files: 2" in rendered
    assert str(p1) in rendered
    assert "bad pdf format" in rendered
    assert str(p2) in rendered
    assert "corrupt zip archive" in rendered
    assert "Total chunk records: 0" in rendered
