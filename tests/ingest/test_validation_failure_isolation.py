"""Validation 8.4: Failure isolation across a mixed root.

Requirements: 9.1, 9.2, 9.3, 9.5.
Boundary: failure isolation.

Observable:
- A root holding a corrupt PDF, a malformed workbook, and an undecodable image
  among good files: each bad file is a failed omission with its reason naming
  each file's format, the run completes, good files are committed, and the state
  store holds a committed row for every good file and none for the bad ones.
- An all-bad root where every file fails returns a report whose failed count
  equals its file count, with zero emitted records and zero rows in the state store.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.pipeline import run_ingest
from npu_rag.ingest.types import OmissionCategory


def _norm(path: Path | str) -> str:
    """Normalize Windows long path prefix and path separators for comparison."""
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    return text.replace("\\", "/").lower()


def test_failure_isolation_mixed_root(
    tmp_path: Path,
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Observable 8.4 (Scenario A): A root holding a corrupt PDF, a malformed
    workbook, and an undecodable image among good files:
      - The report lists exactly the three bad files under failed with a reason
        naming each file's format.
      - The run completes successfully without aborting.
      - The state store holds a committed row for every good file and none for the bad ones.
      - Good files are chunked and emitted in report.records.
    """
    root = tmp_path / "mixed_archive"
    root.mkdir()
    state_path = tmp_path / "mixed_state.sqlite"

    # 1. Good files: 1 Markdown document, 1 plain text document
    good_doc = root / "guide.md"
    good_doc.write_text(
        "# Engineering Handbook\n\n"
        "This is a comprehensive engineering guide with valid prose.\n\n"
        "## Architecture\n\n"
        "The system isolates errors cleanly.",
        encoding="utf-8",
    )

    good_notes = root / "notes.txt"
    good_notes.write_text(
        "Important release notes and observations.\nEverything is verified.",
        encoding="utf-8",
    )

    # 2. Bad files:
    # Corrupt PDF (invalid header/data structure that fails pdfium)
    bad_pdf = root / "corrupt.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4 corrupt header without valid structure or xref table")

    # Malformed workbook (non-zip payload with .xlsx extension that fails openpyxl)
    bad_excel = root / "malformed.xlsx"
    bad_excel.write_bytes(b"PK\x03\x04\x00\x00corrupt-zip-stream-not-a-real-archive")

    # Undecodable image (broken PNG header/payload that fails PIL Image.open)
    bad_image = root / "undecodable.png"
    bad_image.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorrupt-png-data")

    config = IngestConfig(
        roots=(root,),
        token_budget=512,
        prose_overlap_tokens=32,
        state_path=state_path,
    )

    # Execute ingest pipeline
    report = run_ingest(config, runtime_tokenizer)

    # Verify run completed and file counts
    assert report.files_processed == 2, "Expected exactly the 2 good files to be processed"
    assert report.files_failed == 3, "Expected exactly the 3 bad files to be failed"
    assert report.files_skipped == 0
    assert report.files_unchanged == 0
    assert report.no_work_required is False

    # Verify failed omissions
    failed_omissions = [o for o in report.omissions if o.category == OmissionCategory.FAILED]
    assert len(failed_omissions) == 3, "Expected exactly 3 failed omissions"

    omission_by_norm_path = {_norm(o.path): o for o in failed_omissions}
    assert _norm(bad_pdf) in omission_by_norm_path
    assert _norm(bad_excel) in omission_by_norm_path
    assert _norm(bad_image) in omission_by_norm_path

    # Verify format is named in each reason
    pdf_omission = omission_by_norm_path[_norm(bad_pdf)]
    excel_omission = omission_by_norm_path[_norm(bad_excel)]
    image_omission = omission_by_norm_path[_norm(bad_image)]

    assert "pdf" in pdf_omission.reason.lower(), (
        f"PDF omission reason '{pdf_omission.reason}' must name the PDF format"
    )
    assert any(
        fmt in excel_omission.reason.lower() for fmt in ("workbook", "excel", "xlsx")
    ), f"Excel omission reason '{excel_omission.reason}' must name the workbook/excel format"
    assert any(
        fmt in image_omission.reason.lower() for fmt in ("image", "png")
    ), f"Image omission reason '{image_omission.reason}' must name the image format"

    # Verify state store holds committed rows for good files and NONE for bad ones
    with sqlite3.connect(state_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT relative_path FROM file_state")
        committed_relative_paths = {row[0] for row in cursor.fetchall()}

        cursor.execute("SELECT DISTINCT relative_path FROM chunk_registry")
        registry_relative_paths = {row[0] for row in cursor.fetchall()}

    assert "guide.md" in committed_relative_paths
    assert "notes.txt" in committed_relative_paths
    assert "corrupt.pdf" not in committed_relative_paths
    assert "malformed.xlsx" not in committed_relative_paths
    assert "undecodable.png" not in committed_relative_paths

    assert "guide.md" in registry_relative_paths
    assert "notes.txt" in registry_relative_paths
    assert "corrupt.pdf" not in registry_relative_paths
    assert "malformed.xlsx" not in registry_relative_paths
    assert "undecodable.png" not in registry_relative_paths

    # Verify good files' records were emitted in report.records
    assert len(report.records) >= 2
    emitted_source_paths = {_norm(r.source_path) for r in report.records}
    assert _norm(good_doc) in emitted_source_paths
    assert _norm(good_notes) in emitted_source_paths
    assert _norm(bad_pdf) not in emitted_source_paths
    assert _norm(bad_excel) not in emitted_source_paths
    assert _norm(bad_image) not in emitted_source_paths

    # Verify human-readable report rendering states failed files with their reasons
    rendered = report.render()
    assert "Processed: 2" in rendered
    assert "Failed: 3" in rendered
    assert "corrupt.pdf" in rendered
    assert "malformed.xlsx" in rendered
    assert "undecodable.png" in rendered
    assert pdf_omission.reason in rendered
    assert excel_omission.reason in rendered
    assert image_omission.reason in rendered


def test_failure_isolation_all_bad_root(
    tmp_path: Path,
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Observable 8.4 (Scenario B): An all-bad root returns a report whose
    failed count equals its file count, with zero emitted records and zero
    committed state store rows.
    """
    bad_root = tmp_path / "all_bad_archive"
    bad_root.mkdir()
    state_path = tmp_path / "all_bad_state.sqlite"

    bad_pdf = bad_root / "damaged.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4 completely broken body")

    bad_excel = bad_root / "damaged.xlsx"
    bad_excel.write_bytes(b"PK\x03\x04\x00\x00damaged workbook content")

    bad_image = bad_root / "damaged.png"
    bad_image.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00damaged image data")

    config = IngestConfig(
        roots=(bad_root,),
        token_budget=512,
        state_path=state_path,
    )

    report = run_ingest(config, runtime_tokenizer)

    # Failed count equals total file count (Requirement 9.5)
    total_files = 3
    assert report.files_failed == total_files
    assert report.files_processed == 0
    assert report.files_unchanged == 0
    assert report.files_skipped == 0
    assert report.no_work_required is False
    assert len(report.records) == 0
    assert len(report.omissions) == total_files
    assert all(o.category == OmissionCategory.FAILED for o in report.omissions)

    # State store must be empty
    with sqlite3.connect(state_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM file_state")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT COUNT(*) FROM chunk_registry")
        assert cursor.fetchone()[0] == 0

    # Rendered report check
    rendered = report.render()
    assert f"Failed: {total_files}" in rendered
    assert f"Total files: {total_files}" in rendered
    assert "Processed: 0" in rendered
