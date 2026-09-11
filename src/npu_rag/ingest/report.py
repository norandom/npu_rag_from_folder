"""RunReport aggregation and human-readable text rendering (task 7.1).

This module sits at the report layer (same rank as chunking) in design.md's
dependency direction — ``types, errors -> config -> ... -> chunk, report ->
pipeline`` — and imports only from ``types``.

Requirements: 1.4, 2.4, 5.2, 5.3, 8.4, 8.6, 9.2, 9.3, 9.4.
"""

from __future__ import annotations

from npu_rag.ingest.types import (
    ChunkRecord,
    Omission,
    OmissionCategory,
    RunReport,
)

__all__ = [
    "RunReport",
    "omission_counts",
    "render",
]


def omission_counts(report: RunReport) -> dict[OmissionCategory, int]:
    """Return counts of omissions in the report grouped by OmissionCategory."""
    return report.omission_counts


def render(report: RunReport) -> str:
    """Render a human-readable text summary of an ingest RunReport.

    States file counts by status (processed, unchanged, skipped, failed),
    omission counts by category, lists every omission by path and reason
    (explicitly naming any missing capability, e.g. OCR model weights),
    vision requests issued versus cache hits, removed chunk IDs,
    records retained/emitted, and the no-work-required flag.
    """
    lines: list[str] = []
    lines.append("Ingest Run Report")
    lines.append("=================")

    # 1. File Status & Totals (requirements 9.2, 9.5)
    total_files = (
        report.files_processed
        + report.files_unchanged
        + report.files_skipped
        + report.files_failed
    )
    lines.append("Status:")
    lines.append(f"  Processed: {report.files_processed}")
    lines.append(f"  Unchanged: {report.files_unchanged}")
    lines.append(f"  Skipped: {report.files_skipped}")
    lines.append(f"  Failed: {report.files_failed}")
    lines.append(f"  Total files: {total_files}")

    # 2. Work & Records (requirements 8.4, 8.6)
    lines.append("")
    lines.append("Work & Records:")
    lines.append(f"  No work required: {report.no_work_required}")
    if report.no_work_required:
        lines.append(f"  Records retained: {len(report.records)}")
    else:
        lines.append(f"  Total chunk records: {len(report.records)}")

    # Removed chunk IDs
    lines.append(f"  Removed chunk IDs: {len(report.removed_chunk_ids)}")
    if report.removed_chunk_ids:
        for cid in sorted(report.removed_chunk_ids):
            lines.append(f"    - {cid}")

    # 3. Vision metrics (requirements 5.2, 5.3)
    lines.append("")
    lines.append("Vision:")
    lines.append(f"  Vision requests issued: {report.vision_requests_issued}")
    lines.append(f"  Vision cache hits: {report.vision_cache_hits}")

    # 4. Omission counts by category (requirement 9.2)
    lines.append("")
    counts = omission_counts(report)
    lines.append("Omissions by category:")
    if counts:
        for cat, count in sorted(counts.items(), key=lambda item: item[0].value):
            lines.append(f"  {cat.value}: {count}")
    else:
        lines.append("  (none)")

    # 5. Omission details by path and reason, with missing capability named
    # (requirements 1.4, 2.4, 5.2, 9.3, 9.4)
    lines.append("")
    lines.append("Omission details:")
    if report.omissions:
        for omission in report.omissions:
            extra = (
                f" (missing capability: {omission.missing_capability})"
                if omission.missing_capability
                else ""
            )
            lines.append(
                f"  - {omission.path} [{omission.category.value}]{extra}: {omission.reason}"
            )
    else:
        lines.append("  (none)")

    return "\n".join(lines)


# Wire the report rendering implementation to RunReport.render()
setattr(RunReport, "_render_fn", staticmethod(render))
