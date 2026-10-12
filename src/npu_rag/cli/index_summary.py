"""Index command summary in text and JSON."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from npu_rag.cli import render

if TYPE_CHECKING:
    from collections.abc import Mapping
    from npu_rag.cli.config import CliConfig
    from npu_rag.embedding.profiles import ModelProfile
    from npu_rag.ingest.config import IngestConfig
    from npu_rag.ingest.types import RunReport

DURATION_DIGITS: Final = 3
FAILURE_DETAIL_LIMIT: Final = 10
NO_EMBEDDING_PROVIDER: Final = "none (no chunks needed embedding)"

def summary_payload(
    report: RunReport,
    *,
    ingest_config: IngestConfig,
    config: CliConfig,
    profile: ModelProfile,
    full_rebuild: bool,
    duration_seconds: float,
    active_provider: str = NO_EMBEDDING_PROVIDER,
    embedding_performed: bool = False,
    index_written: bool = False,
) -> dict[str, object]:
    """Requirement 7.5's summary as one snake_case mapping (requirement 8.3).

    Two places where `RunReport`'s surface and the requirement's wording do not
    coincide, resolved in the report's favour because inventing a field would
    be worse than naming the limit:

    - **added vs updated.** `RunReport` counts `files_processed`; the state
      store's `FileStatus` distinction between a new and a changed file is not
      carried out of the pipeline, so one number covers both and is labelled as
      such. `files_unchanged` is requirement 7.2's content-hash skip and
      `files_skipped` is discovery's and the router's - two different things
      that the single word "skipped" would merge.
    - **chunks created.** `report.records` is *every* current chunk record: the
      ones this run chunked and the ones re-emitted unchanged from the state
      store. Nothing distinguishes them, so the key is `chunk_records` and the
      count of ids this run retired is reported beside it.

    `failures` carries the path and reason of every `FAILED` omission, and only
    those. Category counts are what requirement 7.5 asks for, but "Failed: 1"
    with no reason leaves the user nothing to act on, and a failure is the one
    category that is always actionable - the others say what the archive
    contains or what capability is missing, which the counts and
    `missing_capabilities` already report.
    """
    from npu_rag.ingest.types import OmissionCategory

    counts = report.omission_counts
    capabilities = sorted(
        {
            omission.missing_capability
            for omission in report.omissions
            if omission.missing_capability
        }
    )
    return {
        "command": "index",
        "archive_path": str(ingest_config.roots[0]),
        "index_path": str(config.index_path.expanduser().resolve()),
        "state_path": str(ingest_config.state_path),
        "model_name": config.model_name,
        "model_id": profile.model_id,
        "token_budget": ingest_config.token_budget,
        "requested_provider": config.provider,
        "active_provider": active_provider,
        "full_rebuild": full_rebuild,
        "embedding_performed": embedding_performed,
        "index_written": index_written,
        "files": {
            "processed": report.files_processed,
            "unchanged": report.files_unchanged,
            "skipped": report.files_skipped,
            "failed": report.files_failed,
            "total": (
                report.files_processed
                + report.files_unchanged
                + report.files_skipped
                + report.files_failed
            ),
        },
        "chunk_records": len(report.records),
        "chunk_ids_removed": len(report.removed_chunk_ids),
        "no_work_required": report.no_work_required,
        "omissions_total": len(report.omissions),
        "omissions_by_category": {
            category.value: counts[category]
            for category in sorted(counts, key=lambda item: item.value)
        },
        "missing_capabilities": capabilities,
        "failures": [
            {"path": str(omission.path), "reason": omission.reason}
            for omission in report.omissions
            if omission.category is OmissionCategory.FAILED
        ],
        "duration_seconds": round(duration_seconds, DURATION_DIGITS),
    }


def _yes_no(flag: bool) -> str:  # noqa: FBT001 - a formatter, not a behaviour switch
    return "yes" if flag else "no"


def _summary_text(payload: Mapping[str, object]) -> str:
    """Render the payload as the terminal summary of requirement 7.5.

    Reads the payload rather than the `RunReport` so the two renderings cannot
    disagree about a count: there is one accounting, and `--json` and the text
    are two views of it.
    """
    files = payload["files"]
    assert isinstance(files, dict)  # noqa: S101 - narrows the payload's own shape
    categories = payload["omissions_by_category"]
    assert isinstance(categories, dict)  # noqa: S101
    capabilities = payload["missing_capabilities"]
    assert isinstance(capabilities, list)  # noqa: S101
    failures = payload["failures"]
    assert isinstance(failures, list)  # noqa: S101

    lines = [
        "Index Run Summary",
        "=" * len("Index Run Summary"),
        f"Archive:  {payload['archive_path']}",
        f"Index:    {payload['index_path']}",
        f"State:    {payload['state_path']}",
        f"Model:    {payload['model_name']} (token budget {payload['token_budget']})",
        f"Provider: requested {payload['requested_provider']}, "
        f"active {payload['active_provider']}",
        f"Embedding performed: {_yes_no(bool(payload['embedding_performed']))}",
        f"Index written: {_yes_no(bool(payload['index_written']))}",
        f"Full rebuild requested: {_yes_no(bool(payload['full_rebuild']))}",
        "",
        "Files:",
        f"  Processed (added or updated): {files['processed']}",
        f"  Unchanged (content hash matched): {files['unchanged']}",
        f"  Skipped (unsupported or unreadable): {files['skipped']}",
        f"  Failed: {files['failed']}",
        f"  Total: {files['total']}",
        "",
        "Chunks:",
        f"  Chunk records: {payload['chunk_records']}",
        f"  Chunk ids removed: {payload['chunk_ids_removed']}",
        f"  No work required: {_yes_no(bool(payload['no_work_required']))}",
        "",
        f"Omissions by category ({payload['omissions_total']} total):",
    ]
    if categories:
        lines.extend(f"  {name}: {count}" for name, count in categories.items())
    else:
        lines.append("  (none)")
    if capabilities:
        lines.append(f"  Missing capabilities: {', '.join(str(c) for c in capabilities)}")
    if failures:
        lines.extend(["", "Failures:"])
        for entry in failures[:FAILURE_DETAIL_LIMIT]:
            assert isinstance(entry, dict)  # noqa: S101
            lines.append(f"  {entry['path']}: {entry['reason']}")
        remaining = len(failures) - FAILURE_DETAIL_LIMIT
        if remaining > 0:
            lines.append(f"  ... and {remaining} more (use --json for all of them)")
    lines.extend(["", f"Duration: {payload['duration_seconds']}s"])
    return "\n".join(lines)


def render_summary(payload: Mapping[str, object], *, as_json: bool) -> str:
    """The summary in whichever rendering requirement 8.3's flag selected.

    `ensure_ascii=False` matches `render.render_search_json`: a path carrying
    Devanagari stays literal for a human and for `jq`, which is why the write
    has to go through `render.emit` (requirement 8.4).
    """
    if as_json:
        return json.dumps(payload, ensure_ascii=False, indent=render.JSON_INDENT)
    return _summary_text(payload)


