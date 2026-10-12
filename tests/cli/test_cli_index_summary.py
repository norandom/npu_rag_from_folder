"""Focused tests for the index command summary renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from npu_rag.cli import cmd_index
from tests.cli.test_cli_index import (
    EXPECTED_CAPABILITIES,
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_FAILURE_PATHS,
    EXPECTED_FILE_KEYS,
    EXPECTED_PAYLOAD_KEYS,
    OMISSION_FIXTURE,
    PROBE_SEQ_LEN,
    make_config,
    make_report,
    probe_profile,
)

# --------------------------------------------------------------------------- #
# Requirement 7.5 / 8.3 - the summary payload
# --------------------------------------------------------------------------- #


def build_payload(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    config = kwargs.pop("config", None) or make_config(tmp_path)
    profile = kwargs.pop("profile", None) or probe_profile()
    report = kwargs.pop("report", None) or make_report()
    return cmd_index.summary_payload(
        report,
        ingest_config=cmd_index.build_ingest_config(config, profile=profile),
        config=config,
        profile=profile,
        full_rebuild=kwargs.pop("full_rebuild", False),
        duration_seconds=kwargs.pop("duration_seconds", 1.23456),
    )


def test_payload_keys_are_exactly_the_documented_schema(tmp_path: Path) -> None:
    payload = build_payload(tmp_path)
    assert frozenset(payload) == EXPECTED_PAYLOAD_KEYS
    assert frozenset(cast("dict[str, Any]", payload["files"])) == EXPECTED_FILE_KEYS


def test_every_payload_key_is_snake_case(tmp_path: Path) -> None:
    """Requirement 8.3's payload is consumed by `jq`; casing is part of it."""
    payload = build_payload(tmp_path)
    keys = list(payload) + list(cast("dict[str, Any]", payload["files"]))
    offenders = [key for key in keys if key != key.lower() or "-" in key or " " in key]
    assert offenders == []


def test_file_counts_come_from_the_matching_report_fields(tmp_path: Path) -> None:
    payload = build_payload(tmp_path)
    files = cast("dict[str, int]", payload["files"])
    assert files == {
        "processed": 7,
        "unchanged": 4,
        "skipped": 2,
        "failed": 3,
        "total": 16,
    }


def test_file_total_is_the_sum_of_the_four_statuses(tmp_path: Path) -> None:
    """Cross-check, so a total copied from one field rather than summed dies."""
    report = make_report(processed=5, unchanged=9, skipped=1, failed=6)
    files = cast("dict[str, int]", build_payload(tmp_path, report=report)["files"])
    assert files["total"] == sum(
        files[key] for key in ("processed", "unchanged", "skipped", "failed")
    )
    assert files["total"] == 21


def test_omissions_are_counted_by_category_against_a_literal_table(
    tmp_path: Path,
) -> None:
    payload = build_payload(tmp_path)
    assert payload["omissions_by_category"] == EXPECTED_CATEGORY_COUNTS
    assert payload["omissions_total"] == len(OMISSION_FIXTURE)
    assert payload["omissions_total"] == sum(EXPECTED_CATEGORY_COUNTS.values())


def test_omission_categories_are_rendered_in_sorted_order(tmp_path: Path) -> None:
    """The fixture's given order is not sorted order (see OMISSION_FIXTURE)."""
    given = [category for category, _path, _cap in OMISSION_FIXTURE]
    assert given != sorted(set(given)), "the fixture would not detect a dropped sort"
    payload = build_payload(tmp_path)
    emitted = list(cast("dict[str, int]", payload["omissions_by_category"]))
    assert emitted == sorted(EXPECTED_CATEGORY_COUNTS)


def test_missing_capabilities_are_deduplicated_and_sorted(tmp_path: Path) -> None:
    assert build_payload(tmp_path)["missing_capabilities"] == EXPECTED_CAPABILITIES


def test_failures_name_the_path_and_reason_of_failed_omissions_only(
    tmp_path: Path,
) -> None:
    failures = cast("list[dict[str, str]]", build_payload(tmp_path)["failures"])
    assert [entry["path"] for entry in failures] == EXPECTED_FAILURE_PATHS
    assert all(entry["reason"] == f"reason for {entry['path']}" for entry in failures)


def test_chunk_counts_come_from_the_report(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, report=make_report(records=11, removed=("x",)))
    assert payload["chunk_records"] == 11
    assert payload["chunk_ids_removed"] == 1


def test_the_two_booleans_the_payload_carries_are_both_live(tmp_path: Path) -> None:
    """`no_work_required` from the report, `full_rebuild` from the flag."""
    assert build_payload(tmp_path)["no_work_required"] is False
    assert (
        build_payload(tmp_path, report=make_report(no_work_required=True))[
            "no_work_required"
        ]
        is True
    )
    assert build_payload(tmp_path, full_rebuild=True)["full_rebuild"] is True
    assert build_payload(tmp_path, full_rebuild=False)["full_rebuild"] is False


def test_duration_is_rounded_to_milliseconds(tmp_path: Path) -> None:
    assert cmd_index.DURATION_DIGITS == 3
    payload = build_payload(tmp_path, duration_seconds=1.23456)
    assert payload["duration_seconds"] == 1.235


def test_the_requested_provider_is_never_passed_off_as_the_active_one(
    tmp_path: Path,
) -> None:
    """Note 3.1: `npu|cpu|auto` is a request, not an attribution."""
    payload = build_payload(tmp_path, config=make_config(tmp_path, provider="npu"))
    assert payload["requested_provider"] == "npu"
    assert payload["active_provider"] == cmd_index.NO_EMBEDDING_PROVIDER
    assert payload["active_provider"] != "npu"
    assert payload["embedding_performed"] is False
    assert payload["index_written"] is False


def test_the_model_and_budget_are_reported(tmp_path: Path) -> None:
    profile = probe_profile()
    payload = build_payload(tmp_path, profile=profile)
    assert payload["model_name"] == "gte-modernbert-base"
    assert payload["model_id"] == profile.model_id
    assert payload["token_budget"] == PROBE_SEQ_LEN


# --------------------------------------------------------------------------- #
# Requirement 7.5 - the two renderings
# --------------------------------------------------------------------------- #


def test_the_text_summary_states_every_element_requirement_7_5_asks_for(
    tmp_path: Path,
) -> None:
    text = cmd_index.render_summary(build_payload(tmp_path), as_json=False)
    assert "added or updated): 7" in text
    assert "Unchanged (content hash matched): 4" in text
    assert "Skipped (unsupported or unreadable): 2" in text
    assert "Failed: 3" in text
    assert "Total: 16" in text
    assert "Chunk records: 11" in text
    assert "Duration: 1.235s" in text
    for category, count in EXPECTED_CATEGORY_COUNTS.items():
        assert f"  {category}: {count}" in text


def test_the_text_summary_names_the_missing_capabilities(tmp_path: Path) -> None:
    text = cmd_index.render_summary(build_payload(tmp_path), as_json=False)
    assert "Missing capabilities: det.onnx, rec.onnx" in text


def test_a_run_with_no_omissions_says_so_rather_than_printing_nothing(
    tmp_path: Path,
) -> None:
    payload = build_payload(tmp_path, report=make_report(omissions=(), failed=0))
    text = cmd_index.render_summary(payload, as_json=False)
    assert "Omissions by category (0 total):" in text
    assert "(none)" in text
    assert "Failures:" not in text
    assert "Missing capabilities" not in text


def test_the_text_summary_truncates_a_long_failure_list(tmp_path: Path) -> None:
    assert cmd_index.FAILURE_DETAIL_LIMIT == 10
    many = tuple(("failed", f"bad{n}.md", None) for n in range(14))
    payload = build_payload(tmp_path, report=make_report(omissions=many))
    text = cmd_index.render_summary(payload, as_json=False)
    assert "bad9.md" in text
    assert "bad10.md" not in text
    assert "... and 4 more" in text
    # The JSON rendering keeps all fourteen: the cap is a terminal concern.
    assert len(cast("list[Any]", payload["failures"])) == 14


def test_the_text_summary_names_the_index_and_active_provider(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, config=make_config(tmp_path, provider="cpu"))
    text = cmd_index.render_summary(payload, as_json=False)
    assert str(payload["index_path"]) in text
    assert cmd_index.NO_EMBEDDING_PROVIDER in text


def test_written_and_embedded_statuses_are_visible_in_text(tmp_path: Path) -> None:
    payload = build_payload(tmp_path)
    payload["embedding_performed"] = True
    payload["index_written"] = True
    text = cmd_index.render_summary(payload, as_json=False)
    assert "Embedding performed: yes" in text
    assert "Index written: yes" in text


def test_the_json_rendering_is_valid_json_carrying_the_same_payload(
    tmp_path: Path,
) -> None:
    payload = build_payload(tmp_path)
    rendered = cmd_index.render_summary(payload, as_json=True)
    assert json.loads(rendered) == json.loads(json.dumps(payload))
    # Non-vacuity control for the flag: `as_json` actually selects a rendering.
    text = cmd_index.render_summary(payload, as_json=False)
    assert text.startswith("Index Run Summary")
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_non_ascii_stays_literal_in_the_json_rendering(tmp_path: Path) -> None:
    """Requirement 8.4: a human and `jq` both see Devanagari, not escapes."""
    config = make_config(tmp_path, archive="गरुड")
    rendered = cmd_index.render_summary(build_payload(tmp_path, config=config), as_json=True)
    assert "गरुड" in rendered
    assert "\\u0917" not in rendered


