"""Unit tests for fingerprints and chunk identity (task 2.1).

Requirements 7.4 and 7.5: a chunk identifier is derived only from file-local
inputs and is identical across runs when those inputs are unchanged.
Requirements 8.2 and 8.5: the parameter fingerprint covers every
output-affecting parameter, so a change to any of them is observable.

``identity.py`` sits to the left of vision in the rank table. The vision
model id and prompt version therefore arrive as arguments (the model id via
``IngestConfig``, the prompt version as its own argument) rather than by
importing the vision module.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.identity import (
    ParamsFingerprint,
    chunk_id,
    content_hash,
    params_fingerprint,
)
from npu_rag.ingest.types import (
    ChunkKind,
    ChunkRecord,
    ImageFileLocator,
    Locator,
    MarkdownLocator,
    PageLocator,
    Provenance,
    SheetLocator,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "ingest" / "identity.py"
)

ROOT = Path("inputs")
BUDGET = 256
TOKENIZER_ID = "Alibaba-NLP/gte-modernbert-base@offline-fixture"
EXTRACTOR_VERSIONS: dict[str, str] = {
    "excel": "1",
    "image": "1",
    "markdown": "1",
    "pdf": "1",
    "text": "1",
}
PROMPT_VERSION = "1"

MARKDOWN_LOCATOR = MarkdownLocator(line_range=(1, 4), ordinal=0)
PAGE_LOCATOR = PageLocator(page=3)
SHEET_LOCATOR = SheetLocator(sheet="Q1", cell_range="A1:C10")
IMAGE_FILE_LOCATOR = ImageFileLocator()

OUTPUT_AFFECTING_CONFIG = (
    ("token_budget", 128),
    ("prose_overlap_tokens", 32),
    ("min_page_chars", 10),
    ("min_image_pixels", 64),
    ("vision_model", "other/vision-model"),
)

UNRELATED_CONFIG = (
    ("roots", (Path("elsewhere"),)),
    ("include", ("**/*.md",)),
    ("exclude", ("**/drafts/**",)),
    ("vision_base_url", "https://example.invalid/v1"),
    ("vision_concurrency", 1),
    ("state_path", Path("tmp") / "other.sqlite"),
)


def make_config(**overrides: object) -> IngestConfig:
    values: dict[str, object] = {
        "roots": (ROOT,),
        "token_budget": BUDGET,
    }
    values.update(overrides)
    return IngestConfig(**values)  # type: ignore[arg-type]


def make_fingerprint(**overrides: object) -> ParamsFingerprint:
    config_keys = {item.name for item in dataclasses.fields(IngestConfig)}
    config_overrides = {
        key: value for key, value in overrides.items() if key in config_keys
    }
    tokenizer_id = overrides.get("tokenizer_id", TOKENIZER_ID)
    extractor_versions = overrides.get("extractor_versions", EXTRACTOR_VERSIONS)
    prompt_version = overrides.get("prompt_version", PROMPT_VERSION)
    assert isinstance(tokenizer_id, str)
    assert isinstance(prompt_version, str)
    if not isinstance(extractor_versions, dict):
        raise TypeError("extractor_versions override must be a dict")
    return params_fingerprint(
        make_config(**config_overrides),
        tokenizer_id,
        extractor_versions,
        prompt_version,
    )


def make_chunk_id(
    *,
    root_id: str = "root-a",
    relative_path: Path | str = Path("alice/notes.md"),
    kind: ChunkKind = ChunkKind.PROSE,
    locator: Locator = MARKDOWN_LOCATOR,
    ordinal: int = 0,
    params: ParamsFingerprint | None = None,
) -> str:
    return chunk_id(
        root_id,
        relative_path,
        kind,
        locator,
        ordinal,
        make_fingerprint() if params is None else params,
    )


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _imported_names(source_text: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source_text)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            assert node.level == 0, "a relative import still reaches a rightward module"
    return names


# --------------------------------------------------------------------------
# params_fingerprint (requirements 8.2, 8.5)
# --------------------------------------------------------------------------


def test_params_fingerprint_is_a_frozen_value_object() -> None:
    """design.md, Domain Model: ParamsFingerprint is a value object."""
    fingerprint = make_fingerprint()
    assert isinstance(fingerprint, ParamsFingerprint)
    assert fingerprint == make_fingerprint()
    with pytest.raises(dataclasses.FrozenInstanceError):
        fingerprint.digest = "mutated"  # type: ignore[misc]


def test_params_fingerprint_is_sha256_over_canonical_json() -> None:
    """design.md, Summary-only components: SHA-256 over canonical JSON."""
    config = make_config()
    versions = {"markdown": "1", "pdf": "2"}
    payload = {
        "extractor_versions": versions,
        "min_image_pixels": config.min_image_pixels,
        "min_page_chars": config.min_page_chars,
        "prompt_version": PROMPT_VERSION,
        "prose_overlap_tokens": config.prose_overlap_tokens,
        "token_budget": config.token_budget,
        "tokenizer_id": TOKENIZER_ID,
        "vision_model": config.vision_model,
    }
    expected = _sha256_hex(_canonical_json(payload))
    fingerprint = params_fingerprint(
        config, TOKENIZER_ID, versions, PROMPT_VERSION
    )
    assert fingerprint.digest == expected
    assert len(fingerprint.digest) == 64
    assert fingerprint.digest == fingerprint.digest.lower()
    int(fingerprint.digest, 16)


def test_an_identical_call_leaves_the_fingerprint_identical() -> None:
    """Observable: changing nothing else leaves it identical."""
    assert make_fingerprint() == make_fingerprint()


@pytest.mark.parametrize("field, value", OUTPUT_AFFECTING_CONFIG)
def test_changing_an_output_affecting_config_field_changes_the_fingerprint(
    field: str, value: object
) -> None:
    """Requirement 8.2 / 8.5: token budget, overlap, routing thresholds,
    and the vision model identifier each enter the fingerprint."""
    baseline = make_fingerprint()
    changed = make_fingerprint(**{field: value})
    assert changed != baseline
    assert changed.digest != baseline.digest


def test_changing_the_prompt_version_changes_the_fingerprint() -> None:
    """Requirement 8.2: prompt version is output-affecting and is not a
    field on IngestConfig, so it arrives as an argument."""
    baseline = make_fingerprint()
    changed = make_fingerprint(prompt_version="2")
    assert changed != baseline


def test_changing_the_tokenizer_identity_changes_the_fingerprint() -> None:
    baseline = make_fingerprint()
    changed = make_fingerprint(tokenizer_id="other-tokenizer@rev")
    assert changed != baseline


def test_changing_an_extractor_version_changes_the_fingerprint() -> None:
    baseline = make_fingerprint()
    changed = make_fingerprint(
        extractor_versions={**EXTRACTOR_VERSIONS, "markdown": "2"}
    )
    assert changed != baseline


def test_extractor_version_key_order_does_not_change_the_fingerprint() -> None:
    """Canonical JSON: sorted keys, so insertion order is not identity."""
    forward = make_fingerprint(
        extractor_versions={"markdown": "1", "pdf": "1"}
    )
    reversed_order = make_fingerprint(
        extractor_versions={"pdf": "1", "markdown": "1"}
    )
    assert forward == reversed_order


@pytest.mark.parametrize("field, value", UNRELATED_CONFIG)
def test_changing_an_unrelated_config_field_leaves_the_fingerprint_identical(
    field: str, value: object
) -> None:
    """Observable: changing nothing listed leaves the fingerprint identical."""
    baseline = make_fingerprint()
    unchanged = make_fingerprint(**{field: value})
    assert unchanged == baseline


# --------------------------------------------------------------------------
# content_hash
# --------------------------------------------------------------------------


def test_content_hash_is_sha256_of_the_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    data = b"hello\n"
    path.write_bytes(data)
    assert content_hash(path) == _sha256_hex(data)


def test_content_hash_depends_on_bytes_not_on_the_path_name(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    assert content_hash(first) == content_hash(second)
    second.write_bytes(b"different")
    assert content_hash(first) != content_hash(second)


def test_content_hash_reads_a_non_ascii_path(tmp_path: Path) -> None:
    path = tmp_path / "åse.txt"
    path.write_bytes(b"x")
    assert content_hash(path) == _sha256_hex(b"x")


def test_an_empty_file_hashes_as_sha256_of_no_bytes(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    assert content_hash(path) == _sha256_hex(b"")


# --------------------------------------------------------------------------
# chunk_id (requirements 7.4, 7.5)
# --------------------------------------------------------------------------


def test_the_same_inputs_on_two_runs_yield_the_same_chunk_id() -> None:
    """Requirement 7.5: identical inputs produce the same identifier."""
    assert make_chunk_id() == make_chunk_id()


def test_changing_an_unrelated_file_leaves_chunk_identifiers_unchanged(
    tmp_path: Path,
) -> None:
    """Requirement 7.4: chunk_id is derived only from file-local inputs.

    An unrelated file's bytes change, which is observable on that file's
    content hash, and no chunk identifier of the first file moves.
    """
    source = tmp_path / "alice" / "notes.md"
    other = tmp_path / "bob" / "other.md"
    source.parent.mkdir()
    other.parent.mkdir()
    source.write_bytes(b"hello")
    other.write_bytes(b"unrelated")
    params = make_fingerprint()
    before = {
        "prose": chunk_id(
            "root-a",
            Path("alice/notes.md"),
            ChunkKind.PROSE,
            MARKDOWN_LOCATOR,
            0,
            params,
        ),
        "figure": chunk_id(
            "root-a",
            Path("alice/notes.md"),
            ChunkKind.FIGURE,
            IMAGE_FILE_LOCATOR,
            1,
            params,
        ),
    }
    other_hash_before = content_hash(other)

    other.write_bytes(b"unrelated changed")

    assert content_hash(other) != other_hash_before
    assert content_hash(source) == _sha256_hex(b"hello")
    assert (
        chunk_id(
            "root-a",
            Path("alice/notes.md"),
            ChunkKind.PROSE,
            MARKDOWN_LOCATOR,
            0,
            params,
        )
        == before["prose"]
    )
    assert (
        chunk_id(
            "root-a",
            Path("alice/notes.md"),
            ChunkKind.FIGURE,
            IMAGE_FILE_LOCATOR,
            1,
            params,
        )
        == before["figure"]
    )


@pytest.mark.parametrize(
    "override",
    [
        {"root_id": "root-b"},
        {"relative_path": Path("bob/notes.md")},
        {"kind": ChunkKind.TABLE},
        {"locator": PAGE_LOCATOR},
        {"ordinal": 1},
    ],
    ids=["root_id", "relative_path", "kind", "locator", "ordinal"],
)
def test_changing_a_file_local_input_changes_the_chunk_id(
    override: dict[str, object],
) -> None:
    assert make_chunk_id(**override) != make_chunk_id()  # type: ignore[arg-type]


def test_changing_params_changes_the_chunk_id() -> None:
    baseline = make_chunk_id()
    changed = make_chunk_id(params=make_fingerprint(token_budget=128))
    assert changed != baseline


@pytest.mark.parametrize(
    "locator",
    [MARKDOWN_LOCATOR, PAGE_LOCATOR, SHEET_LOCATOR, IMAGE_FILE_LOCATOR],
    ids=["markdown", "page", "sheet", "image_file"],
)
def test_chunk_id_reuses_the_chunk_record_locator_encoding(locator: Locator) -> None:
    """Implementation note 1.2: locators have a JSON discriminator via
    ChunkRecord; chunk_id reuses that encoding rather than inventing one."""
    params = make_fingerprint()
    relative_path = Path("alice/notes.md")
    record = ChunkRecord(
        chunk_id="unused",
        source_path=relative_path,
        root_id="root-a",
        author="alice",
        title=None,
        kind=ChunkKind.PROSE if locator is not IMAGE_FILE_LOCATOR else ChunkKind.FIGURE,
        text="hello",
        locator=locator,
        ordinal=0,
        token_count=8,
        truncated=False,
        provenance=(
            None
            if locator is not IMAGE_FILE_LOCATOR
            else Provenance(vision_model="m", prompt_version="1")
        ),
    )
    encoded_locator = json.loads(record.to_json())["locator"]
    payload = {
        "kind": record.kind.value,
        "locator": encoded_locator,
        "ordinal": record.ordinal,
        "params": params.digest,
        "relative_path": relative_path.as_posix(),
        "root_id": record.root_id,
    }
    expected = _sha256_hex(_canonical_json(payload))
    assert (
        chunk_id(
            record.root_id,
            relative_path,
            record.kind,
            locator,
            record.ordinal,
            params,
        )
        == expected
    )


def test_chunk_id_is_stable_across_path_separator_spellings() -> None:
    """relative_path is file-local identity, not a platform string."""
    from_string = make_chunk_id(relative_path="alice/notes.md")
    from_path = make_chunk_id(relative_path=Path("alice") / "notes.md")
    assert from_string == from_path


def test_a_non_ascii_relative_path_is_stable() -> None:
    first = make_chunk_id(relative_path=Path("åse/rapport.md"))
    second = make_chunk_id(relative_path=Path("åse/rapport.md"))
    assert first == second
    assert first != make_chunk_id(relative_path=Path("ase/rapport.md"))


# --------------------------------------------------------------------------
# Boundary guard
# --------------------------------------------------------------------------


def test_identity_does_not_import_rightward_modules() -> None:
    """design.md, Architecture: identity may import config and types
    (leftward). It must not import state, vision, extract, or pipeline.
    The vision model id and prompt version arrive as arguments because
    vision sits to the right and the guard would refuse the import.
    """
    imported = _imported_names(MODULE_PATH.read_text(encoding="utf-8"))
    project = [name for name in imported if name.startswith("npu_rag")]
    allowed_prefixes = (
        "npu_rag.ingest.types",
        "npu_rag.ingest.config",
        "npu_rag.ingest.errors",
        "npu_rag.ingest.credential",
    )
    forbidden = [
        name
        for name in project
        if not any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in allowed_prefixes
        )
    ]
    assert forbidden == []
    joined = "\n".join(imported)
    assert "npu_rag.ingest.vision" not in joined
    assert "npu_rag.ingest.state" not in joined
    assert "npu_rag.ingest.extract" not in joined
    assert "npu_rag.ingest.pipeline" not in joined
    assert "npu_rag.ingest.types" in joined
    assert "npu_rag.ingest.config" in joined
