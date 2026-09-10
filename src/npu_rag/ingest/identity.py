"""Fingerprints and chunk identity (task 2.1).

A parameter fingerprint covers every output-affecting parameter — including
the tokenizer identity, extractor versions, the vision model id, and the
prompt version — as a SHA-256 over canonical JSON. A chunk identifier is
derived only from file-local inputs, so an unrelated file changing cannot
move it.

The vision model id is read from ``IngestConfig`` and the prompt version
arrives as an argument. This module does not import ``vision``: that module
sits to the right in the rank table and the guard would refuse the import.
Extractor versions are likewise an argument so later extractors can pass
them without this module importing ``extract``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.types import ChunkKind, Locator, _locator_to_dict

__all__ = [
    "ParamsFingerprint",
    "chunk_id",
    "content_hash",
    "params_fingerprint",
]

_READ_CHUNK = 1024 * 1024

RelativePath: TypeAlias = Path | str


@dataclass(frozen=True)
class ParamsFingerprint:
    """SHA-256 digest of every parameter that affects a file's output."""

    digest: str


def params_fingerprint(
    config: IngestConfig,
    tokenizer_id: str,
    extractor_versions: Mapping[str, str],
    prompt_version: str,
) -> ParamsFingerprint:
    """Fingerprint of output-affecting parameters (requirements 8.2, 8.5).

    ``prompt_version`` is an argument because it is not a field on
    ``IngestConfig``. ``vision_model`` is taken from ``config``. Neither
    value is imported from the vision module.
    """
    payload = {
        "extractor_versions": dict(extractor_versions),
        "min_image_pixels": config.min_image_pixels,
        "min_page_chars": config.min_page_chars,
        "prompt_version": prompt_version,
        "prose_overlap_tokens": config.prose_overlap_tokens,
        "token_budget": config.token_budget,
        "tokenizer_id": tokenizer_id,
        "vision_model": config.vision_model,
    }
    return ParamsFingerprint(digest=_sha256_hex(_canonical_json(payload)))


def content_hash(path: Path) -> str:
    """SHA-256 of the file's bytes."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def chunk_id(
    root_id: str,
    relative_path: RelativePath,
    kind: ChunkKind,
    locator: Locator,
    ordinal: int,
    params: ParamsFingerprint,
) -> str:
    """Stable identifier derived only from this file's inputs (7.4, 7.5).

    The locator encoding is the same JSON discriminator ``ChunkRecord``
    persists, so identity and the registry speak one locator vocabulary.
    """
    payload = {
        "kind": kind.value,
        "locator": _locator_to_dict(locator),
        "ordinal": ordinal,
        "params": params.digest,
        "relative_path": Path(relative_path).as_posix(),
        "root_id": root_id,
    }
    return _sha256_hex(_canonical_json(payload))


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
