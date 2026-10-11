"""LanceDB schema definitions and record conversion for vector-index."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from lancedb.pydantic import LanceModel, Vector

from npu_rag.index.errors import SchemaError
from npu_rag.ingest.types import (
    ChunkKind,
    ChunkRecord,
    Locator,
    Provenance,
    _locator_from_dict,
    _locator_to_dict,
)

VECTOR_DIMENSION: int = 768

#: Separator between the model repository id and the vector width inside a
#: stored identity. A Hugging Face repository id never contains a colon, so the
#: stamp stays unambiguous and a malformed one is detectable rather than merely
#: odd-looking.
MODEL_IDENTITY_SEPARATOR: str = ":"

#: What ``ChunkSchema.model_identity`` holds when no caller supplied one.
#:
#: The empty string, matching ``root_id`` and ``author``, which are the other
#: string columns with a "nothing was supplied" state. It is deliberately a
#: value and not ``None``: a nullable column would make "unstamped" and "the
#: writer wrote null on purpose" indistinguishable, and every consumer would
#: have to re-derive the same ``or ""`` fallback.
UNSTAMPED_MODEL_IDENTITY: str = ""


def format_model_identity(model_id: str, dimension: int = VECTOR_DIMENSION) -> str:
    """Build the identity stamped on every row embedded by one model.

    ``model_id`` is the Hugging Face repository id, spelled exactly as
    ``npu_rag.embedding.profiles.ModelProfile.model_id`` spells it (for example
    ``google/embeddinggemma-300m``); ``dimension`` is the width of the vectors
    actually stored.

    What the stamp contains, and why:

    * **the repository id**, because it is the single value that selects the
      weights, the pooling rule and the presence of a dense projection stage -
      everything that defines which vector space a row lives in;
    * **the vector width**, because one repository can legitimately be stored
      at a truncated Matryoshka width, and two widths of the same model are two
      vector spaces. It also makes a stamp read back out of an index
      self-describing, which matters precisely when the process that wrote it
      is long gone.

    What the stamp deliberately **omits** is the artifact revision or commit
    sha. It moves when a README moves, so including it would reject indexes
    that are perfectly comparable: it identifies a download, not a vector
    space. Pooling and the dense stage are omitted for the opposite reason -
    they are determined by the repository id through the profile registry, so
    recording them adds no discriminating power and only adds ways for two
    stamps of the same model to disagree.
    """
    cleaned = model_id.strip()
    if not cleaned:
        raise SchemaError(
            "A model identity needs a non-empty model id; pass "
            "UNSTAMPED_MODEL_IDENTITY explicitly if the intent is 'unknown'"
        )
    if MODEL_IDENTITY_SEPARATOR in cleaned:
        raise SchemaError(
            f"Model id must not contain {MODEL_IDENTITY_SEPARATOR!r}, which "
            f"separates the id from the dimension: {model_id!r}"
        )
    if dimension <= 0:
        raise SchemaError(
            f"A model identity needs a positive vector dimension, got {dimension}"
        )
    return f"{cleaned}{MODEL_IDENTITY_SEPARATOR}{dimension}"


def validate_model_identity(model_identity: str) -> str:
    """Return ``model_identity`` if it is storable, else raise ``SchemaError``.

    Exactly two shapes are storable: ``UNSTAMPED_MODEL_IDENTITY``, which states
    that no identity was supplied, and a well-formed
    ``<model_id><sep><dimension>`` stamp as :func:`format_model_identity`
    builds it. Anything else - a bare repository id, a blank-but-not-empty
    string, a trailing separator - is rejected at the write, because a stamp
    that names no model is worse than no stamp: it reads as evidence.
    """
    if model_identity == UNSTAMPED_MODEL_IDENTITY:
        return UNSTAMPED_MODEL_IDENTITY
    if not model_identity.strip():
        raise SchemaError(
            "A blank model identity would be stored as a stamp that names no "
            f"model: {model_identity!r}. Use UNSTAMPED_MODEL_IDENTITY ('') to "
            "state that no identity was supplied."
        )
    model_part, _, dimension_part = model_identity.partition(
        MODEL_IDENTITY_SEPARATOR
    )
    # A missing separator needs no term of its own. A `not separator` test was
    # written here first; it is REACHABLE - a bare repository id partitions to
    # `separator == ""` and the branch does execute - but it is REDUNDANT,
    # because `partition` leaves `dimension_part` empty in exactly that case and
    # `"".isdigit()` is False, so the `isdigit` term below rejects every input
    # the separator term would have. It was removed as redundant rather than
    # kept as a term no test could distinguish.
    if not model_part.strip() or not dimension_part.isdigit():
        raise SchemaError(
            f"Malformed model identity {model_identity!r}: expected "
            f"'<model_id>{MODEL_IDENTITY_SEPARATOR}<dimension>', as "
            "format_model_identity builds it"
        )
    return model_identity


class ChunkSchema(LanceModel):
    """LanceDB table schema for stored chunks, metadata, and embeddings."""

    chunk_id: str
    file_path: str
    file_hash: str
    chunk_index: int
    chunk_kind: str
    title: str | None = None
    headings: list[str] = []
    locator_type: str
    locator_json: str
    text: str
    vector: Vector(VECTOR_DIMENSION)  # type: ignore[valid-type]
    token_count: int
    created_at: float
    root_id: str = ""
    author: str = ""
    truncated: bool = False
    provenance_json: str | None = None
    #: Which embedding model produced ``vector``, as
    #: :func:`format_model_identity` spells it. Per-row rather than a sidecar
    #: manifest because the condition that must be detectable is "this table
    #: holds vectors from more than one model", and only a per-row value can
    #: answer that - or say which rows to delete afterwards.
    model_identity: str = UNSTAMPED_MODEL_IDENTITY


def chunk_record_to_schema(
    record: ChunkRecord,
    vector: Sequence[float],
    *,
    file_hash: str = "",
    headings: Sequence[str] = (),
    created_at: float | None = None,
    model_identity: str = UNSTAMPED_MODEL_IDENTITY,
) -> ChunkSchema:
    """Convert an ingest ChunkRecord and embedding vector to a LanceDB ChunkSchema.

    ``model_identity`` defaults to unstamped because this is a pure converter
    and the write policy does not live here: ``IndexSynchronizer.sync`` is the
    only production writer and is where the compatibility check is enforced.
    """
    if len(vector) != VECTOR_DIMENSION:
        raise SchemaError(
            f"Vector dimension mismatch: expected {VECTOR_DIMENSION}, got {len(vector)}"
        )
    identity = validate_model_identity(model_identity)

    loc_dict = _locator_to_dict(record.locator)
    locator_type = str(loc_dict.get("type", "unknown"))
    locator_json = json.dumps(loc_dict)

    prov_json: str | None = None
    if record.provenance is not None:
        prov_json = json.dumps(
            {
                "vision_model": record.provenance.vision_model,
                "prompt_version": record.provenance.prompt_version,
            }
        )

    return ChunkSchema(
        chunk_id=record.chunk_id,
        file_path=str(record.source_path),
        file_hash=file_hash,
        chunk_index=record.ordinal,
        chunk_kind=record.kind.value,
        title=record.title,
        headings=list(headings),
        locator_type=locator_type,
        locator_json=locator_json,
        text=record.text,
        vector=list(vector),
        token_count=record.token_count,
        created_at=created_at if created_at is not None else time.time(),
        root_id=record.root_id,
        author=record.author,
        truncated=record.truncated,
        provenance_json=prov_json,
        model_identity=identity,
    )


def schema_to_chunk_record(schema: ChunkSchema | dict[str, Any]) -> ChunkRecord:
    """Reconstruct an ingest ChunkRecord from a LanceDB ChunkSchema or result dict."""
    if isinstance(schema, ChunkSchema):
        data = schema.model_dump()
    elif isinstance(schema, dict):
        data = schema
    else:
        raise SchemaError(f"Expected ChunkSchema or dict, got {type(schema).__name__}")

    try:
        loc_dict = json.loads(data["locator_json"])
        locator: Locator = _locator_from_dict(loc_dict)

        provenance: Provenance | None = None
        prov_raw = data.get("provenance_json")
        if prov_raw:
            prov_dict = json.loads(prov_raw)
            provenance = Provenance(
                vision_model=str(prov_dict["vision_model"]),
                prompt_version=str(prov_dict["prompt_version"]),
            )

        return ChunkRecord(
            chunk_id=str(data["chunk_id"]),
            source_path=Path(str(data["file_path"])),
            root_id=str(data.get("root_id", "")),
            author=str(data.get("author", "")),
            title=str(data["title"]) if data.get("title") is not None else None,
            kind=ChunkKind(str(data["chunk_kind"])),
            text=str(data["text"]),
            locator=locator,
            ordinal=int(data["chunk_index"]),
            token_count=int(data["token_count"]),
            truncated=bool(data.get("truncated", False)),
            provenance=provenance,
        )
    except Exception as exc:
        raise SchemaError(f"Failed to reconstruct ChunkRecord: {exc}") from exc
