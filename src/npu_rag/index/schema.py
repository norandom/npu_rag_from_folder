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


def chunk_record_to_schema(
    record: ChunkRecord,
    vector: Sequence[float],
    *,
    file_hash: str = "",
    headings: Sequence[str] = (),
    created_at: float | None = None,
) -> ChunkSchema:
    """Convert an ingest ChunkRecord and embedding vector to a LanceDB ChunkSchema."""
    if len(vector) != VECTOR_DIMENSION:
        raise SchemaError(
            f"Vector dimension mismatch: expected {VECTOR_DIMENSION}, got {len(vector)}"
        )

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
