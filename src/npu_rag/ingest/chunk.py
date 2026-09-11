"""Chunking against the token budget (tasks 6.1 - 6.3).

This module turns extracted segments into ChunkRecords that fit within the
active model's token limit. Every candidate is measured with the document title
attached under the DOCUMENT text kind (requirement 6.2).

Overlap is applied to prose chunks only; tables, spreadsheet blocks, formulas,
and figure segments never overlap (OVERLAP_POLICY).

Precondition:
- budget == tokenizer.max_input_tokens is asserted at construction.
"""

from __future__ import annotations

from typing import Final

from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.embedding.types import DocumentText, TextKind
from npu_rag.ingest.identity import ParamsFingerprint, chunk_id
from npu_rag.ingest.types import (
    ChunkKind,
    ChunkRecord,
    Extracted,
    Locator,
    Provenance,
    SourceFile,
)

__all__ = [
    "OVERLAP_POLICY",
    "Chunker",
]

#: Overlap policy per chunk kind (requirement 6.5, design.md).
#: Only prose chunks apply overlap; all other kinds pass through without overlap.
OVERLAP_POLICY: Final[dict[ChunkKind, bool]] = {
    ChunkKind.PROSE: True,
    ChunkKind.TABLE: False,
    ChunkKind.BLOCK: False,
    ChunkKind.FORMULA: False,
    ChunkKind.FIGURE: False,
}


class Chunker:
    """Chunk segments against a token budget using the embedding runtime's tokenizer."""

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        *,
        budget: int,
        prose_overlap_tokens: int,
    ) -> None:
        if budget != tokenizer.max_input_tokens:
            raise ValueError(
                f"budget ({budget}) must match tokenizer compiled sequence length "
                f"({tokenizer.max_input_tokens})"
            )
        if prose_overlap_tokens < 0 or prose_overlap_tokens >= budget:
            raise ValueError(
                f"prose_overlap_tokens ({prose_overlap_tokens}) must be non-negative "
                f"and less than budget ({budget})"
            )
        self._tokenizer = tokenizer
        self._budget = budget
        self._overlap = prose_overlap_tokens

    @property
    def budget(self) -> int:
        """Configured token budget (equals tokenizer.max_input_tokens)."""
        return self._budget

    @property
    def overlap(self) -> int:
        """Configured prose overlap in tokens."""
        return self._overlap

    def measure(self, text: str, title: str | None) -> int:
        """Count tokens with document title attached under TextKind.DOCUMENT (requirement 6.2)."""
        doc = (
            DocumentText(content=text, title=title)
            if title is not None
            else DocumentText(content=text)
        )
        return self._tokenizer.count_tokens(doc, TextKind.DOCUMENT)

    def exceeds_budget(self, text: str, title: str | None) -> bool:
        """Whether text with title attached exceeds the token budget (requirement 6.3)."""
        return self.measure(text, title) > self._budget

    def assemble_record(
        self,
        *,
        source: SourceFile,
        title: str | None,
        kind: ChunkKind,
        text: str,
        locator: Locator,
        ordinal: int,
        truncated: bool = False,
        provenance: Provenance | None = None,
        params: ParamsFingerprint,
    ) -> ChunkRecord:
        """Assemble a single validated ChunkRecord with stable chunk_id."""
        count = self.measure(text, title)
        cid = chunk_id(
            root_id=source.root.as_posix(),
            relative_path=source.relative_path,
            kind=kind,
            locator=locator,
            ordinal=ordinal,
            params=params,
        )
        return ChunkRecord(
            chunk_id=cid,
            source_path=source.path,
            root_id=source.root.as_posix(),
            author=source.author,
            title=title,
            kind=kind,
            text=text,
            locator=locator,
            ordinal=ordinal,
            token_count=count,
            truncated=truncated,
            provenance=provenance,
        )

    def chunk(
        self,
        source: SourceFile,
        extracted: Extracted,
        *,
        params: ParamsFingerprint,
    ) -> tuple[ChunkRecord, ...]:
        """Convert extracted segments into ChunkRecords (implemented across 6.2/6.3)."""
        # Staged for 6.2 and 6.3
        return ()
