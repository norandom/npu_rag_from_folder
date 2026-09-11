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

from collections.abc import Sequence
from typing import Final

from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.embedding.types import DocumentText, TextKind
from npu_rag.ingest.identity import ParamsFingerprint, chunk_id
from npu_rag.ingest.types import (
    BlockSegment,
    ChunkKind,
    ChunkRecord,
    Extracted,
    FigureSegment,
    FormulaSegment,
    Locator,
    MarkdownLocator,
    ProseSegment,
    Provenance,
    SourceFile,
    TableSegment,
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


def _combine_locators(locators: Sequence[Locator]) -> Locator:
    """Combine a non-empty sequence of locators into a single enclosing locator."""
    if not locators:
        raise ValueError("locators cannot be empty")
    first = locators[0]
    if isinstance(first, MarkdownLocator):
        last = locators[-1]
        last_range = (
            last.line_range
            if isinstance(last, MarkdownLocator)
            else first.line_range
        )
        return MarkdownLocator(
            line_range=(first.line_range[0], last_range[1]),
            ordinal=first.ordinal,
        )
    return first


def _format_block(
    label: str,
    header_row: tuple[str, ...],
    rows: Sequence[tuple[str, ...]],
) -> str:
    """Format a BlockSegment as text with label, header row and data rows."""
    lines: list[str] = []
    if label:
        lines.append(label)
    if header_row:
        lines.append(" | ".join(header_row))
    for row in rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


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

    def _truncate_to_budget(self, text: str, title: str | None) -> str:
        """Truncate text at a token boundary so that measure(truncated, title) <= budget."""
        ids = self._tokenizer.tokenizer.encode(text, add_special_tokens=False)
        low = 0
        high = len(ids)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            candidate = str(
                self._tokenizer.tokenizer.decode(
                    ids[:mid], skip_special_tokens=True
                )
            )
            if self.measure(candidate, title) <= self._budget:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _get_overlap_text(self, text: str) -> str | None:
        """Extract the trailing prose_overlap_tokens from text as a prefix for the next chunk."""
        if self._overlap <= 0:
            return None
        ids = self._tokenizer.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return None
        overlap_len = min(len(ids), self._overlap)
        overlap_ids = ids[-overlap_len:]
        return str(
            self._tokenizer.tokenizer.decode(
                overlap_ids, skip_special_tokens=True
            )
        )

    def split_block(
        self,
        segment: BlockSegment,
        title: str | None,
    ) -> list[tuple[str, bool]]:
        """Split an oversize block by rows, repeating label and header in every piece (4.3).

        Returns a list of (piece_text, truncated) pairs.
        Each piece begins with label and header_row, and together they cover every row once.
        """
        full_text = _format_block(
            segment.label, segment.header_row, segment.rows
        )
        if self.measure(full_text, title) <= self._budget:
            return [(full_text, False)]

        pieces: list[tuple[str, bool]] = []
        current_rows: list[tuple[str, ...]] = []

        for row in segment.rows:
            candidate_rows = current_rows + [row]
            candidate_text = _format_block(
                segment.label, segment.header_row, candidate_rows
            )
            if self.measure(candidate_text, title) <= self._budget:
                current_rows.append(row)
            else:
                if current_rows:
                    piece_text = _format_block(
                        segment.label, segment.header_row, current_rows
                    )
                    pieces.append((piece_text, False))
                    current_rows = []

                # Test if single row fits with label and header
                single_row_text = _format_block(
                    segment.label, segment.header_row, [row]
                )
                if self.measure(single_row_text, title) <= self._budget:
                    current_rows = [row]
                else:
                    # Single row exceeds budget: truncate indivisible unit
                    piece_text = self._truncate_to_budget(
                        single_row_text, title
                    )
                    pieces.append((piece_text, True))
                    current_rows = []

        if current_rows:
            piece_text = _format_block(
                segment.label, segment.header_row, current_rows
            )
            pieces.append((piece_text, False))

        return pieces

    def chunk(
        self,
        source: SourceFile,
        extracted: Extracted,
        *,
        params: ParamsFingerprint,
    ) -> tuple[ChunkRecord, ...]:
        """Convert extracted segments into ChunkRecords (requirements 6.2 - 6.6)."""
        records: list[ChunkRecord] = []
        ordinal = 0

        accumulated_paragraphs: list[str] = []
        accumulated_locators: list[Locator] = []
        current_heading_path: tuple[str, ...] | None = None
        pending_overlap: str | None = None

        def flush_prose(*, allow_overlap: bool) -> None:
            nonlocal ordinal, pending_overlap, accumulated_paragraphs, accumulated_locators, current_heading_path
            if not accumulated_paragraphs:
                return

            body = "\n\n".join(accumulated_paragraphs)
            full_text = (
                f"{pending_overlap}\n\n{body}"
                if pending_overlap is not None
                else body
            )
            locator = _combine_locators(accumulated_locators)
            truncated = False

            if self.measure(full_text, extracted.title) > self._budget:
                if self.measure(body, extracted.title) <= self._budget:
                    overlap_ids = (
                        self._tokenizer.tokenizer.encode(
                            pending_overlap, add_special_tokens=False
                        )
                        if pending_overlap
                        else []
                    )
                    low = 0
                    high = len(overlap_ids)
                    best_overlap = ""
                    while low <= high:
                        mid = (low + high) // 2
                        cand_overlap = (
                            str(
                                self._tokenizer.tokenizer.decode(
                                    overlap_ids[-mid:], skip_special_tokens=True
                                )
                            )
                            if mid > 0
                            else ""
                        )
                        cand_full = (
                            f"{cand_overlap}\n\n{body}"
                            if cand_overlap
                            else body
                        )
                        if self.measure(cand_full, extracted.title) <= self._budget:
                            best_overlap = cand_overlap
                            low = mid + 1
                        else:
                            high = mid - 1
                    full_text = (
                        f"{best_overlap}\n\n{body}"
                        if best_overlap
                        else body
                    )
                else:
                    full_text = self._truncate_to_budget(body, extracted.title)
                    truncated = True

            records.append(
                self.assemble_record(
                    source=source,
                    title=extracted.title,
                    kind=ChunkKind.PROSE,
                    text=full_text,
                    locator=locator,
                    ordinal=ordinal,
                    truncated=truncated,
                    params=params,
                )
            )
            ordinal += 1

            if allow_overlap:
                pending_overlap = self._get_overlap_text(full_text)
            else:
                pending_overlap = None

            accumulated_paragraphs = []
            accumulated_locators = []

        for segment in extracted.segments:
            if isinstance(segment, ProseSegment):
                if (
                    current_heading_path is not None
                    and segment.heading_path != current_heading_path
                ):
                    flush_prose(allow_overlap=False)
                    current_heading_path = segment.heading_path

                if current_heading_path is None:
                    current_heading_path = segment.heading_path

                candidate_body = (
                    "\n\n".join(accumulated_paragraphs + [segment.text])
                    if accumulated_paragraphs
                    else segment.text
                )
                candidate_text = (
                    f"{pending_overlap}\n\n{candidate_body}"
                    if pending_overlap is not None
                    else candidate_body
                )

                if (
                    not accumulated_paragraphs
                    or self.measure(candidate_text, extracted.title) <= self._budget
                ):
                    accumulated_paragraphs.append(segment.text)
                    accumulated_locators.append(segment.locator)
                else:
                    flush_prose(allow_overlap=True)
                    accumulated_paragraphs.append(segment.text)
                    accumulated_locators.append(segment.locator)

            else:
                flush_prose(allow_overlap=False)
                current_heading_path = None
                pending_overlap = None

                if isinstance(segment, TableSegment):
                    table_text = segment.text
                    truncated = False
                    if self.measure(table_text, extracted.title) > self._budget:
                        table_text = self._truncate_to_budget(
                            table_text, extracted.title
                        )
                        truncated = True
                    records.append(
                        self.assemble_record(
                            source=source,
                            title=extracted.title,
                            kind=ChunkKind.TABLE,
                            text=table_text,
                            locator=segment.locator,
                            ordinal=ordinal,
                            truncated=truncated,
                            params=params,
                        )
                    )
                    ordinal += 1

                elif isinstance(segment, FigureSegment):
                    fig_text = segment.text
                    truncated = False
                    if self.measure(fig_text, extracted.title) > self._budget:
                        fig_text = self._truncate_to_budget(
                            fig_text, extracted.title
                        )
                        truncated = True
                    records.append(
                        self.assemble_record(
                            source=source,
                            title=extracted.title,
                            kind=ChunkKind.FIGURE,
                            text=fig_text,
                            locator=segment.locator,
                            ordinal=ordinal,
                            truncated=truncated,
                            provenance=segment.provenance,
                            params=params,
                        )
                    )
                    ordinal += 1

                elif isinstance(segment, FormulaSegment):
                    form_text = segment.text
                    truncated = False
                    if self.measure(form_text, extracted.title) > self._budget:
                        form_text = self._truncate_to_budget(
                            form_text, extracted.title
                        )
                        truncated = True
                    records.append(
                        self.assemble_record(
                            source=source,
                            title=extracted.title,
                            kind=ChunkKind.FORMULA,
                            text=form_text,
                            locator=segment.locator,
                            ordinal=ordinal,
                            truncated=truncated,
                            params=params,
                        )
                    )
                    ordinal += 1

                elif isinstance(segment, BlockSegment):
                    block_pieces = self.split_block(
                        segment, extracted.title
                    )
                    for piece_text, piece_truncated in block_pieces:
                        records.append(
                            self.assemble_record(
                                source=source,
                                title=extracted.title,
                                kind=ChunkKind.BLOCK,
                                text=piece_text,
                                locator=segment.locator,
                                ordinal=ordinal,
                                truncated=piece_truncated,
                                params=params,
                            )
                        )
                        ordinal += 1

        flush_prose(allow_overlap=False)

        return tuple(records)
