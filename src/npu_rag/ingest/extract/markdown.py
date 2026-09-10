"""Markdown extractor: token stream to ordered segments (task 4.2).

markdown-it-py provides the tokens. Front matter and HTML blocks are
dropped; image syntax is dropped from text and emitted as ``ImageRef``s
resolved from the file's directory; GFM tables become ``TableSegment``;
headings maintain a ``heading_path``. Locators convert markdown-it's
0-based exclusive ``token.map`` to 1-based inclusive ranges.

This module sits in the extract rank. It never imports vision, state, or
httpx.
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.token import Token
from PIL import Image, UnidentifiedImageError

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import ExtractionError
from npu_rag.ingest.extract.base import normalise
from npu_rag.ingest.types import (
    Extracted,
    ImageRef,
    MarkdownLocator,
    ProseSegment,
    Segment,
    SourceFile,
    TableSegment,
)

__all__ = ["MarkdownExtractor"]

_TITLE_LINE = re.compile(r"^title:\s*(.*)$", re.MULTILINE)


def _front_matter_rule(
    state: StateBlock, startLine: int, endLine: int, silent: bool
) -> bool:
    """Recognize a leading YAML front-matter block as a single token."""
    marker = "-"
    min_markers = 3
    start = state.bMarks[startLine] + state.tShift[startLine]
    maximum = state.eMarks[startLine]
    if startLine != 0 or not state.src or state.src[0] != marker:
        return False
    pos = start + 1
    while pos <= maximum and pos < len(state.src):
        if state.src[pos] != marker:
            break
        pos += 1
    marker_count = pos - start
    if marker_count < min_markers:
        return False
    if silent:
        return True
    auto_closed = False
    next_line = startLine
    while True:
        next_line += 1
        if next_line >= endLine:
            return False
        start = state.bMarks[next_line] + state.tShift[next_line]
        maximum = state.eMarks[next_line]
        if start < maximum and state.sCount[next_line] < state.blkIndent:
            break
        if start >= maximum or state.src[start] != marker:
            continue
        if state.is_code_block(next_line):
            continue
        pos = start + 1
        while pos < maximum:
            if state.src[pos] != marker:
                break
            pos += 1
        if (pos - start) < marker_count:
            continue
        pos = state.skipSpaces(pos)
        if pos < maximum:
            continue
        auto_closed = True
        break
    old_parent = state.parentType
    old_line_max = state.lineMax
    state.parentType = "container"
    state.lineMax = next_line
    token = state.push("front_matter", "", 0)
    token.hidden = True
    token.markup = marker * min_markers
    token.content = state.src[
        state.bMarks[startLine + 1] : state.eMarks[next_line - 1]
    ]
    token.block = True
    state.parentType = old_parent
    state.lineMax = old_line_max
    state.line = next_line + (1 if auto_closed else 0)
    token.map = [startLine, state.line]
    return True


def _build_parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark").enable("table")
    parser.block.ruler.before(
        "table",
        "front_matter",
        _front_matter_rule,
        {"alt": ["paragraph", "reference", "blockquote", "list"]},
    )
    return parser


_PARSER = _build_parser()


class MarkdownExtractor:
    """Parse Markdown into prose, tables, and image refs. Conforms to ``Extractor``."""

    def extract(self, source: SourceFile, config: IngestConfig) -> Extracted:
        del config
        try:
            raw = source.path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ExtractionError(
                f"could not read {source.relative_path.as_posix()} as UTF-8 text",
                stage="extraction",
                path=source.path,
            ) from exc
        tokens = _PARSER.parse(raw)
        title: str | None = None
        first_heading: str | None = None
        heading_stack: list[tuple[int, str]] = []
        segments: list[Segment] = []
        index = 0
        count = len(tokens)
        while index < count:
            token = tokens[index]
            kind = token.type
            if kind == "front_matter":
                if title is None:
                    title = _title_from_front_matter(token.content)
                index += 1
                continue
            if kind in {"html_block", "hr", "reference"}:
                index += 1
                continue
            if kind == "heading_open":
                level = int(token.tag[1:])
                heading_range = _line_range(token.map)
                heading_text = ""
                heading_images: list[ImageRef] = []
                image_ordinal = 0
                index += 1
                while index < count and tokens[index].type != "heading_close":
                    if tokens[index].type == "inline":
                        children = tokens[index].children or []
                        heading_text = normalise(_visible_text(children))
                        extra, image_ordinal = _image_refs_from_children(
                            children, heading_range, source, image_ordinal
                        )
                        heading_images.extend(extra)
                    index += 1
                if index < count and tokens[index].type == "heading_close":
                    index += 1
                if first_heading is None and heading_text:
                    first_heading = heading_text
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))
                segments.extend(heading_images)
                continue
            if kind == "table_open":
                locator = MarkdownLocator(
                    line_range=_line_range(token.map), ordinal=0
                )
                cells: list[str] = []
                table_images: list[ImageRef] = []
                image_ordinal = 0
                index += 1
                while index < count and tokens[index].type != "table_close":
                    if tokens[index].type == "inline":
                        children = tokens[index].children or []
                        cell = normalise(_visible_text(children))
                        if cell:
                            cells.append(cell)
                        extra, image_ordinal = _image_refs_from_children(
                            children, locator.line_range, source, image_ordinal
                        )
                        table_images.extend(extra)
                    index += 1
                if index < count and tokens[index].type == "table_close":
                    index += 1
                text = normalise(" ".join(cells))
                if text:
                    segments.append(TableSegment(text=text, locator=locator))
                segments.extend(table_images)
                continue
            if kind == "paragraph_open":
                locator = MarkdownLocator(
                    line_range=_line_range(token.map), ordinal=0
                )
                heading_path = tuple(text for _, text in heading_stack)
                index += 1
                while index < count and tokens[index].type != "paragraph_close":
                    if tokens[index].type == "inline":
                        segments.extend(
                            _from_inline(
                                tokens[index], locator, heading_path, source
                            )
                        )
                    index += 1
                if index < count and tokens[index].type == "paragraph_close":
                    index += 1
                continue
            if kind in {"fence", "code_block"}:
                text = normalise(token.content)
                if text:
                    segments.append(
                        ProseSegment(
                            text=text,
                            locator=MarkdownLocator(
                                line_range=_line_range(token.map), ordinal=0
                            ),
                            heading_path=tuple(h for _, h in heading_stack),
                        )
                    )
                index += 1
                continue
            index += 1
        if title is None:
            title = first_heading
        return Extracted(title=title, segments=tuple(segments), omissions=())


def _title_from_front_matter(content: str) -> str | None:
    match = _TITLE_LINE.search(content)
    if match is None:
        return None
    raw = match.group(1).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        raw = raw[1:-1]
    normalised = normalise(raw)
    return normalised or None


def _line_range(token_map: list[int] | None) -> tuple[int, int]:
    """Convert markdown-it ``[start, end)`` 0-based to inclusive 1-based."""
    if token_map is None or len(token_map) != 2:
        return (1, 1)
    return (token_map[0] + 1, token_map[1])


def _visible_text(tokens: list[Token]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token.type in {"image", "html_inline", "html_block", "front_matter"}:
            continue
        if token.type in {"text", "code_inline", "code_block", "fence"}:
            parts.append(token.content)
        elif token.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
        elif token.children:
            parts.append(_visible_text(token.children))
    return "".join(parts)


def _image_refs_from_children(
    tokens: list[Token],
    line_range: tuple[int, int],
    source: SourceFile,
    ordinal: int,
) -> tuple[list[ImageRef], int]:
    """Emit an ImageRef for every image token under this inline, in document order."""
    refs: list[ImageRef] = []
    for token in tokens:
        if token.type == "image":
            src = str(token.attrGet("src") or "")
            refs.append(_image_ref(source, src, line_range, ordinal))
            ordinal += 1
            continue
        if token.children:
            nested, ordinal = _image_refs_from_children(
                token.children, line_range, source, ordinal
            )
            refs.extend(nested)
    return refs, ordinal


def _from_inline(
    inline: Token,
    block_locator: MarkdownLocator,
    heading_path: tuple[str, ...],
    source: SourceFile,
) -> list[Segment]:
    segments: list[Segment] = []
    buffer: list[str] = []
    ordinal = 0

    def flush() -> None:
        text = normalise("".join(buffer))
        buffer.clear()
        if text:
            segments.append(
                ProseSegment(
                    text=text,
                    locator=MarkdownLocator(
                        line_range=block_locator.line_range, ordinal=0
                    ),
                    heading_path=heading_path,
                )
            )

    for child in inline.children or []:
        if child.type == "image":
            flush()
            src = str(child.attrGet("src") or "")
            segments.append(
                _image_ref(source, src, block_locator.line_range, ordinal)
            )
            ordinal += 1
        elif child.type == "html_inline":
            continue
        else:
            piece = _visible_text([child])
            if piece:
                buffer.append(piece)
    flush()
    return segments


def _is_remote(src: str) -> bool:
    lowered = src.lstrip().lower()
    return lowered.startswith(("http://", "https://", "data:", "//"))


def _image_ref(
    source: SourceFile,
    src: str,
    line_range: tuple[int, int],
    ordinal: int,
) -> ImageRef:
    if not src or _is_remote(src):
        raise ExtractionError(
            f"could not read image {src} referenced by "
            f"{source.relative_path.as_posix()}",
            stage="extraction",
            path=source.path,
        )
    relative = Path(unquote(src))
    path = relative if relative.is_absolute() else source.path.parent / relative
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(
            f"could not read image {src} referenced by "
            f"{source.relative_path.as_posix()}",
            stage="extraction",
            path=source.path,
        ) from exc
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            fmt = image.format
    except (UnidentifiedImageError, OSError) as exc:
        raise ExtractionError(
            f"could not decode image {src} referenced by "
            f"{source.relative_path.as_posix()}",
            stage="extraction",
            path=source.path,
        ) from exc
    if not fmt:
        raise ExtractionError(
            f"could not decode image {src} referenced by "
            f"{source.relative_path.as_posix()}",
            stage="extraction",
            path=source.path,
        )
    mime = Image.MIME.get(fmt, "application/octet-stream")
    return ImageRef(
        data=data,
        mime=mime,
        width=width,
        height=height,
        locator=MarkdownLocator(line_range=line_range, ordinal=ordinal),
    )
