"""Rebuild the committed two-page PDF fixture (task 4.3).

pypdfium2's ``save`` is not byte-stable across runs, so the fixture is a
hand-built PDF: page 1 has two text-showing operators in content-stream
order; page 2 draws a rectangle and has no text layer. Re-running this
file must reproduce ``text_and_textless.pdf`` byte for byte.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "FIRST_SENTENCE",
    "PAGE_HEIGHT",
    "PAGE_WIDTH",
    "SECOND_SENTENCE",
    "build_pdf",
    "committed_fixture_bytes",
]

PAGE_WIDTH = 300
PAGE_HEIGHT = 200

FIRST_SENTENCE = "First sentence of extractable prose on the text page."
SECOND_SENTENCE = "Second sentence follows in content-stream order."

_TEXTLESS_STREAM = "20 20 260 160 re\n0.75 g\nf\n"
_FIXTURE_NAME = "text_and_textless.pdf"

PageContent = str | tuple[str, ...] | None


def build_pdf(pages: Sequence[PageContent]) -> bytes:
    """Build a small PDF: ``str``/tuple pages carry a text layer; ``None`` does not."""
    if not pages:
        raise ValueError("build_pdf requires at least one page")
    n_pages = len(pages)
    needs_font = any(page is not None for page in pages)
    page_nums = list(range(3, 3 + n_pages))
    content_nums = list(range(3 + n_pages, 3 + 2 * n_pages))
    font_num = (3 + 2 * n_pages) if needs_font else None
    kids = " ".join(f"{number} 0 R" for number in page_nums)
    objects: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>",
    }
    for page, page_num, content_num in zip(pages, page_nums, content_nums, strict=True):
        stream = _stream_for(page)
        if page is None:
            resources = "/Resources << >>"
        else:
            resources = f"/Resources << /Font << /F1 {font_num} 0 R >> >>"
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Contents {content_num} 0 R {resources} >>"
        )
        objects[content_num] = (
            f"<< /Length {len(stream.encode('ascii'))} >>\nstream\n{stream}endstream"
        )
    if font_num is not None:
        objects[font_num] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    return _assemble(objects)


def committed_fixture_bytes() -> bytes:
    return build_pdf(((FIRST_SENTENCE, SECOND_SENTENCE), None))


def _stream_for(page: PageContent) -> str:
    if page is None:
        return _TEXTLESS_STREAM
    lines = (page,) if isinstance(page, str) else page
    if not lines:
        raise ValueError("a text page needs at least one string")
    parts = ["BT", "/F1 12 Tf", "20 150 Td"]
    for index, line in enumerate(lines):
        if index:
            parts.append("0 -20 Td")
        parts.append(f"({_escape(line)}) Tj")
    parts.append("ET")
    return "\n".join(parts) + "\n"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _assemble(objects: dict[int, str]) -> bytes:
    max_obj = max(objects)
    parts = [b"%PDF-1.4\n"]
    offsets = [0]
    for number in range(1, max_obj + 1):
        offsets.append(sum(len(part) for part in parts))
        parts.append(f"{number} 0 obj\n{objects[number]}\nendobj\n".encode("ascii"))
    xref_at = sum(len(part) for part in parts)
    xref = [
        b"xref\n",
        f"0 {max_obj + 1}\n".encode("ascii"),
        b"0000000000 65535 f \n",
    ]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        f"trailer\n<< /Size {max_obj + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode("ascii")
    return b"".join(parts + xref + [trailer])


def main() -> None:
    target = Path(__file__).with_name(_FIXTURE_NAME)
    target.write_bytes(committed_fixture_bytes())


if __name__ == "__main__":
    main()
