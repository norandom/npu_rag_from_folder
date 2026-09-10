"""The ingest configuration object (task 1.4).

One frozen dataclass carries every tunable the pipeline reads. Roots and the
token budget have no defaults — requirement 1.1 forbids a built-in source
path, and requirement 6.1 forbids this package from choosing the budget.
Everything else takes the value named in design.md's Summary-only components.

This module sits immediately to the right of types/errors; it imports nothing
from this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["IngestConfig"]

_DEFAULT_VISION_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
_DEFAULT_VISION_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_STATE_PATH = Path(".npu_rag") / "ingest.sqlite"


@dataclass(frozen=True)
class IngestConfig:
    """Every ingest tunable, validated at construction.

    ``min_image_pixels`` is applied to an image's shorter side. An empty
    ``include`` tuple means include-all; an empty ``exclude`` tuple means
    exclude-none. Discovery interprets those; this object only stores them.
    """

    roots: tuple[Path, ...]
    token_budget: int
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    prose_overlap_tokens: int = 64
    min_page_chars: int = 50
    min_image_pixels: int = 200
    vision_model: str = _DEFAULT_VISION_MODEL
    vision_base_url: str = _DEFAULT_VISION_BASE_URL
    vision_concurrency: int = 4
    state_path: Path = _DEFAULT_STATE_PATH

    def __post_init__(self) -> None:
        if not self.roots:
            raise ValueError("roots must contain at least one source root")
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.prose_overlap_tokens >= self.token_budget:
            raise ValueError(
                "prose_overlap_tokens must be less than token_budget"
            )
