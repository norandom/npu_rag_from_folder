"""The ingest failure taxonomy (task 1.2), structured so category is a type.

design.md's Error Strategy mirrors the runtime: ``IngestError(message, *,
stage, path)`` with named subclasses, so a caller separates failures with
``except`` rather than by matching strings. Stage is never absent — whoever
raises always knows what it was doing, and where it does not say, its type
does.

Nothing here validates its way into raising from a constructor. An error
class that can fail to be built would replace a diagnosed failure with a
confusing one, at exactly the moment the diagnosis matters most.

This module sits in the leftmost layer of design.md's dependency direction;
it may import ``types`` and nothing else from this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

__all__ = [
    "DiscoveryError",
    "ExtractionError",
    "IngestError",
    "StateError",
    "VisionError",
    "VisionUnavailable",
]


class IngestError(Exception):
    """Root of the taxonomy: carries stage and path.

    Catching this catches every failure this feature raises. A caller that
    cares about the category catches one of the siblings below instead —
    never by inspecting the message, which is written for a human and may
    change.
    """

    default_stage: ClassVar[str] = "unspecified"

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage: str = _cleaned(stage) or self.default_stage
        self.path: Path | None = path

    def __str__(self) -> str:
        path_text = str(self.path) if self.path is not None else "unknown"
        return f"{self.message} [stage={self.stage}, path={path_text}]"


class DiscoveryError(IngestError):
    """A configured root could not be walked."""

    default_stage: ClassVar[str] = "discovery"


class ExtractionError(IngestError):
    """A file could not be turned into segments (corrupt PDF, malformed workbook)."""

    default_stage: ClassVar[str] = "extraction"


class VisionError(IngestError):
    """An image-to-text request failed; the file's other chunks still commit."""

    default_stage: ClassVar[str] = "vision"


class VisionUnavailable(IngestError):
    """The vision credential was rejected (401/403); latched for the rest of the run.

    A sibling of ``VisionError``, not a subclass: catching one must not
    swallow the other, because they map to different omission categories.
    """

    default_stage: ClassVar[str] = "vision_unavailable"


class StateError(IngestError):
    """The state store cannot be used; the one fatal category, aborting the run."""

    default_stage: ClassVar[str] = "state"


def _cleaned(value: str | None) -> str | None:
    """A non-blank string, or ``None``. Absence gets one representation."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
