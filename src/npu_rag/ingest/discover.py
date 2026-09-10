"""Source-root discovery (task 3.1).

Walk every configured root recursively with ``\\?\\``-prefixed absolute
paths on Windows, apply include and exclude globs to the path relative to
the root, de-duplicate files reached through overlapping roots by resolved
absolute path, and derive author as the first directory component beneath
that file's root. A missing or unreadable root is an omission, never a
raised error; remaining roots are still walked.

This module sits to the right of state and to the left of route. It does
not import vision, extract, or state.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.types import Omission, OmissionCategory, SourceFile

__all__ = ["Discoverer", "Discovery"]

_LONG_PREFIX = "\\\\?\\"
_UNC_LONG_PREFIX = "\\\\?\\UNC\\"


@dataclass(frozen=True)
class Discovery:
    """De-duplicated source files and the omissions produced while walking."""

    files: tuple[SourceFile, ...]
    omissions: tuple[Omission, ...]


class Discoverer:
    """Walk configured roots into de-duplicated ``SourceFile``s.

    Requirements 1.1–1.6 and 7.2. Unreadable roots become
    ``Omission(ROOT_UNAVAILABLE)`` and never raise.
    """

    def discover(self, config: IngestConfig) -> Discovery:
        include = _compile_globs(config.include)
        exclude = _compile_globs(config.exclude)
        files: list[SourceFile] = []
        omissions: list[Omission] = []
        seen: set[str] = set()
        for root in config.roots:
            prepared = _prepare_root(root)
            if isinstance(prepared, Omission):
                omissions.append(prepared)
                continue
            for source in _walk_root(prepared, include=include, exclude=exclude):
                key = _dedup_key(source.path)
                if key in seen:
                    continue
                seen.add(key)
                files.append(source)
        return Discovery(files=tuple(files), omissions=tuple(omissions))


def _prepare_root(root: Path) -> Path | Omission:
    try:
        long_root = _windows_long_path(root)
        if not long_root.exists():
            return Omission(
                category=OmissionCategory.ROOT_UNAVAILABLE,
                path=root,
                reason="source root does not exist",
            )
        if not long_root.is_dir():
            return Omission(
                category=OmissionCategory.ROOT_UNAVAILABLE,
                path=root,
                reason="source root is not a directory",
            )
        with os.scandir(long_root) as entries:
            next(entries, None)
        return _windows_long_path(long_root.resolve())
    except OSError as exc:
        return Omission(
            category=OmissionCategory.ROOT_UNAVAILABLE,
            path=root,
            reason=f"source root cannot be read: {exc}",
        )


def _walk_root(
    root: Path,
    *,
    include: tuple[re.Pattern[str], ...],
    exclude: tuple[re.Pattern[str], ...],
) -> Iterator[SourceFile]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            candidate = Path(dirpath) / name
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            relative = _relative_to(candidate, root)
            if not _accepted(relative, include, exclude):
                continue
            yield SourceFile(
                path=_windows_long_path(candidate),
                root=root,
                relative_path=relative,
                author=_author(relative),
            )


def _accepted(
    relative: Path,
    include: tuple[re.Pattern[str], ...],
    exclude: tuple[re.Pattern[str], ...],
) -> bool:
    text = relative.as_posix()
    if include and not any(pattern.fullmatch(text) for pattern in include):
        return False
    if any(pattern.fullmatch(text) for pattern in exclude):
        return False
    return True


def _author(relative: Path) -> str:
    parts = relative.parts
    if len(parts) <= 1:
        return ""
    return parts[0]


def _relative_to(path: Path, root: Path) -> Path:
    return Path(
        os.path.relpath(_without_long_prefix(path), _without_long_prefix(root))
    )


def _dedup_key(path: Path) -> str:
    resolved = _windows_long_path(path).resolve()
    return os.path.normcase(str(resolved))


def _windows_long_path(path: Path) -> Path:
    absolute = path if path.is_absolute() else path.resolve()
    text = _without_long_prefix(absolute)
    if os.name != "nt":
        return Path(text)
    if text.startswith("\\\\"):
        return Path(_UNC_LONG_PREFIX + text[2:])
    return Path(_LONG_PREFIX + text)


def _without_long_prefix(path: Path | str) -> str:
    text = str(path)
    if text.startswith(_UNC_LONG_PREFIX):
        return "\\\\" + text[len(_UNC_LONG_PREFIX) :]
    if text.startswith(_LONG_PREFIX):
        return text[len(_LONG_PREFIX) :]
    return text


def _compile_globs(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    flags = re.IGNORECASE if os.name == "nt" else 0
    return tuple(re.compile(_glob_to_regex(pattern), flags) for pattern in patterns)


def _glob_to_regex(pattern: str) -> str:
    """Translate a glob with ``**`` into a full-match regex over a posix path."""
    posix = pattern.replace("\\", "/")
    i = 0
    n = len(posix)
    parts: list[str] = ["^"]
    while i < n:
        char = posix[i]
        if char == "*":
            if i + 1 < n and posix[i + 1] == "*":
                i += 2
                if i < n and posix[i] == "/":
                    i += 1
                    parts.append("(?:.*/)?")
                else:
                    parts.append(".*")
            else:
                parts.append("[^/]*")
                i += 1
        elif char == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(char))
            i += 1
    parts.append("$")
    return "".join(parts)
