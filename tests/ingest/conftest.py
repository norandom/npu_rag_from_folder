"""Offline tokenizer fixture for ingest tests (task 1.6).

Loads a real ``npu_rag.embedding.tokenize.ModelTokenizer`` for the ungated
Apache-2.0 third candidate from committed files under ``fixtures/tokenizer/``.
The network is not consulted: transformers is asked for ``local_files_only``
and Hub downloads are blocked for the duration of the load. Missing files are
a loud ``pytest.fail``, never a skip, so requirement 6.7's contract test
cannot silently drop out of the standing suite.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from npu_rag.embedding.profiles import profile_for
from npu_rag.embedding.tokenize import TOKENIZER_FILE_PATTERNS, ModelTokenizer

TOKENIZER_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "tokenizer"

#: The files this candidate actually publishes, all of which are named in
#: ``TOKENIZER_FILE_PATTERNS``. A missing name is a loud failure, not a skip.
COMMITTED_TOKENIZER_FILES: tuple[str, ...] = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

_MISSING = set(COMMITTED_TOKENIZER_FILES) - set(TOKENIZER_FILE_PATTERNS)
if _MISSING:  # pragma: no cover - a declaration error, caught at import
    raise ValueError(
        "COMMITTED_TOKENIZER_FILES names files the runtime tokenizer loader "
        f"does not ask for: {sorted(_MISSING)}"
    )

_GTE = profile_for("gte-modernbert-base")


def _missing_tokenizer_message(directory: Path) -> str:
    return (
        f"the committed gte-modernbert-base tokenizer fixture is missing from "
        f"{directory}. Requirement 6.7's two-way contract test must fail "
        f"loudly rather than skip; copy only the files named in "
        f"npu_rag.embedding.tokenize.TOKENIZER_FILE_PATTERNS into "
        f"tests/ingest/fixtures/tokenizer/."
    )


def require_tokenizer_files(directory: Path) -> None:
    """Fail loudly if the committed tokenizer files are not at ``directory``."""
    if not directory.is_dir():
        pytest.fail(_missing_tokenizer_message(directory))
    present = {path.name for path in directory.iterdir() if path.is_file()}
    if not set(COMMITTED_TOKENIZER_FILES).issubset(present):
        pytest.fail(_missing_tokenizer_message(directory))


def _tokenizer_id(directory: Path) -> str:
    revision_path = directory / "REVISION"
    if revision_path.is_file():
        revision = revision_path.read_text(encoding="utf-8").strip()
        if revision:
            return f"{_GTE.model_id}@{revision}"
    return f"{_GTE.model_id}@offline-fixture"


def _blocked_network(*_args: object, **_kwargs: object) -> NoReturn:
    raise OSError(
        "the offline tokenizer fixture must not use the network"
    )


@contextmanager
def _network_unreachable() -> Iterator[None]:
    with (
        patch.object(socket, "create_connection", _blocked_network),
        patch("huggingface_hub.snapshot_download", _blocked_network),
        patch.dict(
            os.environ,
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        ),
    ):
        yield


def load_offline_tokenizer(directory: Path | None = None) -> ModelTokenizer:
    """Load the runtime tokenizer from committed files, with no network."""
    path = TOKENIZER_FIXTURE_DIR if directory is None else directory
    require_tokenizer_files(path)
    with _network_unreachable():
        loaded = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True
        )
    if not isinstance(loaded, PreTrainedTokenizerBase):  # pragma: no cover
        raise TypeError(
            f"{type(loaded).__name__} is not a tokenizer this fixture can "
            "hand to the runtime"
        )
    return ModelTokenizer(
        profile=_GTE,
        tokenizer=loaded,
        tokenizer_id=_tokenizer_id(path),
    )


@pytest.fixture(scope="session")
def runtime_tokenizer() -> ModelTokenizer:
    """The ungated gte-modernbert-base tokenizer, loaded from committed files."""
    return load_offline_tokenizer()
