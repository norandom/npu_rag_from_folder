"""Validation 8.2: Offline and absent-model behaviour.

Requirements: 5.2, 9.4, 10.1, 10.2, 10.3, 10.5.
Boundary: offline validation.

Observable: the failing transport's request counter reads zero after a full run,
the report lists every image by path under vision-unavailable with the capability
named, and the loaded-module set contains no runtime provider or session module.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import npu_rag.ingest.credential as credential_mod
from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.pipeline import run_ingest
from npu_rag.ingest.types import OmissionCategory


def _create_png(path: Path, width: int = 250, height: int = 250) -> None:
    img = Image.new("RGB", (width, height), color=(150, 150, 150))
    img.save(path, format="PNG")


def test_offline_and_absent_model_behaviour(
    tmp_path: Path,
    runtime_tokenizer: ModelTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observable 8.2: Zero network requests issued, absent OCR models produce
    vision-unavailable omissions naming the missing capability, and no runtime
    provider or session module is imported.
    """
    root = tmp_path / "archive"
    root.mkdir()
    state_path = tmp_path / "state.sqlite"

    # Plant files:
    # 1. Markdown with an image reference
    img_embed = root / "chart_embed.png"
    _create_png(img_embed, width=250, height=250)

    doc_path = root / "document.md"
    doc_path.write_text(
        "# Financial Summary\n\n"
        "Here is the revenue performance.\n\n"
        "![Revenue Chart](chart_embed.png)\n\n"
        "Revenue increased significantly.",
        encoding="utf-8",
    )

    # 2. Standalone image file
    standalone_img = root / "standalone.png"
    _create_png(standalone_img, width=300, height=300)

    # 3. Plain text file
    txt_path = root / "notes.txt"
    txt_path.write_text("Unrelated prose notes.", encoding="utf-8")

    # Install network blocker: intercept socket connections
    network_call_count = 0

    original_connect = socket.socket.connect

    def _failing_connect(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal network_call_count
        network_call_count += 1
        raise RuntimeError("Network connection attempt detected during offline test")

    monkeypatch.setattr(socket.socket, "connect", _failing_connect)

    # Track if OpenRouterCredential is consulted (Requirement 10.4)
    credential_consulted = False

    if hasattr(credential_mod, "OpenRouterCredential"):
        original_init = credential_mod.OpenRouterCredential.__init__

        def _tracked_credential_init(self: Any, *args: Any, **kwargs: Any) -> None:
            nonlocal credential_consulted
            credential_consulted = True
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(
            credential_mod.OpenRouterCredential, "__init__", _tracked_credential_init
        )

    # Configure run pointing to non-existent OCR model directory
    config = IngestConfig(
        roots=(root,),
        token_budget=512,
        prose_overlap_tokens=32,
        min_image_pixels=200,
        state_path=state_path,
    )

    # Snapshot loaded modules before run_ingest
    initial_modules = set(sys.modules.keys())

    # Run ingest without injecting a describer
    report = run_ingest(config, runtime_tokenizer, describer=None)

    def _norm(p: Path | str) -> str:
        s = str(p)
        if s.startswith("\\\\?\\UNC\\"):
            s = "\\\\" + s[8:]
        elif s.startswith("\\\\?\\"):
            s = s[4:]
        return s.replace("\\", "/").lower()

    # 1. Zero network requests made
    assert network_call_count == 0

    # 2. Run completed successfully with local extraction paths
    assert report.files_failed == 0
    assert report.files_processed == 4
    assert len(report.records) > 0

    # 3. Every image reference is a vision-unavailable omission naming the missing capability (Requirements 5.2, 9.4, 10.5)
    vision_omissions = [
        o for o in report.omissions if o.category == OmissionCategory.VISION_UNAVAILABLE
    ]
    # Standalone image and the embedded image
    assert len(vision_omissions) >= 2

    omitted_paths = {_norm(o.path) for o in vision_omissions}
    assert _norm(img_embed) in omitted_paths
    assert _norm(standalone_img) in omitted_paths

    for omission in vision_omissions:
        assert omission.missing_capability is not None
        assert len(omission.missing_capability.strip()) > 0

    # 4. OpenRouterCredential was not consulted (Requirement 10.4)
    assert not credential_consulted

    # 5. Loaded modules: assert no runtime provider or session module was loaded during ingest (Requirement 10.1)
    forbidden_prefixes = (
        "npu_rag.embedding.providers",
        "npu_rag.embedding.session",
        "npu_rag.embedding.sessions",
    )
    new_modules = set(sys.modules.keys()) - initial_modules
    for mod_name in new_modules:
        for prefix in forbidden_prefixes:
            assert not mod_name.startswith(
                prefix
            ), f"Forbidden runtime module {mod_name} was loaded during ingest"
