"""The offline tokenizer fixture (task 1.6).

Requirement 6.2: chunk length is measured with the runtime's tokenizer, never
an approximation of this package's own. Requirement 6.7's two-way contract
must run unconditionally, so this fixture loads a real ungated tokenizer from
committed files and fails loudly rather than skip when those files are absent.

design.md, File Structure Plan and Testing Strategy: the files live at
``tests/ingest/fixtures/tokenizer/`` (gte-modernbert-base, Apache-2.0, ungated)
and the session fixture in ``conftest.py`` is the only way the standing suite
obtains them.
"""

from __future__ import annotations

import ast
import socket
from pathlib import Path
from typing import Any

import pytest

from npu_rag.embedding.profiles import DEFAULT_COMPILED_SEQ_LEN, profile_for
from npu_rag.embedding.tokenize import TOKENIZER_FILE_PATTERNS, ModelTokenizer
from npu_rag.embedding.types import DocumentText, TextKind

INGEST_TESTS = Path(__file__).resolve().parent
CONFTEST_PATH = INGEST_TESTS / "conftest.py"
FIXTURE_DIR = INGEST_TESTS / "fixtures" / "tokenizer"

GTE = profile_for("gte-modernbert-base")

WEIGHT_SUFFIXES = {".onnx", ".safetensors", ".bin", ".pt", ".ckpt", ".h5", ".pkl"}
WEIGHT_NAMES = {
    "model.onnx",
    "model.safetensors",
    "pytorch_model.bin",
    "model.bin",
    "model.pt",
    "flax_model.msgpack",
    "tf_model.h5",
}

ALLOWED_FIXTURE_NAMES = {name.lower() for name in TOKENIZER_FILE_PATTERNS} | {
    "readme.md",
    "revision",
}


def _load_helper() -> Any:
    from tests.ingest.conftest import load_offline_tokenizer

    return load_offline_tokenizer


def _conftest_tree() -> ast.AST:
    return ast.parse(CONFTEST_PATH.read_text(encoding="utf-8"))


def _fixture_scope(tree: ast.AST, name: str) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != name:
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                func = decorator.func
                is_fixture = (
                    isinstance(func, ast.Attribute) and func.attr == "fixture"
                ) or (isinstance(func, ast.Name) and func.id == "fixture")
                if not is_fixture:
                    continue
                for keyword in decorator.keywords:
                    if keyword.arg == "scope" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        return str(keyword.value.value)
                return "function"
            if isinstance(decorator, ast.Attribute) and decorator.attr == "fixture":
                return "function"
            if isinstance(decorator, ast.Name) and decorator.id == "fixture":
                return "function"
    return None


def _call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            names.add(func.attr)
        elif isinstance(func, ast.Name):
            names.add(func.id)
    return names


def _imported_from(tree: ast.AST, module: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.extend(alias.name for alias in node.names)
    return names


def _keyword_used(tree: ast.AST, name: str, value: object) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != name or not isinstance(keyword.value, ast.Constant):
                continue
            if keyword.value.value == value:
                return True
    return False


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_args: object, **_kwargs: object) -> None:
        raise OSError("network is unreachable")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.snapshot_download", blocked)


# --------------------------------------------------------------------------
# The session fixture yields the runtime tokenizer (requirements 6.2, 6.7)
# --------------------------------------------------------------------------


def test_the_session_fixture_yields_the_runtime_tokenizer(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    assert isinstance(runtime_tokenizer, ModelTokenizer)
    assert runtime_tokenizer.profile == GTE
    assert runtime_tokenizer.model_id == GTE.model_id


def test_the_fixture_limit_equals_the_compiled_length(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Observable: the fixture yields a runtime tokenizer whose limit equals
    the compiled length, never the architectural context limit."""
    assert runtime_tokenizer.max_input_tokens == GTE.max_input_tokens
    assert runtime_tokenizer.max_input_tokens == GTE.compiled_seq_len
    assert runtime_tokenizer.max_input_tokens == DEFAULT_COMPILED_SEQ_LEN
    assert GTE.architectural_context_limit > GTE.compiled_seq_len
    assert runtime_tokenizer.max_input_tokens != GTE.architectural_context_limit


def test_the_fixture_is_session_scoped() -> None:
    assert CONFTEST_PATH.is_file(), (
        "tests/ingest/conftest.py is missing; the offline tokenizer fixture "
        "has nowhere to live"
    )
    assert _fixture_scope(_conftest_tree(), "runtime_tokenizer") == "session"


# --------------------------------------------------------------------------
# Local files only; the network is unreachable
# --------------------------------------------------------------------------


def test_loading_succeeds_with_the_network_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observable: loading succeeds with the network unreachable."""
    _block_network(monkeypatch)
    with pytest.raises(OSError, match="network is unreachable"):
        socket.create_connection(("huggingface.co", 443), timeout=1)

    tokenizer = _load_helper()()

    assert isinstance(tokenizer, ModelTokenizer)
    assert tokenizer.max_input_tokens == GTE.max_input_tokens


def test_conftest_loads_from_local_files_only() -> None:
    assert CONFTEST_PATH.is_file()
    tree = _conftest_tree()
    assert _keyword_used(tree, "local_files_only", True), (
        "conftest.py must load the tokenizer with local_files_only=True so a "
        "network attempt is never a fallback"
    )


# --------------------------------------------------------------------------
# Missing files fail loudly, never skip (design.md Testing Strategy)
# --------------------------------------------------------------------------


def test_missing_tokenizer_files_fail_loudly_rather_than_skip(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "tokenizer"
    empty.mkdir()
    load_offline_tokenizer = _load_helper()

    try:
        load_offline_tokenizer(empty)
    except pytest.skip.Exception as skipped:
        pytest.fail(
            "missing tokenizer files skipped instead of failing: "
            f"{skipped}"
        )
    except pytest.fail.Exception as failed:
        message = str(failed).lower()
        assert "tokenizer" in message
        assert "skip" in message or "6.7" in message or "missing" in message
    else:
        pytest.fail("missing tokenizer files neither failed nor skipped")


def test_an_absent_fixture_directory_fails_loudly_rather_than_skip(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "no-such-tokenizer"
    load_offline_tokenizer = _load_helper()

    try:
        load_offline_tokenizer(missing)
    except pytest.skip.Exception as skipped:
        pytest.fail(
            "an absent tokenizer directory skipped instead of failing: "
            f"{skipped}"
        )
    except pytest.fail.Exception as failed:
        assert "tokenizer" in str(failed).lower()
    else:
        pytest.fail("an absent tokenizer directory neither failed nor skipped")


def test_conftest_fails_loudly_and_never_skips() -> None:
    assert CONFTEST_PATH.is_file()
    tree = _conftest_tree()
    called = _call_names(tree)
    assert "fail" in called, (
        "conftest.py must pytest.fail when the tokenizer files are missing"
    )
    assert "skip" not in called
    assert "skipif" not in called
    assert "importorskip" not in called
    assert "xfail" not in called


# --------------------------------------------------------------------------
# No homemade token counter (requirement 6.2)
# --------------------------------------------------------------------------


def test_the_fixture_measures_with_the_runtime_tokenizer(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    content = "Partition share."
    counted = runtime_tokenizer.count_tokens(content, TextKind.DOCUMENT)
    rendered = GTE.render_document(DocumentText(content))
    independent = len(
        runtime_tokenizer.tokenizer.encode(
            rendered,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
    )
    assert counted == independent
    assert counted > 0
    assert type(runtime_tokenizer).count_tokens is ModelTokenizer.count_tokens


def test_conftest_does_not_define_a_homemade_token_counter() -> None:
    assert CONFTEST_PATH.is_file()
    tree = _conftest_tree()
    imported = _imported_from(tree, "npu_rag.embedding.tokenize")
    assert "ModelTokenizer" in imported
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "count_tokens" not in defined
    assert "encode" not in defined


# --------------------------------------------------------------------------
# Committed files: tokenizer artifacts only, no weights
# --------------------------------------------------------------------------


def test_the_fixture_directory_contains_tokenizer_files_and_no_weights() -> None:
    assert FIXTURE_DIR.is_dir(), (
        "tests/ingest/fixtures/tokenizer/ is missing; the offline tokenizer "
        "fixture has no committed files to load"
    )
    files = [path for path in FIXTURE_DIR.rglob("*") if path.is_file()]
    names = {path.name for path in files}
    lowered = {name.lower() for name in names}
    assert "tokenizer.json" in lowered
    assert "tokenizer_config.json" in lowered
    assert lowered & WEIGHT_NAMES == set()
    weight_hits = [
        path.as_posix()
        for path in files
        if path.suffix.lower() in WEIGHT_SUFFIXES or path.name.lower() in WEIGHT_NAMES
    ]
    assert weight_hits == []
    unexpected = {
        path.name
        for path in files
        if path.name.lower() not in ALLOWED_FIXTURE_NAMES
    }
    assert unexpected == set(), (
        f"fixture directory contains files outside TOKENIZER_FILE_PATTERNS: "
        f"{sorted(unexpected)}"
    )
