"""Unit tests for the OpenRouter vision credential (task 1.5).

Requirement 10.4: the secret is read from configuration and never written to
a log, report, record or error message. Requirement 10.5, at this boundary:
absence is ``None``, not a stand-in, so later stages can degrade to local
paths rather than inventing a credential.

The object is a mirror of ``HfCredential``. The dotenv helpers and the
redaction marker are imported from the runtime's acquisition module; they
are not copied here.
"""

from __future__ import annotations

import ast
import traceback
from pathlib import Path

import pytest

from npu_rag.embedding.models.acquire import REDACTED
from npu_rag.ingest.credential import OpenRouterCredential, discover_openrouter_credential

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "ingest" / "credential.py"
)

#: A value that would be obvious if it leaked into any rendered form.
SECRET = "or-test-secret-value-do-not-leak"
DOTENV_SECRET = "or-dotenv-secret-value-do-not-leak"
ENV_KEY = "OPENROUTER_API_KEY"


def _credential(value: str = SECRET, *, source: str = "test") -> OpenRouterCredential:
    return OpenRouterCredential(value, source=source)


def _ordinary_renderings(credential: OpenRouterCredential) -> str:
    """Every ordinary way a value escapes: str, repr, f-string, traceback."""
    parts = [
        str(credential),
        repr(credential),
        f"{credential}",
        f"{credential!r}",
        format(credential),
        ascii(credential),
    ]
    try:
        raise RuntimeError(credential)
    except RuntimeError:
        parts.append(traceback.format_exc())
    try:
        raise ValueError(f"auth failed for {credential}")
    except ValueError as error:
        parts.append(str(error))
        parts.append(repr(error))
        parts.append("".join(traceback.format_exception(error)))
    return "\n".join(parts)


# --------------------------------------------------------------------------
# The secret never renders (requirement 10.4)
# --------------------------------------------------------------------------


def test_ordinary_renderings_contain_the_redaction_marker_and_not_the_secret() -> None:
    """Observable: grep the rendered credential for the secret; find nothing."""
    credential = _credential()
    rendered = _ordinary_renderings(credential)

    assert SECRET not in rendered
    assert REDACTED in rendered
    assert REDACTED in str(credential)
    assert REDACTED in repr(credential)


def test_reveal_is_the_only_path_to_the_secret() -> None:
    credential = _credential()

    assert credential.reveal() == SECRET
    assert SECRET not in str(credential)
    assert SECRET not in repr(credential)
    assert SECRET not in f"{credential}"


def test_redact_replaces_the_secret_and_does_not_echo_it() -> None:
    credential = _credential()
    redacted = credential.redact(f"Authorization: Bearer {SECRET}")

    assert SECRET not in redacted
    assert REDACTED in redacted


def test_blank_credential_is_refused() -> None:
    with pytest.raises(ValueError, match="blank"):
        OpenRouterCredential("   ", source="test")


def test_source_is_safe_to_print() -> None:
    credential = _credential(source="the OPENROUTER_API_KEY environment variable")

    assert "environment variable" in repr(credential)
    assert SECRET not in repr(credential)


# --------------------------------------------------------------------------
# Discovery (requirement 10.4 read-from-configuration; 10.5 absence)
# --------------------------------------------------------------------------


def test_environment_variable_wins_over_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{ENV_KEY}={DOTENV_SECRET}\n", encoding="utf-8")

    credential = discover_openrouter_credential(
        env={ENV_KEY: SECRET}, start=tmp_path
    )

    assert credential is not None
    assert credential.reveal() == SECRET
    assert ".env" not in credential.source
    assert ENV_KEY in credential.source


def test_dotenv_is_discovered_by_walking_upward(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{ENV_KEY}={DOTENV_SECRET}\n", encoding="utf-8")
    nested = tmp_path / "src" / "npu_rag"
    nested.mkdir(parents=True)

    credential = discover_openrouter_credential(env={}, start=nested)

    assert credential is not None
    assert credential.reveal() == DOTENV_SECRET
    assert ENV_KEY in credential.source


def test_missing_credential_returns_none_not_a_fake(tmp_path: Path) -> None:
    """Requirement 10.5 at this boundary: absence is None, not a stand-in."""
    assert discover_openrouter_credential(env={}, start=tmp_path) is None


def test_blank_assignments_are_treated_as_absence(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{ENV_KEY}=   \n", encoding="utf-8")

    assert discover_openrouter_credential(
        env={ENV_KEY: "  "}, start=tmp_path
    ) is None


def test_blank_environment_variable_falls_through_to_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{ENV_KEY}={DOTENV_SECRET}\n", encoding="utf-8")

    credential = discover_openrouter_credential(
        env={ENV_KEY: "  "}, start=tmp_path
    )

    assert credential is not None
    assert credential.reveal() == DOTENV_SECRET


def test_a_hub_token_is_not_the_vision_credential(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("HF_TOKEN=hf-not-a-vision-key\n", encoding="utf-8")

    assert discover_openrouter_credential(
        env={"HF_TOKEN": "hf-not-a-vision-key"}, start=tmp_path
    ) is None


def test_discovered_credential_does_not_render_the_secret(tmp_path: Path) -> None:
    credential = discover_openrouter_credential(
        env={ENV_KEY: SECRET}, start=tmp_path
    )

    assert credential is not None
    rendered = _ordinary_renderings(credential)
    assert SECRET not in rendered
    assert REDACTED in rendered


# --------------------------------------------------------------------------
# Boundary: import the runtime helpers; do not copy them; do not import Hub
# --------------------------------------------------------------------------


def test_credential_module_imports_runtime_helpers_and_does_not_copy_them() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_from_acquire: list[str] = []
    defined: set[str] = set()
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
            if node.module == "npu_rag.embedding.models.acquire":
                imported_from_acquire.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)

    assert "REDACTED" in imported_from_acquire
    assert "find_dotenv" in imported_from_acquire
    assert "parse_dotenv" in imported_from_acquire
    assert "find_dotenv" not in defined
    assert "parse_dotenv" not in defined
    assert all(
        not name.startswith("huggingface_hub") for name in imported_modules
    )
    ingest_imports = [
        name for name in imported_modules if name.startswith("npu_rag.ingest")
    ]
    assert ingest_imports == []


def test_credential_is_not_exported_from_the_package() -> None:
    import npu_rag.ingest as ingest

    assert ingest.__all__ == []
    assert not hasattr(ingest, "OpenRouterCredential")
