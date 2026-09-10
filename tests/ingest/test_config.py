"""Unit tests for the ingest configuration object (task 1.4).

``config.py`` sits immediately to the right of types/errors in design.md's
dependency direction. The tests here pin the frozen field set, design.md's
defaults, and the three construction refusals; they also plant the two
negations requirement 1.1 and 6.1 care about — a hardcoded source root, and a
config that would have selected the token budget itself.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from npu_rag.ingest.config import IngestConfig

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "npu_rag" / "ingest" / "config.py"
)

ROOT = Path("inputs")
SECOND_ROOT = Path("models")

# Requirement 6.1: the budget is an input. Tests that need a number pick one
# that is not the runtime's compiled length, so a passing suite cannot be
# read as this package having chosen 512.
BUDGET = 256

DEFAULT_VISION_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
DEFAULT_VISION_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_STATE_PATH = Path(".npu_rag") / "ingest.sqlite"

DESIGN_FIELDS = {
    "roots",
    "include",
    "exclude",
    "token_budget",
    "prose_overlap_tokens",
    "min_page_chars",
    "min_image_pixels",
    "vision_model",
    "vision_base_url",
    "vision_concurrency",
    "state_path",
}


def make_config(**overrides: object) -> IngestConfig:
    values: dict[str, object] = {
        "roots": (ROOT,),
        "token_budget": BUDGET,
    }
    values.update(overrides)
    return IngestConfig(**values)  # type: ignore[arg-type]


def _field(name: str) -> dataclasses.Field[object]:
    fields = {item.name: item for item in dataclasses.fields(IngestConfig)}
    return fields[name]


# --------------------------------------------------------------------------
# Field set, defaults, frozen
# --------------------------------------------------------------------------


def test_ingest_config_carries_exactly_the_design_fields() -> None:
    """design.md, Summary-only components: IngestConfig."""
    assert {item.name for item in dataclasses.fields(IngestConfig)} == DESIGN_FIELDS


def test_required_fields_are_roots_and_token_budget() -> None:
    """Requirement 1.1: roots are configuration, not a default path.
    Requirement 6.1: the token budget is an input and is not selected here."""
    for name in ("roots", "token_budget"):
        field = _field(name)
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING


def test_omitting_roots_is_refused() -> None:
    with pytest.raises(TypeError):
        IngestConfig(token_budget=BUDGET)  # type: ignore[call-arg]


def test_omitting_the_token_budget_is_refused() -> None:
    with pytest.raises(TypeError):
        IngestConfig(roots=(ROOT,))  # type: ignore[call-arg]


def test_design_defaults_apply_when_only_required_fields_are_given() -> None:
    """design.md, Summary-only components: overlap 64, min_page_chars 50,
    min_image_pixels 200, vision model, concurrency 4, state path; include and
    exclude default to empty glob tuples. vision_base_url is the OpenRouter
    chat-completions base (design.md Vision seam: POST {base_url}/chat/completions)."""
    config = make_config()
    assert config.roots == (ROOT,)
    assert config.token_budget == BUDGET
    assert config.include == ()
    assert config.exclude == ()
    assert config.prose_overlap_tokens == 64
    assert config.min_page_chars == 50
    assert config.min_image_pixels == 200
    assert config.vision_model == DEFAULT_VISION_MODEL
    assert config.vision_base_url == DEFAULT_VISION_BASE_URL
    assert config.vision_concurrency == 4
    assert config.state_path == DEFAULT_STATE_PATH


def test_vision_base_url_is_the_openrouter_api_root_not_the_completions_path() -> None:
    """design.md Vision seam posts to ``{base_url}/chat/completions``."""
    config = make_config()
    assert config.vision_base_url == "https://openrouter.ai/api/v1"
    assert not config.vision_base_url.rstrip("/").endswith("chat/completions")


def test_supplied_values_are_stored() -> None:
    config = make_config(
        roots=(ROOT, SECOND_ROOT),
        token_budget=128,
        include=("**/*.md", "**/*.pdf"),
        exclude=("**/drafts/**",),
        prose_overlap_tokens=32,
        min_page_chars=10,
        min_image_pixels=64,
        vision_model="mistralai/pixtral-12b",
        vision_base_url="https://openrouter.ai/api/v1",
        vision_concurrency=2,
        state_path=Path("tmp") / "state.sqlite",
    )
    assert config.roots == (ROOT, SECOND_ROOT)
    assert config.token_budget == 128
    assert config.include == ("**/*.md", "**/*.pdf")
    assert config.exclude == ("**/drafts/**",)
    assert config.prose_overlap_tokens == 32
    assert config.min_page_chars == 10
    assert config.min_image_pixels == 64
    assert config.vision_model == "mistralai/pixtral-12b"
    assert config.vision_concurrency == 2
    assert config.state_path == Path("tmp") / "state.sqlite"


def test_ingest_config_is_frozen() -> None:
    config = make_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.token_budget = 1  # type: ignore[misc]


# --------------------------------------------------------------------------
# Construction refusals (design.md, Summary-only components: IngestConfig)
# --------------------------------------------------------------------------


def test_an_empty_root_tuple_is_refused() -> None:
    with pytest.raises(ValueError, match="root"):
        make_config(roots=())


def test_a_zero_token_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="budget"):
        make_config(token_budget=0)


def test_a_negative_token_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="budget"):
        make_config(token_budget=-1)


def test_overlap_equal_to_the_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="overlap"):
        make_config(token_budget=64, prose_overlap_tokens=64)


def test_overlap_above_the_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="overlap"):
        make_config(token_budget=64, prose_overlap_tokens=65)


def test_a_zero_overlap_below_a_positive_budget_is_accepted() -> None:
    config = make_config(token_budget=1, prose_overlap_tokens=0)
    assert config.prose_overlap_tokens == 0
    assert config.token_budget == 1


# --------------------------------------------------------------------------
# Requirement 1.1 / 6.1 planted negations
# --------------------------------------------------------------------------


def test_config_module_contains_no_hardcoded_source_root() -> None:
    """Requirement 1.1: no source root path as a fixed value.

    The archive path that must not leak in is the one the brief measured;
    any other drive-qualified Source path is the same class of mistake.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "substack" not in lowered
    assert "d:\\source" not in lowered
    assert "d:/source" not in lowered


def test_token_budget_default_is_not_the_runtime_compiled_length() -> None:
    """Requirement 6.1: do not select 512 (or any other budget) here."""
    field = _field("token_budget")
    for value in (field.default, field.default_factory):
        assert value != 512
        assert value is dataclasses.MISSING


def test_config_module_names_only_openrouter_as_the_hosted_base() -> None:
    """design.md Allowed Dependencies: the one outbound host is OpenRouter."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "openrouter.ai" in lowered
    assert "openai.com" not in lowered
    assert "anthropic.com" not in lowered
    assert "googleapis.com" not in lowered


# --------------------------------------------------------------------------
# Boundary guard
# --------------------------------------------------------------------------


def test_config_imports_nothing_from_this_project() -> None:
    """config needs no ingest types; importing rightward would fail the layer
    guard, and importing leftward is unused."""
    imported: list[str] = []
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            assert node.level == 0, "a relative import is still a project import"

    assert [name for name in imported if name.startswith("npu_rag")] == []
