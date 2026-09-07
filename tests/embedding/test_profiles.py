"""Unit tests for the three candidate model profiles (task 2.2).

Requirement 4.1 names exactly three candidates and 4.2 makes EmbeddingGemma the
initial default. Requirement 3.4 makes each model's document-side and query-side
convention part of the runtime's behaviour, and 3.6 makes the published maximum
input token length a contract with ``document-ingest``.

Two of the assertions here are load-bearing rather than descriptive:

* **The published maximum is the compiled length, never the architectural
  limit** (research.md, "Decision: Publish the compiled sequence length, not the
  model's architectural limit"). A consumer chunking to EmbeddingGemma's 2048
  would have three quarters of every chunk silently truncated at 512 while every
  call reported success. Nothing about the shape or the norm of the resulting
  vectors would reveal it, so it is pinned here.
* **The template strings are exact.** A one-character drift - a missing space
  around a pipe, ``search result`` becoming ``search results`` - embeds text the
  model was not trained to see. Retrieval quality drops; no error is raised.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from npu_rag.embedding.postprocess import POOLING_RULES
from npu_rag.embedding.profiles import (
    MISSING_TITLE_SENTINEL,
    PROFILES,
    INITIAL_DEFAULT_MODEL,
    ModelProfile,
    initial_default_profile,
    profile_for,
)
from npu_rag.embedding.types import DocumentText

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "npu_rag"
    / "embedding"
    / "profiles.py"
)

ALL_NAMES = (
    "embeddinggemma-300m",
    "nomic-embed-text-v1.5",
    "gte-modernbert-base",
)

GEMMA = "embeddinggemma-300m"
NOMIC = "nomic-embed-text-v1.5"
#: Requirement 4.1's third slot since the 2026-09-06 amendment; it held
#: ``bge-large-en-v1.5`` until task 5.5.
GTE = "gte-modernbert-base"


def a_profile(**overrides: object) -> ModelProfile:
    """A valid profile, so an invariant test changes exactly one thing."""
    fields: dict[str, object] = {
        "model_id": "vendor/example-model",
        "dimension": 768,
        "compiled_seq_len": 512,
        "architectural_context_limit": 2048,
        "batch_size": 1,
        "pooling": "mean",
        "has_dense_stage": False,
        "document_template": "{content}",
        "query_template": "{content}",
        "license_gated": False,
        "license_acceptance_url": None,
    }
    fields.update(overrides)
    return ModelProfile(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Requirement 4.1: three candidates, and 4.2: which one starts as the default
# --------------------------------------------------------------------------


def test_exactly_the_three_named_candidates_are_declared() -> None:
    """Requirement 4.1 names ``embeddinggemma-300m``, ``nomic-embed-text-v1.5``
    and ``gte-modernbert-base`` - three, no more and no fewer. A fourth model
    that happens to run on this NPU is not a candidate.

    ``bge-large-en-v1.5`` is gone, and its absence is asserted rather than merely
    implied: this test is what makes the swap total instead of additive."""
    assert set(PROFILES) == set(ALL_NAMES)
    assert len(PROFILES) == 3
    assert not [name for name in PROFILES if "bge" in name]


@pytest.mark.parametrize(
    "name, model_id",
    [
        (GEMMA, "google/embeddinggemma-300m"),
        (NOMIC, "nomic-ai/nomic-embed-text-v1.5"),
        (GTE, "Alibaba-NLP/gte-modernbert-base"),
    ],
)
def test_each_candidate_names_its_upstream_repository(
    name: str, model_id: str
) -> None:
    """The key is requirement 4.1's short name; ``model_id`` is what acquisition
    (task 3.1) actually asks Hugging Face for. They must not drift apart."""
    assert PROFILES[name].model_id == model_id
    assert PROFILES[name].name == name
    assert name in model_id


def test_embeddinggemma_is_the_initial_default_until_the_benchmark_says_otherwise(
) -> None:
    """Requirement 4.2: the initial default candidate, held only until the
    benchmark document (requirement 7.3) supersedes the choice."""
    assert INITIAL_DEFAULT_MODEL == GEMMA
    assert initial_default_profile() is PROFILES[GEMMA]


def test_profile_lookup_rejects_an_unknown_model_by_name() -> None:
    with pytest.raises(KeyError) as caught:
        profile_for("all-MiniLM-L6-v2")
    message = str(caught.value)
    assert "all-MiniLM-L6-v2" in message
    for name in ALL_NAMES:
        assert name in message


@pytest.mark.parametrize("name", ALL_NAMES)
def test_profile_lookup_returns_the_declared_profile(name: str) -> None:
    assert profile_for(name) is PROFILES[name]


# --------------------------------------------------------------------------
# Requirement 3.6 / the Observable: compiled length vs architectural limit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_the_published_maximum_is_the_compiled_length(name: str) -> None:
    """Requirement 3.6 reports a maximum input token length, and research.md
    settles which number that is: the length the graph was compiled at, because
    that is the one actually enforced."""
    profile = PROFILES[name]
    assert profile.max_input_tokens == profile.compiled_seq_len


@pytest.mark.parametrize(
    "name, compiled, architectural",
    [
        (GEMMA, 512, 2048),
        (NOMIC, 512, 8192),
        (GTE, 512, 8192),
    ],
)
def test_each_profile_states_both_lengths_separately(
    name: str, compiled: int, architectural: int
) -> None:
    """Two different numbers, two different fields. Collapsing them is the
    failure mode this task exists to prevent."""
    profile = PROFILES[name]
    assert profile.compiled_seq_len == compiled
    assert profile.architectural_context_limit == architectural


@pytest.mark.parametrize("name", ALL_NAMES)
def test_the_long_context_models_publish_the_shorter_compiled_length(
    name: str,
) -> None:
    """The distinction is only observable where the two numbers differ. For
    EmbeddingGemma a consumer trusting 2048 would lose three quarters of every
    chunk to silent truncation at 512 (research.md); for ModernBERT's 8192 it
    would lose fifteen sixteenths.

    Every current candidate is such a case - ``bge-large-en-v1.5`` was the one
    where the two numbers coincided, and it left the lineup at task 5.5. That
    coincidence must stay *permissible* even though no candidate exercises it
    now, which is what the invariant test below pins."""
    profile = PROFILES[name]
    assert profile.compiled_seq_len < profile.architectural_context_limit
    assert profile.max_input_tokens != profile.architectural_context_limit


def test_a_profile_may_still_compile_at_its_architectural_limit() -> None:
    """The two lengths are separate facts, not facts that must differ.

    ``bge-large-en-v1.5`` was the candidate where they legitimately coincided,
    and with it gone the parametrized test above would happily accept an
    implementation that *required* them to differ - which would force a false
    architectural limit into any future candidate that attends exactly what it
    was compiled at."""
    profile = a_profile(compiled_seq_len=512, architectural_context_limit=512)

    assert profile.max_input_tokens == 512


@pytest.mark.parametrize("name", ALL_NAMES)
def test_no_profile_compiles_beyond_what_the_model_can_attend_to(
    name: str,
) -> None:
    profile = PROFILES[name]
    assert profile.compiled_seq_len <= profile.architectural_context_limit


# --------------------------------------------------------------------------
# Requirement 3.4: the document-side and query-side conventions, exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, document_template, query_template",
    [
        (
            GEMMA,
            "title: {title} | text: {content}",
            "task: search result | query: {content}",
        ),
        (
            NOMIC,
            "search_document: {content}",
            "search_query: {content}",
        ),
        (GTE, "{content}", "{content}"),
    ],
)
def test_templates_match_the_published_conventions_character_for_character(
    name: str, document_template: str, query_template: str
) -> None:
    """Requirement 3.4. These strings are quoted from the model cards
    (research.md, "EmbeddingGemma model characteristics"); the spacing around
    EmbeddingGemma's pipes and the trailing space after Nomic's colons are part
    of the convention, not formatting.

    ``gte-modernbert-base`` publishes ``"prompts": {}`` and a null
    ``default_prompt_name`` in its ``config_sentence_transformers.json``: no
    instruction on either side. Both templates are therefore the identity, which
    is a quoted convention like any other rather than a gap to be filled."""
    profile = PROFILES[name]
    assert profile.document_template == document_template
    assert profile.query_template == query_template


def test_the_missing_title_sentinel_is_the_literal_word_none() -> None:
    """research.md: ``title: {title | "none"} | text: {content}``. The absent
    title renders as the word, not as an empty slot and not as Python's
    ``None``."""
    assert MISSING_TITLE_SENTINEL == "none"


@pytest.mark.parametrize(
    "name, document, rendered",
    [
        (
            GEMMA,
            DocumentText("The body text.", "A Real Title"),
            "title: A Real Title | text: The body text.",
        ),
        (
            GEMMA,
            DocumentText("The body text."),
            "title: none | text: The body text.",
        ),
        (
            NOMIC,
            DocumentText("The body text."),
            "search_document: The body text.",
        ),
        (GTE, DocumentText("The body text."), "The body text."),
    ],
)
def test_document_rendering_applies_the_model_convention(
    name: str, document: DocumentText, rendered: str
) -> None:
    assert PROFILES[name].render_document(document) == rendered


@pytest.mark.parametrize(
    "name, rendered",
    [
        (GEMMA, "task: search result | query: how does it work"),
        (NOMIC, "search_query: how does it work"),
        (GTE, "how does it work"),
    ],
)
def test_query_rendering_applies_the_model_convention(
    name: str, rendered: str
) -> None:
    assert PROFILES[name].render_query("how does it work") == rendered


# --------------------------------------------------------------------------
# Requirement 3.4's no-op path: a symmetric model whose templates are identity
#
# Task 5.5's Observable. Nothing exercised this before ``gte-modernbert-base``
# joined the lineup: every earlier candidate decorated at least one side, so an
# identity template ran through the template machinery for the first time here.
# --------------------------------------------------------------------------

#: Awkward on purpose. Leading and trailing whitespace, an internal newline and
#: a colon are exactly what a stray ``strip()``, a ``join`` or a "helpful"
#: prefix would disturb, and all three would survive a fixture reading
#: ``"hello"``.
RAW_TEXT = "  Ada Lovelace:\nthe first programmer.  "


def test_a_symmetric_model_embeds_the_text_it_was_given_and_nothing_else() -> None:
    """Identity means identity: not "close to", not "stripped", not "prefixed".

    The comparison is against the literal input rather than against another
    rendering, so a template that decorated *both* sides identically - which
    equality between the two renderings would happily accept - is still caught.
    """
    profile = PROFILES[GTE]

    assert profile.render_document(DocumentText(RAW_TEXT)) == RAW_TEXT
    assert profile.render_query(RAW_TEXT) == RAW_TEXT


def test_a_symmetric_model_renders_a_document_and_a_query_alike() -> None:
    """Implementation Note 5.1 in reverse.

    That note forbids asserting that a document and a query rendering *differ*,
    because Nomic's two templates legitimately produce equal token counts. This
    is the first profile where the renderings themselves are genuinely equal, so
    the property is asserted positively - and each side is still checked against
    its own independent reference above, never only against the other.
    """
    profile = PROFILES[GTE]
    document = DocumentText(RAW_TEXT)

    assert profile.render_document(document) == profile.render_query(RAW_TEXT)


def test_the_identity_case_is_not_how_every_profile_behaves() -> None:
    """Non-vacuity for the two tests above.

    If `render_document` simply returned its content for every model, both would
    pass and prove nothing about the identity template. The other two candidates
    demonstrably transform the same text.
    """
    for name in (GEMMA, NOMIC):
        assert PROFILES[name].render_document(DocumentText(RAW_TEXT)) != RAW_TEXT
        assert PROFILES[name].render_query(RAW_TEXT) != RAW_TEXT
        assert RAW_TEXT in PROFILES[name].render_document(DocumentText(RAW_TEXT))


def test_the_lineup_spans_both_a_symmetric_and_an_asymmetric_convention() -> None:
    """Non-vacuity for a biconditional asserted in the live tokenizer suite.

    ``test_tokenize_live.py`` asserts that a document and a query rendering are
    equal exactly when the two templates are - a statement that would pass
    trivially if every candidate fell on the same side of it. It is pinned here,
    where no network is involved and nothing can skip it: Implementation Note
    4.2, a gap covered only by a live test is not covered at all.
    """
    symmetric = {
        name
        for name, profile in PROFILES.items()
        if profile.document_template == profile.query_template
    }

    assert symmetric == {GTE}
    assert set(PROFILES) - symmetric == {GEMMA, NOMIC}


@pytest.mark.parametrize("name", [NOMIC, GTE])
def test_a_title_is_dropped_by_a_model_whose_document_form_has_no_slot(
    name: str,
) -> None:
    """Only EmbeddingGemma's document convention has a title slot. The other two
    take the content alone, so a titled document renders exactly as the same
    untitled document does - the title is not smuggled in as a prefix."""
    profile = PROFILES[name]
    assert profile.render_document(
        DocumentText("The body text.", "A Real Title")
    ) == profile.render_document(DocumentText("The body text."))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_only_embeddinggemma_declares_a_title_slot(name: str) -> None:
    assert PROFILES[name].has_title_slot is (name == GEMMA)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_rendering_never_re_interprets_the_text_it_is_given(name: str) -> None:
    """Content is data, not a format string. A chunk containing braces - code
    samples and JSON both do - must survive verbatim rather than raising or
    being substituted into."""
    profile = PROFILES[name]
    hostile = 'a {content} and a {title} and a {0} and a {}'
    assert hostile in profile.render_document(DocumentText(hostile))
    assert hostile in profile.render_query(hostile)


def test_rendering_is_deterministic() -> None:
    """Requirement 3.10 needs identical text in to mean identical text
    embedded; template application is the first place that could break."""
    profile = PROFILES[GEMMA]
    document = DocumentText("The body text.", "A Real Title")
    assert profile.render_document(document) == profile.render_document(document)


def test_an_empty_document_still_renders_its_convention() -> None:
    assert PROFILES[GEMMA].render_document(DocumentText("")) == (
        "title: none | text: "
    )


# --------------------------------------------------------------------------
# Pipeline shape: dimension, pooling, the Dense stage, batch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, dimension", [(GEMMA, 768), (NOMIC, 768), (GTE, 768)]
)
def test_each_profile_declares_its_vector_dimension(
    name: str, dimension: int
) -> None:
    """Requirement 3.6, and a revalidation trigger for ``vector-index``'s
    schema (design.md, Revalidation Triggers)."""
    assert PROFILES[name].dimension == dimension


@pytest.mark.parametrize(
    "name, has_dense_stage",
    [(GEMMA, True), (NOMIC, False), (GTE, False)],
)
def test_only_embeddinggemma_has_a_dense_stage(
    name: str, has_dense_stage: bool
) -> None:
    """research.md: EmbeddingGemma's sentence pipeline is Pooling -> Dense ->
    Normalize. Skipping the Dense stage yields vectors of the right shape and
    the right norm that are semantically wrong, which only the retrieval-quality
    measurement (6.4) would ever catch - so the flag is pinned here."""
    assert PROFILES[name].has_dense_stage is has_dense_stage


@pytest.mark.parametrize(
    "name, pooling", [(GEMMA, "mean"), (NOMIC, "mean"), (GTE, "cls")]
)
def test_each_profile_declares_the_pooling_its_own_model_card_declares(
    name: str, pooling: str
) -> None:
    """Read from each model's ``1_Pooling/config.json``.

    ``gte-modernbert-base`` sets ``pooling_mode_cls_token = True`` and every
    other mode false; the other two pool by mean. Mean-pooling a CLS model
    yields a vector of the right width carrying the right norm and a different
    meaning, so this is pinned per model rather than asserted as a property of
    the set."""
    assert PROFILES[name].pooling == pooling


def test_the_lineup_does_not_pool_by_a_single_rule() -> None:
    """Non-vacuity for the dispatch.

    While every candidate pooled by mean the dispatch could return the mean
    unconditionally and no profile-driven test could see it. Requirement 4.1's
    amendment is what ended that, and this asserts the lineup still spans both
    rules - if a later change collapsed it back to one, the CLS branch would go
    quietly unexercised through the profiles again."""
    assert {profile.pooling for profile in PROFILES.values()} == {"mean", "cls"}


def test_no_profile_declares_a_pooling_rule_the_runtime_cannot_apply() -> None:
    """The two halves of task 5.5 must agree.

    ``profiles`` names the rule and ``postprocess`` implements it, in two
    modules that do not import one another. A profile declaring a rule
    `postprocess.pool` does not implement would raise at embed time - late,
    after preparation and a compile - so the agreement is checked here instead.
    Both directions: an unimplemented declaration *and* an implemented rule no
    candidate uses would both be drift worth seeing."""
    declared = {profile.pooling for profile in PROFILES.values()}

    assert declared <= POOLING_RULES
    assert set(get_args(get_type_hints(ModelProfile)["pooling"])) == POOLING_RULES


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_profile_compiles_at_batch_size_one(name: str) -> None:
    """Static-shape constraint from the requirements' Approach section: fixed
    sequence length, batch size 1 by default."""
    assert PROFILES[name].batch_size == 1


# --------------------------------------------------------------------------
# Requirement 4.5: licence gating, recorded per model
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, license_gated", [(GEMMA, True), (NOMIC, False), (GTE, False)]
)
def test_licence_gating_is_recorded_per_model(
    name: str, license_gated: bool
) -> None:
    """Requirement 4.5 turns on this flag: EmbeddingGemma is gated behind the
    Gemma Terms of Use; the other two are Apache 2.0 and are not."""
    assert PROFILES[name].license_gated is license_gated


def test_the_gated_model_carries_the_acceptance_step() -> None:
    """Requirement 4.5 requires the *acceptance step* to be reported, not just
    the fact of gating - and ``LicenseAcceptanceRequired`` (task 2.1) cannot be
    raised without a URL, so the profile is where that URL comes from."""
    profile = PROFILES[GEMMA]
    assert profile.license_acceptance_url == (
        "https://huggingface.co/google/embeddinggemma-300m"
    )


@pytest.mark.parametrize("name", [NOMIC, GTE])
def test_an_ungated_model_carries_no_acceptance_url(name: str) -> None:
    assert PROFILES[name].license_acceptance_url is None


# --------------------------------------------------------------------------
# Value-object behaviour and invariants
# --------------------------------------------------------------------------


def test_a_profile_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PROFILES[GEMMA].compiled_seq_len = 2048  # type: ignore[misc]


def test_the_profile_mapping_cannot_be_mutated() -> None:
    """A module-level mapping is the whole registry (design.md: "No model
    registry, plugin discovery, or configuration DSL"), so it must not be
    editable at a distance."""
    with pytest.raises(TypeError):
        PROFILES["evil"] = a_profile()  # type: ignore[index]


def test_the_helper_builds_a_valid_profile() -> None:
    """Guards the invariant tests below: they are only meaningful if the
    baseline they perturb is itself accepted."""
    assert a_profile().model_id == "vendor/example-model"


def test_a_profile_cannot_compile_beyond_the_architectural_limit() -> None:
    """design.md, Domain Model: ``compiled_seq_len`` is fixed at export. A value
    above what the model can attend to could not be exported at all."""
    with pytest.raises(ValueError, match="compiled_seq_len"):
        a_profile(compiled_seq_len=4096, architectural_context_limit=2048)


@pytest.mark.parametrize(
    "field", ["dimension", "compiled_seq_len", "batch_size"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_a_profile_rejects_a_non_positive_shape(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        a_profile(**{field: value})


def test_a_gated_profile_must_say_where_to_accept_the_terms() -> None:
    with pytest.raises(ValueError, match="license_acceptance_url"):
        a_profile(license_gated=True, license_acceptance_url=None)


def test_an_ungated_profile_must_not_carry_an_acceptance_url() -> None:
    """The two fields are one fact stated twice; letting them disagree would
    leave requirement 4.5's decision depending on which one a caller read."""
    with pytest.raises(ValueError, match="license_acceptance_url"):
        a_profile(
            license_gated=False,
            license_acceptance_url="https://huggingface.co/vendor/example",
        )


@pytest.mark.parametrize(
    "template", ["title: {title}", "a bare prefix", "{title} {content"]
)
def test_a_document_template_without_a_content_slot_is_rejected(
    template: str,
) -> None:
    with pytest.raises(ValueError, match="document_template"):
        a_profile(document_template=template)


@pytest.mark.parametrize("template", ["query:", "{title}: {content}"])
def test_a_query_template_is_rejected_unless_it_takes_content_alone(
    template: str,
) -> None:
    """A query has no title, so a title slot on the query side would render the
    sentinel into every search string."""
    with pytest.raises(ValueError, match="query_template"):
        a_profile(query_template=template)


@pytest.mark.parametrize("template", ["{content} }", "{content} {"])
def test_a_malformed_template_is_rejected_for_being_malformed(
    template: str,
) -> None:
    """Not merely rejected - rejected with the reason. A syntactically broken
    template would otherwise be reported as a missing ``{content}`` slot, which
    sends the reader looking for the wrong defect."""
    with pytest.raises(ValueError, match="not a valid template"):
        a_profile(document_template=template)


def test_a_template_with_an_unknown_slot_is_rejected() -> None:
    """Only ``title`` and ``content`` are ever substituted. An unrecognised slot
    would raise at render time - one exception per embedded text, long after the
    profile was declared."""
    with pytest.raises(ValueError, match="section"):
        a_profile(document_template="{section}: {content}")


# --------------------------------------------------------------------------
# Dependency direction
# --------------------------------------------------------------------------

#: The module's own dotted package, used to resolve relative imports.
MODULE_PACKAGE = "npu_rag.embedding"

#: design.md, Architecture: "types, errors -> reporting -> profiles ->
#: environment -> models -> providers -> service -> bench". ``profiles`` may
#: read ``types``, ``errors`` and ``reporting``; everything to its right is off
#: limits.
LAYERS_RIGHT_OF_PROFILES = (
    "environment",
    "models",
    "providers",
    "service",
    "bench",
)


def absolute_imports_of(source: str, package: str) -> list[str]:
    """Every name ``source`` imports, as an absolute dotted path.

    Relative imports are resolved rather than skipped: ``from . import
    environment`` and ``import npu_rag.embedding.environment`` are the same
    violation, and a guard that only recognised the second would be evaded by
    writing the first.
    """
    parts = package.split(".")
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                root = node.module or ""
            else:
                base = ".".join(parts[: len(parts) - node.level + 1])
                root = f"{base}.{node.module}" if node.module else base
            names.append(root)
            # ``from X import Y`` may name a submodule, not just an attribute.
            names.extend(f"{root}.{alias.name}" for alias in node.names)
    return names


def test_profiles_imports_nothing_from_a_later_layer() -> None:
    imported = absolute_imports_of(
        MODULE_PATH.read_text(encoding="utf-8"), MODULE_PACKAGE
    )

    forbidden = sorted(
        {
            name
            for name in imported
            for layer in LAYERS_RIGHT_OF_PROFILES
            if name == f"npu_rag.embedding.{layer}"
            or name.startswith(f"npu_rag.embedding.{layer}.")
        }
    )
    assert forbidden == []


@pytest.mark.parametrize(
    "statement",
    [
        "import npu_rag.embedding.models",
        "from npu_rag.embedding.providers import base",
        "from npu_rag.embedding import environment",
        "from . import environment",
        "from .providers import base",
        "from ..embedding.service import EmbeddingService",
    ],
)
def test_the_layer_guard_recognises_every_import_spelling(
    statement: str,
) -> None:
    imported = absolute_imports_of(statement, MODULE_PACKAGE)

    assert any(
        name.startswith(f"npu_rag.embedding.{layer}")
        for name in imported
        for layer in LAYERS_RIGHT_OF_PROFILES
    ), f"{statement!r} resolved to {imported}"
