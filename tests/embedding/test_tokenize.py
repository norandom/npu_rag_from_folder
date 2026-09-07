"""The tokenizer contract and length measurement (task 5.1).

Requirement 3.8 is an *if and only if*: an input the consumer measures as within
the reported maximum must be treated as within limits by the runtime, and an
input the consumer measures as over must be truncated. Both directions are
asserted here, over a sample of real prose, because a counter that is merely
conservative would satisfy one direction and silently shorten chunks
``document-ingest`` believed fit.

**These tests use real tokenizers and no network.** The fixtures build genuine
``tokenizers`` back ends - a WordPiece-style vocabulary with ``[CLS]``/``[SEP]``
post-processing, and a variant that adds a single leading token - wrapped in the
same ``PreTrainedTokenizerFast`` class the Hub returns. Nothing here is a stub
that approximates tokenization; the Rust tokenizer does the work. The
network-backed half, against the three real candidate tokenizers, lives in
``test_tokenize_live.py`` and skips when the Hub is unreachable, so the
requirement is not left uncovered on a CI runner (Implementation Note 4.2).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from npu_rag.embedding.errors import (
    ExecutionError,
    LicenseAcceptanceRequired,
    PreparationError,
)
from npu_rag.embedding.models.acquire import AcquiredModel
from npu_rag.embedding.profiles import ModelProfile, profile_for
from npu_rag.embedding.tokenize import (
    TOKENIZER_FILE_PATTERNS,
    EncodedBatch,
    ModelTokenizer,
    load_tokenizer,
)
from npu_rag.embedding.types import DocumentText, TextKind

GEMMA = profile_for("embeddinggemma-300m")
GTE = profile_for("gte-modernbert-base")
NOMIC = profile_for("nomic-embed-text-v1.5")

#: A resolved commit hash, shaped as `AcquiredModel` insists on.
SHA = "0" * 39 + "a"

#: Realistic prose, in the register the corpus this project indexes is written
#: in. Short enough to be legible, varied enough that no single length dominates.
SAMPLE_PROSE: tuple[str, ...] = (
    "The execution provider partitions the graph and runs whatever it cannot "
    "compile on the host processor, inside a session that still reports "
    "itself as accelerated.",
    "Padding every input to the compiled sequence length costs a full pass "
    "regardless of how little content the input carries, which is why short "
    "chunks are measured separately from long ones.",
    "Acceptance of the licence terms happens once, in a browser, and no "
    "number of retries of the download will substitute for it.",
    "A vector of the right width and the right norm can still be "
    "semantically wrong when a projection stage is skipped.",
    "Chunk sizing downstream is computed against the number reported here, "
    "so publishing the architectural limit rather than the compiled length "
    "would truncate three quarters of every chunk without raising anything.",
)

#: Words the boundary probes are grown from. Ordinary English, so the sample
#: stays prose rather than a repeated token.
FILLER: tuple[str, ...] = (
    "retrieval",
    "quality",
    "degrades",
    "quietly",
    "when",
    "text",
    "is",
    "shortened",
    "without",
    "anyone",
    "noticing",
    "the",
    "difference",
    "between",
    "measured",
    "and",
    "assumed",
    "behaviour",
)


# --------------------------------------------------------------------------
# Real, local tokenizers - built in process, never downloaded
# --------------------------------------------------------------------------


def _vocabulary() -> dict[str, int]:
    words = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "<bos>": 4}
    corpus = " ".join((*SAMPLE_PROSE, *FILLER, *_template_words()))
    for word in corpus.replace("|", " ").replace(":", " ").split():
        words.setdefault(word.strip(".,"), len(words))
    for symbol in (":", "|", ".", ","):
        words.setdefault(symbol, len(words))
    return words


def _template_words() -> tuple[str, ...]:
    return tuple(
        word
        for profile in (GEMMA, GTE, NOMIC)
        for template in (profile.document_template, profile.query_template)
        for word in template.replace("{title}", "").replace("{content}", "").split()
    )


def _backend(post: TemplateProcessing) -> Tokenizer:
    tokenizer = Tokenizer(WordLevel(_vocabulary(), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = post
    return tokenizer


def bracketing_tokenizer() -> PreTrainedTokenizerBase:
    """``[CLS] ... [SEP]``, the shape the BERT-derived candidates use."""
    post = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=_backend(post),
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        model_max_length=512,
    )


def prefixing_tokenizer() -> PreTrainedTokenizerBase:
    """A single leading token, the shape a Gemma-style tokenizer uses."""
    post = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 4)])
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=_backend(post),
        unk_token="[UNK]",
        bos_token="<bos>",
        pad_token="[PAD]",
        model_max_length=512,
    )


def unpadded_tokenizer() -> PreTrainedTokenizerBase:
    """No pad token declared at all."""
    post = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=_backend(post),
        unk_token="[UNK]",
        model_max_length=512,
    )


def model_tokenizer(
    profile: ModelProfile = GTE,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> ModelTokenizer:
    return ModelTokenizer(
        profile=profile,
        tokenizer=bracketing_tokenizer() if tokenizer is None else tokenizer,
        tokenizer_id=f"{profile.model_id}@{SHA}",
    )


@pytest.fixture
def documents() -> ModelTokenizer:
    return model_tokenizer(GEMMA)


@pytest.fixture
def plain() -> ModelTokenizer:
    return model_tokenizer(GTE)


def content_counting(
    subject: ModelTokenizer, target: int, kind: TextKind = TextKind.DOCUMENT
) -> str:
    """Prose whose *rendered* token count is exactly ``target``.

    Every filler word is one token under this vocabulary - known words map to
    themselves and unknown ones to ``[UNK]`` - so the length is reached by
    appending, not by guessing.

    **Bounded, and required to make progress on every step.** The regression
    this suite exists to catch is a measurement that silently truncates - the
    exact divergence between counting and truncation requirement 3.8 forbids -
    and under that defect the count plateaus at the maximum instead of rising.
    An unbounded ``while`` here would spin on that plateau forever, so the most
    valuable failure this suite can produce would arrive as a hang. A hang on
    CI reads as infrastructure flake and gets re-run; a failure gets read.
    """
    words = list(FILLER)
    text = " ".join(words)
    count = subject.count_tokens(text, kind)
    if count > target:
        raise AssertionError(
            f"cannot build a text of exactly {target} tokens; the shortest "
            f"filler already counts {count}"
        )
    # Every appended word costs at least one token, so this many steps is more
    # than enough to reach any reachable target.
    for _ in range(target + len(FILLER)):
        if count == target:
            return text
        words.append(FILLER[len(words) % len(FILLER)])
        text = " ".join(words)
        grown = subject.count_tokens(text, kind)
        if grown <= count:
            _plateaued(grown, target)
        count = grown
    _plateaued(count, target)


def _plateaued(count: int, target: int) -> NoReturn:
    pytest.fail(
        f"count_tokens plateaued at {count} while building a {target}-token "
        "input; counting and truncation may have diverged"
    )


def reference_count(
    tokenizer: PreTrainedTokenizerBase, rendered: str, *, specials: bool = True
) -> int:
    """The count computed independently of the module under test."""
    return len(
        tokenizer.encode(rendered, add_special_tokens=specials, verbose=False)
    )


# --------------------------------------------------------------------------
# Requirement 3.7 - the tokenizer is exposed, and stably identified
# --------------------------------------------------------------------------


def test_the_active_models_tokenizer_is_the_object_a_consumer_receives() -> None:
    tokenizer = bracketing_tokenizer()

    subject = model_tokenizer(GTE, tokenizer)

    assert subject.tokenizer is tokenizer


def test_the_tokenizer_identity_names_the_repository_and_the_resolved_commit(
    tmp_path: Path,
) -> None:
    """design.md's Revalidation Triggers make the tokenizer identity a signal
    downstream specs re-check against, so it must move when the tokenizer does.
    A repository name alone would compare equal to itself forever."""
    acquired = AcquiredModel(
        model_id=GTE.model_id, revision=SHA, local_path=tmp_path
    )

    subject = load_tokenizer(
        GTE,
        acquire=_fixed_acquisition(acquired),
        loader=lambda _path: bracketing_tokenizer(),
    )

    assert subject.tokenizer_id == f"{GTE.model_id}@{SHA}"


def test_loading_asks_acquisition_for_tokenizer_files_and_not_for_weights(
    tmp_path: Path,
) -> None:
    recorded: dict[str, Any] = {}

    def acquire(profile: ModelProfile, **kwargs: Any) -> AcquiredModel:
        recorded.update(kwargs)
        return AcquiredModel(
            model_id=profile.model_id, revision=SHA, local_path=tmp_path
        )

    load_tokenizer(
        GTE, acquire=acquire, loader=lambda _path: bracketing_tokenizer()
    )

    patterns = recorded["allow_patterns"]
    assert patterns is not None
    assert "tokenizer.json" in patterns
    assert "tokenizer_config.json" in patterns
    assert not any(
        pattern.endswith((".safetensors", ".bin", ".onnx")) for pattern in patterns
    )
    assert set(TOKENIZER_FILE_PATTERNS) == set(patterns)


def test_a_gated_tokenizer_reports_the_licence_step_rather_than_a_download_error(
    tmp_path: Path,
) -> None:
    """The acquisition path already turns a closed gate into requirement 4.5's
    error. Re-implementing acquisition here would have lost that."""

    def acquire(profile: ModelProfile, **_: Any) -> AcquiredModel:
        raise LicenseAcceptanceRequired(
            "gated",
            acceptance_url="https://huggingface.co/x",
            model_id=profile.model_id,
        )

    with pytest.raises(LicenseAcceptanceRequired):
        load_tokenizer(
            GEMMA, acquire=acquire, loader=lambda _path: bracketing_tokenizer()
        )


def test_a_tokenizer_that_will_not_load_is_a_preparation_failure_naming_a_stage(
    tmp_path: Path,
) -> None:
    def loader(_path: Path) -> PreTrainedTokenizerBase:
        raise OSError("no tokenizer_config.json in that directory")

    with pytest.raises(PreparationError) as raised:
        load_tokenizer(
            GTE,
            acquire=_fixed_acquisition(
                AcquiredModel(
                    model_id=GTE.model_id, revision=SHA, local_path=tmp_path
                )
            ),
            loader=loader,
        )

    assert raised.value.model_id == GTE.model_id
    assert raised.value.stage
    assert raised.value.stage != "unspecified"


def _fixed_acquisition(acquired: AcquiredModel) -> Any:
    def acquire(_profile: ModelProfile, **_: Any) -> AcquiredModel:
        return acquired

    return acquire


# --------------------------------------------------------------------------
# Requirement 3.6 - the published maximum is the compiled length
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", [GEMMA, GTE, NOMIC])
def test_the_published_maximum_is_the_compiled_length(
    profile: ModelProfile,
) -> None:
    subject = model_tokenizer(profile)

    assert subject.max_input_tokens == profile.compiled_seq_len


@pytest.mark.parametrize("profile", [GEMMA, NOMIC])
def test_the_published_maximum_is_never_the_architectural_limit(
    profile: ModelProfile,
) -> None:
    """Publishing 2048 for EmbeddingGemma or 8192 for nomic would let a
    consumer build chunks that are silently truncated at 512."""
    subject = model_tokenizer(profile)

    assert profile.architectural_context_limit > profile.compiled_seq_len
    assert subject.max_input_tokens != profile.architectural_context_limit


# --------------------------------------------------------------------------
# Counting: the rendered text, with its special tokens
# --------------------------------------------------------------------------


def test_counting_measures_the_rendered_text_not_the_raw_text(
    documents: ModelTokenizer,
) -> None:
    """The template consumes budget the caller cannot see. A count of the raw
    text would under-report, and the runtime would truncate an input the
    consumer had measured as fitting."""
    content = "the quick brown fox"

    counted = documents.count_tokens(DocumentText(content), TextKind.DOCUMENT)

    raw = reference_count(documents.tokenizer, content)
    rendered = reference_count(
        documents.tokenizer, GEMMA.render_document(DocumentText(content))
    )
    assert counted == rendered
    assert counted > raw


def test_counting_includes_the_tokenizers_special_tokens(
    documents: ModelTokenizer,
) -> None:
    rendered = GEMMA.render_document(DocumentText("the quick brown fox"))

    counted = documents.count_tokens(
        DocumentText("the quick brown fox"), TextKind.DOCUMENT
    )

    without = reference_count(documents.tokenizer, rendered, specials=False)
    assert counted == without + documents.tokenizer.num_special_tokens_to_add()
    assert counted > without


def test_a_title_costs_tokens_when_the_model_has_a_title_slot(
    documents: ModelTokenizer,
) -> None:
    untitled = documents.count_tokens(
        DocumentText("the quick brown fox"), TextKind.DOCUMENT
    )

    titled = documents.count_tokens(
        DocumentText("the quick brown fox", title="retrieval quality behaviour"),
        TextKind.DOCUMENT,
    )

    assert GEMMA.has_title_slot
    assert titled > untitled


def test_a_document_given_as_a_bare_string_counts_as_an_untitled_document(
    documents: ModelTokenizer,
) -> None:
    assert documents.count_tokens(
        "the quick brown fox", TextKind.DOCUMENT
    ) == documents.count_tokens(
        DocumentText("the quick brown fox"), TextKind.DOCUMENT
    )


def test_counting_a_query_applies_the_query_convention(
    documents: ModelTokenizer,
) -> None:
    query = "the quick brown fox"

    counted = documents.count_tokens(query, TextKind.QUERY)

    assert counted == reference_count(
        documents.tokenizer, GEMMA.render_query(query)
    )


def test_the_same_text_counts_differently_as_a_document_and_as_a_query(
    documents: ModelTokenizer,
) -> None:
    """Valid *only* because this fixture's vocabulary is controlled: under it
    every template word is exactly one token, and EmbeddingGemma's two
    conventions have different word counts.

    Do not copy this ``!=`` into a test against a real tokenizer. Whether two
    conventions cost different numbers of tokens is a property of the
    vocabulary, not of this module: nomic's ``search_document: `` and
    ``search_query: `` both tokenize to four tokens, so the counts coincide
    while the rendered text differs. ``test_tokenize_live.py`` compares each
    count to its own reference for exactly that reason.
    """
    text = "the quick brown fox"

    assert documents.count_tokens(
        text, TextKind.DOCUMENT
    ) != documents.count_tokens(text, TextKind.QUERY)


def test_a_document_with_no_content_still_costs_its_template(
    documents: ModelTokenizer,
) -> None:
    """``DocumentText`` deliberately leaves content unvalidated - what counts as
    usable input is a service-level judgement - so an empty document is a text
    that renders to the template alone. It costs tokens, and it fits."""
    empty = DocumentText("")

    counted = documents.count_tokens(empty, TextKind.DOCUMENT)

    assert counted == reference_count(
        documents.tokenizer, GEMMA.render_document(empty)
    )
    assert counted > documents.tokenizer.num_special_tokens_to_add()
    assert not documents.exceeds_limit(empty, TextKind.DOCUMENT)

    batch = documents.encode_documents([empty])

    assert batch.truncated_indices == ()
    assert batch.token_counts == (counted,)
    assert int(batch.attention_mask[0].sum()) == counted


def test_a_document_value_cannot_be_measured_as_a_query(
    documents: ModelTokenizer,
) -> None:
    with pytest.raises(ValueError):
        documents.count_tokens(DocumentText("x", title="y"), TextKind.QUERY)


# --------------------------------------------------------------------------
# Requirement 3.8 - the if and only if
# --------------------------------------------------------------------------


def test_an_input_of_exactly_the_maximum_is_within_limits(
    plain: ModelTokenizer,
) -> None:
    text = content_counting(plain, plain.max_input_tokens)

    assert plain.count_tokens(text, TextKind.DOCUMENT) == plain.max_input_tokens
    assert not plain.exceeds_limit(text, TextKind.DOCUMENT)
    assert plain.encode_documents([text]).truncated_indices == ()


def test_an_input_one_token_over_the_maximum_is_truncated(
    plain: ModelTokenizer,
) -> None:
    text = content_counting(plain, plain.max_input_tokens + 1)

    assert plain.exceeds_limit(text, TextKind.DOCUMENT)
    assert plain.encode_documents([text]).truncated_indices == (0,)


def test_counting_and_truncation_agree_in_both_directions_over_real_prose(
    plain: ModelTokenizer,
) -> None:
    """Task 5.1's Observable. Not "the counter is conservative" - agreement,
    in both directions, over a sample that contains inputs on each side of the
    boundary as well as inputs sitting exactly on it."""
    limit = plain.max_input_tokens
    sample: list[str] = [
        *SAMPLE_PROSE,
        content_counting(plain, limit - 1),
        content_counting(plain, limit),
        content_counting(plain, limit + 1),
        content_counting(plain, limit * 2),
    ]

    batch = plain.encode_documents(sample)

    over = {
        index
        for index, text in enumerate(sample)
        if plain.count_tokens(text, TextKind.DOCUMENT) > limit
    }
    assert set(batch.truncated_indices) == over
    # Non-vacuity: the sample must exercise both sides, or "agreement" is the
    # agreement of two empty sets.
    assert over
    assert len(over) < len(sample)


def test_counting_and_truncation_agree_for_titled_documents(
    documents: ModelTokenizer,
) -> None:
    limit = documents.max_input_tokens
    body = content_counting(documents, limit - 4)
    sample = [
        DocumentText(body),
        DocumentText(body, title="a title long enough to push this over"),
    ]

    batch = documents.encode_documents(sample)

    over = {
        index
        for index, text in enumerate(sample)
        if documents.count_tokens(text, TextKind.DOCUMENT) > limit
    }
    assert set(batch.truncated_indices) == over
    assert over == {1}


def test_a_batch_cannot_record_a_truncation_its_counts_do_not_support() -> None:
    """3.8 made structural: the value object refuses to carry a truncation
    claim that disagrees with the measurement it was built from."""
    ids = np.zeros((2, 4), dtype=np.int64)
    mask = np.zeros((2, 4), dtype=np.int64)
    mask[0, :2] = 1
    mask[1, :4] = 1

    with pytest.raises(ValueError):
        EncodedBatch(
            token_ids=ids,
            attention_mask=mask,
            token_counts=(2, 9),
            max_input_tokens=4,
            truncated_indices=(),
        )


def test_a_batch_cannot_claim_a_truncation_that_did_not_happen() -> None:
    ids = np.zeros((1, 4), dtype=np.int64)
    mask = np.zeros((1, 4), dtype=np.int64)
    mask[0, :2] = 1

    with pytest.raises(ValueError):
        EncodedBatch(
            token_ids=ids,
            attention_mask=mask,
            token_counts=(2,),
            max_input_tokens=4,
            truncated_indices=(0,),
        )


def test_a_batch_cannot_carry_a_mask_that_disagrees_with_its_counts() -> None:
    ids = np.zeros((1, 4), dtype=np.int64)
    mask = np.zeros((1, 4), dtype=np.int64)
    mask[0, :3] = 1

    with pytest.raises(ValueError):
        EncodedBatch(
            token_ids=ids,
            attention_mask=mask,
            token_counts=(2,),
            max_input_tokens=4,
            truncated_indices=(),
        )


# --------------------------------------------------------------------------
# Encoding to the pinned model input shape
# --------------------------------------------------------------------------


def test_encoding_produces_the_compiled_shape_and_dtype(
    plain: ModelTokenizer,
) -> None:
    batch = plain.encode_documents(list(SAMPLE_PROSE))

    expected = (len(SAMPLE_PROSE), GTE.compiled_seq_len)
    assert batch.token_ids.shape == expected
    assert batch.attention_mask.shape == expected
    assert batch.token_ids.dtype == np.int64
    assert batch.attention_mask.dtype == np.int64


def test_the_attention_mask_marks_exactly_the_real_tokens(
    plain: ModelTokenizer,
) -> None:
    limit = plain.max_input_tokens
    sample = [SAMPLE_PROSE[0], content_counting(plain, limit + 50)]

    batch = plain.encode_documents(sample)

    for row, count in enumerate(batch.token_counts):
        kept = min(count, limit)
        assert int(batch.attention_mask[row].sum()) == kept
        assert bool(batch.attention_mask[row, :kept].all())
        assert not bool(batch.attention_mask[row, kept:].any())


def test_padding_uses_the_tokenizers_own_pad_token(
    plain: ModelTokenizer,
) -> None:
    batch = plain.encode_documents([SAMPLE_PROSE[0]])

    kept = int(batch.attention_mask[0].sum())
    pad_id = plain.tokenizer.pad_token_id
    assert pad_id is not None
    assert set(batch.token_ids[0, kept:].tolist()) == {pad_id}


def test_padding_falls_back_to_zero_when_no_pad_token_is_declared() -> None:
    subject = model_tokenizer(GTE, unpadded_tokenizer())

    batch = subject.encode_documents([SAMPLE_PROSE[0]])

    kept = int(batch.attention_mask[0].sum())
    assert subject.tokenizer.pad_token_id is None
    assert set(batch.token_ids[0, kept:].tolist()) == {0}


def test_a_truncated_row_keeps_the_models_closing_special_token(
    plain: ModelTokenizer,
) -> None:
    """Truncation goes through the tokenizer's own rule, which drops content
    tokens and keeps the terminator. A raw slice of the encoded ids would end
    the sequence mid-sentence with no ``[SEP]`` - the same length, different
    text, and nothing downstream would notice."""
    text = content_counting(plain, plain.max_input_tokens + 25)

    batch = plain.encode_documents([text])

    assert batch.truncated_indices == (0,)
    assert int(batch.token_ids[0, -1]) == plain.tokenizer.sep_token_id


def test_a_prefixing_tokenizer_is_handled_as_well_as_a_bracketing_one() -> None:
    """The tokenizer is model specific; three profiles, three tokenizers."""
    subject = model_tokenizer(GEMMA, prefixing_tokenizer())
    text = content_counting(
        subject, subject.max_input_tokens + 3, TextKind.DOCUMENT
    )

    batch = subject.encode_documents([text])

    assert subject.tokenizer.num_special_tokens_to_add() == 1
    assert batch.truncated_indices == (0,)
    assert int(batch.attention_mask[0].sum()) == subject.max_input_tokens


def test_the_batch_keeps_one_row_per_input_in_input_order(
    plain: ModelTokenizer,
) -> None:
    sample = list(SAMPLE_PROSE)

    batch = plain.encode_documents(sample)

    assert batch.token_counts == tuple(
        plain.count_tokens(text, TextKind.DOCUMENT) for text in sample
    )
    assert batch.batch_size == len(sample)


def test_an_empty_batch_is_still_shaped_for_the_compiled_length(
    plain: ModelTokenizer,
) -> None:
    batch = plain.encode_documents([])

    assert batch.token_ids.shape == (0, GTE.compiled_seq_len)
    assert batch.token_counts == ()
    assert batch.truncated_indices == ()


def test_queries_are_encoded_under_the_query_convention(
    documents: ModelTokenizer,
) -> None:
    query = "the quick brown fox"

    batch = documents.encode_queries([query])

    assert batch.token_counts == (
        reference_count(documents.tokenizer, GEMMA.render_query(query)),
    )


def test_encoding_the_same_text_twice_produces_identical_arrays(
    plain: ModelTokenizer,
) -> None:
    first = plain.encode_documents(list(SAMPLE_PROSE))
    second = plain.encode_documents(list(SAMPLE_PROSE))

    assert np.array_equal(first.token_ids, second.token_ids)
    assert np.array_equal(first.attention_mask, second.attention_mask)


def test_measuring_an_over_long_input_writes_nothing_to_the_console(
    plain: ModelTokenizer, capfd: pytest.CaptureFixture[str]
) -> None:
    """The tokenizer library warns about sequences longer than its declared
    maximum, on a handler bound to the real file descriptor. design.md's
    Monitoring decision is that callers choose presentation, so measurement
    must not narrate."""
    text = content_counting(plain, plain.max_input_tokens + 200)
    capfd.readouterr()

    plain.count_tokens(text, TextKind.DOCUMENT)
    plain.encode_documents([text])

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --------------------------------------------------------------------------
# The guard against a tokenizer that ignores the limit
# --------------------------------------------------------------------------


class _OverLongTokenizer:
    """A tokenizer whose truncation does not truncate.

    Not a hypothetical: ``truncation`` is a tokenizer-side setting, and a
    tokenizer configured with ``truncation_side`` or a truncation strategy that
    does not apply to a single sequence would hand back more ids than the graph
    has slots for. Silently trimming them would move the truncation decision
    away from the one place that reports it.
    """

    pad_token_id = 0

    def encode(self, text: str, **_: Any) -> list[int]:
        return list(range(1000))

    def num_special_tokens_to_add(self) -> int:
        return 0


def test_a_tokenizer_that_returns_more_ids_than_the_graph_has_slots_is_refused() -> (
    None
):
    subject = ModelTokenizer(
        profile=GTE,
        tokenizer=_OverLongTokenizer(),  # type: ignore[arg-type]
        tokenizer_id="broken@" + SHA,
    )

    with pytest.raises(ExecutionError):
        subject.encode_documents(["anything"])


# --------------------------------------------------------------------------
# Dependency direction (design.md, Architecture)
#
# The package-wide guard in tests/embedding/providers/test_base.py walks only
# modules whose top-level name appears in its layer table, and `tokenize` is not
# in design.md's chain. It is policed here instead: `tokenize` reads `models`
# and everything to its left, and nothing to its right.
# --------------------------------------------------------------------------

MODULE_PACKAGE = "npu_rag.embedding"
MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "npu_rag"
    / "embedding"
    / "tokenize.py"
)


def _imported_names(source: str, containing: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = containing.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            names.append(module)
            names.extend(f"{module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("later", ["providers", "service", "bench"])
def test_tokenize_imports_nothing_from_a_later_layer(later: str) -> None:
    names = _imported_names(
        MODULE_PATH.read_text(encoding="utf-8"), MODULE_PACKAGE
    )

    assert names, "the layer guard found no imports at all"
    assert not [
        name
        for name in names
        if name == f"{MODULE_PACKAGE}.{later}"
        or name.startswith(f"{MODULE_PACKAGE}.{later}.")
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "import npu_rag.embedding.providers.base",
        "from npu_rag.embedding.providers import base",
        "from .providers import base",
        "from .providers.base import resolve_backend",
    ],
)
def test_the_layer_guard_would_catch_an_inversion(statement: str) -> None:
    """Non-vacuity: the guard must recognise every spelling of the import it
    forbids, including the relative ones ``tokenize.py`` would have to use."""
    resolved = _imported_names(statement, MODULE_PACKAGE)

    assert any(
        name.startswith(f"{MODULE_PACKAGE}.providers") for name in resolved
    ), f"{statement!r} resolved to {resolved}"
