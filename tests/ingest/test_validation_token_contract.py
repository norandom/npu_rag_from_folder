"""Validation 8.3: Token contract with the runtime, both directions.

Requirements: 6.7.
Boundary: token contract.
Depends: 1.6.

Observable: the sample contains inputs on both sides of the budget, and for every
one the ingest measurement and the runtime's truncated-index set agree — with a
non-vacuity control asserting the sample is not all on one side.
"""

from __future__ import annotations

import pytest

from npu_rag.embedding.bench.corpus import default_corpus_directory, load_corpus
from npu_rag.embedding.errors import LicenseAcceptanceRequired
from npu_rag.embedding.profiles import profile_for
from npu_rag.embedding.tokenize import ModelTokenizer, load_tokenizer
from npu_rag.embedding.types import DocumentText, TextKind


def _build_evaluation_sample(
    tokenizer: ModelTokenizer,
) -> tuple[list[DocumentText], int, int]:
    """Build an evaluation sample drawn from the committed benchmark corpus
    containing inputs on both sides of the tokenizer budget.
    """
    corpus_dir = default_corpus_directory()
    if not corpus_dir.is_dir():
        pytest.fail(f"Committed benchmark corpus fixture missing at {corpus_dir}")

    corpus = load_corpus(corpus_dir)
    budget = tokenizer.max_input_tokens

    sample: list[DocumentText] = []

    # 1. Within-budget inputs: single chunks from corpus
    for chunk in corpus.chunks[:15]:
        sample.append(DocumentText(content=chunk.text, title=chunk.article_title))

    # 2. Over-budget inputs: concatenated chunks from corpus
    chunks = corpus.chunks
    for i in range(0, min(len(chunks) - 10, 40), 10):
        combined_text = "\n\n".join(c.text for c in chunks[i : i + 10])
        sample.append(DocumentText(content=combined_text, title=chunks[i].article_title))

    # Add a large concatenation to guarantee exceeding budget
    giant_text = "\n\n".join(c.text for c in chunks[:25])
    sample.append(DocumentText(content=giant_text, title="Giant Document"))

    # Non-vacuity check: verify sample contains inputs on both sides
    within_count = 0
    over_count = 0
    for doc in sample:
        count = tokenizer.count_tokens(doc, TextKind.DOCUMENT)
        if count <= budget:
            within_count += 1
        else:
            over_count += 1

    return sample, within_count, over_count


def test_token_contract_unconditional_offline_fixture(
    runtime_tokenizer: ModelTokenizer,
) -> None:
    """Requirement 6.7: Bidirectional token contract against the offline tokenizer fixture.

    Unconditional run: fails rather than skips if files are absent.
    Measurement within budget <=> runtime does not truncate.
    Measurement over budget <=> runtime truncates.
    """
    sample, within_count, over_count = _build_evaluation_sample(runtime_tokenizer)

    # Non-vacuity control: sample is not all on one side
    assert within_count > 0, "Non-vacuity failure: sample has no within-budget inputs"
    assert over_count > 0, "Non-vacuity failure: sample has no over-budget inputs"

    # Encode documents through the runtime
    batch = runtime_tokenizer.encode_documents(sample)
    truncated_set = set(batch.truncated_indices)
    budget = runtime_tokenizer.max_input_tokens

    # Verify bidirectional contract for every input
    for index, doc in enumerate(sample):
        count = runtime_tokenizer.count_tokens(doc, TextKind.DOCUMENT)
        is_over_budget = count > budget
        is_truncated = index in truncated_set

        # Direction 1: within budget implies runtime does not truncate
        if not is_over_budget:
            assert not is_truncated, (
                f"Input {index} ({count} tokens <= budget {budget}) was unexpectedly "
                f"truncated by runtime"
            )

        # Direction 2: over budget implies runtime does truncate
        if is_over_budget:
            assert is_truncated, (
                f"Input {index} ({count} tokens > budget {budget}) was not "
                f"marked truncated by runtime"
            )

        # Both directions equivalence
        assert is_over_budget == is_truncated


def test_token_contract_opt_in_gated_default_model() -> None:
    """Requirement 6.7: Bidirectional token contract against the gated default model
    (embeddinggemma-300m), running only when terms are accepted.
    """
    gemma_profile = profile_for("embeddinggemma-300m")
    try:
        gemma_tokenizer = load_tokenizer(gemma_profile)
    except LicenseAcceptanceRequired as err:
        pytest.skip(f"embeddinggemma-300m is gated and license terms not accepted: {err}")
    except Exception as err:
        pytest.skip(f"embeddinggemma-300m offline/hub unavailable: {err}")

    sample, within_count, over_count = _build_evaluation_sample(gemma_tokenizer)

    assert within_count > 0, "Non-vacuity failure: sample has no within-budget inputs"
    assert over_count > 0, "Non-vacuity failure: sample has no over-budget inputs"

    batch = gemma_tokenizer.encode_documents(sample)
    truncated_set = set(batch.truncated_indices)
    budget = gemma_tokenizer.max_input_tokens

    for index, doc in enumerate(sample):
        count = gemma_tokenizer.count_tokens(doc, TextKind.DOCUMENT)
        is_over_budget = count > budget
        is_truncated = index in truncated_set
        assert is_over_budget == is_truncated
