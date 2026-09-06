"""The length-measurement contract against the three real tokenizers (5.1).

``test_tokenize.py`` proves the logic with tokenizers built in process. This
file proves the same contract with the tokenizers the runtime will actually use,
because the facts that matter most here - how many tokens a real vocabulary
spends on the prefix template, how many special tokens each model adds - are
properties of the downloaded files and of nothing else.

**Nothing here downloads weights.** Acquisition is restricted to the tokenizer
files: roughly a megabyte for the two BERT-derived candidates and about forty
for EmbeddingGemma's Gemma vocabulary, cached by the Hub thereafter. The
candidates' ``model.safetensors`` are between 0.5 and 1.3 GB and are never
fetched.

The whole module skips when the Hub is unreachable, and the gated candidate
skips additionally when this machine holds no accepted credential for it, so
the suite stays green offline and on a runner with no secrets.
"""

from __future__ import annotations

from typing import NoReturn

import pytest
from huggingface_hub import HfApi

from npu_rag.embedding.errors import LicenseAcceptanceRequired
from npu_rag.embedding.profiles import PROFILES, ModelProfile, profile_for
from npu_rag.embedding.tokenize import ModelTokenizer, load_tokenizer
from npu_rag.embedding.types import DocumentText, TextKind

OPEN = profile_for("bge-large-en-v1.5")

#: Real prose, in the register of the archive this project indexes. Deliberately
#: varied: a fragment, an ordinary paragraph, and one long enough that no
#: candidate's 512-token window can hold it.
CORPUS: tuple[str, ...] = (
    "Partition share.",
    "The Vitis AI execution provider partitions an ONNX graph and runs the "
    "subgraphs it cannot compile on the host processor, inside a session that "
    "still reports itself as NPU backed. Session creation therefore proves "
    "nothing; only the node mix of the published context graph does.",
    "Accepting the Gemma terms is a one time action performed in a browser "
    "against a specific Hugging Face account. No number of retries of the "
    "download substitutes for it, which is why the runtime reports the "
    "acceptance step rather than a transport failure.",
    " ".join(
        (
            "Chunk sizing downstream is computed against the maximum this "
            "runtime publishes, and the number it publishes is the length the "
            "graph was compiled at rather than the length the architecture "
            "could attend to.",
        )
        * 40
    ),
)

TITLES: tuple[str | None, ...] = (
    None,
    "Partition verification",
    None,
    "Compiled length versus architectural limit",
)


def _hub_reachable() -> bool:
    try:
        HfApi().model_info(OPEN.model_id, token=False, timeout=10)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hub_reachable(), reason="the Hugging Face Hub is not reachable"
)


def _tokenizer(profile: ModelProfile) -> ModelTokenizer:
    try:
        return load_tokenizer(profile)
    except LicenseAcceptanceRequired as error:
        pytest.skip(f"{profile.name} is gated and not accepted here: {error}")


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_every_candidate_publishes_its_compiled_length_and_a_pinned_identity(
    name: str,
) -> None:
    profile = profile_for(name)

    subject = _tokenizer(profile)

    assert subject.max_input_tokens == profile.compiled_seq_len
    assert subject.max_input_tokens != profile.architectural_context_limit or (
        profile.architectural_context_limit == profile.compiled_seq_len
    )
    repository, _, revision = subject.tokenizer_id.partition("@")
    assert repository == profile.model_id
    assert len(revision) == 40


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_the_real_tokenizer_counts_the_prefix_template_and_its_special_tokens(
    name: str,
) -> None:
    profile = profile_for(name)
    subject = _tokenizer(profile)
    content = CORPUS[1]

    counted = subject.count_tokens(DocumentText(content), TextKind.DOCUMENT)

    bare = len(
        subject.tokenizer.encode(content, add_special_tokens=False, verbose=False)
    )
    rendered = len(
        subject.tokenizer.encode(
            profile.render_document(DocumentText(content)),
            add_special_tokens=True,
            verbose=False,
        )
    )
    assert counted == rendered
    assert counted > bare


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_counting_and_truncation_agree_in_both_directions_on_real_content(
    name: str,
) -> None:
    """Task 5.1's Observable against the tokenizer the runtime really uses.

    An if and only if: every input the consumer measures as over the published
    maximum is truncated, and no input it measures as within is.
    """
    profile = profile_for(name)
    subject = _tokenizer(profile)
    limit = subject.max_input_tokens
    sample = [
        DocumentText(content, title=title)
        for content, title in zip(CORPUS, TITLES, strict=True)
    ]

    batch = subject.encode_documents(sample)

    over = {
        index
        for index, document in enumerate(sample)
        if subject.count_tokens(document, TextKind.DOCUMENT) > limit
    }
    assert set(batch.truncated_indices) == over
    assert over, "the sample must contain an over-long input"
    assert len(over) < len(sample), "the sample must contain a fitting input"
    assert batch.token_ids.shape == (len(sample), limit)


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_a_query_and_a_document_of_the_same_text_measure_by_their_own_rules(
    name: str,
) -> None:
    profile = profile_for(name)
    subject = _tokenizer(profile)
    text = CORPUS[0]

    as_query = subject.count_tokens(text, TextKind.QUERY)
    as_document = subject.count_tokens(text, TextKind.DOCUMENT)

    assert as_query == len(
        subject.tokenizer.encode(
            profile.render_query(text), add_special_tokens=True, verbose=False
        )
    )
    assert as_document == len(
        subject.tokenizer.encode(
            profile.render_document(DocumentText(text)),
            add_special_tokens=True,
            verbose=False,
        )
    )
    # The two conventions render different text. Whether they happen to cost
    # the same number of tokens is a property of the vocabulary - nomic's
    # `search_document: ` and `search_query: ` tokenize to equal lengths - so
    # the counts are compared to their own references, never to each other.
    assert profile.render_query(text) != profile.render_document(
        DocumentText(text)
    )


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_an_input_at_the_published_maximum_is_not_truncated(name: str) -> None:
    """The exact boundary, on a real vocabulary: at the limit is within limits,
    one token past it is not."""
    profile = profile_for(name)
    subject = _tokenizer(profile)
    limit = subject.max_input_tokens

    words = CORPUS[3].split()
    at_limit = _trim_to(subject, words, limit)
    assert subject.count_tokens(at_limit, TextKind.DOCUMENT) == limit
    assert subject.encode_documents([at_limit]).truncated_indices == ()

    over = _grow_past(subject, at_limit, limit)
    assert subject.count_tokens(over, TextKind.DOCUMENT) > limit
    assert subject.encode_documents([over]).truncated_indices == (0,)


def _trim_to(subject: ModelTokenizer, words: list[str], limit: int) -> str:
    """The longest word prefix of ``words`` counting exactly ``limit`` tokens.

    Real sub-word vocabularies do not spend one token per word, so the length
    is searched for rather than computed. A word boundary that lands on the
    limit exactly always exists here because the shortest prefixes count well
    under it and each added word costs at least one token.

    Bounded, for the reason ``content_counting`` in ``test_tokenize.py`` is: a
    measurement that truncates makes the count plateau at the maximum, and an
    unbounded search would hang on exactly the defect requirement 3.8 exists to
    catch instead of reporting it.
    """
    low, high = 0, len(words)
    while low < high:
        middle = (low + high + 1) // 2
        if subject.count_tokens(" ".join(words[:middle]), TextKind.DOCUMENT) <= limit:
            low = middle
        else:
            high = middle - 1
    text = " ".join(words[:low])
    count = subject.count_tokens(text, TextKind.DOCUMENT)
    for _ in range(limit):
        if count == limit:
            return text
        text = f"{text} a"
        grown = subject.count_tokens(text, TextKind.DOCUMENT)
        if grown <= count:
            _plateaued(grown, limit)
        count = grown
    _plateaued(count, limit)


def _grow_past(subject: ModelTokenizer, text: str, limit: int) -> str:
    grown = text
    count = subject.count_tokens(grown, TextKind.DOCUMENT)
    for _ in range(limit):
        if count > limit:
            return grown
        grown = f"{grown} a"
        raised = subject.count_tokens(grown, TextKind.DOCUMENT)
        if raised <= count:
            _plateaued(raised, limit + 1)
        count = raised
    _plateaued(count, limit + 1)


def _plateaued(count: int, target: int) -> NoReturn:
    pytest.fail(
        f"count_tokens plateaued at {count} while building a {target}-token "
        "input; counting and truncation may have diverged"
    )
