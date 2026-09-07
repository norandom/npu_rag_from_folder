"""The embedding service: what it composes, and what it must not lose (task 5.3).

Two fixture decisions here are load-bearing, and both exist because of the
standing lesson in tasks.md about vacuous fixtures.

**``batch_size`` is 3, not 1.** All three shipping profiles compile at batch 1,
where every batch is full, no padding is ever added, and no padded row ever has
to be discarded. A service that forgot to pad a short final batch, or that
returned the padding rows as though they were vectors, would pass a whole suite
built on those profiles. The test profile therefore compiles at 3 and most
batches here are deliberately *not* a multiple of 3.

**The two templates differ in every word.** ``passage title: ... body: ...``
against ``search query: ...``. A service that applied the document convention to
a query - requirement 3.4's exact failure - changes the rendered string, so it
changes the token ids, so it changes the vector. With interchangeable templates
that mistake produces identical vectors and no test can see it.

The fake backend's output is a function of the token ids it was handed, for the
same reason: a backend returning a constant would make input order, template
selection, title rendering and truncation all invisible at once.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from npu_rag.embedding import service as service_module
from npu_rag.embedding.errors import ExecutionError, NpuUnavailableError
from npu_rag.embedding.models.artifacts import (
    ArtifactIdentity,
    ArtifactManifest,
    PreparedArtifact,
)
from npu_rag.embedding.models.export import (
    ACTIVATION_PREFIX,
    BIAS_PREFIX,
    DENSE_FILENAME,
    DENSE_ORDER_KEY,
    ONNX_FILENAME,
    WEIGHT_PREFIX,
)
from npu_rag.embedding.postprocess import DenseLayer
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.providers.base import BackendFactories
from npu_rag.embedding.providers.vitisai import MINIMUM_PARTITION_SHARE
from npu_rag.embedding.reporting import ProgressUpdate, RunSummary
from npu_rag.embedding.service import (
    EMBED_STAGE,
    EmbeddingContract,
    EmbeddingService,
    EmbedResult,
    build_service,
    default_backend_builder,
    load_dense_layers,
)
from npu_rag.embedding.tokenize import ModelTokenizer
from npu_rag.embedding.types import (
    CONDITION_PROVIDER_REGISTERED,
    CapabilityReport,
    Condition,
    DocumentText,
    ExecutionMode,
    ProviderChoice,
    TextKind,
)

SHA = "0" * 40

#: Compiled at 3 so a partial final batch actually occurs. See the module
#: docstring: at batch 1 the padding path is unreachable and untestable.
BATCH = 3

#: Short enough that a handful of words overruns it, so requirement 3.9's
#: shortening path is reachable with readable fixtures rather than filler.
SEQ = 16

PROFILE = ModelProfile(
    model_id="test/three-at-a-time",
    dimension=4,
    compiled_seq_len=SEQ,
    architectural_context_limit=512,
    batch_size=BATCH,
    pooling="mean",
    has_dense_stage=False,
    document_template="passage title: {title} body: {content}",
    query_template="search query: {content}",
    license_gated=False,
)

SAMPLE = (
    "alpha beta gamma",
    "delta epsilon",
    "zeta eta theta iota",
    "kappa",
    "lambda mu nu xi",
)


# --------------------------------------------------------------------------
# A real tokenizer, built in process - never downloaded
# --------------------------------------------------------------------------


def _vocabulary() -> dict[str, int]:
    words = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}
    corpus = " ".join(
        (
            *SAMPLE,
            PROFILE.document_template,
            PROFILE.query_template,
            "none",
            "a title that is real",
            " ".join(f"filler{n}" for n in range(40)),
        )
    )
    for word in corpus.replace("{title}", " ").replace("{content}", " ").split():
        words.setdefault(word.strip(":"), len(words))
    words.setdefault(":", len(words))
    return words


def _tokenizer() -> PreTrainedTokenizerBase:
    backend = Tokenizer(WordLevel(_vocabulary(), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        model_max_length=SEQ,
    )


def model_tokenizer(profile: ModelProfile = PROFILE) -> ModelTokenizer:
    return ModelTokenizer(
        profile=profile,
        tokenizer=_tokenizer(),
        tokenizer_id=f"{profile.model_id}@{SHA}",
    )


# --------------------------------------------------------------------------
# Capability reports
# --------------------------------------------------------------------------


def capable() -> CapabilityReport:
    return CapabilityReport(
        conditions=(
            Condition(
                name=CONDITION_PROVIDER_REGISTERED,
                satisfied=True,
                observed="VitisAIExecutionProvider",
                required="VitisAIExecutionProvider",
                remediation=None,
            ),
        ),
        execution_mode=ExecutionMode.IN_PROCESS,
        driver_version="32.0.20102.3930",
        runtime_version="1.8.0",
        device_name="NPU Compute Accelerator Device",
        power_reporting_supported=True,
    )


def incapable() -> CapabilityReport:
    return CapabilityReport(
        conditions=(
            Condition(
                name=CONDITION_PROVIDER_REGISTERED,
                satisfied=False,
                observed="absent from get_available_providers()",
                required="VitisAIExecutionProvider",
                remediation="install onnxruntime-vitisai from pypi.amd.com",
            ),
        ),
        execution_mode=ExecutionMode.UNAVAILABLE,
        driver_version="32.0.20102.3930",
        runtime_version=None,
        device_name=None,
        power_reporting_supported=True,
    )


def isolated() -> CapabilityReport:
    """The verdict this machine reaches after a bare ``uv sync`` (task 5.4).

    The device is still enumerated by ``xrt-smi`` while the vendor group is gone
    from the interpreter, so the provider is unregistered: device present plus
    provider absent is exactly `ExecutionMode.ISOLATED`. It is a real, reachable
    state rather than a hypothetical one, and no isolated backend exists to
    serve it.
    """
    return CapabilityReport(
        conditions=(
            Condition(
                name=CONDITION_PROVIDER_REGISTERED,
                satisfied=False,
                observed="absent from get_available_providers()",
                required="VitisAIExecutionProvider",
                remediation="run `uv run python -m tools.provision_npu`",
            ),
        ),
        execution_mode=ExecutionMode.ISOLATED,
        driver_version="32.0.20102.3930",
        runtime_version=None,
        device_name="NPU Compute Accelerator Device",
        power_reporting_supported=True,
    )


# --------------------------------------------------------------------------
# A backend whose output depends on its input
# --------------------------------------------------------------------------


class FakeBackend:
    """Token embeddings derived from the token ids, so inputs stay distinct.

    ``embedding[b, t, h] = token_ids[b, t] + h``. Masked mean pooling over row
    ``b`` therefore yields ``[m, m+1, m+2, m+3]`` for that row's mean id ``m``,
    which points in a different direction for every distinct input even after
    normalization. A backend returning a constant would let a service scramble
    the batch, apply the wrong template or drop the title with every assertion
    still passing.
    """

    def __init__(
        self,
        provider: ProviderChoice = ProviderChoice.CPU,
        *,
        execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS,
        share: float | None = None,
        hidden: int = PROFILE.dimension,
        fail_on_call: int | None = None,
    ) -> None:
        self._provider = provider
        self._execution_mode = execution_mode
        self._share = share
        self._hidden = hidden
        self._fail_on_call = fail_on_call
        self.calls: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = []

    @property
    def provider(self) -> ProviderChoice:
        return self._provider

    @property
    def execution_mode(self) -> ExecutionMode:
        return self._execution_mode

    @property
    def npu_partition_share(self) -> float | None:
        return self._share

    def run(
        self,
        token_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        self.calls.append((token_ids.copy(), attention_mask.copy()))
        if self._fail_on_call is not None and len(self.calls) == self._fail_on_call:
            raise ExecutionError(
                "the fake backend was asked to fail",
                provider=self._provider,
                model_id=PROFILE.model_id,
                stage="session",
            )
        batch, seq = token_ids.shape
        out = np.empty((batch, seq, self._hidden), dtype=np.float32)
        for h in range(self._hidden):
            out[:, :, h] = token_ids.astype(np.float32) + float(h)
        return out


class RecordingBuilder:
    """Builds factories, remembering the selection it was asked to close over.

    Note 4.3 requires the NPU factory to be closed over the *caller's* choice,
    because `VitisAIBackend` fails on a weak partition under ``npu`` and merely
    records it under ``auto``. A builder that passed a constant would invert one
    of those two verdicts, so the recorded selections are asserted directly.
    """

    def __init__(
        self,
        npu: FakeBackend | None = None,
        cpu: FakeBackend | None = None,
    ) -> None:
        self._npu = npu
        self._cpu = cpu
        self.requested: list[ProviderChoice] = []
        self.built: list[str] = []

    def __call__(
        self, requested: ProviderChoice = ProviderChoice.NPU, /
    ) -> BackendFactories:
        # The default mirrors `BackendBuilder`'s, which declares the fail-closed
        # reading as part of the contract. No test here relies on it: the
        # service always passes the caller's selection explicitly, and
        # `test_the_npu_factory_is_closed_over_the_callers_selection` asserts
        # exactly that.
        self.requested.append(requested)

        def npu_factory(
            profile: ModelProfile, capability: CapabilityReport, /
        ) -> FakeBackend:
            self.built.append("npu")
            if self._npu is None:
                raise AssertionError("the NPU factory was not expected to run")
            return self._npu

        def cpu_factory(
            profile: ModelProfile, capability: CapabilityReport, /
        ) -> FakeBackend:
            self.built.append("cpu")
            if self._cpu is None:
                raise AssertionError("the CPU factory was not expected to run")
            return self._cpu

        return BackendFactories(npu=npu_factory, cpu=cpu_factory)


class FakeClock:
    """A clock that advances by a fixed step on every reading.

    `RunTracker` reads it three times - at construction, on entry, and on exit -
    so a step of 0.5 makes the run's elapsed time exactly 0.5 seconds. An exact
    number is the point: review found ``elapsed_seconds >= 0.0`` was satisfied
    by a hardcoded literal ``0.0``, which is requirement 8.6 asserted vacuously.
    """

    def __init__(self, step: float = 0.5) -> None:
        self.now = 0.0
        self._step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self._step
        return value


def service(
    *,
    npu: FakeBackend | None = None,
    cpu: FakeBackend | None = None,
    capability: CapabilityReport | None = None,
    profile: ModelProfile = PROFILE,
    dense: Sequence[DenseLayer] = (),
    builder: RecordingBuilder | None = None,
    clock: Callable[[], float] | None = None,
) -> EmbeddingService:
    if cpu is None and npu is None:
        cpu = FakeBackend(ProviderChoice.CPU)
    return EmbeddingService(
        profile=profile,
        capability=capability if capability is not None else capable(),
        tokenizer=model_tokenizer(profile),
        backends=builder if builder is not None else RecordingBuilder(npu, cpu),
        dense=dense,
        **({} if clock is None else {"clock": clock}),
    )


def documents(*texts: str) -> list[DocumentText]:
    return [DocumentText(content=text) for text in texts]


# --------------------------------------------------------------------------
# The published contract (3.6, 3.7)
# --------------------------------------------------------------------------


def test_the_contract_publishes_the_compiled_length_never_the_architectural_limit() -> (
    None
):
    contract = service().contract()

    assert contract.max_input_tokens == PROFILE.compiled_seq_len == SEQ
    assert contract.max_input_tokens != PROFILE.architectural_context_limit


def test_the_contract_reports_the_dimension_model_and_tokenizer() -> None:
    contract = service().contract()

    assert contract == EmbeddingContract(
        model_id=PROFILE.model_id,
        dimension=PROFILE.dimension,
        max_input_tokens=SEQ,
        tokenizer_id=f"{PROFILE.model_id}@{SHA}",
    )


def test_the_exposed_tokenizer_is_the_one_the_runtime_measures_with() -> None:
    subject = service()

    exposed = subject.tokenizer()
    text = "alpha beta gamma"
    rendered = PROFILE.render_query(text)

    assert len(exposed(rendered)["input_ids"]) == subject.count_tokens(
        text, TextKind.QUERY
    )


# --------------------------------------------------------------------------
# One vector per input, in order (3.1)
# --------------------------------------------------------------------------


def test_one_vector_per_input_in_input_order() -> None:
    subject = service()

    batched = subject.embed_documents(documents(*SAMPLE), ProviderChoice.CPU)
    singles = [
        service().embed_documents(documents(text), ProviderChoice.CPU).vectors[0]
        for text in SAMPLE
    ]

    assert batched.vectors.shape == (len(SAMPLE), PROFILE.dimension)
    for index, expected in enumerate(singles):
        np.testing.assert_allclose(
            batched.vectors[index], expected, rtol=1e-6, atol=1e-6
        )


def test_the_inputs_are_distinguishable_so_a_reordering_would_be_visible() -> None:
    """Non-vacuity guard for the order test above.

    If the sample texts produced equal vectors, a service that scrambled the
    batch would satisfy the ordering assertion. This asserts the fixture can
    still tell the rows apart.
    """
    vectors = service().embed_documents(documents(*SAMPLE), ProviderChoice.CPU).vectors

    for left in range(len(SAMPLE)):
        for right in range(left + 1, len(SAMPLE)):
            assert not np.allclose(vectors[left], vectors[right]), (
                f"sample texts {left} and {right} embed identically, so this "
                "fixture cannot see a reordered batch"
            )


# --------------------------------------------------------------------------
# Batching and padding - reachable only because the profile compiles at 3
# --------------------------------------------------------------------------


def test_a_short_final_batch_is_padded_to_the_compiled_batch_size() -> None:
    backend = FakeBackend(ProviderChoice.CPU)

    service(cpu=backend).embed_documents(documents(*SAMPLE), ProviderChoice.CPU)

    assert len(backend.calls) == 2, "five inputs at batch 3 is two forward passes"
    for token_ids, attention_mask in backend.calls:
        assert token_ids.shape == (BATCH, SEQ)
        assert attention_mask.shape == (BATCH, SEQ)


def test_the_padding_rows_of_a_short_batch_never_become_vectors() -> None:
    backend = FakeBackend(ProviderChoice.CPU)

    result = service(cpu=backend).embed_documents(
        documents(*SAMPLE[:4]), ProviderChoice.CPU
    )

    assert len(backend.calls) == 2
    assert backend.calls[1][0].shape[0] == BATCH
    assert result.vectors.shape[0] == 4
    assert result.input_count == 4


def test_a_padding_row_carries_no_attention_and_is_not_pooled() -> None:
    """A padded row has an all-zero mask, which masked mean pooling refuses.

    So a service that handed the padding through post-processing would raise
    rather than return a wrong number quietly. This pins the row is genuinely
    empty, which is what makes that refusal the backstop it is.
    """
    backend = FakeBackend(ProviderChoice.CPU)

    service(cpu=backend).embed_documents(documents(*SAMPLE[:4]), ProviderChoice.CPU)

    final_mask = backend.calls[1][1]
    assert final_mask[0].sum() > 0, "the one real row of the final batch"
    assert final_mask[1].sum() == 0
    assert final_mask[2].sum() == 0


def test_an_exactly_full_batch_adds_no_padding() -> None:
    backend = FakeBackend(ProviderChoice.CPU)

    result = service(cpu=backend).embed_documents(
        documents(*SAMPLE[:3]), ProviderChoice.CPU
    )

    assert len(backend.calls) == 1
    assert result.vectors.shape[0] == 3
    assert backend.calls[0][1].sum(axis=1).min() > 0, "every row is real"


# --------------------------------------------------------------------------
# Normalization (3.5)
# --------------------------------------------------------------------------


def test_every_returned_vector_is_unit_norm() -> None:
    result = service().embed_documents(documents(*SAMPLE), ProviderChoice.CPU)

    np.testing.assert_allclose(
        np.linalg.norm(result.vectors, axis=1),
        np.ones(len(SAMPLE)),
        rtol=1e-6,
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# Pooling comes from the active profile (4.1, task 5.5)
#
# Implementation Note 5.4: "a fixture can be non-vacuous against a system that
# was never assembled that way". `postprocess`'s own tests prove the CLS branch
# computes the right thing; only this proves the assembled service actually
# reaches it, because `ModelProfile.pooling` spent tasks 2.2 to 5.4 with no
# production consumer at all and a widened annotation alone would change
# nothing.
# --------------------------------------------------------------------------

#: A profile identical to `PROFILE` in every respect except the pooling rule, so
#: the only thing that can move the vectors is the rule itself.
CLS_PROFILE = dataclasses.replace(PROFILE, pooling="cls")


def _expected_from_first_position(
    token_ids: npt.NDArray[np.int64], width: int
) -> list[list[float]]:
    """CLS-pooled, normalized vectors, computed without the module under test.

    `FakeBackend` returns ``token_ids[b, t] + h``, so the first position of row
    ``b`` is ``[id, id+1, ...]``. Plain Python arithmetic on the ids the backend
    was actually handed - no NumPy reduction and nothing from ``postprocess``,
    so a shared mistake cannot survive in both (Implementation Note 5.1).
    """
    rows: list[list[float]] = []
    for row in token_ids.tolist():
        raw = [float(row[0]) + h for h in range(width)]
        length = math.sqrt(sum(value * value for value in raw))
        rows.append([value / length for value in raw])
    return rows


def test_the_service_pools_by_the_rule_the_active_profile_declares() -> None:
    backend = FakeBackend(ProviderChoice.CPU)

    result = service(cpu=backend, profile=CLS_PROFILE).embed_documents(
        documents(*SAMPLE[:3]), ProviderChoice.CPU
    )

    token_ids = backend.calls[0][0]
    np.testing.assert_allclose(
        result.vectors,
        _expected_from_first_position(token_ids, CLS_PROFILE.dimension),
        rtol=1e-6,
        atol=1e-6,
    )


def test_the_same_texts_embed_differently_under_the_two_pooling_rules() -> None:
    """Non-vacuity for the test above, and the sharpest assertion here.

    Both services differ in exactly one field. If the service ignored
    ``profile.pooling`` - which is what it did until task 5.5 - both would
    return the mean-pooled vectors and the test above would still pass, because
    the CLS reference would simply be wrong in a way nothing compared it to.
    Both results are unit-norm and the same shape, so only the numbers differ.
    """
    by_cls = service(profile=CLS_PROFILE).embed_documents(
        documents(*SAMPLE[:3]), ProviderChoice.CPU
    )
    by_mean = service(profile=PROFILE).embed_documents(
        documents(*SAMPLE[:3]), ProviderChoice.CPU
    )

    assert by_cls.vectors.shape == by_mean.vectors.shape
    assert not np.allclose(by_cls.vectors, by_mean.vectors)
    for row in range(by_cls.vectors.shape[0]):
        assert not np.allclose(by_cls.vectors[row], by_mean.vectors[row]), (
            f"row {row}'s first token must not coincide with its mean token, "
            "or that row cannot tell the two rules apart"
        )


def test_a_profile_declaring_an_unimplemented_rule_fails_rather_than_guesses() -> (
    None
):
    """Requirement 4.6's spirit at the pooling seam: no silent substitution.

    The refusal happens where the vectors would be produced, so it carries the
    provider and the model requirement 8.1 asks for.
    """
    unimplemented = dataclasses.replace(
        PROFILE,
        pooling="lasttoken",  # type: ignore[arg-type]
    )

    with pytest.raises(ExecutionError) as caught:
        service(profile=unimplemented).embed_documents(
            documents(*SAMPLE[:3]), ProviderChoice.CPU
        )

    assert "lasttoken" in str(caught.value)
    assert caught.value.model_id == PROFILE.model_id
    assert caught.value.provider is ProviderChoice.CPU


# --------------------------------------------------------------------------
# Per-kind conventions (3.2, 3.3, 3.4)
# --------------------------------------------------------------------------


def test_a_document_and_a_query_of_the_same_text_embed_differently() -> None:
    text = "alpha beta gamma"

    as_document = service().embed_documents(documents(text), ProviderChoice.CPU)
    as_query = service().embed_queries([text], ProviderChoice.CPU)

    assert not np.allclose(as_document.vectors[0], as_query.vectors[0]), (
        "the two conventions render different text, so they must embed "
        "differently (requirement 3.4)"
    )


def test_the_document_title_reaches_the_template() -> None:
    titled = service().embed_documents(
        [DocumentText(content="alpha beta gamma", title="a title that is real")],
        ProviderChoice.CPU,
    )
    untitled = service().embed_documents(
        documents("alpha beta gamma"), ProviderChoice.CPU
    )

    assert not np.allclose(titled.vectors[0], untitled.vectors[0])


def test_an_absent_title_renders_the_sentinel_rather_than_an_empty_slot() -> None:
    subject = service()

    assert subject.count_tokens(
        DocumentText(content="alpha"), TextKind.DOCUMENT
    ) == subject.count_tokens(
        DocumentText(content="alpha", title="none"), TextKind.DOCUMENT
    )


def test_there_is_no_entry_point_that_takes_the_kind_as_an_argument() -> None:
    """Requirement 3.3 made structural rather than checked at runtime."""
    public = {name for name in dir(EmbeddingService) if not name.startswith("_")}

    assert {"embed_documents", "embed_queries"} <= public
    assert "embed" not in public


# --------------------------------------------------------------------------
# Truncation (3.8, 3.9)
# --------------------------------------------------------------------------


def _over_long() -> str:
    return " ".join(f"filler{n}" for n in range(40))


def test_an_over_long_input_is_shortened_and_identified() -> None:
    texts = documents("alpha", _over_long(), "kappa")

    result = service().embed_documents(texts, ProviderChoice.CPU)

    assert result.truncated_indices == (1,)
    assert result.vectors.shape[0] == 3


def test_count_tokens_agrees_with_the_truncation_decision() -> None:
    subject = service()
    texts = [*SAMPLE, _over_long(), "alpha " * 5]

    result = subject.embed_documents(documents(*texts), ProviderChoice.CPU)
    measured = {
        index
        for index, text in enumerate(texts)
        if subject.count_tokens(DocumentText(content=text), TextKind.DOCUMENT)
        > subject.contract().max_input_tokens
    }

    assert measured == set(result.truncated_indices)


def test_the_truncation_fixture_actually_contains_both_kinds_of_input() -> None:
    """Non-vacuity guard: an empty or total truncation set proves nothing."""
    subject = service()
    texts = [*SAMPLE, _over_long(), "alpha " * 5]

    result = subject.embed_documents(documents(*texts), ProviderChoice.CPU)

    assert 0 < len(result.truncated_indices) < len(texts)


# --------------------------------------------------------------------------
# Provider selection and reporting (2.2-2.6)
# --------------------------------------------------------------------------


def test_the_serving_provider_is_reported_on_a_successful_operation() -> None:
    result = service(cpu=FakeBackend(ProviderChoice.CPU)).embed_documents(
        documents("alpha"), ProviderChoice.CPU
    )

    assert result.provider_served is ProviderChoice.CPU
    assert result.execution_mode is ExecutionMode.IN_PROCESS


def test_the_execution_mode_is_the_serving_backends_not_the_environments() -> None:
    """Requirement 5.3, on a run where the two sources disagree.

    Every fixture had `capable().execution_mode` equal to the fake backend's,
    so reading the mode off the capability report instead of off the backend
    that served was invisible - a vacuous fixture in the standing lesson's exact
    sense. Under `incapable()` the report says UNAVAILABLE while a CPU backend
    serves in-process, which is the pair that discriminates. It is also the
    shape requirement 5.3 cares about: an isolated backend must be reported as
    isolated for every operation it serves, whatever the environment said.
    """
    summaries: list[RunSummary] = []
    backend = FakeBackend(
        ProviderChoice.CPU, execution_mode=ExecutionMode.ISOLATED
    )

    result = service(cpu=backend, capability=incapable()).embed_documents(
        documents("alpha"), ProviderChoice.AUTO, on_finish=summaries.append
    )

    assert incapable().execution_mode is ExecutionMode.UNAVAILABLE
    assert result.execution_mode is ExecutionMode.ISOLATED
    assert summaries[0].execution_mode is ExecutionMode.ISOLATED


def test_the_result_and_its_summary_agree_on_every_shared_field() -> None:
    """Note 2.3: `EmbedResult` and `RunSummary` duplicate four fields verbatim.

    5.3 gives them one producer by reading all four off the summary rather than
    re-deriving them. This pins that; two sources that agree today are two
    sources that can drift tomorrow.
    """
    summaries: list[RunSummary] = []

    result = service(
        cpu=FakeBackend(ProviderChoice.CPU), capability=incapable()
    ).embed_documents(
        documents(*SAMPLE), ProviderChoice.AUTO, on_finish=summaries.append
    )
    summary = summaries[0]

    assert result.provider_served is summary.provider_served
    assert result.execution_mode is summary.execution_mode
    assert result.input_count == summary.input_count
    assert result.elapsed_seconds == summary.elapsed_seconds
    assert result.fallback_reason == summary.fallback_reason


def test_explicit_npu_on_an_unusable_npu_fails_and_embeds_nothing() -> None:
    backend = FakeBackend(ProviderChoice.CPU)

    with pytest.raises(NpuUnavailableError) as raised:
        service(cpu=backend, capability=incapable()).embed_documents(
            documents("alpha"), ProviderChoice.NPU
        )

    assert "install onnxruntime-vitisai" in str(raised.value)
    assert backend.calls == [], "no CPU work may be done for an NPU request"


def test_auto_reports_the_specific_reason_before_falling_back_to_cpu() -> None:
    result = service(
        cpu=FakeBackend(ProviderChoice.CPU), capability=incapable()
    ).embed_documents(documents("alpha"), ProviderChoice.AUTO)

    assert result.provider_served is ProviderChoice.CPU
    assert result.fallback_reason is not None
    assert "install onnxruntime-vitisai" in result.fallback_reason


def test_auto_prefers_the_npu_where_it_is_reachable() -> None:
    result = service(npu=FakeBackend(ProviderChoice.NPU, share=1.0)).embed_documents(
        documents("alpha"), ProviderChoice.AUTO
    )

    assert result.provider_served is ProviderChoice.NPU
    assert result.fallback_reason is None


def test_the_npu_factory_is_closed_over_the_callers_selection() -> None:
    """Note 4.3: the selection decides fail-versus-warn on a weak partition."""
    builder = RecordingBuilder(npu=FakeBackend(ProviderChoice.NPU, share=1.0))

    service(builder=builder).embed_documents(documents("alpha"), ProviderChoice.AUTO)
    service(builder=builder).embed_documents(documents("alpha"), ProviderChoice.NPU)

    assert builder.requested == [ProviderChoice.AUTO, ProviderChoice.NPU]


def test_a_plain_string_provider_selects_the_same_backend_as_the_member() -> None:
    builder = RecordingBuilder(cpu=FakeBackend(ProviderChoice.CPU))

    result = service(builder=builder).embed_documents(
        documents("alpha"),
        "cpu",  # type: ignore[arg-type]
    )

    assert result.provider_served is ProviderChoice.CPU
    # Identity, not equality. `ProviderChoice` is a `StrEnum`, so the raw string
    # "cpu" compares equal to the member and an equality assertion here passes
    # whether or not the service converted anything - which is what review
    # found. Every branch downstream is written with `is`.
    assert builder.requested[0] is ProviderChoice.CPU
    assert type(builder.requested[0]) is ProviderChoice


# --------------------------------------------------------------------------
# Partition verification, derived service-side (Note 4.3)
# --------------------------------------------------------------------------


def test_partition_verification_does_not_apply_to_the_cpu() -> None:
    result = service(cpu=FakeBackend(ProviderChoice.CPU)).embed_documents(
        documents("alpha"), ProviderChoice.CPU
    )

    assert result.partition_verified is None


@pytest.mark.parametrize(
    ("share", "verified"),
    [
        (1.0, True),
        (MINIMUM_PARTITION_SHARE, True),
        (MINIMUM_PARTITION_SHARE - 0.01, False),
        (0.0, False),
        (None, False),
    ],
)
def test_the_partition_share_is_verified_against_the_threshold(
    share: float | None, verified: bool
) -> None:
    result = service(npu=FakeBackend(ProviderChoice.NPU, share=share)).embed_documents(
        documents("alpha"), ProviderChoice.AUTO
    )

    assert result.partition_verified is verified


# --------------------------------------------------------------------------
# Progress and summaries (8.3, 8.4, 8.6) - Note 4.1's carrier
# --------------------------------------------------------------------------


def test_progress_reports_inputs_completed_and_remaining() -> None:
    seen: list[ProgressUpdate] = []

    service().embed_documents(
        documents(*SAMPLE), ProviderChoice.CPU, progress=seen.append
    )

    assert [update.completed for update in seen] == [0, 3, 5]
    assert [update.remaining for update in seen] == [5, 2, 0]
    assert {update.total for update in seen} == {5}


def test_a_successful_run_reports_elapsed_time_and_the_input_count() -> None:
    summaries: list[RunSummary] = []

    result = service().embed_documents(
        documents(*SAMPLE), ProviderChoice.CPU, on_finish=summaries.append
    )

    assert result.input_count == len(SAMPLE)
    assert result.elapsed_seconds >= 0.0
    assert len(summaries) == 1
    assert summaries[0].completed_count == len(SAMPLE)
    assert summaries[0].interruption is None


def test_the_reported_elapsed_time_is_the_runs_actual_duration() -> None:
    """Requirement 8.6, pinned to a number rather than to a sign.

    ``elapsed_seconds >= 0.0`` was the whole assertion until review planted a
    hardcoded ``0.0`` and watched it pass all 1107 tests. `RunTracker` reads the
    clock three times - at construction, on entry, on exit - so a half-second
    step makes the run's duration exactly one step, and the service must publish
    the same number its own summary carries.
    """
    summaries: list[RunSummary] = []

    result = service(clock=FakeClock(step=0.5)).embed_documents(
        documents(*SAMPLE), ProviderChoice.CPU, on_finish=summaries.append
    )

    assert result.elapsed_seconds == pytest.approx(0.5)
    assert result.elapsed_seconds == summaries[0].elapsed_seconds


def test_a_slower_run_reports_a_longer_elapsed_time() -> None:
    """Non-vacuity guard: the number must track the clock, not be a constant."""
    quick = service(clock=FakeClock(step=0.5)).embed_documents(
        documents("alpha"), ProviderChoice.CPU
    )
    slow = service(clock=FakeClock(step=4.0)).embed_documents(
        documents("alpha"), ProviderChoice.CPU
    )

    assert slow.elapsed_seconds > quick.elapsed_seconds


def test_the_run_summary_counts_the_inputs_that_were_shortened() -> None:
    """The result names *which* inputs were shortened; the summary counts them.

    Planted mutant M16 - ``record_truncated(0)`` - survived the first round of
    this suite: every truncation assertion read `EmbedResult.truncated_indices`,
    which comes straight from the encoder, so the tracker could be told anything
    at all. Two of four inputs are over-long here, so neither zero nor the whole
    batch satisfies this.
    """
    summaries: list[RunSummary] = []
    texts = documents("alpha", _over_long(), "kappa", _over_long())

    result = service().embed_documents(
        texts, ProviderChoice.CPU, on_finish=summaries.append
    )

    assert result.truncated_indices == (1, 3)
    assert result.truncated_count == 2
    assert summaries[0].truncated_count == 2
    assert 0 < summaries[0].truncated_count < len(texts)


def test_an_interrupted_batch_reports_how_many_inputs_completed() -> None:
    summaries: list[RunSummary] = []
    backend = FakeBackend(ProviderChoice.CPU, fail_on_call=2)

    with pytest.raises(ExecutionError):
        service(cpu=backend).embed_documents(
            documents(*SAMPLE), ProviderChoice.CPU, on_finish=summaries.append
        )

    assert len(summaries) == 1
    assert summaries[0].completed_count == BATCH
    assert summaries[0].input_count == len(SAMPLE)
    assert summaries[0].interrupted


def test_the_provider_is_named_when_the_operation_fails() -> None:
    summaries: list[RunSummary] = []
    backend = FakeBackend(ProviderChoice.CPU, fail_on_call=1)

    with pytest.raises(ExecutionError) as raised:
        service(cpu=backend).embed_documents(
            documents(*SAMPLE), ProviderChoice.CPU, on_finish=summaries.append
        )

    assert raised.value.provider is ProviderChoice.CPU
    assert summaries[0].provider_served is ProviderChoice.CPU


def test_the_fallback_reason_reaches_the_run_summary() -> None:
    """Note 4.1: `RunSummary` carried this field with no producer until now."""
    summaries: list[RunSummary] = []

    service(cpu=FakeBackend(ProviderChoice.CPU), capability=incapable()).embed_documents(
        documents("alpha"), ProviderChoice.AUTO, on_finish=summaries.append
    )

    assert summaries[0].fallback_reason is not None
    assert summaries[0].fallback_reason == summaries[0].fallback_reason.strip()
    assert "install onnxruntime-vitisai" in summaries[0].fallback_reason


def test_an_npu_run_records_no_fallback_reason() -> None:
    summaries: list[RunSummary] = []

    service(npu=FakeBackend(ProviderChoice.NPU, share=1.0)).embed_documents(
        documents("alpha"), ProviderChoice.AUTO, on_finish=summaries.append
    )

    assert summaries[0].fallback_reason is None


def test_each_call_builds_its_own_tracker() -> None:
    """Note 2.3: `RunTracker` is single-use, so a shared one fails the second."""
    subject = service()

    first = subject.embed_documents(documents("alpha"), ProviderChoice.CPU)
    second = subject.embed_documents(documents("kappa"), ProviderChoice.CPU)

    assert first.input_count == second.input_count == 1


# --------------------------------------------------------------------------
# Determinism and preconditions (3.10)
# --------------------------------------------------------------------------


def test_the_same_text_under_the_same_provider_embeds_identically() -> None:
    subject = service()

    first = subject.embed_documents(documents(*SAMPLE), ProviderChoice.CPU)
    second = subject.embed_documents(documents(*SAMPLE), ProviderChoice.CPU)

    np.testing.assert_array_equal(first.vectors, second.vectors)


def test_an_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        service().embed_documents([], ProviderChoice.CPU)


def test_an_empty_query_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        service().embed_queries([], ProviderChoice.CPU)


# --------------------------------------------------------------------------
# The Dense stage
# --------------------------------------------------------------------------


def _cycle() -> DenseLayer:
    """A 3-cycle permutation with a bias only two of whose slots are nonzero.

    Deliberately asymmetric on both axes. Task 5.2's review found a Dense
    fixture whose weight equalled its own transpose and whose bias aligned with
    the identity scale, which made a bias-before-projection mutant numerically
    invisible. This one distinguishes both.
    """
    return DenseLayer(
        name="0_Dense",
        weight=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        bias=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        activation="Identity",
    )


def test_the_dense_stage_changes_the_vectors_it_is_given() -> None:
    plain = service().embed_documents(documents(*SAMPLE), ProviderChoice.CPU)
    projected = service(dense=(_cycle(),)).embed_documents(
        documents(*SAMPLE), ProviderChoice.CPU
    )

    assert not np.allclose(plain.vectors, projected.vectors)
    np.testing.assert_allclose(
        np.linalg.norm(projected.vectors, axis=1),
        np.ones(len(SAMPLE)),
        rtol=1e-6,
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# The production wiring: `default_backend_builder`, `load_dense_layers`,
# `build_service`
#
# Review found all three shipped with no coverage at all: five mutants inside
# them passed the entire 1107-test suite, including one that made the CPU
# factory open an NPU artifact and one that hardcoded the NPU adapter's
# selection to `auto` - which is exactly the requirement 2.2 substitution that
# Note 4.3 exists to prevent. None of that needs hardware: `ArtifactPreparer` is
# an injectable seam, and the adapters are module attributes.
# --------------------------------------------------------------------------


class RecordingPreparer:
    """An `ArtifactPreparer` that records what it was asked to prepare."""

    def __init__(self, artifact: PreparedArtifact) -> None:
        self.artifact = artifact
        self.calls: list[tuple[str, ProviderChoice, Path]] = []

    def __call__(
        self, profile: ModelProfile, provider: ProviderChoice, root: Path, /
    ) -> PreparedArtifact:
        self.calls.append((profile.model_id, provider, root))
        return self.artifact

    @property
    def providers(self) -> list[ProviderChoice]:
        return [provider for _, provider, _ in self.calls]


class BackendRecorder:
    """Stands in for a real adapter constructor and records its arguments.

    `CpuBackend` is constructed with two positional arguments and
    `VitisAIBackend` with three, so ``requested`` stays ``None`` for the CPU and
    carries the caller's selection for the NPU - which is the whole question
    Note 4.3 asks.
    """

    def __init__(self, provider: ProviderChoice) -> None:
        self._provider = provider
        self.requested: list[ProviderChoice | None] = []
        self.artifacts: list[PreparedArtifact] = []
        self.arity: list[int] = []

    def __call__(
        self,
        artifact: PreparedArtifact,
        profile: ModelProfile,
        *rest: ProviderChoice,
    ) -> FakeBackend:
        # `*rest` rather than `requested=None, **_`. The lenient version
        # absorbed an arity mistake: `CpuBackend(artifact, profile, requested)`
        # is wrong against the real two-positional-argument constructor, and
        # only mypy caught it. Recording the actual arity means the CPU shape
        # is asserted rather than tolerated.
        self.artifacts.append(artifact)
        self.arity.append(len(rest))
        self.requested.append(rest[0] if rest else None)
        return FakeBackend(self._provider, share=1.0)


def prepared(tmp_path: Path, *, dense: bool = False) -> PreparedArtifact:
    """A published artifact directory, with the files a manifest may name."""
    directory = tmp_path / "artifact"
    directory.mkdir(exist_ok=True)
    onnx_path = directory / ONNX_FILENAME
    onnx_path.write_bytes(b"not a real graph; no session is ever built from it")
    dense_path: Path | None = None
    if dense:
        dense_path = directory / DENSE_FILENAME
        write_dense(dense_path)
    return PreparedArtifact(
        directory=directory,
        onnx_path=onnx_path,
        context_path=None,
        dense_path=dense_path,
        manifest=ArtifactManifest(
            identity=ArtifactIdentity(
                model_id=PROFILE.model_id,
                revision=SHA,
                provider=ProviderChoice.CPU.value,
                compiled_seq_len=PROFILE.compiled_seq_len,
                batch_size=PROFILE.batch_size,
                onnxruntime_version="1.23.2",
                ryzen_ai_version=None,
                driver_version=None,
            ),
            observed_partition_share=None,
            files=(ONNX_FILENAME,),
        ),
        reused=False,
        reason="built for this test",
        elapsed_seconds=0.0,
    )


#: Stage names chosen so alphabetical order is the *wrong* order. Note 3.2
#: records that task 3.2's fixture named its stages alphabetically, which made a
#: `sorted()` substitute indistinguishable from reading the recorded order. The
#: widths chain 4 -> 6 -> 4, so applying them backwards is also a shape error.
_STAGE_ORDER = ("z_first", "a_second")


def write_dense(path: Path) -> None:
    """A two-stage ``dense.npz`` under `models/export.py`'s key names."""
    rng = np.random.default_rng(20260906)
    np.savez(
        path,
        **{
            DENSE_ORDER_KEY: np.array(list(_STAGE_ORDER)),
            f"{WEIGHT_PREFIX}z_first": rng.normal(size=(6, 4)).astype(np.float32),
            f"{BIAS_PREFIX}z_first": rng.normal(size=(6,)).astype(np.float32),
            f"{ACTIVATION_PREFIX}z_first": np.array("Identity"),
            f"{WEIGHT_PREFIX}a_second": rng.normal(size=(4, 6)).astype(np.float32),
            f"{BIAS_PREFIX}a_second": rng.normal(size=(4,)).astype(np.float32),
            f"{ACTIVATION_PREFIX}a_second": np.array("Identity"),
        },
    )


DENSE_PROFILE = ModelProfile(
    model_id="test/with-a-dense-stage",
    dimension=4,
    compiled_seq_len=SEQ,
    architectural_context_limit=512,
    batch_size=BATCH,
    pooling="mean",
    has_dense_stage=True,
    document_template="passage title: {title} body: {content}",
    query_template="search query: {content}",
    license_gated=False,
)


def test_the_cpu_factory_prepares_a_cpu_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An NPU artifact carries a compiled snapshot a CPU session must not open."""
    recorder = BackendRecorder(ProviderChoice.CPU)
    monkeypatch.setattr(service_module, "CpuBackend", recorder)
    preparer = RecordingPreparer(prepared(tmp_path))

    factories = default_backend_builder(tmp_path, prepare=preparer)(
        ProviderChoice.CPU
    )
    backend = factories.cpu(PROFILE, capable())

    assert preparer.providers == [ProviderChoice.CPU]
    assert preparer.calls[0][2] == tmp_path
    assert recorder.artifacts == [preparer.artifact]
    # The CPU adapter takes two positional arguments and no selection: it has
    # no fail-versus-warn reading to make, so handing it one would be wrong.
    assert recorder.arity == [0]
    assert backend.provider is ProviderChoice.CPU


def test_the_npu_factory_prepares_an_npu_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = BackendRecorder(ProviderChoice.NPU)
    monkeypatch.setattr(service_module, "VitisAIBackend", recorder)
    preparer = RecordingPreparer(prepared(tmp_path))

    factories = default_backend_builder(tmp_path, prepare=preparer)(
        ProviderChoice.NPU
    )
    factories.npu(PROFILE, capable())

    assert preparer.providers == [ProviderChoice.NPU]


@pytest.mark.parametrize(
    "selection", [ProviderChoice.NPU, ProviderChoice.AUTO]
)
def test_the_callers_selection_reaches_the_npu_adapter(
    selection: ProviderChoice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Note 4.3: this is what decides fail-versus-warn on a weak partition.

    Under ``npu`` `VitisAIBackend` refuses a below-threshold or unverifiable
    partition; under ``auto`` it records the weakness and proceeds. A builder
    that passed a constant would invert exactly one of those two verdicts, and
    the ``auto`` constant is the dangerous one - an explicit ``npu`` request
    would then be served by a graph the NPU barely touched.
    """
    recorder = BackendRecorder(ProviderChoice.NPU)
    monkeypatch.setattr(service_module, "VitisAIBackend", recorder)

    factories = default_backend_builder(
        tmp_path, prepare=RecordingPreparer(prepared(tmp_path))
    )(selection)
    factories.npu(PROFILE, capable())

    assert recorder.requested == [selection]


def test_the_builder_defaults_to_the_fail_closed_npu_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Note 4.3: "If it ever defaults, default to NPU (fail-closed)"."""
    recorder = BackendRecorder(ProviderChoice.NPU)
    monkeypatch.setattr(service_module, "VitisAIBackend", recorder)

    factories = default_backend_builder(
        tmp_path, prepare=RecordingPreparer(prepared(tmp_path))
    )()
    factories.npu(PROFILE, capable())

    assert recorder.requested == [ProviderChoice.NPU]


def test_a_profile_with_no_dense_stage_reads_no_weights(tmp_path: Path) -> None:
    assert load_dense_layers(prepared(tmp_path), PROFILE) == ()


def test_the_dense_layers_are_read_in_the_recorded_pipeline_order(
    tmp_path: Path,
) -> None:
    layers = load_dense_layers(prepared(tmp_path, dense=True), DENSE_PROFILE)

    assert [layer.name for layer in layers] == list(_STAGE_ORDER)
    assert [layer.weight.shape for layer in layers] == [(6, 4), (4, 6)]


def test_the_recorded_order_is_not_the_alphabetical_one(tmp_path: Path) -> None:
    """Non-vacuity guard for the test above.

    Note 3.2: task 3.2's fixture named its stages alphabetically, so reading the
    recorded order and sorting the weight names produced the same answer and the
    substitution was invisible. This asserts the fixture can still tell them
    apart.
    """
    assert list(_STAGE_ORDER) != sorted(_STAGE_ORDER)


def test_a_declared_dense_stage_with_no_weights_is_a_failure(
    tmp_path: Path,
) -> None:
    """design.md's own named risk: right shape, right norm, wrong meaning."""
    with pytest.raises(ExecutionError) as raised:
        load_dense_layers(prepared(tmp_path), DENSE_PROFILE, ProviderChoice.CPU)

    assert raised.value.stage == EMBED_STAGE
    assert raised.value.model_id == DENSE_PROFILE.model_id
    assert raised.value.provider is ProviderChoice.CPU


def test_the_dense_keys_are_the_exporters_own_constants(tmp_path: Path) -> None:
    """The four key names belong to `models/export.py` and are passed in.

    ``postprocess`` sits below ``models`` and requires them as arguments so the
    two cannot drift. Writing the file under those exact constants and reading
    it back is what proves this layer supplies them rather than a local copy: a
    renamed prefix makes the read fail.
    """
    artifact = prepared(tmp_path, dense=True)
    with np.load(artifact.dense_path) as arrays:  # type: ignore[arg-type]
        keys = set(arrays.files)

    assert DENSE_ORDER_KEY in keys
    assert f"{WEIGHT_PREFIX}z_first" in keys
    assert f"{BIAS_PREFIX}z_first" in keys
    assert f"{ACTIVATION_PREFIX}z_first" in keys
    assert load_dense_layers(artifact, DENSE_PROFILE)[0].name == "z_first"


def test_a_dense_file_written_under_other_key_names_is_refused(
    tmp_path: Path,
) -> None:
    """Non-vacuity guard: the test above must depend on the constants."""
    artifact = prepared(tmp_path, dense=True)
    assert artifact.dense_path is not None
    np.savez(artifact.dense_path, **{"a_different_order_key": np.array(["x"])})

    with pytest.raises(ExecutionError, match=DENSE_ORDER_KEY):
        load_dense_layers(artifact, DENSE_PROFILE)


def test_build_service_prepares_for_the_callers_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        service_module, "load_tokenizer", lambda profile, **_: model_tokenizer()
    )
    preparer = RecordingPreparer(prepared(tmp_path))

    subject = build_service(
        PROFILE, capable(), tmp_path, provider=ProviderChoice.CPU, prepare=preparer
    )

    assert preparer.providers == [ProviderChoice.CPU]
    assert subject.contract().model_id == PROFILE.model_id
    assert subject.contract().max_input_tokens == SEQ


def test_build_service_defaults_to_the_npu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        service_module, "load_tokenizer", lambda profile, **_: model_tokenizer()
    )
    preparer = RecordingPreparer(prepared(tmp_path))

    build_service(PROFILE, capable(), tmp_path, prepare=preparer)

    assert preparer.providers == [ProviderChoice.NPU]


def test_build_service_gives_the_service_its_dense_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The join between `load_dense_layers` and `EmbeddingService`.

    Review found both ends covered and the wire between them covered by
    nothing: every `build_service` test used a profile with no Dense stage, so
    ``dense=load_dense_layers(...)`` could be replaced by ``dense=()`` and the
    whole suite stayed green. That is design.md's own named risk for this
    boundary - right shape, right norm, wrong meaning, invisible to every
    assertion except a retrieval-quality measurement.
    """
    monkeypatch.setattr(
        service_module,
        "load_tokenizer",
        lambda profile, **_: model_tokenizer(DENSE_PROFILE),
    )
    monkeypatch.setattr(service_module, "CpuBackend", BackendRecorder(ProviderChoice.CPU))
    preparer = RecordingPreparer(prepared(tmp_path, dense=True))

    built = build_service(
        DENSE_PROFILE,
        capable(),
        tmp_path,
        provider=ProviderChoice.CPU,
        prepare=preparer,
    )
    with_dense = built.embed_documents(documents(*SAMPLE), ProviderChoice.CPU)
    # The comparison service declares *no* Dense stage rather than declaring one
    # and omitting it: since task 5.4 the latter is unconstructible, which is
    # the point of that guard. Everything else about the two profiles - id
    # aside - is identical, so the only difference in the vectors is the
    # projection.
    without = service(
        profile=dataclasses.replace(DENSE_PROFILE, has_dense_stage=False),
        cpu=FakeBackend(ProviderChoice.CPU),
    ).embed_documents(documents(*SAMPLE), ProviderChoice.CPU)

    assert not np.allclose(with_dense.vectors, without.vectors), (
        "the service build_service returned applies no Dense projection"
    )


def test_build_service_propagates_its_preparer_into_the_backend_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping ``prepare=prepare`` silently reverts to the real `ensure_prepared`.

    The earlier tests asserted only on the direct `prepare(...)` call that reads
    the Dense weights, never on the builder the service embeds through, so the
    seam was half-wired and nothing said so.
    """
    monkeypatch.setattr(
        service_module, "load_tokenizer", lambda profile, **_: model_tokenizer()
    )
    monkeypatch.setattr(service_module, "CpuBackend", BackendRecorder(ProviderChoice.CPU))
    preparer = RecordingPreparer(prepared(tmp_path))

    built = build_service(
        PROFILE, capable(), tmp_path, provider=ProviderChoice.CPU, prepare=preparer
    )
    built.embed_documents(documents("alpha"), ProviderChoice.CPU)

    assert preparer.providers == [ProviderChoice.CPU, ProviderChoice.CPU], (
        "one preparation for the dense weights, one for the backend"
    )
    assert {root for _, _, root in preparer.calls} == {tmp_path}


# --------------------------------------------------------------------------
# Task 5.4, defect 1: the isolated verdict meets the REAL factory pair
#
# Task 5.3 proved the resolution policy against a stub backend that reported
# ISOLATED - an adapter the shipped system does not contain. Both halves were
# individually correct and individually tested; the join was untested and
# wrong. These tests therefore go through `default_backend_builder`, the only
# NPU factory that exists, with `ensure_prepared` replaced at its own injectable
# seam so the assertion "nothing was prepared" is observable rather than argued.
# --------------------------------------------------------------------------


def isolated_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    capability: CapabilityReport | None = None,
) -> tuple[EmbeddingService, RecordingPreparer, BackendRecorder]:
    """A service wired to the production builder in an isolated-verdict world."""
    npu_recorder = BackendRecorder(ProviderChoice.NPU)
    monkeypatch.setattr(service_module, "VitisAIBackend", npu_recorder)
    monkeypatch.setattr(
        service_module, "CpuBackend", BackendRecorder(ProviderChoice.CPU)
    )
    preparer = RecordingPreparer(prepared(tmp_path))
    subject = EmbeddingService(
        profile=PROFILE,
        capability=isolated() if capability is None else capability,
        tokenizer=model_tokenizer(),
        backends=default_backend_builder(tmp_path, prepare=preparer),
    )
    return subject, preparer, npu_recorder


def test_auto_under_an_isolated_verdict_serves_the_cpu_without_preparing_the_npu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirements 2.4, 2.5 and 1.4 through the wiring that actually ships.

    Preparing for the NPU here is a 160-310 s compile followed by a session that
    cannot register the provider. The old policy called it "available", so an
    ``auto`` run paid that cost and then raised instead of reporting the reason
    and serving on the CPU.
    """
    subject, preparer, npu_recorder = isolated_service(tmp_path, monkeypatch)

    result = subject.embed_documents(documents(*SAMPLE), ProviderChoice.AUTO)

    assert result.provider_served is ProviderChoice.CPU
    assert result.fallback_reason is not None
    assert ExecutionMode.ISOLATED.value in result.fallback_reason
    assert len(result.vectors) == len(SAMPLE)
    assert preparer.providers == [ProviderChoice.CPU], (
        "an NPU artifact was prepared for a route no adapter can take"
    )
    assert npu_recorder.artifacts == []


def test_explicit_npu_under_an_isolated_verdict_prepares_nothing_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirements 2.2 and 1.4: refused at resolution, not at session creation.

    Failing later is not equivalent. The diagnostic would name the session as
    the failing stage and misattribute the cause, and the compile would already
    have been paid for.
    """
    subject, preparer, npu_recorder = isolated_service(tmp_path, monkeypatch)

    with pytest.raises(NpuUnavailableError) as raised:
        subject.embed_documents(documents(*SAMPLE), ProviderChoice.NPU)

    assert preparer.calls == []
    assert npu_recorder.artifacts == []
    assert ExecutionMode.ISOLATED.value in str(raised.value)
    assert CONDITION_PROVIDER_REGISTERED in str(raised.value)


def test_the_same_wiring_does_prepare_the_npu_when_it_is_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for both tests above.

    Their load-bearing assertion is that the preparer was never called for the
    NPU. A builder that prepared nothing under any verdict - a broken seam, a
    monkeypatch that silently took no effect - would satisfy that for the wrong
    reason. Under an in-process verdict the identical wiring must prepare an NPU
    artifact and serve on the NPU.
    """
    subject, preparer, npu_recorder = isolated_service(
        tmp_path, monkeypatch, capability=capable()
    )

    result = subject.embed_documents(documents(*SAMPLE), ProviderChoice.AUTO)

    assert result.provider_served is ProviderChoice.NPU
    assert result.fallback_reason is None
    assert preparer.providers == [ProviderChoice.NPU]
    assert npu_recorder.requested == [ProviderChoice.AUTO]


# --------------------------------------------------------------------------
# Task 5.4, defect 2: a declared Dense stage cannot be left out
# --------------------------------------------------------------------------


def test_a_service_for_a_dense_profile_refuses_an_empty_dense_stage() -> None:
    """`load_dense_layers` guards the missing *file*; this guards the omitted
    *argument*, which no downstream check can see.

    The trunk's hidden width equals the published dimension - EmbeddingGemma
    chains 768 -> 3072 -> 768 - so skipping the projection yields vectors of the
    right width, the right dtype and unit norm that mean something else. design.md
    records this as the risk only requirement 6.4's retrieval-quality
    measurement can detect. `build_service` wires the layers correctly today,
    and task 6.3's harness is the second construction site.
    """
    with pytest.raises(ValueError, match="Dense"):
        EmbeddingService(
            profile=DENSE_PROFILE,
            capability=capable(),
            tokenizer=model_tokenizer(DENSE_PROFILE),
            backends=RecordingBuilder(cpu=FakeBackend(ProviderChoice.CPU)),
        )


def test_the_dense_refusal_names_the_model_and_the_way_out() -> None:
    """Requirement 8.1's spirit at construction: say which model, and what to do."""
    with pytest.raises(ValueError) as raised:
        EmbeddingService(
            profile=DENSE_PROFILE,
            capability=capable(),
            tokenizer=model_tokenizer(DENSE_PROFILE),
            backends=RecordingBuilder(cpu=FakeBackend(ProviderChoice.CPU)),
            dense=(),
        )

    message = str(raised.value)
    assert DENSE_PROFILE.model_id in message
    assert "load_dense_layers" in message


def test_a_dense_profile_constructs_once_it_is_given_its_layers(
    tmp_path: Path,
) -> None:
    layers = load_dense_layers(prepared(tmp_path, dense=True), DENSE_PROFILE)

    subject = EmbeddingService(
        profile=DENSE_PROFILE,
        capability=capable(),
        tokenizer=model_tokenizer(DENSE_PROFILE),
        backends=RecordingBuilder(cpu=FakeBackend(ProviderChoice.CPU)),
        dense=layers,
    )

    assert subject.contract().model_id == DENSE_PROFILE.model_id


def test_a_profile_with_no_dense_stage_still_constructs_with_no_layers() -> None:
    """Non-vacuity: the guard keys on the profile's declaration, not on
    emptiness. A check that simply rejected an empty tuple would break every
    model that has no Dense stage - two of the three shipping profiles."""
    assert PROFILE.has_dense_stage is False

    assert service().contract().model_id == PROFILE.model_id


# --------------------------------------------------------------------------
# EmbedResult's own invariants
# --------------------------------------------------------------------------


def _result(**overrides: Any) -> EmbedResult:
    fields: dict[str, Any] = {
        "vectors": np.zeros((2, PROFILE.dimension), dtype=np.float32),
        "provider_served": ProviderChoice.CPU,
        "execution_mode": ExecutionMode.IN_PROCESS,
        "partition_verified": None,
        "fallback_reason": None,
        "truncated_indices": (),
        "elapsed_seconds": 0.5,
        "input_count": 2,
    }
    fields.update(overrides)
    return EmbedResult(**fields)


def test_a_result_is_constructible_from_consistent_values() -> None:
    assert _result().input_count == 2


def test_a_result_whose_row_count_disagrees_with_its_input_count_is_refused() -> None:
    with pytest.raises(ValueError, match="one vector per input"):
        _result(input_count=3)


def test_a_result_never_names_auto_as_the_serving_provider() -> None:
    with pytest.raises(ValueError, match="request, not an outcome"):
        _result(provider_served=ProviderChoice.AUTO)


def test_a_cpu_result_may_not_claim_a_verified_partition() -> None:
    with pytest.raises(ValueError, match="does not apply to the cpu"):
        _result(partition_verified=True)


def test_an_npu_result_must_state_whether_its_partition_was_verified() -> None:
    with pytest.raises(ValueError, match="partition_verified"):
        _result(
            provider_served=ProviderChoice.NPU,
            partition_verified=None,
            vectors=np.zeros((2, PROFILE.dimension), dtype=np.float32),
        )


def test_a_fallback_reason_belongs_only_to_a_cpu_result() -> None:
    with pytest.raises(ValueError, match="cpu"):
        _result(
            provider_served=ProviderChoice.NPU,
            partition_verified=True,
            fallback_reason="the NPU was busy",
        )


def test_a_blank_fallback_reason_explains_nothing_and_is_refused() -> None:
    with pytest.raises(ValueError, match="fallback_reason"):
        _result(fallback_reason="   ")


def test_a_truncated_index_outside_the_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="truncated_indices"):
        _result(truncated_indices=(5,))


def test_truncated_indices_must_be_ordered_and_distinct() -> None:
    with pytest.raises(ValueError, match="truncated_indices"):
        _result(truncated_indices=(1, 1))
