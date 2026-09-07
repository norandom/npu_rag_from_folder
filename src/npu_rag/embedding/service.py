"""The embedding service: text in, unit vectors out, and a truthful account
of what produced them (task 5.3).

This is the assembly point. Everything it needs already exists and was built to
be composed here: `ModelTokenizer` renders and encodes, `resolve_backend` binds
one provider to one operation, `TransformerBackend.run` returns token embeddings
alone, `postprocess.finalize` pools, projects and normalizes them, and
`RunTracker` records what happened. This module adds no numerical behaviour of
its own - it decides *what to call in what order*, and it owns the two things
nobody below it can: the batching that satisfies the graph's static shape, and
the report requirement 2.6 makes about every operation.

**Why there is no ``embed(texts, kind)``.** Requirement 3.3 says a request that
does not declare its text kind is rejected rather than defaulted.
`embed_documents` and `embed_queries` make that a call-site impossibility rather
than a runtime rejection, and `DocumentText` carries the title slot that
requirement 3.4's document convention has and a bare string cannot.

**Why the service pads, and then discards what it padded.** NPU execution
requires static shapes, so the graph is compiled at exactly
``(batch_size, compiled_seq_len)`` and `TransformerBackend` documents that a
partial batch is padded *by the caller*. A run of five inputs at batch three is
two forward passes, the second of which is one real row and two filler rows. The
filler is cut off before post-processing ever sees it. That ordering is not a
tidiness preference: a padded row has an all-zero attention mask, and
`masked_mean_pool` refuses to divide by zero - so passing the padding through
would raise rather than quietly return a mean over nothing. The refusal is the
backstop; slicing first is the behaviour.

**Why the filler repeats the last real row.** Its mask is zero, so its content
cannot reach any result, and any value would do arithmetically. It repeats a row
that this tokenizer actually produced so the ids are certainly inside the
model's vocabulary. A hard-coded zero would rely on 0 being a valid embedding
index for every model this service will ever run, which is an assumption with no
reason to be made.

**Where ``partition_verified`` is decided.** Here, not in the adapter. The
adapter reports the raw share it measured, or ``None`` where it established
none; comparing that against `MINIMUM_PARTITION_SHARE` is a policy judgement,
and keeping it on this side means the CPU's "the question does not apply" and
the NPU's "measured and below threshold" are produced by one piece of code
rather than two that could disagree.

This module sits in the ``service`` layer of design.md's dependency direction -
``types, errors -> postprocess -> reporting -> profiles -> environment -> models
-> providers, tokenize -> service -> bench`` - so it may read everything to its
left and must never reach ``bench``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
from transformers import PreTrainedTokenizerBase

from npu_rag.embedding.errors import ExecutionError
from npu_rag.embedding.models.artifacts import PreparedArtifact, ensure_prepared
from npu_rag.embedding.models.export import (
    ACTIVATION_PREFIX,
    BIAS_PREFIX,
    DENSE_ORDER_KEY,
    WEIGHT_PREFIX,
)
from npu_rag.embedding.postprocess import (
    DenseLayer,
    dense_layers_from_arrays,
    finalize,
)
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.providers.base import (
    BackendFactories,
    TransformerBackend,
    resolve_backend,
)
from npu_rag.embedding.providers.cpu import CpuBackend
from npu_rag.embedding.providers.vitisai import (
    MINIMUM_PARTITION_SHARE,
    VitisAIBackend,
)
from npu_rag.embedding.reporting import (
    ProgressCallback,
    RunTracker,
    SummaryCallback,
)
from npu_rag.embedding.tokenize import EncodedBatch, ModelTokenizer, load_tokenizer
from npu_rag.embedding.types import (
    CapabilityReport,
    DocumentText,
    ExecutionMode,
    ProviderChoice,
    TextKind,
)

__all__ = [
    "EMBED_STAGE",
    "ArtifactPreparer",
    "BackendBuilder",
    "EmbedResult",
    "EmbeddingContract",
    "EmbeddingService",
    "build_service",
    "default_backend_builder",
    "load_dense_layers",
]

#: The stage a failure inside an embedding operation belongs to (8.1, 8.2).
#: Distinct from the preparation stages ``acquire``/``export``/``compile`` and
#: from ``session``, so requirement 8.2's three categories stay separable.
EMBED_STAGE: Final = "embed"

#: What `RunSummary.operation` is called for each entry point. Requirement 8.3's
#: progress and 8.6's completion report name the operation, and a consumer that
#: renders documents and queries differently needs to tell them apart.
_OPERATION: Final[dict[TextKind, str]] = {
    TextKind.DOCUMENT: "embed_documents",
    TextKind.QUERY: "embed_queries",
}


# --------------------------------------------------------------------------
# What the service publishes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingContract:
    """What a consumer needs to know before sending anything (3.6, 3.7).

    ``max_input_tokens`` is the **compiled** length. `ModelProfile` keeps the
    architectural limit in a separate field precisely so this one cannot
    accidentally carry it: a consumer told the larger number would chunk to it
    and have every chunk silently shortened, which requirement 3.9 exists to
    make visible rather than to absorb.
    """

    model_id: str
    dimension: int
    max_input_tokens: int
    tokenizer_id: str


@dataclass(frozen=True)
class EmbedResult:
    """The vectors, and an account of what produced them.

    Every field is a claim about a *completed* operation, which is why this type
    lives here rather than in ``types.py`` with the rest of the vocabulary: the
    component that produces the claims is the one that can enforce them. The
    invariants below are requirements 2.5, 2.6 and 3.1 made unconstructible to
    violate rather than promised in prose.
    """

    vectors: npt.NDArray[np.float32]
    provider_served: ProviderChoice
    execution_mode: ExecutionMode
    #: ``None`` when the CPU served, because the question does not apply to it.
    #: ``False`` on an NPU run means *assumed, not verified* - either no share
    #: was established or the measured share was below threshold - which is a
    #: different statement from ``True`` and must not collapse into it.
    partition_verified: bool | None
    fallback_reason: str | None
    truncated_indices: tuple[int, ...]
    elapsed_seconds: float
    input_count: int

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[0] != self.input_count:
            raise ValueError(
                f"requirement 3.1 is one vector per input: {self.input_count} "
                f"inputs produced an array shaped {self.vectors.shape}"
            )
        if self.provider_served is ProviderChoice.AUTO:
            raise ValueError(
                "provider_served names what actually ran; 'auto' is a "
                "request, not an outcome (requirement 2.6)"
            )
        self._check_partition()
        self._check_fallback()
        self._check_truncated()

    def _check_partition(self) -> None:
        served_on_cpu = self.provider_served is ProviderChoice.CPU
        if served_on_cpu and self.partition_verified is not None:
            raise ValueError(
                "graph partitioning does not apply to the cpu, so "
                f"partition_verified must be None, got "
                f"{self.partition_verified!r}"
            )
        if not served_on_cpu and self.partition_verified is None:
            raise ValueError(
                "an npu run must state whether its partition was verified; "
                "partition_verified=None would leave 'measured and below "
                "threshold' indistinguishable from 'never checked'"
            )

    def _check_fallback(self) -> None:
        if self.fallback_reason is None:
            return
        if not self.fallback_reason.strip():
            raise ValueError(
                "fallback_reason must say specifically why the NPU was not "
                f"used, got {self.fallback_reason!r}: a blank reason reports a "
                "fallback and explains nothing (requirement 2.5)"
            )
        if self.provider_served is not ProviderChoice.CPU:
            raise ValueError(
                "fallback_reason explains why the NPU was not used, so the cpu "
                f"must be what served; got provider_served="
                f"{self.provider_served.value!r} (requirement 2.5)"
            )

    def _check_truncated(self) -> None:
        previous = -1
        for index in self.truncated_indices:
            if not 0 <= index < self.input_count:
                raise ValueError(
                    f"truncated_indices names input {index}, which is outside "
                    f"a batch of {self.input_count} (requirement 3.9)"
                )
            if index <= previous:
                raise ValueError(
                    "truncated_indices must be strictly increasing so each "
                    f"shortened input is named once, got "
                    f"{self.truncated_indices!r}"
                )
            previous = index

    @property
    def truncated_count(self) -> int:
        """How many inputs were shortened to fit (3.9)."""
        return len(self.truncated_indices)


# --------------------------------------------------------------------------
# The seams
# --------------------------------------------------------------------------


class BackendBuilder(Protocol):
    """Produces the factory pair for one selection.

    `resolve_backend` takes `BackendFactories`, but the NPU adapter needs to
    know what the *caller* asked for: `VitisAIBackend` fails on a weak or
    unverifiable partition under an explicit ``npu`` request and merely records
    it under ``auto``. A single fixed pair could not express both, so the pair is
    built per operation with the selection closed over it.

    ``requested`` is **required here, with no default**, which is design.md's
    Preconditions for this boundary: "a provider choice is always explicit, with
    no default". A default on this protocol was tried and reverted - it says
    only that an implementation *may* be called with no argument, never which
    value it would then use, so a builder defaulting to ``auto`` satisfies it
    with no type error. It would have widened the contract to permit the
    argument-less call while enforcing nothing about it, which is the opposite
    of the precondition.
    """

    def __call__(self, requested: ProviderChoice, /) -> BackendFactories: ...


class DefaultingBackendBuilder(Protocol):
    """A `BackendBuilder` that also answers a call with no selection at all.

    The concrete shape `default_backend_builder` returns. Any callable with a
    default satisfies both this and `BackendBuilder`, since a default only makes
    a signature more permissive.

    This type permits the argument-less call; it does not pin what that call
    resolves to. Nothing in the type system can - that is a value, and mypy
    checks arity and types. The fail-closed ``NPU`` reading is a property of the
    implementation below and is held there by test, not by this annotation.
    """

    def __call__(
        self, requested: ProviderChoice = ..., /
    ) -> BackendFactories: ...


class ArtifactPreparer(Protocol):
    """`ensure_prepared`, narrowed to what the default builder uses."""

    def __call__(
        self, profile: ModelProfile, provider: ProviderChoice, root: Path, /
    ) -> PreparedArtifact: ...


def load_dense_layers(
    artifact: PreparedArtifact,
    profile: ModelProfile,
    provider: ProviderChoice | None = None,
) -> tuple[DenseLayer, ...]:
    """The Dense stages this profile's pipeline applies, in pipeline order.

    The four key names come from `models/export.py`, which wrote the file.
    ``postprocess`` requires them as arguments precisely so they cannot be
    duplicated into a module that sits below ``models`` and drift; supplying
    them is this layer's job, and this is the one place it happens.

    A profile that declares a Dense stage and an artifact that has no
    ``dense.npz`` is a failure, not an empty tuple: skipping the projection
    yields vectors of the right width and the right norm that mean something
    else, which only the retrieval-quality measurement could ever detect.
    """
    if not profile.has_dense_stage:
        return ()
    if artifact.dense_path is None:
        raise ExecutionError(
            f"{profile.model_id} has a Dense stage between pooling and "
            "normalization, but the prepared artifact carries no dense "
            "weights. Embedding without it would produce correctly shaped, "
            "correctly normalized vectors that are semantically wrong",
            provider=provider,
            model_id=profile.model_id,
            stage=EMBED_STAGE,
        )
    with np.load(artifact.dense_path) as arrays:
        return dense_layers_from_arrays(
            arrays,
            order_key=DENSE_ORDER_KEY,
            weight_prefix=WEIGHT_PREFIX,
            bias_prefix=BIAS_PREFIX,
            activation_prefix=ACTIVATION_PREFIX,
            provider=provider,
            model_id=profile.model_id,
        )


def default_backend_builder(
    root: Path,
    *,
    prepare: ArtifactPreparer = ensure_prepared,
) -> DefaultingBackendBuilder:
    """Wire preparation to the two real adapters.

    Each factory prepares for the provider *it* serves, which is the whole
    reason preparation is not hoisted out: an NPU artifact carries a compiled
    snapshot a CPU session must not open, and a CPU artifact has none for an NPU
    session to find.

    ``requested`` defaults to ``NPU`` rather than ``AUTO``. Both are accepted by
    `VitisAIBackend`, and they differ in exactly one way: under ``npu`` a weak
    partition is a failure, under ``auto`` it is a recorded weakness. A caller
    who reaches this function without stating a selection gets the strict
    reading, so the mistake surfaces instead of quietly downgrading to the
    lenient one.
    """

    def build(
        requested: ProviderChoice = ProviderChoice.NPU, /
    ) -> BackendFactories:
        def npu(
            profile: ModelProfile, capability: CapabilityReport, /
        ) -> TransformerBackend:
            artifact = prepare(profile, ProviderChoice.NPU, root)
            return VitisAIBackend(artifact, profile, requested)

        def cpu(
            profile: ModelProfile, capability: CapabilityReport, /
        ) -> TransformerBackend:
            artifact = prepare(profile, ProviderChoice.CPU, root)
            return CpuBackend(artifact, profile)

        return BackendFactories(npu=npu, cpu=cpu)

    return build


# --------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------


class EmbeddingService:
    """Turns declared-kind text into unit vectors and reports what happened.

    **A declared Dense stage cannot be left out** (task 5.4, defect 2).
    ``dense`` has a default because most profiles have no Dense stage, and a
    default is exactly how the one that does could be skipped: applying no
    projection is `apply_dense_stages(pooled, ())`, the identity, followed by
    normalization - correctly shaped, correctly normalized, semantically wrong,
    which design.md names as this boundary's own risk. `load_dense_layers`
    guards the missing *file*; nothing guarded the omitted *argument* until the
    constructor did. `build_service` has always wired it, and task 6.3's
    benchmark harness is the second construction site this exists for.
    """

    def __init__(
        self,
        profile: ModelProfile,
        capability: CapabilityReport,
        tokenizer: ModelTokenizer,
        *,
        backends: BackendBuilder,
        dense: Sequence[DenseLayer] = (),
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if profile.has_dense_stage and not dense:
            raise ValueError(
                f"{profile.model_id} applies a Dense stage between pooling and "
                "normalization and no layers were supplied. Embedding without "
                "it produces vectors of the right width, the right dtype and "
                "unit norm that mean something else - the trunk's hidden width "
                "equals the published dimension, so not even a dimension check "
                "can see it, and only a retrieval-quality measurement ever "
                "would. Pass dense=load_dense_layers(artifact, profile), or "
                "use build_service, which does it"
            )
        self._profile = profile
        self._capability = capability
        self._tokens = tokenizer
        self._backends = backends
        self._dense = tuple(dense)
        self._clock = clock

    # -- the published contract (3.6, 3.7) ---------------------------------

    def contract(self) -> EmbeddingContract:
        """The model's dimension, its maximum input length, and its tokenizer."""
        return EmbeddingContract(
            model_id=self._profile.model_id,
            dimension=self._profile.dimension,
            max_input_tokens=self._profile.max_input_tokens,
            tokenizer_id=self._tokens.tokenizer_id,
        )

    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer this runtime measures with (3.7).

        Exposed so a consumer can measure by the same rule rather than a similar
        one; requirement 3.8 is the promise that the two agree.
        """
        return self._tokens.tokenizer

    def count_tokens(
        self, text: str | DocumentText, kind: TextKind
    ) -> int:
        """The token length of ``text`` under ``kind``'s convention (3.8).

        Accepts `DocumentText` as well as a bare string, because a document's
        length includes the rendered title slot. Measuring the content alone
        would disagree with the truncation decision for exactly the documents
        that carry a title, which is the drift requirement 3.8 forbids.
        """
        return self._tokens.count_tokens(text, kind)

    # -- embedding (3.1-3.5, 3.9, 3.10) ------------------------------------

    def embed_documents(
        self,
        texts: Sequence[DocumentText],
        provider: ProviderChoice,
        progress: ProgressCallback | None = None,
        *,
        on_finish: SummaryCallback | None = None,
    ) -> EmbedResult:
        """Embed corpus content under the document-side convention (3.4)."""
        return self._embed(texts, TextKind.DOCUMENT, provider, progress, on_finish)

    def embed_queries(
        self,
        texts: Sequence[str],
        provider: ProviderChoice,
        progress: ProgressCallback | None = None,
        *,
        on_finish: SummaryCallback | None = None,
    ) -> EmbedResult:
        """Embed search queries under the query-side convention (3.4)."""
        return self._embed(texts, TextKind.QUERY, provider, progress, on_finish)

    def _embed(
        self,
        texts: Sequence[str | DocumentText],
        kind: TextKind,
        provider: ProviderChoice,
        progress: ProgressCallback | None,
        on_finish: SummaryCallback | None,
    ) -> EmbedResult:
        """Tokenize, bind one provider, run every batch, report.

        ``on_finish`` is how requirement 8.4 is reachable at all. A run that is
        interrupted raises, so its `EmbedResult` is never returned - but the
        summary naming how many inputs completed is delivered on the way out,
        because `RunTracker.__exit__` runs whether the block ended cleanly or
        not. design.md's interface sketch shows only ``progress``; the summary
        callback is the channel 8.4 and 8.6 need, and `reporting` already
        defines it.
        """
        if len(texts) == 0:
            raise ValueError(
                "an embedding request must carry at least one input: an empty "
                "batch has no provider to report and no vectors to return"
            )
        choice = ProviderChoice(provider)
        encoded = self._tokens.encode(list(texts), kind)
        # Bound once, before any work, and never rebound inside the loop below:
        # requirement 2.7 is that the provider cannot change part-way through an
        # operation, and `BoundBackend` is frozen so it structurally cannot.
        bound, reason = resolve_backend(
            choice, self._profile, self._capability, factories=self._backends(choice)
        )
        tracker = RunTracker(
            operation=_OPERATION[kind],
            provider_served=ProviderChoice(bound.provider),
            execution_mode=bound.execution_mode,
            input_count=len(texts),
            progress=progress,
            on_finish=on_finish,
            fallback_reason=reason,
            clock=self._clock,
        )
        with tracker:
            tracker.record_truncated(len(encoded.truncated_indices))
            vectors = self._forward(bound, encoded, tracker)
        summary = tracker.result
        if summary is None:
            # `RunTracker.__exit__` stores the summary before it returns, so
            # this is unreachable today. It is a raise rather than an `assert`
            # because `python -O` strips assertions, and the stripped build
            # would reach the attribute access below and report requirement
            # 8.6's missing elapsed time as an AttributeError on None.
            raise ExecutionError(
                "the run tracker produced no summary for a completed "
                "operation, so there is no elapsed time to report",
                provider=ProviderChoice(bound.provider),
                model_id=self._profile.model_id,
                stage=EMBED_STAGE,
            )
        # Every field this result shares with `RunSummary` is read off the
        # summary, never re-derived from `bound` or `texts`. Note 2.3 raised
        # this: design.md's flat `EmbedResult` duplicates four fields verbatim
        # with `RunSummary`, and asked 5.3 to give requirements 2.6 and 8.6 ONE
        # producer. Composing the two types would have done it at the cost of a
        # design amendment; reading them from the one object the tracker built
        # does it without changing the shape design.md publishes. Two sources
        # that agree today are two sources that can disagree later - review
        # found exactly that, with `execution_mode` taken from the capability
        # report rather than from the backend that actually served.
        return EmbedResult(
            vectors=vectors,
            provider_served=summary.provider_served,
            execution_mode=summary.execution_mode,
            partition_verified=self._partition_verified(bound),
            fallback_reason=summary.fallback_reason,
            truncated_indices=encoded.truncated_indices,
            elapsed_seconds=summary.elapsed_seconds,
            input_count=summary.input_count,
        )

    def _forward(
        self,
        bound: TransformerBackend,
        encoded: EncodedBatch,
        tracker: RunTracker,
    ) -> npt.NDArray[np.float32]:
        """Run every batch at the compiled shape and post-process the real rows."""
        size = self._profile.batch_size
        total = encoded.token_ids.shape[0]
        provider = ProviderChoice(bound.provider)
        chunks: list[npt.NDArray[np.float32]] = []
        for start in range(0, total, size):
            stop = min(start + size, total)
            actual = stop - start
            token_ids = encoded.token_ids[start:stop]
            attention_mask = encoded.attention_mask[start:stop]
            if actual < size:
                token_ids, attention_mask = _pad_to(token_ids, attention_mask, size)
            embeddings = bound.run(token_ids, attention_mask)
            # Slice before post-processing, never after: the filler rows carry
            # an all-zero mask, which masked mean pooling refuses rather than
            # averaging over nothing.
            chunks.append(
                finalize(
                    embeddings[:actual],
                    attention_mask[:actual],
                    # The active model's own rule, never a default: two
                    # candidates pool by masked mean and one by CLS, and the
                    # wrong choice is the right width and the right norm
                    # carrying a different meaning (task 5.5).
                    pooling=self._profile.pooling,
                    dense=self._dense,
                    provider=provider,
                    model_id=self._profile.model_id,
                )
            )
            tracker.advance(actual)
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    def _partition_verified(self, bound: TransformerBackend) -> bool | None:
        """Whether the NPU demonstrably took the graph (design.md, 2026-09-06).

        ``None`` for the CPU: the question does not apply. Otherwise a share is
        verified only if one was established *and* it reaches the threshold, so
        an unestablished share reads as ``False`` - assumed, not verified -
        rather than as a passing check.
        """
        if ProviderChoice(bound.provider) is ProviderChoice.CPU:
            return None
        share = bound.npu_partition_share
        return share is not None and share >= MINIMUM_PARTITION_SHARE


def _pad_to(
    token_ids: npt.NDArray[np.int64],
    attention_mask: npt.NDArray[np.int64],
    size: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Fill a short batch out to the compiled batch size.

    The filler repeats the last real row with its attention zeroed. The ids are
    then certainly inside this model's vocabulary, and the zero mask means the
    row's output is discarded before post-processing and cannot reach a result.
    """
    missing = size - token_ids.shape[0]
    filler = np.repeat(token_ids[-1:], missing, axis=0)
    return (
        np.concatenate([token_ids, filler]),
        np.concatenate(
            [attention_mask, np.zeros_like(filler, dtype=np.int64)]
        ),
    )


def build_service(
    profile: ModelProfile,
    capability: CapabilityReport,
    root: Path,
    *,
    provider: ProviderChoice = ProviderChoice.NPU,
    prepare: ArtifactPreparer = ensure_prepared,
    progress: ProgressCallback | None = None,
) -> EmbeddingService:
    """Assemble a service against real artifacts under ``root``.

    ``provider`` decides which artifact the Dense weights are read from, and it
    should be the same selection the caller will embed with. The weights
    themselves are identical either way - ``dense.npz`` is written once at
    export and copied, not recompiled - but reading them from the artifact that
    will actually serve keeps a missing file surfacing as a preparation failure
    for the provider in use, rather than for a provider nobody asked for.
    """
    artifact = prepare(profile, ProviderChoice(provider), root)
    return EmbeddingService(
        profile=profile,
        capability=capability,
        tokenizer=load_tokenizer(profile, progress=progress),
        backends=default_backend_builder(root, prepare=prepare),
        dense=load_dense_layers(artifact, profile, ProviderChoice(provider)),
    )
