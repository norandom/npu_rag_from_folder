"""The tokenizer, and the length-measurement contract around it (task 5.1).

Requirement 3.7 exposes the active model's tokenizer so a consumer can measure
input length by the same rule the runtime applies. Requirement 3.8 says what
that is worth: an input the consumer measures as within the reported maximum is
treated as within limits **if and only if** the runtime agrees. Both directions.
A counter that were merely conservative would satisfy the easy half and still
let `document-ingest` size chunks that this runtime silently shortens - content
lost, with no error raised anywhere and no shape or norm changed to notice it
by. design.md's Revalidation Triggers list the tokenizer identity precisely
because downstream specs build against this number.

**Three things make disagreement structurally impossible here.**

1. *One measurement.* `ModelTokenizer.count_tokens` and
   `ModelTokenizer.encode` both call `_measure`, and neither has any other way
   to learn a length. Changing the rule changes both.
2. *One boundary.* Whether a count is over the limit is decided by
   `_over_limit` and by nothing else - not by an inline ``>`` at a call site,
   where a later ``>=`` would move the boundary for truncation without moving
   it for counting.
3. *One invariant.* `EncodedBatch` recomputes the truncation set from the
   counts it carries and refuses to be constructed if the two disagree. A
   caller cannot hand the service a batch whose truncation report is a fiction.

**What is measured is the rendered text.** The prefix template spends budget the
caller cannot see - EmbeddingGemma's document form is ``title: {title} | text:
{content}`` and its retrieval query form ``task: search result | query:
{content}`` - and so do the tokenizer's own special tokens. Counting the raw
string would under-report by exactly that much, which is the 3.8 failure this
module exists to prevent.

**The published maximum is `ModelProfile.max_input_tokens`**: the length the
graph was *compiled* at, never the architectural context limit. For
EmbeddingGemma those differ by a factor of four and for nomic by sixteen
(requirement 3.6, and profiles.py's own note).

**Truncation goes through the tokenizer.** Slicing the encoded ids to length
would drop the trailing ``[SEP]`` and end the sequence mid-sentence: the same
shape, the same norm, different text. The tokenizer's own ``truncation`` removes
content tokens and keeps the terminators, which is what the model was trained to
see. Padding, by contrast, is done here rather than by the tokenizer, so the
pinned ``(batch, compiled_seq_len)`` shape the backends require does not depend
on a tokenizer's padding configuration being set correctly.

This module sits between ``models`` and ``providers`` in design.md's dependency
direction - ``types, errors -> reporting -> profiles -> environment -> models ->
providers -> service -> bench``. It reuses `acquire_model` for the download
rather than re-implementing acquisition, so the gated-repository path
(requirement 4.5) and the credential handling are the ones already reviewed in
``models/acquire.py``; a tokenizer for EmbeddingGemma sits behind the same gate
as its weights. It imports nothing from ``providers``, ``service`` or ``bench``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from npu_rag.embedding.errors import ExecutionError, PreparationError
from npu_rag.embedding.models.acquire import (
    DEFAULT_REVISION,
    AcquiredModel,
    acquire_model,
)
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.reporting import ProgressCallback
from npu_rag.embedding.types import DocumentText, TextKind

__all__ = [
    "TOKENIZATION_STAGE",
    "TOKENIZER_FILE_PATTERNS",
    "TOKENIZER_STAGE",
    "EncodedBatch",
    "ModelAcquirer",
    "ModelTokenizer",
    "TokenizerLoader",
    "load_tokenizer",
]

#: The stage a failure to obtain or load a tokenizer reports (8.1, 4.6). It is a
#: preparation stage: the model has not been made ready.
TOKENIZER_STAGE: Final = "tokenizer"

#: The stage a failure *while measuring or encoding* reports. Distinct from
#: `TOKENIZER_STAGE` because the tokenizer loaded fine and the failure belongs
#: to the embedding operation in flight, which is what 8.2 separates.
TOKENIZATION_STAGE: Final = "tokenize"

#: Everything a tokenizer needs and nothing a tokenizer does not. Acquisition is
#: restricted to these because the candidates' weights are 0.5-1.3 GB each and
#: measuring an input's length must not cost a model download. ``config.json``
#: is included because ``AutoTokenizer`` falls back to the model type declared
#: there when ``tokenizer_config.json`` names no tokenizer class.
#:
#: The list is a superset covering all three candidates' formats - WordPiece
#: vocabularies, a SentencePiece model, and the fast ``tokenizer.json`` - so a
#: pattern that matches nothing in a given repository simply matches nothing.
TOKENIZER_FILE_PATTERNS: Final[tuple[str, ...]] = (
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
)

#: What pads a short sequence when the tokenizer declares no pad token. The
#: padded positions are masked out, so the value is never attended to; a
#: concrete number is still needed because the array has a fixed width.
_FALLBACK_PAD_ID: Final = 0


# --------------------------------------------------------------------------
# The boundary, in one place
# --------------------------------------------------------------------------


def _over_limit(counts: Iterable[int], max_input_tokens: int) -> tuple[int, ...]:
    """The positions whose measured length exceeds the published maximum.

    The single expression of requirement 3.8's boundary. A count *equal* to the
    maximum fits - the graph has exactly that many slots - so the comparison is
    strictly greater. Everything that needs to know "is this too long", from
    `ModelTokenizer.exceeds_limit` to `EncodedBatch`'s own invariant, asks here
    rather than writing the comparison again.
    """
    return tuple(
        index for index, count in enumerate(counts) if count > max_input_tokens
    )


# --------------------------------------------------------------------------
# What encoding produces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedBatch:
    """Model inputs at the pinned shape, and what measuring them established.

    ``token_ids`` and ``attention_mask`` are ``(batch, compiled_seq_len)``
    ``int64``, which is `TransformerBackend.run`'s precondition: NPU execution
    needs static shapes, so the caller pads rather than the session reshaping.

    ``token_counts`` are the *measured* lengths - what the rendered text
    actually tokenizes to, before any truncation. That is deliberately not
    recoverable from the arrays: a truncated row's mask sums to the maximum and
    says nothing about how much was dropped, and requirement 3.9 reports that
    an input was shortened.

    Invariant, and the reason this is a class rather than a tuple:
    ``truncated_indices`` is recomputed from ``token_counts`` and rejected if it
    disagrees. Requirement 3.8's *if and only if* is therefore not a property of
    the code that happened to build this object - it is a property of the object
    existing at all.
    """

    token_ids: npt.NDArray[np.int64]
    attention_mask: npt.NDArray[np.int64]
    #: The untruncated length of each input, in input order.
    token_counts: tuple[int, ...]
    #: The published maximum these counts were judged against (3.6).
    max_input_tokens: int
    #: Which inputs were shortened, in input order (3.9's carrier).
    truncated_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0:
            raise ValueError(
                f"max_input_tokens must be positive, got {self.max_input_tokens}"
            )
        for name in ("token_ids", "attention_mask"):
            array: npt.NDArray[np.int64] = getattr(self, name)
            if array.ndim != 2 or array.shape[1] != self.max_input_tokens:
                raise ValueError(
                    f"{name} is shaped {array.shape}; the backends require "
                    f"(batch, {self.max_input_tokens}) because the graph is "
                    "compiled at a static length"
                )
            if array.dtype != np.int64:
                raise ValueError(
                    f"{name} has dtype {array.dtype}; the exported graph "
                    "declares int64 inputs"
                )
        if self.token_ids.shape != self.attention_mask.shape:
            raise ValueError(
                f"token_ids {self.token_ids.shape} and attention_mask "
                f"{self.attention_mask.shape} describe different batches"
            )
        rows = int(self.token_ids.shape[0])
        if len(self.token_counts) != rows:
            raise ValueError(
                f"{len(self.token_counts)} token counts for {rows} rows: every "
                "input is measured, so the two cannot differ"
            )
        if any(count < 0 for count in self.token_counts):
            raise ValueError(f"negative token count in {self.token_counts}")

        expected = _over_limit(self.token_counts, self.max_input_tokens)
        if self.truncated_indices != expected:
            raise ValueError(
                f"truncated_indices {self.truncated_indices} disagrees with the "
                f"measurement {expected} at a maximum of {self.max_input_tokens}"
                " tokens; requirement 3.8 makes the consumer's measurement and "
                "this runtime's truncation decision the same decision, so a "
                "batch that reported otherwise would be the exact silent "
                "shortening that requirement exists to prevent"
            )

        for row, count in enumerate(self.token_counts):
            kept = min(count, self.max_input_tokens)
            mask = self.attention_mask[row]
            if int(mask.sum()) != kept:
                raise ValueError(
                    f"row {row} masks {int(mask.sum())} tokens but measured "
                    f"{count} against a maximum of {self.max_input_tokens}"
                )
            if not bool(mask[:kept].all()) or bool(mask[kept:].any()):
                raise ValueError(
                    f"row {row}'s attention mask is not a run of real tokens "
                    "followed by padding; masked mean pooling reads this mask "
                    "and would average over padding"
                )

    @property
    def batch_size(self) -> int:
        """How many inputs this batch carries."""
        return int(self.token_ids.shape[0])

    @property
    def any_truncated(self) -> bool:
        """Whether requirement 3.9 has anything to report for this batch."""
        return bool(self.truncated_indices)


# --------------------------------------------------------------------------
# The seams
# --------------------------------------------------------------------------


class TokenizerLoader(Protocol):
    """Turns a directory of tokenizer files into a tokenizer."""

    def __call__(self, path: Path, /) -> PreTrainedTokenizerBase: ...


class ModelAcquirer(Protocol):
    """The part of `acquire_model` this module uses.

    Narrow on purpose. Acquisition, its credential handling and its
    gated-repository diagnosis are `models.acquire`'s, not this module's; this
    protocol exists so a test can supply a directory without a network, not so
    an alternative acquisition can be written.
    """

    def __call__(
        self,
        profile: ModelProfile,
        /,
        *,
        allow_patterns: Sequence[str] | None = ...,
        cache_dir: Path | None = ...,
        revision: str = ...,
        progress: ProgressCallback | None = ...,
    ) -> AcquiredModel: ...


def _load_pretrained(path: Path) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if not isinstance(tokenizer, PreTrainedTokenizerBase):  # pragma: no cover
        raise TypeError(
            f"{type(tokenizer).__name__} is not a tokenizer this runtime can "
            "measure with"
        )
    return tokenizer


# --------------------------------------------------------------------------
# The tokenizer, and measurement by its rule
# --------------------------------------------------------------------------


class ModelTokenizer:
    """The active model's tokenizer, plus the rules the runtime measures by.

    Requirement 3.7's exposure is `tokenizer`; requirement 3.8's guarantee is
    that `count_tokens`, `exceeds_limit` and `encode` are one computation seen
    from three angles.
    """

    __slots__ = ("_profile", "_tokenizer", "_tokenizer_id")

    def __init__(
        self,
        *,
        profile: ModelProfile,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_id: str,
    ) -> None:
        if not tokenizer_id.strip():
            raise ValueError(
                "a tokenizer must be identifiable: design.md makes the "
                "tokenizer identity a revalidation trigger for downstream "
                "specs, and a blank one can never trigger it"
            )
        self._profile = profile
        self._tokenizer = tokenizer
        self._tokenizer_id = tokenizer_id

    @property
    def profile(self) -> ModelProfile:
        """The active model's profile."""
        return self._profile

    @property
    def model_id(self) -> str:
        """The active model's repository id."""
        return self._profile.model_id

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The active model's tokenizer itself (requirement 3.7)."""
        return self._tokenizer

    @property
    def tokenizer_id(self) -> str:
        """Stable identity of this tokenizer: repository and resolved commit.

        The value `EmbeddingContract.tokenizer_id` carries at task 5.3. It is
        pinned to a commit rather than a branch so that an upstream tokenizer
        change moves the identity, which is what makes design.md's revalidation
        trigger observable at all.
        """
        return self._tokenizer_id

    @property
    def max_input_tokens(self) -> int:
        """The published maximum input length (requirement 3.6).

        The compiled sequence length. Reading it through
        `ModelProfile.max_input_tokens` rather than off ``compiled_seq_len``
        keeps the choice of *which number is the contract* in the one place
        profiles.py made it.
        """
        return self._profile.max_input_tokens

    # -- measuring ---------------------------------------------------------

    def render(self, text: str | DocumentText, kind: TextKind) -> str:
        """The text as the model will actually see it (requirement 3.4).

        A bare string given as a document is an untitled document, which renders
        the absence sentinel into the title slot. A `DocumentText` given as a
        query is a caller error rather than a coercion: a title has no place in
        a query convention, and silently dropping it would embed something the
        caller did not ask for.
        """
        if kind is TextKind.QUERY:
            if isinstance(text, DocumentText):
                raise ValueError(
                    "a DocumentText cannot be measured or embedded as a query: "
                    "the query convention has no title slot, and dropping the "
                    "title silently would change the text being embedded"
                )
            return self._profile.render_query(text)
        document = DocumentText(text) if isinstance(text, str) else text
        return self._profile.render_document(document)

    def count_tokens(self, text: str | DocumentText, kind: TextKind) -> int:
        """How many tokens this input costs, by the rule the runtime applies.

        Counted over the *rendered* text, including the prefix template and the
        tokenizer's special tokens, because those are what the model is fed.
        This is the number requirement 3.8 tells a consumer to compare against
        `max_input_tokens`, and it is the same number `encode` decides
        truncation from.
        """
        return len(self._measure(self.render(text, kind)))

    def exceeds_limit(self, text: str | DocumentText, kind: TextKind) -> bool:
        """Whether this input will be shortened (requirements 3.8, 3.9)."""
        return bool(
            _over_limit([self.count_tokens(text, kind)], self.max_input_tokens)
        )

    def _measure(self, rendered: str) -> list[int]:
        """The complete, untruncated token ids for one rendered text.

        The single measurement. ``verbose=False`` suppresses the tokenizer
        library's own warning about sequences longer than its declared maximum:
        over-length input is a routine, reported condition here (3.9), and
        design.md's Monitoring decision leaves presentation to the caller, so
        measurement must not narrate to a stream nobody chose.
        """
        return [
            int(token)
            for token in self._tokenizer.encode(
                rendered,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
        ]

    def _fit(self, rendered: str) -> list[int]:
        """The rendered text shortened to the compiled length by the model's
        own truncation rule, which keeps the closing special token."""
        return [
            int(token)
            for token in self._tokenizer.encode(
                rendered,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_input_tokens,
                verbose=False,
            )
        ]

    # -- encoding ----------------------------------------------------------

    def encode_documents(
        self, texts: Sequence[str | DocumentText]
    ) -> EncodedBatch:
        """Model inputs for corpus content, under the document convention."""
        return self.encode(texts, TextKind.DOCUMENT)

    def encode_queries(self, texts: Sequence[str]) -> EncodedBatch:
        """Model inputs for search queries, under the query convention."""
        return self.encode(list(texts), TextKind.QUERY)

    def encode(
        self, texts: Sequence[str | DocumentText], kind: TextKind
    ) -> EncodedBatch:
        """Render, measure, shorten what does not fit, and pad to the graph.

        One row per input in input order (requirement 3.1), each row exactly
        ``compiled_seq_len`` wide. Truncation is decided from the measurement
        `count_tokens` returns and from nothing else, so the two cannot drift.
        """
        limit = self.max_input_tokens
        counts: list[int] = []
        rows: list[list[int]] = []
        for text in texts:
            rendered = self.render(text, kind)
            measured = self._measure(rendered)
            counts.append(len(measured))
            rows.append(
                self._fit(rendered)
                if _over_limit([len(measured)], limit)
                else measured
            )

        token_ids = np.full((len(rows), limit), self._pad_id(), dtype=np.int64)
        attention_mask = np.zeros((len(rows), limit), dtype=np.int64)
        for index, ids in enumerate(rows):
            if len(ids) > limit:
                raise ExecutionError(
                    f"the tokenizer returned {len(ids)} tokens for input "
                    f"{index} after being asked to truncate to {limit}; the "
                    "graph has no more slots, and quietly trimming here would "
                    "move the truncation decision away from the one place that "
                    "reports it",
                    model_id=self.model_id,
                    stage=TOKENIZATION_STAGE,
                )
            token_ids[index, : len(ids)] = ids
            attention_mask[index, : len(ids)] = 1

        return EncodedBatch(
            token_ids=token_ids,
            attention_mask=attention_mask,
            token_counts=tuple(counts),
            max_input_tokens=limit,
            truncated_indices=_over_limit(counts, limit),
        )

    def _pad_id(self) -> int:
        pad = self._tokenizer.pad_token_id
        return _FALLBACK_PAD_ID if pad is None else int(pad)


# --------------------------------------------------------------------------
# Obtaining the tokenizer
# --------------------------------------------------------------------------


def load_tokenizer(
    profile: ModelProfile,
    *,
    acquire: ModelAcquirer = acquire_model,
    loader: TokenizerLoader = _load_pretrained,
    revision: str = DEFAULT_REVISION,
    cache_dir: Path | None = None,
    progress: ProgressCallback | None = None,
) -> ModelTokenizer:
    """The active model's tokenizer, obtained through the acquisition path.

    Acquisition is `models.acquire.acquire_model`, restricted to
    `TOKENIZER_FILE_PATTERNS`, so a gated candidate raises
    `LicenseAcceptanceRequired` carrying the acceptance step (requirement 4.5)
    and the credential is discovered and redacted by the code already built for
    it. Nothing about credentials or gating is re-implemented here.

    Failures that are not acquisition's - a directory with no tokenizer in it, a
    file the tokenizer library cannot parse - surface as `PreparationError`
    naming `TOKENIZER_STAGE` and the model (requirements 4.6, 8.1).
    """
    acquired = acquire(
        profile,
        allow_patterns=TOKENIZER_FILE_PATTERNS,
        cache_dir=cache_dir,
        revision=revision,
        progress=progress,
    )
    try:
        tokenizer = loader(acquired.local_path)
    except Exception as error:
        raise PreparationError(
            f"the tokenizer for {profile.model_id} could not be loaded from "
            f"{acquired.local_path}: {type(error).__name__}: {error}",
            model_id=profile.model_id,
            stage=TOKENIZER_STAGE,
        ) from None
    return ModelTokenizer(
        profile=profile,
        tokenizer=tokenizer,
        tokenizer_id=f"{profile.model_id}@{acquired.revision}",
    )
