"""The three candidate model profiles, as data (task 2.2).

Requirement 4.1 names exactly three candidates - ``embeddinggemma-300m``,
``bge-large-en-v1.5`` and ``nomic-embed-text-v1.5`` - and requirement 4.2 makes
the first of them the *initial* default, held only until the benchmark document
supersedes that choice on measured evidence. This module is where each model's
behaviour is declared rather than coded: design.md's simplification note is
explicit that there is "no model registry, plugin discovery, or configuration
DSL. Three profiles in a module-level mapping."

**The one number that must not be confused with another.** A model's
architectural context limit and the length its graph was actually compiled at
are different facts, and requirement 3.6 publishes the second one. NPU
compilation fixes the shape, so a consumer that chunked to EmbeddingGemma's
2048-token limit would have three quarters of every chunk silently truncated at
512 while every call reported success (research.md, "Decision: Publish the
compiled sequence length, not the model's architectural limit"). Both numbers
are therefore carried as separate fields, and ``max_input_tokens`` - the value
3.6 reports - is unambiguously the compiled one.

**Templates are quoted, not composed.** Requirement 3.4's "document-side
convention" is a published string belonging to the model, and EmbeddingGemma's
document form is a template with a *title* slot rather than a bare prefix. A
one-character drift embeds text the model was not trained to see: the vectors
keep their shape and their norm, retrieval quality drops, and nothing raises.

This module sits between ``types``/``errors``/``reporting`` and everything else
in design.md's dependency direction, so it reads ``types`` and nothing to its
right.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter
from types import MappingProxyType
from typing import Literal

from npu_rag.embedding.types import DocumentText

__all__ = [
    "DEFAULT_COMPILED_SEQ_LEN",
    "INITIAL_DEFAULT_MODEL",
    "MISSING_TITLE_SENTINEL",
    "PROFILES",
    "ModelProfile",
    "initial_default_profile",
    "profile_for",
]

#: Rendered into a document template's title slot when the document has no
#: title. research.md writes the convention as ``title: {title | "none"} | text:
#: {content}`` - the absent title becomes the literal word, not an empty slot.
MISSING_TITLE_SENTINEL = "none"

#: The static length every profile is compiled at unless a benchmark result
#: argues otherwise. research.md records 512 as an assumption to be validated:
#: padding every input to 512 may dominate throughput for short chunks, in which
#: case a 256 profile is warranted - a benchmark output, not a design
#: commitment.
DEFAULT_COMPILED_SEQ_LEN = 512

#: The only slots any template may contain. ``title`` appears on the document
#: side alone; a query has no title to render.
_DOCUMENT_SLOTS = frozenset({"title", "content"})
_QUERY_SLOTS = frozenset({"content"})


@dataclass(frozen=True)
class ModelProfile:
    """Everything the runtime needs to know about one candidate model.

    A value object (design.md, Domain Model), and the aggregate root for model
    behaviour: ``compiled_seq_len`` and ``batch_size`` are fixed at export and
    must equal the values recorded in any artifact manifest claiming to serve
    this profile (requirement 4.7).
    """

    #: Hugging Face repository id, as acquisition asks for it.
    model_id: str
    #: Output vector width. A revalidation trigger for ``vector-index``.
    dimension: int
    #: The static length the graph is compiled at - and the value requirement
    #: 3.6 publishes as the maximum input token length. Read
    #: ``max_input_tokens`` rather than this field where the *contract* is
    #: meant, so the intent survives a later reader.
    compiled_seq_len: int
    #: What the model architecture could attend to if the shape were not fixed.
    #: Recorded because it is a genuine property of the model and because the
    #: gap between it and ``compiled_seq_len`` is exactly what a consumer must
    #: not be told. Never published as the maximum.
    architectural_context_limit: int
    #: Inputs per forward pass, fixed at export alongside the sequence length.
    batch_size: int
    #: How token embeddings collapse to one vector. Masked mean pooling for all
    #: three candidates; the annotation is the constraint.
    pooling: Literal["mean"]
    #: Whether the sentence pipeline has a Dense projection between pooling and
    #: normalization. EmbeddingGemma does, and skipping it produces vectors of
    #: the right shape and the right norm that are semantically wrong - a defect
    #: only the retrieval-quality measurement (6.4) can detect.
    has_dense_stage: bool
    #: Requirement 3.4's document-side convention, verbatim from the model card.
    document_template: str
    #: Requirement 3.4's query-side convention, verbatim from the model card.
    query_template: str
    #: Whether the repository requires licence acceptance before download
    #: (requirement 4.5).
    license_gated: bool
    #: Where the operator accepts those terms. Present exactly when
    #: ``license_gated`` is true, because ``LicenseAcceptanceRequired`` cannot
    #: be raised without one.
    license_acceptance_url: str | None = None

    def __post_init__(self) -> None:
        for field in ("dimension", "compiled_seq_len", "batch_size"):
            value: int = getattr(self, field)
            if value <= 0:
                raise ValueError(
                    f"{field} must be positive, got {value}: a profile "
                    "describes a graph that was actually exported"
                )
        if self.compiled_seq_len > self.architectural_context_limit:
            raise ValueError(
                f"compiled_seq_len {self.compiled_seq_len} exceeds the "
                f"architectural context limit {self.architectural_context_limit}"
                f" of {self.model_id}: no such graph can be exported"
            )
        if self.license_gated and not self.license_acceptance_url:
            raise ValueError(
                "a gated model must carry a license_acceptance_url: "
                "requirement 4.5 reports the acceptance step, not just the fact "
                "that acceptance is needed"
            )
        if not self.license_gated and self.license_acceptance_url:
            raise ValueError(
                "an ungated model must not carry a license_acceptance_url: the "
                "two fields state one fact, and disagreeing would leave "
                "requirement 4.5's decision depending on which one was read"
            )
        _validate_template(
            self.document_template, "document_template", _DOCUMENT_SLOTS
        )
        _validate_template(self.query_template, "query_template", _QUERY_SLOTS)

    @property
    def name(self) -> str:
        """The short name requirement 4.1 uses, and this profile's key."""
        return self.model_id.rpartition("/")[2]

    @property
    def max_input_tokens(self) -> int:
        """Requirement 3.6's maximum input token length.

        The compiled length, never the architectural limit. This property exists
        so the choice is made once, here, instead of at every call site that
        might reach for the larger and more flattering number.
        """
        return self.compiled_seq_len

    @property
    def has_title_slot(self) -> bool:
        """Whether this model's document convention renders a title at all."""
        return "title" in _slots_of(self.document_template)

    def render_document(self, document: DocumentText) -> str:
        """Apply the document-side convention to a corpus text (3.4).

        An absent title becomes ``MISSING_TITLE_SENTINEL``. Models whose
        convention has no title slot ignore the title rather than inventing a
        place to put it - inserting it anyway would embed text the model was not
        trained on, which is the failure this template is here to prevent.
        """
        title = (
            MISSING_TITLE_SENTINEL if document.title is None else document.title
        )
        return self.document_template.format(
            title=title, content=document.content
        )

    def render_query(self, query: str) -> str:
        """Apply the query-side convention to a search string (3.4)."""
        return self.query_template.format(content=query)


def _slots_of(template: str, field: str = "template") -> frozenset[str]:
    """The substitution slots in ``template``.

    A malformed template raises here, at declaration, rather than once per
    embedded text much later.
    """
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"{field} is not a valid template: {error}") from error
    return frozenset(name for _, name, _, _ in parsed if name is not None)


def _validate_template(template: str, field: str, allowed: frozenset[str]) -> None:
    slots = _slots_of(template, field)
    unknown = sorted(slots - allowed)
    if unknown:
        raise ValueError(
            f"{field} has slots nothing will fill: {', '.join(unknown)}. "
            f"Only {', '.join(sorted(allowed))} are substituted"
        )
    if "content" not in slots:
        raise ValueError(
            f"{field} has no {{content}} slot, so the text being embedded would "
            "be dropped entirely"
        )


_EMBEDDINGGEMMA_300M = ModelProfile(
    model_id="google/embeddinggemma-300m",
    dimension=768,
    compiled_seq_len=DEFAULT_COMPILED_SEQ_LEN,
    # Gemma 3 based, 2048 tokens - four times what the graph is compiled at.
    architectural_context_limit=2048,
    batch_size=1,
    pooling="mean",
    # Pooling -> Dense -> Normalize. The Dense stage is not optional.
    has_dense_stage=True,
    document_template="title: {title} | text: {content}",
    # The query prefix is task-specific; retrieval is the task this project
    # has. The other published tasks (question answering, fact checking,
    # classification, clustering, sentence similarity, code retrieval) are not
    # modelled, because nothing here would exercise them.
    query_template="task: search result | query: {content}",
    # Gemma Terms of Use, `gated: manual` on Hugging Face.
    license_gated=True,
    license_acceptance_url="https://huggingface.co/google/embeddinggemma-300m",
)

_BGE_LARGE_EN_V1_5 = ModelProfile(
    model_id="BAAI/bge-large-en-v1.5",
    dimension=1024,
    compiled_seq_len=DEFAULT_COMPILED_SEQ_LEN,
    # The one candidate whose architectural limit and compiled length coincide.
    # They are still two facts: this profile publishes 512 because it compiled
    # at 512, not because the model stops there.
    architectural_context_limit=512,
    batch_size=1,
    pooling="mean",
    has_dense_stage=False,
    # Asymmetric the other way round: the instruction goes on the query, and
    # documents are embedded bare.
    document_template="{content}",
    query_template=(
        "Represent this sentence for searching relevant passages: {content}"
    ),
    license_gated=False,
)

_NOMIC_EMBED_TEXT_V1_5 = ModelProfile(
    model_id="nomic-ai/nomic-embed-text-v1.5",
    dimension=768,
    compiled_seq_len=DEFAULT_COMPILED_SEQ_LEN,
    architectural_context_limit=8192,
    batch_size=1,
    pooling="mean",
    has_dense_stage=False,
    document_template="search_document: {content}",
    query_template="search_query: {content}",
    license_gated=False,
)

#: Requirement 4.1's three candidates, keyed by the short names it uses.
PROFILES: Mapping[str, ModelProfile] = MappingProxyType(
    {
        "embeddinggemma-300m": _EMBEDDINGGEMMA_300M,
        "bge-large-en-v1.5": _BGE_LARGE_EN_V1_5,
        "nomic-embed-text-v1.5": _NOMIC_EMBED_TEXT_V1_5,
    }
)

#: Requirement 4.2: the *initial* default candidate. It is held only until the
#: benchmark document (requirement 7.3) recommends a default on measured
#: evidence; when that happens this constant changes and the profiles do not.
INITIAL_DEFAULT_MODEL = "embeddinggemma-300m"

_mismatched = sorted(
    key for key, profile in PROFILES.items() if key != profile.name
)
if _mismatched:  # pragma: no cover - a declaration error, caught at import
    raise ValueError(f"profile keys disagree with their model ids: {_mismatched}")

if INITIAL_DEFAULT_MODEL not in PROFILES:  # pragma: no cover - as above
    raise ValueError(f"the initial default {INITIAL_DEFAULT_MODEL!r} is undeclared")


def profile_for(name: str) -> ModelProfile:
    """The profile for one of requirement 4.1's three candidates.

    An unknown name is a caller error, not a fallback: requirement 4.6 forbids
    substituting a different model, and the quietest possible substitution would
    be one made here.
    """
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; requirement 4.1 declares exactly "
            f"{', '.join(sorted(PROFILES))}"
        ) from None


def initial_default_profile() -> ModelProfile:
    """The candidate used until the benchmark supersedes it (4.2)."""
    return PROFILES[INITIAL_DEFAULT_MODEL]
