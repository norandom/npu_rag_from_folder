"""Embedding-model compatibility policy for an existing vector index.

One module so the policy exists once. Both ``npu-rag index`` (through
``IndexSynchronizer.sync``, which calls it before writing anything) and
``npu-rag search`` (through :meth:`npu_rag.index.service.VectorIndex.verify_model_identity`)
reach the same decision, with the same message, from the same table scan.

The failure this guards is specific and quiet. A *dimension* change already
fails at the write, in ``chunk_record_to_schema``. A **same-dimension
different model** fails nowhere: EmbeddingGemma 2 is also 768-d, so pointing
``model_name`` at it over an existing index would produce one table holding two
incompatible vector spaces, with no error, no way to tell the rows apart
afterwards, and every similarity score that crossed the boundary meaningless
but entirely plausible-looking.
"""

from __future__ import annotations

from npu_rag.index.errors import ConfigurationError
from npu_rag.index.schema import UNSTAMPED_MODEL_IDENTITY
from npu_rag.index.store import LanceIndexStore

#: How an absent identity is rendered inside a diagnostic. Never ``""``, which
#: would leave the operator reading an empty gap in a sentence.
UNSTAMPED_LABEL: str = "(no identity recorded)"


def _describe(identity: str) -> str:
    """Render one stored identity for a human reading an error message."""
    return UNSTAMPED_LABEL if identity == UNSTAMPED_MODEL_IDENTITY else identity


def _chunks(count: int) -> str:
    """``"1 chunk"`` / ``"2 chunks"`` - the message is read by an operator."""
    return f"{count} chunk" if count == 1 else f"{count} chunks"


def _inventory(counts: dict[str, int]) -> str:
    """Render ``identity (N chunks)`` pairs in a stable order."""
    return ", ".join(
        f"{_describe(identity)} ({_chunks(counts[identity])})"
        for identity in sorted(counts)
    )


def verify_model_identity(
    store: LanceIndexStore,
    configured_identity: str,
) -> None:
    """Raise unless ``configured_identity`` may write to, or query, ``store``.

    ``configured_identity`` is the stamp the caller is about to use, as
    :func:`npu_rag.index.schema.format_model_identity` builds it, or
    ``UNSTAMPED_MODEL_IDENTITY`` when the caller has none to offer.

    The decision table, in the order the branches are taken:

    ===========================  ==========================  ===========
    index holds                  caller supplies             outcome
    ===========================  ==========================  ===========
    nothing (empty table)        anything                    accepted
    two or more identities       anything                    **mixed**
    one identity + unstamped     anything                    **mixed**
    the same identity            that identity               accepted
    only unstamped rows          an identity                 **unstamped**
    only unstamped rows          nothing                     accepted
    one identity                 nothing                     **unsupplied**
    one identity                 a different identity        **mismatch**
    ===========================  ==========================  ===========

    An empty index is accepted because the first sync has to be able to happen.

    An entirely unstamped index is **rejected** for a caller that does have an
    identity, rather than being waved through as "unknown". A pass would be an
    unverifiable claim about rows whose vector space nobody recorded, and the
    cheap-migration argument for passing does not apply: no index exists yet
    anywhere, so every unstamped index that will ever be met is a caller that
    forgot to stamp - which is the thing that must be loud.

    Raises:
        ConfigurationError: the index and the configured model disagree, or the
            index already disagrees with itself. Chosen from the existing
            taxonomy because nothing is malformed (``SchemaError``), no storage
            operation failed (``StoreError``), no query was wrong
            (``QueryError``), and the condition has to be raisable from the
            search path as well as the sync path (``SyncError``).
    """
    counts = store.model_identity_counts()
    if not counts:
        return

    stored = tuple(sorted(counts))
    location = store.db_path

    if len(stored) > 1:
        raise ConfigurationError(
            f"The index at {location} already holds vectors from more than one "
            # "vector space", not "embedding model": the partially stamped case
            # may name only one model, and unstamped rows are a second space
            # whether or not they came from a second model.
            f"vector space: {_inventory(counts)}. Similarity scores that "
            "cross that boundary are meaningless, so no model can be declared "
            "compatible with it. Delete the index and rebuild it with a single "
            f"model (the configured one is {_describe(configured_identity)})."
        )

    only = stored[0]
    if only == configured_identity:
        return

    total = counts[only]
    if only == UNSTAMPED_MODEL_IDENTITY:
        raise ConfigurationError(
            f"The index at {location} holds {_chunks(total)} with no recorded "
            "embedding model, so there is no way to tell whether they came "
            f"from {_describe(configured_identity)}. The index predates "
            "embedding-model stamping or was written by a caller that supplied "
            "none. Delete the index and rebuild it so every row is stamped."
        )

    if configured_identity == UNSTAMPED_MODEL_IDENTITY:
        raise ConfigurationError(
            f"The index at {location} was built with {only} "
            f"({_chunks(total)}), but this run supplied no embedding-model "
            "identity. Writing unstamped rows into it would make the table "
            "permanently ambiguous. Pass the identity of the model being "
            "used, or delete the index and rebuild it."
        )

    raise ConfigurationError(
        f"Embedding model mismatch: the index at {location} was built with "
        f"{only} ({_chunks(total)}), but the configured model is "
        f"{configured_identity}. Vectors from different models are not "
        "comparable even at the same width, so mixing them would produce "
        f"plausible but meaningless scores. Either set the model back to "
        f"{only}, or delete the index and rebuild it with "
        f"{configured_identity}."
    )
