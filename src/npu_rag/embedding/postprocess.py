"""Pooling, the Dense stage, and L2 normalization (task 5.2).

design.md's decision "Post-process pooling, Dense, and normalization outside the
ONNX graph" puts these three stages here rather than in the exported model.
research.md's finding is the reason: an ONNX export converts only the
transformer trunk and emits *token* embeddings, while a sentence-embedding model
is a pipeline - Pooling -> Dense -> Normalize for EmbeddingGemma, Pooling ->
Normalize for the other two candidates. Folding those stages into the graph
would give every backend its own copy of them and let BF16 rounding differ
inside the normalization itself. Keeping them here makes requirement 3.5 true by
construction for every backend and collapses requirement 5.4's cross-backend
equivalence surface down to the transformer alone.

**Nothing here knows what ran the transformer.** This module is a function of
arrays. It never names a provider, a session, or ONNX Runtime, because the only
way pooling and normalization can be identical across three adapters is for
there to be one implementation that cannot tell them apart.

## The three ways this goes silently wrong

Each of them produces vectors of the right shape carrying the right norm, so no
shape check and no norm assertion downstream can see any of them. Only a
retrieval-quality measurement (6.4) would eventually notice, long after the
index was built.

1. **Mask-blind pooling.** Inputs are padded to the compiled length - 512 for
   EmbeddingGemma - so a mean over all positions is mostly a mean over padding.
   design.md calls the padding-mask interaction "where mean pooling silently
   breaks". `masked_mean_pool` therefore *selects* the attended positions rather
   than multiplying by the mask: a padded slot is not read at all, so whatever
   it holds - including a value arithmetic cannot survive - cannot reach the
   result. Dividing by the attended count, not the padded length, is the other
   half of the same guarantee.
2. **A skipped or misapplied Dense stage.** EmbeddingGemma chains two
   projections, 768 -> 3072 -> 768. Omitting them is undetectable downstream,
   and so is applying the wrong activation, which is why an activation this
   module does not recognise is refused rather than assumed to be the identity.
3. **The Dense order re-derived rather than read.** `dense.npz` records the
   pipeline order as data (`models/export.py`, `DENSE_ORDER_KEY`). Implementation
   Note 3.2: "Task 5.2 must consume ``order`` and never re-derive it by sorting
   the weight names." Sorting EmbeddingGemma's stage names would apply
   768 -> 3072 -> 768 backwards, and both directions chain, so nothing but the
   recorded order can tell them apart. `dense_layers_from_arrays` is the reader
   that consumes it; it also refuses a weight the order does not name, because
   dropping a projection nobody listed is the same corruption as skipping one
   that is listed.
4. **The wrong pooling rule** (task 5.5). Two of requirement 4.1's candidates
   pool by masked mean and `gte-modernbert-base` pools by CLS - the first
   position, which is what its own ``1_Pooling/config.json`` declares
   (``pooling_mode_cls_token = True``). Mean-pooling a CLS model is the same
   class of defect as skipping a Dense stage: same width, same norm, different
   meaning. ``ModelProfile.pooling`` says which rule the model wants, `pool`
   dispatches on it, and `finalize` takes it as a **required** argument. Until
   task 5.5 that field had no production consumer at all, so widening its
   annotation alone would have left a CLS profile silently mean-pooled - which
   is why the rule travels in rather than being assumed here.

## Why the CLS branch cannot reintroduce hazard 1

`cls_pool` has no ``attention_mask`` parameter. Not "ignores the mask" - cannot
be handed one. Position 0 is a real token for every batch this runtime produces
because `tokenize.EncodedBatch.__post_init__` refuses to construct a row whose
mask is anything but a run of real tokens followed by padding, so the padding is
always on the right and the first position is always the model's own ``[CLS]``.
That invariant is upstream of here and is not re-checked here; what is checked is
that a rule nobody implemented is refused rather than quietly replaced.

## What it refuses rather than papering over

A row that attends nothing has no mean. Returning zeros would hand the caller a
vector that cannot be normalized, and dividing by the count would produce a row
of `NaN` that survives every subsequent stage and lands in the index. Both are
refused: `ExecutionError` names the offending rows. The same applies to a vector
whose length is zero, which has no direction to normalize to.

Failures are `ExecutionError` - post-processing runs inside an embedding call,
not during preparation, so requirement 8.2's category is execution - and every
one of them carries the provider, the model, and the stage requirement 8.1 asks
for, which is why `provider` and `model_id` travel in as optional arguments.

## Where this sits

Directly above ``types`` and ``errors`` in design.md's dependency direction
(``types, errors -> reporting -> profiles -> environment -> models -> providers
-> service -> bench``) and below everything else. It needs the error taxonomy
and the provider vocabulary for its own failure reports, NumPy, and nothing
else. In particular it does not read ``profiles`` or ``models``: the caller
decides whether a profile declares a Dense stage and loads the weights, and
hands the arrays in. That is what keeps this a pure function of its arguments,
and it is also what lets the package-wide layer guard forbid the most - a module
placed this low may import almost nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from npu_rag.embedding.errors import ExecutionError
from npu_rag.embedding.types import ProviderChoice

__all__ = [
    "POOLING_RULES",
    "POSTPROCESS_STAGE",
    "SUPPORTED_ACTIVATIONS",
    "DenseLayer",
    "apply_dense_stages",
    "cls_pool",
    "dense_layers_from_arrays",
    "finalize",
    "l2_normalize",
    "masked_mean_pool",
    "pool",
]

#: The stage every failure here reports (8.1). Named for the phase of the
#: embedding flow rather than for this file, because that is what an operator
#: reading a traceback is trying to locate.
POSTPROCESS_STAGE: Final = "postprocess"

#: The pooling rules this module actually implements, spelled the way
#: ``ModelProfile.pooling`` spells them. `pool` refuses anything outside this
#: set, so a profile that declares a rule nobody wrote is a loud failure instead
#: of a vector that is the right width, the right norm and the wrong meaning.
#:
#: This set and ``ModelProfile.pooling``'s annotation state the same fact in two
#: places - ``profiles`` sits above this module and could import the set, but
#: making the declaration depend on the implementation would put behaviour in
#: what is meant to be data. ``tests/embedding/test_profiles.py`` compares the
#: two directly, so they cannot drift apart in silence.
POOLING_RULES: Final[frozenset[str]] = frozenset({"mean", "cls"})


def _identity(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return values


def _relu(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.maximum(values, np.zeros((), dtype=np.float32))


def _tanh(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.tanh(values, dtype=np.float32)


#: The activations this module computes exactly, keyed by the final component of
#: the dotted class path sentence-transformers records in a Dense module's
#: ``config.json`` (``torch.nn.modules.linear.Identity`` and friends), lowercased.
#:
#: The list is short on purpose. All three candidate models use ``Identity``
#: today (`models/export.py` read the real ``google/embeddinggemma-300m``
#: repository), and ``Tanh`` is sentence-transformers' own default for a Dense
#: module, so both are worth having. Anything else - ``GELU`` above all, whose
#: exact form needs an error function NumPy does not ship and whose tanh
#: approximation is a *different* function - is refused by
#: `apply_dense_stages` rather than approximated or quietly treated as the
#: identity. An approximation here would be the third silent-corruption path in
#: this module's docstring wearing a different name.
SUPPORTED_ACTIVATIONS: Final[
    Mapping[str, Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]]
] = {
    "identity": _identity,
    "linear": _identity,
    "relu": _relu,
    "tanh": _tanh,
}


@dataclass(frozen=True)
class DenseLayer:
    """One projection of the sentence pipeline, ready to apply.

    Deliberately *not* `models.export.DenseStage`, though it carries the same
    four facts. Importing that type would drag this module up above ``models``
    in the dependency direction for no gain: what post-processing needs is the
    arrays, and whoever loaded them - the service, in task 5.3 - is already in a
    layer that can see both. Keeping the value type local is what lets this
    module sit one rank above ``errors``.

    ``weight`` is ``(out_features, in_features)``, the orientation
    sentence-transformers stores and `models/export.py` persists, so nothing
    between the checkpoint and here has to guess a convention.
    """

    name: str
    weight: npt.NDArray[np.float32]
    bias: npt.NDArray[np.float32] | None
    activation: str


def _fail(
    message: str,
    *,
    provider: ProviderChoice | None,
    model_id: str | None,
) -> ExecutionError:
    return ExecutionError(
        message,
        provider=provider,
        model_id=model_id,
        stage=POSTPROCESS_STAGE,
    )


def _float32_matrix(
    values: npt.NDArray[Any],
    *,
    what: str,
    provider: ProviderChoice | None,
    model_id: str | None,
) -> npt.NDArray[np.float32]:
    """A two-dimensional float32 array, or a diagnosed failure."""
    if values.ndim != 2:
        raise _fail(
            f"{what} is {values.ndim}-dimensional; post-processing works on "
            f"(batch, width) matrices",
            provider=provider,
            model_id=model_id,
        )
    if values.dtype != np.float32:
        raise _fail(
            f"{what} carries {values.dtype} values, not float32. The backends "
            "return float32 and the whole post-processing path stays there, so "
            "a different precision here would be a cast nobody asked for",
            provider=provider,
            model_id=model_id,
        )
    return values


def masked_mean_pool(
    token_embeddings: npt.NDArray[np.float32],
    attention_mask: npt.NDArray[Any],
    *,
    provider: ProviderChoice | None = None,
    model_id: str | None = None,
) -> npt.NDArray[np.float32]:
    """Average each row's *attended* token embeddings, and nothing else.

    ``token_embeddings`` is ``(batch, compiled_seq_len, hidden)`` exactly as
    `TransformerBackend.run` returns it, and ``attention_mask`` is the
    ``(batch, compiled_seq_len)`` array the caller itself supplied to that run -
    design.md, TransformerBackend: the backend returns token embeddings alone,
    so there is only ever one mask and it cannot disagree with itself.

    Padded positions are *selected out* rather than multiplied by zero. Under a
    fixed 512 most positions of a short input are padding, and a multiply would
    carry whatever the graph left there into the sum whenever it is not an
    ordinary finite number. The divisor is the attended count of each row
    separately, which is the difference between a mean of the input and a mean
    of the input diluted by its padding.

    Raises `ExecutionError` for a row that attends nothing: it has no mean, and
    both alternatives - a row of ``NaN`` from dividing by zero, or a silently
    substituted zero vector - would travel all the way into the index.
    """
    if token_embeddings.ndim != 3:
        raise _fail(
            f"token embeddings are {token_embeddings.ndim}-dimensional; the "
            "backend contract is (batch, sequence, hidden)",
            provider=provider,
            model_id=model_id,
        )
    if token_embeddings.dtype != np.float32:
        raise _fail(
            f"token embeddings carry {token_embeddings.dtype} values, not "
            "float32, which is what every backend returns",
            provider=provider,
            model_id=model_id,
        )
    mask = np.asarray(attention_mask)
    if mask.ndim != 2:
        raise _fail(
            f"the attention mask is {mask.ndim}-dimensional; it is "
            "(batch, sequence), one flag per position",
            provider=provider,
            model_id=model_id,
        )
    if mask.shape != token_embeddings.shape[:2]:
        raise _fail(
            f"the attention mask is shaped {tuple(mask.shape)} but the token "
            f"embeddings are {tuple(token_embeddings.shape)}; a mask that does "
            "not line up would pool the wrong positions rather than fail",
            provider=provider,
            model_id=model_id,
        )
    if not bool(np.all((mask == 0) | (mask == 1))):
        raise _fail(
            "the attention mask carries values other than 0 and 1, so it is "
            "not a statement about which positions were attended; a weighted "
            "mask would change the pooling rule without changing any shape",
            provider=provider,
            model_id=model_id,
        )

    attended = mask.astype(bool)
    counts = attended.sum(axis=1)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise _fail(
            f"rows {empty.tolist()} attend no positions at all, so they have "
            "no mean. Dividing by zero would put NaN in the index and "
            "substituting a zero vector would put a direction there that no "
            "text produced",
            provider=provider,
            model_id=model_id,
        )

    selected = np.where(
        attended[:, :, None], token_embeddings, np.zeros((), dtype=np.float32)
    )
    totals = selected.sum(axis=1, dtype=np.float32)
    pooled: npt.NDArray[np.float32] = (
        totals / counts[:, None].astype(np.float32)
    ).astype(np.float32, copy=False)
    return pooled


def cls_pool(
    token_embeddings: npt.NDArray[np.float32],
    *,
    provider: ProviderChoice | None = None,
    model_id: str | None = None,
) -> npt.NDArray[np.float32]:
    """Take each row's **first** position and read nothing else (4.1).

    ``gte-modernbert-base`` declares ``pooling_mode_cls_token = True`` in its own
    ``1_Pooling/config.json``: the sentence vector is the ``[CLS]`` position's
    token embedding, not an average over the sequence.

    **There is deliberately no ``attention_mask`` parameter.** The mask-blind
    pooling hazard in this module's docstring is a hazard of reading padded
    positions; a function that cannot be handed a mask cannot be written to
    misuse one, and one that took a mask and ignored it would invite a later
    reader to "fix" the omission. Position 0 is guaranteed to be a real token
    rather than padding by `tokenize.EncodedBatch`, which refuses to construct a
    row whose mask is not a run of real tokens followed by padding.

    Raises `ExecutionError` for a zero-length sequence, which has no first
    position; NumPy would otherwise raise ``IndexError`` from inside a
    post-processing call that carries none of requirement 8.1's context.
    """
    if token_embeddings.ndim != 3:
        raise _fail(
            f"token embeddings are {token_embeddings.ndim}-dimensional; the "
            "backend contract is (batch, sequence, hidden)",
            provider=provider,
            model_id=model_id,
        )
    if token_embeddings.dtype != np.float32:
        raise _fail(
            f"token embeddings carry {token_embeddings.dtype} values, not "
            "float32, which is what every backend returns",
            provider=provider,
            model_id=model_id,
        )
    if token_embeddings.shape[1] == 0:
        raise _fail(
            "the sequence dimension is empty, so there is no first position to "
            "pool. A CLS model's vector is that position and nothing else, so "
            "there is no fallback that would mean anything",
            provider=provider,
            model_id=model_id,
        )

    # A copy, not the view ``token_embeddings[:, 0, :]`` would be: every other
    # function here hands back an array the caller does not already own, and a
    # view would alias the backend's output buffer.
    pooled: npt.NDArray[np.float32] = np.array(
        token_embeddings[:, 0, :], dtype=np.float32, copy=True
    )
    return pooled


def pool(
    token_embeddings: npt.NDArray[np.float32],
    attention_mask: npt.NDArray[Any],
    *,
    rule: str,
    provider: ProviderChoice | None = None,
    model_id: str | None = None,
) -> npt.NDArray[np.float32]:
    """Collapse token embeddings to one vector per row, by the declared rule.

    ``rule`` is ``ModelProfile.pooling``, carried in by the caller rather than
    decided here - the same reasoning that makes `dense_layers_from_arrays`'
    key names required arguments. An unrecognised rule is **refused**: falling
    back to the mean would give a CLS model vectors of the right width and the
    right norm that mean something else, which is exactly the failure this
    module's docstring exists to enumerate.

    ``attention_mask`` is accepted for every rule and forwarded only to the ones
    that pool over more than one position. The CLS branch never receives it.
    """
    if rule == "mean":
        return masked_mean_pool(
            token_embeddings,
            attention_mask,
            provider=provider,
            model_id=model_id,
        )
    if rule == "cls":
        return cls_pool(
            token_embeddings, provider=provider, model_id=model_id
        )
    raise _fail(
        f"the active model declares {rule!r} pooling, which this runtime does "
        "not implement. It is refused rather than pooled by some other rule: "
        "every rule here produces a vector of the same width carrying the same "
        "norm, so a substituted one would be undetectable downstream and would "
        f"only ever surface as poor retrieval. Known: {sorted(POOLING_RULES)}",
        provider=provider,
        model_id=model_id,
    )


def _activation_of(
    layer: DenseLayer,
    *,
    provider: ProviderChoice | None,
    model_id: str | None,
) -> Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]:
    key = layer.activation.rsplit(".", 1)[-1].strip().lower()
    function = SUPPORTED_ACTIVATIONS.get(key)
    if function is None:
        raise _fail(
            f"dense stage {layer.name!r} declares the activation "
            f"{layer.activation!r}, which this runtime does not compute. It is "
            "refused rather than treated as the identity or approximated: "
            "either would produce vectors of the right width and the right "
            f"norm that mean something else. Known: "
            f"{sorted(SUPPORTED_ACTIVATIONS)}",
            provider=provider,
            model_id=model_id,
        )
    return function


def apply_dense_stages(
    vectors: npt.NDArray[np.float32],
    stages: Sequence[DenseLayer],
    *,
    provider: ProviderChoice | None = None,
    model_id: str | None = None,
) -> npt.NDArray[np.float32]:
    """Apply the projections **in the order given**, which is pipeline order.

    ``stages`` is a sequence, not a mapping, precisely so the order is the
    caller's recorded one and never re-derived here. `dense_layers_from_arrays`
    is the supported way to build it from a loaded ``dense.npz``.

    Each stage is a linear projection followed by its declared activation:
    ``activation(x @ weight.T + bias)``. Widths are checked stage by stage, so
    weights that do not chain are reported where the break is rather than
    surfacing as a NumPy shape error with no idea which model it belongs to.
    """
    current = np.array(
        _float32_matrix(
            vectors,
            what="the pooled vectors",
            provider=provider,
            model_id=model_id,
        ),
        dtype=np.float32,
        copy=True,
    )

    for layer in stages:
        weight = _float32_matrix(
            layer.weight,
            what=f"the weight of dense stage {layer.name!r}",
            provider=provider,
            model_id=model_id,
        )
        if weight.shape[1] != current.shape[1]:
            raise _fail(
                f"dense stage {layer.name!r} consumes {weight.shape[1]} "
                f"features but is being given {current.shape[1]}. The stages "
                "must be applied in the recorded pipeline order; two "
                "projections that chain in both directions each look plausible "
                "alone, and the wrong order is the same shape and the same norm "
                "carrying a different meaning",
                provider=provider,
                model_id=model_id,
            )

        projected = current @ weight.T
        if layer.bias is not None:
            bias = layer.bias
            if bias.dtype != np.float32:
                raise _fail(
                    f"the bias of dense stage {layer.name!r} carries "
                    f"{bias.dtype} values, not float32",
                    provider=provider,
                    model_id=model_id,
                )
            if bias.shape != (weight.shape[0],):
                raise _fail(
                    f"the bias of dense stage {layer.name!r} is shaped "
                    f"{tuple(bias.shape)}, which does not match its "
                    f"{weight.shape[0]} outputs",
                    provider=provider,
                    model_id=model_id,
                )
            projected = projected + bias

        activation = _activation_of(
            layer, provider=provider, model_id=model_id
        )
        current = activation(projected.astype(np.float32, copy=False)).astype(
            np.float32, copy=False
        )

    return current


def l2_normalize(
    vectors: npt.NDArray[np.float32],
    *,
    provider: ProviderChoice | None = None,
    model_id: str | None = None,
) -> npt.NDArray[np.float32]:
    """Scale every row to unit length (3.5).

    Requirement 3.5 exists so cosine similarity and dot-product similarity are
    the same operation downstream; `vector-index` is told not to re-normalize
    (requirements.md, Adjacent expectations), so this is the only place it
    happens.

    The lengths are accumulated in float64 and the result is returned in
    float32. That is a single rounding on the way out instead of two roundings
    plus a float32 accumulation over 768 squares, so the rows come back closer
    to unit length; it is deterministic either way, and the values it divides
    are still exactly the ones the Dense stage produced.

    Raises `ExecutionError` for a row of zeros, which has no direction.
    """
    matrix = _float32_matrix(
        vectors,
        what="the vectors to normalize",
        provider=provider,
        model_id=model_id,
    )
    lengths = np.sqrt(np.square(matrix, dtype=np.float64).sum(axis=1))
    unusable = np.flatnonzero(~(lengths > 0.0))
    if unusable.size:
        raise _fail(
            f"rows {unusable.tolist()} have no length to normalize by, so they "
            "have no direction either. A zero or non-finite vector cannot be "
            "made unit-norm, and passing it on would break the equivalence of "
            "cosine and dot-product similarity that requirement 3.5 exists to "
            "guarantee",
            provider=provider,
            model_id=model_id,
        )
    scaled: npt.NDArray[np.float32] = (matrix / lengths[:, None]).astype(
        np.float32, copy=False
    )
    return scaled


def finalize(
    token_embeddings: npt.NDArray[np.float32],
    attention_mask: npt.NDArray[Any],
    *,
    pooling: str,
    dense: Sequence[DenseLayer] = (),
    provider: ProviderChoice | None = None,
    model_id: str | None = None,
) -> npt.NDArray[np.float32]:
    """Token embeddings in, one unit vector per input row out.

    The single entry point design.md's traceability table names for requirements
    3.5 and 3.10, and the order is the sentence pipeline's own: pool, then
    project, then normalize. Normalizing *before* the projections would still
    produce unit-length output - the projections carry a bias, so it would point
    somewhere else - which is why the order is fixed here rather than left to
    the caller to assemble.

    ``pooling`` is ``ModelProfile.pooling`` and has **no default** (task 5.5).
    Defaulting it to ``"mean"`` would be the whole defect back again: a caller
    that forgot to pass it would mean-pool a CLS model and produce vectors of
    the right width and the right norm that mean something else. Requiring it
    makes stating the model's rule the only way to call this function at all.

    ``dense`` is empty for a profile that declares no Dense stage, and is the
    recorded pipeline order for one that does. Row order is the input's
    (requirement 3.1) and identical input gives bitwise identical output
    (requirement 3.10): every step is a fixed sequence of NumPy reductions over
    the same values, with no dependence on batch position, iteration order, or
    anything the caller did before.
    """
    pooled = pool(
        token_embeddings,
        attention_mask,
        rule=pooling,
        provider=provider,
        model_id=model_id,
    )
    projected = apply_dense_stages(
        pooled, dense, provider=provider, model_id=model_id
    )
    return l2_normalize(projected, provider=provider, model_id=model_id)


def _one_string(
    value: npt.NDArray[Any],
    *,
    what: str,
    provider: ProviderChoice | None,
    model_id: str | None,
) -> str:
    flat = np.asarray(value).ravel()
    if flat.size != 1:
        raise _fail(
            f"{what} holds {flat.size} values where exactly one was expected",
            provider=provider,
            model_id=model_id,
        )
    return str(flat[0])


def dense_layers_from_arrays(
    arrays: Mapping[str, npt.NDArray[Any]],
    *,
    order_key: str,
    weight_prefix: str,
    bias_prefix: str,
    activation_prefix: str,
    provider: ProviderChoice | None = None,
    model_id: str | None = None,
) -> tuple[DenseLayer, ...]:
    """Read the Dense stages out of a loaded ``dense.npz``, in pipeline order.

    The four key names are **required arguments with no defaults**, and that is
    deliberate. They belong to `models/export.py`, which writes the file; this
    module sits below ``models`` and must not import it, and copying its
    constants here would let the two drift apart silently. Passing them in makes
    the caller state the contract it is reading under, the same reasoning that
    made `resolve_backend`'s ``factories`` required rather than defaulted.

    The order comes from ``arrays[order_key]`` and from nowhere else. Sorting the
    weight names instead would apply EmbeddingGemma's 768 -> 3072 -> 768 chain
    backwards - both directions chain, so neither the shapes nor the norms
    change - which Implementation Note 3.2 records as the reason the order is
    persisted as data at all.

    A weight the recorded order does not name is refused rather than skipped: a
    projection dropped because nobody listed it is the same silent corruption as
    one skipped because it was listed in the wrong place.
    """
    if order_key not in arrays:
        raise _fail(
            f"the dense weights carry no {order_key!r} entry, so the pipeline "
            "order is unknown. It is not recoverable by sorting the weight "
            "names: the stages chain in both directions, and applying them "
            "backwards changes neither the width nor the norm of the result",
            provider=provider,
            model_id=model_id,
        )

    order = [
        str(name) for name in np.asarray(arrays[order_key]).ravel().tolist()
    ]
    layers: list[DenseLayer] = []
    for name in order:
        weight_key = f"{weight_prefix}{name}"
        if weight_key not in arrays:
            raise _fail(
                f"the pipeline order names a dense stage {name!r} whose "
                f"weights ({weight_key!r}) are not in the file",
                provider=provider,
                model_id=model_id,
            )
        activation_key = f"{activation_prefix}{name}"
        if activation_key not in arrays:
            raise _fail(
                f"dense stage {name!r} records no activation. It is refused "
                "rather than assumed to be the identity, because a stage whose "
                "activation is guessed wrong produces the right width and the "
                "right norm and the wrong meaning",
                provider=provider,
                model_id=model_id,
            )
        bias = arrays.get(f"{bias_prefix}{name}")
        layers.append(
            DenseLayer(
                name=name,
                weight=np.asarray(arrays[weight_key], dtype=np.float32),
                bias=(
                    None if bias is None else np.asarray(bias, dtype=np.float32)
                ),
                activation=_one_string(
                    arrays[activation_key],
                    what=f"the activation of dense stage {name!r}",
                    provider=provider,
                    model_id=model_id,
                ),
            )
        )

    named = set(order)
    unlisted = sorted(
        key[len(weight_prefix) :]
        for key in arrays
        if key.startswith(weight_prefix)
        and key[len(weight_prefix) :] not in named
    )
    if unlisted:
        raise _fail(
            f"the dense weights carry projections {unlisted} that the recorded "
            f"pipeline order does not name. Applying only the listed ones would "
            "drop a stage of the sentence pipeline, which is invisible in the "
            "shape and in the norm of every vector that follows",
            provider=provider,
            model_id=model_id,
        )

    return tuple(layers)
