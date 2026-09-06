"""Post-processing shared by every backend (task 5.2).

design.md's decision "Post-process pooling, Dense, and normalization outside the
ONNX graph" puts three stages in NumPy on the service side of the port. All
three fail *silently*: a mask-blind pool, a skipped Dense stage, and a Dense
stage applied in the wrong order each produce vectors of the right shape with
the right norm and the wrong meaning. No shape check and no norm check can see
any of them, so the tests below pin numbers, not properties.

Every reference value here is computed **by hand** and written as a literal. The
fixture is chosen so the arithmetic is exact in float32 - the attended counts are
2 and 4, and every token component is a multiple of 0.5 - which is what lets the
pooling assertions be exact equality rather than a tolerance wide enough to hide
a real defect.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from npu_rag.embedding.errors import EmbeddingRuntimeError, ExecutionError
from npu_rag.embedding.postprocess import (
    POSTPROCESS_STAGE,
    DenseLayer,
    apply_dense_stages,
    dense_layers_from_arrays,
    finalize,
    l2_normalize,
    masked_mean_pool,
)
from npu_rag.embedding.types import ProviderChoice

# --------------------------------------------------------------------------
# The fixture, and the reference values computed by hand from it
# --------------------------------------------------------------------------

#: What a padded position holds. Large, finite, and nowhere near any attended
#: value, so a pool that reaches into padding is off by hundreds rather than by
#: a rounding error.
PAD_VALUE = 1000.0

#: Two rows padded to a fixed length of six, exactly as a backend returns them.
#: Row 0 attends four positions, row 1 attends two - deliberately different, so
#: dividing by any single count is wrong for one of them.
TOKEN_EMBEDDINGS: npt.NDArray[np.float32] = np.array(
    [
        [
            [1.0, 2.0, -1.0],
            [3.0, 4.0, 1.0],
            [0.5, 1.0, 2.0],
            [-0.5, 5.0, 6.0],
            [PAD_VALUE, PAD_VALUE, PAD_VALUE],
            [PAD_VALUE, PAD_VALUE, PAD_VALUE],
        ],
        [
            [2.0, -3.0, 0.5],
            [4.0, 1.0, 1.5],
            [PAD_VALUE, PAD_VALUE, PAD_VALUE],
            [PAD_VALUE, PAD_VALUE, PAD_VALUE],
            [PAD_VALUE, PAD_VALUE, PAD_VALUE],
            [PAD_VALUE, PAD_VALUE, PAD_VALUE],
        ],
    ],
    dtype=np.float32,
)

ATTENTION_MASK: npt.NDArray[np.int64] = np.array(
    [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)

#: Hand-computed masked mean.
#:
#: Row 0, over its four attended positions only:
#:   column 0: (1.0 + 3.0 + 0.5 - 0.5) / 4 = 4.0 / 4 = 1.0
#:   column 1: (2.0 + 4.0 + 1.0 + 5.0) / 4 = 12.0 / 4 = 3.0
#:   column 2: (-1.0 + 1.0 + 2.0 + 6.0) / 4 = 8.0 / 4 = 2.0
#: Row 1, over its two attended positions only:
#:   column 0: (2.0 + 4.0) / 2 = 3.0
#:   column 1: (-3.0 + 1.0) / 2 = -1.0
#:   column 2: (0.5 + 1.5) / 2 = 1.0
POOLED_BY_HAND: npt.NDArray[np.float32] = np.array(
    [[1.0, 3.0, 2.0], [3.0, -1.0, 1.0]], dtype=np.float32
)

#: Two projections whose names sort into the *reverse* of their pipeline order,
#: which is the point: ``sorted(("2_Dense", "10_Dense"))`` is
#: ``["10_Dense", "2_Dense"]``. Task 3.2's review rejected a test whose stage
#: names happened to be alphabetical, because sorting them changed nothing and
#: the assertion proved nothing.
FIRST_STAGE_NAME = "2_Dense"
SECOND_STAGE_NAME = "10_Dense"

#: Both are 3x3, so *either* order is shape-valid and a wrong order produces a
#: different number rather than a convenient exception. Neither is a scalar
#: multiple of the other and they do not commute.
#:
#: **Three separate degeneracies are avoided here, and each had to be checked on
#: its own axis.** Task 5.2's first review rejected a fixture that was
#: discriminating for the property it had been designed for - stage order - and
#: degenerate for two others nobody had asked about:
#:
#: 1. A weight equal to its own transpose pins no orientation, so
#:    ``x @ weight`` and ``x @ weight.T`` agree and the
#:    ``(out_features, in_features)`` convention is asserted by nothing.
#:    `SECOND_WEIGHT` is therefore the 3-cycle, whose transpose is the *inverse*
#:    cycle. (`FIRST_WEIGHT` is diagonal and unavoidably symmetric; the second
#:    stage is what carries the orientation assertion.)
#: 2. A bias supported only on components its own weight scales by exactly 1.0
#:    makes ``activation(x @ W.T + b)`` and ``activation((x + b) @ W.T)`` agree,
#:    so *when* the bias is added is pinned by nothing. `FIRST_BIAS` is
#:    therefore nonzero in component 1, which `FIRST_WEIGHT` scales by 2.0.
#: 3. Stage names that sort into pipeline order, which is the one this fixture
#:    was already built against (see the names above).
#:
#: The three non-vacuity tests below assert each of these properties rather than
#: trusting the comment: non-vacuity is a claim about one property at a time,
#: not a badge a fixture earns once.
FIRST_WEIGHT: npt.NDArray[np.float32] = np.array(
    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)
FIRST_BIAS: npt.NDArray[np.float32] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
SECOND_WEIGHT: npt.NDArray[np.float32] = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
)

IDENTITY_ACTIVATION = "torch.nn.modules.linear.Identity"


def pipeline_stages() -> tuple[DenseLayer, ...]:
    """The two projections in pipeline order."""
    return (
        DenseLayer(
            name=FIRST_STAGE_NAME,
            weight=FIRST_WEIGHT,
            bias=FIRST_BIAS,
            activation=IDENTITY_ACTIVATION,
        ),
        DenseLayer(
            name=SECOND_STAGE_NAME,
            weight=SECOND_WEIGHT,
            bias=None,
            activation=IDENTITY_ACTIVATION,
        ),
    )


#: Hand-computed Dense output, applied to `POOLED_BY_HAND` in pipeline order.
#:
#: A stage is ``activation(W x + b)`` for a row ``x``. "2_Dense" is
#: ``W = diag(1, 2, 1)`` with ``b = [1, 1, 0]``; "10_Dense" is the 3-cycle
#: ``W x = [x1, x2, x0]`` with no bias.
#:
#: Row 0 is [1, 3, 2].
#:   after "2_Dense":  W x  = [1*1, 2*3, 1*2] = [1, 6, 2]
#:                     + b  = [1+1, 6+1, 2+0] = [2, 7, 2]
#:   after "10_Dense": [y1, y2, y0]           = [7, 2, 2]
#: Row 1 is [3, -1, 1].
#:   after "2_Dense":  W x  = [3, -2, 1]
#:                     + b  = [4, -1, 1]
#:   after "10_Dense": [y1, y2, y0]           = [-1, 1, 4]
DENSED_BY_HAND: npt.NDArray[np.float32] = np.array(
    [[7.0, 2.0, 2.0], [-1.0, 1.0, 4.0]], dtype=np.float32
)

#: The same two projections applied in the *reversed* order, which is what
#: sorting the stage names would produce.
#:
#: Row 0 [1, 3, 2]  -> "10_Dense" -> [3, 2, 1]
#:                  -> "2_Dense"  -> [3, 4, 1] + [1, 1, 0] = [4, 5, 1]
#: Row 1 [3, -1, 1] -> "10_Dense" -> [-1, 1, 3]
#:                  -> "2_Dense"  -> [-1, 2, 3] + [1, 1, 0] = [0, 3, 3]
REVERSED_BY_HAND: npt.NDArray[np.float32] = np.array(
    [[4.0, 5.0, 1.0], [0.0, 3.0, 3.0]], dtype=np.float32
)


def unit(row: list[float]) -> list[float]:
    """L2-normalize one row with plain Python arithmetic.

    Deliberately not NumPy and deliberately not the module under test: the
    reference has to be computed by a route that shares no code with the thing
    it is checking, or a shared mistake survives it (Implementation Note 5.1).
    """
    length = math.sqrt(sum(value * value for value in row))
    return [value / length for value in row]


#: finalize(TOKEN_EMBEDDINGS, ATTENTION_MASK, dense=pipeline_stages()),
#: computed end to end by hand: pool, then both projections, then normalize.
FINALIZED_BY_HAND: list[list[float]] = [
    unit([7.0, 2.0, 2.0]),
    unit([-1.0, 1.0, 4.0]),
]


# --------------------------------------------------------------------------
# Non-vacuity: the fixture can actually tell the mistakes apart
#
# Task 4.2's review established the habit. Two earlier tasks shipped assertions
# that were true for the wrong reason - an all-ones mask, alphabetical stage
# names - so the fixture's discriminating power is asserted rather than assumed.
# --------------------------------------------------------------------------


def test_the_mask_fixture_would_expose_a_mask_blind_pool() -> None:
    assert not np.all(ATTENTION_MASK == 1), "an all-ones mask proves nothing"

    counts = ATTENTION_MASK.sum(axis=1)
    assert counts.tolist() == [4, 2], (
        "the two rows must attend different numbers of positions, or dividing "
        "by a single count would look correct"
    )

    over_everything = TOKEN_EMBEDDINGS.mean(axis=1)
    assert not np.allclose(over_everything, POOLED_BY_HAND), (
        "pooling over all positions must give a visibly different answer"
    )

    by_sequence_length = np.array(
        [
            TOKEN_EMBEDDINGS[row][ATTENTION_MASK[row] == 1].sum(axis=0)
            / ATTENTION_MASK.shape[1]
            for row in range(TOKEN_EMBEDDINGS.shape[0])
        ]
    )
    assert not np.allclose(by_sequence_length, POOLED_BY_HAND), (
        "dividing by the padded length instead of the attended count must give "
        "a visibly different answer"
    )


def test_the_dense_fixture_would_expose_a_sorted_order() -> None:
    recorded = [FIRST_STAGE_NAME, SECOND_STAGE_NAME]

    assert sorted(recorded) != recorded, (
        "the stage names must not be alphabetical, or sorting them changes "
        "nothing and the ordering assertions are vacuous"
    )
    assert sorted(recorded) == list(reversed(recorded)), (
        "sorting these names must actually reverse the pipeline"
    )
    assert not np.allclose(DENSED_BY_HAND, REVERSED_BY_HAND), (
        "the two orders must produce different vectors"
    )
    assert FIRST_WEIGHT.shape == SECOND_WEIGHT.shape, (
        "both orders must be shape-valid, so a wrong order is caught by its "
        "numbers rather than by an exception it happens to raise"
    )


def test_the_dense_fixture_would_expose_a_transposed_weight() -> None:
    """A weight equal to its own transpose pins no orientation.

    `models/export.py` stores ``(out_features, in_features)``, the orientation
    sentence-transformers writes, and `apply_dense_stages` consumes it as
    ``x @ weight.T``. With a symmetric weight both readings agree, so the
    convention would be enforced by a docstring and by nothing else - and a
    transposed read of a real 768x3072 matrix is a shape error only by luck.
    """
    assert not np.array_equal(SECOND_WEIGHT, SECOND_WEIGHT.T), (
        "a symmetric weight cannot tell x @ W from x @ W.T"
    )
    assert not np.allclose(
        POOLED_BY_HAND @ SECOND_WEIGHT.T, POOLED_BY_HAND @ SECOND_WEIGHT
    ), "the two orientations must produce different vectors"
    assert SECOND_WEIGHT.shape[0] == SECOND_WEIGHT.shape[1], (
        "the weight must be square, so a transposed read is caught by its "
        "numbers rather than by a shape error it happens to raise"
    )


def test_the_dense_fixture_would_expose_a_bias_added_too_early() -> None:
    """A bias supported only where its weight scales by 1.0 pins no ordering.

    ``W x + b`` and ``W (x + b)`` are the same vector whenever ``W b == b``,
    which was true of the first version of this fixture: the bias sat in the one
    component ``diag(1, 2, 1)`` leaves alone. A stage that added its bias before
    the projection would then have been invisible in every assertion here,
    including the exporter round-trip.
    """
    assert not np.allclose(
        (POOLED_BY_HAND + FIRST_BIAS) @ FIRST_WEIGHT.T,
        POOLED_BY_HAND @ FIRST_WEIGHT.T + FIRST_BIAS,
    ), "adding the bias before the projection must change the answer"
    assert not np.allclose(FIRST_WEIGHT @ FIRST_BIAS, FIRST_BIAS), (
        "the bias must not be a fixed point of its own weight, or the two "
        "orderings agree by construction"
    )


def test_the_dense_fixture_would_expose_a_skipped_stage() -> None:
    assert not np.allclose(DENSED_BY_HAND, POOLED_BY_HAND), (
        "skipping the projections entirely must give a different answer"
    )
    assert not np.allclose(
        np.array(FINALIZED_BY_HAND, dtype=np.float32),
        np.array([unit([1.0, 3.0, 2.0]), unit([3.0, -1.0, 1.0])]),
    ), "normalizing the pooled vector without the projections must differ"


# --------------------------------------------------------------------------
# Masked mean pooling (3.5, the Observable's first limb)
# --------------------------------------------------------------------------


def test_pooling_a_padded_input_matches_the_hand_computed_reference() -> None:
    pooled = masked_mean_pool(TOKEN_EMBEDDINGS, ATTENTION_MASK)

    np.testing.assert_array_equal(pooled, POOLED_BY_HAND)


def test_padding_contributes_nothing_whatever_it_holds() -> None:
    """A padded position is not merely down-weighted, it is not read.

    The padding here is filled with values arithmetic cannot survive. A pool
    that multiplies by the mask and sums propagates them; a pool that selects
    the attended positions does not. design.md calls the padding-mask
    interaction "where mean pooling silently breaks", and under a fixed 512 this
    is the only difference between the two implementations.
    """
    poisoned = TOKEN_EMBEDDINGS.copy()
    poisoned[0, 4, :] = np.float32("nan")
    poisoned[0, 5, :] = np.float32("inf")
    poisoned[1, 2:, :] = np.float32("-inf")

    pooled = masked_mean_pool(poisoned, ATTENTION_MASK)

    np.testing.assert_array_equal(pooled, POOLED_BY_HAND)


def test_pooling_is_unaffected_by_what_other_rows_hold() -> None:
    """Rows are independent, so one row's content cannot move another's vector."""
    alone = masked_mean_pool(
        TOKEN_EMBEDDINGS[:1].copy(), ATTENTION_MASK[:1].copy()
    )

    np.testing.assert_array_equal(alone[0], POOLED_BY_HAND[0])


def test_pooling_preserves_row_order() -> None:
    order = [1, 0]
    swapped = masked_mean_pool(
        TOKEN_EMBEDDINGS[order].copy(), ATTENTION_MASK[order].copy()
    )

    np.testing.assert_array_equal(swapped, POOLED_BY_HAND[order])


def test_pooling_returns_float32_of_the_trunk_width() -> None:
    pooled = masked_mean_pool(TOKEN_EMBEDDINGS, ATTENTION_MASK)

    assert pooled.dtype == np.float32
    assert pooled.shape == (
        TOKEN_EMBEDDINGS.shape[0],
        TOKEN_EMBEDDINGS.shape[2],
    )


def test_pooling_does_not_modify_its_arguments() -> None:
    tokens = TOKEN_EMBEDDINGS.copy()
    mask = ATTENTION_MASK.copy()

    masked_mean_pool(tokens, mask)

    np.testing.assert_array_equal(tokens, TOKEN_EMBEDDINGS)
    np.testing.assert_array_equal(mask, ATTENTION_MASK)


def test_a_row_attending_nothing_is_refused_rather_than_divided_by_zero() -> None:
    mask = ATTENTION_MASK.copy()
    mask[1, :] = 0

    with pytest.raises(ExecutionError) as caught:
        masked_mean_pool(TOKEN_EMBEDDINGS, mask)

    assert "[1]" in str(caught.value), (
        "the report must name the offending row, and a bare '1' appears in "
        "most of these messages by accident"
    )
    assert caught.value.stage == POSTPROCESS_STAGE


def test_a_mask_carrying_values_other_than_zero_and_one_is_refused() -> None:
    mask = ATTENTION_MASK.copy()
    mask[0, 0] = 2

    with pytest.raises(ExecutionError):
        masked_mean_pool(TOKEN_EMBEDDINGS, mask)


@pytest.mark.parametrize(
    ("tokens", "mask"),
    [
        (np.zeros((2, 6), dtype=np.float32), ATTENTION_MASK),
        (TOKEN_EMBEDDINGS, np.zeros((2, 6, 1), dtype=np.int64)),
        (TOKEN_EMBEDDINGS, np.ones((3, 6), dtype=np.int64)),
        (TOKEN_EMBEDDINGS, np.ones((2, 5), dtype=np.int64)),
    ],
)
def test_a_shape_that_cannot_be_pooled_is_refused(
    tokens: npt.NDArray[Any], mask: npt.NDArray[Any]
) -> None:
    with pytest.raises(ExecutionError):
        masked_mean_pool(tokens, mask)


# --------------------------------------------------------------------------
# The Dense stage (3.5; Implementation Note 3.2)
# --------------------------------------------------------------------------


def test_the_dense_stages_are_applied_in_the_order_they_are_given() -> None:
    projected = apply_dense_stages(POOLED_BY_HAND, pipeline_stages())

    np.testing.assert_array_equal(projected, DENSED_BY_HAND)


def test_reversing_the_dense_order_changes_the_answer() -> None:
    """The assertion the whole ordering contract rests on.

    If this passed for both orders, nothing downstream could tell 768->3072->768
    from 768<-3072<-768, which Implementation Note 3.2 records as right shape,
    right norm, wrong meaning.
    """
    reversed_stages = tuple(reversed(pipeline_stages()))

    projected = apply_dense_stages(POOLED_BY_HAND, reversed_stages)

    np.testing.assert_array_equal(projected, REVERSED_BY_HAND)
    assert not np.allclose(projected, DENSED_BY_HAND)


def test_no_dense_stages_leaves_the_pooled_vector_untouched() -> None:
    projected = apply_dense_stages(POOLED_BY_HAND, ())

    np.testing.assert_array_equal(projected, POOLED_BY_HAND)


def test_the_dense_stage_never_hands_back_the_array_it_was_given() -> None:
    """With no stages the arithmetic is a no-op, and the cheapest no-op is to
    return the argument - which would make the caller's later write to its own
    array reach into a vector it had already been handed."""
    pooled = POOLED_BY_HAND.copy()

    projected = apply_dense_stages(pooled, ())
    pooled[:] = 0.0

    assert projected is not pooled
    np.testing.assert_array_equal(projected, POOLED_BY_HAND)


def test_a_bias_is_added_and_a_missing_bias_is_not_invented() -> None:
    with_bias = apply_dense_stages(
        POOLED_BY_HAND,
        (
            DenseLayer(
                name="only",
                weight=FIRST_WEIGHT,
                bias=FIRST_BIAS,
                activation=IDENTITY_ACTIVATION,
            ),
        ),
    )
    without_bias = apply_dense_stages(
        POOLED_BY_HAND,
        (
            DenseLayer(
                name="only",
                weight=FIRST_WEIGHT,
                bias=None,
                activation=IDENTITY_ACTIVATION,
            ),
        ),
    )

    # diag(1, 2, 1) applied to [1, 3, 2] gives [1, 6, 2] and to [3, -1, 1]
    # gives [3, -2, 1]; the bias [1, 1, 0] is added to those *products*, not to
    # the inputs. Adding it first would give diag(1, 2, 1) @ [2, 4, 2] =
    # [2, 8, 2] for row 0, which these literals distinguish.
    np.testing.assert_array_equal(
        with_bias, np.array([[2.0, 7.0, 2.0], [4.0, -1.0, 1.0]], np.float32)
    )
    np.testing.assert_array_equal(
        without_bias, np.array([[1.0, 6.0, 2.0], [3.0, -2.0, 1.0]], np.float32)
    )


def test_a_projection_that_does_not_chain_is_refused() -> None:
    widening = DenseLayer(
        name="wide",
        weight=np.ones((5, 3), dtype=np.float32),
        bias=None,
        activation=IDENTITY_ACTIVATION,
    )
    same_again = DenseLayer(
        name="again",
        weight=np.ones((5, 3), dtype=np.float32),
        bias=None,
        activation=IDENTITY_ACTIVATION,
    )

    with pytest.raises(ExecutionError):
        apply_dense_stages(POOLED_BY_HAND, (widening, same_again))


def test_a_projection_whose_width_does_not_match_the_trunk_is_refused() -> None:
    mismatched = DenseLayer(
        name="foreign",
        weight=np.ones((3, 7), dtype=np.float32),
        bias=None,
        activation=IDENTITY_ACTIVATION,
    )

    with pytest.raises(ExecutionError):
        apply_dense_stages(POOLED_BY_HAND, (mismatched,))


def test_a_tanh_activation_is_applied_rather_than_ignored() -> None:
    stage = DenseLayer(
        name="tanh",
        weight=np.eye(3, dtype=np.float32),
        bias=None,
        activation="torch.nn.modules.activation.Tanh",
    )

    projected = apply_dense_stages(POOLED_BY_HAND, (stage,))

    expected = [
        [math.tanh(value) for value in row] for row in POOLED_BY_HAND.tolist()
    ]
    np.testing.assert_allclose(projected, expected, rtol=1e-6, atol=1e-7)
    assert not np.allclose(projected, POOLED_BY_HAND), (
        "tanh must actually change these values, or the assertion is vacuous"
    )


def test_a_relu_activation_is_applied_rather_than_ignored() -> None:
    stage = DenseLayer(
        name="relu",
        weight=np.eye(3, dtype=np.float32),
        bias=None,
        activation="torch.nn.modules.activation.ReLU",
    )

    projected = apply_dense_stages(POOLED_BY_HAND, (stage,))

    np.testing.assert_array_equal(
        projected, np.array([[1.0, 3.0, 2.0], [3.0, 0.0, 1.0]], np.float32)
    )


def test_an_unrecognised_activation_is_refused_not_treated_as_identity() -> None:
    """The failure Implementation Note 3.2 left open when it deferred activation
    fidelity to this task: assuming the identity for a stage that has some other
    activation is the same right-shape, right-norm, wrong-meaning defect."""
    stage = DenseLayer(
        name="gelu",
        weight=np.eye(3, dtype=np.float32),
        bias=None,
        activation="torch.nn.modules.activation.GELU",
    )

    with pytest.raises(ExecutionError) as caught:
        apply_dense_stages(POOLED_BY_HAND, (stage,))

    assert "GELU" in str(caught.value)


@pytest.mark.parametrize(
    "layer",
    [
        DenseLayer(
            name="float64 weight",
            weight=np.eye(3, dtype=np.float64),
            bias=None,
            activation=IDENTITY_ACTIVATION,
        ),
        DenseLayer(
            name="one-dimensional weight",
            weight=np.ones(3, dtype=np.float32),
            bias=None,
            activation=IDENTITY_ACTIVATION,
        ),
        DenseLayer(
            name="float64 bias",
            weight=np.eye(3, dtype=np.float32),
            bias=np.zeros(3, dtype=np.float64),
            activation=IDENTITY_ACTIVATION,
        ),
        DenseLayer(
            name="bias of the wrong length",
            weight=np.eye(3, dtype=np.float32),
            bias=np.zeros(4, dtype=np.float32),
            activation=IDENTITY_ACTIVATION,
        ),
    ],
)
def test_a_projection_that_is_not_what_it_claims_is_refused(
    layer: DenseLayer,
) -> None:
    """Each of these would otherwise surface as a NumPy broadcasting error or a
    silent float64 promotion, neither of which names the model or the stage
    requirement 8.1 asks for."""
    with pytest.raises(ExecutionError) as caught:
        apply_dense_stages(POOLED_BY_HAND, (layer,))

    assert caught.value.stage == POSTPROCESS_STAGE


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((2, 3, 1), dtype=np.float32),
        np.ones((2, 3), dtype=np.float64),
    ],
)
def test_a_matrix_that_cannot_be_projected_or_normalized_is_refused(
    vectors: npt.NDArray[Any],
) -> None:
    with pytest.raises(ExecutionError):
        apply_dense_stages(vectors, pipeline_stages())
    with pytest.raises(ExecutionError):
        l2_normalize(vectors)


def test_the_dense_stage_does_not_modify_its_arguments() -> None:
    pooled = POOLED_BY_HAND.copy()
    weight = FIRST_WEIGHT.copy()
    stage = DenseLayer(
        name="only", weight=weight, bias=None, activation=IDENTITY_ACTIVATION
    )

    apply_dense_stages(pooled, (stage,))

    np.testing.assert_array_equal(pooled, POOLED_BY_HAND)
    np.testing.assert_array_equal(weight, FIRST_WEIGHT)


# --------------------------------------------------------------------------
# L2 normalization (3.5, the Observable's second limb)
# --------------------------------------------------------------------------


def test_every_normalized_row_has_unit_length() -> None:
    normalized = l2_normalize(DENSED_BY_HAND)

    norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    np.testing.assert_allclose(norms, np.ones(norms.shape), rtol=0, atol=1e-6)


def test_normalization_matches_the_hand_computed_reference() -> None:
    normalized = l2_normalize(DENSED_BY_HAND)

    np.testing.assert_allclose(
        normalized,
        [unit([7.0, 2.0, 2.0]), unit([-1.0, 1.0, 4.0])],
        rtol=1e-6,
        atol=1e-7,
    )


def test_normalization_preserves_direction() -> None:
    normalized = l2_normalize(DENSED_BY_HAND)

    for original, scaled in zip(
        DENSED_BY_HAND.astype(np.float64), normalized.astype(np.float64)
    ):
        ratios = scaled[original != 0.0] / original[original != 0.0]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)
        assert ratios[0] > 0.0


def test_a_zero_vector_cannot_be_normalized_and_is_refused() -> None:
    vectors = DENSED_BY_HAND.copy()
    vectors[1, :] = 0.0

    with pytest.raises(ExecutionError) as caught:
        l2_normalize(vectors)

    assert caught.value.stage == POSTPROCESS_STAGE


def test_normalization_does_not_modify_its_argument() -> None:
    vectors = DENSED_BY_HAND.copy()

    l2_normalize(vectors)

    np.testing.assert_array_equal(vectors, DENSED_BY_HAND)


# --------------------------------------------------------------------------
# finalize: the whole path, in the one order that is correct
# --------------------------------------------------------------------------


def test_finalize_pools_then_projects_then_normalizes() -> None:
    vectors = finalize(
        TOKEN_EMBEDDINGS, ATTENTION_MASK, dense=pipeline_stages()
    )

    np.testing.assert_allclose(
        vectors, FINALIZED_BY_HAND, rtol=1e-6, atol=1e-7
    )


def test_finalize_normalizes_after_the_dense_stage_not_before() -> None:
    """Order matters beyond magnitude: the projections carry a bias, so
    normalizing first changes the *direction* the caller receives, not merely
    its length - and the result is still unit-norm, so no norm check sees it."""
    vectors = finalize(
        TOKEN_EMBEDDINGS, ATTENTION_MASK, dense=pipeline_stages()
    )

    normalize_first = l2_normalize(
        apply_dense_stages(l2_normalize(POOLED_BY_HAND), pipeline_stages())
    )

    assert not np.allclose(vectors, normalize_first), (
        "normalizing before the projections must be distinguishable, or this "
        "test cannot see the mistake it exists for"
    )


def test_finalize_without_a_dense_stage_still_returns_unit_vectors() -> None:
    vectors = finalize(TOKEN_EMBEDDINGS, ATTENTION_MASK)

    np.testing.assert_allclose(
        vectors,
        [unit([1.0, 3.0, 2.0]), unit([3.0, -1.0, 1.0])],
        rtol=1e-6,
        atol=1e-7,
    )


def test_finalize_returns_one_unit_row_per_input_row_in_order() -> None:
    """Requirement 3.1 and 3.5 together."""
    order = [1, 0]
    vectors = finalize(
        TOKEN_EMBEDDINGS[order].copy(),
        ATTENTION_MASK[order].copy(),
        dense=pipeline_stages(),
    )

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(
        vectors,
        [FINALIZED_BY_HAND[i] for i in order],
        rtol=1e-6,
        atol=1e-7,
    )
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    np.testing.assert_allclose(norms, np.ones(2), rtol=0, atol=1e-6)


def test_finalize_reports_the_dimension_the_projections_end_at() -> None:
    widening = DenseLayer(
        name="widen",
        weight=np.eye(5, 3, dtype=np.float32),
        bias=None,
        activation=IDENTITY_ACTIVATION,
    )

    vectors = finalize(TOKEN_EMBEDDINGS, ATTENTION_MASK, dense=(widening,))

    assert vectors.shape == (2, 5)


# --------------------------------------------------------------------------
# Determinism (3.10, the Observable's third limb)
# --------------------------------------------------------------------------


def test_the_same_input_twice_returns_bitwise_identical_vectors() -> None:
    first = finalize(TOKEN_EMBEDDINGS, ATTENTION_MASK, dense=pipeline_stages())
    second = finalize(TOKEN_EMBEDDINGS, ATTENTION_MASK, dense=pipeline_stages())

    assert first.tobytes() == second.tobytes()


def test_the_same_row_twice_in_one_batch_returns_identical_vectors() -> None:
    """3.10 within a single call, not merely across two: a row's vector must not
    depend on where in the batch it sits."""
    tokens = np.stack(
        [TOKEN_EMBEDDINGS[0], TOKEN_EMBEDDINGS[1], TOKEN_EMBEDDINGS[0]]
    )
    mask = np.stack([ATTENTION_MASK[0], ATTENTION_MASK[1], ATTENTION_MASK[0]])

    vectors = finalize(tokens, mask, dense=pipeline_stages())

    assert vectors[0].tobytes() == vectors[2].tobytes()


def test_finalize_does_not_modify_its_arguments() -> None:
    tokens = TOKEN_EMBEDDINGS.copy()
    mask = ATTENTION_MASK.copy()

    finalize(tokens, mask, dense=pipeline_stages())

    np.testing.assert_array_equal(tokens, TOKEN_EMBEDDINGS)
    np.testing.assert_array_equal(mask, ATTENTION_MASK)


def test_a_returned_vector_is_not_a_view_onto_the_caller_s_array() -> None:
    tokens = TOKEN_EMBEDDINGS.copy()
    vectors = finalize(tokens, ATTENTION_MASK)

    tokens[:] = 0.0

    np.testing.assert_allclose(
        vectors,
        [unit([1.0, 3.0, 2.0]), unit([3.0, -1.0, 1.0])],
        rtol=1e-6,
        atol=1e-7,
    )


# --------------------------------------------------------------------------
# Failures carry the three facts requirement 8.1 demands
# --------------------------------------------------------------------------


def test_a_postprocessing_failure_names_provider_model_and_stage() -> None:
    mask = ATTENTION_MASK.copy()
    mask[0, :] = 0

    with pytest.raises(ExecutionError) as caught:
        finalize(
            TOKEN_EMBEDDINGS,
            mask,
            provider=ProviderChoice.NPU,
            model_id="google/embeddinggemma-300m",
        )

    assert caught.value.provider is ProviderChoice.NPU
    assert caught.value.model_id == "google/embeddinggemma-300m"
    assert caught.value.stage == POSTPROCESS_STAGE


def test_postprocessing_failures_are_execution_failures_not_preparation() -> None:
    """Requirement 8.2 is a type question (errors.py). Post-processing runs
    inside an embedding call, so its failures belong to the execution
    category."""
    mask = ATTENTION_MASK.copy()
    mask[0, :] = 0

    with pytest.raises(EmbeddingRuntimeError) as caught:
        finalize(TOKEN_EMBEDDINGS, mask)

    assert isinstance(caught.value, ExecutionError)


# --------------------------------------------------------------------------
# Reading the recorded pipeline order out of a loaded dense.npz
#
# Implementation Note 3.2: "Task 5.2 must consume `order` and never re-derive it
# by sorting the weight names."
# --------------------------------------------------------------------------

ORDER_KEY = "order"
WEIGHT_PREFIX = "weight__"
BIAS_PREFIX = "bias__"
ACTIVATION_PREFIX = "activation__"


def loaded_arrays() -> dict[str, npt.NDArray[Any]]:
    """A dense.npz-shaped mapping whose *insertion* order is the wrong one.

    Both the mapping's iteration order and the sorted key order disagree with
    the recorded pipeline order, so only a reader that consumes ``order`` gets
    the right answer.
    """
    return {
        ACTIVATION_PREFIX + SECOND_STAGE_NAME: np.array(IDENTITY_ACTIVATION),
        WEIGHT_PREFIX + SECOND_STAGE_NAME: SECOND_WEIGHT,
        ORDER_KEY: np.array([FIRST_STAGE_NAME, SECOND_STAGE_NAME]),
        WEIGHT_PREFIX + FIRST_STAGE_NAME: FIRST_WEIGHT,
        BIAS_PREFIX + FIRST_STAGE_NAME: FIRST_BIAS,
        ACTIVATION_PREFIX + FIRST_STAGE_NAME: np.array(IDENTITY_ACTIVATION),
    }


def read_back(arrays: dict[str, npt.NDArray[Any]]) -> tuple[DenseLayer, ...]:
    return dense_layers_from_arrays(
        arrays,
        order_key=ORDER_KEY,
        weight_prefix=WEIGHT_PREFIX,
        bias_prefix=BIAS_PREFIX,
        activation_prefix=ACTIVATION_PREFIX,
    )


def test_the_recorded_order_is_used_and_not_the_mapping_s_own() -> None:
    arrays = loaded_arrays()

    assert [
        key for key in arrays if key.startswith(WEIGHT_PREFIX)
    ] != [WEIGHT_PREFIX + FIRST_STAGE_NAME, WEIGHT_PREFIX + SECOND_STAGE_NAME], (
        "the fixture must not present the weights in pipeline order, or "
        "iterating the mapping would look correct"
    )

    layers = read_back(arrays)

    assert [layer.name for layer in layers] == [
        FIRST_STAGE_NAME,
        SECOND_STAGE_NAME,
    ]


def test_layers_read_from_a_mapping_reproduce_the_hand_computed_pipeline() -> None:
    layers = read_back(loaded_arrays())

    np.testing.assert_array_equal(
        apply_dense_stages(POOLED_BY_HAND, layers), DENSED_BY_HAND
    )


def test_sorting_the_weight_names_would_have_given_the_wrong_answer() -> None:
    """Non-vacuity for the ordering contract: prove the sorted route is wrong
    here, so the assertion above is distinguishing rather than lucky."""
    arrays = loaded_arrays()
    sorted_names = sorted(
        key[len(WEIGHT_PREFIX) :]
        for key in arrays
        if key.startswith(WEIGHT_PREFIX)
    )

    assert sorted_names == [SECOND_STAGE_NAME, FIRST_STAGE_NAME]

    by_sorting = tuple(
        DenseLayer(
            name=name,
            weight=arrays[WEIGHT_PREFIX + name].astype(np.float32),
            bias=(
                None
                if BIAS_PREFIX + name not in arrays
                else arrays[BIAS_PREFIX + name].astype(np.float32)
            ),
            activation=IDENTITY_ACTIVATION,
        )
        for name in sorted_names
    )
    np.testing.assert_array_equal(
        apply_dense_stages(POOLED_BY_HAND, by_sorting), REVERSED_BY_HAND
    )


def test_a_stage_named_in_the_order_but_missing_its_weight_is_refused() -> None:
    arrays = loaded_arrays()
    del arrays[WEIGHT_PREFIX + SECOND_STAGE_NAME]

    with pytest.raises(ExecutionError) as caught:
        read_back(arrays)

    assert SECOND_STAGE_NAME in str(caught.value)


def test_a_weight_absent_from_the_recorded_order_is_refused() -> None:
    """Silently dropping a projection nobody listed is the same failure as
    skipping one that is listed."""
    arrays = loaded_arrays()
    arrays[WEIGHT_PREFIX + "99_Dense"] = np.eye(3, dtype=np.float32)

    with pytest.raises(ExecutionError) as caught:
        read_back(arrays)

    assert "99_Dense" in str(caught.value)


def test_a_mapping_with_no_recorded_order_is_refused() -> None:
    arrays = loaded_arrays()
    del arrays[ORDER_KEY]

    with pytest.raises(ExecutionError) as caught:
        read_back(arrays)

    assert ORDER_KEY in str(caught.value)


def test_a_missing_activation_is_refused_rather_than_assumed() -> None:
    arrays = loaded_arrays()
    del arrays[ACTIVATION_PREFIX + FIRST_STAGE_NAME]

    with pytest.raises(ExecutionError):
        read_back(arrays)


def test_an_activation_entry_that_is_not_one_name_is_refused() -> None:
    arrays = loaded_arrays()
    arrays[ACTIVATION_PREFIX + FIRST_STAGE_NAME] = np.array(
        [IDENTITY_ACTIVATION, IDENTITY_ACTIVATION]
    )

    with pytest.raises(ExecutionError) as caught:
        read_back(arrays)

    assert FIRST_STAGE_NAME in str(caught.value)


def test_an_absent_bias_is_read_as_absent() -> None:
    layers = read_back(loaded_arrays())

    assert layers[0].bias is not None
    assert layers[1].bias is None


def test_an_empty_mapping_reads_as_no_stages() -> None:
    layers = read_back({ORDER_KEY: np.array([], dtype="<U1")})

    assert layers == ()


def test_the_reader_accepts_what_the_exporter_actually_writes(
    tmp_path: Path,
) -> None:
    """The one place this task touches ``models/export.py``: through its file,
    not through an import in the source. The prefixes and the order key are the
    exporter's constants, passed in by the caller, so a rename there is a loud
    failure here rather than a silent divergence."""
    from npu_rag.embedding.models.export import (
        ACTIVATION_PREFIX as EXPORT_ACTIVATION_PREFIX,
    )
    from npu_rag.embedding.models.export import (
        BIAS_PREFIX as EXPORT_BIAS_PREFIX,
    )
    from npu_rag.embedding.models.export import (
        DENSE_ORDER_KEY,
        DenseStage,
        _write_dense,
    )
    from npu_rag.embedding.models.export import (
        WEIGHT_PREFIX as EXPORT_WEIGHT_PREFIX,
    )

    stages = (
        DenseStage(
            name=FIRST_STAGE_NAME,
            weight=FIRST_WEIGHT,
            bias=FIRST_BIAS,
            activation=IDENTITY_ACTIVATION,
        ),
        DenseStage(
            name=SECOND_STAGE_NAME,
            weight=SECOND_WEIGHT,
            bias=None,
            activation=IDENTITY_ACTIVATION,
        ),
    )
    written = tmp_path / "dense.npz"
    _write_dense(stages, written)

    with np.load(written) as handle:
        arrays = {key: handle[key] for key in handle.files}

    layers = dense_layers_from_arrays(
        arrays,
        order_key=DENSE_ORDER_KEY,
        weight_prefix=EXPORT_WEIGHT_PREFIX,
        bias_prefix=EXPORT_BIAS_PREFIX,
        activation_prefix=EXPORT_ACTIVATION_PREFIX,
    )

    assert [layer.name for layer in layers] == [
        FIRST_STAGE_NAME,
        SECOND_STAGE_NAME,
    ]
    np.testing.assert_array_equal(
        apply_dense_stages(POOLED_BY_HAND, layers), DENSED_BY_HAND
    )


# --------------------------------------------------------------------------
# Backend independence and dependency direction (design.md, Architecture)
# --------------------------------------------------------------------------

MODULE_PACKAGE = "npu_rag.embedding"
MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "npu_rag"
    / "embedding"
    / "postprocess.py"
)


def module_source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def imported_names(source: str, containing: str) -> list[str]:
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


@pytest.mark.parametrize(
    "later",
    ["reporting", "profiles", "environment", "models", "providers", "tokenize",
     "service", "bench"],
)
def test_postprocess_imports_nothing_from_a_later_layer(later: str) -> None:
    """``postprocess`` sits directly above ``types``/``errors``: it needs the
    error taxonomy and the provider vocabulary for its own failure reports, and
    nothing else in this package. Placing it that low is what makes the
    package-wide guard forbid the most."""
    names = imported_names(module_source(), MODULE_PACKAGE)

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
        "from npu_rag.embedding.models import export",
        "from .providers import base",
        "from .profiles import profile_for",
    ],
)
def test_the_layer_guard_would_catch_an_inversion(statement: str) -> None:
    resolved = imported_names(statement, MODULE_PACKAGE)

    assert any(
        name.startswith(f"{MODULE_PACKAGE}.{layer}")
        for name in resolved
        for layer in ("providers", "models", "profiles")
    ), f"{statement!r} resolved to {resolved}"


def test_the_module_mentions_no_backend_session_or_runtime() -> None:
    """Backend independence, design.md line 120: this path executes identically
    whichever adapter produced the token embeddings, which it can only do if it
    has never heard of one."""
    text = module_source().lower()

    for banned in (
        "onnxruntime",
        "inferencesession",
        "vitisai",
        "execution provider",
        "get_providers",
    ):
        assert banned not in text, f"postprocess.py mentions {banned!r}"


def test_the_module_neither_prints_nor_logs() -> None:
    text = module_source()

    assert [
        name
        for name in imported_names(text, MODULE_PACKAGE)
        for banned in ("logging", "sys", "warnings")
        if name == banned or name.startswith(f"{banned}.")
    ] == []

    called = {
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
    }
    assert [
        name
        for name in called
        if name == "print" or name.startswith(("logging.", "warnings.", "sys.std"))
    ] == []


def test_the_module_carries_no_unfinished_work() -> None:
    text = module_source().upper()

    for marker in ("TODO", "FIXME", "TBD", "XXX"):
        assert marker not in text
