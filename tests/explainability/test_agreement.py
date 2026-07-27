import numpy as np
import pytest

from src.explainability.agreement import (
    attribution_agreement,
    densify_weights,
    top_positive_mask,
)


def test_densify_places_weights_at_their_segment_index():
    dense = densify_weights([(2, 0.5), (0, -0.25)], n_segments=4)

    np.testing.assert_allclose(dense, [-0.25, 0.0, 0.5, 0.0])


def test_top_positive_mask_ignores_negative_values():
    mask = top_positive_mask(np.array([0.9, -5.0, 0.1, 0.4]), top_k=2)

    np.testing.assert_array_equal(mask, [True, False, False, True])


def test_top_positive_mask_with_no_positive_values_is_empty():
    mask = top_positive_mask(np.array([-1.0, -2.0]), top_k=2)

    assert not mask.any()


def test_identical_vectors_agree_completely():
    values = np.array([0.5, -0.2, 0.9, 0.1])

    agreement = attribution_agreement(values, values, top_k=2)

    assert agreement["iou_topk"] == 1.0
    assert agreement["spearman"] == pytest.approx(1.0)
    assert agreement["sign_agreement"] == 1.0


def test_opposite_vectors_disagree_completely():
    values = np.array([0.5, -0.2, 0.9, 0.1])

    agreement = attribution_agreement(values, -values, top_k=2)

    assert agreement["spearman"] == pytest.approx(-1.0)
    assert agreement["sign_agreement"] == 0.0


def test_constant_vector_yields_zero_correlation():
    agreement = attribution_agreement(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]), top_k=2)

    assert agreement["spearman"] == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="longitud distinta"):
        attribution_agreement(np.zeros(3), np.zeros(4), top_k=2)
