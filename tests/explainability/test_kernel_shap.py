import itertools
from math import factorial

import numpy as np
import pytest

from src.explainability.kernel_shap import build_background, explain_with_kernel_shap

_SEGMENT_VALUES = {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.05}
_INTERACTION_BONUS = 0.25


def _four_block_segments() -> np.ndarray:
    segments = np.zeros((8, 8), dtype=np.int64)
    segments[:4, 4:] = 1
    segments[4:, :4] = 2
    segments[4:, 4:] = 3
    return segments


def _four_block_image() -> np.ndarray:
    segments = _four_block_segments()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for segment_id in range(4):
        image[segments == segment_id] = (segment_id + 1) * 50
    return image


def _set_value(visible: frozenset) -> float:
    """Funcion de coalicion con una interaccion, para que los valores no sean triviales."""
    total = sum(_SEGMENT_VALUES[segment_id] for segment_id in visible)
    if {0, 1} <= visible:
        total += _INTERACTION_BONUS
    return total


def _predict_fn(batch: np.ndarray) -> np.ndarray:
    """Deduce que superpixeles quedaron visibles por el color de cada bloque."""
    segments = _four_block_segments()
    scores = np.array(
        [
            _set_value(
                frozenset(
                    segment_id for segment_id in range(4) if image[segments == segment_id].max() > 0
                )
            )
            for image in batch
        ]
    )
    return np.stack([scores, 1.0 - scores], axis=1)


def _exact_shapley() -> np.ndarray:
    players = list(range(4))
    phi = np.zeros(4)
    for player in players:
        others = [other for other in players if other != player]
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                weight = factorial(size) * factorial(3 - size) / factorial(4)
                phi[player] += weight * (
                    _set_value(frozenset(subset) | {player}) - _set_value(frozenset(subset))
                )
    return phi


def _explain(**overrides):
    kwargs = {
        "image_np": _four_block_image(),
        "segments": _four_block_segments(),
        "predict_fn": _predict_fn,
        "target_idx": 0,
        "nsamples": 64,
        "batch_size": 8,
    }
    kwargs.update(overrides)
    return explain_with_kernel_shap(**kwargs)


def test_matches_exact_shapley_values():
    explanation = _explain()

    np.testing.assert_allclose(explanation.values, _exact_shapley(), atol=1e-6)


def test_is_additive():
    explanation = _explain()

    total = explanation.values.sum() + explanation.expected_value

    assert total == pytest.approx(_set_value(frozenset(range(4))), abs=1e-6)


def test_expected_value_is_the_fully_masked_prediction():
    explanation = _explain()

    assert explanation.expected_value == pytest.approx(_set_value(frozenset()), abs=1e-9)


def test_is_deterministic_when_coalitions_are_sampled():
    first = _explain(nsamples=10)
    second = _explain(nsamples=10)

    np.testing.assert_array_equal(first.values, second.values)


def test_does_not_leak_the_global_random_state():
    np.random.seed(1234)
    expected = np.random.random()

    np.random.seed(1234)
    _explain(nsamples=10)
    actual = np.random.random()

    assert actual == expected


def test_black_background_is_all_zeros():
    background = build_background(_four_block_image(), "black")

    assert not background.any()


def test_mean_background_is_constant_per_channel():
    background = build_background(_four_block_image(), "mean")

    assert len(np.unique(background.reshape(-1, 3), axis=0)) == 1


def test_rejects_unknown_background():
    with pytest.raises(ValueError, match="desconocida"):
        build_background(_four_block_image(), "inpaint")
