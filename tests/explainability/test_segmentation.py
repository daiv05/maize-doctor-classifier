import numpy as np
import pytest

from src.explainability.segmentation import build_segments


def _synthetic_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_labels_are_consecutive_from_zero():
    segments = build_segments(_synthetic_image(), n_segments=16)

    unique = np.unique(segments)
    np.testing.assert_array_equal(unique, np.arange(unique.size))


def test_is_deterministic_across_calls():
    image = _synthetic_image()

    np.testing.assert_array_equal(
        build_segments(image, n_segments=16), build_segments(image, n_segments=16)
    )


def test_shape_matches_the_image():
    image = _synthetic_image()

    assert build_segments(image, n_segments=16).shape == image.shape[:2]


def test_quickshift_is_supported():
    segments = build_segments(_synthetic_image(), algorithm="quickshift")

    assert segments.min() == 0


def test_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="desconocido"):
        build_segments(_synthetic_image(), algorithm="felzenszwalb")
