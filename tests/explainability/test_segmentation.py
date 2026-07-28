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


def _structured_image() -> np.ndarray:
    """Imagen con cuatro cuadrantes de color solido y contiguo.

    Ruido uniforme (usado en `_synthetic_image`) no tiene regiones coherentes de color, y
    quickshift puede colapsar ese tipo de imagen a un unico segmento, lo que volveria
    vacua cualquier asercion de "mas de un segmento" (ver el mismo problema documentado en
    `tests/explainability/test_compare_report.py` para SLIC). Cuatro bloques bien
    diferenciados son justo lo que quickshift esta disenado para separar; se verifico
    empiricamente que produce 4 segmentos sobre esta imagen.
    """
    size = 64
    half = size // 2
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:half, :half] = (220, 40, 40)
    image[:half, half:] = (40, 200, 60)
    image[half:, :half] = (40, 80, 220)
    image[half:, half:] = (230, 200, 40)
    return image


def test_quickshift_is_supported():
    image = _structured_image()
    segments = build_segments(image, algorithm="quickshift")

    assert segments.shape == image.shape[:2]
    assert int(segments.max()) + 1 > 1


def test_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="desconocido"):
        build_segments(_synthetic_image(), algorithm="felzenszwalb")
