import numpy as np

from src.explainability.leaf_mask import is_coverage_degenerate, leaf_mask, mask_coverage


def _half_leaf_image() -> np.ndarray:
    """Mitad izquierda verde (hoja), mitad derecha gris parduzco (suelo)."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (40, 160, 40)
    image[:, 16:] = (140, 120, 100)
    return image


def test_detects_the_green_half():
    mask = leaf_mask(_half_leaf_image())

    assert mask[:, :16].all()
    assert not mask[:, 16:].any()


def test_coverage_matches_the_green_fraction():
    coverage = mask_coverage(leaf_mask(_half_leaf_image()))

    assert coverage == 0.5


def test_uniform_image_yields_degenerate_coverage():
    uniform = np.full((32, 32, 3), 128, dtype=np.uint8)

    coverage = mask_coverage(leaf_mask(uniform))

    assert is_coverage_degenerate(coverage)


def test_healthy_coverage_is_not_degenerate():
    assert not is_coverage_degenerate(0.5)
    assert is_coverage_degenerate(0.01)
    assert is_coverage_degenerate(0.99)
