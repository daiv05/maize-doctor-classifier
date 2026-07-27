"""Mascara de vegetacion para separar atribucion sobre hoja de atribucion sobre fondo."""

import cv2
import numpy as np

_COVERAGE_LOW = 0.05
_COVERAGE_HIGH = 0.95


def leaf_mask(image_np: np.ndarray) -> np.ndarray:
    """
    Segmenta hoja contra fondo con indice de exceso de verde (ExG) y umbral de Otsu.

    Es una heuristica: sobre hojas cloroticas (deficiencia de nitrogeno) puede degradarse.
    El consumidor debe validar el resultado con `mask_coverage` e `is_coverage_degenerate`
    antes de derivar metricas de el.

    @param {np.ndarray} image_np Imagen HWC uint8.
    @returns {np.ndarray} Mascara booleana HW; True donde hay vegetacion.
    """
    channels = image_np.astype(np.float32)
    excess_green = 2.0 * channels[..., 1] - channels[..., 0] - channels[..., 2]
    normalized = cv2.normalize(excess_green, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(bool)


def mask_coverage(mask: np.ndarray) -> float:
    """
    Fraccion de pixeles clasificados como vegetacion.

    @param {np.ndarray} mask Mascara booleana.
    @returns {float} Cobertura en [0, 1].
    """
    return float(mask.mean())


def is_coverage_degenerate(
    coverage: float, low: float = _COVERAGE_LOW, high: float = _COVERAGE_HIGH
) -> bool:
    """
    Indica si la cobertura delata una segmentacion inservible (casi todo o casi nada).

    @param {float} coverage Cobertura devuelta por `mask_coverage`.
    @param {float} low Cota inferior aceptable.
    @param {float} high Cota superior aceptable.
    @returns {bool} True si la mascara no es utilizable para el ratio hoja/fondo.
    """
    return coverage < low or coverage > high
