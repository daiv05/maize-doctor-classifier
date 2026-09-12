"""Mascara de vegetacion para separar atribucion sobre hoja de atribucion sobre fondo."""

import cv2
import numpy as np

_COVERAGE_LOW = 0.05
_COVERAGE_HIGH = 0.95
_EXG_THRESHOLD = 0.10
_FRAGMENTATION_HIGH = 0.60
_OPENING_KERNEL = (5, 5)
_CLOSING_KERNEL = (15, 15)
_EPSILON = 1e-6


def leaf_mask(image_np: np.ndarray, threshold: float = _EXG_THRESHOLD) -> np.ndarray:
    """
    Segmenta hoja contra fondo con exceso de verde (ExG) cromatico y umbral absoluto.

    El ExG se calcula sobre canales normalizados por la suma RGB, lo que lo hace robusto a
    cambios de iluminacion, y se corta con un umbral fijo. No usa Otsu: Otsu elige el corte
    a partir del histograma de cada imagen, asi que siempre parte la imagen en dos aunque no
    haya vegetacion (sobre ruido puro devolvia 50% de cobertura).

    Sigue siendo una heuristica: sobre hojas cloroticas puede degradarse. Validar con
    `mask_coverage` / `mask_fragmentation` e `is_mask_unreliable` antes de derivar metricas.

    @param {np.ndarray} image_np Imagen HWC uint8.
    @param {float} threshold Corte absoluto sobre el ExG cromatico.
    @returns {np.ndarray} Mascara booleana HW; True donde hay vegetacion.
    """
    channels = image_np.astype(np.float32) / 255.0
    total = channels.sum(axis=2) + _EPSILON
    red, green, blue = (channels[..., index] / total for index in range(3))
    binary = ((2.0 * green - red - blue) > threshold).astype(np.uint8)

    # Apertura: borra pixeles verdes sueltos sin comerse la hoja. Es lo que separa una
    # hoja real del ruido, que sobrevivia al umbral de color por pura probabilidad.
    opened = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _OPENING_KERNEL)
    )

    # Cierre: reincorpora las lesiones internas, que no son verdes y quedaban fuera. Sin
    # esto la mascara excluye el sintoma (medido: 19% del interior en common_rust) y el
    # ratio castiga al modelo justo por mirar donde debe.
    return cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _CLOSING_KERNEL)
    ).astype(bool)


def mask_fragmentation(mask: np.ndarray) -> float:
    """
    Fraccion de la mascara que no pertenece a su componente conexa mas grande.

    Una hoja real es un objeto compacto; pixeles verdes dispersos por ruido o textura de
    fondo producen muchas islas pequenas. Es la senal que la cobertura sola no captura.

    @param {np.ndarray} mask Mascara booleana.
    @returns {float} Fragmentacion en [0, 1]; 0 si la mascara esta vacia.
    """
    if not mask.any():
        return 0.0
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0.0
    largest = np.bincount(labels.ravel())[1:].max()
    return float(1.0 - largest / mask.sum())


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
    """Compatibilidad pública; NaN/infinito también invalida la cobertura."""
    return not np.isfinite(coverage) or coverage < low or coverage > high


def is_mask_unreliable(
    coverage: float,
    fragmentation: float,
    low: float = _COVERAGE_LOW,
    high: float = _COVERAGE_HIGH,
    max_fragmentation: float = _FRAGMENTATION_HIGH,
) -> bool:
    """
    Indica si la mascara no es utilizable para el ratio hoja/fondo.

    Rechaza por cobertura degenerada (casi todo o casi nada) y por fragmentacion alta, que
    delata pixeles verdes dispersos en vez de una hoja: la cobertura sola no distingue una
    hoja que ocupa medio cuadro de ruido repartido por todo el cuadro.

    @param {float} coverage Cobertura devuelta por `mask_coverage`.
    @param {float} fragmentation Valor devuelto por `mask_fragmentation`.
    @param {float} low Cota inferior de cobertura aceptable.
    @param {float} high Cota superior de cobertura aceptable.
    @param {float} max_fragmentation Fragmentacion maxima aceptable.
    @returns {bool} True si la mascara no es utilizable.
    """
    return (
        is_coverage_degenerate(coverage, low, high)
        or not np.isfinite(fragmentation)
        or fragmentation > max_fragmentation
    )
