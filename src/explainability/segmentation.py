"""Segmentacion canonica compartida por LIME y KernelSHAP en el panel comparado."""

import numpy as np
from skimage.segmentation import quickshift, slic

_QUICKSHIFT_KERNEL_SIZE = 4
_QUICKSHIFT_MAX_DIST = 200
_QUICKSHIFT_RATIO = 0.2


def build_segments(
    image_np: np.ndarray,
    algorithm: str = "slic",
    n_segments: int = 50,
    compactness: float = 10.0,
) -> np.ndarray:
    """
    Calcula el mapa de superpixeles que comparten LIME y KernelSHAP.

    Ninguno de los dos algoritmos usa aleatoriedad, asi que el mapa es reproducible sin
    semilla. Las etiquetas se devuelven consecutivas desde 0 para poder indexar el vector
    de atribuciones directamente con `values[segments]`.

    Los parametros de quickshift replican los que `lime_image` usa por defecto, de modo
    que esa opcion reproduce las regiones del panel `visual`.

    @param {np.ndarray} image_np Imagen HWC uint8 ya reescalada a target_size.
    @param {str} algorithm "slic" o "quickshift".
    @param {int} n_segments Numero objetivo de superpixeles de SLIC; ignorado por quickshift.
    @param {float} compactness Peso del termino espacial de SLIC; ignorado por quickshift.
    @returns {np.ndarray} Mapa HW int64 con etiquetas consecutivas desde 0.
    @throws {ValueError} Si el algoritmo no es uno de los soportados.
    """
    if algorithm == "slic":
        segments = slic(image_np, n_segments=n_segments, compactness=compactness, start_label=0)
    elif algorithm == "quickshift":
        segments = quickshift(
            image_np,
            kernel_size=_QUICKSHIFT_KERNEL_SIZE,
            max_dist=_QUICKSHIFT_MAX_DIST,
            ratio=_QUICKSHIFT_RATIO,
        )
    else:
        raise ValueError(
            f"Algoritmo de segmentacion desconocido: {algorithm!r}. Usa 'slic' o 'quickshift'."
        )

    _, relabeled = np.unique(segments, return_inverse=True)
    return relabeled.reshape(segments.shape).astype(np.int64)
