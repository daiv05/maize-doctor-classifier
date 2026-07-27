"""Acuerdo entre dos vectores de atribucion calculados sobre los mismos superpixeles."""

import numpy as np
from scipy.stats import spearmanr

from src.explainability.stability import mask_iou


def densify_weights(local_exp: list[tuple[int, float]], n_segments: int) -> np.ndarray:
    """
    Convierte la lista dispersa (segmento, peso) de LIME en un vector denso.

    @param {list[tuple[int, float]]} local_exp Pares (id de segmento, peso).
    @param {int} n_segments Cantidad total de superpixeles.
    @returns {np.ndarray} Vector de longitud n_segments; 0.0 en los segmentos ausentes.
    """
    dense = np.zeros(n_segments, dtype=np.float64)
    for segment_id, weight in local_exp:
        dense[int(segment_id)] = float(weight)
    return dense


def top_positive_mask(values: np.ndarray, top_k: int) -> np.ndarray:
    """
    Marca los `top_k` segmentos con mayor atribucion positiva.

    @param {np.ndarray} values Vector de atribuciones por segmento.
    @param {int} top_k Cantidad de segmentos a marcar.
    @returns {np.ndarray} Mascara booleana de la misma longitud que `values`.
    """
    mask = np.zeros(values.shape, dtype=bool)
    positive = np.flatnonzero(values > 0)
    if positive.size == 0:
        return mask
    mask[positive[np.argsort(-values[positive])][:top_k]] = True
    return mask


def attribution_agreement(
    lime_weights: np.ndarray, shap_values: np.ndarray, top_k: int
) -> dict[str, float]:
    """
    Compara dos vectores de atribucion definidos sobre los mismos superpixeles.

    `iou_topk` responde si coinciden en que mirar, `spearman` si coinciden en el orden de
    importancia, y `sign_agreement` si coinciden en la direccion del empuje.

    @param {np.ndarray} lime_weights Pesos de la regresion local de LIME por segmento.
    @param {np.ndarray} shap_values Valores de Shapley por segmento.
    @param {int} top_k Segmentos positivos a considerar en el IoU.
    @returns {dict[str, float]} Claves iou_topk, spearman y sign_agreement.
    @throws {ValueError} Si los vectores no tienen la misma longitud.
    """
    if lime_weights.shape != shap_values.shape:
        raise ValueError(
            f"Vectores de longitud distinta: {lime_weights.shape} vs {shap_values.shape}"
        )

    if lime_weights.std() == 0 or shap_values.std() == 0:
        correlation = 0.0
    else:
        correlation = float(spearmanr(lime_weights, shap_values)[0])
        if np.isnan(correlation):
            correlation = 0.0

    return {
        "iou_topk": mask_iou(
            top_positive_mask(lime_weights, top_k), top_positive_mask(shap_values, top_k)
        ),
        "spearman": correlation,
        "sign_agreement": float(np.mean(np.sign(lime_weights) == np.sign(shap_values))),
    }
