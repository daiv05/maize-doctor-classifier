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


def expected_iou(tamano_a: int, tamano_b: int, total: int) -> float:
    """Solapamiento esperado entre dos selecciones independientes del mismo tamano.

    Es la referencia sin la cual `iou_topk` no se puede leer: con pocos segmentos, dos
    selecciones al azar ya se solapan de forma apreciable, asi que un valor observado alto
    no implica acuerdo entre las dos tecnicas.

    @param {int} tamano_a Segmentos seleccionados por la primera tecnica.
    @param {int} tamano_b Segmentos seleccionados por la segunda.
    @param {int} total Segmentos disponibles.
    @returns {float} IoU esperado bajo independencia.
    """
    if total <= 0 or tamano_a == 0 or tamano_b == 0:
        return 0.0
    interseccion = tamano_a * tamano_b / total
    union = tamano_a + tamano_b - interseccion
    return float(interseccion / union) if union > 0 else 0.0


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
    @returns {dict[str, float]} Cada metrica junto a su valor esperado bajo
                                independencia, con sufijo `_null`.
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

    mascara_lime = top_positive_mask(lime_weights, top_k)
    mascara_shap = top_positive_mask(shap_values, top_k)
    proporcion_lime = float((lime_weights > 0).mean())
    proporcion_shap = float((shap_values > 0).mean())

    return {
        "iou_topk": mask_iou(mascara_lime, mascara_shap),
        "iou_topk_null": expected_iou(
            int(mascara_lime.sum()), int(mascara_shap.sum()), lime_weights.size
        ),
        "spearman": correlation,
        "spearman_null": 0.0,
        "sign_agreement": float(np.mean(np.sign(lime_weights) == np.sign(shap_values))),
        "sign_agreement_null": (
            proporcion_lime * proporcion_shap
            + (1.0 - proporcion_lime) * (1.0 - proporcion_shap)
        ),
    }
