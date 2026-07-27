"""KernelSHAP sobre superpixeles para el panel comparado del pipeline principal."""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
import shap

_BLUR_SIGMA = 15


@dataclass(frozen=True)
class ShapExplanation:
    """Valores de Shapley por superpixel para una imagen y una clase."""

    values: np.ndarray
    expected_value: float
    target_idx: int
    n_evals: int


def build_background(image_np: np.ndarray, background: str) -> np.ndarray:
    """
    Construye la imagen de referencia que reemplaza a los superpixeles ausentes.

    @param {np.ndarray} image_np Imagen HWC uint8.
    @param {str} background "black" (paridad con el hide_color=0 de LIME), "mean" o "blur".
    @returns {np.ndarray} Imagen HWC uint8 del mismo tamano.
    @throws {ValueError} Si la linea base no es una de las soportadas.
    """
    if background == "black":
        return np.zeros_like(image_np)
    if background == "mean":
        channel_mean = image_np.reshape(-1, image_np.shape[-1]).mean(axis=0)
        return np.full_like(image_np, channel_mean.astype(np.uint8))
    if background == "blur":
        return cv2.GaussianBlur(image_np, (0, 0), sigmaX=_BLUR_SIGMA)
    raise ValueError(f"Linea base desconocida: {background!r}. Usa 'black', 'mean' o 'blur'.")


def _build_coalition_fn(
    image_np: np.ndarray,
    segments: np.ndarray,
    background_np: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    target_idx: int,
    batch_size: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Arma la funcion de coalicion que evalua KernelSHAP: z[i]=1 deja visible el superpixel i.

    @param {np.ndarray} image_np Imagen HWC uint8.
    @param {np.ndarray} segments Mapa de superpixeles con etiquetas desde 0.
    @param {np.ndarray} background_np Imagen de referencia para los superpixeles ausentes.
    @param {Callable} predict_fn Mapea un batch HWC uint8 a probabilidades por clase.
    @param {int} target_idx Indice de la clase explicada.
    @param {int} batch_size Imagenes por forward pass.
    @returns {Callable} Funcion que mapea una matriz (n, k) de coaliciones a (n,) scores.
    """
    segment_masks = np.stack(
        [segments == segment_id for segment_id in range(int(segments.max()) + 1)]
    )

    def coalition_fn(coalitions: np.ndarray) -> np.ndarray:
        coalitions = np.atleast_2d(np.asarray(coalitions))
        scores = np.empty(len(coalitions), dtype=np.float64)
        for start in range(0, len(coalitions), batch_size):
            chunk = coalitions[start : start + batch_size]
            batch = np.empty((len(chunk), *image_np.shape), dtype=np.uint8)
            for position, row in enumerate(chunk):
                visible = segment_masks[row > 0.5].any(axis=0)
                batch[position] = np.where(visible[..., None], image_np, background_np)
            scores[start : start + batch_size] = predict_fn(batch)[:, target_idx]
        return scores

    return coalition_fn


def explain_with_kernel_shap(
    image_np: np.ndarray,
    segments: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    target_idx: int,
    nsamples: int = 2048,
    batch_size: int = 128,
    background: str = "black",
    seed: int = 42,
) -> ShapExplanation:
    """
    Calcula los valores de Shapley por superpixel con KernelSHAP.

    Con k superpixeles KernelSHAP enumera todas las coaliciones si `nsamples >= 2**k` y
    las muestrea si no. El muestreo consume el generador global de NumPy, asi que se fija
    la semilla y se restaura el estado previo: el determinismo entre corridas es lo que
    distingue a SHAP de LIME y no debe depender de como venga sembrado el proceso, ni
    filtrar la resiembra al resto del pipeline.

    `l1_reg` se fija en `num_features(k)` para desactivar la seleccion de variables: con
    regularizacion activa algunos superpixeles reciben exactamente cero por decision del
    lasso y no por su contribucion real, lo que rompe la aditividad y la comparacion con
    LIME.

    @param {np.ndarray} image_np Imagen HWC uint8 ya reescalada a target_size.
    @param {np.ndarray} segments Mapa de superpixeles con etiquetas consecutivas desde 0.
    @param {Callable} predict_fn Mapea un batch HWC uint8 a probabilidades por clase.
    @param {int} target_idx Indice de la clase a explicar.
    @param {int} nsamples Evaluaciones del modelo por imagen.
    @param {int} batch_size Imagenes por forward pass.
    @param {str} background Linea base de enmascarado.
    @param {int} seed Semilla del muestreo de coaliciones.
    @returns {ShapExplanation} Valores por segmento, valor esperado y metadatos.
    """
    n_segments = int(segments.max()) + 1
    coalition_fn = _build_coalition_fn(
        image_np=image_np,
        segments=segments,
        background_np=build_background(image_np, background),
        predict_fn=predict_fn,
        target_idx=target_idx,
        batch_size=batch_size,
    )

    previous_state = np.random.get_state()
    np.random.seed(seed)
    try:
        explainer = shap.KernelExplainer(coalition_fn, np.zeros((1, n_segments)))
        raw_values = explainer.shap_values(
            np.ones((1, n_segments)),
            nsamples=nsamples,
            l1_reg=f"num_features({n_segments})",
            silent=True,
        )
    finally:
        np.random.set_state(previous_state)

    return ShapExplanation(
        values=np.asarray(raw_values, dtype=np.float64).reshape(n_segments),
        expected_value=float(np.asarray(explainer.expected_value).reshape(-1)[0]),
        target_idx=int(target_idx),
        n_evals=int(nsamples),
    )
