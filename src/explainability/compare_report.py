"""Panel comparado LIME | SHAP | Grad-CAM sobre una segmentacion compartida."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from PIL import Image

from src.explainability.agreement import attribution_agreement, densify_weights
from src.explainability.gradcam import GradCAM, build_gradcam_overlay, get_target_layer
from src.explainability.kernel_shap import explain_with_kernel_shap
from src.explainability.segmentation import build_segments
from src.explainability.visual_report import (
    build_importance_heatmap,
    build_predict_fn,
    build_validation_transform,
    explanation_dispersion,
    prepare_lime_image,
    run_lime_explanation,
)

logger = logging.getLogger(__name__)

_TITLE_COLOR = "#2C3E50"


def render_comparison(
    image: Image.Image,
    model: nn.Module,
    model_name: str | None,
    idx_to_class: dict[int, str],
    target_size: tuple[int, int],
    output_path: Path,
    lime_cfg: dict,
    shap_cfg: dict,
    device: torch.device,
) -> dict:
    """
    Genera el panel comparado de una imagen y persiste sus artefactos numericos.

    LIME y KernelSHAP se ejecutan sobre el mismo mapa de superpixeles, calculado una sola
    vez con `build_segments`, y sobre la misma clase (el argmax del modelo). Esa doble
    identidad es lo que hace que las metricas de acuerdo comparen lo mismo.

    Las regiones LIME de este panel no coinciden con las del panel `visual`: alli LIME
    segmenta con quickshift y aqui recibe la segmentacion impuesta (por defecto SLIC).

    @param {Image.Image} image Imagen original, sin reescalar.
    @param {nn.Module} model Modelo en modo eval.
    @param {str|None} model_name Nombre registrado para Grad-CAM, o None para omitirlo.
    @param {dict[int, str]} idx_to_class Mapeo indice->clase del head entrenado.
    @param {tuple[int, int]} target_size Tamano de entrada del checkpoint.
    @param {Path} output_path Ruta del PNG; los sidecars usan el mismo stem.
    @param {dict} lime_cfg Bloque `lime` de config/dataset.yaml.
    @param {dict} shap_cfg Bloque `shap` de config/dataset.yaml.
    @param {torch.device} device Dispositivo de computo.
    @returns {dict} Prediccion, confianza, dispersion de SHAP, metricas de acuerdo y
        `agreement_reliable` (False cuando `shap_cfg["background"]` no es "black" y las
        metricas de acuerdo por lo tanto no son comparables, ver D7).
    """
    model.eval()
    image_np = prepare_lime_image(image, target_size)
    image_rgb01 = image_np.astype(float) / 255.0
    predict_fn = build_predict_fn(model, device, target_size)

    probabilities = predict_fn(image_np[np.newaxis, ...])[0]
    target_idx = int(np.argmax(probabilities))

    segments = build_segments(
        image_np,
        algorithm=shap_cfg["segmentation"],
        n_segments=shap_cfg["n_segments"],
        compactness=shap_cfg["compactness"],
    )
    n_segments = int(segments.max()) + 1

    lime_explanation = run_lime_explanation(
        image_np,
        predict_fn,
        num_labels=len(idx_to_class),
        num_samples=lime_cfg["num_samples"],
        seed=lime_cfg["seed"],
        hide_color=0,
        segments=segments,
    )
    lime_weights = densify_weights(lime_explanation.local_exp[target_idx], n_segments)

    shap_explanation = explain_with_kernel_shap(
        image_np=image_np,
        segments=segments,
        predict_fn=predict_fn,
        target_idx=target_idx,
        nsamples=shap_cfg["nsamples"],
        batch_size=shap_cfg["batch_size"],
        background=shap_cfg["background"],
        seed=shap_cfg["seed"],
    )

    agreement = attribution_agreement(
        lime_weights, shap_explanation.values, lime_cfg["num_features"]
    )
    agreement_reliable = shap_cfg["background"] == "black"
    if not agreement_reliable:
        logger.warning(
            f"shap.background={shap_cfg['background']!r}: LIME siempre enmascara con "
            "hide_color=0 (negro, D7), asi que las metricas de acuerdo (iou_topk, "
            "spearman, sign_agreement) comparan nociones de ausencia distintas entre "
            "tecnicas y no son comparables entre si. Usa shap.background=black para "
            "paridad si necesitas confiar en el acuerdo."
        )

    lime_panel, lime_norm = build_importance_heatmap(
        image_rgb01, segments, list(enumerate(lime_weights))
    )
    shap_panel, shap_norm = build_importance_heatmap(
        image_rgb01, segments, list(enumerate(shap_explanation.values))
    )
    gradcam_panel = _build_gradcam_panel(
        model, model_name, image, image_rgb01, target_size, target_idx, device
    )

    predicted_label = idx_to_class.get(target_idx, str(target_idx))
    predicted_prob = float(probabilities[target_idx])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(
        original=image_rgb01,
        lime_panel=lime_panel,
        lime_norm=lime_norm,
        shap_panel=shap_panel,
        shap_norm=shap_norm,
        gradcam_panel=gradcam_panel,
        predicted_label=predicted_label,
        predicted_prob=predicted_prob,
        agreement=agreement,
        segmentation=shap_cfg["segmentation"],
        output_path=output_path,
    )

    dispersion = explanation_dispersion(list(enumerate(shap_explanation.values)))
    _save_artifacts(
        output_path=output_path,
        segments=segments,
        metadata={
            "predicted_label": predicted_label,
            "predicted_prob": predicted_prob,
            "target_idx": target_idx,
            "n_segments": n_segments,
            "segmentation": shap_cfg["segmentation"],
            "shap_expected_value": shap_explanation.expected_value,
            "shap_n_evals": shap_explanation.n_evals,
            "shap_dispersion": dispersion,
            "lime_weights": [float(weight) for weight in lime_weights],
            "shap_values": [float(value) for value in shap_explanation.values],
            "agreement": agreement,
            "agreement_reliable": agreement_reliable,
        },
    )

    return {
        "predicted_label": predicted_label,
        "predicted_prob": predicted_prob,
        "dispersion": dispersion,
        "agreement": agreement,
        "agreement_reliable": agreement_reliable,
    }


def _build_gradcam_panel(
    model: nn.Module,
    model_name: str | None,
    image: Image.Image,
    image_rgb01: np.ndarray,
    target_size: tuple[int, int],
    target_idx: int,
    device: torch.device,
) -> np.ndarray | None:
    """
    Calcula el overlay de Grad-CAM, o None si la arquitectura no lo soporta.

    @param {nn.Module} model Modelo en modo eval.
    @param {str|None} model_name Nombre registrado, o None para omitir el panel.
    @param {Image.Image} image Imagen original, sin reescalar.
    @param {np.ndarray} image_rgb01 Imagen reescalada en [0, 1].
    @param {tuple[int, int]} target_size Tamano de entrada del checkpoint.
    @param {int} target_idx Clase explicada.
    @param {torch.device} device Dispositivo de computo.
    @returns {np.ndarray|None} Overlay RGB en [0, 1], o None.
    """
    if model_name is None:
        return None
    try:
        target_layer = get_target_layer(model, model_name)
    except KeyError as error:
        logger.warning(f"Grad-CAM omitido: {error}")
        return None

    input_tensor = build_validation_transform(target_size)(image).unsqueeze(0).to(device)
    with GradCAM(model, target_layer) as cam:
        heatmap = cam(input_tensor, class_idx=target_idx)
    return build_gradcam_overlay(image_rgb01, heatmap, target_size)


def _save_artifacts(output_path: Path, segments: np.ndarray, metadata: dict) -> None:
    """
    Persiste los sidecars .json y .npy junto al PNG.

    @param {Path} output_path Ruta del PNG.
    @param {np.ndarray} segments Mapa de superpixeles compartido.
    @param {dict} metadata Contenido del sidecar .json.
    """
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(output_path.with_suffix(".npy"), segments)


def _save_figure(
    original: np.ndarray,
    lime_panel: np.ndarray,
    lime_norm: plt.Normalize,
    shap_panel: np.ndarray,
    shap_norm: plt.Normalize,
    gradcam_panel: np.ndarray | None,
    predicted_label: str,
    predicted_prob: float,
    agreement: dict[str, float],
    segmentation: str,
    output_path: Path,
) -> None:
    """
    Dibuja original, LIME, SHAP y (si existe) Grad-CAM, con una barra de color por metodo.

    @param {np.ndarray} original Imagen reescalada en [0, 1].
    @param {np.ndarray} lime_panel Overlay de importancia de LIME.
    @param {plt.Normalize} lime_norm Normalizacion de la barra de color de LIME.
    @param {np.ndarray} shap_panel Overlay de valores de Shapley.
    @param {plt.Normalize} shap_norm Normalizacion de la barra de color de SHAP.
    @param {np.ndarray|None} gradcam_panel Overlay de Grad-CAM, o None.
    @param {str} predicted_label Clase predicha.
    @param {float} predicted_prob Confianza de la clase predicha.
    @param {dict[str, float]} agreement Metricas de acuerdo entre LIME y SHAP.
    @param {str} segmentation Algoritmo de segmentacion usado.
    @param {Path} output_path Ruta del PNG de salida.
    """
    panels = [(original, "Imagen Original", None), (lime_panel, "LIME", lime_norm)]
    panels.append((shap_panel, "SHAP (valores de Shapley)", shap_norm))
    if gradcam_panel is not None:
        panels.append((gradcam_panel, "Grad-CAM", None))

    fig = plt.figure(figsize=(5 * len(panels), 6), facecolor="white")
    grid = gridspec.GridSpec(1, len(panels) + 2, width_ratios=[1] * len(panels) + [0.06, 0.06])

    for position, (panel, title, _) in enumerate(panels):
        axis = fig.add_subplot(grid[0, position])
        axis.imshow(panel)
        axis.set_title(title, fontsize=13, fontweight="bold", color=_TITLE_COLOR, pad=12)
        axis.axis("off")

    for offset, (norm, label) in enumerate(((lime_norm, "LIME"), (shap_norm, "SHAP"))):
        axis = fig.add_subplot(grid[0, len(panels) + offset])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap="RdYlGn"), cax=axis, label=label)

    fig.suptitle(
        f"Diagnostico: {predicted_label} - Confianza: {predicted_prob * 100:.1f}%",
        fontsize=16,
        fontweight="bold",
        color=_TITLE_COLOR,
    )
    fig.text(
        0.5,
        0.01,
        f"Segmentacion compartida ({segmentation}) - "
        f"IoU top-k: {agreement['iou_topk']:.2f} | "
        f"Spearman: {agreement['spearman']:.2f} | "
        f"Acuerdo de signo: {agreement['sign_agreement']:.2f}",
        ha="center",
        fontsize=9,
        fontstyle="italic",
        color="#95A5A6",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Panel comparado guardado en {output_path}")
