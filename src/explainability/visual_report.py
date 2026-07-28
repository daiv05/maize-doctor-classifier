import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from lime import lime_image
from matplotlib import cm, colormaps, gridspec
from matplotlib import pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries

from src.data.loader import load_and_normalize_image
from src.explainability.gradcam import GradCAM, build_gradcam_overlay, get_target_layer

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_HIGHLIGHT_COLOR = (0.2, 0.8, 0.3)
_DIM_FACTOR = 0.45

# Nombres legibles en español para el diagnóstico. Cualquier clase no listada
# cae al fallback de _humanize_class_name (title-case con espacios).
_DISPLAY_NAMES: dict[str, str] = {
    "common_rust": "Roya Común",
    "fall_armyworm": "Gusano Cogollero",
    "gray_leaf_spot": "Mancha Gris",
    "healthy": "Sana",
    "lethal_necrosis": "Necrosis Letal",
    "nitrogen_deficiency": "Deficiencia de Nitrógeno",
    "northern_corn_leaf_blight": "Tizón Foliar Norteño",
    "phosphorus_deficiency": "Deficiencia de Fósforo",
    "potassium_deficiency": "Deficiencia de Potasio",
}

# Color del título principal según la naturaleza del diagnóstico.
_COLOR_HEALTHY = "#27AE60"
_COLOR_DEFICIENCY = "#E67E22"
_COLOR_DISEASE = "#C0392B"


def sample_balanced(test_df: pd.DataFrame, num_per_class: int, seed: int) -> pd.DataFrame:
    """Muestra reproducible de hasta `num_per_class` filas por cada clase de `test_df`."""
    sampled = [
        group.sample(n=min(num_per_class, len(group)), random_state=seed)
        for _, group in test_df.groupby("label")
    ]
    return pd.concat(sampled, ignore_index=True)


def explanation_dispersion(local_exp: list[tuple[int, float]]) -> float:
    """
    Desviación estándar de los pesos LIME normalizados por el máximo absoluto: valores
    bajos indican una explicación concentrada en pocos superpíxeles dominantes; valores
    altos indican pesos repartidos de forma pareja (explicación más difícil de
    interpretar visualmente, como se observó en las clases con fondo ruidoso de campo).
    """
    weights = np.array([w for _, w in local_exp])
    max_abs = np.abs(weights).max()
    return float(np.std(weights / max_abs)) if max_abs > 0 else 0.0


def _humanize_class_name(class_name: str) -> str:
    return _DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())


def _diagnosis_color(class_name: str) -> str:
    if class_name == "healthy":
        return _COLOR_HEALTHY
    if "deficiency" in class_name:
        return _COLOR_DEFICIENCY
    return _COLOR_DISEASE


def build_validation_transform(target_size: tuple[int, int]) -> T.Compose:
    """Resize + normalize deterministas de validación, compartidos entre `predict_fn`
    (LIME) y la construcción del tensor de entrada para Grad-CAM."""
    return T.Compose(
        [
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def prepare_lime_image(image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """Lleva la imagen al tamaño de entrada como array HWC uint8 para LimeImageExplainer.

    Usa la misma `T.Resize` que `_build_validation_transform`, no `PIL.Image.resize`: el
    array resultante vuelve a pasar por esa transform dentro de `predict_fn`, y un
    remuestreo distinto (PIL usa bicúbico por defecto, `T.Resize` bilineal) hacía que el
    modelo evaluara píxeles distintos a los del pipeline de entrenamiento. Sobre imágenes
    de alta resolución la diferencia bastaba para cambiar el argmax en predicciones de
    margen estrecho. `T.Resize` además toma (alto, ancho), el mismo orden que
    `target_size`, mientras que `PIL.Image.resize` espera (ancho, alto).

    @param {Image.Image} image Imagen original, sin reescalar.
    @param {tuple[int, int]} target_size Tamaño de entrada del checkpoint, (alto, ancho).
    @returns {np.ndarray} Array HWC uint8 con la imagen reescalada.
    """
    return np.array(T.Resize(target_size)(image))


def build_predict_fn(
    model: nn.Module, device: torch.device, target_size: tuple[int, int]
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Envuelve `model` en la función predict_fn que espera `LimeImageExplainer`: recibe
    un batch de imágenes HWC uint8 y devuelve las probabilidades softmax por clase,
    aplicando las mismas transforms deterministas de validación (resize + normalize).
    """
    validation_transform = build_validation_transform(target_size)

    @torch.no_grad()
    def predict_fn(images: np.ndarray) -> np.ndarray:
        batch = torch.stack([validation_transform(Image.fromarray(img)) for img in images]).to(
            device
        )
        probs = model(batch).softmax(dim=1)
        return probs.cpu().numpy()

    return predict_fn


def build_positive_region_panel(image_rgb01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Superpíxeles positivos resaltados en verde semitransparente (mezcla 50/50 con
    `_HIGHLIGHT_COLOR`); el resto de la imagen se atenúa (no se oculta) y luego se
    dibujan los bordes gruesos verdes con `mark_boundaries`.
    """
    region = image_rgb01.copy()
    is_important = mask > 0
    region[is_important] = image_rgb01[is_important] * 0.5 + np.array(_HIGHLIGHT_COLOR) * 0.5
    region[~is_important] = image_rgb01[~is_important] * _DIM_FACTOR
    return mark_boundaries(region, mask, color=(0, 1, 0), mode="thick", outline_color=(0, 1, 0))


def build_importance_heatmap(
    image_rgb01: np.ndarray, segments: np.ndarray, local_exp: list[tuple[int, float]]
) -> tuple[np.ndarray, plt.Normalize]:
    """
    Mapa de importancia continuo: cada superpíxel toma el peso LIME que le asignó
    la regresión local, coloreado con RdYlGn y superpuesto sobre la imagen original.
    """
    weight_map = np.zeros(segments.shape, dtype=float)
    for segment_id, weight in local_exp:
        weight_map[segments == segment_id] = weight

    max_abs_weight = np.abs(weight_map).max()
    if max_abs_weight == 0:
        max_abs_weight = 1.0
    norm = plt.Normalize(vmin=-max_abs_weight, vmax=max_abs_weight)

    heatmap_rgba = colormaps["RdYlGn"](norm(weight_map))
    overlay = image_rgb01 * 0.35 + heatmap_rgba[..., :3] * 0.65
    return np.clip(overlay, 0, 1), norm


def render_visual_explanation(
    image: Image.Image,
    model: nn.Module,
    idx_to_class: dict[int, str],
    target_size: tuple[int, int],
    output_path: Path,
    num_samples: int = 300,
    num_features: int = 8,
    seed: int = 42,
    device: torch.device | None = None,
    model_name: str | None = None,
    segments: np.ndarray | None = None,
) -> dict:
    """
    Genera el reporte visual (original / regiones positivas / heatmap de importancia
    LIME, y opcionalmente Grad-CAM como 4to panel) para una única imagen y lo guarda
    como PNG en `output_path`.

    `model_name` es opcional (default None): si se provee y está registrado en
    `GRADCAM_TARGET_LAYERS`, añade el panel Grad-CAM; si es None o la arquitectura no
    está soportada, el reporte mantiene el layout de 3 paneles original.

    `segments` es opcional (default None): con None, LIME segmenta internamente con
    quickshift, que es su comportamiento historico y el de los reportes ya publicados.
    Con un mapa explicito, LIME atribuye sobre esas mismas regiones, que es lo que
    permite comparar sus pesos con los valores de Shapley segmento a segmento.

    Devuelve un dict con la predicción y probabilidad, útil para logging/metadata.

    @param {np.ndarray|None} segments Mapa de superpixeles a imponer, o None.
    """
    device = device or torch.device("cpu")
    model.eval()

    image_np = prepare_lime_image(image, target_size)
    image_rgb01 = image_np.astype(float) / 255.0

    predict_fn = build_predict_fn(model, device, target_size)
    explainer = lime_image.LimeImageExplainer(random_state=seed)

    explain_kwargs = {}
    if segments is not None:
        explain_kwargs["segmentation_fn"] = lambda _image: segments

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=len(idx_to_class),
        hide_color=0,
        num_samples=num_samples,
        random_seed=seed,
        **explain_kwargs,
    )

    pred_idx = explanation.top_labels[0]
    pred_class = idx_to_class.get(pred_idx, str(pred_idx))
    pred_prob = float(predict_fn(image_np[np.newaxis, ...])[0, pred_idx])

    _, mask = explanation.get_image_and_mask(
        pred_idx,
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )
    region_panel = build_positive_region_panel(image_rgb01, mask)

    local_exp = explanation.local_exp[pred_idx]
    heatmap_panel, norm = build_importance_heatmap(image_rgb01, explanation.segments, local_exp)

    gradcam_panel = None
    if model_name is not None:
        try:
            target_layer = get_target_layer(model, model_name)
            input_tensor = build_validation_transform(target_size)(image).unsqueeze(0).to(device)
            with GradCAM(model, target_layer) as cam:
                heatmap = cam(input_tensor, class_idx=pred_idx)
            gradcam_panel = build_gradcam_overlay(image_rgb01, heatmap, target_size)
        except KeyError as e:
            logger.warning(f"Grad-CAM omitido: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(
        original=image_rgb01,
        region_panel=region_panel,
        heatmap_panel=heatmap_panel,
        heatmap_norm=norm,
        pred_class=pred_class,
        pred_prob=pred_prob,
        output_path=output_path,
        gradcam_panel=gradcam_panel,
    )
    _save_explanation_artifacts(
        output_path=output_path,
        pred_class=pred_class,
        pred_prob=pred_prob,
        local_exp=local_exp,
        segments=explanation.segments,
    )

    return {"predicted_label": pred_class, "predicted_prob": pred_prob}


def _save_explanation_artifacts(
    output_path: Path,
    pred_class: str,
    pred_prob: float,
    local_exp: list[tuple[int, float]],
    segments: np.ndarray,
) -> None:
    """
    Persiste junto al PNG (mismo stem) los datos numéricos que LIME ya calculó: un
    .json con la predicción y los pesos por superpíxel, y un .npy con el mapa de
    segmentos, necesarios para recalcular el heatmap o cruzarlo con otras máscaras
    sin tener que re-ejecutar LIME.
    """
    metadata = {
        "predicted_label": pred_class,
        "predicted_prob": pred_prob,
        "top_features": [
            {"segment_id": int(seg_id), "weight": float(weight)} for seg_id, weight in local_exp
        ],
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    np.save(output_path.with_suffix(".npy"), segments)


def _save_figure(
    original: np.ndarray,
    region_panel: np.ndarray,
    heatmap_panel: np.ndarray,
    heatmap_norm: plt.Normalize,
    pred_class: str,
    pred_prob: float,
    output_path: Path,
    gradcam_panel: np.ndarray | None = None,
) -> None:
    title_color = _diagnosis_color(pred_class)
    diagnosis_name = _humanize_class_name(pred_class)

    has_gradcam = gradcam_panel is not None
    n_image_panels = 4 if has_gradcam else 3
    width_ratios = [1] * n_image_panels + [0.08]

    fig = plt.figure(figsize=(16 + (4 if has_gradcam else 0), 6), facecolor="white")
    grid = gridspec.GridSpec(1, n_image_panels + 1, width_ratios=width_ratios)

    ax_original = fig.add_subplot(grid[0, 0])
    ax_original.imshow(original)
    ax_original.set_title(
        "Imagen Original", fontsize=13, fontweight="bold", color="#2C3E50", pad=12
    )
    ax_original.axis("off")

    ax_regions = fig.add_subplot(grid[0, 1])
    ax_regions.imshow(region_panel)
    ax_regions.set_title(
        "Zonas que Determinan el Diagnóstico",
        fontsize=13,
        fontweight="bold",
        color="#2C3E50",
        pad=12,
    )
    ax_regions.axis("off")

    ax_heatmap = fig.add_subplot(grid[0, 2])
    ax_heatmap.imshow(heatmap_panel)
    ax_heatmap.set_title(
        "Mapa de Importancia", fontsize=13, fontweight="bold", color="#2C3E50", pad=12
    )
    ax_heatmap.axis("off")

    if has_gradcam:
        ax_gradcam = fig.add_subplot(grid[0, 3])
        ax_gradcam.imshow(gradcam_panel)
        ax_gradcam.set_title("Grad-CAM", fontsize=13, fontweight="bold", color="#2C3E50", pad=12)
        ax_gradcam.axis("off")

    ax_colorbar = fig.add_subplot(grid[0, n_image_panels])
    mappable = cm.ScalarMappable(norm=heatmap_norm, cmap="RdYlGn")
    fig.colorbar(mappable, cax=ax_colorbar, label="Importancia")

    fig.suptitle(
        f"Diagnóstico: {diagnosis_name} - Confianza: {pred_prob * 100:.1f}%",
        fontsize=16,
        fontweight="bold",
        color=title_color,
    )
    fig.text(
        0.5,
        0.01,
        "Las zonas verdes resaltadas son las regiones de la hoja que más influyeron "
        "en el diagnóstico del modelo.",
        ha="center",
        fontsize=9,
        fontstyle="italic",
        color="#95A5A6",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Reporte visual LIME guardado en {output_path}")


def explain_model_visual(
    model: nn.Module,
    model_name: str,
    test_df: pd.DataFrame,
    dataset_root: Path,
    idx_to_class: dict[int, str],
    target_size: tuple[int, int],
    output_dir: Path,
    images_per_class: int,
    num_features: int,
    num_samples: int,
    seed: int,
    device: torch.device,
    enable_gradcam: bool = True,
) -> None:
    """
    Genera el reporte visual (3 paneles LIME, 4 si `enable_gradcam`) para una muestra
    balanceada de `test_df` (`images_per_class` imágenes por clase) y las guarda como
    PNG bajo `<output_dir>/explain_visual/`. `output_dir` ya debe ser el directorio de la
    corrida concreta (el caller es responsable de incluir model_name/run_id).
    """
    model.eval()
    df_sample = sample_balanced(test_df, images_per_class, seed)
    lime_dir = output_dir / "explain_visual"

    for _, row in df_sample.iterrows():
        img_path = dataset_root / row["image_path"]
        true_label = row["label"]

        try:
            image = load_and_normalize_image(img_path)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"[{model_name}] Saltando {img_path}: {e}")
            continue

        stem = Path(str(row["image_path"])).stem
        output_path = lime_dir / f"{stem}__true-{true_label}.png"

        result = render_visual_explanation(
            image=image,
            model=model,
            idx_to_class=idx_to_class,
            target_size=target_size,
            output_path=output_path,
            num_samples=num_samples,
            num_features=num_features,
            seed=seed,
            device=device,
            model_name=model_name if enable_gradcam else None,
        )
        logger.info(
            f"[{model_name}] {img_path.name}: predicho={result['predicted_label']} "
            f"({result['predicted_prob'] * 100:.1f}%)"
        )

    logger.info(f"[{model_name}] Paneles visuales guardados en {lime_dir}")
