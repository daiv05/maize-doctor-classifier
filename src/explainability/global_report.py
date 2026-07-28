"""Perfil global por clase: mapa espacial medio de atribucion y ratio hoja/fondo."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.explainability.leaf_mask import is_coverage_degenerate, leaf_mask, mask_coverage
from src.explainability.visual_report import explanation_dispersion

logger = logging.getLogger(__name__)

_UNRELIABLE_REJECTION_RATIO = 0.3


class GlobalAccumulator:
    """
    Acumula atribuciones SHAP de muchas imagenes en un perfil por clase.

    Cada mapa se normaliza por su propio maximo absoluto antes de acumularse: sin eso, una
    imagen con atribuciones de gran magnitud dominaria el promedio de la clase y el mapa
    dejaria de responder "donde mira el modelo" para responder "que imagen grito mas".
    """

    def __init__(self, unreliable_ratio: float = _UNRELIABLE_REJECTION_RATIO):
        self._maps: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._rows: list[dict] = []
        self._unreliable_ratio = unreliable_ratio

    def accumulate(
        self,
        label: str,
        correct: bool,
        shap_values: np.ndarray,
        segments: np.ndarray,
        image_np: np.ndarray,
    ) -> None:
        """
        Incorpora la explicacion de una imagen al perfil de su clase verdadera.

        @param {str} label Clase verdadera de la imagen.
        @param {bool} correct Si el modelo acerto en esa imagen.
        @param {np.ndarray} shap_values Valores de Shapley por segmento.
        @param {np.ndarray} segments Mapa de superpixeles con etiquetas desde 0.
        @param {np.ndarray} image_np Imagen HWC uint8 reescalada a target_size.
        """
        weight_map = shap_values[segments]
        max_abs = np.abs(weight_map).max()
        normalized = weight_map / max_abs if max_abs > 0 else weight_map

        accumulated = self._maps.get(label)
        self._maps[label] = (
            np.abs(normalized) if accumulated is None else accumulated + np.abs(normalized)
        )
        self._counts[label] = self._counts.get(label, 0) + 1

        mask = leaf_mask(image_np)
        coverage = mask_coverage(mask)
        rejected = is_coverage_degenerate(coverage)
        positive = np.clip(normalized, 0.0, None)
        positive_total = positive.sum()
        ratio_undefined = not rejected and positive_total <= 0
        usable = not rejected and positive_total > 0

        self._rows.append(
            {
                "label": label,
                "correct": bool(correct),
                "leaf_attribution_ratio": (
                    float(positive[mask].sum() / positive_total) if usable else float("nan")
                ),
                "mask_coverage": coverage,
                "mask_rejected": rejected,
                "ratio_undefined": ratio_undefined,
                "abs_attribution": float(np.abs(normalized).mean()),
                "dispersion": explanation_dispersion(list(enumerate(shap_values))),
            }
        )

    def is_empty(self) -> bool:
        """
        Indica si no se acumulo ninguna imagen.

        `summary()` agrupa por columnas que solo existen cuando hay al menos una fila, asi
        que sobre un acumulador vacio falla dentro de pandas con un error opaco. Los
        callers consultan esto antes de pedir el resumen.

        @returns {bool} True si el acumulador no recibio ninguna imagen.
        """
        return not self._rows

    def summary(self) -> pd.DataFrame:
        """
        Agrega las filas acumuladas por clase y correctitud.

        Las imagenes con mascara descartada entran en `n` y en `n_mask_rejected`, pero su
        ratio es NaN y queda fuera del promedio: se cuentan sin contaminar la metrica. Lo
        mismo aplica a las imagenes con mascara valida pero sin atribucion positiva que
        repartir (`n_ratio_undefined`): son una causa distinta del mismo sintoma - un ratio
        no confiable - y se cuentan por separado porque el diagnostico que sugieren al
        lector humano es distinto (mascara de vegetacion fallando vs. modelo sin atribucion
        positiva). `ratio_reliable` se apaga cuando la suma de ambos supera el umbral: un
        numero o es confiable o se declara no confiable, nunca miente en silencio.

        @returns {pd.DataFrame} Una fila por (label, correct) con las metricas agregadas.
        """
        frame = pd.DataFrame(self._rows)
        grouped = (
            frame.groupby(["label", "correct"])
            .agg(
                n=("mask_coverage", "size"),
                n_mask_rejected=("mask_rejected", "sum"),
                n_ratio_undefined=("ratio_undefined", "sum"),
                mean_leaf_attribution_ratio=("leaf_attribution_ratio", "mean"),
                mean_mask_coverage=("mask_coverage", "mean"),
                mean_abs_attribution=("abs_attribution", "mean"),
                mean_dispersion=("dispersion", "mean"),
            )
            .reset_index()
        )
        grouped["ratio_reliable"] = (
            (grouped["n_mask_rejected"] + grouped["n_ratio_undefined"]) / grouped["n"]
        ) <= self._unreliable_ratio
        return grouped

    def class_maps(self) -> dict[str, np.ndarray]:
        """
        Mapa espacial medio de |atribucion| por clase.

        @returns {dict[str, np.ndarray]} Un mapa HW por clase acumulada.
        """
        return {label: total / self._counts[label] for label, total in self._maps.items()}


def write_global_report(accumulator: GlobalAccumulator, output_dir: Path) -> None:
    """
    Escribe los mapas por clase y la tabla agregada del perfil global.

    @param {GlobalAccumulator} accumulator Acumulador ya alimentado.
    @param {Path} output_dir Directorio destino (`<run_dir>/explain_global/`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, class_map in accumulator.class_maps().items():
        figure, axis = plt.subplots(figsize=(5, 5), facecolor="white")
        image = axis.imshow(class_map, cmap="inferno")
        axis.set_title(f"Atribucion media - {label}", fontsize=12, fontweight="bold")
        axis.axis("off")
        figure.colorbar(image, ax=axis, label="|SHAP| normalizado")
        figure.savefig(
            output_dir / f"{label}_attribution_map.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

    summary = accumulator.summary()
    summary.to_csv(output_dir / "global_summary.csv", index=False)
    (output_dir / "global_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )
    logger.info(f"Perfil global guardado en {output_dir}")
