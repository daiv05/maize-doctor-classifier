"""Perfil global por clase: mapa espacial medio de atribucion y ratio hoja/fondo."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data.provenance import source_from_path
from src.explainability.leaf_mask import (
    is_mask_unreliable,
    leaf_mask,
    mask_coverage,
    mask_fragmentation,
)
from src.explainability.visual_report import explanation_dispersion

logger = logging.getLogger(__name__)

_UNRELIABLE_REJECTION_RATIO = 0.3
_MASK_SAMPLES_PER_CLASS = 3
_MASK_DIM_FACTOR = 0.25


class GlobalAccumulator:
    """
    Acumula atribuciones SHAP de muchas imagenes en un perfil por clase.

    El perfil es deliberadamente NO espacial. Promediar los mapas en coordenadas de pixel
    mezcla imagenes donde la hoja cae en distinta posicion y angulo, asi que el resultado
    converge a un blob centrado que refleja el encuadre del dataset, no el comportamiento
    del modelo. Las metricas que se reportan (ratio hoja/fondo, dispersion, magnitud) son
    invariantes a donde caiga la hoja.

    Cada mapa se normaliza por su propio maximo absoluto: sin eso una imagen con
    atribuciones de gran magnitud dominaria el promedio de la clase.
    """

    def __init__(
        self,
        unreliable_ratio: float = _UNRELIABLE_REJECTION_RATIO,
        mask_samples_per_class: int = _MASK_SAMPLES_PER_CLASS,
    ):
        self._maps: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._rows: list[dict] = []
        self._mask_samples: list[dict] = []
        self._unreliable_ratio = unreliable_ratio
        self._mask_samples_per_class = mask_samples_per_class

    def accumulate(
        self,
        label: str,
        correct: bool,
        shap_values: np.ndarray,
        segments: np.ndarray,
        image_np: np.ndarray,
        relative_path: str = "",
        external_mask: np.ndarray | None = None,
    ) -> None:
        """
        Incorpora la explicacion de una imagen al perfil de su clase verdadera.

        @param {str} label Clase verdadera de la imagen.
        @param {bool} correct Si el modelo acerto en esa imagen.
        @param {np.ndarray} shap_values Valores de Shapley por segmento.
        @param {np.ndarray} segments Mapa de superpixeles con etiquetas desde 0.
        @param {np.ndarray} image_np Imagen HWC uint8 reescalada a target_size.
        @param {str} relative_path Ruta de la imagen relativa a la raiz del dataset, para
                                   poder desagregar despues por procedencia.
        @param {np.ndarray | None} external_mask Mascara de hoja precalculada. Cuando se
                                   entrega, sustituye a la heuristica de color, que sobre
                                   algunas clases marca la imagen entera como hoja y deja
                                   el ratio sin significado.
        """
        weight_map = shap_values[segments]
        max_abs = np.abs(weight_map).max()
        normalized = weight_map / max_abs if max_abs > 0 else weight_map

        accumulated = self._maps.get(label)
        self._maps[label] = (
            np.abs(normalized) if accumulated is None else accumulated + np.abs(normalized)
        )
        self._counts[label] = self._counts.get(label, 0) + 1

        mask = leaf_mask(image_np) if external_mask is None else external_mask
        coverage = mask_coverage(mask)
        fragmentation = mask_fragmentation(mask)
        rejected = is_mask_unreliable(coverage, fragmentation)
        positive = np.clip(normalized, 0.0, None)
        positive_total = positive.sum()
        ratio_undefined = not rejected and positive_total <= 0
        usable = not rejected and positive_total > 0

        if sum(sample["label"] == label for sample in self._mask_samples) < (
            self._mask_samples_per_class
        ):
            self._mask_samples.append(
                {
                    "label": label,
                    "image": image_np,
                    "mask": mask,
                    "coverage": coverage,
                    "fragmentation": fragmentation,
                    "rejected": rejected,
                }
            )

        self._rows.append(
            {
                "label": label,
                "image_path": relative_path,
                "source_id": source_from_path(relative_path) if relative_path else "",
                "mask_source": "segmentador" if external_mask is not None else "exg",
                "correct": bool(correct),
                "leaf_attribution_ratio": (
                    float(positive[mask].sum() / positive_total) if usable else float("nan")
                ),
                "mask_coverage": coverage,
                "mask_fragmentation": fragmentation,
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
                mean_mask_fragmentation=("mask_fragmentation", "mean"),
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

        Solo diagnostico de encuadre: ver la nota de la clase sobre por que no se publica
        como "donde mira el modelo".

        @returns {dict[str, np.ndarray]} Un mapa HW por clase acumulada.
        """
        return {label: total / self._counts[label] for label, total in self._maps.items()}

    def mask_samples(self) -> list[dict]:
        """
        Muestras guardadas para auditar visualmente la mascara de hoja.

        @returns {list[dict]} Imagen, mascara y sus metricas, por muestra.
        """
        return self._mask_samples

    def rows(self) -> pd.DataFrame:
        """
        Filas por imagen, sin agregar.

        @returns {pd.DataFrame} Una fila por imagen acumulada.
        """
        return pd.DataFrame(self._rows)


def _plot_mask_audit(samples: list[dict], output_path: Path) -> None:
    """
    Contactsheet imagen | mascara para auditar a ojo la segmentacion de hoja.

    El ratio hoja/fondo no es interpretable sin ver que considera "hoja" la mascara, que es
    una heuristica de color y no un segmentador aprendido.

    @param {list[dict]} samples Muestras devueltas por `GlobalAccumulator.mask_samples`.
    @param {Path} output_path Ruta del PNG de salida.
    """
    if not samples:
        return

    ordered = sorted(samples, key=lambda sample: (sample["label"], -sample["coverage"]))
    columns = 6
    rows_count = -(-len(ordered) // columns)
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(2.2 * columns, 2.5 * rows_count), facecolor="white"
    )

    for axis, sample in zip(np.atleast_1d(axes).ravel(), ordered):
        overlay = sample["image"].astype(float) / 255.0
        overlay[~sample["mask"]] *= _MASK_DIM_FACTOR
        axis.imshow(np.clip(overlay, 0, 1))
        status = "RECHAZADA" if sample["rejected"] else "ok"
        color = "#C0392B" if sample["rejected"] else "#27AE60"
        axis.set_title(
            f"{sample['label']}\ncob {sample['coverage']:.2f} | frg "
            f"{sample['fragmentation']:.2f} | {status}",
            fontsize=6.5,
            color=color,
        )
        axis.axis("off")

    for axis in np.atleast_1d(axes).ravel()[len(ordered) :]:
        axis.axis("off")

    figure.suptitle(
        "Auditoria de la mascara de hoja - zona atenuada = fondo", fontsize=12, fontweight="bold"
    )
    figure.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_class_profile(rows: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    """
    Perfil por clase invariante a la posicion de la hoja.

    Panel izquierdo: distribucion del ratio hoja/fondo por clase, separando aciertos de
    errores; la linea de referencia es la cobertura media de la mascara, que es el ratio
    que daria una atribucion repartida al azar. Panel derecho: dispersion, que distingue
    una explicacion concentrada en pocos superpixeles de una repartida.

    @param {pd.DataFrame} rows Filas por imagen del acumulador.
    @param {pd.DataFrame} summary Tabla agregada, para marcar clases no confiables.
    @param {Path} output_path Ruta del PNG de salida.
    """
    labels = sorted(rows["label"].unique())
    unreliable = set(summary.loc[~summary["ratio_reliable"], "label"])

    figure, (ratio_axis, dispersion_axis) = plt.subplots(
        1, 2, figsize=(max(9, 1.6 * len(labels)), 5.5), facecolor="white"
    )

    for axis, column, title, xlabel in (
        (ratio_axis, "leaf_attribution_ratio", "Atribucion sobre hoja", "ratio hoja/fondo"),
        (dispersion_axis, "dispersion", "Concentracion de la explicacion", "dispersion"),
    ):
        data = [rows.loc[rows["label"] == label, column].dropna().to_numpy() for label in labels]
        positions = range(len(labels))
        axis.boxplot(
            [values if values.size else [np.nan] for values in data],
            positions=list(positions),
            vert=False,
            widths=0.6,
            showfliers=False,
        )
        for position, values in zip(positions, data):
            if values.size:
                axis.scatter(
                    values,
                    np.full(values.size, position)
                    + np.random.default_rng(0).uniform(-0.12, 0.12, values.size),
                    s=10,
                    alpha=0.45,
                    color="#2C7FB8",
                )
        axis.set_yticks(list(positions))
        axis.set_yticklabels(
            [
                f"{label} (!)" if label in unreliable and column.startswith("leaf") else label
                for label in labels
            ],
            fontsize=9,
        )
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel(xlabel, fontsize=9)
        axis.grid(axis="x", alpha=0.25)

    coverage = rows["mask_coverage"].mean()
    ratio_axis.axvline(coverage, color="#C0392B", linestyle="--", linewidth=1.2)
    ratio_axis.text(
        coverage,
        len(labels) - 0.4,
        f" azar ({coverage:.2f})",
        color="#C0392B",
        fontsize=8,
        va="top",
    )

    figure.suptitle(
        "Perfil global por clase - (!) = ratio no confiable", fontsize=13, fontweight="bold"
    )
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_global_report(accumulator: GlobalAccumulator, output_dir: Path) -> None:
    """
    Escribe los mapas por clase y la tabla agregada del perfil global.

    @param {GlobalAccumulator} accumulator Acumulador ya alimentado.
    @param {Path} output_dir Directorio destino (`<run_dir>/explain_global/`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    maps_dir = output_dir / "framing_diagnostics"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for label, class_map in accumulator.class_maps().items():
        figure, axis = plt.subplots(figsize=(5, 5), facecolor="white")
        image = axis.imshow(class_map, cmap="inferno")
        axis.set_title(f"Encuadre medio - {label}", fontsize=12, fontweight="bold")
        axis.axis("off")
        figure.colorbar(image, ax=axis, label="|SHAP| normalizado")
        figure.text(
            0.5,
            0.02,
            "Diagnostico de encuadre, no 'donde mira el modelo':\n"
            "promediar en pixeles mezcla hojas en distinta posicion y angulo.",
            ha="center",
            fontsize=7,
            fontstyle="italic",
            color="#95A5A6",
        )
        figure.savefig(
            maps_dir / f"{label}_framing.png", dpi=150, bbox_inches="tight", facecolor="white"
        )
        plt.close(figure)

    summary = accumulator.summary()
    _plot_class_profile(accumulator.rows(), summary, output_dir / "class_profile.png")
    _plot_mask_audit(accumulator.mask_samples(), output_dir / "mask_audit.png")
    # El exceso sobre la cobertura de la mascara es la lectura util: la cobertura es el
    # ratio que daria una atribucion repartida al azar, asi que sin restarla el ratio no
    # distingue atribuir a la hoja de atribuir a cualquier sitio.
    summary["attribution_excess"] = (
        summary["mean_leaf_attribution_ratio"] - summary["mean_mask_coverage"]
    )
    summary.to_csv(output_dir / "global_summary.csv", index=False)

    por_imagen = pd.DataFrame(accumulator.rows())
    if not por_imagen.empty:
        por_imagen["attribution_excess"] = (
            por_imagen["leaf_attribution_ratio"] - por_imagen["mask_coverage"]
        )
        por_imagen.to_csv(output_dir / "global_per_image.csv", index=False)
    (output_dir / "global_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )
    logger.info(f"Perfil global guardado en {output_dir}")
