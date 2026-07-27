"""Metricas post-hoc calculadas sobre predictions.csv, sin GPU.

Cubren los tres huecos que dejo la primera fase: calibracion (el reporte documenta que
el modelo se equivoca con 0.914 de confianza media), desglose lab/real (common_rust es
95% lab, lo que abre la sospecha de shortcut learning) y la metrica agrupada N/P/K
(el 97% de los errores de deficiencia se quedan dentro del propio bloque).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    num_bins: int = 15,
) -> float:
    """
    Calcula el Expected Calibration Error con bins uniformes.

    @param {list[float]} confidences Confianza de la clase predicha por muestra.
    @param {list[bool]} correct True si la prediccion fue correcta.
    @param {int} num_bins Numero de bins uniformes sobre [0, 1].
    @returns {float} Promedio ponderado de |accuracy - confianza| por bin.
    """
    confidence_array = np.asarray(confidences, dtype=float)
    correct_array = np.asarray(correct, dtype=bool)
    if confidence_array.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence_array > lower) & (confidence_array <= upper)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        error += weight * abs(correct_array[in_bin].mean() - confidence_array[in_bin].mean())
    return float(error)


def compute_calibration_metrics(
    predictions_df: pd.DataFrame,
    class_to_idx: dict[str, int],
) -> dict:
    """
    Resume calibracion: ECE, Brier multiclase y confianza media de aciertos vs. fallos.

    @param {pd.DataFrame} predictions_df Columnas label, pred_label y pred_prob.
    @param {dict[str,int]} class_to_idx Mapeo canonico clase->indice.
    @returns {dict} Metricas de calibracion del run.
    """
    correct = (predictions_df["label"] == predictions_df["pred_label"]).tolist()
    confidences = predictions_df["pred_prob"].tolist()

    is_correct = predictions_df["label"] == predictions_df["pred_label"]
    hits = predictions_df.loc[is_correct, "pred_prob"]
    misses = predictions_df.loc[~is_correct, "pred_prob"]

    confidence_array = np.asarray(confidences, dtype=float)
    correct_array = np.asarray(correct, dtype=float)
    brier = float(np.mean((confidence_array - correct_array) ** 2))

    return {
        "ece": expected_calibration_error(confidences, correct),
        "brier": brier,
        "mean_confidence_hits": float(hits.mean()) if len(hits) else 0.0,
        "mean_confidence_misses": float(misses.mean()) if len(misses) else 0.0,
        "n_hits": int(len(hits)),
        "n_misses": int(len(misses)),
        "num_classes": len(class_to_idx),
    }


def compute_environment_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Desglosa accuracy y macro-F1 por entorno de captura.

    @param {pd.DataFrame} predictions_df Debe incluir la columna environment.
    @returns {pd.DataFrame} Una fila por entorno, con la n visible para leer su fiabilidad.
    """
    rows = []
    for environment, group in predictions_df.groupby("environment"):
        rows.append(
            {
                "environment": environment,
                "n": len(group),
                "accuracy": accuracy_score(group["label"], group["pred_label"]),
                "macro_f1": f1_score(
                    group["label"], group["pred_label"], average="macro", zero_division=0
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_grouped_metrics(predictions_df: pd.DataFrame, groups: dict[str, str]) -> dict:
    """
    Recalcula las metricas colapsando las clases indicadas en una sola categoria.

    @param {pd.DataFrame} predictions_df Columnas label y pred_label.
    @param {dict[str,str]} groups Mapeo clase original -> nombre de la clase agrupada.
    @returns {dict} Metricas antes y despues de agrupar.
    """
    grouped_labels = predictions_df["label"].replace(groups)
    grouped_predictions = predictions_df["pred_label"].replace(groups)

    return {
        "groups": groups,
        "ungrouped_accuracy": accuracy_score(predictions_df["label"], predictions_df["pred_label"]),
        "ungrouped_macro_f1": f1_score(
            predictions_df["label"],
            predictions_df["pred_label"],
            average="macro",
            zero_division=0,
        ),
        "grouped_accuracy": accuracy_score(grouped_labels, grouped_predictions),
        "grouped_macro_f1": f1_score(
            grouped_labels, grouped_predictions, average="macro", zero_division=0
        ),
    }
