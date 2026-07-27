"""Escritura de los artefactos de un run de entrenamiento.

Compartido por baselines y pipeline principal para que ambos produzcan el mismo
esquema de salida y explain_report.py pueda leer los dos indistintamente.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.training.evaluation import (
    compute_calibration_metrics,
    compute_environment_metrics,
    compute_grouped_metrics,
)

NPK_GROUPS: dict[str, str] = {
    "nitrogen_deficiency": "nutrient_deficiency",
    "phosphorus_deficiency": "nutrient_deficiency",
    "potassium_deficiency": "nutrient_deficiency",
}


def write_test_outputs(
    run_dir: Path,
    idx_to_class: dict[int, str],
    labels: list[int],
    predictions: list[int],
) -> None:
    """
    Escribe el classification report y la matriz de confusion del split de test.

    @param {Path} run_dir Directorio del run.
    @param {dict[int,str]} idx_to_class Mapeo indice->clase.
    @param {list[int]} labels Etiquetas reales codificadas.
    @param {list[int]} predictions Predicciones codificadas.
    """
    target_ids = sorted(idx_to_class)
    target_names = [idx_to_class[idx] for idx in target_ids]

    report = classification_report(
        labels,
        predictions,
        labels=target_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(run_dir / "test_classification_report.csv")

    matrix = confusion_matrix(labels, predictions, labels=target_ids)
    pd.DataFrame(matrix, index=target_names, columns=target_names).to_csv(
        run_dir / "test_confusion_matrix.csv"
    )


def write_predictions_csv(
    run_dir: Path,
    test_dataset,
    idx_to_class: dict[int, str],
    predictions: list[int],
    probs: list[float],
) -> pd.DataFrame:
    """
    Escribe predictions.csv con una fila por imagen de test.

    @param {Path} run_dir Directorio del run.
    @param {CornDataset} test_dataset Dataset de test, fuente de rutas y etiquetas.
    @param {dict[int,str]} idx_to_class Mapeo indice->clase.
    @param {list[int]} predictions Predicciones codificadas.
    @param {list[float]} probs Confianza de la clase predicha.
    @returns {pd.DataFrame} El propio dataframe escrito.
    """
    frame = pd.DataFrame(
        {
            "image_path": test_dataset.data_frame["image_path"].tolist(),
            "label": test_dataset.data_frame["label"].tolist(),
            "pred_label": [idx_to_class[p] for p in predictions],
            "pred_prob": probs,
        }
    )
    if "environment" in test_dataset.data_frame.columns:
        frame["environment"] = test_dataset.data_frame["environment"].tolist()
    frame.to_csv(run_dir / "predictions.csv", index=False)
    return frame


def write_extended_metrics(
    run_dir: Path,
    predictions_df: pd.DataFrame,
    class_to_idx: dict[str, int],
    npk_groups: dict[str, str] = NPK_GROUPS,
) -> None:
    """
    Escribe calibracion, desglose por environment y metricas agrupadas N/P/K.

    @param {Path} run_dir Directorio del run.
    @param {pd.DataFrame} predictions_df Salida de write_predictions_csv.
    @param {dict[str,int]} class_to_idx Mapeo canonico clase->indice.
    @param {dict[str,str]} npk_groups Mapeo de clases a agrupar.
    """
    calibration = compute_calibration_metrics(predictions_df, class_to_idx)
    (run_dir / "test_calibration.json").write_text(json.dumps(calibration, indent=2))

    if "environment" in predictions_df.columns:
        compute_environment_metrics(predictions_df).to_csv(
            run_dir / "test_by_environment.csv", index=False
        )

    grouped = compute_grouped_metrics(predictions_df, npk_groups)
    (run_dir / "test_grouped_metrics.json").write_text(json.dumps(grouped, indent=2))


def write_summary(run_dir: Path, payload: dict) -> None:
    """
    Persiste summary.json, fuente de verdad del run para los scripts de explicabilidad.

    @param {Path} run_dir Directorio del run.
    @param {dict} payload Configuracion y metricas del run.
    """
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
