"""Escritura de los artefactos de un run de entrenamiento.

Compartido por baselines y pipeline principal para que ambos produzcan el mismo
esquema de salida y los subcomandos `fidelity`/`errors` de scripts/pipeline/explain.py
puedan leer los dos indistintamente.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data.identity import align_manifest_to_sample_ids, dataset_manifest, sample_ids_from
from src.data.preparation import atomic_write_json
from src.data.provenance import provenance_from_path
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
    labels: list[int],
    predictions: list[int],
    probs: list[float],
    *,
    filename: str = "predictions.csv",
) -> pd.DataFrame:
    """
    Escribe predictions.csv con una fila por imagen de test.

    @param {Path} run_dir Directorio del run.
    @param {CornDataset} test_dataset Dataset de test, fuente de rutas y etiquetas.
    @param {dict[int,str]} idx_to_class Mapeo indice->clase.
    @param {list[int]} labels Etiquetas reales en el orden efectivo de inferencia.
    @param {list[int]} predictions Predicciones codificadas.
    @param {list[float]} probs Confianza de la clase predicha.
    @returns {pd.DataFrame} El propio dataframe escrito.
    """
    if not (len(labels) == len(predictions) == len(probs)):
        raise ValueError("Labels, predicciones y probabilidades deben tener la misma longitud")
    sample_ids = sample_ids_from(predictions, field_name="predicciones")
    label_sample_ids = sample_ids_from(labels, field_name="etiquetas")
    if sample_ids != label_sample_ids:
        raise ValueError("Las etiquetas y predicciones no conservan el mismo orden de sample_id")

    source_manifest = dataset_manifest(test_dataset)
    if source_manifest is None:
        raise ValueError("El dataset de test no expone un manifiesto con sample_id")
    manifest = align_manifest_to_sample_ids(source_manifest, sample_ids)
    true_labels = [idx_to_class[value] for value in labels]
    if manifest["label"].tolist() != true_labels:
        raise ValueError("Las etiquetas inferidas no coinciden con el manifiesto por sample_id")

    frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_path": manifest["image_path"].tolist(),
            "label": true_labels,
            "true_label": true_labels,
            "pred_label": [idx_to_class[p] for p in predictions],
            "pred_prob": probs,
        }
    )
    if "environment" in manifest.columns:
        frame["environment"] = manifest["environment"].tolist()
    frame["source_id"] = [provenance_from_path(path) for path in frame["image_path"]]
    frame.to_csv(run_dir / filename, index=False)
    return frame


def write_validation_outputs(run_dir, dataset, idx_to_class, labels, predictions, probs):
    """Diagnóstico del checkpoint elegido, con las métricas e identidad existentes."""
    frame = write_predictions_csv(
        run_dir,
        dataset,
        idx_to_class,
        labels,
        predictions,
        probs,
        filename="validation_predictions.csv",
    )
    mapping = {name: index for index, name in idx_to_class.items()}
    diagnostics = {
        "calibration": compute_calibration_metrics(frame, mapping),
        "per_class": classification_report(
            frame["label"],
            frame["pred_label"],
            labels=list(mapping),
            output_dict=True,
            zero_division=0,
        ),
        "npk_grouped": compute_grouped_metrics(frame, NPK_GROUPS),
    }
    atomic_write_json(run_dir / "validation_diagnostics.json", diagnostics)
    return diagnostics


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

    # La media agregada oculta que una fuente de laboratorio pueda dar 0.99 mientras
    # una de campo da 0.58. Para un destino movil la cifra que importa es la peor
    # fuente, no el promedio.
    if "source_id" in predictions_df.columns:
        compute_environment_metrics(predictions_df, group_column="source_id").to_csv(
            run_dir / "test_by_source.csv", index=False
        )

    grouped = compute_grouped_metrics(predictions_df, npk_groups)
    (run_dir / "test_grouped_metrics.json").write_text(json.dumps(grouped, indent=2))


def write_summary(run_dir: Path, payload: dict) -> None:
    """
    Persiste summary.json, fuente de verdad del run para los scripts de explicabilidad.

    @param {Path} run_dir Directorio del run.
    @param {dict} payload Configuracion y metricas del run.
    """
    if payload.get("schema_version") is not None:
        from src.training.runs import write_run_contract

        write_run_contract(run_dir, payload)
    else:
        # Compatibilidad para utilidades/tests que escriben JSON auxiliares; los nuevos
        # runs pasan siempre por write_run_contract y su esquema estricto.
        atomic_write_json(run_dir / "summary.json", payload)
