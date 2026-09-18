"""Persistencia de predicciones por imagen.

Las métricas agregadas no se pueden auditar ni desagregar después. Guardando la
predicción de cada imagen junto a su ruta y su procedencia, cualquier cifra publicada
se recomputa desde el archivo y se puede partir por fuente sin volver a entrenar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data.identity import sample_id_for_path
from src.data.provenance import source_from_path


def write_per_image_predictions(
    destination: Path,
    sample_ids: Sequence[str],
    image_paths: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    idx_to_class: dict[int, str] | None = None,
    confidence: Sequence[float] | None = None,
) -> Path:
    """Escribe un CSV con una fila por imagen evaluada.

    ``sample_ids`` e ``image_paths`` deben viajar en el orden efectivo del DataLoader.
    Cada par se valida contra la identidad canónica para impedir atribuciones por
    posición del manifiesto original.

    @param {Path} destination Ruta del CSV a escribir.
    @param {Sequence[str]} sample_ids IDs observados durante inferencia.
    @param {Sequence[str]} image_paths Rutas relativas, en el orden del manifiesto.
    @param {Sequence[int]} y_true Índices de clase verdaderos.
    @param {Sequence[int]} y_pred Índices de clase predichos.
    @param {dict[int, str] | None} idx_to_class Mapeo para añadir columnas legibles.
    @param {Sequence[float] | None} confidence Probabilidad de la clase predicha.
    @returns {Path} Ruta escrita.
    """
    if not (len(sample_ids) == len(image_paths) == len(y_true) == len(y_pred)):
        raise ValueError(
            "Las predicciones no se alinean con el manifiesto: "
            f"{len(sample_ids)} IDs, {len(image_paths)} rutas, "
            f"{len(y_true)} verdaderos y {len(y_pred)} predichos."
        )
    expected_ids = [sample_id_for_path(path) for path in image_paths]
    if list(sample_ids) != expected_ids:
        raise ValueError("sample_id no corresponde a image_path en el orden de inferencia")

    frame = pd.DataFrame(
        {
            "sample_id": list(sample_ids),
            "image_path": list(image_paths),
            "source_id": [source_from_path(path) for path in image_paths],
            "y_true": list(y_true),
            "y_pred": list(y_pred),
        }
    )
    if idx_to_class:
        frame["true_label"] = frame["y_true"].map(idx_to_class)
        frame["pred_label"] = frame["y_pred"].map(idx_to_class)
    if confidence is not None:
        if len(confidence) != len(frame):
            raise ValueError("La confianza no tiene una entrada por imagen.")
        frame["confidence"] = list(confidence)

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
