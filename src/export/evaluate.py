"""Evaluación de un modelo ya exportado (.onnx / .tflite) sobre el split de test completo.

Complementa la validación de paridad de `src/export/parity.py`, que compara PyTorch contra
el archivo exportado sobre una **muestra** (30 imágenes por defecto) y solo mira diferencias
numéricas. Aquí se corre el split de test entero y se miden accuracy y macro-F1 reales del
artefacto que se va a embarcar. La distinción importa sobre todo con `--quantize int8`: una
cuantización puede pasar una tolerancia laxa de paridad y aun así perder puntos de macro-F1
en las clases minoritarias.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data.identity import (
    align_manifest_to_sample_ids,
    dataset_manifest,
    unpack_batch,
)
from src.export.runtime import load_exported_runner

logger = logging.getLogger(__name__)


@dataclass
class ExportEvaluation:
    format: str
    model_path: str
    quantize: str | None
    n_samples: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    per_class_support: dict[str, int]
    by_environment: dict[str, dict[str, float]]
    torch_accuracy: float | None = None
    torch_macro_f1: float | None = None
    agreement_rate: float | None = None
    accuracy_delta: float | None = None
    macro_f1_delta: float | None = None
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    warnings: list[str] = field(default_factory=list)


def _predict_exported(
    runner, test_loader: DataLoader
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """
    Corre el modelo exportado sobre todo el loader.

    @param {ExportedRunner} runner Callable del modelo exportado.
    @param {DataLoader} test_loader Loader del split de test.
    @returns {tuple} (predicciones, confianzas, labels) como arrays de numpy.
    """
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    identified: bool | None = None
    for batch in test_loader:
        images, batch_labels, batch_sample_ids = unpack_batch(batch)
        batch_identified = batch_sample_ids is not None
        if identified is not None and identified != batch_identified:
            raise ValueError("El loader mezcló batches con y sin sample_id")
        identified = batch_identified
        if batch_sample_ids is not None:
            sample_ids.extend(batch_sample_ids)
        logits = runner(images.cpu().numpy())
        probs = torch.from_numpy(np.asarray(logits, dtype=np.float32)).softmax(dim=1).numpy()
        predictions.append(probs.argmax(axis=1))
        confidences.append(probs.max(axis=1))
        labels.append(batch_labels.numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(confidences),
        np.concatenate(labels),
        sample_ids if identified else None,
    )


def _predict_torch(
    model: torch.nn.Module, test_loader: DataLoader, device
) -> tuple[np.ndarray, list[str] | None]:
    model.eval()
    predictions: list[np.ndarray] = []
    sample_ids: list[str] = []
    identified: bool | None = None
    with torch.no_grad():
        for batch in test_loader:
            images, _, batch_sample_ids = unpack_batch(batch)
            batch_identified = batch_sample_ids is not None
            if identified is not None and identified != batch_identified:
                raise ValueError("El loader mezcló batches con y sin sample_id")
            identified = batch_identified
            if batch_sample_ids is not None:
                sample_ids.extend(batch_sample_ids)
            logits = model(images.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions), sample_ids if identified else None


def _environment_breakdown(
    frame: pd.DataFrame, idx_to_class: dict[int, str]
) -> dict[str, dict[str, float]]:
    """Accuracy y macro-F1 por entorno de captura (lab / real)."""
    if "environment" not in frame.columns:
        return {}
    breakdown: dict[str, dict[str, float]] = {}
    for environment, group in frame.groupby("environment"):
        breakdown[str(environment)] = {
            "n": int(len(group)),
            "accuracy": float(accuracy_score(group["label_idx"], group["pred_idx"])),
            "macro_f1": float(
                f1_score(
                    group["label_idx"], group["pred_idx"], average="macro", zero_division=0
                )
            ),
        }
    return breakdown


def evaluate_exported_model(
    model_path: Path,
    format_name: str,
    test_loader: DataLoader,
    idx_to_class: dict[int, str],
    *,
    torch_model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    quantize: str | None = None,
    environments: pd.Series | None = None,
) -> tuple[ExportEvaluation, pd.DataFrame]:
    """
    Evalúa el archivo exportado sobre el split de test completo.

    @param {Path} model_path Ruta al .onnx o .tflite.
    @param {str} format_name "onnx" o "tflite".
    @param {DataLoader} test_loader Loader del split de test (pipeline 'test', sin shuffle).
    @param {dict[int,str]} idx_to_class Mapeo índice->clase del run.
    @param {torch.nn.Module|None} torch_model Modelo original, para comparar; None lo omite.
    @param {torch.device|None} device Dispositivo del modelo PyTorch.
    @param {str|None} quantize Cuantización aplicada al exportar, solo para el reporte.
    @param {pd.Series|None} environments Entorno por imagen, alineado al orden del loader.
    @returns {tuple[ExportEvaluation, pd.DataFrame]} Métricas y predicciones por imagen.
    """
    runner = load_exported_runner(model_path, format_name)
    predictions, confidences, labels, sample_ids = _predict_exported(runner, test_loader)

    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    label_indices = sorted(idx_to_class)

    accuracy = float(accuracy_score(labels, predictions))
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    per_class = f1_score(
        labels, predictions, average=None, labels=label_indices, zero_division=0
    )
    per_class_f1 = {idx_to_class[i]: float(per_class[n]) for n, i in enumerate(label_indices)}
    per_class_support = {
        idx_to_class[i]: int((labels == i).sum()) for i in label_indices
    }

    frame = pd.DataFrame(
        {
            "label_idx": labels,
            "pred_idx": predictions,
            "label": [idx_to_class[int(i)] for i in labels],
            "true_label": [idx_to_class[int(i)] for i in labels],
            "pred_label": [idx_to_class[int(i)] for i in predictions],
            "confidence": confidences,
        }
    )
    manifest = dataset_manifest(test_loader.dataset)
    if sample_ids is not None:
        if manifest is None:
            raise ValueError("El loader tiene sample_id pero su dataset no expone manifiesto")
        aligned_manifest = align_manifest_to_sample_ids(manifest, sample_ids)
        if aligned_manifest["label"].tolist() != frame["label"].tolist():
            raise ValueError("Las etiquetas exportadas no coinciden con sample_id")
        frame.insert(0, "image_path", aligned_manifest["image_path"].tolist())
        frame.insert(0, "sample_id", sample_ids)
        if "environment" in aligned_manifest.columns:
            frame["environment"] = aligned_manifest["environment"].tolist()
    if (
        "environment" not in frame.columns
        and environments is not None
        and len(environments) == len(frame)
    ):
        frame["environment"] = environments.to_numpy()

    warnings: list[str] = []
    torch_accuracy = torch_macro_f1 = agreement_rate = None
    accuracy_delta = macro_f1_delta = None
    if torch_model is not None:
        torch_predictions, torch_sample_ids = _predict_torch(torch_model, test_loader, device)
        if sample_ids != torch_sample_ids:
            raise ValueError("PyTorch y el modelo exportado evaluaron sample_id en distinto orden")
        torch_accuracy = float(accuracy_score(labels, torch_predictions))
        torch_macro_f1 = float(
            f1_score(labels, torch_predictions, average="macro", zero_division=0)
        )
        agreement_rate = float((torch_predictions == predictions).mean())
        accuracy_delta = accuracy - torch_accuracy
        macro_f1_delta = macro_f1 - torch_macro_f1
        frame["torch_pred_label"] = [idx_to_class[int(i)] for i in torch_predictions]
        if macro_f1_delta < -0.01:
            warnings.append(
                f"macro-F1 del modelo exportado cae {abs(macro_f1_delta):.4f} vs PyTorch"
            )

    evaluation = ExportEvaluation(
        format=format_name,
        model_path=str(model_path),
        quantize=quantize,
        n_samples=int(len(labels)),
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        per_class_support=per_class_support,
        by_environment=_environment_breakdown(frame, idx_to_class),
        torch_accuracy=torch_accuracy,
        torch_macro_f1=torch_macro_f1,
        agreement_rate=agreement_rate,
        accuracy_delta=accuracy_delta,
        macro_f1_delta=macro_f1_delta,
        warnings=warnings,
    )
    _ = class_names  # orden canónico ya reflejado en per_class_f1
    return evaluation, frame


def write_evaluation(run_dir: Path, evaluation: ExportEvaluation, frame: pd.DataFrame) -> Path:
    """
    Persiste <run_dir>/export/eval_<formato>.json y eval_<formato>_predictions.csv.

    @param {Path} run_dir Directorio del run.
    @param {ExportEvaluation} evaluation Métricas a serializar.
    @param {pd.DataFrame} frame Predicciones por imagen.
    @returns {Path} Ruta del JSON escrito.
    """
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    suffix = evaluation.format
    if evaluation.quantize:
        suffix = f"{suffix}_{evaluation.quantize}"

    json_path = export_dir / f"eval_{suffix}.json"
    json_path.write_text(json.dumps(asdict(evaluation), indent=2))
    frame.to_csv(export_dir / f"eval_{suffix}_predictions.csv", index=False)
    return json_path
