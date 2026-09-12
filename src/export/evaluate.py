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

from src.data.identity import align_predictions, dataset_frame, identified_batches
from src.export.runtime import load_exported_runner
from src.provenance import atomic_json, contract_hash, sha256_file

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
    by_environment: dict[str, dict]
    torch_accuracy: float | None = None
    torch_macro_f1: float | None = None
    agreement_rate: float | None = None
    accuracy_delta: float | None = None
    macro_f1_delta: float | None = None
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    warnings: list[str] = field(default_factory=list)
    model_sha256: str | None = None
    sample_ids_hash: str | None = None
    expected_samples: int | None = None
    evaluated_split_sha256: str | None = None


def _predict_exported(runner, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Corre el modelo exportado sobre todo el loader.

    @param {ExportedRunner} runner Callable del modelo exportado.
    @param {DataLoader} test_loader Loader del split de test.
    @returns {tuple} (predicciones, confianzas, labels) como arrays de numpy.
    """
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in test_loader:
        logits = runner(images.cpu().numpy())
        probs = torch.from_numpy(np.asarray(logits, dtype=np.float32)).softmax(dim=1).numpy()
        predictions.append(probs.argmax(axis=1))
        confidences.append(probs.max(axis=1))
        labels.append(batch_labels.numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(confidences),
        np.concatenate(labels),
    )


def _predict_torch(model: torch.nn.Module, test_loader: DataLoader, device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in test_loader:
            logits = model(images.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def _environment_breakdown(frame: pd.DataFrame, idx_to_class: dict[int, str]) -> dict[str, dict]:
    """Macro-F1 on supported true classes, not a prediction-dependent class universe."""
    if "environment" not in frame.columns:
        return {}
    breakdown: dict[str, dict] = {}
    for environment, group in frame.groupby("environment"):
        supported = sorted(group["label_idx"].unique())
        breakdown[str(environment)] = {
            "n": int(len(group)),
            "accuracy": float(accuracy_score(group["label_idx"], group["pred_idx"])),
            "macro_f1": float(
                f1_score(
                    group["label_idx"],
                    group["pred_idx"],
                    labels=supported,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1_policy": "supported_true_classes",
            "supported_classes": [idx_to_class[int(i)] for i in supported],
            "per_class_support": {
                name: int((group.label_idx == index).sum()) for index, name in idx_to_class.items()
            },
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
    predictions, confidences, labels, ids, torch_predictions = [], [], [], [], []
    if torch_model is not None:
        torch_model.eval()
    # Both runtimes consume the identical tensor in a single pass, even with shuffle.
    with torch.no_grad():
        for images, batch_labels, batch_ids in identified_batches(test_loader):
            logits = np.asarray(runner(images.cpu().numpy()), dtype=np.float32)
            if logits.shape != (len(images), len(idx_to_class)) or not np.isfinite(logits).all():
                raise ValueError("Exported logits have invalid shape/values")
            probs = torch.from_numpy(logits).softmax(dim=1).numpy()
            predictions.extend(probs.argmax(axis=1))
            confidences.extend(probs.max(axis=1))
            labels.extend(batch_labels.tolist())
            if batch_ids is not None:
                ids.extend(batch_ids)
            if torch_model is not None:
                output = torch_model(images.to(device or "cpu"))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                torch_predictions.extend(output.argmax(dim=1).cpu().tolist())
    if not labels:
        raise ValueError("Cannot evaluate an empty export dataset")
    predictions, labels = np.asarray(predictions), np.asarray(labels)

    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    label_indices = sorted(idx_to_class)

    accuracy = float(accuracy_score(labels, predictions))
    macro_f1 = float(
        f1_score(labels, predictions, labels=label_indices, average="macro", zero_division=0)
    )
    per_class = f1_score(labels, predictions, average=None, labels=label_indices, zero_division=0)
    per_class_f1 = {idx_to_class[i]: float(per_class[n]) for n, i in enumerate(label_indices)}
    per_class_support = {idx_to_class[i]: int((labels == i).sum()) for i in label_indices}

    frame = pd.DataFrame(
        {
            "label_idx": labels,
            "pred_idx": predictions,
            "label": [idx_to_class[int(i)] for i in labels],
            "pred_label": [idx_to_class[int(i)] for i in predictions],
            "confidence": confidences,
        }
    )
    metadata = dataset_frame(test_loader.dataset)
    if metadata is not None:
        aligned = align_predictions(metadata, ids)
        frame["sample_id"] = ids
        frame["image_path"] = aligned["image_path"].to_numpy()
        if "environment" in aligned:
            frame["environment"] = aligned["environment"].to_numpy()
    elif environments is not None:
        raise ValueError("Environment attribution requires sample IDs, not positional Series")

    warnings: list[str] = []
    torch_accuracy = torch_macro_f1 = agreement_rate = None
    accuracy_delta = macro_f1_delta = None
    if torch_model is not None:
        torch_predictions = np.asarray(torch_predictions)
        torch_accuracy = float(accuracy_score(labels, torch_predictions))
        torch_macro_f1 = float(
            f1_score(
                labels, torch_predictions, labels=label_indices, average="macro", zero_division=0
            )
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
        model_sha256=sha256_file(model_path) if model_path.is_file() else None,
        sample_ids_hash=contract_hash(ids) if ids else None,
        expected_samples=len(test_loader.dataset),
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
    predictions_path = export_dir / f"eval_{suffix}_predictions.csv"
    frame.to_csv(predictions_path, index=False)
    payload = asdict(evaluation)
    payload["predictions_sha256"] = sha256_file(predictions_path)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        training = json.loads(summary_path.read_text())
        payload.update(
            {
                "run_id": run_dir.name,
                "checkpoint_sha256": training.get("checkpoint_sha256"),
                "preprocessing_id": contract_hash(training["preprocessing"]),
                "split_hashes": training.get("split_hashes", {}),
            }
        )
    atomic_json(json_path, payload)
    return json_path
