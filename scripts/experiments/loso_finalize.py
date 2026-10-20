"""Lock a completed LOSO checkpoint, then evaluate its unseen-source holdout.

The holdout CSV is not read until the selection lock has been written and checked.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from scripts.experiments.prepare_loso_baseline import SOURCES, sha256
from src.data.dataset import CornDataset
from src.data.identity import unpack_batch
from src.training.evaluation import expected_calibration_error
from src.training.runs import load_validated_run

BASELINE_CONFIG_SHA256 = "53cc091e505057f6751a7b6151f403e03f830fac3c0966df9ba51e2df2814e8b"


def timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def lock_selection(run_dir: Path, splits_dir: Path, source: str) -> dict:
    """Check the completed validation-only run before any holdout manifest is opened."""
    if source not in SOURCES:
        raise ValueError("Fuente no predefinida")
    summary_path = run_dir / "summary.json"
    checkpoint = run_dir / "best.pth"
    lock_path = splits_dir / "manifest.lock.json"
    selection_path = run_dir / "selection.lock.json"
    if selection_path.exists():
        raise FileExistsError("El selection lock ya existe; no duplicar evaluación")
    if (run_dir / "holdout_predictions.csv").exists():
        raise ValueError("Hay predicciones holdout antes del selection lock")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if split_lock["held_out_source"] != source:
        raise ValueError("El lock de splits corresponde a otra fuente")
    if summary["model"] != "efficientnet_lite0" or summary["seed"] != 42:
        raise ValueError("Modelo o semilla distintos del protocolo")
    if summary["config_sha256"] != BASELINE_CONFIG_SHA256:
        raise ValueError("La configuración efectiva difiere del baseline congelado")
    if summary["split_manifest_sha256"] != sha256(lock_path):
        raise ValueError("El run no usó estos splits derivados")
    if summary.get("test_used") is not False or summary.get("evaluation_mode") != "validation_only":
        raise ValueError("El run no es validation-only")
    if "test" in summary or "test" in summary["metrics"]:
        raise ValueError("El run contiene evaluación de test antes del lock")
    if sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("SHA-256 del checkpoint no coincide con el contrato")
    history = pd.read_csv(run_dir / "train_history.csv")
    if history.empty or history["val_macro_f1"].isna().any():
        raise ValueError("Historial de validation incompleto")
    best = history.loc[history["val_macro_f1"].idxmax()]
    if int(best["epoch"]) != summary["best_epoch"] or abs(
        float(best["val_macro_f1"]) - summary["best_val_macro_f1"]
    ) > 1e-10:
        raise ValueError("Best epoch/validation no coinciden con el historial")
    locked = {
        "schema_version": 1,
        "protocol": "loso_baseline_seed42_v1",
        "held_out_source": source,
        "run_id": summary["run_id"],
        "selection_criterion": "internal_validation_macro_f1",
        "best_epoch": summary["best_epoch"],
        "best_val_macro_f1": summary["best_val_macro_f1"],
        "checkpoint_sha256": sha256(checkpoint),
        "summary_sha256": sha256(summary_path),
        "split_lock_sha256": sha256(lock_path),
        "checkpoint_selected_at_utc": timestamp(checkpoint),
        "training_completed_at_utc": timestamp(summary_path),
        "selection_locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_evaluated": False,
    }
    if datetime.fromisoformat(locked["selection_locked_at_utc"]) <= datetime.fromisoformat(
        locked["training_completed_at_utc"]
    ):
        raise ValueError("El lock no es posterior al fin del entrenamiento")
    selection_path.write_text(json.dumps(locked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return locked


def evaluate_holdout(run_dir: Path, splits_dir: Path, source: str,
                     batch_size: int = 32, workers: int = 16) -> dict:
    """Evaluate only after a validated selection lock is persisted."""
    selection_path = run_dir / "selection.lock.json"
    locked = json.loads(selection_path.read_text(encoding="utf-8"))
    if locked["held_out_source"] != source or locked["holdout_evaluated"]:
        raise ValueError("Selection lock incorrecto o evaluación repetida")
    if (run_dir / "holdout_predictions.csv").exists():
        raise FileExistsError("Ya existen predicciones holdout")
    checkpoint = run_dir / "best.pth"
    if sha256(checkpoint) != locked["checkpoint_sha256"]:
        raise ValueError("Checkpoint cambió después del selection lock")
    if sha256(run_dir / "summary.json") != locked["summary_sha256"]:
        raise ValueError("Summary cambió después del selection lock")
    split_lock_path = splits_dir / "manifest.lock.json"
    if sha256(split_lock_path) != locked["split_lock_sha256"]:
        raise ValueError("Splits cambiaron después del selection lock")
    split_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
    holdout_path = splits_dir / "holdout.csv"
    if sha256(holdout_path) != split_lock["derived_sha256"]["holdout.csv"]:
        raise ValueError("Holdout CSV no coincide con su lock")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_validated_run(
        checkpoint, expected_model="efficientnet_lite0", splits_dir=splits_dir, device=device
    )
    mapping = loaded.summary["class_to_idx"]
    dataset = CornDataset(
        str(holdout_path), transform=loaded.factory.get_pipeline("test"), class_to_idx=mapping
    )
    if set(dataset.data_frame["source_id"]) != {source}:
        raise ValueError("Holdout contiene otra fuente")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    inverse = {index: name for name, index in mapping.items()}
    with torch.inference_mode():
        for batch in loader:
            images, targets, sample_ids = unpack_batch(batch)
            if sample_ids is None:
                raise ValueError("El batch no contiene sample_id")
            scores = loaded.model(images.to(device)).softmax(dim=1)
            confidence, predicted = scores.max(dim=1)
            for sid, label, prediction, prob in zip(
                sample_ids, targets.tolist(), predicted.cpu().tolist(),
                confidence.cpu().tolist(), strict=True
            ):
                rows.append({"sample_id": sid, "label": inverse[label],
                             "pred_label": inverse[prediction], "pred_prob": prob})
    predictions = pd.DataFrame(rows)
    manifest = dataset.data_frame
    if predictions["sample_id"].duplicated().any() or set(predictions["sample_id"]) != set(
        manifest["sample_id"]
    ):
        raise ValueError("Predicciones holdout incompletas o duplicadas")
    predictions = manifest[["sample_id", "image_path", "source_id", "historical_split"]].merge(
        predictions, on="sample_id", validate="one_to_one"
    )
    if predictions["label"].tolist() != manifest["label"].tolist():
        raise ValueError("Etiquetas inferidas contradicen el holdout")
    present = sorted(predictions["label"].unique())
    truth, predicted = predictions["label"], predictions["pred_label"]
    correct = truth.eq(predicted)
    confidence = predictions["pred_prob"].astype(float)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=present, zero_division=0
    )
    class_metrics = pd.DataFrame({"class": present, "precision": precision,
                                  "recall": recall, "f1": f1, "support": support})
    classes = [name for name, _index in sorted(mapping.items(), key=lambda item: item[1])]
    matrix = confusion_matrix(truth, predicted, labels=classes)
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        run_dir / "holdout_confusion_matrix.csv"
    )
    pd.DataFrame(normalized, index=classes, columns=classes).to_csv(
        run_dir / "holdout_confusion_normalized.csv"
    )
    predictions.to_csv(run_dir / "holdout_predictions.csv", index=False)
    class_metrics.to_csv(run_dir / "holdout_class_metrics.csv", index=False)
    metrics = {
        "protocol": "loso_baseline_seed42_v1",
        "held_out_source": source,
        "run_id": locked["run_id"],
        "selection_lock_sha256": sha256(selection_path),
        "checkpoint_sha256": locked["checkpoint_sha256"],
        "holdout_sha256": sha256(holdout_path),
        "holdout_started_at_utc": started,
        "holdout_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(predictions),
        "classes_present": present,
        "macro_f1_definition": "unweighted mean over classes present in ground truth",
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=present,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=present,
                                      average="weighted", zero_division=0)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "errors": int((~correct).sum()),
        "error_rate": float((~correct).mean()),
        "mean_confidence_correct": float(confidence[correct].mean())
        if correct.any() else None,
        "mean_confidence_errors": float(confidence[~correct].mean())
        if (~correct).any() else None,
        "ece_15_bins": float(expected_calibration_error(confidence.tolist(), correct.tolist())),
        "npk_present": sorted(set(present) & {"nitrogen_deficiency", "phosphorus_deficiency",
                                              "potassium_deficiency"}),
    }
    (run_dir / "holdout_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    locked = lock_selection(args.run_dir, args.splits_dir, args.source)
    print(json.dumps(locked, indent=2))
    result = evaluate_holdout(args.run_dir, args.splits_dir, args.source,
                              workers=args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
