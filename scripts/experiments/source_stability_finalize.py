"""Lock validation-selected checkpoints, then evaluate each final split exactly once.

This command never trains. It is intentionally separate from the training batch so
no test/holdout can influence a later checkpoint or a pre-registered seed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.data.dataset import CornDataset
from src.data.identity import unpack_batch
from src.data.preparation import sha256_file
from src.training.evaluation import expected_calibration_error
from src.training.runs import load_validated_run, validate_run_contract

MODEL = "efficientnet_lite0"
SEEDS = (123, 2026)
SCENARIOS = ("baseline", "maize-diseases", "multicrop-disease-maiz")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_validation_run(
    run_dir: Path, splits_dir: Path, scenario: str, seed: int, reference: dict
) -> dict:
    """Pre-test check; reads no final CSV or image."""
    if scenario not in SCENARIOS or seed not in SEEDS:
        raise ValueError("Escenario/seed no pre-registrado")
    lock_path = splits_dir / "manifest.lock.json"
    validated = validate_run_contract(
        run_dir,
        expected_model=MODEL,
        expected_class_to_idx=reference["class_to_idx"],
        expected_preprocessing=reference["preprocessing"],
        splits_dir=splits_dir,
    )
    summary = validated.summary
    for field in ("architecture", "hyperparameters", "preprocessing", "class_to_idx"):
        if summary[field] != reference[field]:
            raise ValueError(f"Drift del baseline congelado: {field}")
    if (
        summary["seed"] != seed
        or summary["split_manifest_sha256"] != sha256_file(lock_path)
        or summary.get("test_used") is not False
        or summary.get("evaluation_mode") != "validation_only"
        or "test" in summary.get("metrics", {})
    ):
        raise ValueError("Run no es validation-only bajo los splits esperados")
    split_lock = _read(lock_path)
    if scenario != "baseline" and (
        split_lock.get("held_out_source") != scenario
        or split_lock.get("protocol") != "loso_baseline_seed42_v1"
    ):
        raise ValueError("Holdout LOSO no corresponde al escenario")
    history = pd.read_csv(run_dir / "train_history.csv")
    if history.empty or history["val_macro_f1"].isna().any():
        raise ValueError("Historial de validación incompleto")
    best = history.loc[history["val_macro_f1"].idxmax()]
    if (
        int(best["epoch"]) != summary["best_epoch"]
        or abs(float(best["val_macro_f1"]) - summary["best_val_macro_f1"]) > 1e-10
    ):
        raise ValueError("Best checkpoint no coincide con validación")
    return summary


def lock_selection(
    run_dir: Path,
    splits_dir: Path,
    scenario: str,
    seed: int,
    reference: dict,
    training_metadata: dict,
) -> dict:
    """Persist immutable selection evidence before any final split is parsed."""
    prefix = "test" if scenario == "baseline" else "holdout"
    selection_path = run_dir / "selection.lock.json"
    if selection_path.exists() or (run_dir / f"{prefix}_predictions.csv").exists():
        raise FileExistsError("Selection lock o evaluación final ya existe")
    summary = verify_validation_run(run_dir, splits_dir, scenario, seed, reference)
    if (
        training_metadata.get("status") != "validation_complete"
        or training_metadata.get("scenario") != scenario
        or training_metadata.get("seed") != seed
        or training_metadata.get("run_id") != summary["run_id"]
        or training_metadata.get("checkpoint_sha256") != summary["checkpoint_sha256"]
    ):
        raise ValueError("Metadatos de entrenamiento incompatibles con el checkpoint")
    split_lock = _read(splits_dir / "manifest.lock.json")
    if scenario == "baseline":
        manifest_hashes = {
            "master_manifest.csv": split_lock["master_manifest_sha256"],
            "train.csv": split_lock["train_sha256"],
            "val.csv": split_lock["val_sha256"],
            "test.csv": split_lock["test_sha256"],
        }
    else:
        manifest_hashes = {
            "master_manifest.csv": split_lock["frozen_sha256"]["master_manifest.csv"],
            **split_lock["derived_sha256"],
        }
    payload = {
        "schema_version": 1,
        "protocol": "source_stability_3seed_v1",
        "scenario": scenario,
        "held_out_source": None if scenario == "baseline" else scenario,
        "seed": seed,
        "run_id": summary["run_id"],
        "selection_criterion": "internal_validation_macro_f1",
        "best_epoch": summary["best_epoch"],
        "best_val_macro_f1": summary["best_val_macro_f1"],
        "validation_macro_f1": summary["best_val_macro_f1"],
        "checkpoint_path": str(run_dir / "best.pth"),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "config_sha256": summary["config_sha256"],
        "summary_sha256": sha256_file(run_dir / "summary.json"),
        "split_lock_sha256": sha256_file(splits_dir / "manifest.lock.json"),
        "manifest_hashes": manifest_hashes,
        "git_commit": training_metadata["git_commit"],
        "git_dirty": training_metadata["git_dirty"],
        "training_completed_at_utc": training_metadata["finished_at_utc"],
        "selection_locked_at_utc": _utc(),
        "test_or_holdout_used_before_lock": False,
        "final_evaluation_count": 0,
    }
    if payload["selection_locked_at_utc"] <= payload["training_completed_at_utc"]:
        raise ValueError("Selection lock debe ser posterior al entrenamiento")
    with selection_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return payload


def evaluate_final(
    run_dir: Path,
    splits_dir: Path,
    scenario: str,
    seed: int,
    batch_size: int = 32,
    workers: int = 16,
    commit_guard: Callable[[], None] | None = None,
) -> dict:
    """A single guarded test/holdout pass with exact sample-id reconciliation."""
    if scenario not in SCENARIOS or seed not in SEEDS:
        raise ValueError("Escenario/seed no pre-registrado")
    prefix = "test" if scenario == "baseline" else "holdout"
    selection_path = run_dir / "selection.lock.json"
    locked = _read(selection_path)
    if (
        locked["scenario"] != scenario
        or locked["seed"] != seed
        or locked["final_evaluation_count"] != 0
    ):
        raise ValueError("Selection lock incompatible")
    for filename, expected in (
        (run_dir / "best.pth", locked["checkpoint_sha256"]),
        (run_dir / "summary.json", locked["summary_sha256"]),
        (splits_dir / "manifest.lock.json", locked["split_lock_sha256"]),
    ):
        if sha256_file(filename) != expected:
            raise ValueError(f"Artefacto modificado tras lock: {filename}")
    split_lock = _read(splits_dir / "manifest.lock.json")
    final_path = splits_dir / f"{prefix}.csv"
    expected_final_sha = (
        split_lock["test_sha256"]
        if prefix == "test"
        else split_lock["derived_sha256"]["holdout.csv"]
    )
    if sha256_file(final_path) != expected_final_sha:
        raise ValueError("Final manifest difiere de su lock")
    guard = run_dir / "final_evaluation.started.json"
    with guard.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "scenario": scenario,
                "seed": seed,
                "evaluation_count": 1,
                "selection_lock_sha256": sha256_file(selection_path),
                "started_at_utc": _utc(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    if commit_guard is not None:
        commit_guard()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_validated_run(
        run_dir / "best.pth", expected_model=MODEL, splits_dir=splits_dir, device=device
    )
    mapping = loaded.summary["class_to_idx"]
    dataset = CornDataset(
        str(final_path), transform=loaded.factory.get_pipeline("test"), class_to_idx=mapping
    )
    manifest = dataset.data_frame
    if scenario != "baseline" and set(manifest["source_id"]) != {scenario}:
        raise ValueError("Holdout contiene fuente distinta")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    inverse = {index: name for name, index in mapping.items()}
    rows = []
    with torch.inference_mode():
        for batch in loader:
            images, targets, sample_ids = unpack_batch(batch)
            if sample_ids is None:
                raise ValueError("Batch sin sample_id")
            scores = loaded.model(images.to(device)).softmax(dim=1)
            confidence, predicted = scores.max(dim=1)
            rows.extend(
                {
                    "sample_id": sid,
                    "label": inverse[label],
                    "pred_label": inverse[prediction],
                    "pred_prob": prob,
                }
                for sid, label, prediction, prob in zip(
                    sample_ids,
                    targets.tolist(),
                    predicted.cpu().tolist(),
                    confidence.cpu().tolist(),
                    strict=True,
                )
            )
    inferred = pd.DataFrame(rows)
    if inferred["sample_id"].duplicated().any() or set(inferred["sample_id"]) != set(
        manifest["sample_id"]
    ):
        raise ValueError("Evaluación final incompleta o duplicada")
    columns = [
        c
        for c in ("sample_id", "image_path", "source_id", "historical_split")
        if c in manifest.columns
    ]
    predictions = manifest[columns].merge(inferred, on="sample_id", validate="one_to_one")
    if predictions["label"].tolist() != manifest["label"].tolist():
        raise ValueError("Etiquetas inferidas no coinciden por sample_id")
    truth, predicted = predictions["label"], predictions["pred_label"]
    present = sorted(truth.unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=present, zero_division=0
    )
    class_metrics = pd.DataFrame(
        {"class": present, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    classes = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    matrix = confusion_matrix(truth, predicted, labels=classes)
    correct = truth.eq(predicted)
    confidence = predictions["pred_prob"].astype(float)
    metrics = {
        "protocol": "source_stability_3seed_v1",
        "scenario": scenario,
        "seed": seed,
        "held_out_source": None if scenario == "baseline" else scenario,
        "run_id": locked["run_id"],
        "evaluation_count": 1,
        "selection_lock_sha256": sha256_file(selection_path),
        "checkpoint_sha256": locked["checkpoint_sha256"],
        "final_manifest_sha256": expected_final_sha,
        "finished_at_utc": _utc(),
        "samples": len(predictions),
        "classes_present": present,
        "macro_f1_definition": "unweighted mean over classes present in ground truth",
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=present, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predicted, labels=present, average="weighted", zero_division=0)
        ),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "errors": int((~correct).sum()),
        "error_rate": float((~correct).mean()),
        "ece_15_bins": float(expected_calibration_error(confidence.tolist(), correct.tolist())),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "mean_confidence_errors": float(confidence[~correct].mean()) if (~correct).any() else None,
    }
    predictions.to_csv(run_dir / f"{prefix}_predictions.csv", index=False)
    class_metrics.to_csv(run_dir / f"{prefix}_class_metrics.csv", index=False)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        run_dir / f"{prefix}_confusion_matrix.csv"
    )
    pd.DataFrame(
        matrix / matrix.sum(axis=1, keepdims=True).clip(min=1), index=classes, columns=classes
    ).to_csv(run_dir / f"{prefix}_confusion_normalized.csv")
    with (run_dir / f"{prefix}_metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return metrics
