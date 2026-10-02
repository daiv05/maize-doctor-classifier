"""Lock all historical ensemble members before one multicrop LOSO inference per seed."""

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

from scripts.modal.ensemble_loso_train import (
    LOCK_SHA256,
    MODELS,
    ROOT,
    SEEDS,
    SOURCE,
    SPLITS,
    sha256_file,
    verify_run,
    verify_splits,
    write_json,
)
from src.data.dataset import CornDataset
from src.data.identity import unpack_batch
from src.training.evaluation import expected_calibration_error
from src.training.runs import load_validated_run, validate_run_contract

MEMBERS = ("efficientnet_lite0", *MODELS)
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
LITE_ROOT = Path("/outputs/loso/efficientnet_lite0_baseline") / SOURCE
HOLDOUT_SHA256 = "d36f938bdb6db7505722e5b7cf6458adc5568f3a94525884943e47eb0a8b0688"
HISTORICAL_ENSEMBLE_SHA256 = "dcafbb4c21e2d69646b8a04cd1bbf425474b615f5eb6b44d594abd469b6f90c2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def member_run(seed: int, model: str) -> Path:
    if seed not in SEEDS or model not in MEMBERS:
        raise ValueError("Seed o modelo fuera del protocolo")
    if model == "efficientnet_lite0":
        roots = LITE_ROOT / f"seed_{seed}" / "runs" / model
    else:
        roots = ROOT / f"seed_{seed}" / model / "runs" / model
    runs = [path for path in roots.iterdir() if path.is_dir() and (path / "summary.json").is_file()]
    if len(runs) != 1:
        raise ValueError(f"Se esperaba un run completo en {roots}; hay {len(runs)}")
    return runs[0]


def validated_member(seed: int, model: str, historical: dict) -> dict:
    run = member_run(seed, model)
    if model == "efficientnet_lite0":
        summary = validate_run_contract(run, expected_model=model, splits_dir=SPLITS).summary
        old_lock_path = run / "selection.lock.json"
        old_lock = json.loads(old_lock_path.read_text(encoding="utf-8"))
        if (
            old_lock["checkpoint_sha256"] != summary["checkpoint_sha256"]
            or old_lock["split_lock_sha256"] != LOCK_SHA256
        ):
            raise ValueError("Lock previo Lite0 incompatible")
    else:
        summary = verify_run(run, model, seed, historical)
    if (
        summary["seed"] != seed
        or summary["split_manifest_sha256"] != LOCK_SHA256
        or sha256_file(run / "best.pth") != summary["checkpoint_sha256"]
        or summary.get("test_used") is not False
        or summary.get("evaluation_mode") != "validation_only"
    ):
        raise ValueError("Checkpoint o split incompatible")
    history = pd.read_csv(run / "train_history.csv")
    best = history.loc[history["val_macro_f1"].idxmax()]
    if (
        int(best["epoch"]) != summary["best_epoch"]
        or abs(float(best["val_macro_f1"]) - summary["best_val_macro_f1"]) > 1e-10
    ):
        raise ValueError("Selección no coincide con validación")
    return summary


def lock_all(historical: dict) -> list[Path]:
    """No parsea ni infiere sobre holdout; registra 6 locks nuevos y 3 ensembles."""
    if sha256_file(SPLITS / "manifest.lock.json") != LOCK_SHA256:
        raise ValueError("Manifest LOSO alterado")
    split_lock = verify_splits(SOURCE, SPLITS)
    for name in ("train.csv", "val.csv"):
        if SOURCE in set(pd.read_csv(SPLITS / name, usecols=["source_id"])["source_id"]):
            raise ValueError(f"Fuente holdout presente en {name}")
    written: list[Path] = []
    for seed in SEEDS:
        members = []
        lite_summary = validated_member(seed, "efficientnet_lite0", historical)
        for model in MEMBERS:
            run = member_run(seed, model)
            summary = (
                lite_summary
                if model == "efficientnet_lite0"
                else validated_member(seed, model, historical[model])
            )
            if (
                summary["class_to_idx"] != lite_summary["class_to_idx"]
                or summary["preprocessing"] != lite_summary["preprocessing"]
            ):
                raise ValueError("Clases o preprocesamiento distintos")
            selection_path = run / "selection.lock.json"
            if model != "efficientnet_lite0":
                metadata = json.loads(
                    (run.parents[2] / "experiment_metadata.json").read_text(encoding="utf-8")
                )
                if (
                    metadata["status"] != "validation_complete"
                    or metadata["model"] != model
                    or metadata["seed"] != seed
                    or metadata["run_id"] != summary["run_id"]
                    or metadata["checkpoint_sha256"] != summary["checkpoint_sha256"]
                    or metadata["split_lock_sha256"] != LOCK_SHA256
                ):
                    raise ValueError("Metadatos de entrenamiento incompatibles")
                payload = {
                    "schema_version": 1,
                    "protocol": "historical_ensemble_multicrop_loso_v1",
                    "seed": seed,
                    "model": model,
                    "held_out_source": SOURCE,
                    "selection_criterion": "internal_validation_macro_f1",
                    "best_epoch": summary["best_epoch"],
                    "best_val_macro_f1": summary["best_val_macro_f1"],
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "summary_sha256": sha256_file(run / "summary.json"),
                    "split_lock_sha256": LOCK_SHA256,
                    "manifest_hashes": split_lock["derived_sha256"],
                    "config_sha256": summary["config_sha256"],
                    "training_completed_at_utc": metadata["finished_at_utc"],
                    "git_commit": metadata["git_commit"],
                    "git_dirty": metadata["git_dirty"],
                    "holdout_used_before_lock": False,
                    "final_evaluation_count": 0,
                    "locked_at_utc": utc_now(),
                }
                if payload["locked_at_utc"] <= payload["training_completed_at_utc"]:
                    raise ValueError("Selection lock anterior al fin del entrenamiento")
                if selection_path.exists():
                    existing = json.loads(selection_path.read_text(encoding="utf-8"))
                    comparable = {k: v for k, v in payload.items() if k != "locked_at_utc"}
                    if {k: existing.get(k) for k in comparable} != comparable:
                        raise ValueError(f"Lock previo incompatible: {selection_path}")
                else:
                    with selection_path.open("x", encoding="utf-8") as stream:
                        json.dump(payload, stream, indent=2, sort_keys=True)
                        stream.write("\n")
                    written.append(selection_path)
            members.append(
                {
                    "model": model,
                    "run_id": summary["run_id"],
                    "run_dir": str(run),
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "summary_sha256": sha256_file(run / "summary.json"),
                    "selection_lock_sha256": sha256_file(selection_path),
                    "best_epoch": summary["best_epoch"],
                    "best_val_macro_f1": summary["best_val_macro_f1"],
                }
            )
        group = ROOT / f"seed_{seed}" / "ensemble.selection.lock.json"
        group.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "protocol": "historical_ensemble_multicrop_loso_v1",
            "seed": seed,
            "held_out_source": SOURCE,
            "split_lock_sha256": LOCK_SHA256,
            "manifest_hashes": split_lock["derived_sha256"],
            "holdout_sha256": HOLDOUT_SHA256,
            "historical_ensemble_evidence_sha256": HISTORICAL_ENSEMBLE_SHA256,
            "members": members,
            "fusion": "mean_of_softmax_probabilities",
            "weights": list(WEIGHTS),
            "decision_rule": "argmax",
            "holdout_used_before_lock": False,
            "final_evaluation_count": 0,
            "locked_at_utc": utc_now(),
        }
        if group.exists():
            existing = json.loads(group.read_text(encoding="utf-8"))
            comparable = {k: v for k, v in payload.items() if k != "locked_at_utc"}
            if {k: existing.get(k) for k in comparable} != comparable:
                raise ValueError(f"Lock ensemble incompatible: {group}")
        else:
            with group.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            written.append(group)
    return written


def fuse_softmax(probs: list[torch.Tensor]) -> torch.Tensor:
    """Same float32 weighted softmax mean as the historical SoftVotingEnsemble."""
    if len(probs) != len(MEMBERS):
        raise ValueError("Faltan miembros del ensamble histórico")
    stacked = torch.stack(probs, dim=0)
    weights = torch.tensor(WEIGHTS, dtype=torch.float32, device=stacked.device).view(-1, 1, 1)
    return (stacked * weights).sum(dim=0)


def aligned_predictions(manifest: pd.DataFrame, inferred: pd.DataFrame) -> pd.DataFrame:
    if inferred["sample_id"].duplicated().any() or manifest["sample_id"].duplicated().any():
        raise ValueError("sample_id duplicado")
    if set(inferred["sample_id"]) != set(manifest["sample_id"]):
        raise ValueError("Componentes no cubren exactamente el holdout")
    columns = ["sample_id", "image_path", "source_id", "historical_split", "label"]
    aligned = manifest[columns].merge(inferred, on="sample_id", validate="one_to_one")
    if len(aligned) != len(manifest):
        raise ValueError("Alineación incompleta")
    return aligned


def metrics_for(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    truth, predicted = predictions["label"], predictions["pred_label"]
    present = sorted(truth.unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=present, zero_division=0
    )
    class_metrics = pd.DataFrame(
        {"class": present, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    classes = [name.removeprefix("prob_") for name in predictions if name.startswith("prob_")]
    if not classes:
        classes = sorted(set(truth) | set(predicted))
    matrix = pd.DataFrame(
        confusion_matrix(truth, predicted, labels=classes), index=classes, columns=classes
    )
    correct = truth.eq(predicted)
    confidence = predictions["confidence"].astype(float)
    metrics = {
        "samples": len(predictions),
        "classes_present": present,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=present, average="macro", zero_division=0)
        ),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "weighted_f1": float(
            f1_score(truth, predicted, labels=present, average="weighted", zero_division=0)
        ),
        "ece_15_bins": float(expected_calibration_error(confidence.tolist(), correct.tolist())),
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_errors": float(confidence[~correct].mean()),
        "error_rate": float((~correct).mean()),
        "errors": int((~correct).sum()),
    }
    return metrics, class_metrics, matrix


def evaluate_seed(seed: int, *, commit_guard: Callable[[], None] | None = None) -> dict:
    """One guarded pass; models are fixed by all selection locks before reading holdout."""
    if seed not in SEEDS:
        raise ValueError("Seed no registrada")
    group_path = ROOT / f"seed_{seed}" / "ensemble.selection.lock.json"
    group = json.loads(group_path.read_text(encoding="utf-8"))
    if (
        group["seed"] != seed
        or group["held_out_source"] != SOURCE
        or group["split_lock_sha256"] != LOCK_SHA256
        or group["holdout_sha256"] != HOLDOUT_SHA256
        or group["historical_ensemble_evidence_sha256"] != HISTORICAL_ENSEMBLE_SHA256
        or [member["model"] for member in group["members"]] != list(MEMBERS)
    ):
        raise ValueError("Lock ensemble inválido")
    if group["weights"] != list(WEIGHTS) or group["fusion"] != "mean_of_softmax_probabilities":
        raise ValueError("Fusion distinta del protocolo")
    if sha256_file(SPLITS / "manifest.lock.json") != LOCK_SHA256:
        raise ValueError("Manifest LOSO alterado")
    for member in group["members"]:
        run = Path(member["run_dir"])
        for filename, digest in (
            ("best.pth", member["checkpoint_sha256"]),
            ("summary.json", member["summary_sha256"]),
            ("selection.lock.json", member["selection_lock_sha256"]),
        ):
            if sha256_file(run / filename) != digest:
                raise ValueError(f"Artefacto alterado: {run / filename}")
    out = ROOT / f"seed_{seed}" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ensemble_metrics.json").exists():
        raise FileExistsError("Evaluación ya finalizada")
    guard = out / "final_evaluation.started.json"
    with guard.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": seed,
                "evaluation_count": 1,
                "ensemble_selection_lock_sha256": sha256_file(group_path),
                "started_at_utc": utc_now(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    if commit_guard is not None:
        commit_guard()
    if sha256_file(SPLITS / "holdout.csv") != HOLDOUT_SHA256:
        raise ValueError("Holdout alterado")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {
        member["model"]: load_validated_run(
            Path(member["run_dir"]) / "best.pth",
            expected_model=member["model"],
            splits_dir=SPLITS,
            device=device,
        )
        for member in group["members"]
    }
    reference = loaded[MEMBERS[0]]
    mapping = reference.summary["class_to_idx"]
    for run in loaded.values():
        if (
            run.summary["class_to_idx"] != mapping
            or run.summary["preprocessing"] != reference.summary["preprocessing"]
        ):
            raise ValueError("Miembros incompatibles para inferencia compartida")
        run.model.eval()
    inverse = {index: name for name, index in mapping.items()}
    class_names = [inverse[i] for i in range(len(inverse))]
    dataset = CornDataset(
        str(SPLITS / "holdout.csv"),
        transform=reference.factory.get_pipeline("test"),
        class_to_idx=mapping,
    )
    manifest = dataset.data_frame
    if len(manifest) != 5816 or set(manifest["source_id"]) != {SOURCE}:
        raise ValueError("Holdout distinto")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)
    rows: dict[str, list[dict]] = {model: [] for model in (*MEMBERS, "ensemble")}
    with torch.inference_mode():
        for batch in loader:
            images, targets, sample_ids = unpack_batch(batch)
            if sample_ids is None:
                raise ValueError("Batch sin sample_id")
            images = images.to(device)
            probabilities = [loaded[model].model(images).softmax(dim=1) for model in MEMBERS]
            outputs = dict(zip(MEMBERS, probabilities, strict=True))
            outputs["ensemble"] = fuse_softmax(probabilities)
            for model, scores in outputs.items():
                confidence, predicted = scores.max(dim=1)
                for sid, label, pred, conf, probs in zip(
                    sample_ids,
                    targets.tolist(),
                    predicted.cpu().tolist(),
                    confidence.cpu().tolist(),
                    scores.cpu().tolist(),
                    strict=True,
                ):
                    rows[model].append(
                        {
                            "sample_id": sid,
                            "true_label_from_batch": inverse[label],
                            "pred_label": inverse[pred],
                            "confidence": conf,
                            **{
                                f"prob_{name}": prob
                                for name, prob in zip(class_names, probs, strict=True)
                            },
                        }
                    )
    results = {}
    for model, inferred_rows in rows.items():
        predictions = aligned_predictions(manifest, pd.DataFrame(inferred_rows))
        if not predictions["label"].eq(predictions["true_label_from_batch"]).all():
            raise ValueError(f"Label mismatch por sample_id: {model}")
        predictions = predictions.drop(columns="true_label_from_batch")
        metrics, classes, matrix = metrics_for(predictions)
        metrics.update(
            protocol="historical_ensemble_multicrop_loso_v1",
            seed=seed,
            model=model,
            evaluation_count=1,
            ensemble_selection_lock_sha256=sha256_file(group_path),
            final_manifest_sha256=HOLDOUT_SHA256,
            finished_at_utc=utc_now(),
        )
        predictions.to_csv(out / f"{model}_predictions.csv", index=False)
        classes.to_csv(out / f"{model}_class_metrics.csv", index=False)
        matrix.to_csv(out / f"{model}_confusion_matrix.csv")
        write_json(out / f"{model}_metrics.json", metrics)
        results[model] = metrics
    return results
