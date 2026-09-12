"""Read-only audit of historical runs; recalculation is not a new model evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.analysis.fairness import compute_disparity_metrics, compute_subgroup_metrics
from src.data.identity import ensure_sample_ids
from src.models import build_model
from src.provenance import atomic_json, sha256_file


def audit(bundle: Path, stage2: Path) -> dict:
    split_dir = bundle / "splits/seed_42"
    splits = {
        s: ensure_sample_ids(pd.read_csv(split_dir / f"{s}.csv")) for s in ("train", "val", "test")
    }
    payload = {
        "schema_version": 1,
        "mode": "saved_artifact_audit_no_image_inference",
        "bundle": str(bundle.resolve()),
        "runs": [],
        "split_sha256": {s: sha256_file(split_dir / f"{s}.csv") for s in splits},
        "split_counts": {s: len(f) for s, f in splits.items()},
        "within_pipeline_path_overlap": {
            f"{a}/{b}": len(set(splits[a].image_path) & set(splits[b].image_path))
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        },
    }
    hpo = json.loads((bundle / "tuning/efficientnet_b0/best_params.json").read_text())
    trials = pd.read_csv(bundle / "tuning/efficientnet_b0/trials.csv")
    payload["hpo"] = {"declared": hpo, "trial_states": trials.state.value_counts().to_dict()}
    for path in sorted((bundle / "main").glob("*/*/summary.json")):
        summary = json.loads(path.read_text())
        run = path.parent
        pred = ensure_sample_ids(pd.read_csv(run / "predictions.csv"))
        test = splits["test"].set_index("sample_id")
        assert set(test.index) == set(pred.sample_id), "Predictions do not cover test IDs"
        aligned = test.loc[pred.sample_id].reset_index()
        assert pred.label.tolist() == aligned.label.tolist(), "Labels disagree"
        assert pred.environment.tolist() == aligned.environment.tolist(), "Environments disagree"
        mapping = summary["class_to_idx"]
        names = sorted(mapping, key=mapping.get)
        true, predicted = pred.label.map(mapping), pred.pred_label.map(mapping)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, predicted, labels=list(range(len(names))), average="macro", zero_division=0
        )
        metrics = {
            "accuracy": float(accuracy_score(true, predicted)),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "errors": int((true != predicted).sum()),
        }
        matrix = confusion_matrix(true, predicted, labels=list(range(len(names))))
        saved_matrix = pd.read_csv(run / "test_confusion_matrix.csv", index_col=0).loc[names, names]
        history = pd.read_csv(run / "train_history.csv")
        best_row = history.loc[history.val_macro_f1.idxmax()]
        state = torch.load(run / "best.pth", weights_only=True, map_location="cpu")
        last = torch.load(run / "last.pth", weights_only=True, map_location="cpu")
        model = build_model(summary["model"], num_classes=len(names), pretrained=False)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            shape = list(model(torch.zeros(1, 3, *summary["image_size"])).shape)
        assert shape == [1, len(names)]
        subgroup = compute_subgroup_metrics(
            true.tolist(), predicted.tolist(), pred.environment.tolist(), names
        )
        item = {
            "model": summary["model"],
            "run_id": run.name,
            "recomputed": metrics,
            "declared": summary,
            "history_best_epoch": int(best_row.epoch),
            "history_best_macro_f1": float(best_row.val_macro_f1),
            "history_epochs": len(history),
            "training_minutes": float(history.epoch_seconds.sum() / 60),
            "matrix_matches_predictions": bool(np.array_equal(matrix, saved_matrix.to_numpy())),
            "summary_metrics_match": all(
                abs(metrics[k] - summary["test"][k]) < 1e-12 for k in ("accuracy", "macro_f1")
            ),
            "checkpoint_strict_load": True,
            "structural_output_shape": shape,
            "checkpoint_sha256": sha256_file(run / "best.pth"),
            "summary_sha256": sha256_file(path),
            "predictions_sha256": sha256_file(run / "predictions.csv"),
            "best_and_last_differ": any(not torch.equal(v, last[k]) for k, v in state.items()),
            "hpo_differences": {
                k: {"hpo": v, "run": summary.get(k)}
                for k, v in hpo["best_params"].items()
                if summary.get(k) != v
            },
            "subgroups_recomputed": subgroup,
            "disparity_recomputed": compute_disparity_metrics(subgroup),
            "legacy_limitations": [
                "no original byte hashes in splits",
                "no serialized preprocessing or training code revision",
                "saved path alignment cannot exclude historical image substitution",
                "best checkpoint epoch cannot be proved from bare state_dict",
            ],
        }
        payload["runs"].append(item)
        del model, state, last
    stage_pred = pd.read_csv(stage2 / "final/predictions.csv")
    counts = confusion_matrix(stage_pred.label_idx, stage_pred.pred_idx, labels=range(9))
    labels = list(json.loads((stage2 / "dataset_summary.json").read_text())["classes"])
    archived = pd.read_csv(stage2 / "final/confusion_matrix.csv", index_col=0).loc[labels, labels]
    precision, recall, f1, _ = precision_recall_fscore_support(
        stage_pred.label_idx, stage_pred.pred_idx, labels=range(9), average="macro", zero_division=0
    )
    payload["stage2"] = {
        "n": len(stage_pred),
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "accuracy": float(np.trace(counts) / counts.sum()),
        "matrix_matches": bool(np.array_equal(counts, archived.to_numpy())),
        "predictions_sha256": sha256_file(stage2 / "final/predictions.csv"),
    }
    stage_paths = set(stage_pred.image_path)
    payload["cross_experiment_overlap"] = {
        "stage2_holdout_vs_main": {
            s: len(stage_paths & set(frame.image_path)) for s, frame in splits.items()
        },
        "same_holdout": stage_paths == set(splits["test"].image_path),
        "interpretation": (
            "Different protocols; cross-experiment overlap alone is not within-run leakage"
        ),
    }
    payload["ensemble"] = {
        "declared": json.loads((bundle / "ensemble/ensemble_summary.json").read_text()),
        "recalculation": (
            "unavailable: full per-class probabilities and ensemble predictions missing"
        ),
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, default=Path("outputs/etapa_2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new audit output; historical evidence is not overwritten")
    payload = audit(args.bundle, args.stage2)
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "runs": [
                    {
                        "model": r["model"],
                        "metrics": r["recomputed"],
                        "hpo_differences": r["hpo_differences"],
                    }
                    for r in payload["runs"]
                ],
                "stage2": payload["stage2"],
                "overlap": payload["cross_experiment_overlap"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
