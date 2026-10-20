"""Compile two completed, locked LOSO runs into reproducible thesis tables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.experiments.loso_finalize import BASELINE_CONFIG_SHA256
from scripts.experiments.prepare_loso_baseline import FROZEN, OUTPUT, SOURCES, sha256
from src.config import PROJECT_ROOT

EVIDENCE = PROJECT_ROOT / "docs/es/reproducibilidad/evidencia/loso_baseline"
SOURCE_ANALYSIS = PROJECT_ROOT / "docs/es/reproducibilidad/evidencia/source_analysis"


def collect(base: Path, evidence: Path) -> dict:
    master = pd.read_csv(FROZEN / "master_manifest.csv")
    historical = pd.read_csv(SOURCE_ANALYSIS / "validation_metrics_by_source.csv")
    historical_test = pd.read_csv(SOURCE_ANALYSIS / "test_metrics_by_source.csv")
    historical_class = pd.read_csv(SOURCE_ANALYSIS / "class_metrics_by_source.csv")
    summary_rows, class_rows, run_records = [], [], []
    evidence.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        root = base / source / "seed_42"
        split_dir = root / "splits"
        split_lock_path = split_dir / "manifest.lock.json"
        split_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
        candidates = sorted((root / "runs" / "efficientnet_lite0").glob("*/summary.json"))
        if len(candidates) != 1:
            raise ValueError(f"{source}: se esperaba exactamente una run completa")
        run = candidates[0].parent
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selection_path = run / "selection.lock.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        metrics = json.loads((run / "holdout_metrics.json").read_text(encoding="utf-8"))
        if summary["config_sha256"] != BASELINE_CONFIG_SHA256:
            raise ValueError(f"{source}: baseline config SHA diferente")
        if summary["split_manifest_sha256"] != sha256(split_lock_path):
            raise ValueError(f"{source}: splits SHA diferente")
        if summary.get("test_used") is not False or "test" in summary["metrics"]:
            raise ValueError(f"{source}: entrenamiento accedió a test")
        if sha256(run / "best.pth") != selection["checkpoint_sha256"]:
            raise ValueError(f"{source}: checkpoint SHA diferente")
        if sha256(summary_path) != selection["summary_sha256"]:
            raise ValueError(f"{source}: summary cambió tras el lock")
        if metrics["selection_lock_sha256"] != sha256(selection_path):
            raise ValueError(f"{source}: lock no coincide con evaluación")
        if metrics["holdout_sha256"] != split_lock["derived_sha256"]["holdout.csv"]:
            raise ValueError(f"{source}: holdout SHA diferente")
        if not (datetime.fromisoformat(selection["training_completed_at_utc"])
                < datetime.fromisoformat(selection["selection_locked_at_utc"])
                < datetime.fromisoformat(metrics["holdout_started_at_utc"])
                < datetime.fromisoformat(metrics["holdout_finished_at_utc"])):
            raise ValueError(f"{source}: timestamps violan el orden selection → holdout")
        history = pd.read_csv(run / "train_history.csv")
        best = history.loc[history["epoch"] == summary["best_epoch"]].iloc[0]
        holdout = pd.read_csv(split_dir / "holdout.csv", dtype={"sample_id": str})
        predictions = pd.read_csv(run / "holdout_predictions.csv", dtype={"sample_id": str})
        if len(holdout) != metrics["samples"] or set(holdout["sample_id"]) != set(
            predictions["sample_id"]
        ) or predictions["sample_id"].duplicated().any():
            raise ValueError(f"{source}: cobertura de predicciones incorrecta")
        if predictions.set_index("sample_id").loc[holdout["sample_id"], "label"].tolist() != (
            holdout["label"].tolist()
        ):
            raise ValueError(f"{source}: etiquetas de predicciones incorrectas")

        old_val = historical.loc[(historical["model"] == "baseline") &
                                 (historical["source_id"] == source)].iloc[0]
        old_test = historical_test.loc[(historical_test["model"] == "baseline") &
                                      (historical_test["source_id"] == source)].iloc[0]
        counts = split_lock["preflight"]["counts"]
        summary_rows.append({
            "source_id": source,
            "run_id": summary["run_id"],
            "seed": summary["seed"],
            "train_n": counts["train"],
            "val_n": counts["val"],
            "holdout_n": counts["holdout"],
            "classes_present": ";".join(metrics["classes_present"]),
            "best_epoch": summary["best_epoch"],
            "epochs_run": summary["epochs_run"],
            "train_macro_f1_at_best": best["train_macro_f1"],
            "internal_val_macro_f1": summary["best_val_macro_f1"],
            "train_val_gap_at_best": best["train_macro_f1"] - best["val_macro_f1"],
            "duration_seconds_sum_epochs": history["epoch_seconds"].sum(),
            "loso_macro_f1": metrics["macro_f1"],
            "loso_accuracy": metrics["accuracy"],
            "loso_errors": metrics["errors"],
            "loso_error_rate": metrics["error_rate"],
            "loso_ece_15_bins": metrics["ece_15_bins"],
            "mean_confidence_correct": metrics["mean_confidence_correct"],
            "mean_confidence_errors": metrics["mean_confidence_errors"],
            "historical_val_macro_f1": old_val["macro_f1"],
            "historical_test_macro_f1": old_test["macro_f1"],
            "delta_loso_minus_historical_val": metrics["macro_f1"] - old_val["macro_f1"],
            "original_train_n": split_lock["preflight"]["original_train"],
            "train_delta_n": split_lock["preflight"]["train_delta"],
            "train_delta_percent": split_lock["preflight"]["train_delta_percent"],
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "selection_lock_sha256": sha256(selection_path),
            "split_lock_sha256": sha256(split_lock_path),
        })
        current_class = pd.read_csv(run / "holdout_class_metrics.csv")
        remaining_train = pd.read_csv(split_dir / "train.csv")
        for _, row in current_class.iterrows():
            name = row["class"]
            previous = historical_class.loc[
                (historical_class["model"] == "baseline") &
                (historical_class["split"] == "validation") &
                (historical_class["source_id"] == source) &
                (historical_class["class"] == name)
            ].iloc[0]
            alternatives = sorted(set(master.loc[(master["source_id"] != source) &
                                                  (master["label"] == name), "source_id"]))
            class_rows.append({
                "source_id": source,
                "class": name,
                "holdout_support": row["support"],
                "train_without_source": int((remaining_train["label"] == name).sum()),
                "other_source_count": len(alternatives),
                "other_sources": ";".join(alternatives),
                "historical_val_support": previous["support"],
                "historical_val_f1": previous["f1"],
                "loso_precision": row["precision"],
                "loso_recall": row["recall"],
                "loso_f1": row["f1"],
                "delta_loso_minus_historical_val": row["f1"] - previous["f1"],
            })
        slug = source.replace("-", "_")
        for suffix, input_name in (("", "holdout_confusion_matrix.csv"),
                                   ("_normalized", "holdout_confusion_normalized.csv")):
            matrix = pd.read_csv(run / input_name, index_col=0)
            matrix.to_csv(evidence / f"loso_confusion_{slug}{suffix}.csv")
        run_records.append({
            "source_id": source,
            "run_id": summary["run_id"],
            "run_dir": str(run),
            "selection": selection,
            "holdout_metrics": metrics,
        })

    results = pd.DataFrame(summary_rows)
    results.to_csv(evidence / "loso_results.csv", index=False)
    pd.DataFrame(class_rows).to_csv(evidence / "loso_class_metrics.csv", index=False)
    payload = {"schema_version": 1, "protocol": "loso_baseline_seed42_v1",
               "training_runs": len(run_records), "results": run_records}
    (evidence / "loso_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=OUTPUT)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    result = collect(args.base, args.evidence)
    print(f"Compiladas {result['training_runs']} runs LOSO con locks y holdouts verificados")


if __name__ == "__main__":
    main()
