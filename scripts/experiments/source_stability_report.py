"""Consolidate the pre-registered 3-seed baseline and two LOSO experiments.

Reads completed local artifacts only. Never trains, selects checkpoints, or runs
inference. Final-split metrics are checked against the persisted predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.config import PROJECT_ROOT
from src.data.preparation import sha256_file
from src.training.evaluation import expected_calibration_error

SEEDS = (42, 123, 2026)
SCENARIOS = ("baseline", "maize-diseases", "multicrop-disease-maiz")
EVIDENCE = PROJECT_ROOT / "docs/es/reproducibilidad/evidencia"
OUTPUT = EVIDENCE / "multiseed_source_stability"
BASELINE_42 = (
    PROJECT_ROOT
    / "outputs/archives"
    / "DoctorMaiz_efficientnet_lite0_20260921_204608"
    / "efficientnet_lite0/20260921_204608"
)
LOSO = PROJECT_ROOT / "outputs/loso/efficientnet_lite0_baseline"
HISTORICAL_REFERENCE = {
    "maize-diseases": 0.9978925393906257,
    "multicrop-disease-maiz": 0.9972173913043477,
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict:
    """A descriptive n=3 summary; SD is sample SD (ddof=1)."""
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("Se requieren exactamente tres valores finitos")
    array = np.asarray(values, dtype=float)
    return {
        "n": 3,
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sample_sd": float(array.std(ddof=1)),
        "ddof": 1,
        "min": float(array.min()),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
    }


def paths(scenario: str, seed: int) -> tuple[Path, Path, Path]:
    if scenario not in SCENARIOS or seed not in SEEDS:
        raise ValueError("Escenario/seed no pre-registrado")
    if scenario == "baseline":
        splits = PROJECT_ROOT / "outputs/multiseed-inputs/frozen/seed_42"
        if seed == 42:
            return BASELINE_42, splits, BASELINE_42
        root = PROJECT_ROOT / f"outputs/multiseed/source_stability/baseline/seed_{seed}"
    else:
        splits = LOSO / scenario / "seed_42/splits"
        root = LOSO / scenario / f"seed_{seed}"
    runs = list((root / "runs/efficientnet_lite0").glob("*/summary.json"))
    if len(runs) != 1:
        raise ValueError(f"Se esperaba un run completo en {root}: {len(runs)}")
    return runs[0].parent, splits, root


def _predictions(path: Path, manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    if (
        frame["sample_id"].duplicated().any()
        or manifest["sample_id"].duplicated().any()
        or set(frame["sample_id"]) != set(manifest["sample_id"])
    ):
        raise ValueError(f"sample_id incompleto o duplicado: {path}")
    aligned = manifest[["sample_id", "label"]].merge(
        frame, on="sample_id", validate="one_to_one", suffixes=("_manifest", "")
    )
    if aligned["label_manifest"].tolist() != aligned["label"].tolist():
        raise ValueError(f"Etiqueta diferente del manifest: {path}")
    return aligned


def _metrics(frame: pd.DataFrame) -> dict:
    labels = sorted(frame["label"].unique())
    precision, recall, _f1, _support = precision_recall_fscore_support(
        frame["label"], frame["pred_label"], labels=labels, zero_division=0
    )
    correct = frame["label"].eq(frame["pred_label"])
    confidence = frame["pred_prob"].astype(float)
    return {
        "macro_f1": float(
            f1_score(
                frame["label"], frame["pred_label"], labels=labels, average="macro", zero_division=0
            )
        ),
        "accuracy": float(accuracy_score(frame["label"], frame["pred_label"])),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "weighted_f1": float(
            f1_score(
                frame["label"],
                frame["pred_label"],
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "error_rate": float((~correct).mean()),
        "mean_confidence_correct": (float(confidence[correct].mean()) if correct.any() else None),
        "mean_confidence_errors": (
            float(confidence[~correct].mean()) if (~correct).any() else None
        ),
        "ece_15_bins": float(expected_calibration_error(confidence.tolist(), correct.tolist())),
    }


def _class_rows(frame: pd.DataFrame, scenario: str, seed: int, stage: str) -> list[dict]:
    rows = []
    for name in sorted(frame["label"].unique()):
        precision, recall, f1, support = precision_recall_fscore_support(
            frame["label"], frame["pred_label"], labels=[name], zero_division=0
        )
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "stage": stage,
                "class": name,
                "support": int(support[0]),
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
            }
        )
    return rows


def collect() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, class_rows, references = [], [], {}
    for scenario in SCENARIOS:
        for seed in SEEDS:
            run, splits, root = paths(scenario, seed)
            summary = _read(run / "summary.json")
            if (
                summary["seed"] != seed
                or summary["split_manifest_sha256"] != sha256_file(splits / "manifest.lock.json")
                or summary["checkpoint_sha256"] != sha256_file(run / "best.pth")
            ):
                raise ValueError(f"Contrato o checkpoint inválido: {run}")
            history = pd.read_csv(run / "train_history.csv")
            best = history.loc[history["val_macro_f1"].idxmax()]
            if (
                int(best["epoch"]) != summary["best_epoch"]
                or abs(float(best["val_macro_f1"]) - summary["best_val_macro_f1"]) > 1e-10
            ):
                raise ValueError(f"Best validation inconsistente: {run}")
            validation_path = (
                EVIDENCE / "source_analysis/baseline_validation_predictions.csv"
                if scenario == "baseline" and seed == 42
                else run / "validation_predictions.csv"
            )
            final_prefix = "test" if scenario == "baseline" else "holdout"
            final_predictions = (
                run / "predictions.csv"
                if scenario == "baseline" and seed == 42
                else run / f"{final_prefix}_predictions.csv"
            )
            val = _predictions(validation_path, splits / "val.csv")
            final = _predictions(final_predictions, splits / f"{final_prefix}.csv")
            val_metrics, final_metrics = _metrics(val), _metrics(final)
            if abs(val_metrics["macro_f1"] - summary["best_val_macro_f1"]) > 1e-6:
                raise ValueError(f"Validation distinta del summary: {run}")
            recorded = (
                summary["metrics"]["test"]
                if scenario == "baseline" and seed == 42
                else _read(run / f"{final_prefix}_metrics.json")
            )
            for metric_name in (
                "macro_f1",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "weighted_f1",
                "ece_15_bins",
            ):
                if (
                    metric_name in recorded
                    and abs(final_metrics[metric_name] - recorded[metric_name]) > 1e-6
                ):
                    raise ValueError(f"Final {metric_name} distinto del reporte: {run}")
            if seed != 42:
                guard = _read(run / "final_evaluation.started.json")
                lock = _read(run / "selection.lock.json")
                if (
                    guard["evaluation_count"] != 1
                    or lock["selection_locked_at_utc"] >= guard["started_at_utc"]
                    or guard["selection_lock_sha256"] != sha256_file(run / "selection.lock.json")
                ):
                    raise ValueError(f"Orden lock/evaluación inválido: {run}")
                metadata = _read(root / "experiment_metadata.json")
                duration = metadata["duration_seconds"]
                environment = metadata["environment"]
                gpu = environment["gpu"]
                duration_basis = "wall_start_to_validation_complete"
                selection_locked_at = lock["selection_locked_at_utc"]
                evaluation_started_at = guard["started_at_utc"]
                evaluation_finished_at = recorded["finished_at_utc"]
                evaluation_count = guard["evaluation_count"]
            else:
                duration = float(history["epoch_seconds"].sum())
                gpu = None
                environment = {}
                duration_basis = "sum_epoch_seconds"
                historical_lock = (
                    _read(run / "selection.lock.json")
                    if (run / "selection.lock.json").exists()
                    else None
                )
                selection_locked_at = (
                    historical_lock["selection_locked_at_utc"] if historical_lock else None
                )
                evaluation_started_at = recorded.get("holdout_started_at_utc")
                evaluation_finished_at = recorded.get("holdout_finished_at_utc")
                evaluation_count = None  # No one-shot marker for historical seed 42.
            split_lock = _read(splits / "manifest.lock.json")
            dataset_fingerprint = (
                split_lock.get("master_manifest_sha256")
                or (split_lock["frozen_sha256"]["master_manifest.csv"])
            )
            row = {
                "scenario": scenario,
                "seed": seed,
                "run_id": summary["run_id"],
                "reused": seed == 42,
                "best_epoch": summary["best_epoch"],
                "epochs_run": summary["epochs_run"],
                "train_macro_f1_at_best": float(best["train_macro_f1"]),
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_ece_15_bins": val_metrics["ece_15_bins"],
                "train_val_gap": float(best["train_macro_f1"] - val_metrics["macro_f1"]),
                "final_macro_f1": final_metrics["macro_f1"],
                "final_accuracy": final_metrics["accuracy"],
                "final_macro_precision": final_metrics["macro_precision"],
                "final_macro_recall": final_metrics["macro_recall"],
                "final_weighted_f1": final_metrics["weighted_f1"],
                "final_error_rate": final_metrics["error_rate"],
                "mean_confidence_correct": final_metrics["mean_confidence_correct"],
                "mean_confidence_errors": final_metrics["mean_confidence_errors"],
                "held_out_source": None if scenario == "baseline" else scenario,
                "selection_locked_at_utc": selection_locked_at,
                "evaluation_started_at_utc": evaluation_started_at,
                "evaluation_finished_at_utc": evaluation_finished_at,
                "final_evaluation_count": evaluation_count,
                "final_predictions_sha256": sha256_file(final_predictions),
                "selection_lock_sha256": (
                    sha256_file(run / "selection.lock.json")
                    if (run / "selection.lock.json").exists()
                    else None
                ),
                "config_sha256": summary["config_sha256"],
                "python": environment.get("python"),
                "pytorch": environment.get("torch"),
                "timm": environment.get("timm"),
                "cuda": environment.get("cuda"),
                "git_commit": metadata["git_commit"] if seed != 42 else None,
                "git_dirty": metadata["git_dirty"] if seed != 42 else None,
                "metadata_sha256": sha256_file(root / "experiment_metadata.json")
                if seed != 42
                else None,
                "dataset_fingerprint": dataset_fingerprint,
                "final_ece_15_bins": final_metrics["ece_15_bins"],
                "duration_seconds": duration,
                "duration_basis": duration_basis,
                "gpu": gpu,
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "manifest_lock_sha256": summary["split_manifest_sha256"],
                "train_size": len(pd.read_csv(splits / "train.csv")),
                "validation_size": len(val),
                "final_size": len(final),
                "run_path": str(run.relative_to(PROJECT_ROOT)),
            }
            if scenario != "baseline":
                row["historical_reference_f1"] = HISTORICAL_REFERENCE[scenario]
                row["delta_vs_historical"] = (
                    final_metrics["macro_f1"] - HISTORICAL_REFERENCE[scenario]
                )
            rows.append(row)
            class_rows.extend(_class_rows(val, scenario, seed, "validation"))
            class_rows.extend(_class_rows(final, scenario, seed, final_prefix))
            references[f"{scenario}/seed_{seed}"] = {
                "run_path": row["run_path"],
                "run_id": summary["run_id"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "split_lock_sha256": summary["split_manifest_sha256"],
                "validation_predictions_sha256": sha256_file(validation_path),
                "final_predictions_sha256": sha256_file(final_predictions),
                "selection_lock_sha256": sha256_file(run / "selection.lock.json")
                if (run / "selection.lock.json").exists()
                else None,
                "selection_lock_status": (
                    "historical_pre_final"
                    if scenario != "baseline" and seed == 42
                    else "not_pre_final_historical"
                    if seed == 42
                    else "pre_final"
                ),
            }
    return pd.DataFrame(rows), pd.DataFrame(class_rows), references


def summarize_classes(classes: pd.DataFrame) -> pd.DataFrame:
    """Per-class seed values plus n=3 mean and sample SD."""
    rows = []
    for (scenario, stage, name), group in classes.groupby(["scenario", "stage", "class"]):
        by_seed = group.set_index("seed")["f1"]
        if set(by_seed.index) != set(SEEDS) or len(by_seed) != 3:
            raise ValueError(f"Faltan seeds para {scenario}/{stage}/{name}")
        if group["support"].nunique() != 1:
            raise ValueError(f"Soporte distinto entre seeds: {scenario}/{stage}/{name}")
        summary = stats([float(by_seed.loc[seed]) for seed in SEEDS])
        rows.append(
            {
                "scenario": scenario,
                "stage": stage,
                "class": name,
                **{f"seed_{seed}": float(by_seed.loc[seed]) for seed in SEEDS},
                "mean": summary["mean"],
                "sample_sd": summary["sample_sd"],
                "support": int(group["support"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _plots(runs: pd.DataFrame, classes: pd.DataFrame, out: Path) -> None:
    baseline = runs[runs.scenario == "baseline"].sort_values("seed")
    loso = runs[runs.scenario != "baseline"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(baseline.seed, baseline.val_macro_f1, "o-", label="Validación")
    ax.plot(baseline.seed, baseline.final_macro_f1, "s-", label="Test")
    ax.set(xlabel="Seed", ylabel="Macro-F1", title="Baseline estándar por seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "baseline_f1_by_seed.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for source, group in loso.groupby("scenario"):
        group = group.sort_values("seed")
        ax.plot(group.seed, group.final_macro_f1, "o-", label=source)
    ax.set(xlabel="Seed", ylabel="LOSO Macro-F1", title="LOSO por fuente y seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loso_f1_by_source_seed.png", dpi=160)
    plt.close(fig)

    names = list(SCENARIOS)
    summaries = [
        stats(runs.loc[runs.scenario == name, "final_macro_f1"].tolist()) for name in names
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        names,
        [item["mean"] for item in summaries],
        yerr=[item["sample_sd"] for item in summaries],
        fmt="o",
        capsize=5,
    )
    ax.tick_params(axis="x", rotation=12)
    ax.set(ylabel="Macro-F1 final", title="Media ± SD muestral (n=3)")
    fig.tight_layout()
    fig.savefig(out / "scenario_mean_sd.png", dpi=160)
    plt.close(fig)

    lethal = classes[
        (classes.scenario == "multicrop-disease-maiz")
        & (classes.stage == "holdout")
        & (classes["class"] == "lethal_necrosis")
    ].sort_values("seed")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(lethal.seed.astype(str), lethal.f1)
    ax.set(xlabel="Seed", ylabel="F1", title="Lethal necrosis bajo LOSO multicrop", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out / "multicrop_lethal_necrosis_by_seed.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for source, group in loso.groupby("scenario"):
        group = group.sort_values("seed")
        ax.plot(group.seed, group.delta_vs_historical, "o-", label=source)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(
        xlabel="Seed", ylabel="LOSO − validation histórica", title="Delta descriptivo por fuente"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loso_delta_by_source.png", dpi=160)
    plt.close(fig)


def main() -> None:
    runs, classes, references = collect()
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    runs[runs.scenario == "baseline"].to_csv(output / "baseline_multiseed.csv", index=False)
    runs[runs.scenario != "baseline"].to_csv(output / "loso_multiseed.csv", index=False)
    classes[classes.scenario != "baseline"].to_csv(output / "loso_class_multiseed.csv", index=False)
    classes[classes.scenario == "baseline"].to_csv(
        output / "baseline_class_multiseed.csv", index=False
    )
    registry_columns = [
        "scenario",
        "seed",
        "run_id",
        "reused",
        "run_path",
        "held_out_source",
        "selection_locked_at_utc",
        "evaluation_started_at_utc",
        "evaluation_finished_at_utc",
        "final_evaluation_count",
        "checkpoint_sha256",
        "config_sha256",
        "manifest_lock_sha256",
        "selection_lock_sha256",
        "final_predictions_sha256",
        "git_commit",
        "git_dirty",
    ]
    runs[registry_columns].to_csv(output / "evaluation_registry.csv", index=False)
    class_summary = summarize_classes(classes)
    class_summary[class_summary.scenario != "baseline"].to_csv(
        output / "loso_class_summary.csv", index=False
    )
    class_summary[class_summary.scenario == "baseline"].to_csv(
        output / "baseline_class_summary.csv", index=False
    )
    summary = {
        "schema_version": 1,
        "n_seeds": 3,
        "seeds": list(SEEDS),
        "ddof": 1,
        "scenario_stats": {
            scenario: {
                metric: stats(runs.loc[runs.scenario == scenario, metric].tolist())
                for metric in (
                    "val_macro_f1",
                    "final_macro_f1",
                    "final_accuracy",
                    "final_ece_15_bins",
                )
            }
            for scenario in SCENARIOS
        },
        "historical_references": HISTORICAL_REFERENCE,
        "ensemble_gain_original": 0.9566569450594972 - 0.9483330101876505,
        "b0_tuning_gain_original": 0.9483330101876505 - 0.9426381998733162,
        "checkpoint_references": references,
        "baseline_npk_f1": class_summary[
            (class_summary.scenario == "baseline")
            & (class_summary.stage == "test")
            & (
                class_summary["class"].isin(
                    ("nitrogen_deficiency", "phosphorus_deficiency", "potassium_deficiency")
                )
            )
        ].to_dict(orient="records"),
        "multicrop_lethal_necrosis_f1": class_summary[
            (class_summary.scenario == "multicrop-disease-maiz")
            & (class_summary.stage == "holdout")
            & (class_summary["class"] == "lethal_necrosis")
        ].to_dict(orient="records"),
        "limitations": [
            "n=3 is descriptive, not a significance test",
            "LOSO also changes training size and class distribution",
            "baseline seed42 test predates this selection-lock protocol",
            "historical in-distribution source validation is not fully matched to LOSO holdout",
            "historical seed42 package versions/pretrained weight revision are not fully archived",
        ],
    }
    (output / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest_references.json").write_text(
        json.dumps(references, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plots(runs, classes, output)
    print(json.dumps(summary["scenario_stats"], indent=2))


if __name__ == "__main__":
    main()
