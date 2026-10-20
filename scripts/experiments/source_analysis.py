"""Análisis descriptivo por procedencia sobre manifests y predicciones congeladas.

Uso: python -m scripts.experiments.source_analysis --splits-dir ... --baseline-test ...
     --baseline-validation ... --hpo-validation ... --hpo-test ... --output-dir ...
No carga imágenes, no entrena ni modifica los splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.training.evaluation import expected_calibration_error

NUTRITION_CLASSES = (
    "nitrogen_deficiency",
    "phosphorus_deficiency",
    "potassium_deficiency",
)
EXPECTED_SPLIT_SIZES = {"train": 23400, "validation": 5014, "test": 5015}
EXPECTED_HASHES = {
    "master_manifest.csv": "64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163",
    "manifest.lock.json": "0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8",
    "train.csv": "231048178f3450bf84925f8d19eb5a672669ee9f2658a23fe87d88d6e9949434",
    "val.csv": "6f37710ebd797e470bc918fec6bf9502f0635e8220542336e88fe5c13b5ae6a5",
    "test.csv": "08c81aeec5a57e04416edf2c94f997c422bfcada357c6a3434e73aa9f771f724",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_ids(frame: pd.DataFrame, name: str) -> None:
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"{name}: sample_id vacío o duplicado")


def load_frozen_splits(directory: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Comprueba hash, membresía, identidad y totales antes de cualquier agregado."""
    hashes = {}
    for name, expected in EXPECTED_HASHES.items():
        actual = sha256_file(directory / name)
        if actual != expected:
            raise ValueError(f"{name}: SHA-256 {actual} != {expected}")
        hashes[name] = actual
    master = pd.read_csv(directory / "master_manifest.csv", dtype={"sample_id": str})
    _unique_ids(master, "master_manifest")
    if len(master) != 33429 or master["source_id"].isna().any():
        raise ValueError("El master no tiene 33,429 muestras con source_id")
    parts = []
    for split, filename in (
        ("train", "train.csv"),
        ("validation", "val.csv"),
        ("test", "test.csv"),
    ):
        frame = pd.read_csv(directory / filename, dtype={"sample_id": str})
        _unique_ids(frame, split)
        if len(frame) != EXPECTED_SPLIT_SIZES[split]:
            raise ValueError(f"{split}: total inesperado {len(frame)}")
        joined = frame.merge(
            master[["sample_id", "image_path", "label", "environment", "source_id"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_master"),
        )
        if joined["source_id_master"].isna().any():
            raise ValueError(f"{split}: hay sample_id ausentes del master")
        for column in ("image_path", "label", "environment", "source_id"):
            if not joined[column].equals(joined[f"{column}_master"]):
                raise ValueError(f"{split}: {column} contradice al master")
        parts.append(frame.assign(split=split))
    eligible = pd.concat(parts, ignore_index=True)
    _unique_ids(eligible, "splits concatenados")
    if set(eligible["sample_id"]) != set(master["sample_id"]):
        raise ValueError("Los splits no cubren exactamente el master")
    return eligible, hashes


def attach_predictions(path: Path, split: pd.DataFrame) -> pd.DataFrame:
    """Valida cobertura y une predicciones por sample_id, nunca por orden de fila."""
    predictions = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "label", "pred_label", "pred_prob"}
    if not required.issubset(predictions):
        raise ValueError(f"{path}: faltan {sorted(required - set(predictions))}")
    _unique_ids(predictions, str(path))
    if set(predictions["sample_id"]) != set(split["sample_id"]):
        raise ValueError(f"{path}: cobertura de sample_id distinta del split")
    attached = split[["sample_id", "label", "source_id", "environment", "image_path"]].merge(
        predictions,
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("_manifest", ""),
    )
    for column in ("label", "source_id", "environment", "image_path"):
        if column in predictions and not attached[column].equals(attached[f"{column}_manifest"]):
            raise ValueError(f"{path}: {column} contradice al manifest")
    if "true_label" in predictions and not attached["true_label"].equals(attached["label"]):
        raise ValueError(f"{path}: true_label contradice al manifest")
    if attached[["pred_label", "pred_prob"]].isna().any().any():
        raise ValueError(f"{path}: predicción o confianza vacía")
    if not attached["pred_prob"].between(0, 1).all():
        raise ValueError(f"{path}: confianza fuera de [0,1]")
    return attached.rename(
        columns={"label_manifest": "true_label_manifest", "source_id_manifest": "source_manifest"}
    )


def source_distribution(eligible: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(eligible["source_id"], eligible["split"])
    counts = counts.reindex(columns=["train", "validation", "test"], fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts = counts.reset_index().sort_values("total", ascending=False)
    for part in ("total", "train", "validation", "test"):
        counts[f"pct_{part}"] = 100 * counts[part] / counts[part].sum()
    return counts[
        [
            "source_id",
            "total",
            "train",
            "validation",
            "test",
            "pct_total",
            "pct_train",
            "pct_validation",
            "pct_test",
        ]
    ]


def source_class_distribution(eligible: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    sources = sorted(eligible["source_id"].unique())
    index = pd.MultiIndex.from_product([sources, classes], names=["source_id", "class"])
    counts = eligible.groupby(["source_id", "label"]).size().reindex(index, fill_value=0)
    result = counts.rename("total").reset_index()
    for split in ("train", "validation", "test"):
        part = eligible.loc[eligible["split"] == split].groupby(["source_id", "label"]).size()
        result[split] = part.reindex(index, fill_value=0).to_numpy()
    source_totals = result.groupby("source_id")["total"].transform("sum")
    class_totals = result.groupby("class")["total"].transform("sum")
    result["pct_within_source"] = 100 * result["total"] / source_totals
    result["pct_of_class"] = 100 * result["total"] / class_totals
    return result


def association(matrix: pd.DataFrame) -> dict:
    table = matrix.pivot(index="source_id", columns="class", values="total")
    chi2, _p, _dof, _expected = chi2_contingency(table.to_numpy(), correction=False)
    n = int(table.to_numpy().sum())
    cramer_v = float(np.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1))))
    source_entropy = {}
    for source, row in table.iterrows():
        probabilities = row.to_numpy() / row.sum()
        nonzero = probabilities[probabilities > 0]
        source_entropy[source] = float(-(nonzero * np.log2(nonzero)).sum())
    return {
        "n": n,
        "sources": int(table.shape[0]),
        "classes": int(table.shape[1]),
        "chi2_descriptive": float(chi2),
        "cramers_v": cramer_v,
        "entropy_bits_by_source": source_entropy,
    }


def source_metrics(predictions: pd.DataFrame, model: str, split: str) -> pd.DataFrame:
    rows = []
    for source, group in predictions.groupby("source_id"):
        truth, predicted = group["label"], group["pred_label"]
        classes_present = sorted(truth.unique())
        correct = truth.eq(predicted)
        confidence = group["pred_prob"].astype(float)
        rows.append(
            {
                "split": split,
                "model": model,
                "source_id": source,
                "samples": len(group),
                "classes_present": ";".join(classes_present),
                "n_classes_present": len(classes_present),
                "accuracy": accuracy_score(truth, predicted),
                "macro_f1": f1_score(
                    truth, predicted, labels=classes_present, average="macro", zero_division=0
                ),
                "weighted_f1": f1_score(
                    truth, predicted, labels=classes_present, average="weighted", zero_division=0
                ),
                "macro_precision": precision_score(
                    truth, predicted, labels=classes_present, average="macro", zero_division=0
                ),
                "macro_recall": recall_score(
                    truth, predicted, labels=classes_present, average="macro", zero_division=0
                ),
                "mean_confidence": confidence.mean(),
                "ece_15_bins": expected_calibration_error(confidence.tolist(), correct.tolist())
                if len(group) >= 30
                else np.nan,
                "errors": int((~correct).sum()),
                "error_rate": float((~correct).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("samples", ascending=False)


def class_metrics(
    predictions: pd.DataFrame, model: str, split: str, classes: list[str]
) -> pd.DataFrame:
    rows = []
    for source, group in predictions.groupby("source_id"):
        for name in classes:
            truth = group["label"] == name
            predicted = group["pred_label"] == name
            support = int(truth.sum())
            if not support:
                continue
            tp = int((truth & predicted).sum())
            fp = int((~truth & predicted).sum())
            fn = int((truth & ~predicted).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            hits = group.loc[truth & predicted, "pred_prob"]
            misses = group.loc[truth & ~predicted, "pred_prob"]
            confusions = group.loc[truth & ~predicted, "pred_label"].value_counts()
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "source_id": source,
                    "class": name,
                    "support": support,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "errors": fn,
                    "false_positives": fp,
                    "top_confusion": confusions.index[0] if len(confusions) else "",
                    "top_confusion_count": int(confusions.iloc[0]) if len(confusions) else 0,
                    "mean_confidence_hits": float(hits.mean()) if len(hits) else np.nan,
                    "mean_confidence_errors": float(misses.mean()) if len(misses) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[
        metrics["model"] == "baseline", ["source_id", "samples", "classes_present", "macro_f1"]
    ]
    hpo = metrics.loc[
        metrics["model"] == "hpo_trial_0", ["source_id", "samples", "classes_present", "macro_f1"]
    ]
    joined = baseline.merge(
        hpo, on="source_id", suffixes=("_baseline", "_hpo"), validate="one_to_one"
    )
    if not joined["samples_baseline"].equals(joined["samples_hpo"]) or not joined[
        "classes_present_baseline"
    ].equals(joined["classes_present_hpo"]):
        raise ValueError("Baseline y HPO no cubren los mismos soportes/clases por fuente")
    return pd.DataFrame(
        {
            "source_id": joined["source_id"],
            "support": joined["samples_baseline"],
            "classes_present": joined["classes_present_baseline"],
            "baseline_macro_f1": joined["macro_f1_baseline"],
            "hpo_macro_f1": joined["macro_f1_hpo"],
            "delta_hpo_minus_baseline": joined["macro_f1_hpo"] - joined["macro_f1_baseline"],
        }
    ).sort_values("delta_hpo_minus_baseline", ascending=False)


def confusion_rows(
    predictions: pd.DataFrame, model: str, split: str, classes: list[str], selected: set[str]
) -> pd.DataFrame:
    rows = []
    for source, group in predictions.groupby("source_id"):
        if source not in selected:
            continue
        counts = pd.crosstab(group["label"], group["pred_label"]).reindex(
            index=classes, columns=classes, fill_value=0
        )
        for true_label in classes:
            support = int(counts.loc[true_label].sum())
            if not support:
                continue
            for predicted_label in classes:
                count = int(counts.loc[true_label, predicted_label])
                rows.append(
                    {
                        "split": split,
                        "model": model,
                        "source_id": source,
                        "true_label": true_label,
                        "pred_label": predicted_label,
                        "count": count,
                        "row_normalized": count / support,
                    }
                )
    return pd.DataFrame(rows)


def loso_feasibility(eligible: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    rows = []
    for source, group in eligible.groupby("source_id"):
        remaining_train = eligible.loc[
            (eligible["split"] == "train") & (eligible["source_id"] != source)
        ]
        held_classes = sorted(group["label"].unique())
        missing = sorted(set(held_classes) - set(remaining_train["label"]))
        rows.append(
            {
                "source_id": source,
                "holdout_approx": len(group),
                "train_remaining": len(remaining_train),
                "classes_present": ";".join(held_classes),
                "n_classes_present": len(held_classes),
                "classes_without_remaining_train": ";".join(missing),
                "n_classes_without_remaining_train": len(missing),
                "all_nine_classes_in_remaining_train": set(classes).issubset(
                    set(remaining_train["label"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("holdout_approx", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "splits-dir",
        "baseline-validation",
        "baseline-test",
        "hpo-validation",
        "hpo-test",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    with (Path("config") / "dataset.yaml").open(encoding="utf-8") as handle:
        classes = list(yaml.safe_load(handle)["dataset"]["classes"])
    if len(classes) != 9:
        raise ValueError("Se esperaban nueve clases contractuales")
    eligible, split_hashes = load_frozen_splits(args.splits_dir)
    if set(classes) != set(eligible["label"]):
        raise ValueError("Las clases de config/dataset.yaml no coinciden con el master congelado")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    distribution = source_distribution(eligible)
    matrix = source_class_distribution(eligible, classes)
    distribution.to_csv(output / "source_distribution.csv", index=False)
    matrix.to_csv(output / "source_class_distribution.csv", index=False)
    pred_paths = {
        ("validation", "baseline"): args.baseline_validation,
        ("validation", "hpo_trial_0"): args.hpo_validation,
        ("test", "baseline"): args.baseline_test,
        ("test", "hpo_trial_0"): args.hpo_test,
    }
    predictions = {}
    source_tables = []
    class_tables = []
    for (split, model), path in pred_paths.items():
        subset = eligible.loc[eligible["split"] == split]
        predictions[(split, model)] = attach_predictions(path, subset)
        source_tables.append(source_metrics(predictions[(split, model)], model, split))
        class_tables.append(class_metrics(predictions[(split, model)], model, split, classes))
    by_source = pd.concat(source_tables, ignore_index=True)
    by_class = pd.concat(class_tables, ignore_index=True)
    by_source.loc[by_source["split"] == "validation"].to_csv(
        output / "validation_metrics_by_source.csv", index=False
    )
    by_source.loc[by_source["split"] == "test"].to_csv(
        output / "test_metrics_by_source.csv", index=False
    )
    by_class.to_csv(output / "class_metrics_by_source.csv", index=False)
    by_class.loc[by_class["class"].isin(NUTRITION_CLASSES)].to_csv(
        output / "nutrition_metrics_by_source.csv", index=False
    )
    comparisons = {}
    for split in ("validation", "test"):
        comparisons[split] = comparison(by_source.loc[by_source["split"] == split]).assign(
            split=split
        )
    pd.concat(comparisons.values(), ignore_index=True).to_csv(
        output / "baseline_vs_hpo_by_source.csv", index=False
    )
    selected = set(distribution.loc[distribution["total"] >= 500, "source_id"])
    selected.update(("maize-diseases", "multicrop-disease-maiz"))
    confusion = pd.concat(
        [
            confusion_rows(frame, model, split, classes, selected)
            for (split, model), frame in predictions.items()
        ],
        ignore_index=True,
    )
    confusion.to_csv(output / "confusion_by_source.csv", index=False)
    feasibility = loso_feasibility(eligible, classes)
    feasibility.to_csv(output / "loso_feasibility.csv", index=False)
    extension = eligible.assign(
        file_extension=eligible["image_path"].map(lambda path: Path(path).suffix.lower())
    )
    extension.groupby(["source_id", "file_extension"]).size().rename("count").reset_index().to_csv(
        output / "source_file_extensions.csv", index=False
    )
    summary = {
        "status": "complete_descriptive",
        "corpus": "corn-clean:/clean",
        "splits": "corn-outputs:/splits/seed_42",
        "split_hashes": split_hashes,
        "prediction_sha256": {
            f"{split}_{model}": sha256_file(path) for (split, model), path in pred_paths.items()
        },
        "n_eligible": len(eligible),
        "split_counts": {
            key: int(value) for key, value in eligible["split"].value_counts().items()
        },
        "n_sources": len(distribution),
        "source_label_association": association(matrix),
        "mean_source_macro_f1": {
            f"{split}_{model}": float(
                by_source.loc[
                    (by_source["split"] == split) & (by_source["model"] == model), "macro_f1"
                ].mean()
            )
            for split, model in pred_paths
        },
        "interpretation": "Post-hoc descriptive analysis; no model selection on test",
    }
    (output / "source_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sources": len(distribution),
                "association_cramers_v": summary["source_label_association"]["cramers_v"],
                "outputs": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
