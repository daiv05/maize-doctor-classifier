"""Compile locked multicrop ensemble predictions without model inference or tuning."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from scripts.experiments.source_stability_report import stats
from src.data.preparation import sha256_file
from src.training.evaluation import expected_calibration_error

SOURCE = "multicrop-disease-maiz"
SEEDS = (42, 123, 2026)
MEMBERS = ("efficientnet_lite0", "efficientnet_b0", "shufflenet_v2_x1_0")
MODELS = (*MEMBERS, "ensemble")
HISTORICAL_DELTA = 0.9566569450594972 - 0.9483330101876505
OUTPUT = Path("outputs/ensemble_loso") / SOURCE
EVIDENCE = Path("docs/es/reproducibilidad/evidencia/ensemble_loso")
LITE = Path("outputs/loso/efficientnet_lite0_baseline") / SOURCE


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_errors(predictions: dict[str, pd.DataFrame], seed: int) -> tuple[pd.DataFrame, dict]:
    """Describe paired corrections and member diversity on aligned sample IDs."""
    ensemble = predictions["ensemble"]
    ids = ensemble.index
    if ensemble.index.has_duplicates:
        raise ValueError("sample_id duplicado en ensemble")
    for model in MEMBERS:
        frame = predictions[model]
        if frame.index.has_duplicates or set(frame.index) != set(ids):
            raise ValueError(f"sample_id no alineado: {model}")
        if not frame.loc[ids, "label"].equals(ensemble["label"]):
            raise ValueError(f"Etiquetas incompatibles: {model}")
    reference = predictions["efficientnet_b0"].loc[ids]
    truth = ensemble["label"]
    correct_ref = reference["pred_label"].eq(truth)
    correct_ens = ensemble["pred_label"].eq(truth)
    member_preds = {model: predictions[model].loc[ids, "pred_label"] for model in MEMBERS}
    member_wrong = {model: ~member_preds[model].eq(truth) for model in MEMBERS}
    disagreement = pd.DataFrame(member_preds).nunique(axis=1).gt(1)
    corrected = ~correct_ref & correct_ens
    introduced = correct_ref & ~correct_ens
    all_wrong = pd.DataFrame(member_wrong).all(axis=1)
    selection = corrected | introduced | all_wrong | disagreement
    errors = pd.DataFrame(
        {
            "seed": seed,
            "sample_id": ids,
            "label": truth.values,
            "reference_pred": reference["pred_label"].values,
            "reference_confidence": reference["confidence"].values,
            "ensemble_pred": ensemble["pred_label"].values,
            "ensemble_confidence": ensemble["confidence"].values,
            **{f"pred_{model}": member_preds[model].values for model in MEMBERS},
            "corrected_by_ensemble": corrected.values,
            "introduced_by_ensemble": introduced.values,
            "all_members_wrong": all_wrong.values,
            "member_disagreement": disagreement.values,
        }
    )
    errors = errors.loc[selection.values].copy()
    diversity = {
        "seed": seed,
        "corrected_count": int(corrected.sum()),
        "corrected_fraction": float(corrected.mean()),
        "introduced_count": int(introduced.sum()),
        "introduced_fraction": float(introduced.mean()),
        "all_members_wrong_count": int(all_wrong.sum()),
        "all_members_wrong_fraction": float(all_wrong.mean()),
        "member_disagreement_count": int(disagreement.sum()),
        "member_disagreement_fraction": float(disagreement.mean()),
        "corrected_mean_confidence": float(ensemble.loc[corrected, "confidence"].mean())
        if corrected.any()
        else None,
        "introduced_mean_confidence": float(ensemble.loc[introduced, "confidence"].mean())
        if introduced.any()
        else None,
        "pairwise_disagreement": {},
        "pairwise_shared_errors": {},
        "exclusive_errors": {},
        "corrected_by_class": {
            name: int((corrected & truth.eq(name)).sum()) for name in sorted(truth.unique())
        },
        "introduced_by_class": {
            name: int((introduced & truth.eq(name)).sum()) for name in sorted(truth.unique())
        },
    }
    for i, first in enumerate(MEMBERS):
        diversity["exclusive_errors"][first] = int(
            (
                member_wrong[first]
                & ~pd.DataFrame({m: member_wrong[m] for m in MEMBERS if m != first}).any(axis=1)
            ).sum()
        )
        for second in MEMBERS[i + 1 :]:
            key = f"{first}__{second}"
            diversity["pairwise_disagreement"][key] = float(
                member_preds[first].ne(member_preds[second]).mean()
            )
            diversity["pairwise_shared_errors"][key] = int(
                (member_wrong[first] & member_wrong[second]).sum()
            )
    return errors, diversity


def one_seed(seed: int) -> tuple[list[dict], list[dict], pd.DataFrame, dict, dict]:
    root = OUTPUT / f"seed_{seed}"
    group_path = root / "ensemble.selection.lock.json"
    group = read_json(group_path)
    out = root / "evaluation"
    guard = read_json(out / "final_evaluation.started.json")
    group_sha = sha256_file(group_path)
    if (
        group["seed"] != seed
        or group["held_out_source"] != SOURCE
        or group["historical_ensemble_evidence_sha256"]
        != "dcafbb4c21e2d69646b8a04cd1bbf425474b615f5eb6b44d594abd469b6f90c2"
        or [member["model"] for member in group["members"]] != list(MEMBERS)
        or guard["evaluation_count"] != 1
        or guard["ensemble_selection_lock_sha256"] != group_sha
        or datetime.fromisoformat(group["locked_at_utc"])
        >= datetime.fromisoformat(guard["started_at_utc"])
    ):
        raise ValueError(f"Selección/evaluación inválida en seed_{seed}")
    rows, class_rows, predictions = [], [], {}
    files = {
        "ensemble_selection_lock": group_path,
        "evaluation_guard": out / "final_evaluation.started.json",
    }
    for model in MODELS:
        metric_path = out / f"{model}_metrics.json"
        pred_path = out / f"{model}_predictions.csv"
        class_path = out / f"{model}_class_metrics.csv"
        metrics = read_json(metric_path)
        pred = pd.read_csv(pred_path, dtype={"sample_id": str})
        classes = pd.read_csv(class_path)
        if (
            metrics["seed"] != seed
            or metrics["model"] != model
            or metrics["evaluation_count"] != 1
            or metrics["ensemble_selection_lock_sha256"] != group_sha
            or metrics["samples"] != 5816
            or len(pred) != 5816
            or pred["sample_id"].duplicated().any()
        ):
            raise ValueError(f"Artefactos inválidos: {model}/seed_{seed}")
        present = sorted(pred["label"].unique())
        if present != ["common_rust", "fall_armyworm", "lethal_necrosis"]:
            raise ValueError("Holdout de clases distintas")
        calculated = f1_score(
            pred["label"], pred["pred_label"], labels=present, average="macro", zero_division=0
        )
        if abs(calculated - metrics["macro_f1"]) > 1e-12:
            raise ValueError("Macro-F1 no recomputa")
        if abs(accuracy_score(pred["label"], pred["pred_label"]) - metrics["accuracy"]) > 1e-12:
            raise ValueError("Accuracy no recomputa")
        correct = pred["label"].eq(pred["pred_label"])
        ece = expected_calibration_error(pred["confidence"].tolist(), correct.tolist())
        if abs(ece - metrics["ece_15_bins"]) > 1e-12:
            raise ValueError("ECE no recomputa")
        precision, recall, f1, support = precision_recall_fscore_support(
            pred["label"], pred["pred_label"], labels=present, zero_division=0
        )
        expected_classes = (
            pd.DataFrame(
                {
                    "class": present,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }
            )
            .sort_values("class")
            .reset_index(drop=True)
        )
        recorded_classes = classes.sort_values("class").reset_index(drop=True)
        if not expected_classes["class"].equals(recorded_classes["class"]):
            raise ValueError("Clases reportadas distintas")
        for column in ("precision", "recall", "f1", "support"):
            if not np.allclose(expected_classes[column], recorded_classes[column], atol=1e-12):
                raise ValueError(f"Métricas por clase distintas: {column}")
        probability_columns = [column for column in pred if column.startswith("prob_")]
        if len(probability_columns) != 9:
            raise ValueError("Vector de probabilidades incompleto")
        probability_frame = pred[probability_columns]
        if not np.allclose(probability_frame.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Probabilidades no normalizadas")
        argmax = probability_frame.idxmax(axis=1).str.removeprefix("prob_")
        if not argmax.equals(pred["pred_label"]):
            raise ValueError("Predicción no corresponde al argmax")
        if not np.allclose(probability_frame.max(axis=1), pred["confidence"], atol=1e-6):
            raise ValueError("Confianza no corresponde a la probabilidad máxima")
        rows.append({"seed": seed, "model": model, **metrics})
        for _, class_row in classes.iterrows():
            class_rows.append({"seed": seed, "model": model, **class_row.to_dict()})
        predictions[model] = pred.set_index("sample_id", verify_integrity=True)
        files[f"{model}_metrics"] = metric_path
        files[f"{model}_predictions"] = pred_path
        files[f"{model}_class_metrics"] = class_path
        files[f"{model}_confusion_matrix"] = out / f"{model}_confusion_matrix.csv"
    ids = set(predictions["ensemble"].index)
    for model in MODELS:
        if set(predictions[model].index) != ids:
            raise ValueError("Componentes no alinean por sample_id")
        if (
            not predictions[model]
            .loc[predictions["ensemble"].index, "label"]
            .equals(predictions["ensemble"]["label"])
        ):
            raise ValueError("Etiquetas verdaderas incompatibles")
    legacy_root = LITE / f"seed_{seed}" / "runs" / "efficientnet_lite0"
    legacy_runs = [path for path in legacy_root.iterdir() if (path / "summary.json").is_file()]
    if len(legacy_runs) != 1:
        raise ValueError("Referencia Lite0 previa ambigua")
    legacy = pd.read_csv(
        legacy_runs[0] / "holdout_predictions.csv", dtype={"sample_id": str}
    ).set_index("sample_id", verify_integrity=True)
    if set(legacy.index) != ids or not legacy.loc[
        predictions["efficientnet_lite0"].index, "pred_label"
    ].equals(predictions["efficientnet_lite0"]["pred_label"]):
        raise ValueError("Reinferencia Lite0 no coincide con evaluación LOSO previa")
    errors, diversity = analyze_errors(predictions, seed)
    registry = {
        "seed": seed,
        "evaluation_count": 1,
        "ensemble_selection_lock_sha256": group_sha,
        "locked_at_utc": group["locked_at_utc"],
        "started_at_utc": guard["started_at_utc"],
        "finished_at_utc": read_json(out / "ensemble_metrics.json")["finished_at_utc"],
        **{f"sha256_{name}": sha256_file(path) for name, path in files.items()},
    }
    return rows, class_rows, errors, diversity, registry


def paired_results(components: pd.DataFrame) -> pd.DataFrame:
    """Pair models strictly within the same seed, never by input row order."""
    results = []
    for seed in SEEDS:
        indexed = components.loc[components.seed.eq(seed)].set_index("model")
        if set(indexed.index) != set(MODELS) or indexed.index.duplicated().any():
            raise ValueError(f"Faltan modelos o hay duplicados en seed_{seed}")
        reference = float(indexed.loc["efficientnet_b0", "macro_f1"])
        lite = float(indexed.loc["efficientnet_lite0", "macro_f1"])
        ensemble = float(indexed.loc["ensemble", "macro_f1"])
        results.append(
            {
                "seed": seed,
                "reference_b0_macro_f1": reference,
                "baseline_lite0_macro_f1": lite,
                "shufflenet_macro_f1": float(indexed.loc["shufflenet_v2_x1_0", "macro_f1"]),
                "ensemble_macro_f1": ensemble,
                "delta_vs_b0": ensemble - reference,
                "delta_vs_lite0": ensemble - lite,
                "reference_b0_accuracy": float(indexed.loc["efficientnet_b0", "accuracy"]),
                "ensemble_accuracy": float(indexed.loc["ensemble", "accuracy"]),
            }
        )
    return pd.DataFrame(results)


def compile_report() -> dict:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    all_rows, all_classes, all_errors, diversities, registries = [], [], [], [], []
    for seed in SEEDS:
        rows, classes, errors, diversity, registry = one_seed(seed)
        all_rows.extend(rows)
        all_classes.extend(classes)
        all_errors.append(errors)
        diversities.append(diversity)
        registries.append(registry)
    components = pd.DataFrame(all_rows).sort_values(["seed", "model"])
    classes = pd.DataFrame(all_classes).sort_values(["seed", "model", "class"])
    errors = pd.concat(all_errors, ignore_index=True)
    results_frame = paired_results(components)
    deltas = results_frame["delta_vs_b0"].tolist()
    summary = {
        "protocol": "historical_ensemble_multicrop_loso_v1",
        "held_out_source": SOURCE,
        "seeds": list(SEEDS),
        "n_trainings_new": 6,
        "n_reused_lite0": 3,
        "holdout_samples": 5816,
        "historical_test_delta_vs_b0": HISTORICAL_DELTA,
        "reference_b0_macro_f1": stats(results_frame["reference_b0_macro_f1"].tolist()),
        "baseline_lite0_macro_f1": stats(results_frame["baseline_lite0_macro_f1"].tolist()),
        "ensemble_macro_f1": stats(results_frame["ensemble_macro_f1"].tolist()),
        "paired_delta_vs_b0": stats(deltas),
        "paired_delta_vs_lite0": stats(results_frame["delta_vs_lite0"].tolist()),
        "positive_seeds": sum(delta > 0 for delta in deltas),
        "negative_seeds": sum(delta < 0 for delta in deltas),
        "tied_seeds": sum(delta == 0 for delta in deltas),
        "class_summary": {},
        "diversity": diversities,
        "global_metric_summary": {},
    }
    for metric in (
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "weighted_f1",
        "ece_15_bins",
        "mean_confidence_correct",
        "mean_confidence_errors",
        "error_rate",
    ):
        reference_values = (
            components.loc[components.model.eq("efficientnet_b0")]
            .sort_values("seed")[metric]
            .tolist()
        )
        ensemble_values = (
            components.loc[components.model.eq("ensemble")].sort_values("seed")[metric].tolist()
        )
        summary["global_metric_summary"][metric] = {
            "reference_b0": stats(reference_values),
            "ensemble": stats(ensemble_values),
            "paired_delta": stats(
                [new - old for old, new in zip(reference_values, ensemble_values, strict=True)]
            ),
        }
    for class_name in ("common_rust", "fall_armyworm", "lethal_necrosis"):
        summary["class_summary"][class_name] = {}
        for model in MODELS:
            subset = classes.loc[
                classes["class"].eq(class_name) & classes.model.eq(model)
            ].sort_values("seed")
            if len(subset) != 3:
                raise ValueError("Faltan métricas por clase")
            summary["class_summary"][class_name][model] = {
                metric: stats(subset[metric].tolist()) for metric in ("precision", "recall", "f1")
            }
            summary["class_summary"][class_name][model]["support"] = int(subset.support.iloc[0])
    components.to_csv(EVIDENCE / "ensemble_component_results.csv", index=False)
    classes.to_csv(EVIDENCE / "ensemble_loso_class_metrics.csv", index=False)
    results_frame.to_csv(EVIDENCE / "ensemble_loso_results.csv", index=False)
    errors.to_csv(EVIDENCE / "ensemble_disagreements.csv", index=False)
    pd.DataFrame(registries).to_csv(EVIDENCE / "ensemble_evaluation_registry.csv", index=False)
    (EVIDENCE / "ensemble_loso_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    references = {
        "manifest_lock_sha256": "d8aac805a3d768166a63be664f1a658478926ab7d44a48e617c950f795e1537b",
        "holdout_sha256": "d36f938bdb6db7505722e5b7cf6458adc5568f3a94525884943e47eb0a8b0688",
        "historical_evidence_sha256": sha256_file(
            Path("docs/es/resultados/evidencia/ensamble_resumen.json")
        ),
        "historical_b0_checkpoint_sha256": (
            "0e367a5318c068c9bcc90a6bec0d17a0b029251ca5f98400034bd09de7524fdb"
        ),
        "historical_shuffle_checkpoint_sha256": (
            "b681a3fd5fffaa5d078802369dff04b46c91cbfed7b333ea80c9d6965fd82c9e"
        ),
        "members": [],
    }
    split_lock_path = LITE / "seed_42/splits/manifest.lock.json"
    if sha256_file(split_lock_path) != references["manifest_lock_sha256"]:
        raise ValueError("Manifest local distinto")
    split_lock = read_json(split_lock_path)
    references["derived_manifest_sha256"] = split_lock["derived_sha256"]
    references["frozen_manifest_sha256"] = split_lock["frozen_sha256"]
    for seed in SEEDS:
        group = read_json(OUTPUT / f"seed_{seed}" / "ensemble.selection.lock.json")
        for member in group["members"]:
            run_dir = Path(member["run_dir"])
            local_run = Path(str(run_dir).removeprefix("/outputs/"))
            if not local_run.exists():
                local_run = Path("outputs") / local_run
            if not local_run.exists():
                raise FileNotFoundError(local_run)
            if sha256_file(local_run / "best.pth") != member["checkpoint_sha256"]:
                raise ValueError("Checkpoint local distinto")
            item = {"seed": seed, **member, "local_run_dir": str(local_run)}
            if member["model"] != "efficientnet_lite0":
                metadata = read_json(local_run.parents[2] / "experiment_metadata.json")
                item["duration_seconds"] = metadata["duration_seconds"]
                item["gpu"] = metadata["environment"]["gpu"]
            references["members"].append(item)
    (EVIDENCE / "manifest_checkpoint_references.json").write_text(
        json.dumps(references, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for seed in SEEDS:
        matrix_path = OUTPUT / f"seed_{seed}" / "evaluation/ensemble_confusion_matrix.csv"
        pd.read_csv(matrix_path, index_col=0).to_csv(
            EVIDENCE / f"ensemble_confusion_matrix_seed_{seed}.csv"
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [str(seed) for seed in SEEDS],
        deltas,
        color=["#168466" if delta > 0 else "#b94d4d" for delta in deltas],
    )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axhline(HISTORICAL_DELTA, color="#4671a6", linestyle="--", label="Delta histórico, test")
    ax.set_ylabel("Delta Macro-F1 vs B0 LOSO")
    ax.set_xlabel("Semilla")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVIDENCE / "ensemble_paired_deltas.png", dpi=160)
    plt.close(fig)
    matrices = [
        pd.read_csv(EVIDENCE / f"ensemble_confusion_matrix_seed_{seed}.csv", index_col=0)
        for seed in SEEDS
    ]
    summed = matrices[0] + matrices[1] + matrices[2]
    summed.to_csv(EVIDENCE / "ensemble_confusion_matrix_aggregate.csv")
    normalized = summed.div(summed.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(normalized.columns)), normalized.columns, rotation=90)
    ax.set_yticks(range(len(normalized.index)), normalized.index)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    fig.colorbar(image, ax=ax, label="Proporción por fila")
    fig.tight_layout()
    fig.savefig(EVIDENCE / "ensemble_confusion_normalized.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    compile_report()
