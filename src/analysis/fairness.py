"""Descriptive domain performance and spatial sensitivity, without causal certification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data.identity import identified_batches

logger = logging.getLogger(__name__)


def _compute_confusion_matrix_np(y_true, y_pred, num_classes, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(float)
    if normalize:
        sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, sums, out=np.zeros_like(cm), where=sums > 0)
    return cm


def wilson_interval(successes, total):
    """95% binomial interval; undefined when there are no observations."""
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0, center - radius)), float(min(1, center + radius))]


def _metrics(y_true, y_pred, class_names, average_ids=None):
    n = len(class_names)
    cm = _compute_confusion_matrix_np(y_true, y_pred, n)
    support = cm.sum(axis=1).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    ids = list(np.flatnonzero(support)) if average_ids is None else list(average_ids)
    total = len(y_true)
    return {
        "sample_count": total,
        "accuracy": float(np.mean(y_true == y_pred)) if total else None,
        "accuracy_ci95": wilson_interval(int(np.trace(cm)), total),
        "macro_precision": float(np.mean(p[ids])) if ids else None,
        "macro_recall": float(np.mean(r[ids])) if ids else None,
        "macro_f1": float(np.mean(f[ids])) if ids else None,
        "macro_policy": "true-supported classes"
        if average_ids is None
        else "common true-supported classes",
        "averaged_classes": [class_names[i] for i in ids],
        "class_support": {name: int(support[i]) for i, name in enumerate(class_names)},
        "class_fnr": {
            name: float(1 - r[i]) if support[i] else None for i, name in enumerate(class_names)
        },
        "class_recall_ci95": {
            name: wilson_interval(int(cm[i, i]), int(support[i]))
            for i, name in enumerate(class_names)
        },
    }


def _compute_classification_metrics_np(y_true, y_pred, num_classes):
    metrics = _metrics(
        np.asarray(y_true),
        np.asarray(y_pred),
        [str(i) for i in range(num_classes)],
        list(range(num_classes)),
    )
    recalls = [
        None if metrics["class_fnr"][str(i)] is None else 1 - metrics["class_fnr"][str(i)]
        for i in range(num_classes)
    ]
    return (
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["macro_precision"],
        metrics["macro_recall"],
        recalls,
    )


def compute_subgroup_metrics(y_true, y_pred, subgroups, class_names):
    yt, yp, groups = (
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        np.asarray(subgroups, dtype=str),
    )
    if not (len(yt) == len(yp) == len(groups)):
        raise ValueError("Predictions, targets and subgroup identities must have matching lengths")
    if len(class_names) != len(set(class_names)) or not class_names:
        raise ValueError("Class names must be unique and nonempty")
    if (
        np.any(yt < 0)
        or np.any(yp < 0)
        or np.any(yt >= len(class_names))
        or np.any(yp >= len(class_names))
    ):
        raise ValueError("Prediction or target outside class contract")
    unique = sorted(set(groups))
    supported = [set(yt[groups == group]) for group in unique]
    common = sorted(set.intersection(*supported)) if supported else []
    results = {
        "schema_version": 2,
        "overall": _metrics(yt, yp, class_names),
        "common_classes": [class_names[i] for i in common],
        "subgroups": {},
    }
    for group in unique:
        mask = groups == group
        metrics = _metrics(yt[mask], yp[mask], class_names)
        comparable = mask & np.isin(yt, common)
        metrics["comparable_metrics"] = _metrics(
            yt[comparable], yp[comparable], class_names, common
        )
        results["subgroups"][group] = metrics
    return results


def compute_disparity_metrics(subgroup_results):
    groups = subgroup_results.get("subgroups", {})
    keys = ["lab", "real"] if {"lab", "real"} <= set(groups) else sorted(groups)
    result = {
        "status": "insufficient",
        "metric_name": "common_class_macro_f1_ratio",
        "ratio": None,
        "delta_macro_f1": None,
        "delta_accuracy": None,
        "common_classes": subgroup_results.get("common_classes", []),
        "fnr_disparity_by_class": {},
        "threshold": None,
        "conclusion": "Insufficient groups or common-class support; no parity conclusion.",
    }
    if len(keys) != 2 or not result["common_classes"]:
        return result
    a, b = (groups[key].get("comparable_metrics") for key in keys)
    if not a or not b or not a["sample_count"] or not b["sample_count"]:
        return result
    maximum = max(a["macro_f1"], b["macro_f1"])
    result.update(
        status="descriptive",
        group_a=keys[0],
        group_b=keys[1],
        ratio=float(min(a["macro_f1"], b["macro_f1"]) / maximum) if maximum else None,
        delta_macro_f1=abs(a["macro_f1"] - b["macro_f1"]),
        delta_accuracy=abs(a["accuracy"] - b["accuracy"]),
        comparison_support={key: groups[key]["comparable_metrics"]["sample_count"] for key in keys},
        conclusion="Descriptive performance on common classes; not a fairness certification.",
    )
    result["fnr_disparity_by_class"] = {
        name: abs(a["class_fnr"][name] - b["class_fnr"][name]) for name in result["common_classes"]
    }
    return result


def apply_spatial_mask(images, mode, generator=None):
    """Zero normalized pixels; central rectangle spans 60% per axis (~36% area)."""
    _, _, h, w = images.shape
    hs, he, ws, we = int(h * 0.2), int(h * 0.8), int(w * 0.2), int(w * 0.8)
    mask = torch.zeros((len(images), 1, h, w), dtype=torch.bool, device=images.device)
    if mode == "center_occlusion":
        mask[:, :, hs:he, ws:we] = True
    elif mode in ("peripheral_occlusion", "inverse_occlusion", "center_only"):
        mask[:] = True
        mask[:, :, hs:he, ws:we] = False
    elif mode == "edge_only":
        mask[:, :, int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)] = True
    elif mode == "random_occlusion":
        mh, mw = he - hs, we - ws
        for i in range(len(images)):
            top = int(torch.randint(h - mh + 1, (1,), generator=generator))
            left = int(torch.randint(w - mw + 1, (1,), generator=generator))
            mask[i, :, top : top + mh, left : left + mw] = True
    else:
        raise ValueError(f"Unknown mask_mode: {mode}")
    return images.masked_fill(mask, 0.0), float(mask.float().mean())


def evaluate_background_shortcut(model, loader, device, mask_mode="center_occlusion"):
    """Measure sensitivity of a fixed original predicted class, without a causal verdict."""
    if mask_mode not in {
        "center_occlusion",
        "peripheral_occlusion",
        "inverse_occlusion",
        "center_only",
        "edge_only",
        "random_occlusion",
    }:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")
    model.eval()
    records = []
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for images, targets, sample_ids in identified_batches(loader):
            images, targets = images.to(device), targets.to(device)
            original = model(images).softmax(-1)
            predicted = original.argmax(-1)
            masked, fraction = apply_spatial_mask(images, mask_mode, generator)
            occluded = model(masked).softmax(-1)
            after = occluded.argmax(-1)
            original_confidence = original.gather(1, predicted[:, None]).squeeze(1)
            fixed_confidence = occluded.gather(1, predicted[:, None]).squeeze(1)
            for i in range(len(targets)):
                records.append(
                    {
                        "sample_id": None if sample_ids is None else sample_ids[i],
                        "target": int(targets[i]),
                        "fixed_class": int(predicted[i]),
                        "prediction_original": int(predicted[i]),
                        "prediction_masked": int(after[i]),
                        "confidence_original": float(original_confidence[i]),
                        "confidence_masked_fixed_class": float(fixed_confidence[i]),
                        "correct_original": bool(predicted[i] == targets[i]),
                        "correct_masked": bool(after[i] == targets[i]),
                        "masked_fraction": fraction,
                    }
                )
    if not records:
        return {"status": "insufficient", "mask_mode": mask_mode, "total_samples": 0, "samples": []}
    before = np.array([r["confidence_original"] for r in records])
    after = np.array([r["confidence_masked_fixed_class"] for r in records])
    hits = np.array([r["correct_original"] for r in records])
    masked_hits = np.array([r["correct_masked"] for r in records])
    return {
        "status": "sensitivity_only",
        "mask_mode": mask_mode,
        "total_samples": len(records),
        "fixed_class": "original_prediction",
        "mask_fill": "zero in normalized space (normalization mean in RGB)",
        "masked_fraction": float(np.mean([r["masked_fraction"] for r in records])),
        "mean_original_confidence": float(before.mean()),
        "mean_masked_confidence": float(after.mean()),
        "confidence_drop": float((before - after).mean()),
        "confidence_retention_ratio": float(after.mean() / before.mean()),
        "accuracy_original": float(hits.mean()),
        "accuracy_masked": float(masked_hits.mean()),
        "accuracy_drop": float(hits.mean() - masked_hits.mean()),
        "flip_rate": float((hits & ~masked_hits).sum() / hits.sum()) if hits.any() else None,
        "accuracy_masked_ci95": wilson_interval(int(masked_hits.sum()), len(records)),
        "samples": records,
    }


def evaluate_dual_shortcut_audit(model, loader, device):
    results = {
        mode: evaluate_background_shortcut(model, loader, device, mode)
        for mode in ("center_occlusion", "peripheral_occlusion", "random_occlusion")
    }
    return {
        **results,
        "schema_version": 2,
        "status": "sensitivity_only",
        "diagnostic_summary": (
            "Spatial sensitivity; rectangles do not identify leaf, lesion or background. "
            "No causal shortcut conclusion."
        ),
    }


def plot_disaggregated_confusion_matrices(
    y_true_lab: list[int] | np.ndarray,
    y_pred_lab: list[int] | np.ndarray,
    y_true_real: list[int] | np.ndarray,
    y_pred_real: list[int] | np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Matrices de confusión normalizadas de campo y laboratorio, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    num_classes = len(class_names)

    # Matriz Laboratorio
    cm_lab = _compute_confusion_matrix_np(
        np.asarray(y_true_lab),
        np.asarray(y_pred_lab),
        num_classes=num_classes,
        normalize=True,
    )
    im0 = axes[0].imshow(cm_lab, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title(
        "Matriz de Confusión: Entorno Laboratorio (lab)", fontsize=12, fontweight="bold", pad=12
    )
    axes[0].set_xticks(range(num_classes))
    axes[0].set_yticks(range(num_classes))
    axes[0].set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    axes[0].set_yticklabels(class_names, fontsize=9)
    axes[0].set_xlabel("Predicción", fontweight="bold", labelpad=8)
    axes[0].set_ylabel("Etiqueta Real", fontweight="bold")

    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_lab[i, j]
            color = "white" if val > 0.5 else "black"
            axes[0].text(
                j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold"
            )

    # Matriz Campo Real
    cm_real = _compute_confusion_matrix_np(
        np.asarray(y_true_real),
        np.asarray(y_pred_real),
        num_classes=num_classes,
        normalize=True,
    )
    im1 = axes[1].imshow(cm_real, interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title(
        "Matriz de Confusión: Entorno Campo Real (real)", fontsize=12, fontweight="bold", pad=12
    )
    axes[1].set_xticks(range(num_classes))
    axes[1].set_yticks(range(num_classes))
    axes[1].set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    axes[1].set_yticklabels(class_names, fontsize=9)
    axes[1].set_xlabel("Predicción", fontweight="bold", labelpad=8)
    axes[1].set_ylabel("Etiqueta Real", fontweight="bold")

    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_real[i, j]
            color = "white" if val > 0.5 else "black"
            axes[1].text(
                j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold"
            )

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Matrices de confusión desagregadas guardadas en: %s", output_path)


def plot_subgroup_disparity_bars(
    subgroup_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Compara macro-F1, accuracy, precision y recall entre subgrupos."""
    subgroups = subgroup_metrics.get("subgroups", {})
    if not subgroups:
        return

    groups = list(subgroups.keys())
    metrics_names = ["macro_f1", "accuracy", "macro_precision", "macro_recall"]
    display_names = ["Macro F1", "Accuracy", "Precision", "Recall"]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    colors = ["#2b5c8f", "#2e7d32", "#e65100", "#6a1b9a"]
    for idx, g in enumerate(groups):
        values = [subgroups[g].get(m, 0.0) for m in metrics_names]
        offset = (idx - (len(groups) - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            values,
            width,
            label=f"Entorno: {g}",
            color=colors[idx % len(colors)],
            alpha=0.85,
        )
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Puntuación (0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Comparativa de Rendimiento por Subgrupo de Entorno (Fairness Audit)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico de barras de disparidad guardado en: %s", output_path)
