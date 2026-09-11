"""Módulo de Equidad Algorítmica y Análisis de Sesgos (Criterio 4 - Rúbrica Etapa 2).

Proporciona funciones para:
1. Evaluación desagregada de métricas por subgrupos (Entorno de Laboratorio vs Campo Real).
2. Cálculo de métricas de disparidad (Δ_F1, Disparate Impact Ratio DIR, Error Rate Ratio).
3. Análisis de Falsos Negativos (FNR = 1 - Recall) por clase (mayoritarias vs minoritarias).
4. Test de control negativo contra atajos visuales (Shortcut Learning / Efecto Clever Hans).
5. Visualización de matrices de confusión desagregadas y gráficos de disparidad.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _compute_confusion_matrix_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    normalize: bool = False,
) -> np.ndarray:
    """Calcula la matriz de confusión en NumPy puro sin depender de scipy/sklearn DLLs."""
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1.0

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return cm


def _compute_classification_metrics_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> tuple[float, float, float, float, list[float]]:
    """Calcula Accuracy, Macro F1, Macro Precision, Macro Recall y Recall por clase en NumPy puro."""
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0, [0.0] * num_classes

    acc = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return acc, macro_f1, macro_prec, macro_rec, recalls


def compute_subgroup_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    subgroups: list[str] | np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Calcula métricas de rendimiento desagregadas por subgrupo (p.ej. 'lab' vs 'real')

    y la tasa de falsos negativos (FNR) por clase dentro de cada subgrupo.

    Returns:
        dict con métricas desagregadas por subgrupo y métricas globales.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    subgroups_arr = np.asarray(subgroups)
    num_classes = len(class_names)

    unique_subgroups = sorted(list(set(subgroups_arr)))

    # Global
    g_acc, g_f1, g_prec, g_rec, g_recalls = _compute_classification_metrics_np(
        y_true_arr, y_pred_arr, num_classes
    )

    results: dict[str, Any] = {
        "subgroups": {},
        "overall": {
            "macro_f1": g_f1,
            "accuracy": g_acc,
            "macro_precision": g_prec,
            "macro_recall": g_rec,
            "sample_count": int(len(y_true_arr)),
            "class_fnr": {
                cls: float(1.0 - g_recalls[i]) for i, cls in enumerate(class_names)
            },
        },
    }

    for group in unique_subgroups:
        mask = subgroups_arr == group
        if not np.any(mask):
            continue

        yt_g = y_true_arr[mask]
        yp_g = y_pred_arr[mask]

        acc_g, f1_g, prec_g, rec_g, recalls_g = _compute_classification_metrics_np(
            yt_g, yp_g, num_classes
        )

        fnr_by_class = {
            cls: float(1.0 - recalls_g[i]) for i, cls in enumerate(class_names)
        }

        results["subgroups"][str(group)] = {
            "sample_count": int(np.sum(mask)),
            "macro_f1": f1_g,
            "accuracy": acc_g,
            "macro_precision": prec_g,
            "macro_recall": rec_g,
            "class_fnr": fnr_by_class,
        }

    return results


def compute_disparity_metrics(subgroup_results: dict[str, Any]) -> dict[str, Any]:
    """Calcula las brechas de paridad matemática entre subgrupos (especialmente 'lab' vs 'real').

    Métricas calculadas:
    - delta_macro_f1: |F1_real - F1_lab|
    - delta_accuracy: |Acc_real - Acc_lab|
    - disparate_impact_ratio: min(F1_1, F1_2) / max(F1_1, F1_2)
    - four_fifths_rule_passed: True si DIR >= 0.80
    - max_fnr_disparity_by_class: Máxima diferencia de FNR entre subgrupos por cada patología
    """
    subgroups = subgroup_results.get("subgroups", {})
    if "lab" not in subgroups or "real" not in subgroups:
        keys = list(subgroups.keys())
        if len(keys) < 2:
            return {
                "delta_macro_f1": 0.0,
                "delta_accuracy": 0.0,
                "disparate_impact_ratio": 1.0,
                "four_fifths_rule_passed": True,
                "note": "Menos de 2 subgrupos disponibles.",
            }
        g1, g2 = keys[0], keys[1]
    else:
        g1, g2 = "real", "lab"

    m1 = subgroups[g1]
    m2 = subgroups[g2]

    f1_1 = m1["macro_f1"]
    f1_2 = m2["macro_f1"]
    acc_1 = m1["accuracy"]
    acc_2 = m2["accuracy"]

    delta_f1 = float(abs(f1_1 - f1_2))
    delta_acc = float(abs(acc_1 - acc_2))

    max_f1 = max(f1_1, f1_2)
    min_f1 = min(f1_1, f1_2)
    dir_ratio = float(min_f1 / max_f1) if max_f1 > 1e-6 else 1.0
    passed_80_rule = bool(dir_ratio >= 0.80)

    # Disparidad de FNR por patología
    fnr_1 = m1.get("class_fnr", {})
    fnr_2 = m2.get("class_fnr", {})
    fnr_disparity = {
        cls: float(abs(fnr_1.get(cls, 0.0) - fnr_2.get(cls, 0.0)))
        for cls in fnr_1
    }

    return {
        "group_a": g1,
        "group_b": g2,
        "delta_macro_f1": round(delta_f1, 4),
        "delta_accuracy": round(delta_acc, 4),
        "disparate_impact_ratio": round(dir_ratio, 4),
        "four_fifths_rule_passed": passed_80_rule,
        "fnr_disparity_by_class": fnr_disparity,
    }


def evaluate_background_shortcut(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_mode: str = "center_occlusion",
) -> dict[str, Any]:
    """Test de ablación contra atajos visuales (Clever Hans Effect).

    Modos soportados:
    - 'center_occlusion': Ocluye el 60% central (donde reside la lesión patológica).
      Si el modelo retiene alta confianza mirando solo la periferia/fondo, confirma
      aprendizaje de atajos espurios (Shortcut Learning).
    - 'peripheral_occlusion': Ocluye el 40% periférico dejando visible únicamente
      el 60% central (lesión pura sin entorno). Si la confianza o exactitud colapsa
      al retirar el fondo, confirma que la red dependía del contexto exterior.
    - 'edge_only': Modo complementario de oclusión del 80% central.
    """
    model.eval()
    orig_confidences: list[float] = []
    masked_confidences: list[float] = []
    orig_correct: int = 0
    masked_correct: int = 0
    correct_flips: int = 0
    total_samples: int = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            bs = images.size(0)
            total_samples += bs

            # Inferencia original
            logits_orig = model(images)
            probs_orig = torch.softmax(logits_orig, dim=-1)
            conf_orig, preds_orig = torch.max(probs_orig, dim=-1)
            orig_confidences.extend(conf_orig.cpu().tolist())
            orig_correct += int((preds_orig == targets).sum().item())

            # Crear imagen enmascarada
            _, _, h, w = images.shape
            h_start, h_end = int(h * 0.2), int(h * 0.8)
            w_start, w_end = int(w * 0.2), int(w * 0.8)

            if mask_mode == "center_occlusion":
                # Ocluir el 60% central (tapar la lesión, dejar solo fondo/bordes)
                masked_images = images.clone()
                masked_images[:, :, h_start:h_end, w_start:w_end] = 0.0
            elif mask_mode in ("peripheral_occlusion", "inverse_occlusion", "center_only"):
                # Control inverso: tapar el 40% periférico, dejando visible ÚNICAMENTE el 60% central
                masked_images = torch.zeros_like(images)
                masked_images[:, :, h_start:h_end, w_start:w_end] = images[:, :, h_start:h_end, w_start:w_end]
            elif mask_mode == "edge_only":
                # Ocluir el 80% central dejando solo bordes extremos
                h_edge_start, h_edge_end = int(h * 0.1), int(h * 0.9)
                w_edge_start, w_edge_end = int(w * 0.1), int(w * 0.9)
                masked_images = images.clone()
                masked_images[:, :, h_edge_start:h_edge_end, w_edge_start:w_edge_end] = 0.0
            else:
                raise ValueError(f"mask_mode '{mask_mode}' no reconocido.")

            # Inferencia sobre imagen enmascarada
            logits_masked = model(masked_images)
            probs_masked = torch.softmax(logits_masked, dim=-1)
            conf_masked, preds_masked = torch.max(probs_masked, dim=-1)
            masked_confidences.extend(conf_masked.cpu().tolist())
            masked_correct += int((preds_masked == targets).sum().item())

            # Contar aciertos originales que cambiaron a error (flip)
            flips = (preds_masked != targets) & (preds_orig == targets)
            correct_flips += int(flips.sum().item())

    mean_orig_conf = float(np.mean(orig_confidences)) if orig_confidences else 0.0
    mean_masked_conf = float(np.mean(masked_confidences)) if masked_confidences else 0.0
    confidence_drop = float(mean_orig_conf - mean_masked_conf)
    shortcut_vulnerability_score = float(mean_masked_conf / (mean_orig_conf + 1e-6))

    acc_orig = float(orig_correct / total_samples) if total_samples > 0 else 0.0
    acc_masked = float(masked_correct / total_samples) if total_samples > 0 else 0.0
    acc_drop = float(acc_orig - acc_masked)
    flip_rate = float(correct_flips / max(orig_correct, 1))

    # Diagnóstico riguroso según la dirección de la oclusión:
    if mask_mode == "center_occlusion":
        # En oclusión central: si la confianza se mantiene alta (poca caída) o retiene > 75%,
        # significa que la red clasifica por fondo/bordes -> ALERTA DE ATAJO CONFIRMADA.
        collapse_confirmed = bool(confidence_drop > 0.40 and shortcut_vulnerability_score < 0.60)
        shortcut_detected = bool(shortcut_vulnerability_score >= 0.75 or confidence_drop < 0.20)
        risk_level = "CRITICAL" if shortcut_detected else ("MODERATE" if confidence_drop < 0.35 else "LOW")
    elif mask_mode in ("peripheral_occlusion", "inverse_occlusion", "center_only"):
        # En control inverso (solo centro):
        # Si la precisión o confianza colapsa al retirar el fondo, el modelo depende del fondo.
        collapse_confirmed = bool(acc_drop > 0.25 or confidence_drop > 0.30)
        shortcut_detected = bool(acc_drop > 0.20 or confidence_drop > 0.25)
        risk_level = "CRITICAL" if shortcut_detected else ("MODERATE" if acc_drop > 0.10 else "LOW")
    else:
        collapse_confirmed = bool(confidence_drop > 0.15 or shortcut_vulnerability_score < 0.75)
        shortcut_detected = not collapse_confirmed
        risk_level = "MODERATE"

    return {
        "mask_mode": mask_mode,
        "total_samples": total_samples,
        "mean_original_confidence": round(mean_orig_conf, 4),
        "mean_masked_confidence": round(mean_masked_conf, 4),
        "confidence_drop": round(confidence_drop, 4),
        "shortcut_vulnerability_ratio": round(shortcut_vulnerability_score, 4),
        "accuracy_original": round(acc_orig, 4),
        "accuracy_masked": round(acc_masked, 4),
        "accuracy_drop": round(acc_drop, 4),
        "flip_rate": round(flip_rate, 4),
        "collapse_confirmed": collapse_confirmed,
        "shortcut_detected": shortcut_detected,
        "risk_level": risk_level,
    }


def evaluate_dual_shortcut_audit(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Ejecuta una auditoría dual completa de atajos visuales:

    1. Oclusión Central: mide la retención espuria de certeza ante pérdida de la lesión.
    2. Oclusión Periférica (Control Inverso): mide la capacidad diagnóstica sobre la lesión pura sin fondo.
    """
    center_res = evaluate_background_shortcut(model, loader, device, mask_mode="center_occlusion")
    peripheral_res = evaluate_background_shortcut(model, loader, device, mask_mode="peripheral_occlusion")

    # Diagnóstico conjunto
    shortcut_confirmed = bool(center_res["shortcut_detected"] or peripheral_res["shortcut_detected"])

    if center_res["shortcut_vulnerability_ratio"] >= 0.85:
        verdict = (
            f"VULNERABILIDAD SEVERA (Clever Hans Confirmado): El modelo retiene el "
            f"{center_res['shortcut_vulnerability_ratio']*100:.1f}% de su confianza sin ver la lesión central. "
            f"El {100 - center_res['confidence_drop']*100:.1f}% de su comportamiento está anclado a artefactos periféricos."
        )
    else:
        verdict = "Comportamiento dentro de márgenes esperados de atención foliar."

    return {
        "center_occlusion": center_res,
        "peripheral_occlusion": peripheral_res,
        "shortcut_confirmed": shortcut_confirmed,
        "overall_risk_level": "CRITICAL" if shortcut_confirmed else "LOW",
        "diagnostic_summary": verdict,
    }


def plot_disaggregated_confusion_matrices(
    y_true_lab: list[int] | np.ndarray,
    y_pred_lab: list[int] | np.ndarray,
    y_true_real: list[int] | np.ndarray,
    y_pred_real: list[int] | np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Genera y guarda una figura con las matrices de confusión normalizadas de Campo Real vs Laboratorio lado a lado."""
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
    axes[0].set_title("Matriz de Confusión: Entorno Laboratorio (lab)", fontsize=12, fontweight="bold", pad=12)
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
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")

    # Matriz Campo Real
    cm_real = _compute_confusion_matrix_np(
        np.asarray(y_true_real),
        np.asarray(y_pred_real),
        num_classes=num_classes,
        normalize=True,
    )
    im1 = axes[1].imshow(cm_real, interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Matriz de Confusión: Entorno Campo Real (real)", fontsize=12, fontweight="bold", pad=12)
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
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")

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
    """Genera un gráfico de barras comparativo de Macro F1, Accuracy, Precision y Recall entre subgrupos."""
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
        rects = ax.bar(x + offset, values, width, label=f"Entorno: {g}", color=colors[idx % len(colors)], alpha=0.85)
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
    ax.set_title("Comparativa de Rendimiento por Subgrupo de Entorno (Fairness Audit)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.80, color="gray", linestyle="--", alpha=0.6, label="Umbral 80% (Fairness Reference)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico de barras de disparidad guardado en: %s", output_path)
