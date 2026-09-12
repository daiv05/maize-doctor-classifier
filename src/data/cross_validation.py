"""Módulo de Validación Cruzada Estratificada Jerárquica (Criterio 3 - Rúbrica Etapa 2).

Genera K folds estratificados considerando conjuntamente la patología (label)
y el entorno de captura (environment: lab vs real) para garantizar evaluación
estadísticamente rigurosa sin fuga de datos.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class KFoldSplit:
    """Representa una partición individual (Fold) con índices o DataFrames."""

    fold_index: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame


class HierarchicalKFoldSplitter:
    """Generador de K-Folds estratificados jerárquicamente por Clase y Entorno."""

    def __init__(self, n_splits: int = 5, seed: int = 42, shuffle: bool = True) -> None:
        if n_splits < 2:
            raise ValueError("n_splits debe ser al menos 2.")
        self.n_splits = n_splits
        self.seed = seed
        self.shuffle = shuffle
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    def split(self, data_df: pd.DataFrame) -> list[KFoldSplit]:
        """Divide el DataFrame en K folds homogéneos y disjuntos."""
        if "label" not in data_df.columns:
            raise ValueError("El DataFrame debe contener la columna 'label'.")

        # Generar súper-etiqueta jerárquica si existe columna 'environment'
        if "environment" in data_df.columns:
            stratify_col = data_df["label"].astype(str) + "_" + data_df["environment"].astype(str)
        else:
            stratify_col = data_df["label"].astype(str)

        splits: list[KFoldSplit] = []
        indices = np.arange(len(data_df))

        for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(data_df, stratify_col)):
            train_df = data_df.iloc[train_idx].copy().reset_index(drop=True)
            val_df = data_df.iloc[val_idx].copy().reset_index(drop=True)
            splits.append(KFoldSplit(fold_index=fold_idx + 1, train_df=train_df, val_df=val_df))

        return splits


def compute_aggregate_statistics(metrics_list: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Calcula media, desviación estándar e intervalo de confianza al 95% para cada métrica."""
    if not metrics_list:
        return {}

    keys = list(metrics_list[0].keys())
    k = len(metrics_list)
    t_val = 1.96 if k >= 30 else 2.776 if k == 5 else 2.0  # Aproximación t de Student para 95% CI

    summary: dict[str, Any] = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if not values:
            continue
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        margin_error = float(t_val * (std_val / np.sqrt(k))) if k > 1 else 0.0

        summary[key] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "ci_95_lower": round(max(0.0, mean_val - margin_error), 4),
            "ci_95_upper": round(min(1.0, mean_val + margin_error), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
        }

    return summary


def plot_kfold_boxplot(
    metrics_list: Sequence[dict[str, float]],
    output_path: Path,
    model_name: str = "EfficientNet-B0",
) -> None:
    """Genera un gráfico Boxplot mostrando la estabilidad de métricas a través de los K folds."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys_to_plot = ["macro_f1", "accuracy", "macro_precision", "macro_recall"]
    labels = ["Macro F1", "Accuracy", "Macro Precision", "Macro Recall"]

    data = []
    actual_labels = []
    for key, label in zip(keys_to_plot, labels):
        vals = [m[key] for m in metrics_list if key in m]
        if vals:
            data.append(vals)
            actual_labels.append(label)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(data, patch_artist=True, tick_labels=actual_labels, medianprops=dict(color="black", linewidth=1.5))

    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Añadir puntos de cada fold
    for i, vals in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.plot(x, vals, "r.", alpha=0.8, markersize=8)

    ax.set_title(f"Validación Cruzada ({len(metrics_list)} Folds) — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Puntuación (0.0 a 1.0)", fontsize=11)
    ax.set_ylim(bottom=max(0.0, min([min(v) for v in data]) - 0.05), top=1.02)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    fig.tight_layout()

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
