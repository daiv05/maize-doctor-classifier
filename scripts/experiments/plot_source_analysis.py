"""Figuras compactas del paquete de análisis por procedencia ya calculado."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    counts = pd.read_csv(root / "source_distribution.csv")
    matrix = pd.read_csv(root / "source_class_distribution.csv")
    validation = pd.read_csv(root / "validation_metrics_by_source.csv")
    nutrition = pd.read_csv(root / "nutrition_metrics_by_source.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(counts["source_id"][::-1], counts["total"][::-1], color="#356a91")
    ax.set_xlabel("Muestras elegibles")
    ax.set_title("Tamaño de las fuentes en seed_42")
    fig.tight_layout()
    fig.savefig(root / "source_sizes.png", dpi=160)
    plt.close(fig)

    pivot = matrix.pivot(index="source_id", columns="class", values="pct_within_source")
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Composición por clase dentro de cada fuente (%)")
    fig.colorbar(image, ax=ax, label="% dentro de fuente")
    fig.tight_layout()
    fig.savefig(root / "source_label_composition.png", dpi=160)
    plt.close(fig)

    scores = validation.pivot(index="source_id", columns="model", values="macro_f1")
    scores = scores.loc[counts["source_id"]]
    y = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y - 0.18, scores["baseline"], height=0.35, label="Baseline")
    ax.barh(y + 0.18, scores["hpo_trial_0"], height=0.35, label="HPO trial 0")
    ax.set_yticks(y, scores.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro-F1 (solo clases presentes; soportes distintos)")
    ax.set_title("Validation: baseline vs. HPO por fuente")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "validation_macro_f1_by_source.png", dpi=160)
    plt.close(fig)

    potassium = nutrition.loc[
        (nutrition["split"] == "validation")
        & (nutrition["model"] == "hpo_trial_0")
        & (nutrition["class"] == "potassium_deficiency")
    ].sort_values("support", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3))
    bars = ax.barh(potassium["source_id"][::-1], potassium["f1"][::-1], color="#9b6c25")
    for bar, support in zip(bars, potassium["support"][::-1]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"n={support}", va="center"
        )
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 de potassium_deficiency")
    ax.set_title("Validation HPO: potasio por fuente")
    fig.tight_layout()
    fig.savefig(root / "potassium_validation_by_source.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
