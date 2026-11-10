"""Genera la figura de los 25 ensayos de la búsqueda de hiperparámetros de EfficientNet-Lite0.

Cada ensayo completado o podado se dibuja con su mejor Macro F1 de validación, junto con la
configuración inicial como referencia. Los ensayos cortados al ajustar el presupuesto de cómputo
no tienen valor y se marcan al pie de la gráfica.

Uso:
    python scripts/experiments/figura_hpo_ensayos.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EVIDENCIA = Path("docs/es/reproducibilidad/evidencia/hpo_lite0_seed42")
ESTILOS = {
    "COMPLETE": ("Completado", "#2e86c1", "o"),
    "PRUNED": ("Podado", "#a6acaf", "o"),
}


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(description="Figura de los ensayos de Optuna para EfficientNet-Lite0.")
    parser.add_argument("--evidencia", type=Path, default=EVIDENCIA)
    parser.add_argument("--destinos", type=Path, nargs="+",
                        default=[Path("reports/tesina/images/tuning/hpo_ensayos.png")])
    return parser.parse_args()


def dibujar(evidencia: Path, destinos: list[Path]) -> None:
    """Dibuja el mejor Macro F1 de validación de cada ensayo según su estado.

    @param {Path} evidencia Carpeta con `trials.csv` y `comparison_vs_baseline.csv`.
    @param {list[Path]} destinos Rutas donde se guarda la figura.
    """
    ensayos = pd.read_csv(evidencia / "trials.csv")
    comparacion = pd.read_csv(evidencia / "comparison_vs_baseline.csv")
    referencia = comparacion.loc[comparacion["configuration"].str.startswith("baseline"), "val_macro_f1"].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    con_valor = ensayos.dropna(subset=["value"]).sort_values("trial_number")
    ax.plot(con_valor["trial_number"], con_valor["value"], color="#7f8c8d", linewidth=1.2, alpha=0.7, zorder=2)
    for estado, (etiqueta, color, marcador) in ESTILOS.items():
        grupo = ensayos[ensayos["state"] == estado]
        ax.scatter(grupo["trial_number"], grupo["value"], s=70, color=color, marker=marcador,
                   edgecolor="white", linewidth=0.8, zorder=3, label=f"{etiqueta} ({len(grupo)})")

    cortados = ensayos[ensayos["state"] == "FAIL"]
    piso = ensayos["value"].min() - 0.015
    ax.scatter(cortados["trial_number"], [piso] * len(cortados), s=70, color="#c0392b", marker="x",
               linewidth=2, zorder=3, label=f"Cortado por presupuesto ({len(cortados)})")

    ax.axhline(referencia, color="#1e8449", linestyle="--", linewidth=1.6,
               label=f"Configuración inicial ({referencia:.4f})")

    mejor = ensayos.loc[ensayos["value"].idxmax()]
    ax.annotate(f"Mejor ensayo ({mejor['value']:.4f})", (mejor["trial_number"], mejor["value"]),
                xytext=(10, 8), textcoords="offset points", fontsize=11, va="bottom")

    ax.set_xlabel("Ensayo", fontsize=12)
    ax.set_ylabel("Mejor Macro $F_1$ de validación", fontsize=12)
    ax.set_xticks(range(0, len(ensayos), 2))
    ax.set_ylim(piso - 0.01, 0.97)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    fig.tight_layout()
    for destino in destinos:
        destino.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    argumentos = parse_args()
    dibujar(argumentos.evidencia, argumentos.destinos)
