"""Genera la figura de convergencia de las tres arquitecturas con los hiperparámetros iniciales.

Las curvas provienen de `train_history.csv` de cada corrida de la segunda etapa sobre la
partición `seed_42`.

Uso:
    python scripts/experiments/figura_convergencia.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CORRIDAS = (
    ("EfficientNet-Lite0", "efficientnet_lite0/20260812_221429", "#c0392b"),
    ("EfficientNet-B0", "efficientnet_b0/20260811_211306", "#2e86c1"),
    ("ShuffleNet-V2-x1.0", "shufflenet_v2_x1_0/20260812_185239", "#3d8b37"),
)


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(description="Figura de convergencia del entrenamiento.")
    parser.add_argument("--raiz", type=Path, default=Path("outputs-remote/main"))
    parser.add_argument("--destinos", type=Path, nargs="+",
                        default=[Path("public/training/training_convergence.png"),
                                 Path("reports/tesina/images/training/training_convergence.png")])
    return parser.parse_args()


def dibujar(raiz: Path, destinos: list[Path]) -> None:
    """Dibuja el Macro F1 de validación y la pérdida por época de cada arquitectura.

    @param {Path} raiz Directorio que contiene las corridas por arquitectura.
    @param {list[Path]} destinos Rutas donde se guarda la figura.
    """
    fig, (ax_f1, ax_loss) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    minimo_f1 = 1.0

    for nombre, corrida, color in CORRIDAS:
        historial = pd.read_csv(raiz / corrida / "train_history.csv")
        mejor = historial.loc[historial["val_macro_f1"].idxmax()]
        minimo_f1 = min(minimo_f1, historial["val_macro_f1"].min())

        ax_f1.plot(historial["epoch"], historial["val_macro_f1"], color=color, linewidth=2,
                   label=f"{nombre} (máximo {mejor['val_macro_f1']:.4f} en la época {int(mejor['epoch'])})")
        ax_f1.scatter(mejor["epoch"], mejor["val_macro_f1"], color=color, s=45, zorder=3)

        ax_loss.plot(historial["epoch"], historial["train_loss"], color=color, linewidth=2, label=nombre)
        ax_loss.plot(historial["epoch"], historial["val_loss"], color=color, linewidth=1.5, linestyle="--")

    ax_f1.set_title("a) Macro $F_1$ en validación", loc="left", fontsize=13, fontweight="bold")
    ax_f1.set_ylabel("Macro $F_1$", fontsize=12)
    ax_f1.set_ylim(max(0.0, minimo_f1 - 0.02), 1.0)
    ax_f1.legend(loc="lower right", fontsize=10)

    ax_loss.plot([], [], color="gray", linewidth=2, label="Entrenamiento")
    ax_loss.plot([], [], color="gray", linewidth=1.5, linestyle="--", label="Validación")
    ax_loss.set_title("b) Pérdida", loc="left", fontsize=13, fontweight="bold")
    ax_loss.set_ylabel("Pérdida", fontsize=12)
    ax_loss.set_xlabel("Época", fontsize=12)
    ax_loss.legend(loc="upper right", fontsize=10, ncol=2)

    for ax in (ax_f1, ax_loss):
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    for destino in destinos:
        destino.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    argumentos = parse_args()
    dibujar(argumentos.raiz, argumentos.destinos)
