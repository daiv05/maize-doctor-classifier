"""Genera la figura de ablación espacial con su control nulo y el área visible de cada condición.

Las cifras provienen de `shortcut_learning_audit` en el fichero de métricas de equidad, que se
calcula sobre las 5 015 imágenes del conjunto de prueba con el checkpoint desplegado.

Uso:
    python scripts/experiments/figura_ablacion.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LADO_ENTRADA = 224
RECORTE = (int(LADO_ENTRADA * 0.2), int(LADO_ENTRADA * 0.8))
AREA_CENTRAL = ((RECORTE[1] - RECORTE[0]) / LADO_ENTRADA) ** 2
COLORES = ("#7f8c8d", "#c0392b", "#2e86c1")


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(description="Figura de ablación espacial con control nulo.")
    parser.add_argument("--metricas", type=Path,
                        default=Path("docs/es/resultados/evidencia/equidad_metricas.json"))
    parser.add_argument("--destinos", type=Path, nargs="+",
                        default=[Path("public/resultados/ablacion_control_nulo.png"),
                                 Path("reports/second-phase/images/resultados/ablacion_control_nulo.png")])
    return parser.parse_args()


def leer_condiciones(metricas: Path) -> tuple[list[str], list[float], list[float], float, int]:
    """Extrae las exactitudes de cada condición de oclusión y el control nulo.

    @param {Path} metricas Fichero de métricas de equidad.
    @returns {tuple} Etiquetas, exactitudes, áreas visibles, exactitud nula y tamaño de muestra.
    """
    datos = json.loads(metricas.read_text(encoding="utf-8"))
    auditoria = datos["shortcut_learning_audit"]
    centro = auditoria["center_occlusion"]
    periferia = auditoria["peripheral_occlusion"]

    etiquetas = [
        "Imagen\ncompleta",
        f"Sólo el anillo periférico\n(área visible {1 - AREA_CENTRAL:.1%})",
        f"Sólo la caja central\n(área visible {AREA_CENTRAL:.1%})",
    ]
    exactitudes = [
        centro["accuracy_original"],
        centro["accuracy_masked"],
        periferia["accuracy_masked"],
    ]
    areas = [1.0, 1 - AREA_CENTRAL, AREA_CENTRAL]
    return etiquetas, exactitudes, areas, datos["control_nulo_clase_mayoritaria"], centro["total_samples"]


def main() -> None:
    """Dibuja la figura y la escribe en cada destino solicitado."""
    args = parse_args()
    etiquetas, exactitudes, _, nulo, muestras = leer_condiciones(args.metricas)

    figura, eje = plt.subplots(figsize=(12, 7.5))
    barras = eje.bar(etiquetas, exactitudes, color=COLORES, width=0.62)

    for barra, valor in zip(barras, exactitudes):
        eje.text(barra.get_x() + barra.get_width() / 2, valor + 0.018, f"{valor:.4f}",
                 ha="center", fontsize=15, fontweight="bold")
        eje.text(barra.get_x() + barra.get_width() / 2, valor - 0.08,
                 f"+{valor - nulo:.3f}\nsobre el nulo", ha="center", color="white",
                 fontsize=12, fontweight="bold")

    eje.axhline(nulo, color="#d4a017", linestyle="--", linewidth=2.5)
    eje.text(0.335, nulo + 0.018, f"control nulo  {nulo:.4f}", transform=eje.get_yaxis_transform(),
             ha="center", color="#b8860b", fontsize=12, fontweight="bold")

    eje.set_title("Exactitud según la región visible, frente al control nulo",
                  fontsize=16, fontweight="bold", pad=16)
    eje.set_ylabel(f"Exactitud sobre test ({muestras:,} imágenes)".replace(",", " "), fontsize=12)
    eje.set_ylim(0, 1.08)
    eje.grid(axis="y", linestyle=":", alpha=0.5)
    eje.set_axisbelow(True)

    figura.text(0.5, 0.015,
                "Las dos condiciones de oclusión no están igualadas en área: el anillo muestra "
                "1.75 veces más píxeles que la caja central.",
                ha="center", fontsize=11, style="italic", color="#555555")
    figura.tight_layout(rect=(0, 0.04, 1, 1))

    for destino in args.destinos:
        destino.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(destino, dpi=130)
        print(f"[*] {destino}")
    plt.close(figura)


if __name__ == "__main__":
    main()
