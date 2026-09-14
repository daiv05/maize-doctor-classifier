"""Genera la comparativa de equidad por entorno restringida a las clases comparables.

El subgrupo de laboratorio solo cubre tres de las nueve clases del corpus, de modo que un macro
promediado sobre todas las categorias de cada subgrupo compara conjuntos distintos. Esta figura
restringe ambos subgrupos a las tres clases que comparten y deja la exactitud global como
referencia sobre el conjunto completo.

Uso:
    python scripts/experiments/figura_equidad_entorno.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

COMPARTIDAS = ("common_rust", "gray_leaf_spot", "northern_corn_leaf_blight")
COLORES = {"lab": "#3d6f9e", "real": "#4a9160"}


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(description="Comparativa de equidad por entorno.")
    parser.add_argument("--predicciones", type=Path,
                        default=Path("outputs/fase_g/por_entorno/fairness_predictions.csv"))
    parser.add_argument("--particion", type=Path,
                        default=Path("outputs/splits/seed_42/test.csv"))
    parser.add_argument("--destinos", type=Path, nargs="+",
                        default=[Path("public/fairness/fairness_disparity.png"),
                                 Path("reports/second-phase/images/fairness/fairness_disparity.png")])
    return parser.parse_args()


def unir(predicciones: Path, particion: Path) -> pd.DataFrame:
    """Añade la columna de entorno a las predicciones por imagen.

    @param {Path} predicciones CSV con una fila por imagen del conjunto de prueba.
    @param {Path} particion CSV de la particion de prueba, que registra el entorno.
    @returns {pd.DataFrame} Predicciones con su entorno asignado.
    """
    izquierda = pd.read_csv(predicciones)
    derecha = pd.read_csv(particion)
    clave = lambda serie: serie.astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]
    izquierda["clave"] = clave(izquierda.image_path)
    derecha["clave"] = clave(derecha.image_path)
    unido = izquierda.merge(derecha[["clave", "environment"]], on="clave", how="left")
    if unido.environment.isna().any():
        raise ValueError(f"{int(unido.environment.isna().sum())} imagenes sin entorno asignado.")
    return unido


def metricas(marco: pd.DataFrame) -> dict[str, float]:
    """Calcula las cuatro metricas sobre las clases compartidas.

    @param {pd.DataFrame} marco Predicciones de un subgrupo, ya filtradas.
    @returns {dict[str, float]} Macro F1, exactitud, macro precision y macro recall.
    """
    verdad, prediccion = marco.true_label, marco.pred_label
    comun = {"labels": list(COMPARTIDAS), "average": "macro", "zero_division": 0}
    return {
        "Macro F1": f1_score(verdad, prediccion, **comun),
        "Accuracy": accuracy_score(verdad, prediccion),
        "Precision": precision_score(verdad, prediccion, **comun),
        "Recall": recall_score(verdad, prediccion, **comun),
    }


def main() -> None:
    """Dibuja la comparativa y la escribe en cada destino solicitado."""
    args = parse_args()
    unido = unir(args.predicciones, args.particion)
    restringido = unido[unido.true_label.isin(COMPARTIDAS)]

    valores = {env: metricas(restringido[restringido.environment == env]) for env in COLORES}
    tamanos = {env: int((restringido.environment == env).sum()) for env in COLORES}
    etiquetas = list(valores["lab"].keys())
    posiciones = np.arange(len(etiquetas))
    ancho = 0.38

    figura, eje = plt.subplots(figsize=(13, 7.5))
    for desplazamiento, (env, color) in zip((-ancho / 2, ancho / 2), COLORES.items()):
        alturas = [valores[env][m] for m in etiquetas]
        barras = eje.bar(posiciones + desplazamiento, alturas, ancho, color=color,
                         label=f"Entorno: {env} (N = {tamanos[env]:,})".replace(",", " "))
        for barra, altura in zip(barras, alturas):
            eje.text(barra.get_x() + barra.get_width() / 2, altura + 0.012, f"{altura:.3f}",
                     ha="center", fontsize=12, fontweight="bold")

    eje.axhline(0.80, color="#999999", linestyle="--", linewidth=1.8,
                label="Umbral 80 % (referencia de equidad)")
    eje.set_title("Equidad por entorno sobre las tres clases presentes en ambos subgrupos",
                  fontsize=16, fontweight="bold", pad=16)
    eje.set_ylabel("Puntuación (0 - 1.0)", fontsize=12, fontweight="bold")
    eje.set_xticks(posiciones)
    eje.set_xticklabels(etiquetas, fontsize=13, fontweight="bold")
    eje.set_ylim(0, 1.12)
    eje.grid(axis="y", linestyle=":", alpha=0.5)
    eje.set_axisbelow(True)
    eje.legend(loc="lower right", fontsize=11)

    dir_f1 = min(valores["lab"]["Macro F1"], valores["real"]["Macro F1"]) / max(
        valores["lab"]["Macro F1"], valores["real"]["Macro F1"])
    dir_acc = min(valores["lab"]["Accuracy"], valores["real"]["Accuracy"]) / max(
        valores["lab"]["Accuracy"], valores["real"]["Accuracy"])
    figura.text(0.5, 0.015,
                f"Ratio de impacto dispar: {dir_f1:.4f} en Macro F1 y {dir_acc:.4f} en exactitud. "
                f"Clases comparadas: {', '.join(COMPARTIDAS)}.",
                ha="center", fontsize=11, style="italic", color="#555555")
    figura.tight_layout(rect=(0, 0.04, 1, 1))

    for destino in args.destinos:
        destino.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(destino, dpi=130)
        print(f"[*] {destino}")
    plt.close(figura)


if __name__ == "__main__":
    main()
