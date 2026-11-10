"""Calcula el Disparate Impact Ratio entre laboratorio y campo sobre las clases compartidas.

Uso:
    python -m scripts.experiments.equidad_dir \
        --predicciones oficial_20260921_204608=<predicciones_test.csv> \
        etapa2_20260812_221429=<predicciones_test.csv> \
        --salida docs/es/resultados/evidencia/equidad_dir_clases_compartidas.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from scripts.experiments.figura_equidad_entorno import COMPARTIDAS, metricas
from src.data.preparation import sha256_file


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--predicciones",
        nargs="+",
        required=True,
        metavar="NOMBRE=CSV",
        help="CSV por imagen con columnas environment, pred_label y label o true_label.",
    )
    parser.add_argument("--salida", type=Path, required=True)
    return parser.parse_args()


def evaluar(ruta: Path) -> dict:
    """Calcula métricas por entorno, DIR y F1 por clase sobre las clases compartidas.

    @param {Path} ruta CSV de predicciones de prueba con su entorno.
    @returns {dict} Métricas por entorno, cocientes DIR y F1 por clase y entorno.
    """
    marco = pd.read_csv(ruta)
    if "true_label" not in marco.columns:
        marco = marco.rename(columns={"label": "true_label"})
    restringido = marco[marco.true_label.isin(COMPARTIDAS)]
    por_entorno = {}
    for entorno in ("lab", "real"):
        grupo = restringido[restringido.environment == entorno]
        valores = metricas(grupo)
        por_entorno[entorno] = {
            "muestras": int(len(grupo)),
            "macro_f1": float(valores["Macro F1"]),
            "accuracy": float(valores["Accuracy"]),
        }
    cocientes = {}
    for clave in ("macro_f1", "accuracy"):
        a, b = por_entorno["lab"][clave], por_entorno["real"][clave]
        cocientes[clave] = min(a, b) / max(a, b)
    por_clase = {}
    for clase in COMPARTIDAS:
        por_clase[clase] = {}
        for entorno in ("lab", "real"):
            grupo = marco[marco.environment == entorno]
            f1 = f1_score(
                grupo.true_label, grupo.pred_label, labels=[clase], average="macro", zero_division=0
            )
            por_clase[clase][entorno] = {
                "muestras": int((grupo.true_label == clase).sum()),
                "f1": float(f1),
            }
    return {
        "fuente": ruta.as_posix(),
        "sha256_fuente": sha256_file(ruta),
        "clases_compartidas": list(COMPARTIDAS),
        "por_entorno": por_entorno,
        "dir": cocientes,
        "umbral_cuatro_quintos": 0.8,
        "f1_por_clase": por_clase,
    }


def main() -> None:
    """Evalúa cada CSV solicitado y escribe el resultado en JSON."""
    args = parse_args()
    resultado = {}
    for especificacion in args.predicciones:
        nombre, _, ruta = especificacion.partition("=")
        resultado[nombre] = evaluar(Path(ruta))
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    contenido = json.dumps(resultado, indent=2, ensure_ascii=False) + "\n"
    args.salida.write_text(contenido, encoding="utf-8")


if __name__ == "__main__":
    main()
