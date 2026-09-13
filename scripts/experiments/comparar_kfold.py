"""Compara las celdas del experimento 2x2 de validación cruzada.

Las cuatro celdas cruzan protocolo de partición (estratificada frente a agrupada por
procedencia) con configuración (valores por defecto frente a hiperparámetros afinados).
Cada cifra se recomputa desde las predicciones por imagen de cada pliegue, no se lee del
resumen, y se acompaña del acierto de un predictor constante de la clase mayoritaria.

Uso:
    python scripts/experiments/comparar_kfold.py --raiz outputs/fase_f
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

CELDAS = {
    "estratificado_actual": ("estratificada", "por defecto"),
    "estratificado_afinada": ("estratificada", "afinada"),
    "agrupado_actual": ("agrupada", "por defecto"),
    "agrupado_afinada": ("agrupada", "afinada"),
}


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(description="Compara las celdas del 2x2 de K-Fold.")
    parser.add_argument("--raiz", type=Path, default=Path("outputs/fase_f"))
    parser.add_argument("--salida", type=Path, default=Path("outputs/fase_f/comparacion.csv"))
    return parser.parse_args()


def metricas_de_pliegue(csv: Path) -> dict[str, float]:
    """Recomputa las métricas de un pliegue desde sus predicciones por imagen.

    @param {Path} csv Ruta al predictions.csv del pliegue.
    @returns {dict[str, float]} Métricas del pliegue, incluido el control nulo.
    """
    marco = pd.read_csv(csv)
    con_soporte = sorted(marco.y_true.unique())
    mayoritaria = marco.y_true.value_counts().iloc[0] / len(marco)
    return {
        "n": len(marco),
        "clases_con_soporte": len(con_soporte),
        "accuracy": accuracy_score(marco.y_true, marco.y_pred),
        "macro_f1": f1_score(marco.y_true, marco.y_pred, average="macro", zero_division=0),
        "macro_f1_evaluable": f1_score(
            marco.y_true, marco.y_pred, average="macro", labels=con_soporte, zero_division=0
        ),
        "control_nulo": mayoritaria,
        "fuentes": marco.source_id.nunique() if "source_id" in marco.columns else np.nan,
    }


def resumir_celda(directorio: Path) -> dict[str, object] | None:
    """Agrega los pliegues de una celda con media e intervalo de confianza al 95 %.

    @param {Path} directorio Carpeta de la celda.
    @returns {dict | None} Resumen de la celda, o None si no hay pliegues.
    """
    pliegues = sorted(directorio.glob("fold_*/predictions.csv"))
    if not pliegues:
        return None

    filas = [metricas_de_pliegue(p) for p in pliegues]
    resumen: dict[str, object] = {"pliegues": len(filas)}
    for clave in ("macro_f1_evaluable", "macro_f1", "accuracy", "control_nulo"):
        valores = np.array([f[clave] for f in filas], dtype=float)
        media = valores.mean()
        # t de Student a 95 % con n-1 grados de libertad, aproximada para k pequeño.
        t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(valores), 1.96)
        margen = t * valores.std(ddof=1) / np.sqrt(len(valores)) if len(valores) > 1 else 0.0
        resumen[clave] = media
        resumen[f"{clave}_ic95"] = margen
    resumen["clases_min"] = min(f["clases_con_soporte"] for f in filas)
    resumen["clases_max"] = max(f["clases_con_soporte"] for f in filas)
    return resumen


def main() -> None:
    """Construye la tabla comparativa de las cuatro celdas."""
    args = parse_args()
    filas = []
    for carpeta, (protocolo, configuracion) in CELDAS.items():
        resumen = resumir_celda(args.raiz / carpeta)
        if resumen is None:
            print(f"[!] sin pliegues en {carpeta}", flush=True)
            continue
        filas.append({"celda": carpeta, "protocolo": protocolo,
                      "configuracion": configuracion, **resumen})

    if not filas:
        raise SystemExit("No se encontraron pliegues en ninguna celda.")

    tabla = pd.DataFrame(filas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(args.salida, index=False)

    print(f"\n{'celda':26s} {'macroF1_ev':>11s} {'IC95':>8s} {'accuracy':>9s} "
          f"{'nulo':>7s} {'clases':>8s} {'k':>3s}")
    for _, r in tabla.iterrows():
        print(f"{r.celda:26s} {r.macro_f1_evaluable:11.4f} {r.macro_f1_evaluable_ic95:8.4f} "
              f"{r.accuracy:9.4f} {r.control_nulo:7.4f} "
              f"{int(r.clases_min)}-{int(r.clases_max):<6d} {int(r.pliegues):3d}")

    for configuracion in tabla.configuracion.unique():
        par = tabla[tabla.configuracion == configuracion].set_index("protocolo")
        if {"estratificada", "agrupada"} <= set(par.index):
            caida = (par.loc["estratificada", "macro_f1_evaluable"]
                     - par.loc["agrupada", "macro_f1_evaluable"])
            print(f"\n  {configuracion}: estratificada -> agrupada = {caida:+.4f} de macro-F1")

    print(f"\n[*] tabla en {args.salida}", flush=True)


if __name__ == "__main__":
    main()
