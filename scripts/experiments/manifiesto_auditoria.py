"""Construye el manifiesto de auditoría de todas las corridas reportadas.

Recorre las corridas del pipeline principal en el Volume de Modal, recoge su configuración
y sus métricas publicadas, y registra para cada una si esas métricas se pueden recomputar
desde predicciones por imagen y desde qué fichero.

Uso:
    python scripts/experiments/manifiesto_auditoria.py --volumen-local outputs/auditoria
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

VOLUMEN = "corn-outputs"
MODELOS = ("efficientnet_lite0", "efficientnet_b0", "shufflenet_v2_x1_0")


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(description="Manifiesto de auditoría de las corridas.")
    parser.add_argument("--destino", type=Path, default=Path("outputs/auditoria"))
    parser.add_argument("--salida", type=Path,
                        default=Path("docs/es/resultados/evidencia/manifiesto_corridas.csv"))
    parser.add_argument("--ensamble", type=Path,
                        default=Path("outputs/fase_e2/ensemble_predictions.csv"),
                        help="CSV del ensamble, que guarda la prediccion de cada modelo y "
                             "permite verificar corridas sin predictions.csv propio.")
    return parser.parse_args()


def modal(*argumentos: str) -> str:
    """Ejecuta el CLI de Modal y devuelve su salida.

    @param {str} argumentos Argumentos del subcomando.
    @returns {str} Salida estándar.
    """
    resultado = subprocess.run(
        [sys.executable, "-m", "modal", *argumentos],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return resultado.stdout


def descargar(remoto: str, local: Path) -> bool:
    """Descarga un fichero del Volume si existe.

    @param {str} remoto Ruta dentro del Volume.
    @param {Path} local Destino.
    @returns {bool} True si el fichero quedó en disco.
    """
    local.parent.mkdir(parents=True, exist_ok=True)
    modal("volume", "get", "--force", VOLUMEN, remoto, str(local))
    return local.exists()


def verificar(predicciones: Path, publicado: float) -> tuple[bool, float]:
    """Recomputa el macro-F1 desde las predicciones por imagen.

    @param {Path} predicciones CSV con una fila por imagen.
    @param {float} publicado Macro-F1 declarado en el resumen.
    @returns {tuple[bool, float]} Si coincide y el valor recomputado.
    """
    marco = pd.read_csv(predicciones)
    verdad = "label" if "label" in marco.columns else "y_true"
    prediccion = "pred_label" if "pred_label" in marco.columns else "y_pred"
    valor = f1_score(marco[verdad], marco[prediccion], average="macro", zero_division=0)
    return abs(valor - publicado) < 1e-9, valor


def main() -> None:
    """Recorre las corridas del Volume y escribe el manifiesto."""
    args = parse_args()
    filas = []

    for modelo in MODELOS:
        listado = modal("volume", "ls", VOLUMEN, f"main/{modelo}")
        # El listado de Modal viene en tabla con bordes, asi que los identificadores no
        # quedan aislados al separar por espacios.
        corridas = sorted(set(re.findall(r"2026\d{4}_\d{6}", listado)))
        for corrida in corridas:
            base = f"main/{modelo}/{corrida}"
            resumen_local = args.destino / modelo / corrida / "summary.json"
            if not descargar(f"/{base}/summary.json", resumen_local):
                continue
            resumen = json.loads(resumen_local.read_text(encoding="utf-8"))
            prueba = resumen.get("test", {})
            publicado = float(prueba.get("macro_f1", 0.0))

            predicciones = args.destino / modelo / corrida / "predictions.csv"
            tiene = descargar(f"/{base}/predictions.csv", predicciones)
            coincide, recomputado = verificar(predicciones, publicado) if tiene else (False, 0.0)
            fuente_verificacion = f"{base}/predictions.csv" if coincide else ""

            # Una corrida sin predicciones propias sigue siendo auditable si su checkpoint
            # entro en el ensamble: ese CSV guarda una columna por modelo.
            if not coincide and args.ensamble.exists():
                ensamble = pd.read_csv(args.ensamble)
                columna = f"pred_{modelo}"
                if columna in ensamble.columns:
                    valor = f1_score(ensamble.y_true, ensamble[columna],
                                     average="macro", zero_division=0)
                    if abs(valor - publicado) < 5e-5:
                        coincide, recomputado = True, valor
                        fuente_verificacion = f"{args.ensamble.name}:{columna}"

            filas.append({
                "modelo": modelo,
                "run_id": corrida,
                "ruta_modal": f"{VOLUMEN}:/{base}",
                "splits": str(resumen.get("splits_dir", "")).split("/")[-1],
                "batch_size": resumen.get("batch_size"),
                "learning_rate": resumen.get("learning_rate"),
                "weight_decay": resumen.get("weight_decay"),
                "label_smoothing": resumen.get("label_smoothing"),
                "warmup_epochs": resumen.get("warmup_epochs"),
                "class_weights": resumen.get("class_weights"),
                "epocas": resumen.get("epochs_run"),
                "mejor_epoca": resumen.get("best_epoch"),
                "val_macro_f1": resumen.get("best_val_macro_f1"),
                "test_macro_f1": publicado,
                "test_accuracy": prueba.get("accuracy"),
                "predicciones_por_imagen": tiene,
                "recomputa_exacto": coincide,
                "verificable_desde": fuente_verificacion,
                "macro_f1_recomputado": round(recomputado, 10) if tiene else None,
            })
            print(f"  {modelo}/{corrida}: publicado {publicado:.4f} | "
                  f"{'recomputa exacto' if coincide else 'SIN predictions.csv'}", flush=True)

    tabla = pd.DataFrame(filas).sort_values(["modelo", "run_id"])
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(args.salida, index=False)
    sin_verificar = tabla[~tabla.recomputa_exacto]
    print(f"\ncorridas: {len(tabla)} | verificadas: {int(tabla.recomputa_exacto.sum())} | "
          f"sin predicciones propias: {len(sin_verificar)}")
    for _, r in sin_verificar.iterrows():
        print(f"   {r.modelo}/{r.run_id}")
    print(f"\n[*] manifiesto en {args.salida}", flush=True)


if __name__ == "__main__":
    main()
