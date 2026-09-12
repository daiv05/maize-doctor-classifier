"""Evalua un modelo ya exportado (.onnx / .tflite) sobre el split de test completo.

La validacion de paridad que corre dentro de `export.py` compara PyTorch contra el archivo
exportado sobre una muestra pequena y solo mira diferencias numericas. Este script mide
accuracy y macro-F1 reales del artefacto sobre todo el split de test, que es el numero que
decide si el modelo se puede embarcar en la app.

Uso:
    python scripts/pipeline/evaluate_export.py --models shufflenet_v2_x1_0 --formats onnx,tflite
    python scripts/pipeline/evaluate_export.py --models efficientnet_b0 --quantize int8
"""

import argparse
import json
import logging
from pathlib import Path

from src.config import PROJECT_ROOT, get_output_root
from src.export.common import (
    export_artifact_name,
    load_checkpoint_for_export,
    parse_export_formats,
    parse_quantize,
    resolve_export_inputs,
)
from src.export.data import build_test_loader, resolve_test_csv
from src.export.evaluate import evaluate_exported_model, write_evaluation
from src.models import list_models
from src.provenance import sha256_file
from src.training.common import resolve_run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua modelos exportados sobre el split de test completo."
    )
    parser.add_argument("--models", nargs="+", required=True, choices=list_models())
    parser.add_argument("--run", default=None, help="run_id; por defecto usa latest.json.")
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio de runs del pipeline principal (default: <outputs>/main).",
    )
    parser.add_argument(
        "--formats", default="onnx", help="Formatos a evaluar, CSV (ej: 'onnx,tflite')."
    )
    parser.add_argument(
        "--quantize",
        default=None,
        help="Variante a evaluar: 'int8' o 'none' (default: none / FP32).",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con test.csv (default: el 'splits_dir' de summary.json).",
    )
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--no-torch-baseline",
        action="store_true",
        dest="no_torch_baseline",
        help="No corre el modelo PyTorch de referencia (mas rapido, sin deltas).",
    )
    parser.add_argument(
        "--max-macro-f1-drop",
        type=float,
        default=0.01,
        dest="max_macro_f1_drop",
        help="Caida maxima de macro-F1 vs PyTorch antes de fallar (default: 0.01).",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    return parser.parse_args()


def _evaluate_one(args: argparse.Namespace, model_name: str, output_dir: Path) -> bool:
    """
    Evalua todas las variantes pedidas de un modelo. Devuelve True si alguna falló.

    @param {argparse.Namespace} args Argumentos ya parseados.
    @param {str} model_name Modelo a evaluar.
    @param {Path} output_dir Directorio de runs del pipeline principal.
    @returns {bool} True si algun formato falto o degrado mas de lo permitido.
    """
    config_path = Path(args.config)
    formats = parse_export_formats(args.formats)
    quantize = parse_quantize(args.quantize)

    run_dir = resolve_run_dir(output_dir, model_name, args.run)
    class_to_idx, idx_to_class, image_size = resolve_export_inputs(run_dir, model_name, config_path)
    device = select_device()

    test_csv = resolve_test_csv(run_dir, args.splits_dir)
    test_loader, environments = build_test_loader(
        test_csv,
        config_path,
        class_to_idx,
        image_size,
        args.batch_size,
        preprocessing=json.loads((run_dir / "summary.json").read_text())["preprocessing"],
    )

    torch_model = None
    if not args.no_torch_baseline:
        torch_model = load_checkpoint_for_export(
            run_dir / "best.pth", model_name, class_to_idx, device
        )

    print(f"\nModelo: {model_name}  (cuantizacion: {quantize or 'none'})")
    print(f"Run: {run_dir}")

    failed = False
    for format_name in formats:
        model_path = run_dir / "export" / export_artifact_name(format_name, quantize)
        if not model_path.exists():
            print(f"  [FALTA] {format_name}: no existe {model_path}")
            failed = True
            continue

        evaluation, frame = evaluate_exported_model(
            model_path,
            format_name,
            test_loader,
            idx_to_class,
            torch_model=torch_model,
            device=device,
            quantize=quantize,
            environments=environments if len(environments) else None,
        )
        evaluation.evaluated_split_sha256 = sha256_file(test_csv)
        json_path = write_evaluation(run_dir, evaluation, frame)

        print(
            f"  [{format_name}] n={evaluation.n_samples} "
            f"accuracy={evaluation.accuracy:.4f} macro_f1={evaluation.macro_f1:.4f}"
        )
        if evaluation.torch_macro_f1 is not None:
            print(
                f"    PyTorch: accuracy={evaluation.torch_accuracy:.4f} "
                f"macro_f1={evaluation.torch_macro_f1:.4f} | "
                f"delta_macro_f1={evaluation.macro_f1_delta:+.4f} "
                f"acuerdo={evaluation.agreement_rate:.4f}"
            )
            if evaluation.macro_f1_delta < -args.max_macro_f1_drop:
                print(
                    f"    FALLO: la caida de macro-F1 supera el limite "
                    f"({args.max_macro_f1_drop:.4f})"
                )
                failed = True
        for environment, metrics in evaluation.by_environment.items():
            print(
                f"    {environment}: n={metrics['n']} "
                f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
            )
        print(f"    reporte: {json_path}")

    return failed


def main() -> None:
    args = _parse_args()
    output_root = get_output_root()
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "main"

    if not parse_export_formats(args.formats):
        raise SystemExit("Debes indicar al menos un formato en --formats.")

    failures = [
        model_name for model_name in args.models if _evaluate_one(args, model_name, output_dir)
    ]

    if failures:
        print(f"\nModelos con problemas: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
