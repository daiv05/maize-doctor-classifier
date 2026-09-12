"""Exporta checkpoints del pipeline principal a ONNX/TFLite.

Opera sobre uno o varios modelos (`--models`), cada uno resuelto a su run más reciente
(o el que indique `--run`). Valida paridad numérica contra una muestra del split de test
y escribe `<run>/export/export_summary[_<quant>].json`.

Para medir el artefacto exportado sobre el split de test **completo**, usar después
`scripts/pipeline/evaluate_export.py` (`make eval-export-main`).
"""

import argparse
import json
import logging
from pathlib import Path

from src.config import PROJECT_ROOT, get_output_root
from src.export.common import (
    export_model,
    load_checkpoint_for_export,
    parse_export_formats,
    parse_quantize,
    resolve_export_inputs,
    write_export_summary,
    write_labels_json,
)
from src.export.data import build_test_loader, resolve_split_csv
from src.models import list_models
from src.models.feature_exposed import FeatureExposedModel
from src.training.common import resolve_run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta checkpoints del pipeline principal a ONNX/TFLite."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=list_models(),
        help="Uno o varios modelos a exportar.",
    )
    parser.add_argument("--run", default=None, help="run_id; por defecto usa latest.json.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta explicita a un checkpoint .pth (solo valido con un unico modelo).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio de runs del pipeline principal (default: <outputs>/main).",
    )
    parser.add_argument(
        "--formats", default="onnx", help="Formatos a exportar, CSV (ej: 'onnx,tflite')."
    )
    parser.add_argument(
        "--quantize",
        default=None,
        help="Cuantizacion a aplicar: 'int8' o 'none' (default: none / FP32).",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con val.csv para paridad (default: 'splits_dir' de summary.json).",
    )
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--parity-sample-size", type=int, default=30, dest="parity_sample_size")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Tolerancia de paridad; por defecto depende de --quantize.",
    )
    parser.add_argument(
        "--min-agreement-rate",
        type=float,
        default=None,
        dest="min_agreement_rate",
        help="Acuerdo top-1 minimo; por defecto depende de --quantize.",
    )
    parser.add_argument(
        "--no-parity",
        action="store_true",
        dest="no_parity",
        help="Omite la validacion de paridad numerica (no recomendado).",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    return parser.parse_args()


def _export_one(args: argparse.Namespace, model_name: str, output_dir: Path) -> bool:
    """
    Exporta un modelo y reporta por stdout. Devuelve True si algo falló.

    @param {argparse.Namespace} args Argumentos ya parseados.
    @param {str} model_name Modelo a exportar.
    @param {Path} output_dir Directorio de runs del pipeline principal.
    @returns {bool} True si hubo un fallo de exportacion o de paridad.
    """
    config_path = Path(args.config)
    formats = parse_export_formats(args.formats)
    quantize = parse_quantize(args.quantize)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        run_dir = checkpoint_path.parent
    else:
        run_dir = resolve_run_dir(output_dir, model_name, args.run)
        checkpoint_path = run_dir / "best.pth"

    class_to_idx, _, image_size = resolve_export_inputs(run_dir, model_name, config_path)
    write_labels_json(run_dir, class_to_idx, model_name, image_size)
    device = select_device()
    model = load_checkpoint_for_export(checkpoint_path, model_name, class_to_idx, device)
    # El segundo output (features pooled pre-head) alimenta el detector OOD por
    # distancia de Mahalanobis en la app; no cambia el output[0] (logits), que
    # sigue siendo lo único que valida la paridad numérica.
    model = FeatureExposedModel(model, model_name)

    test_loader = None
    if not args.no_parity:
        test_csv = resolve_split_csv(run_dir, args.splits_dir, "val")
        test_loader, _ = build_test_loader(
            test_csv,
            config_path,
            class_to_idx,
            image_size,
            args.batch_size,
            preprocessing=json.loads((run_dir / "summary.json").read_text())["preprocessing"],
        )

    report = export_model(
        model=model,
        run_dir=run_dir,
        model_name=model_name,
        class_to_idx=class_to_idx,
        image_size=image_size,
        formats=formats,
        test_loader=test_loader,
        device=device,
        tolerance=args.tolerance,
        parity_sample_size=args.parity_sample_size,
        skip_parity=args.no_parity,
        quantize=quantize,
        min_agreement_rate=args.min_agreement_rate,
    )
    summary_path = write_export_summary(run_dir, report)

    print(f"\nModelo: {model_name}  (cuantizacion: {quantize or 'none'})")
    print(f"Run: {run_dir}")
    failed = False
    for result in report.formats:
        status = "OK" if result.succeeded else "FALLO"
        print(f"  [{status}] {result.format} -> {result.output_path}")
        if result.error:
            print(f"    error: {result.error}")
        if result.parity is not None:
            parity_status = "paso" if result.parity.passed else "NO PASO"
            print(
                f"    paridad: {parity_status} "
                f"(max_abs_prob_diff={result.parity.max_abs_prob_diff:.6f}, "
                f"tolerance={result.parity.tolerance:.6f}, "
                f"agreement_rate={result.parity.agreement_rate:.4f})"
            )
        if not result.succeeded or (result.parity is not None and not result.parity.passed):
            failed = True
    print(f"  resumen: {summary_path}")
    return failed


def main() -> None:
    args = _parse_args()
    output_root = get_output_root()
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "main"

    if not parse_export_formats(args.formats):
        raise SystemExit("Debes indicar al menos un formato en --formats.")
    if args.checkpoint and len(args.models) > 1:
        raise SystemExit(
            "--checkpoint apunta a un unico archivo; no se puede usar con varios --models."
        )

    failures = [
        model_name for model_name in args.models if _export_one(args, model_name, output_dir)
    ]

    if failures:
        print(f"\nModelos con problemas: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
