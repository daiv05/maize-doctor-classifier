from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.transforms import CornTransformFactory
from src.export.parity import ParityResult, validate_onnx_parity, validate_tflite_parity
from src.provenance import atomic_json, contract_hash

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("onnx", "tflite")
SUPPORTED_QUANTIZATIONS = ("int8",)

_EXPORT_MODULE = {"onnx": "src.export.onnx_export", "tflite": "src.export.tflite_export"}
_EXPORT_FUNCTION = {"onnx": "export_to_onnx", "tflite": "export_to_tflite"}
_VALIDATE_PARITY = {"onnx": validate_onnx_parity, "tflite": validate_tflite_parity}

# La cuantización cambia los números a propósito, así que exigirle la tolerancia de FP32
# la reprobaría siempre. Estos son los umbrales por defecto según el modo; el CLI los
# puede sobrescribir con --tolerance / --min-agreement-rate.
_PARITY_DEFAULTS = {
    None: {"tolerance": 1e-3, "min_agreement_rate": 1.0},
    "int8": {"tolerance": 0.15, "min_agreement_rate": 0.95},
}


class ExportDependencyError(RuntimeError):
    """Falta una dependencia opcional (onnxruntime / litert-torch) para exportar o validar."""


@dataclass
class ExportFormatResult:
    format: str
    output_path: Path | None
    succeeded: bool
    parity: ParityResult | None = None
    error: str | None = None
    feature_parity: dict | None = None


@dataclass
class ExportReport:
    run_dir: Path
    model_name: str
    formats: list[ExportFormatResult] = field(default_factory=list)
    exported_at: str = field(default_factory=lambda: datetime.now().isoformat())
    library_versions: dict[str, str] = field(default_factory=dict)
    quantize: str | None = None


def parse_quantize(raw: str | None) -> str | None:
    """
    Normaliza el modo de cuantización pedido por CLI.

    @param {str|None} raw "int8", "none"/"" o None.
    @returns {str|None} El modo válido, o None para FP32.
    @throws {SystemExit} Si el modo no está soportado.
    """
    if not raw or raw.strip().lower() in {"none", "fp32", "float32"}:
        return None
    mode = raw.strip().lower()
    if mode not in SUPPORTED_QUANTIZATIONS:
        raise SystemExit(
            f"Cuantizacion desconocida: '{mode}'. "
            f"Soportadas: {list(SUPPORTED_QUANTIZATIONS)} (o 'none')."
        )
    return mode


def parse_export_formats(raw: str | list[str] | None) -> list[str]:
    """
    Parsea una lista de formatos de exportación separados por coma.

    @param {str|list[str]|None} raw Formatos como CSV ("onnx,tflite") o lista. None/'' -> [].
    @returns {list[str]} Formatos válidos, sin duplicados, orden de aparición.
    @throws {SystemExit} Si algún formato no está en SUPPORTED_FORMATS.
    """
    if not raw:
        return []
    items = raw.split(",") if isinstance(raw, str) else raw
    formats = [item.strip().lower() for item in items if item.strip()]
    unknown = [f for f in formats if f not in SUPPORTED_FORMATS]
    if unknown:
        raise SystemExit(
            f"Formato(s) de exportacion desconocidos: {unknown}. "
            f"Soportados: {list(SUPPORTED_FORMATS)}"
        )
    seen: dict[str, None] = {}
    for f in formats:
        seen.setdefault(f, None)
    return list(seen)


def _load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Checkpoint invalido: {checkpoint_path}")
    return checkpoint


def load_checkpoint_for_export(
    checkpoint_path: Path,
    model_name: str,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> torch.nn.Module:
    """
    Reconstruye un modelo del registry y carga los pesos de un checkpoint, en eval().

    @param {Path} checkpoint_path Ruta al archivo .pth.
    @param {str} model_name Nombre del modelo registrado en MODEL_REGISTRY.
    @param {dict[str,int]} class_to_idx Mapeo clase->índice con el que se entrenó.
    @param {torch.device} device Dispositivo destino del modelo.
    @returns {torch.nn.Module} Modelo cargado, en modo eval().
    """
    from src.training.runs import load_run

    run = load_run(checkpoint_path, model_name, device)
    if run.class_to_idx != class_to_idx:
        raise ValueError("Export class order differs from checkpoint contract")
    return run.model


def resolve_export_inputs(
    run_dir: Path, model_name: str, config_path: Path
) -> tuple[dict[str, int], dict[int, str], tuple[int, int]]:
    """
    Resuelve class_to_idx/idx_to_class/image_size desde summary.json de un run.

    @param {Path} run_dir Directorio del run.
    @param {str} model_name Nombre del modelo (solo para mensajes de error).
    @param {Path} config_path Ruta al YAML de configuración (no usado si hay summary.json).
    @returns {tuple} class_to_idx, idx_to_class, image_size (h, w).
    @throws {SystemExit} Si no existe summary.json en run_dir.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"No existe {summary_path}. La exportacion requiere un run completo "
            f"con summary.json (modelo '{model_name}')."
        )
    summary = json.loads(summary_path.read_text())
    if summary.get("model") != model_name:
        raise ValueError("Export architecture differs from training contract")
    CornTransformFactory.from_contract(summary["preprocessing"])
    class_to_idx = {str(name): int(idx) for name, idx in summary["class_to_idx"].items()}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    if sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
        raise ValueError("Invalid export class order")
    image_size = summary.get("image_size")
    if not (isinstance(image_size, list) and len(image_size) == 2):
        raise SystemExit(f"summary.json en {run_dir} no tiene 'image_size' valido.")
    if image_size != summary["preprocessing"]["target_size"]:
        raise ValueError("Export image size differs from preprocessing contract")
    return class_to_idx, idx_to_class, (int(image_size[0]), int(image_size[1]))


def _library_versions(formats: list[str]) -> dict[str, str]:
    versions = {"torch": torch.__version__}
    if "onnx" in formats:
        try:
            import onnx

            versions["onnx"] = onnx.__version__
        except ImportError:
            pass
        try:
            import onnxruntime

            versions["onnxruntime"] = onnxruntime.__version__
        except ImportError:
            pass
    if "tflite" in formats:
        try:
            import litert_torch

            versions["litert_torch"] = getattr(litert_torch, "__version__", None) or getattr(
                litert_torch.version, "__version__", "desconocida"
            )
        except (ImportError, AttributeError):
            pass
        try:
            import ai_edge_litert

            versions["ai_edge_litert"] = getattr(ai_edge_litert, "__version__", "desconocida")
        except ImportError:
            pass
    return versions


def export_artifact_name(format_name: str, quantize: str | None) -> str:
    """
    Nombre del archivo exportado: `model.<fmt>` en FP32, `model_<quant>.<fmt>` cuantizado.

    Se separan para que un run pueda tener ambas variantes a la vez y se puedan comparar
    sin re-exportar.

    @param {str} format_name "onnx" o "tflite".
    @param {str|None} quantize Modo de cuantización, o None para FP32.
    @returns {str} Nombre del archivo.
    """
    stem = "model" if quantize is None else f"model_{quantize}"
    return f"{stem}.{format_name}"


def _export_single_format(
    format_name: str,
    model: torch.nn.Module,
    export_dir: Path,
    image_size: tuple[int, int],
    device: torch.device,
    test_loader: DataLoader | None,
    tolerance: float,
    parity_sample_size: int,
    skip_parity: bool,
    quantize: str | None,
    min_agreement_rate: float,
) -> ExportFormatResult:
    output_path = export_dir / export_artifact_name(format_name, quantize)
    try:
        import importlib

        export_module = importlib.import_module(_EXPORT_MODULE[format_name])
        export_fn = getattr(export_module, _EXPORT_FUNCTION[format_name])
        export_fn(model, output_path, image_size, device, quantize=quantize)
    except ExportDependencyError as e:
        logger.error("Exportacion a %s fallo: %s", format_name, e)
        return ExportFormatResult(
            format=format_name, output_path=None, succeeded=False, error=str(e)
        )
    except Exception as e:
        logger.exception("Exportacion a %s fallo con un error inesperado.", format_name)
        return ExportFormatResult(
            format=format_name, output_path=None, succeeded=False, error=str(e)
        )

    if skip_parity or test_loader is None:
        reason = "skip_parity=True" if skip_parity else "no se proveyo test_loader"
        logger.warning(
            "Paridad numerica omitida para %s (%s). No exportar a produccion sin validar.",
            format_name,
            reason,
        )
        return ExportFormatResult(
            format=format_name,
            output_path=output_path,
            succeeded=False,
            error=f"Unvalidated artifact: {reason}",
        )

    try:
        parity = _VALIDATE_PARITY[format_name](
            model,
            output_path,
            test_loader,
            device,
            parity_sample_size,
            tolerance,
            min_agreement_rate,
        )
    except Exception as e:
        logger.error("Validacion de paridad %s fallo: %s", format_name, e)
        return ExportFormatResult(
            format=format_name, output_path=output_path, succeeded=False, error=str(e)
        )

    if not parity.passed:
        logger.error(
            "Paridad %s NO paso: max_abs_prob_diff=%.6f (tolerancia=%.6f), "
            "agreement_rate=%.4f (minimo=%.4f)",
            format_name,
            parity.max_abs_prob_diff,
            parity.tolerance,
            parity.agreement_rate,
            parity.min_agreement_rate,
        )

    feature_parity = None
    try:
        import numpy as np

        from src.export.runtime import load_exported_runner

        images, _ = next(iter(test_loader))
        images = images[:parity_sample_size].to(device)
        with torch.no_grad():
            reference = model(images)
        if isinstance(reference, (tuple, list)) and len(reference) > 1:
            runner = load_exported_runner(output_path, format_name)
            outputs = runner.all_outputs(images.cpu().numpy())
            expected = reference[1].detach().cpu().numpy()
            actual = outputs[1]
            passed = (
                expected.shape == actual.shape
                and np.isfinite(actual).all()
                and np.allclose(expected, actual, atol=tolerance, rtol=tolerance)
            )
            feature_parity = {
                "passed": bool(passed),
                "n_samples": len(images),
                "feature_dim": int(expected.shape[-1]),
                "tolerance": tolerance,
                "kind": "pooled_pre_head",
            }
    except Exception as error:
        feature_parity = {"passed": False, "error": str(error)}
    return ExportFormatResult(
        format=format_name,
        output_path=output_path,
        succeeded=parity.passed,
        parity=parity,
        feature_parity=feature_parity,
    )


def write_labels_json(
    run_dir: Path,
    class_to_idx: dict[str, int],
    model_name: str,
    image_size: tuple[int, int],
) -> Path:
    """
    Persiste <run_dir>/export/labels.json con el orden de clases del modelo.

    Existe para que ningun consumidor del modelo exportado (la app movil, un script de
    evaluacion externo) tenga que re-derivar o hardcodear el orden de clases: lo lee de
    aqui. Un desajuste de orden entre este archivo y el modelo real es indetectable en
    runtime (el modelo igual devuelve 9 logits validos), asi que la unica fuente de verdad
    aceptable es `class_to_idx` de `summary.json`, nunca una lista transcrita a mano.

    @param {Path} run_dir Directorio del run.
    @param {dict[str,int]} class_to_idx Mapeo clase->indice con el que se entreno/exporto.
    @param {str} model_name Nombre del modelo (metadata informativa).
    @param {tuple[int,int]} image_size Alto y ancho de entrada esperado (h, w).
    @returns {Path} Ruta del archivo escrito.
    @throws {ValueError} Si `class_to_idx` no tiene indices contiguos 0..N-1 (por ejemplo,
        un `summary.json` editado a mano con un indice repetido o un hueco).
    """
    if sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
        raise ValueError(f"class_to_idx no es contiguo 0..N-1: {class_to_idx}")
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    payload = {
        "schema_version": 1,
        "model": model_name,
        "image_size": list(image_size),
        "labels": labels,
    }
    output_path = export_dir / "labels.json"
    atomic_json(output_path, payload)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("class_to_idx") != class_to_idx or summary.get("model") != model_name:
            raise ValueError("Export labels differ from training contract")
        factory = CornTransformFactory.from_contract(summary["preprocessing"])
        atomic_json(export_dir / "preprocessing.json", factory.to_contract())
    return output_path


def export_model(
    model: torch.nn.Module,
    run_dir: Path,
    model_name: str,
    class_to_idx: dict[str, int],
    image_size: tuple[int, int],
    formats: list[str],
    *,
    test_loader: DataLoader | None = None,
    device: torch.device | None = None,
    tolerance: float | None = None,
    parity_sample_size: int = 30,
    skip_parity: bool = False,
    quantize: str | None = None,
    min_agreement_rate: float | None = None,
) -> ExportReport:
    """
    Exporta `model` a cada formato de `formats` bajo <run_dir>/export/ y valida paridad.

    Tambien escribe `export/labels.json` con el orden de clases (`write_labels_json`), una
    sola vez por llamada y sin condicionarlo al exito de ningun formato individual: para
    cuando `export_model()` corre, el checkpoint ya se cargo correctamente, asi que el
    archivo siempre describe un modelo real aunque alguna conversion puntual falle.

    @param {torch.nn.Module} model Modelo ya construido (se fuerza a eval() internamente).
    @param {Path} run_dir Directorio del run de entrenamiento.
    @param {str} model_name Nombre del modelo (para el reporte).
    @param {dict[str,int]} class_to_idx Mapeo clase->índice del run.
    @param {tuple[int,int]} image_size Alto y ancho de entrada (h, w).
    @param {list[str]} formats Formatos a exportar ("onnx", "tflite").
    @param {DataLoader|None} test_loader Loader de test para validar paridad; None la omite.
    @param {torch.device|None} device Dispositivo; por defecto el de `model`.
    @param {float|None} tolerance Tolerancia de diferencia de probabilidad; None usa el
        default del modo de cuantización.
    @param {int} parity_sample_size Número de muestras a usar en la validación de paridad.
    @param {bool} skip_parity Omite la validación de paridad aunque haya test_loader.
    @param {str|None} quantize "int8" para cuantizar; None exporta en FP32.
    @param {float|None} min_agreement_rate Acuerdo top-1 mínimo; None usa el default del modo.
    @returns {ExportReport} Resultado de la exportación, un ExportFormatResult por formato.
    """
    model.eval()
    resolved_device = device or next(model.parameters()).device
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    defaults = _PARITY_DEFAULTS[quantize]
    resolved_tolerance = defaults["tolerance"] if tolerance is None else tolerance
    resolved_min_agreement = (
        defaults["min_agreement_rate"] if min_agreement_rate is None else min_agreement_rate
    )

    results = [
        _export_single_format(
            fmt,
            model,
            export_dir,
            image_size,
            resolved_device,
            test_loader,
            resolved_tolerance,
            parity_sample_size,
            skip_parity,
            quantize,
            resolved_min_agreement,
        )
        for fmt in formats
    ]

    write_labels_json(run_dir, class_to_idx, model_name, image_size)

    return ExportReport(
        run_dir=run_dir,
        model_name=model_name,
        formats=results,
        library_versions=_library_versions(formats),
        quantize=quantize,
    )


def _sha256_file(path: Path) -> str:
    """
    Calcula el digest SHA-256 de un archivo, leyendo en bloques para no cargarlo entero en memoria.

    @param {Path} path Archivo a hashear.
    @returns {str} Digest hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_export_summary(run_dir: Path, report: ExportReport) -> Path:
    """
    Persiste <run_dir>/export/export_summary.json (o ..._<quant>.json si se cuantizó).

    Se separa por modo para que exportar int8 no pise el reporte de la variante FP32.

    @param {Path} run_dir Directorio del run.
    @param {ExportReport} report Reporte de exportación a serializar.
    @returns {Path} Ruta del archivo escrito.
    """
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "model": report.model_name,
        "exported_at": report.exported_at,
        "quantize": report.quantize,
        "library_versions": report.library_versions,
        "formats": [
            {
                "format": f.format,
                "output_path": (str(f.output_path.relative_to(run_dir)) if f.output_path else None),
                "created": f.output_path is not None and f.output_path.is_file(),
                "succeeded": bool(f.succeeded and f.parity and f.parity.passed),
                "validation_status": "validated"
                if f.succeeded and f.parity and f.parity.passed
                else "failed"
                if f.parity
                else "unvalidated",
                "deliverable": False,  # Full-split evaluation + bundle preflight remain mandatory.
                "error": f.error,
                "feature_parity": f.feature_parity,
                "sha256": _sha256_file(f.output_path) if f.output_path else None,
                "parity": (
                    None
                    if f.parity is None
                    else {
                        "n_samples": f.parity.n_samples,
                        "torch_top1_accuracy": f.parity.torch_top1_accuracy,
                        "exported_top1_accuracy": f.parity.exported_top1_accuracy,
                        "agreement_rate": f.parity.agreement_rate,
                        "max_abs_prob_diff": f.parity.max_abs_prob_diff,
                        "mean_abs_prob_diff": f.parity.mean_abs_prob_diff,
                        "tolerance": f.parity.tolerance,
                        "min_agreement_rate": f.parity.min_agreement_rate,
                        "passed": f.parity.passed,
                        "warnings": f.parity.warnings,
                    }
                ),
            }
            for f in report.formats
        ],
    }
    training_path = run_dir / "summary.json"
    if training_path.exists():
        training = json.loads(training_path.read_text())
        payload["checkpoint_sha256"] = training.get("checkpoint_sha256")
        payload["preprocessing_id"] = contract_hash(training["preprocessing"])
        payload["class_to_idx"] = training["class_to_idx"]
        payload["training_summary_sha256"] = _sha256_file(training_path)
    payload["asset_hashes"] = {
        name: _sha256_file(export_dir / name)
        for name in ("labels.json", "preprocessing.json")
        if (export_dir / name).is_file()
    }
    name = (
        "export_summary.json"
        if report.quantize is None
        else f"export_summary_{report.quantize}.json"
    )
    summary_path = export_dir / name
    atomic_json(summary_path, payload)
    return summary_path
