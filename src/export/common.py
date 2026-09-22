from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.preparation import atomic_write_json, sha256_file
from src.export.parity import ParityResult, validate_onnx_parity, validate_tflite_parity
from src.training.runs import load_validated_run, read_run_contract

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


class ExportIntegrityError(RuntimeError):
    """El artefacto exportado o su metadata no coincide con el run contractual."""


@dataclass
class ExportFormatResult:
    format: str
    output_path: Path | None
    succeeded: bool
    parity: ParityResult | None = None
    error: str | None = None


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


def load_checkpoint_for_export(
    checkpoint_path: Path,
    model_name: str,
    class_to_idx: dict[str, int],
    device: torch.device,
    *,
    image_size: tuple[int, int] | None = None,
    splits_dir: Path | None = None,
) -> torch.nn.Module:
    """
    Reconstruye un modelo del registry y carga los pesos de un checkpoint, en eval().

    @param {Path} checkpoint_path Ruta al archivo .pth.
    @param {str} model_name Nombre del modelo registrado en MODEL_REGISTRY.
    @param {dict[str,int]} class_to_idx Mapeo clase->índice con el que se entrenó.
    @param {torch.device} device Dispositivo destino del modelo.
    @returns {torch.nn.Module} Modelo cargado, en modo eval().
    """
    loaded = load_validated_run(
        checkpoint_path,
        expected_model=model_name,
        expected_num_classes=len(class_to_idx),
        expected_class_to_idx=class_to_idx,
        expected_input_size=image_size,
        splits_dir=splits_dir,
        device=device,
    )
    return loaded.model


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
    summary = read_run_contract(run_dir)
    if summary["model"] != model_name:
        raise ValueError(
            f"El run {summary['run_id']} es de {summary['model']!r}, no de {model_name!r}."
        )
    class_to_idx = dict(summary["class_to_idx"])
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    image_size = summary["architecture"]["input_size"]
    if not (isinstance(image_size, list) and len(image_size) == 2):
        raise SystemExit(f"summary.json en {run_dir} no tiene 'image_size' valido.")
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
        return ExportFormatResult(format=format_name, output_path=output_path, succeeded=True)

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
    except ExportDependencyError as e:
        logger.error("Validacion de paridad %s fallo: %s", format_name, e)
        return ExportFormatResult(
            format=format_name, output_path=output_path, succeeded=True, error=str(e)
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

    return ExportFormatResult(
        format=format_name, output_path=output_path, succeeded=True, parity=parity
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
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = read_run_contract(run_dir)
        payload.update(
            {
                "run_id": summary["run_id"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "class_to_idx": summary["class_to_idx"],
                "preprocessing": summary["preprocessing"],
            }
        )
    output_path = export_dir / "labels.json"
    atomic_write_json(output_path, payload)
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
        "schema_version": 1,
        "run_id": run_dir.name,
        "model": report.model_name,
        "exported_at": report.exported_at,
        "quantize": report.quantize,
        "library_versions": report.library_versions,
        "formats": [
            {
                "format": f.format,
                "output_path": (str(f.output_path.relative_to(run_dir)) if f.output_path else None),
                "succeeded": f.succeeded,
                "error": f.error,
                "sha256": sha256_file(f.output_path) if f.output_path else None,
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
                        "cross_runtime_agreement_rate": f.parity.cross_runtime_agreement_rate,
                        "cross_runtime_max_abs_prob_diff": f.parity.cross_runtime_max_abs_prob_diff,
                        "cross_runtime_tolerance": f.parity.cross_runtime_tolerance,
                        "per_channel_fully_connected": f.parity.per_channel_fully_connected,
                        "passed": f.parity.passed,
                        "warnings": f.parity.warnings,
                    }
                ),
            }
            for f in report.formats
        ],
    }
    training_summary_path = run_dir / "summary.json"
    if training_summary_path.is_file():
        training_summary = read_run_contract(run_dir)
        payload.update(
            {
                "run_id": training_summary["run_id"],
                "checkpoint_sha256": training_summary["checkpoint_sha256"],
                "class_to_idx": training_summary["class_to_idx"],
                "preprocessing": training_summary["preprocessing"],
            }
        )
    name = (
        "export_summary.json"
        if report.quantize is None
        else f"export_summary_{report.quantize}.json"
    )
    summary_path = export_dir / name
    atomic_write_json(summary_path, payload)
    return summary_path


def validate_export_artifact(
    run_dir: Path,
    model_path: Path,
    format_name: str,
    quantize: str | None,
) -> dict:
    """Rechaza un export modificado o desvinculado del contrato antes de inferencia."""
    name = "export_summary.json" if quantize is None else f"export_summary_{quantize}.json"
    metadata_path = run_dir / "export" / name
    if not metadata_path.is_file():
        raise ExportIntegrityError(f"Falta metadata contractual del export: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportIntegrityError(
            f"Metadata de export inválida en {metadata_path}: {error}"
        ) from error
    if payload.get("schema_version") != 1:
        raise ExportIntegrityError(
            f"Esquema de export incompatible en {metadata_path}: {payload.get('schema_version')!r}."
        )
    run = read_run_contract(run_dir)
    for metadata_field in ("run_id", "checkpoint_sha256", "class_to_idx", "preprocessing"):
        if payload.get(metadata_field) != run[metadata_field]:
            raise ExportIntegrityError(
                f"Metadata de export incompatible en {metadata_field}: "
                f"esperado {run[metadata_field]!r}, "
                f"encontrado {payload.get(metadata_field)!r}."
            )
    entry = next(
        (item for item in payload.get("formats", []) if item.get("format") == format_name),
        None,
    )
    if entry is None or not entry.get("succeeded"):
        raise ExportIntegrityError(f"No hay export exitoso registrado para {format_name}.")
    output_path = entry.get("output_path")
    if not isinstance(output_path, str) or Path(output_path).is_absolute():
        raise ExportIntegrityError(f"Ruta exportada insegura en {metadata_path}: {output_path!r}.")
    expected_path = run_dir / output_path
    if run_dir.resolve() not in expected_path.resolve().parents:
        raise ExportIntegrityError(f"La ruta exportada escapa del run: {output_path!r}.")
    if expected_path.resolve() != model_path.resolve():
        raise ExportIntegrityError(
            f"Ruta exportada distinta: esperada {expected_path}, solicitada {model_path}."
        )
    if not model_path.is_file():
        raise ExportIntegrityError(f"No existe el artefacto exportado: {model_path}")
    actual_hash = sha256_file(model_path)
    if actual_hash != entry.get("sha256"):
        raise ExportIntegrityError(
            f"SHA-256 del export no coincide para {model_path}: "
            f"esperado {entry.get('sha256')}, actual {actual_hash}."
        )
    return payload
