from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.identity import unpack_batch

logger = logging.getLogger(__name__)

_MIN_RELIABLE_SAMPLES = 20

_CROSS_RUNTIME_TOLERANCE = 0.05


def _extract_logits(model_output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Extrae el tensor de logits de la salida de un modelo, sea de un solo output
    (Tensor) o de un `FeatureExposedModel` (tupla `(logits, features)`).

    @param {torch.Tensor|tuple[torch.Tensor, ...]} model_output Salida de `model(images)`.
    @returns {torch.Tensor} El tensor de logits (primer elemento si es tupla).
    """
    return model_output[0] if isinstance(model_output, tuple) else model_output


@dataclass
class ParityResult:
    format: str
    n_samples: int
    torch_top1_accuracy: float
    exported_top1_accuracy: float
    agreement_rate: float
    max_abs_prob_diff: float
    mean_abs_prob_diff: float
    tolerance: float
    passed: bool
    min_agreement_rate: float = 1.0
    warnings: list[str] = field(default_factory=list)
    cross_runtime_agreement_rate: float | None = None
    cross_runtime_max_abs_prob_diff: float | None = None
    cross_runtime_tolerance: float | None = None
    per_channel_fully_connected: int | None = None


def _collect_samples(
    test_loader: DataLoader, device: torch.device, sample_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Toma las primeras `sample_size` muestras de `test_loader` (pipeline 'test',
    determinista, sin augmentation).

    @param {DataLoader} test_loader Loader de test ya construido.
    @param {torch.device} device Dispositivo para las imágenes recolectadas.
    @param {int} sample_size Número máximo de muestras a recolectar.
    @returns {tuple[torch.Tensor, torch.Tensor]} Imágenes apiladas y sus labels.
    """
    images_batches: list[torch.Tensor] = []
    labels_batches: list[torch.Tensor] = []
    collected = 0
    for batch in test_loader:
        images, labels, _ = unpack_batch(batch)
        images_batches.append(images)
        labels_batches.append(labels)
        collected += images.size(0)
        if collected >= sample_size:
            break
    images = torch.cat(images_batches, dim=0)[:sample_size].to(device)
    labels = torch.cat(labels_batches, dim=0)[:sample_size]
    return images, labels


def _build_parity_result(
    format_name: str,
    torch_probs: np.ndarray,
    exported_probs: np.ndarray,
    labels: np.ndarray,
    tolerance: float,
    min_agreement_rate: float = 1.0,
) -> ParityResult:
    torch_preds = torch_probs.argmax(axis=1)
    exported_preds = exported_probs.argmax(axis=1)

    torch_top1_accuracy = float((torch_preds == labels).mean())
    exported_top1_accuracy = float((exported_preds == labels).mean())
    agreement_rate = float((torch_preds == exported_preds).mean())

    abs_diff = np.abs(torch_probs - exported_probs)
    max_abs_prob_diff = float(abs_diff.max())
    mean_abs_prob_diff = float(abs_diff.mean())

    warnings: list[str] = []
    n_samples = len(labels)
    if n_samples < _MIN_RELIABLE_SAMPLES:
        warnings.append(
            f"n_samples={n_samples} < {_MIN_RELIABLE_SAMPLES}, resultado poco confiable"
        )

    passed = max_abs_prob_diff <= tolerance and agreement_rate >= min_agreement_rate

    return ParityResult(
        format=format_name,
        n_samples=n_samples,
        torch_top1_accuracy=torch_top1_accuracy,
        exported_top1_accuracy=exported_top1_accuracy,
        agreement_rate=agreement_rate,
        max_abs_prob_diff=max_abs_prob_diff,
        mean_abs_prob_diff=mean_abs_prob_diff,
        tolerance=tolerance,
        passed=passed,
        min_agreement_rate=min_agreement_rate,
        warnings=warnings,
    )


def validate_onnx_parity(
    torch_model: torch.nn.Module,
    onnx_path: Path,
    test_loader: DataLoader,
    device: torch.device,
    sample_size: int = 30,
    tolerance: float = 1e-3,
    min_agreement_rate: float = 1.0,
) -> ParityResult:
    """
    Compara probabilidades del modelo PyTorch contra el modelo ONNX exportado.

    @param {torch.nn.Module} torch_model Modelo original en modo eval().
    @param {Path} onnx_path Ruta al modelo .onnx exportado.
    @param {DataLoader} test_loader Loader del split de test (pipeline 'test').
    @param {torch.device} device Dispositivo para correr el modelo PyTorch.
    @param {int} sample_size Número de muestras a comparar.
    @param {float} tolerance Tolerancia máxima de diferencia absoluta de probabilidad.
    @param {float} min_agreement_rate Acuerdo top-1 mínimo exigido (1.0 = exacto).
    @returns {ParityResult} Resultado de la comparación.
    @throws {ExportDependencyError} Si onnxruntime no está instalado.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        from src.export.common import ExportDependencyError

        raise ExportDependencyError(
            "Validacion de paridad ONNX requiere 'onnxruntime'. "
            "Instala con: pip install -e '.[export]'"
        ) from e

    images, labels = _collect_samples(test_loader, device, sample_size)

    torch_model.eval()
    with torch.no_grad():
        torch_logits = _extract_logits(torch_model(images))
        torch_probs = torch_logits.softmax(dim=1).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: images.cpu().numpy()})[0]
    exported_probs = torch.from_numpy(onnx_logits).softmax(dim=1).numpy()

    return _build_parity_result(
        "onnx", torch_probs, exported_probs, labels.numpy(), tolerance, min_agreement_rate
    )


def _resolve_tflite_interpreter() -> tuple[type, type | None]:
    """
    Localiza la clase `Interpreter` disponible y, si existe, su `OpResolverType`.

    @returns {tuple[type, type|None]} Clase del intérprete y el enum de resolvers, o None
      si el paquete instalado no lo expone.
    @throws {ExportDependencyError} Si no hay ningún intérprete TFLite instalado.
    """
    candidates: list[tuple[type, type | None]] = []
    for module_name in ("ai_edge_litert.interpreter", "tensorflow.lite"):
        try:
            module = __import__(module_name, fromlist=["Interpreter"])
        except ImportError:
            continue
        interpreter_cls = getattr(module, "Interpreter", None)
        if interpreter_cls is None:
            continue
        resolver = getattr(module, "OpResolverType", None)
        if resolver is None:
            experimental = getattr(module, "experimental", None)
            resolver = getattr(experimental, "OpResolverType", None)
        candidates.append((interpreter_cls, resolver))

    for interpreter_cls, resolver in candidates:
        if resolver is not None:
            return interpreter_cls, resolver
    if candidates:
        return candidates[0]

    from src.export.common import ExportDependencyError

    raise ExportDependencyError(
        "Validacion de paridad TFLite requiere 'ai-edge-litert' o 'tensorflow'. "
        "Instala con: pip install -e '.[export]'"
    )


def _count_per_channel_fully_connected(interpreter) -> int | None:
    """
    Cuenta los `FULLY_CONNECTED` cuyo tensor de pesos lleva escalas distintas entre canales.

    Es la configuración que el kernel integrado de TFLite no sabe aplicar: devuelve los logits
    multiplicados por `1/escala_c` sin señalar el error. Lo que importa es que las escalas
    difieran, no cuántas haya: el exportador emite la cuantización per-tensor como un tensor
    per-axis con todas las escalas iguales, y ese caso el kernel sí lo resuelve bien.

    La comprobación es estructural a propósito, porque no depende de qué runtimes haya
    instalados para correr el artefacto.

    @param {object} interpreter Intérprete con los tensores ya asignados.
    @returns {int|None} Cuántos ops incumplen, o None si el paquete instalado no expone el
      detalle de operaciones.
    """
    get_ops = getattr(interpreter, "_get_ops_details", None)
    if get_ops is None:
        return None

    scales_by_tensor = {
        d["index"]: np.asarray(d["quantization_parameters"]["scales"], dtype=np.float64)
        for d in interpreter.get_tensor_details()
    }
    offenders = 0
    for op in get_ops():
        if op.get("op_name") != "FULLY_CONNECTED":
            continue
        for tensor_index in op.get("inputs", [])[1:2]:
            scales = scales_by_tensor.get(tensor_index)
            if scales is None or scales.size < 2:
                continue
            smallest = float(np.abs(scales).min())
            if smallest <= 0 or float(np.abs(scales).max()) / smallest > 1 + 1e-6:
                offenders += 1
    return offenders


def _run_tflite(interpreter, images_np: np.ndarray) -> np.ndarray:
    """
    Pasa un lote imagen por imagen por un intérprete TFLite ya construido.

    @param {object} interpreter Intérprete con los tensores ya asignados.
    @param {np.ndarray} images_np Lote NCHW float32.
    @returns {np.ndarray} Logits, uno por imagen.
    """
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    logits = np.zeros((images_np.shape[0], output_detail["shape"][-1]), dtype=np.float32)
    for i in range(images_np.shape[0]):
        interpreter.set_tensor(input_detail["index"], images_np[i : i + 1])
        interpreter.invoke()
        logits[i] = interpreter.get_tensor(output_detail["index"])[0]
    return logits


def validate_tflite_parity(
    torch_model: torch.nn.Module,
    tflite_path: Path,
    test_loader: DataLoader,
    device: torch.device,
    sample_size: int = 30,
    tolerance: float = 1e-3,
    min_agreement_rate: float = 1.0,
) -> ParityResult:
    """
    Compara probabilidades del modelo PyTorch contra el modelo TFLite exportado, y el
    artefacto contra sí mismo bajo dos intérpretes.

    El intérprete por defecto activa el delegado XNNPACK; el que embarca la app móvil no.
    Un artefacto cuyos dos intérpretes discrepan produce en el teléfono números distintos
    de los que reporta la evaluación del servidor, así que esa discrepancia invalida el
    export aunque la paridad contra PyTorch pase.

    @param {torch.nn.Module} torch_model Modelo original en modo eval().
    @param {Path} tflite_path Ruta al modelo .tflite exportado.
    @param {DataLoader} test_loader Loader del split de test (pipeline 'test').
    @param {torch.device} device Dispositivo para correr el modelo PyTorch.
    @param {int} sample_size Número de muestras a comparar.
    @param {float} tolerance Tolerancia máxima de diferencia absoluta de probabilidad.
    @param {float} min_agreement_rate Acuerdo top-1 mínimo exigido (1.0 = exacto).
    @returns {ParityResult} Resultado de la comparación.
    @throws {ExportDependencyError} Si no hay un intérprete TFLite disponible.
    """
    interpreter_cls, resolver_enum = _resolve_tflite_interpreter()

    images, labels = _collect_samples(test_loader, device, sample_size)

    torch_model.eval()
    with torch.no_grad():
        torch_logits = _extract_logits(torch_model(images))
        torch_probs = torch_logits.softmax(dim=1).cpu().numpy()

    images_np = images.cpu().numpy()

    interpreter = interpreter_cls(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    exported_probs = (
        torch.from_numpy(_run_tflite(interpreter, images_np)).softmax(dim=1).numpy()
    )

    result = _build_parity_result(
        "tflite", torch_probs, exported_probs, labels.numpy(), tolerance, min_agreement_rate
    )

    result.per_channel_fully_connected = _count_per_channel_fully_connected(interpreter)
    if result.per_channel_fully_connected is None:
        result.warnings.append(
            "el paquete TFLite instalado no expone el detalle de operaciones: no se pudo "
            "verificar que los FULLY_CONNECTED no lleven escalas por canal"
        )
    elif result.per_channel_fully_connected > 0:
        result.warnings.append(
            f"{result.per_channel_fully_connected} FULLY_CONNECTED con escalas de peso por "
            "canal: el kernel integrado de TFLite no las aplica y el telefono calculara "
            "logits escalados por 1/escala_c, distinta por clase"
        )
        result.passed = False

    if resolver_enum is None:
        result.warnings.append(
            "el paquete TFLite instalado no expone OpResolverType: no se pudo comparar el "
            "artefacto sin delegados, que es como corre en el telefono"
        )
        return result

    bare = interpreter_cls(
        model_path=str(tflite_path),
        experimental_op_resolver_type=resolver_enum.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    bare.allocate_tensors()
    bare_probs = torch.from_numpy(_run_tflite(bare, images_np)).softmax(dim=1).numpy()

    result.cross_runtime_tolerance = _CROSS_RUNTIME_TOLERANCE
    result.cross_runtime_agreement_rate = float(
        (exported_probs.argmax(axis=1) == bare_probs.argmax(axis=1)).mean()
    )
    result.cross_runtime_max_abs_prob_diff = float(
        np.abs(exported_probs - bare_probs).max()
    )

    cross_ok = (
        result.cross_runtime_max_abs_prob_diff <= _CROSS_RUNTIME_TOLERANCE
        and result.cross_runtime_agreement_rate >= min_agreement_rate
    )
    if not cross_ok:
        result.warnings.append(
            "el artefacto da resultados distintos con y sin delegado "
            f"(max|dprob|={result.cross_runtime_max_abs_prob_diff:.4f} > "
            f"{_CROSS_RUNTIME_TOLERANCE}, acuerdo={result.cross_runtime_agreement_rate:.4f}): "
            "las metricas del servidor no describen lo que calcula el telefono"
        )
    result.passed = result.passed and cross_ok

    return result
