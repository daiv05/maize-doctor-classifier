"""Carga de modelos ya exportados (.onnx / .tflite) para inferencia en CPU.

Existe para que la evaluación sobre el split de test corra contra el **archivo exportado**
y no contra el modelo PyTorch: es la única forma de detectar que la conversión degradó el
modelo. Es también la referencia de cómo debe comportarse el runtime en la app móvil.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ExportedRunner(Protocol):
    """Callable que mapea un batch (N,3,H,W) float32 a logits (N,C) float32."""

    def __call__(self, batch: np.ndarray) -> np.ndarray: ...


def _load_onnx_runner(model_path: Path) -> ExportedRunner:
    try:
        import onnxruntime as ort
    except ImportError as e:
        from src.export.common import ExportDependencyError

        raise ExportDependencyError(
            "Evaluar un modelo ONNX requiere 'onnxruntime'. Instala con: pip install -e '.[export]'"
        ) from e

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def run(batch: np.ndarray) -> np.ndarray:
        return session.run(None, {input_name: batch.astype(np.float32)})[0]

    run.all_outputs = lambda batch: session.run(None, {input_name: batch.astype(np.float32)})
    return run


def _load_tflite_runner(model_path: Path) -> ExportedRunner:
    interpreter_cls = None
    try:
        from ai_edge_litert.interpreter import Interpreter as interpreter_cls
    except ImportError:
        try:
            from tensorflow.lite import Interpreter as interpreter_cls
        except ImportError:
            pass

    if interpreter_cls is None:
        from src.export.common import ExportDependencyError

        raise ExportDependencyError(
            "Evaluar un modelo TFLite requiere 'ai-edge-litert' o 'tensorflow'. "
            "Instala con: pip install -e '.[export]'"
        )

    interpreter = interpreter_cls(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    output_detail = output_details[0]
    n_classes = int(output_detail["shape"][-1])

    def run(batch: np.ndarray) -> np.ndarray:
        # El modelo se exporta con batch fijo=1, así que se itera imagen por imagen.
        logits = np.zeros((batch.shape[0], n_classes), dtype=np.float32)
        for i in range(batch.shape[0]):
            interpreter.set_tensor(
                input_detail["index"], batch[i : i + 1].astype(input_detail["dtype"])
            )
            interpreter.invoke()
            logits[i] = interpreter.get_tensor(output_detail["index"])[0]
        return logits

    def all_outputs(batch):
        outputs = [[] for _ in output_details]
        for item in batch:
            interpreter.set_tensor(input_detail["index"], item[None].astype(input_detail["dtype"]))
            interpreter.invoke()
            for values, detail in zip(outputs, output_details):
                values.append(interpreter.get_tensor(detail["index"]).copy())
        return [np.concatenate(values) for values in outputs]

    run.all_outputs = all_outputs
    return run


_LOADERS: dict[str, Callable[[Path], ExportedRunner]] = {
    "onnx": _load_onnx_runner,
    "tflite": _load_tflite_runner,
}


def load_exported_runner(model_path: Path, format_name: str) -> ExportedRunner:
    """
    Devuelve un callable que corre inferencia sobre el modelo exportado.

    @param {Path} model_path Ruta al archivo .onnx o .tflite.
    @param {str} format_name "onnx" o "tflite".
    @returns {ExportedRunner} Callable batch (N,3,H,W) -> logits (N,C).
    @throws {SystemExit} Si el formato no se reconoce o el archivo no existe.
    @throws {ExportDependencyError} Si falta el runtime correspondiente.
    """
    if format_name not in _LOADERS:
        raise SystemExit(
            f"Formato desconocido para evaluacion: '{format_name}'. Soportados: {sorted(_LOADERS)}"
        )
    if not model_path.exists():
        raise SystemExit(
            f"No existe el modelo exportado: {model_path}. Corre primero 'make export-main'."
        )
    return _LOADERS[format_name](model_path)
