from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _import_litert_torch():
    """
    Importa `litert_torch`, el sucesor de `ai-edge-torch`.

    El paquete `ai-edge-torch` quedó deprecado: las versiones >=0.7 son un stub que solo
    emite un DeprecationWarning y no exponen `convert()`, así que importarlo "con éxito"
    no significa que se pueda exportar. Por eso solo aceptamos `litert_torch`.

    @returns {module} El módulo `litert_torch`.
    @throws {ExportDependencyError} Si `litert-torch` no está instalado.
    """
    try:
        import litert_torch
    except ImportError as e:
        from src.export.common import ExportDependencyError

        raise ExportDependencyError(
            "TFLite export requiere 'litert-torch' (sucesor de 'ai-edge-torch', que fue "
            "deprecado y ya no expone convert()). Solo soporta Linux. "
            "Instala con: pip install -e '.[export]'"
        ) from e
    return litert_torch


def _quantize_pt2e(model: torch.nn.Module, sample_inputs: tuple[torch.Tensor, ...]):
    """
    Aplica cuantización dinámica int8 (PT2E simétrica) sobre el grafo.

    Dinámica y no estática a propósito: las escalas de activación se calculan en runtime,
    así que no hace falta un set de calibración y el resultado no depende de qué imágenes
    se le pasen. La pasada de `sample_inputs` solo puebla los observers de pesos.

    El backbone va per-channel y la capa lineal final per-tensor, porque cada mitad tiene
    una restricción distinta:

    - El kernel integrado de TFLite para `FULLY_CONNECTED` dinámico aplica una sola escala
      por tensor. Con escalas por canal devuelve los logits multiplicados por `1/escala_c`,
      una constante distinta por clase, sin error ni aviso. El delegado XNNPACK sí soporta
      per-channel, así que la discrepancia solo aparece en runtimes que no lo activan —
      entre ellos el que embarca la app móvil.
    - Poner *todo* per-tensor no es alternativa: las convoluciones depthwise de EfficientNet
      tienen rangos de peso muy distintos entre canales y una sola escala las destruye. Medido
      sobre `efficientnet_lite0`, el acuerdo top-1 contra PyTorch cae a 0,30.

    La capa se selecciona por tipo de módulo con dos claves porque `_get_module_type_filter`
    compara contra `nn_module_stack`, y `torch.export` guarda ahí el tipo como cadena en unas
    versiones y como clase en otras; la que no case no anota nada.

    `validate_tflite_parity` comprueba después que ningún `FULLY_CONNECTED` quedó con escalas
    por canal, y compara el artefacto con y sin delegado, para que una discrepancia como esa
    no vuelva a pasar inadvertida.

    @param {torch.nn.Module} model Modelo en modo eval().
    @param {tuple[torch.Tensor, ...]} sample_inputs Entradas de ejemplo para el trazado.
    @returns {tuple} (grafo cuantizado, quantizer) para pasar a `litert_torch.convert`.
    @throws {ExportDependencyError} Si falta `torchao`.
    """
    try:
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    except ImportError as e:
        from src.export.common import ExportDependencyError

        raise ExportDependencyError(
            "Cuantizacion int8 de TFLite requiere 'torchao' (viene con litert-torch). "
            "Instala con: pip install -e '.[export]'"
        ) from e

    from litert_torch.quantize.pt2e_quantizer import (
        PT2EQuantizer,
        get_symmetric_quantization_config,
    )

    per_tensor = get_symmetric_quantization_config(is_per_channel=False, is_dynamic=True)
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )
    for linear_key in (torch.nn.Linear, "torch.nn.modules.linear.Linear"):
        quantizer.set_module_type(linear_key, per_tensor)
    graph = torch.export.export(model, sample_inputs).module()
    graph = prepare_pt2e(graph, quantizer)
    graph(*sample_inputs)  # puebla observers
    graph = convert_pt2e(graph, fold_quantize=False)
    return graph, quantizer


def export_to_tflite(
    model: torch.nn.Module,
    output_path: Path,
    image_size: tuple[int, int],
    device: torch.device,
    quantize: str | None = None,
) -> Path:
    """
    Exporta `model` a TFLite con batch fijo=1, vía litert-torch (ecosistema LiteRT).

    @param {torch.nn.Module} model Modelo en modo eval().
    @param {Path} output_path Ruta destino del archivo .tflite.
    @param {tuple[int, int]} image_size Alto y ancho de entrada (h, w).
    @param {torch.device} device Dispositivo donde correr el tracing.
    @param {str|None} quantize "int8" para cuantización dinámica; None/"none" para FP32.
    @returns {Path} `output_path`.
    @throws {ExportDependencyError} Si litert-torch no está instalado.
    """
    litert_torch = _import_litert_torch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    # litert-torch traza en CPU; forzarlo evita que un modelo en CUDA falle en torch.export.
    cpu_model = model.to("cpu")
    sample_inputs = (torch.randn(1, 3, *image_size),)

    if quantize == "int8":
        graph, quantizer = _quantize_pt2e(cpu_model, sample_inputs)
        from litert_torch.quantize.quant_config import QuantConfig

        edge_model = litert_torch.convert(
            graph, sample_inputs, quant_config=QuantConfig(pt2e_quantizer=quantizer)
        )
    else:
        edge_model = litert_torch.convert(cpu_model, sample_inputs)

    edge_model.export(str(output_path))
    model.to(device)
    return output_path
