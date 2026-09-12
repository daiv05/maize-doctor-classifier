import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps

GRADCAM_TARGET_LAYERS: dict[str, str] = {
    "efficientnet_b0": "features.-1",
    "efficientnet_b4": "conv_head",
    "efficientnet_lite0": "conv_head",
    "mobilenet_v3_large": "blocks.-1",
    "mobilenet_v3_small": "blocks.-1",
    "shufflenet_v2_x1_0": "conv5",
    "ghostnetv2_100": "blocks.-1",
    "fastvit_t8": "final_conv",
}


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Resuelve el nn.Module objetivo para Grad-CAM según GRADCAM_TARGET_LAYERS.
    Soporta dot-path con índice final (p.ej. "features.-1")."""
    if model_name not in GRADCAM_TARGET_LAYERS:
        raise KeyError(
            f"Sin capa Grad-CAM registrada para '{model_name}'. "
            f"Arquitecturas soportadas: {sorted(GRADCAM_TARGET_LAYERS)}"
        )
    target = model
    for part in GRADCAM_TARGET_LAYERS[model_name].split("."):
        target = target[int(part)] if part.lstrip("-").isdigit() else getattr(target, part)
    return target


class GradCAM:
    """
    Grad-CAM con hooks nativos de PyTorch, sin dependencias externas nuevas.

    Captura el gradiente vía `Tensor.register_hook` sobre la salida de la capa objetivo
    (no `register_full_backward_hook` a nivel de módulo): varias arquitecturas del
    registry (p.ej. ghostnetv2_100) aplican una ReLU inplace inmediatamente después de
    la capa objetivo, lo que rompe register_full_backward_hook ("view is being modified
    inplace") porque el hook de módulo intercepta el tensor antes de que downstream lo
    module in-place. Enganchar el gradiente directo del tensor de salida evita ese
    conflicto y funciona igual en todas las arquitecturas.

    Uso: with GradCAM(model, target_layer) as cam: heatmap = cam(input_tensor, class_idx)

    Requiere gradientes habilitados -- no debe invocarse dentro de un bloque
    torch.no_grad()/torch.inference_mode() externo, o el backward fallará.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inp, out):
        self._activations = out
        if out.requires_grad:
            out.register_hook(self._save_gradients)

    def _save_gradients(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> torch.Tensor:
        """input_tensor: [1,C,H,W]. Devuelve el heatmap [H,W] normalizado en [0,1]
        (tamaño espacial de la capa objetivo, sin upsample -- eso lo hace el caller)."""
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        logits[0, class_idx].backward()

        assert self._gradients is not None and self._activations is not None, (
            "Los hooks no capturaron activaciones/gradientes -- ¿se llamó dentro de "
            "torch.no_grad()/inference_mode()?"
        )
        activations = self._activations.detach()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1)).squeeze(0)
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            return (cam - cam_min) / (cam_max - cam_min)
        return torch.zeros_like(cam)

    def close(self) -> None:
        self._fwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def build_gradcam_overlay(
    image_rgb01: np.ndarray, cam: torch.Tensor, target_size: tuple[int, int]
) -> np.ndarray:
    """
    Upsample bilinear del heatmap Grad-CAM a target_size y blend con la imagen original.
    Usa 'jet': la magnitud de Grad-CAM es no-negativa, a diferencia de las atribuciones
    con signo de LIME/SHAP que usan un divergente (`ATTRIBUTION_CMAP`).
    """
    upsampled = (
        F.interpolate(cam[None, None, :, :], size=target_size, mode="bilinear", align_corners=False)
        .squeeze()
        .cpu()
        .numpy()
    )
    heatmap_rgba = colormaps["jet"](upsampled)
    return np.clip(image_rgb01 * 0.4 + heatmap_rgba[..., :3] * 0.6, 0, 1)
