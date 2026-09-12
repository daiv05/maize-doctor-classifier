"""Módulo de Ensamble por Soft Voting (Criterio 2 - Rúbrica Etapa 2).

Combina múltiples arquitecturas de Deep Learning (ej. EfficientNet-B0, ShuffleNetV2-x1.0,
EfficientNet-Lite0) mediante promedio ponderado de probabilidades de salida (Softmax),
reduciendo la varianza y elevando la robustez de clasificación.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from src.models.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)



class SoftVotingEnsemble(nn.Module):
    """Ensamble que promedia las probabilidades (Softmax) de múltiples clasificadores."""

    def __init__(
        self,
        models: Sequence[nn.Module],
        weights: Sequence[float] | None = None,
        model_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("El ensamble debe contener al menos un modelo.")

        self.models = nn.ModuleList(models)
        self.model_names = list(model_names) if model_names else [f"model_{i}" for i in range(len(models))]

        if weights is None:
            raw_weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError(
                    f"La cantidad de pesos ({len(weights)}) no coincide con la cantidad de modelos ({len(models)})."
                )
            total = sum(weights)
            if total <= 0:
                raise ValueError("La suma de los pesos del ensamble debe ser mayor a 0.")
            raw_weights = [w / total for w in weights]

        self.register_buffer("weights", torch.tensor(raw_weights, dtype=torch.float32))

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula el tensor de probabilidades promedio [Batch, Num_Classes]."""
        probs_list: list[torch.Tensor] = []
        for model in self.models:
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            probs_list.append(probs)

        # Promedio ponderado: sum(w_i * P_i)
        stacked = torch.stack(probs_list, dim=0)  # [Num_Models, Batch, Num_Classes]
        weights = self.weights.view(-1, 1, 1).to(x.device)  # [Num_Models, 1, 1]
        weighted_probs = (stacked * weights).sum(dim=0)  # [Batch, Num_Classes]
        return weighted_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve log-probabilidades (logits estables) del ensamble para compatibilidad con loss/argmax."""
        probs = self.predict_probabilities(x)
        return torch.log(probs.clamp(min=1e-12))

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve los índices de clases predichas con mayor probabilidad [Batch]."""
        probs = self.predict_probabilities(x)
        return torch.argmax(probs, dim=-1)

    @classmethod
    def from_checkpoints(
        cls,
        model_names: Sequence[str],
        checkpoint_paths: Sequence[str | Path],
        num_classes: int = 9,
        weights: Sequence[float] | None = None,
        device: str | torch.device = "cpu",
    ) -> SoftVotingEnsemble:
        """Construye un ensamble cargando los pesos entrenados de cada modelo desde disco."""
        if len(model_names) != len(checkpoint_paths):
            raise ValueError(
                f"Número de nombres ({len(model_names)}) difiere de checkpoints ({len(checkpoint_paths)})."
            )

        loaded_models: list[nn.Module] = []
        for name, ckpt_path in zip(model_names, checkpoint_paths):
            path = Path(ckpt_path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint no encontrado: {path}")

            model = MODEL_REGISTRY.build(name, num_classes=num_classes, pretrained=False)
            checkpoint = torch.load(path, map_location=device)
            state_dict = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            loaded_models.append(model)
            logger.info("Modelo cargado en el ensamble: %s desde %s", name, path)

        return cls(models=loaded_models, weights=weights, model_names=model_names)
