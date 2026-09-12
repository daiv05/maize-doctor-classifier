"""Módulo de Ensamble por Soft Voting (Criterio 2 - Rúbrica Etapa 2).

Combina múltiples arquitecturas de Deep Learning (ej. EfficientNet-B0, ShuffleNetV2-x1.0,
EfficientNet-Lite0) mediante promedio ponderado de probabilidades de salida (Softmax),
reduciendo la varianza y elevando la robustez de clasificación.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

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
        self.model_names = (
            list(model_names) if model_names else [f"model_{i}" for i in range(len(models))]
        )

        if weights is None:
            raw_weights = [1.0 / len(models)] * len(models)
        else:
            if any(not math.isfinite(w) or w < 0 for w in weights):
                raise ValueError("Ensemble weights must be finite and nonnegative")
            if len(weights) != len(models):
                raise ValueError(
                    f"Cantidad de pesos ({len(weights)}) no coincide con modelos ({len(models)})."
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
        """Devuelve log-probabilidades para compatibilidad con loss/argmax."""
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
                f"Nombres ({len(model_names)}) difieren de checkpoints ({len(checkpoint_paths)})."
            )

        from src.training.runs import load_run, validate_ensemble_runs

        runs = [load_run(path, name, device) for name, path in zip(model_names, checkpoint_paths)]
        validate_ensemble_runs(runs, shared_input=True)
        if len(runs[0].class_to_idx) != num_classes:
            raise ValueError("num_classes differs from member contracts")
        return cls(models=[run.model for run in runs], weights=weights, model_names=model_names)
