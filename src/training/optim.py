"""Scheduler y parada temprana del pipeline principal.

Cosine con warmup lineal es la eleccion por determinismo: a diferencia de
ReduceLROnPlateau, la curva de lr no depende del ruido de validacion, asi que dos
corridas con la misma semilla recorren exactamente el mismo camino.
"""

from __future__ import annotations

import math

import torch

_SCHEDULERS = ("none", "cosine")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: str,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Construye el scheduler por epoca: warmup lineal seguido de cosine annealing.

    @param {torch.optim.Optimizer} optimizer Optimizador cuyo lr se modula.
    @param {str} kind Una de "none" o "cosine".
    @param {int} total_epochs Total de epocas previstas.
    @param {int} warmup_epochs Epocas de subida lineal desde ~0 hasta el lr base.
    @param {float} min_lr Piso del descenso coseno.
    @returns {LRScheduler|None} Scheduler listo para `.step()` por epoca, o None si kind es "none".
    """
    if kind not in _SCHEDULERS:
        raise ValueError(f"Scheduler desconocido: '{kind}'. Validos: {', '.join(_SCHEDULERS)}")
    if kind == "none":
        return None

    base_lr = optimizer.param_groups[0]["lr"]
    decay_epochs = max(total_epochs - warmup_epochs, 1)
    floor_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = min((epoch - warmup_epochs) / decay_epochs, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Detiene el entrenamiento tras `patience` epocas sin mejora de la metrica vigilada."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """
        @param {int} patience Epocas malas consecutivas que disparan la parada: con
            patience=N, step() devuelve True en la N-esima epoca consecutiva sin mejora.
        @param {float} min_delta Mejora minima para considerarse tal.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.num_bad_epochs = 0

    def step(self, metric: float) -> bool:
        """
        Registra la metrica de la epoca y decide si hay que parar.

        @param {float} metric Valor a maximizar (val_macro_f1).
        @returns {bool} True si se agoto la paciencia.
        """
        if metric > self.best + self.min_delta:
            self.best = metric
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
