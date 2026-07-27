"""Construccion de la funcion de perdida del pipeline principal.

El pipeline principal balancea con perdida ponderada en vez de WeightedRandomSampler:
sobre el dataset completo el desbalance llega a 32.9x, donde combinar sampler y pesos
sobre-compensaria el mismo desbalance por dos vias. `sqrt_inverse` es el default porque
comprime el rango de pesos de 32.9x a 5.7x, evitando que las 186 imagenes de train de
potassium_deficiency dominen el gradiente.
"""

from __future__ import annotations

from collections import Counter

import torch

_STRATEGIES = ("none", "inverse", "sqrt_inverse")


def compute_class_weights(
    labels: list[str],
    class_to_idx: dict[str, int],
    strategy: str,
) -> torch.Tensor | None:
    """
    Calcula el peso por clase a partir de la distribucion del split de entrenamiento.

    @param {list[str]} labels Etiquetas de texto de cada muestra de entrenamiento.
    @param {dict[str,int]} class_to_idx Mapeo canonico clase->indice.
    @param {str} strategy Una de "none", "inverse" o "sqrt_inverse".
    @returns {torch.Tensor|None} Pesos ordenados por indice de clase, o None si strategy es "none".
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Estrategia de pesos desconocida: '{strategy}'. Validas: {', '.join(_STRATEGIES)}"
        )
    if strategy == "none":
        return None

    counts = Counter(list(labels))
    missing = set(class_to_idx) - set(counts)
    if missing:
        raise ValueError(f"Clases sin muestras en el split de train: {sorted(missing)}")

    total = sum(counts.values())
    num_classes = len(class_to_idx)
    weights = torch.zeros(num_classes, dtype=torch.float)
    for class_name, class_idx in class_to_idx.items():
        weight = total / (num_classes * counts[class_name])
        weights[class_idx] = weight**0.5 if strategy == "sqrt_inverse" else weight
    return weights


def build_criterion(
    labels: list[str],
    class_to_idx: dict[str, int],
    strategy: str = "none",
    label_smoothing: float = 0.0,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Construye la CrossEntropy del run, con pesos de clase y label smoothing opcionales.

    @param {list[str]} labels Etiquetas del split de entrenamiento.
    @param {dict[str,int]} class_to_idx Mapeo canonico clase->indice.
    @param {str} strategy Estrategia de pesos.
    @param {float} label_smoothing Suavizado de etiquetas; 0.0 lo desactiva.
    @param {torch.device|None} device Dispositivo al que mover los pesos.
    @returns {torch.nn.Module} Criterio listo para el loop.
    """
    weights = compute_class_weights(labels, class_to_idx, strategy)
    if weights is not None and device is not None:
        weights = weights.to(device)
    return torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
