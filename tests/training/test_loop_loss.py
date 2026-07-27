"""Fija como promedia `run_epoch` la perdida cuando el criterio lleva pesos de clase.

`CrossEntropyLoss(weight=..., reduction="mean")` divide por la suma de los pesos del
batch, no por su numero de muestras. Reponderar por el tamano del batch mezclaria las
dos normalizaciones y sub-representaria justo a los batches ricos en clases minoritarias.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.loop import run_epoch


class _FixedLogitsModel(torch.nn.Module):
    """Modelo determinista: devuelve los logits que se le pasan como entrada."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images


def _two_batches_with_different_class_mix() -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
    """
    Construye dos batches de composicion de clases distinta sobre 3 clases.

    @returns {tuple} Loader de batches de tamano 3, logits y etiquetas completos.
    """
    logits = torch.tensor(
        [
            [2.0, 0.5, 0.1],
            [0.2, 1.5, 0.3],
            [0.1, 0.4, 2.5],
            [1.0, 0.2, 0.7],
            [0.3, 0.9, 1.8],
            [0.6, 2.2, 0.4],
        ]
    )
    labels = torch.tensor([0, 0, 2, 0, 2, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=3, shuffle=False)
    return loader, logits, labels


def _manual_weighted_epoch_loss(
    logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor
) -> float:
    """
    Calcula a mano `sum(loss_i * w_i) / sum(w_i)` sobre todas las muestras.

    `reduction="none"` con `weight` ya devuelve `w_i * ce_i`, asi que el numerador se
    construye sobre la cross-entropy sin pesos para no aplicarlos dos veces.

    @param {torch.Tensor} logits Logits de todas las muestras.
    @param {torch.Tensor} labels Etiquetas reales de todas las muestras.
    @param {torch.Tensor} weight Pesos por clase.
    @returns {float} Perdida de epoca esperada.
    """
    per_sample = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)
    sample_weights = weight[labels]
    return float((per_sample * sample_weights).sum() / sample_weights.sum())


def test_perdida_con_pesos_iguala_el_promedio_ponderado_manual():
    loader, logits, labels = _two_batches_with_different_class_mix()
    weight = torch.tensor([1.0, 5.0, 12.0])
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    metrics, _, _, _ = run_epoch(
        _FixedLogitsModel(), loader, criterion, torch.device("cpu"), desc="test"
    )

    expected = _manual_weighted_epoch_loss(logits, labels, weight)
    assert metrics["loss"] == pytest.approx(expected)


def test_perdida_con_pesos_difiere_del_promedio_por_tamano_de_batch():
    """El bug corregido: ponderar por batch_size da otro numero con batches desbalanceados."""
    loader, logits, labels = _two_batches_with_different_class_mix()
    weight = torch.tensor([1.0, 5.0, 12.0])
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    metrics, _, _, _ = run_epoch(
        _FixedLogitsModel(), loader, criterion, torch.device("cpu"), desc="test"
    )

    naive_total = 0.0
    seen = 0
    for batch_logits, batch_labels in loader:
        naive_total += float(criterion(batch_logits, batch_labels)) * batch_labels.size(0)
        seen += batch_labels.size(0)
    naive = naive_total / seen

    assert metrics["loss"] != pytest.approx(naive)
    assert metrics["loss"] == pytest.approx(_manual_weighted_epoch_loss(logits, labels, weight))


def test_perdida_sin_pesos_es_el_promedio_simple_por_muestra():
    """Neutralidad de baselines: sin pesos, mean == sum/N y el numero no cambia."""
    loader, logits, labels = _two_batches_with_different_class_mix()
    criterion = torch.nn.CrossEntropyLoss()

    metrics, _, _, _ = run_epoch(
        _FixedLogitsModel(), loader, criterion, torch.device("cpu"), desc="test"
    )

    expected = float(torch.nn.CrossEntropyLoss(reduction="mean")(logits, labels))
    assert metrics["loss"] == pytest.approx(expected)


def test_perdida_sin_pesos_coincide_con_ponderar_por_tamano_de_batch():
    """La ruta sin pesos queda identica al comportamiento previo a la correccion."""
    loader, _, _ = _two_batches_with_different_class_mix()
    criterion = torch.nn.CrossEntropyLoss()

    metrics, _, _, _ = run_epoch(
        _FixedLogitsModel(), loader, criterion, torch.device("cpu"), desc="test"
    )

    legacy_total = 0.0
    seen = 0
    for batch_logits, batch_labels in loader:
        legacy_total += float(criterion(batch_logits, batch_labels)) * batch_labels.size(0)
        seen += batch_labels.size(0)

    assert metrics["loss"] == pytest.approx(legacy_total / seen)
