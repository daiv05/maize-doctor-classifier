import math

import pytest
import torch

from src.training.losses import build_criterion, compute_class_weights

CLASS_TO_IDX = {"healthy": 0, "common_rust": 1, "potassium_deficiency": 2}
LABELS = ["healthy"] * 12 + ["common_rust"] * 6 + ["potassium_deficiency"] * 2


def test_strategy_none_no_devuelve_pesos():
    assert compute_class_weights(LABELS, CLASS_TO_IDX, "none") is None


def test_inverse_da_ratio_igual_al_desbalance():
    weights = compute_class_weights(LABELS, CLASS_TO_IDX, "inverse")
    assert weights[2] / weights[0] == pytest.approx(6.0, abs=1e-5)


def test_sqrt_inverse_comprime_el_ratio():
    weights = compute_class_weights(LABELS, CLASS_TO_IDX, "sqrt_inverse")
    assert weights[2] / weights[0] == pytest.approx(math.sqrt(6.0), abs=1e-5)


def test_los_pesos_respetan_el_orden_de_class_to_idx():
    weights = compute_class_weights(LABELS, CLASS_TO_IDX, "inverse")
    assert weights[0] < weights[1] < weights[2]
    assert weights.shape == (3,)


def test_estrategia_desconocida_falla():
    with pytest.raises(ValueError, match="Estrategia de pesos desconocida"):
        compute_class_weights(LABELS, CLASS_TO_IDX, "magica")


def test_build_criterion_propaga_label_smoothing():
    criterion = build_criterion(LABELS, CLASS_TO_IDX, "sqrt_inverse", label_smoothing=0.1)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.1
    assert criterion.weight is not None


def test_build_criterion_sin_pesos_equivale_a_crossentropy_plana():
    criterion = build_criterion(LABELS, CLASS_TO_IDX, "none")
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.8, 0.3]])
    targets = torch.tensor([0, 1])
    expected = torch.nn.CrossEntropyLoss()(logits, targets)
    assert criterion(logits, targets) == pytest.approx(expected.item(), abs=1e-6)
