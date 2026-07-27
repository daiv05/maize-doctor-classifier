import pytest
import torch

from src.training.optim import EarlyStopping, build_scheduler


def _optimizer(lr: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_kind_none_no_devuelve_scheduler():
    assert build_scheduler(_optimizer(), "none", total_epochs=10) is None


def test_warmup_arranca_por_debajo_del_lr_base():
    optimizer = _optimizer(lr=1e-4)
    build_scheduler(optimizer, "cosine", total_epochs=10, warmup_epochs=3)
    assert optimizer.param_groups[0]["lr"] < 1e-4


def test_warmup_alcanza_el_lr_base_al_terminar():
    optimizer = _optimizer(lr=1e-4)
    scheduler = build_scheduler(optimizer, "cosine", total_epochs=10, warmup_epochs=3)
    for _ in range(3):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4, rel=1e-3)


def test_cosine_desciende_tras_el_warmup():
    optimizer = _optimizer(lr=1e-4)
    scheduler = build_scheduler(optimizer, "cosine", total_epochs=10, warmup_epochs=3)
    for _ in range(3):
        scheduler.step()
    peak = optimizer.param_groups[0]["lr"]
    for _ in range(5):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] < peak


def test_early_stopping_dispara_tras_patience_epocas_sin_mejora():
    stopper = EarlyStopping(patience=3)
    assert stopper.step(0.80) is False
    assert stopper.step(0.79) is False
    assert stopper.step(0.78) is False
    assert stopper.step(0.77) is True


def test_early_stopping_se_reinicia_con_una_mejora():
    stopper = EarlyStopping(patience=2)
    stopper.step(0.80)
    stopper.step(0.79)
    assert stopper.step(0.85) is False
    assert stopper.num_bad_epochs == 0
    assert stopper.best == pytest.approx(0.85)


def test_min_delta_ignora_mejoras_insignificantes():
    stopper = EarlyStopping(patience=1, min_delta=0.01)
    stopper.step(0.80)
    assert stopper.step(0.8005) is True
