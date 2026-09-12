import pytest
import torch
import torch.nn as nn

from src.models.ensemble import SoftVotingEnsemble


class DummyClassifier(nn.Module):
    """Modelo dummy que devuelve logits fijos predefinidos para pruebas."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.logits.repeat(batch_size, 1)


def test_ensemble_validation_errors():
    with pytest.raises(ValueError, match="al menos un modelo"):
        SoftVotingEnsemble(models=[])

    m1 = DummyClassifier(torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="no coincide"):
        SoftVotingEnsemble(models=[m1], weights=[0.5, 0.5])

    with pytest.raises(ValueError, match="mayor a 0"):
        SoftVotingEnsemble(models=[m1], weights=[0.0])


def test_ensemble_uniform_weights():
    # Modelo 1 favorece clase 0: [2.0, 0.0]
    # Modelo 2 favorece clase 1: [0.0, 2.0]
    m1 = DummyClassifier(torch.tensor([2.0, 0.0]))
    m2 = DummyClassifier(torch.tensor([0.0, 2.0]))

    ensemble = SoftVotingEnsemble(models=[m1, m2])
    assert torch.allclose(ensemble.weights, torch.tensor([0.5, 0.5]))

    x = torch.zeros((2, 3, 224, 224))
    probs = ensemble.predict_probabilities(x)

    assert probs.shape == (2, 2)
    # Con pesos iguales y logits simétricos, las probabilidades para clase 0 y clase 1 deben ser 0.5
    assert torch.allclose(probs[:, 0], torch.tensor([0.5, 0.5]), atol=1e-5)
    assert torch.allclose(probs[:, 1], torch.tensor([0.5, 0.5]), atol=1e-5)


def test_ensemble_weighted_voting():
    # Modelo 1 da prob 1.0 a clase 0: [100.0, 0.0]
    # Modelo 2 da prob 1.0 a clase 1: [0.0, 100.0]
    m1 = DummyClassifier(torch.tensor([100.0, 0.0]))
    m2 = DummyClassifier(torch.tensor([0.0, 100.0]))

    # Ponderación 80% Modelo 1, 20% Modelo 2
    ensemble = SoftVotingEnsemble(models=[m1, m2], weights=[0.8, 0.2])

    x = torch.zeros((3, 3, 224, 224))
    probs = ensemble.predict_probabilities(x)
    preds = ensemble.predict_classes(x)

    assert torch.allclose(probs[:, 0], torch.tensor([0.8, 0.8, 0.8]), atol=1e-3)
    assert torch.allclose(probs[:, 1], torch.tensor([0.2, 0.2, 0.2]), atol=1e-3)
    assert (preds == 0).all()


def test_ensemble_forward_logits_compatibility():
    m1 = DummyClassifier(torch.tensor([1.0, 2.0, 3.0]))
    m2 = DummyClassifier(torch.tensor([3.0, 2.0, 1.0]))
    m3 = DummyClassifier(torch.tensor([2.0, 3.0, 1.0]))

    ensemble = SoftVotingEnsemble(models=[m1, m2, m3], model_names=["effnet", "shufflenet", "lite"])

    x = torch.zeros((4, 3, 224, 224))
    logits = ensemble(x)
    assert logits.shape == (4, 3)

    # Las log-probabilidades exponenciadas deben sumar 1.0
    probs = torch.exp(logits)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)
