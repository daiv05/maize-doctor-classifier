import json

import pytest
import torch

from src.training.loop import _metrics_from_predictions, fit


@pytest.mark.parametrize("persist", [True, False])
def test_restores_best_epoch_even_without_run_dir(monkeypatch, tmp_path, persist):
    model = torch.nn.Linear(1, 1, bias=False)
    epoch = [0]

    def controlled_epoch(model, loader, criterion, device, optimizer=None, **kwargs):
        if optimizer is not None:
            epoch[0] += 1
            with torch.no_grad():
                model.weight.fill_(epoch[0])
        score = [0.9, 0.7, 0.1][epoch[0] - 1]
        return {"loss": 1.0, "accuracy": score, "macro_f1": score}, [], [], []

    monkeypatch.setattr("src.training.loop.run_epoch", controlled_epoch)
    fit(
        model,
        [],
        [],
        None,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.device("cpu"),
        epochs=3,
        model_name="controlled",
        run_dir=tmp_path if persist else None,
    )
    assert model.weight.item() == 1.0
    if persist:
        assert torch.load(tmp_path / "last.pth", weights_only=True)["weight"].item() == 3.0
        assert json.loads((tmp_path / "training_state.json").read_text())["best_epoch"] == 1


def test_metrics_are_computed_independently():
    result = _metrics_from_predictions([0, 0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 0, 1, 1], 0.0, [0, 1, 2])
    assert result["accuracy"] == pytest.approx(4 / 7)
    assert (
        len(
            {
                round(result[key], 6)
                for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall")
            }
        )
        == 4
    )
