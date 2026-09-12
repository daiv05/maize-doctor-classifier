"""Evaluacion de un modelo exportado sobre el split de test.

Se inyecta un runner falso en lugar de cargar un .onnx/.tflite real: lo que se prueba aqui
es el calculo de metricas y la deteccion de degradacion, no el runtime de inferencia.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.export.evaluate import evaluate_exported_model, write_evaluation

IDX_TO_CLASS = {0: "healthy", 1: "common_rust", 2: "gray_leaf_spot"}


def _loader(labels: list[int]) -> DataLoader:
    images = torch.zeros(len(labels), 3, 8, 8)
    return DataLoader(TensorDataset(images, torch.tensor(labels)), batch_size=2)


def _runner_from(predictions: list[int]):
    """Runner que devuelve logits one-hot segun una lista fija de predicciones."""
    state = {"i": 0}

    def run(batch: np.ndarray) -> np.ndarray:
        n = batch.shape[0]
        chunk = predictions[state["i"] : state["i"] + n]
        state["i"] += n
        logits = np.full((n, len(IDX_TO_CLASS)), -10.0, dtype=np.float32)
        for row, pred in enumerate(chunk):
            logits[row, pred] = 10.0
        return logits

    return run


@pytest.fixture
def patched_loader(monkeypatch):
    def _apply(predictions):
        monkeypatch.setattr(
            "src.export.evaluate.load_exported_runner",
            lambda path, fmt: _runner_from(predictions),
        )

    return _apply


def test_evaluacion_perfecta(tmp_path, patched_loader):
    labels = [0, 1, 2, 0]
    patched_loader(labels)
    evaluation, frame = evaluate_exported_model(
        tmp_path / "model.onnx", "onnx", _loader(labels), IDX_TO_CLASS
    )
    assert evaluation.accuracy == 1.0
    assert evaluation.macro_f1 == 1.0
    assert evaluation.n_samples == 4
    assert len(frame) == 4


def test_per_class_y_support_cubren_todas_las_clases(tmp_path, patched_loader):
    labels = [0, 1, 2, 0]
    patched_loader([0, 1, 1, 0])  # falla la clase 2
    evaluation, _ = evaluate_exported_model(
        tmp_path / "model.onnx", "onnx", _loader(labels), IDX_TO_CLASS
    )
    assert set(evaluation.per_class_f1) == set(IDX_TO_CLASS.values())
    assert evaluation.per_class_f1["gray_leaf_spot"] == 0.0
    assert evaluation.per_class_support == {"healthy": 2, "common_rust": 1, "gray_leaf_spot": 1}


def test_desglose_por_entorno(tmp_path, patched_loader):
    labels = [0, 1, 2, 0]
    patched_loader(labels)
    environments = pd.Series(["lab", "lab", "real", "real"])
    with pytest.raises(ValueError, match="sample IDs"):
        evaluate_exported_model(
            tmp_path / "model.onnx",
            "onnx",
            _loader(labels),
            IDX_TO_CLASS,
            environments=environments,
        )


def test_environment_macro_f1_does_not_add_absent_predicted_class():
    from src.export.evaluate import _environment_breakdown

    frame = pd.DataFrame({"label_idx": [0, 0], "pred_idx": [0, 1], "environment": ["lab", "lab"]})
    result = _environment_breakdown(frame, IDX_TO_CLASS)["lab"]
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["accuracy"] == 0.5
    assert result["supported_classes"] == ["healthy"]
    assert result["per_class_support"]["common_rust"] == 0


def test_degradacion_vs_torch_queda_registrada(tmp_path, patched_loader):
    """Si el exportado predice peor que PyTorch, el delta debe ser negativo y avisar."""
    labels = [0, 1, 2, 0]
    patched_loader([0, 1, 1, 0])  # el exportado falla una

    class PerfectModel(torch.nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            out = torch.full((batch, 3), -10.0)
            for row in range(batch):
                out[row, labels[PerfectModel.seen + row]] = 10.0
            PerfectModel.seen += batch
            return out

    PerfectModel.seen = 0
    evaluation, frame = evaluate_exported_model(
        tmp_path / "model.onnx",
        "onnx",
        _loader(labels),
        IDX_TO_CLASS,
        torch_model=PerfectModel(),
        device=torch.device("cpu"),
    )
    assert evaluation.torch_accuracy == 1.0
    assert evaluation.accuracy < 1.0
    assert evaluation.macro_f1_delta < 0
    assert evaluation.agreement_rate < 1.0
    assert evaluation.warnings
    assert "torch_pred_label" in frame.columns


def test_write_evaluation_separa_fp32_de_int8(tmp_path, patched_loader):
    labels = [0, 1, 2, 0]
    patched_loader(labels)
    evaluation, frame = evaluate_exported_model(
        tmp_path / "model_int8.tflite",
        "tflite",
        _loader(labels),
        IDX_TO_CLASS,
        quantize="int8",
    )
    path = write_evaluation(tmp_path, evaluation, frame)
    assert path.name == "eval_tflite_int8.json"
    assert (tmp_path / "export" / "eval_tflite_int8_predictions.csv").exists()
    payload = json.loads(path.read_text())
    assert payload["quantize"] == "int8"
    assert payload["accuracy"] == 1.0
