import importlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.config


class _FakeOpResolverType:
    BUILTIN_WITHOUT_DEFAULT_DELEGATES = "sin-delegados"


class _FakeInterpreter:
    escala_sin_delegados = 1.0

    def __init__(self, model_path, experimental_op_resolver_type=None):
        sin_delegados = (
            experimental_op_resolver_type
            == _FakeOpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        self._escala = self.escala_sin_delegados if sin_delegados else 1.0
        self._logits = None

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [{"index": 0, "dtype": "float32"}]

    def get_output_details(self):
        return [{"index": 1, "shape": [1, 3]}]

    def set_tensor(self, index, value):
        self._logits = np.array([1.0, 0.2, -0.5], dtype="float32") * self._escala

    def invoke(self):
        return None

    def get_tensor(self, index):
        return self._logits[None]


def _build_test_loader(splits_dir, dataset_root, image_size=(32, 32)):
    os.environ["DATASET_ROOT"] = str(dataset_root)
    importlib.reload(src.config)

    from src.data.dataset import CornDataset
    from src.data.transforms import CornTransformFactory

    factory = CornTransformFactory(target_size=image_size)
    test_dataset = CornDataset(
        csv_path=str(splits_dir / "test.csv"),
        transform=factory.get_pipeline("test"),
    )
    return test_dataset, DataLoader(test_dataset, batch_size=4, shuffle=False)


def _preparar(monkeypatch, tmp_splits_dir, fake_image_root, escala_sin_delegados):
    from src.export import parity
    from src.models import build_model

    _FakeInterpreter.escala_sin_delegados = escala_sin_delegados
    monkeypatch.setattr(
        parity, "_resolve_tflite_interpreter", lambda: (_FakeInterpreter, _FakeOpResolverType)
    )

    test_dataset, test_loader = _build_test_loader(tmp_splits_dir, fake_image_root)
    model = build_model(
        "shufflenet_v2_x1_0", num_classes=len(test_dataset.class_to_idx), pretrained=False
    )
    model.eval()
    return model, test_loader


def test_paridad_tflite_falla_si_los_dos_interpretes_discrepan(
    monkeypatch, tmp_path, tmp_splits_dir, fake_image_root
):
    from src.export.parity import validate_tflite_parity

    model, test_loader = _preparar(monkeypatch, tmp_splits_dir, fake_image_root, 300.0)

    result = validate_tflite_parity(
        model,
        tmp_path / "model_int8.tflite",
        test_loader,
        torch.device("cpu"),
        sample_size=8,
        tolerance=1.0,
        min_agreement_rate=0.0,
    )

    assert not result.passed
    assert result.cross_runtime_max_abs_prob_diff > result.cross_runtime_tolerance
    assert any("con y sin delegado" in w for w in result.warnings)


def test_paridad_tflite_no_penaliza_interpretes_que_coinciden(
    monkeypatch, tmp_path, tmp_splits_dir, fake_image_root
):
    from src.export.parity import validate_tflite_parity

    model, test_loader = _preparar(monkeypatch, tmp_splits_dir, fake_image_root, 1.0)

    result = validate_tflite_parity(
        model,
        tmp_path / "model.tflite",
        test_loader,
        torch.device("cpu"),
        sample_size=8,
        tolerance=1.0,
        min_agreement_rate=0.0,
    )

    assert result.cross_runtime_max_abs_prob_diff == 0.0
    assert result.cross_runtime_agreement_rate == 1.0
    assert not any("con y sin delegado" in w for w in result.warnings)


class _InterpreteConOps(_FakeInterpreter):
    escalas_pesos = [0.003]

    def get_tensor_details(self):
        return [
            {"index": 2, "quantization_parameters": {"scales": self.escalas_pesos}},
        ]

    def _get_ops_details(self):
        return [{"op_name": "FULLY_CONNECTED", "inputs": [0, 2, 3]}]


def _contar(escalas):
    from src.export.parity import _count_per_channel_fully_connected

    _InterpreteConOps.escalas_pesos = escalas
    return _count_per_channel_fully_connected(_InterpreteConOps("x"))


def test_cuenta_como_infractor_el_fully_connected_con_escalas_distintas():
    assert _contar([0.003067, 0.003037, 0.002890]) == 1


def test_no_cuenta_per_tensor_emitido_como_per_axis():
    assert _contar([0.003493, 0.003493, 0.003493]) == 0
    assert _contar([0.003493]) == 0


def test_devuelve_none_si_el_interprete_no_expone_las_operaciones():
    from src.export.parity import _count_per_channel_fully_connected

    assert _count_per_channel_fully_connected(_FakeInterpreter("x")) is None
