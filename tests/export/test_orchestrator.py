import importlib
import json
import os

import pytest
import torch
from torch.utils.data import DataLoader

import src.config

pytest.importorskip("onnxruntime")


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
    loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return test_dataset, loader


def test_export_model_onnx_escribe_artefactos_y_pasa_paridad(
    tmp_path, tmp_splits_dir, fake_image_root
):
    from src.export.common import export_model, write_export_summary
    from src.models import build_model

    test_dataset, test_loader = _build_test_loader(tmp_splits_dir, fake_image_root)
    model = build_model(
        "shufflenet_v2_x1_0", num_classes=len(test_dataset.class_to_idx), pretrained=False
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = export_model(
        model=model,
        run_dir=run_dir,
        model_name="shufflenet_v2_x1_0",
        class_to_idx=test_dataset.class_to_idx,
        image_size=(32, 32),
        formats=["onnx"],
        test_loader=test_loader,
        device=torch.device("cpu"),
        parity_sample_size=8,
    )
    write_export_summary(run_dir, report)

    assert (run_dir / "export" / "model.onnx").exists()
    assert (run_dir / "export" / "labels.json").exists()
    assert report.formats[0].succeeded
    assert report.formats[0].parity.passed

    payload = json.loads((run_dir / "export" / "export_summary.json").read_text())
    assert payload["formats"][0]["parity"]["passed"] is True

    labels_payload = json.loads((run_dir / "export" / "labels.json").read_text())
    idx_to_class = {idx: name for name, idx in test_dataset.class_to_idx.items()}
    assert labels_payload["labels"] == [idx_to_class[i] for i in range(len(idx_to_class))]


def test_export_model_formato_desconocido_no_bloquea_los_demas(
    tmp_path, tmp_splits_dir, fake_image_root, monkeypatch
):
    from src.export.common import ExportDependencyError, export_model
    from src.models import build_model

    test_dataset, test_loader = _build_test_loader(tmp_splits_dir, fake_image_root)
    model = build_model(
        "shufflenet_v2_x1_0", num_classes=len(test_dataset.class_to_idx), pretrained=False
    )

    def _fail_tflite(*args, **kwargs):
        raise ExportDependencyError("dependencia ausente simulada")

    monkeypatch.setattr("src.export.tflite_export.export_to_tflite", _fail_tflite, raising=False)

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = export_model(
        model=model,
        run_dir=run_dir,
        model_name="shufflenet_v2_x1_0",
        class_to_idx=test_dataset.class_to_idx,
        image_size=(32, 32),
        formats=["onnx", "tflite"],
        test_loader=test_loader,
        device=torch.device("cpu"),
        parity_sample_size=8,
    )

    onnx_result = next(f for f in report.formats if f.format == "onnx")
    tflite_result = next(f for f in report.formats if f.format == "tflite")

    assert onnx_result.succeeded
    assert not tflite_result.succeeded
    assert "dependencia ausente simulada" in tflite_result.error


def test_export_model_sin_test_loader_omite_paridad(tmp_path, tmp_splits_dir, fake_image_root):
    from src.export.common import export_model
    from src.models import build_model

    test_dataset, _ = _build_test_loader(tmp_splits_dir, fake_image_root)
    model = build_model(
        "shufflenet_v2_x1_0", num_classes=len(test_dataset.class_to_idx), pretrained=False
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = export_model(
        model=model,
        run_dir=run_dir,
        model_name="shufflenet_v2_x1_0",
        class_to_idx=test_dataset.class_to_idx,
        image_size=(32, 32),
        formats=["onnx"],
        test_loader=None,
        device=torch.device("cpu"),
    )

    assert not report.formats[0].succeeded
    assert report.formats[0].output_path.is_file()
    assert report.formats[0].parity is None
    assert "Unvalidated" in report.formats[0].error
