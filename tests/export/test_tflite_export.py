import importlib
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import src.config

pytest.importorskip("litert_torch")


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


def test_export_to_tflite_produce_archivo(tmp_path, tmp_splits_dir, fake_image_root):
    from src.export.runtime import load_exported_runner
    from src.export.tflite_export import export_to_tflite
    from src.models import build_model

    test_dataset, _ = _build_test_loader(tmp_splits_dir, fake_image_root)
    model = build_model(
        "shufflenet_v2_x1_0", num_classes=len(test_dataset.class_to_idx), pretrained=False
    )
    model.eval()

    output_path = tmp_path / "export" / "model.tflite"
    result = export_to_tflite(model, output_path, (32, 32), torch.device("cpu"))

    assert result == output_path
    assert output_path.exists()
    batch = torch.randn(3, 3, 32, 32)
    with torch.inference_mode():
        expected = model(batch).numpy()
    actual = load_exported_runner(output_path, "tflite")(batch.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)


def test_tflite_logits_and_features_runtime_parity(tmp_path):
    from src.export.runtime import load_exported_runner
    from src.export.tflite_export import export_to_tflite

    class WithFeatures(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 2)

        def forward(self, image):
            features = image.mean(dim=(2, 3))
            return self.head(features), features

    torch.manual_seed(42)
    model = WithFeatures().eval()
    path = tmp_path / "multi.tflite"
    export_to_tflite(model, path, (8, 8), torch.device("cpu"))
    batch = torch.randn(3, 3, 8, 8)
    with torch.inference_mode():
        expected = [tensor.numpy() for tensor in model(batch)]
    runner = load_exported_runner(path, "tflite")
    outputs = runner.all_outputs(batch.numpy())
    assert len(outputs) == 2
    for actual, reference in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(runner(batch.numpy()), expected[0], rtol=1e-4, atol=1e-5)
