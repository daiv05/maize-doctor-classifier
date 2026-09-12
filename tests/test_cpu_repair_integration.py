"""Actual CPU training -> best checkpoint -> shuffled IDs -> ONNX -> mobile gate.

Synthetic data tests integrity, not agronomic accuracy or device readiness.
"""

import json

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.pipeline.sync_mobile_model import sync_mobile_model
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory
from src.export.common import export_model, write_export_summary
from src.export.evaluate import evaluate_exported_model, write_evaluation
from src.provenance import sha256_file
from src.training.artifacts import write_predictions_csv, write_summary
from src.training.loop import fit, run_epoch


def test_real_cpu_pipeline(tmp_path, fake_image_root, tmp_splits_dir, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr("src.data.dataset.get_dataset_root", lambda: fake_image_root)
    torch.manual_seed(17)
    factory = CornTransformFactory(target_size=(16, 16), clahe=True)
    dataset = CornDataset(tmp_splits_dir / "train.csv", transform=factory.get_pipeline("test"))
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, len(dataset.class_to_idx)),
    )
    run = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    history = fit(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        epochs=2,
        model_name="tiny",
        run_dir=run,
    )
    best = torch.load(run / "best.pth", weights_only=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in best.items())
    _, _, preds, probs = run_epoch(model, loader, criterion, torch.device("cpu"))
    predictions = write_predictions_csv(run, dataset, dataset.idx_to_class, preds, probs)
    assert set(predictions.sample_id) == set(dataset.data_frame.sample_id)
    write_summary(
        run,
        {
            "model": "tiny",
            "class_to_idx": dataset.class_to_idx,
            "image_size": [16, 16],
            "preprocessing": factory.to_contract(),
            "splits_dir": str(tmp_splits_dir),
            "history": history,
        },
    )
    report = export_model(
        model,
        run,
        "tiny",
        dataset.class_to_idx,
        (16, 16),
        ["onnx"],
        test_loader=loader,
        device=torch.device("cpu"),
        parity_sample_size=20,
    )
    assert report.formats[0].succeeded, report.formats[0].error
    write_export_summary(run, report)
    evaluation, frame = evaluate_exported_model(
        run / "export/model.onnx", "onnx", loader, dataset.idx_to_class, torch_model=model
    )
    assert evaluation.agreement_rate == 1.0
    assert evaluation.macro_f1_delta == 0.0
    assert frame.sample_id.nunique() == len(dataset)
    metadata = dataset.data_frame.set_index("sample_id")
    assert frame.environment.tolist() == metadata.loc[frame.sample_id, "environment"].tolist()
    evaluation.evaluated_split_sha256 = sha256_file(tmp_splits_dir / "test.csv")
    write_evaluation(run, evaluation, frame)
    path = sync_mobile_model(run, tmp_path / "bundle", fmt="onnx", quantize=None, require_ood=False)
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "validated_bundle"
    assert manifest["device_validation"] == "not_performed"
    assert len(pd.read_csv(run / "export/eval_onnx_predictions.csv")) == len(dataset)
