from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.fairness import (
    compute_disparity_metrics,
    compute_subgroup_metrics,
    evaluate_background_shortcut,
    evaluate_dual_shortcut_audit,
    plot_disaggregated_confusion_matrices,
    plot_subgroup_disparity_bars,
)


def test_compute_subgroup_metrics_exact():
    class_names = ["healthy", "common_rust", "blight"]
    # 6 muestras lab (3 acertadas, 3 erradas) -> Acc 0.50
    # 6 muestras real (6 acertadas) -> Acc 1.00
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2]  # lab: 0->0, 1->1, 2->0, 0->0, 1->0, 2->0
    subgroups = ["lab"] * 6 + ["real"] * 6

    results = compute_subgroup_metrics(y_true, y_pred, subgroups, class_names)

    assert "subgroups" in results
    assert "lab" in results["subgroups"]
    assert "real" in results["subgroups"]
    assert results["subgroups"]["real"]["accuracy"] == pytest.approx(1.0)
    assert results["subgroups"]["lab"]["accuracy"] == pytest.approx(0.50)

    # Dos muestras blight en lab, ambas predichas como 0: recall=0, FNR=1.
    assert results["subgroups"]["lab"]["class_fnr"]["blight"] == pytest.approx(1.0)
    assert results["subgroups"]["real"]["class_fnr"]["blight"] == pytest.approx(0.0)


def test_compute_disparity_metrics_perfect_and_disparate():
    result = compute_subgroup_metrics(
        [0, 1, 0, 1], [0, 1, 0, 1], ["lab", "lab", "real", "real"], ["a", "b"]
    )
    disparity = compute_disparity_metrics(result)
    assert disparity["status"] == "descriptive"
    assert disparity["ratio"] == pytest.approx(1.0)
    assert disparity["threshold"] is None
    result = compute_subgroup_metrics(
        [0, 1, 0, 1], [0, 1, 0, 0], ["lab", "lab", "real", "real"], ["a", "b"]
    )
    disparity = compute_disparity_metrics(result)
    assert disparity["ratio"] == pytest.approx(1 / 3)
    assert disparity["delta_macro_f1"] == pytest.approx(2 / 3)
    assert disparity["fnr_disparity_by_class"]["b"] == pytest.approx(1.0)


class DummyModel(nn.Module):
    """Modelo dummy que atiende al centro y predice la clase según el tensor."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.fc(h)


def test_evaluate_background_shortcut():
    device = torch.device("cpu")
    model = DummyModel().to(device)

    # Dataset de tensores sintéticos [N, C, H, W]
    images = torch.randn(10, 3, 64, 64)
    targets = torch.randint(0, 3, (10,))
    ds = TensorDataset(images, targets)
    loader = DataLoader(ds, batch_size=4)

    # 1. Test oclusión central
    res_center = evaluate_background_shortcut(
        model=model,
        loader=loader,
        device=device,
        mask_mode="center_occlusion",
    )

    assert "mean_original_confidence" in res_center
    assert "mean_masked_confidence" in res_center
    assert "confidence_drop" in res_center
    assert "confidence_retention_ratio" in res_center
    assert "accuracy_original" in res_center
    assert "accuracy_masked" in res_center
    assert "flip_rate" in res_center
    assert res_center["status"] == "sensitivity_only"
    assert "risk_level" not in res_center
    assert res_center["fixed_class"] == "original_prediction"

    # 2. Test oclusión periférica (control inverso)
    res_periph = evaluate_background_shortcut(
        model=model,
        loader=loader,
        device=device,
        mask_mode="peripheral_occlusion",
    )

    assert res_periph["mask_mode"] == "peripheral_occlusion"
    assert "accuracy_drop" in res_periph
    assert res_periph["status"] == "sensitivity_only"

    # 3. Test auditoría dual completa
    res_dual = evaluate_dual_shortcut_audit(
        model=model,
        loader=loader,
        device=device,
    )

    assert "center_occlusion" in res_dual
    assert "peripheral_occlusion" in res_dual
    assert res_dual["status"] == "sensitivity_only"
    assert "random_occlusion" in res_dual
    assert "diagnostic_summary" in res_dual


def test_plot_generation(tmp_path: Path):
    class_names = ["healthy", "common_rust", "blight"]
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 0, 2]
    subgroups = ["lab"] * 3 + ["real"] * 3

    subgroup_metrics = compute_subgroup_metrics(y_true, y_pred, subgroups, class_names)

    bars_path = tmp_path / "bars.png"
    plot_subgroup_disparity_bars(subgroup_metrics, bars_path)
    assert bars_path.exists()
    assert bars_path.stat().st_size > 0

    cm_path = tmp_path / "cm.png"
    plot_disaggregated_confusion_matrices(
        y_true_lab=[0, 1, 2],
        y_pred_lab=[0, 1, 2],
        y_true_real=[0, 1, 2],
        y_pred_real=[0, 0, 2],
        class_names=class_names,
        output_path=cm_path,
    )
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0
