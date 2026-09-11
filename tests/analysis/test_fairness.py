import json
from pathlib import Path
import numpy as np
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

    # Verificar FNR de blight (clase 2) en lab: 2 muestras de clase 2, ambas predichas como 0 -> recall 0.0 -> FNR 1.0
    assert results["subgroups"]["lab"]["class_fnr"]["blight"] == pytest.approx(1.0)
    assert results["subgroups"]["real"]["class_fnr"]["blight"] == pytest.approx(0.0)


def test_compute_disparity_metrics_perfect_and_disparate():
    # Caso 1: Paridad perfecta
    subgroup_res_perfect = {
        "subgroups": {
            "lab": {"macro_f1": 0.92, "accuracy": 0.94, "class_fnr": {"healthy": 0.05}},
            "real": {"macro_f1": 0.92, "accuracy": 0.94, "class_fnr": {"healthy": 0.05}},
        }
    }
    disp_perf = compute_disparity_metrics(subgroup_res_perfect)
    assert disp_perf["delta_macro_f1"] == pytest.approx(0.0)
    assert disp_perf["disparate_impact_ratio"] == pytest.approx(1.0)
    assert disp_perf["four_fifths_rule_passed"] is True

    # Caso 2: Disparidad notable (lab 0.90 vs real 0.60)
    subgroup_res_disp = {
        "subgroups": {
            "lab": {"macro_f1": 0.90, "accuracy": 0.92, "class_fnr": {"healthy": 0.05}},
            "real": {"macro_f1": 0.60, "accuracy": 0.65, "class_fnr": {"healthy": 0.35}},
        }
    }
    disp = compute_disparity_metrics(subgroup_res_disp)
    assert disp["delta_macro_f1"] == pytest.approx(0.30)
    assert disp["disparate_impact_ratio"] == pytest.approx(0.60 / 0.90, abs=1e-3)
    assert disp["four_fifths_rule_passed"] is False
    assert disp["fnr_disparity_by_class"]["healthy"] == pytest.approx(0.30)


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
    assert "shortcut_vulnerability_ratio" in res_center
    assert "accuracy_original" in res_center
    assert "accuracy_masked" in res_center
    assert "flip_rate" in res_center
    assert "shortcut_detected" in res_center
    assert "risk_level" in res_center
    assert isinstance(res_center["collapse_confirmed"], bool)

    # 2. Test oclusión periférica (control inverso)
    res_periph = evaluate_background_shortcut(
        model=model,
        loader=loader,
        device=device,
        mask_mode="peripheral_occlusion",
    )

    assert res_periph["mask_mode"] == "peripheral_occlusion"
    assert "accuracy_drop" in res_periph
    assert isinstance(res_periph["shortcut_detected"], bool)

    # 3. Test auditoría dual completa
    res_dual = evaluate_dual_shortcut_audit(
        model=model,
        loader=loader,
        device=device,
    )

    assert "center_occlusion" in res_dual
    assert "peripheral_occlusion" in res_dual
    assert "shortcut_confirmed" in res_dual
    assert "overall_risk_level" in res_dual
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
