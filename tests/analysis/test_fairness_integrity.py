import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.fairness import (
    apply_spatial_mask,
    compute_disparity_metrics,
    compute_subgroup_metrics,
    evaluate_background_shortcut,
)


def test_absent_class_is_undefined_and_single_group_insufficient():
    result = compute_subgroup_metrics([0, 0], [0, 0], ["lab", "lab"], ["a", "b"])
    assert result["subgroups"]["lab"]["class_fnr"]["b"] is None
    assert result["subgroups"]["lab"]["class_support"]["b"] == 0
    assert compute_disparity_metrics(result)["status"] == "insufficient"
    json.dumps(result, allow_nan=False)


def test_comparison_uses_common_class_support():
    result = compute_subgroup_metrics(
        [0, 0, 1, 1], [0, 0, 0, 0], ["lab", "real", "real", "real"], ["a", "b"]
    )
    disparity = compute_disparity_metrics(result)
    assert disparity["common_classes"] == ["a"]
    assert disparity["ratio"] == 1.0
    assert disparity["comparison_support"] == {"lab": 1, "real": 1}


def test_rectangles_cover_36_and_64_percent_and_are_complementary():
    image = torch.ones(2, 3, 100, 100)
    center, fraction = apply_spatial_mask(image, "center_occlusion")
    peripheral, inverse_fraction = apply_spatial_mask(image, "peripheral_occlusion")
    assert fraction == pytest.approx(0.36)
    assert inverse_fraction == pytest.approx(0.64)
    assert torch.equal(center + peripheral, image)
    random, random_fraction = apply_spatial_mask(
        image, "random_occlusion", torch.Generator().manual_seed(4)
    )
    assert random_fraction == pytest.approx(0.36)
    assert not torch.equal(random, center)


def test_confidence_tracks_original_class_even_when_prediction_flips():
    class Center(torch.nn.Module):
        def forward(self, x):
            center = x[:, :, 2:8, 2:8].mean((1, 2, 3))
            return torch.stack((10 * center, 10 * (1 - center)), dim=1)

    loader = DataLoader(TensorDataset(torch.ones(2, 3, 10, 10), torch.zeros(2, dtype=torch.long)))
    result = evaluate_background_shortcut(Center(), loader, torch.device("cpu"))
    assert result["mean_masked_confidence"] < 0.001
    assert result["flip_rate"] == 1.0
    assert result["accuracy_drop"] == 1.0
    assert result["status"] == "sensitivity_only"
    with pytest.raises(ValueError):
        evaluate_background_shortcut(Center(), loader, torch.device("cpu"), "unknown")
