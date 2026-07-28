import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.explainability.compare_report import render_comparison

_TARGET_SIZE = (16, 16)
_IDX_TO_CLASS = {0: "healthy", 1: "common_rust"}
_LIME_CFG = {"num_samples": 20, "num_features": 3, "seed": 42}
_SHAP_CFG = {
    "segmentation": "slic",
    "n_segments": 4,
    "compactness": 10.0,
    "nsamples": 32,
    "batch_size": 8,
    "background": "black",
    "seed": 42,
}


def _dummy_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 2)).eval()


def _dummy_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8))


def _render(tmp_path):
    return render_comparison(
        image=_dummy_image(),
        model=_dummy_model(),
        model_name=None,
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_path=tmp_path / "compare.png",
        lime_cfg=_LIME_CFG,
        shap_cfg=_SHAP_CFG,
        device=torch.device("cpu"),
    )


def test_writes_png_and_sidecars(tmp_path):
    _render(tmp_path)

    assert (tmp_path / "compare.png").exists()
    assert (tmp_path / "compare.json").exists()
    assert (tmp_path / "compare.npy").exists()


def test_result_reports_prediction_and_agreement(tmp_path):
    result = _render(tmp_path)

    assert result["predicted_label"] in _IDX_TO_CLASS.values()
    assert 0.0 <= result["predicted_prob"] <= 1.0
    assert set(result["agreement"]) == {"iou_topk", "spearman", "sign_agreement"}


def test_sidecar_holds_both_attribution_vectors(tmp_path):
    _render(tmp_path)

    metadata = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))
    segments = np.load(tmp_path / "compare.npy")

    n_segments = int(segments.max()) + 1
    assert len(metadata["lime_weights"]) == n_segments
    assert len(metadata["shap_values"]) == n_segments


def test_lime_and_shap_share_the_segmentation(tmp_path):
    _render(tmp_path)

    metadata = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))

    assert len(metadata["lime_weights"]) == len(metadata["shap_values"])
    assert metadata["n_segments"] == len(metadata["shap_values"])
