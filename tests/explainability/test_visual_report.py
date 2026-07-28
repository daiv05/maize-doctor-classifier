import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from lime import lime_image
from PIL import Image

from src.explainability.visual_report import explain_model_visual, render_visual_explanation

_TARGET_SIZE = (8, 8)
_IDX_TO_CLASS = {0: "healthy", 1: "common_rust"}


class _FakeExplanation:
    """Sustituto de la explicacion de LIME, para no pagar el muestreo en los tests."""

    def __init__(self, segments: np.ndarray):
        self.segments = segments
        self.top_labels = [0]
        self.local_exp = {0: [(0, 1.0), (1, -0.5)]}

    def get_image_and_mask(self, label, positive_only, num_features, hide_rest):
        return None, (self.segments == 0).astype(int)


@pytest.fixture
def captured_kwargs(monkeypatch) -> dict:
    captured: dict = {}

    def fake_explain_instance(self, image, classifier_fn, **kwargs):
        captured.update(kwargs)
        segments = np.zeros(image.shape[:2], dtype=np.int64)
        segments[image.shape[0] // 2 :, :] = 1
        return _FakeExplanation(segments)

    monkeypatch.setattr(lime_image.LimeImageExplainer, "explain_instance", fake_explain_instance)
    return captured


def _dummy_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2)).eval()


def _dummy_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8))


def test_without_segments_lime_keeps_its_own_segmentation(captured_kwargs, tmp_path):
    render_visual_explanation(
        image=_dummy_image(),
        model=_dummy_model(),
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_path=tmp_path / "panel.png",
        num_samples=4,
        num_features=2,
    )

    assert "segmentation_fn" not in captured_kwargs


def test_segments_are_forwarded_to_lime(captured_kwargs, tmp_path):
    segments = np.zeros(_TARGET_SIZE, dtype=np.int64)
    segments[4:, :] = 1

    render_visual_explanation(
        image=_dummy_image(),
        model=_dummy_model(),
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_path=tmp_path / "panel.png",
        num_samples=4,
        num_features=2,
        segments=segments,
    )

    forwarded = captured_kwargs["segmentation_fn"](np.zeros((8, 8, 3), dtype=np.uint8))
    np.testing.assert_array_equal(forwarded, segments)


def test_writes_png_and_sidecars(captured_kwargs, tmp_path):
    output_path = tmp_path / "panel.png"

    render_visual_explanation(
        image=_dummy_image(),
        model=_dummy_model(),
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_path=output_path,
        num_samples=4,
        num_features=2,
    )

    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    assert output_path.with_suffix(".npy").exists()


def test_explain_model_visual_writes_under_explain_visual(
    captured_kwargs, tmp_path, fake_image_root, tmp_splits_dir
):
    run_dir = tmp_path / "run"

    explain_model_visual(
        model=_dummy_model(),
        model_name="dummy",
        test_df=pd.read_csv(tmp_splits_dir / "test.csv"),
        dataset_root=fake_image_root,
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_dir=run_dir,
        images_per_class=1,
        num_features=2,
        num_samples=4,
        seed=42,
        device=torch.device("cpu"),
        enable_gradcam=False,
    )

    assert (run_dir / "explain_visual").is_dir()
    assert not (run_dir / "lime_visual").exists()
