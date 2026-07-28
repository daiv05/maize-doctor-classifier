import json

import numpy as np

from src.explainability.global_report import GlobalAccumulator, write_global_report


def _half_leaf_image() -> np.ndarray:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :8] = (40, 160, 40)
    image[:, 8:] = (140, 120, 100)
    return image


def _half_segments() -> np.ndarray:
    segments = np.zeros((16, 16), dtype=np.int64)
    segments[:, 8:] = 1
    return segments


def _uniform_image() -> np.ndarray:
    return np.full((16, 16, 3), 128, dtype=np.uint8)


def test_attribution_on_the_leaf_yields_ratio_one():
    accumulator = GlobalAccumulator()
    accumulator.accumulate(
        label="healthy",
        correct=True,
        shap_values=np.array([1.0, 0.0]),
        segments=_half_segments(),
        image_np=_half_leaf_image(),
    )

    row = accumulator.summary().iloc[0]

    assert row["mean_leaf_attribution_ratio"] == 1.0
    assert row["mean_mask_coverage"] == 0.5
    assert row["n"] == 1
    assert row["n_mask_rejected"] == 0
    assert bool(row["ratio_reliable"])


def test_attribution_on_the_background_yields_ratio_zero():
    accumulator = GlobalAccumulator()
    accumulator.accumulate(
        label="healthy",
        correct=True,
        shap_values=np.array([0.0, 1.0]),
        segments=_half_segments(),
        image_np=_half_leaf_image(),
    )

    assert accumulator.summary().iloc[0]["mean_leaf_attribution_ratio"] == 0.0


def test_degenerate_mask_is_rejected_and_flags_the_class():
    accumulator = GlobalAccumulator()
    accumulator.accumulate(
        label="nitrogen_deficiency",
        correct=False,
        shap_values=np.array([1.0, 0.0]),
        segments=_half_segments(),
        image_np=_uniform_image(),
    )

    row = accumulator.summary().iloc[0]

    assert row["n_mask_rejected"] == 1
    assert not bool(row["ratio_reliable"])


def test_class_map_averages_over_the_accumulated_images():
    accumulator = GlobalAccumulator()
    for _ in range(2):
        accumulator.accumulate(
            label="healthy",
            correct=True,
            shap_values=np.array([1.0, 0.0]),
            segments=_half_segments(),
            image_np=_half_leaf_image(),
        )

    class_map = accumulator.class_maps()["healthy"]

    assert class_map.shape == (16, 16)
    assert class_map[:, :8].mean() == 1.0
    assert class_map[:, 8:].mean() == 0.0


def test_ratio_undefined_when_attribution_is_entirely_non_positive():
    accumulator = GlobalAccumulator()
    accumulator.accumulate(
        label="healthy",
        correct=True,
        shap_values=np.array([-1.0, -2.0]),
        segments=_half_segments(),
        image_np=_half_leaf_image(),
    )

    row = accumulator.summary().iloc[0]

    assert row["n"] == 1
    assert row["n_mask_rejected"] == 0
    assert row["n_ratio_undefined"] == 1
    assert np.isnan(row["mean_leaf_attribution_ratio"])
    assert not bool(row["ratio_reliable"])


def test_ratio_reliable_is_true_exactly_at_the_30_percent_boundary():
    accumulator = GlobalAccumulator()
    for _ in range(3):
        accumulator.accumulate(
            label="nitrogen_deficiency",
            correct=False,
            shap_values=np.array([1.0, 0.0]),
            segments=_half_segments(),
            image_np=_uniform_image(),
        )
    for _ in range(7):
        accumulator.accumulate(
            label="nitrogen_deficiency",
            correct=False,
            shap_values=np.array([1.0, 0.0]),
            segments=_half_segments(),
            image_np=_half_leaf_image(),
        )

    row = accumulator.summary().iloc[0]

    assert row["n"] == 10
    assert row["n_mask_rejected"] == 3
    assert row["n_ratio_undefined"] == 0
    assert bool(row["ratio_reliable"])


def test_write_global_report_emits_maps_and_summary(tmp_path):
    accumulator = GlobalAccumulator()
    accumulator.accumulate(
        label="healthy",
        correct=True,
        shap_values=np.array([1.0, 0.0]),
        segments=_half_segments(),
        image_np=_half_leaf_image(),
    )

    write_global_report(accumulator, tmp_path)

    assert (tmp_path / "healthy_attribution_map.png").exists()
    assert (tmp_path / "global_summary.csv").exists()
    payload = json.loads((tmp_path / "global_summary.json").read_text(encoding="utf-8"))
    assert payload[0]["label"] == "healthy"
