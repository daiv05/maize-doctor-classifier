"""Contratos del análisis descriptivo por procedencia."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.experiments.source_analysis import (
    attach_predictions,
    comparison,
    source_class_distribution,
    source_distribution,
    source_metrics,
)


@pytest.fixture
def eligible() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "image_path": ["clean/a.jpg", "clean/b.jpg", "clean/c.jpg", "clean/d.jpg"],
            "label": ["healthy", "healthy", "rust", "rust"],
            "source_id": ["s1", "s1", "s2", "s2"],
            "environment": ["real"] * 4,
            "split": ["train", "validation", "test", "test"],
        }
    )


def test_join_uses_sample_id_not_row_order(tmp_path, eligible):
    split = eligible.loc[eligible["split"] == "test"]
    predictions = pd.DataFrame(
        {
            "sample_id": ["d", "c"],
            "label": ["rust", "rust"],
            "pred_label": ["healthy", "rust"],
            "pred_prob": [0.8, 0.9],
        }
    )
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    attached = attach_predictions(path, split)
    assert attached["sample_id"].tolist() == ["c", "d"]
    assert attached["pred_label"].tolist() == ["rust", "healthy"]


def test_join_rejects_wrong_identity_or_incomplete_coverage(tmp_path, eligible):
    split = eligible.loc[eligible["split"] == "test"]
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"sample_id": ["c"], "label": ["rust"], "pred_label": ["rust"], "pred_prob": [0.9]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cobertura"):
        attach_predictions(path, split)
    pd.DataFrame(
        {
            "sample_id": ["c", "d"],
            "label": ["healthy", "rust"],
            "pred_label": ["rust", "rust"],
            "pred_prob": [0.9, 0.9],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="label contradice"):
        attach_predictions(path, split)


def test_source_counts_and_source_class_zero_cells(eligible):
    counts = source_distribution(eligible).set_index("source_id")
    assert counts.loc["s1", ["total", "train", "validation", "test"]].tolist() == [2, 1, 1, 0]
    assert counts["total"].sum() == 4
    assert counts[["train", "validation", "test"]].sum().to_dict() == {
        "train": 1,
        "validation": 1,
        "test": 2,
    }
    matrix = source_class_distribution(eligible, ["healthy", "rust"])
    assert len(matrix) == 4
    assert matrix.query('source_id == "s1" and `class` == "rust"')["total"].item() == 0
    assert matrix.query('source_id == "s2" and `class` == "rust"')["pct_of_class"].item() == 100


def test_metrics_ignore_absent_ground_truth_classes(eligible):
    frame = eligible.loc[eligible["source_id"] == "s1"].copy()
    frame["pred_label"] = ["healthy", "rust"]
    frame["pred_prob"] = [0.8, 0.7]
    row = source_metrics(frame, "baseline", "validation").iloc[0]
    assert row["classes_present"] == "healthy"
    assert row["n_classes_present"] == 1
    assert row["macro_f1"] == pytest.approx(2 / 3)
    assert pd.isna(row["ece_15_bins"])


def test_comparison_delta_and_support():
    frame = pd.DataFrame(
        {
            "model": ["baseline", "hpo_trial_0"],
            "source_id": ["s", "s"],
            "samples": [8, 8],
            "classes_present": ["healthy;rust"] * 2,
            "macro_f1": [0.6, 0.7],
        }
    )
    result = comparison(frame).iloc[0]
    assert result["delta_hpo_minus_baseline"] == pytest.approx(0.1)
    frame.loc[1, "samples"] = 7
    with pytest.raises(ValueError, match="soportes"):
        comparison(frame)
