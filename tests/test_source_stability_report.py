"""Statistical and identity checks for the three-seed evidence builder."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.experiments.source_stability_report import _predictions, paths, stats


def test_sample_sd_uses_ddof_one() -> None:
    result = stats([0.8, 0.9, 1.0])
    assert result["n"] == 3
    assert result["ddof"] == 1
    assert math.isclose(result["mean"], 0.9)
    assert math.isclose(result["median"], 0.9)
    assert math.isclose(result["sample_sd"], 0.1)
    assert math.isclose(result["range"], 0.2)
    with pytest.raises(ValueError):
        stats([0.8, 0.9])


def test_predictions_join_on_sample_id_not_row_position(tmp_path) -> None:
    manifest = tmp_path / "val.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "label": ["healthy", "rust"]}).to_csv(
        manifest, index=False
    )
    pd.DataFrame(
        {
            "sample_id": ["b", "a"],
            "label": ["rust", "healthy"],
            "pred_label": ["rust", "healthy"],
            "pred_prob": [0.8, 0.9],
        }
    ).to_csv(predictions, index=False)
    aligned = _predictions(predictions, manifest)
    assert aligned["sample_id"].tolist() == ["a", "b"]
    pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "label": ["healthy", "healthy"],
            "pred_label": ["healthy", "healthy"],
            "pred_prob": [0.9, 0.9],
        }
    ).to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="sample_id"):
        _predictions(predictions, manifest)


def test_seed42_paths_are_reused_without_new_run_lookup() -> None:
    baseline, baseline_splits, _ = paths("baseline", 42)
    assert baseline.name == "20260921_204608"
    assert baseline_splits.name == "seed_42"
    with pytest.raises(ValueError):
        paths("baseline", 3407)


def test_final_metrics_and_class_rows_include_precision_recall_confidence() -> None:
    from scripts.experiments.source_stability_report import _class_rows, _metrics

    frame = pd.DataFrame(
        {
            "label": ["healthy", "healthy", "rust"],
            "pred_label": ["healthy", "rust", "healthy"],
            "pred_prob": [0.9, 0.8, 0.7],
        }
    )
    metrics = _metrics(frame)
    assert math.isclose(metrics["accuracy"], 1 / 3)
    assert math.isclose(metrics["error_rate"], 2 / 3)
    assert math.isclose(metrics["macro_precision"], 0.25)
    assert math.isclose(metrics["mean_confidence_correct"], 0.9)
    assert math.isclose(metrics["mean_confidence_errors"], 0.75)
    healthy = next(
        row for row in _class_rows(frame, "baseline", 42, "test") if row["class"] == "healthy"
    )
    assert healthy["support"] == 2
    assert math.isclose(healthy["precision"], 0.5)
    assert math.isclose(healthy["recall"], 0.5)
    assert math.isclose(healthy["f1"], 0.5)


def test_archived_interrupted_attempt_is_not_selected(tmp_path, monkeypatch) -> None:
    from scripts.experiments import source_stability_report as report

    monkeypatch.setattr(report, "LOSO", tmp_path)
    root = tmp_path / "maize-diseases" / "seed_123"
    valid = root / "runs" / "efficientnet_lite0" / "valid"
    valid.mkdir(parents=True)
    (valid / "summary.json").write_text("{}", encoding="utf-8")
    archived = root / "interrupted_attempts" / "attempt_1" / "efficientnet_lite0" / "old"
    archived.mkdir(parents=True)
    (archived / "summary.json").write_text("{}", encoding="utf-8")

    selected, _splits, selected_root = report.paths("maize-diseases", 123)
    assert selected == valid
    assert selected_root == root
