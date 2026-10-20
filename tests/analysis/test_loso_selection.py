"""Selection lock must precede any LOSO holdout access."""

import json

import pandas as pd
import pytest

from scripts.experiments.loso_finalize import (
    BASELINE_CONFIG_SHA256,
    evaluate_holdout,
    lock_selection,
)
from scripts.experiments.prepare_loso_baseline import sha256


def fixture_run(tmp_path):
    source = "maize-diseases"
    run = tmp_path / "runs" / "efficientnet_lite0" / "20260930_144843"
    splits = tmp_path / "splits"
    run.mkdir(parents=True)
    splits.mkdir()
    split_lock = splits / "manifest.lock.json"
    split_lock.write_text(json.dumps({"held_out_source": source}))
    checkpoint = run / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    summary = {
        "model": "efficientnet_lite0",
        "seed": 42,
        "config_sha256": BASELINE_CONFIG_SHA256,
        "split_manifest_sha256": sha256(split_lock),
        "test_used": False,
        "evaluation_mode": "validation_only",
        "metrics": {"best_validation": {"macro_f1": 0.8}},
        "checkpoint_sha256": sha256(checkpoint),
        "best_epoch": 2,
        "best_val_macro_f1": 0.8,
        "run_id": run.name,
    }
    (run / "summary.json").write_text(json.dumps(summary))
    pd.DataFrame({"epoch": [1, 2], "val_macro_f1": [0.7, 0.8]}).to_csv(
        run / "train_history.csv", index=False
    )
    return run, splits, source


def test_evaluation_requires_selection_lock(tmp_path):
    run, splits, source = fixture_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="selection.lock.json"):
        evaluate_holdout(run, splits, source)
    locked = lock_selection(run, splits, source)
    assert locked["checkpoint_sha256"] == sha256(run / "best.pth")
    assert locked["selection_criterion"] == "internal_validation_macro_f1"
    assert not (run / "holdout_predictions.csv").exists()
    with pytest.raises(FileExistsError, match="ya existe"):
        lock_selection(run, splits, source)


def test_test_metrics_forbid_lock(tmp_path):
    run, splits, source = fixture_run(tmp_path)
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["metrics"]["test"] = {"macro_f1": 0.9}
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="test antes del lock"):
        lock_selection(run, splits, source)


def test_changed_config_forbids_lock(tmp_path):
    run, splits, source = fixture_run(tmp_path)
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["config_sha256"] = "0" * 64
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="configuración efectiva"):
        lock_selection(run, splits, source)
