"""A restarted Modal finalizer must not evaluate a final split twice."""

import json

import pytest

from scripts.modal import source_stability_finalize as remote


def _run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "selection.lock.json").write_text("{}")
    return run


def test_pending_final_evaluation_is_safe_to_start(tmp_path):
    run = _run(tmp_path)
    assert remote.final_status(run, "baseline", 123) == ("pending", None)


def test_started_but_incomplete_evaluation_is_not_repeated(tmp_path):
    run = _run(tmp_path)
    (run / "final_evaluation.started.json").write_text("{}")
    with pytest.raises(RuntimeError, match="no repetir"):
        remote.final_status(run, "baseline", 123)


def test_complete_evaluation_is_verified_and_skipped(tmp_path):
    run = _run(tmp_path)
    lock_sha = remote.sha256(run / "selection.lock.json")
    marker = {
        "scenario": "maize-diseases",
        "seed": 2026,
        "evaluation_count": 1,
        "selection_lock_sha256": lock_sha,
    }
    metrics = {**marker, "macro_f1": 0.9}
    (run / "final_evaluation.started.json").write_text(json.dumps(marker))
    (run / "holdout_metrics.json").write_text(json.dumps(metrics))
    for suffix in (
        "predictions.csv",
        "class_metrics.csv",
        "confusion_matrix.csv",
        "confusion_normalized.csv",
    ):
        (run / f"holdout_{suffix}").write_text("verified")
    assert remote.final_status(run, "maize-diseases", 2026) == ("complete", metrics)
    (run / "holdout_predictions.csv").unlink()
    with pytest.raises(RuntimeError, match="no repetir"):
        remote.final_status(run, "maize-diseases", 2026)
