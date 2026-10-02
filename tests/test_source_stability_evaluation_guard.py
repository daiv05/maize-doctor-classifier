"""Final evaluation is one-shot and records the attempt before loading images."""

from __future__ import annotations

import json

import pytest

from scripts.experiments import source_stability_finalize as finalize


def _setup(tmp_path):
    run, splits = tmp_path / "run", tmp_path / "splits"
    run.mkdir()
    splits.mkdir()
    (run / "best.pth").write_bytes(b"checkpoint")
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (splits / "test.csv").write_text("sample_id,label\na,healthy\n", encoding="utf-8")
    (splits / "manifest.lock.json").write_text(
        json.dumps({"test_sha256": finalize.sha256_file(splits / "test.csv")}), encoding="utf-8"
    )
    (run / "selection.lock.json").write_text(
        json.dumps(
            {
                "scenario": "baseline",
                "seed": 123,
                "final_evaluation_count": 0,
                "checkpoint_sha256": finalize.sha256_file(run / "best.pth"),
                "summary_sha256": finalize.sha256_file(run / "summary.json"),
                "split_lock_sha256": finalize.sha256_file(splits / "manifest.lock.json"),
            }
        ),
        encoding="utf-8",
    )
    return run, splits


def test_repeated_evaluation_is_rejected_before_model_load(tmp_path, monkeypatch) -> None:
    run, splits = _setup(tmp_path)
    (run / "final_evaluation.started.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        finalize, "load_validated_run", lambda *_a, **_k: pytest.fail("model must not load")
    )
    with pytest.raises(FileExistsError):
        finalize.evaluate_final(run, splits, "baseline", 123)


def test_evaluation_count_written_before_any_image_load(tmp_path, monkeypatch) -> None:
    run, splits = _setup(tmp_path)
    committed = []

    def commit_guard() -> None:
        committed.append(json.loads((run / "final_evaluation.started.json").read_text()))

    monkeypatch.setattr(
        finalize, "load_validated_run", lambda *_a, **_k: pytest.fail("stop after guard")
    )
    with pytest.raises(pytest.fail.Exception):
        finalize.evaluate_final(run, splits, "baseline", 123, commit_guard=commit_guard)
    marker = json.loads((run / "final_evaluation.started.json").read_text())
    assert committed == [marker]
    assert marker["evaluation_count"] == 1
    assert marker["selection_lock_sha256"] == finalize.sha256_file(run / "selection.lock.json")
