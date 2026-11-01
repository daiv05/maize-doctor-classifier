"""Guards for the fixed six-run plan and validation-before-final-evaluation gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.experiments import source_stability_finalize as finalize
from scripts.modal import source_stability_train as train


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def test_training_plan_never_uses_seed42_or_final_split() -> None:
    assert len(train.SCENARIOS) * len(train.SEEDS) == 6
    assert train.SEEDS == (123, 2026)
    for scenario in train.SCENARIOS:
        for seed in train.SEEDS:
            splits, root = train.scenario_paths(scenario, seed)
            cmd = train.command(scenario, seed, splits, root)
            assert cmd[cmd.index("--seed") + 1] == str(seed)
            assert "--skip-test" in cmd
            assert "--best-params" not in cmd
            assert "--max-per-class" in cmd and "--no-clahe" in cmd
            assert "seed_42" in str(splits) and f"seed_{seed}" in str(root)
    with pytest.raises(ValueError):
        train.scenario_paths("baseline", 42)


def test_frozen_manifests_checked_by_hash_without_rewriting(tmp_path, monkeypatch) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    hashes = {}
    for name in ("master_manifest.csv", "train.csv", "val.csv", "test.csv"):
        path = splits / name
        _write(path, f"{name}\n")
        hashes[name] = train.sha256_file(path)
    lock = {
        "master_manifest_sha256": hashes["master_manifest.csv"],
        "train_sha256": hashes["train.csv"],
        "val_sha256": hashes["val.csv"],
        "test_sha256": hashes["test.csv"],
    }
    lock_path = splits / "manifest.lock.json"
    _write(lock_path, json.dumps(lock))
    monkeypatch.setitem(train.LOCK_SHA256, "baseline", train.sha256_file(lock_path))
    before = {p.name: p.read_bytes() for p in splits.iterdir()}
    assert train.verify_splits("baseline", splits) == lock
    assert before == {p.name: p.read_bytes() for p in splits.iterdir()}
    _write(splits / "test.csv", "changed\n")
    with pytest.raises(ValueError, match="test.csv"):
        train.verify_splits("baseline", splits)


def test_selection_lock_is_after_validation_and_before_final_csv(tmp_path, monkeypatch) -> None:
    run_dir, splits = tmp_path / "run", tmp_path / "splits"
    run_dir.mkdir()
    splits.mkdir()
    _write(run_dir / "summary.json", "{}")
    _write(
        splits / "manifest.lock.json",
        json.dumps(
            {
                "master_manifest_sha256": "m" * 64,
                "train_sha256": "t" * 64,
                "val_sha256": "v" * 64,
                "test_sha256": "e" * 64,
            }
        ),
    )
    summary = {
        "run_id": "run",
        "best_epoch": 2,
        "best_val_macro_f1": 0.9,
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "c" * 64,
    }
    metadata = {
        "status": "validation_complete",
        "scenario": "baseline",
        "seed": 123,
        "run_id": "run",
        "checkpoint_sha256": "a" * 64,
        "git_commit": "commit",
        "git_dirty": True,
        "finished_at_utc": "2026-10-01T00:00:00+00:00",
    }
    monkeypatch.setattr(finalize, "verify_validation_run", lambda *_: summary)
    lock = finalize.lock_selection(run_dir, splits, "baseline", 123, {}, metadata)
    assert lock["final_evaluation_count"] == 0
    assert lock["selection_criterion"] == "internal_validation_macro_f1"
    assert lock["checkpoint_path"] == str(run_dir / "best.pth")
    assert lock["checkpoint_sha256"] == "a" * 64
    assert lock["config_sha256"] == "c" * 64
    assert lock["manifest_hashes"]["test.csv"] == "e" * 64
    assert lock["git_dirty"] is True
    assert lock["test_or_holdout_used_before_lock"] is False
    assert (run_dir / "selection.lock.json").is_file()
    assert not (run_dir / "test_predictions.csv").exists()
    with pytest.raises(FileExistsError):
        finalize.lock_selection(run_dir, splits, "baseline", 123, {}, metadata)


def test_incomplete_validation_run_rejected_before_lock(tmp_path, monkeypatch) -> None:
    run_dir, splits = tmp_path / "run", tmp_path / "splits"
    run_dir.mkdir()
    splits.mkdir()
    _write(splits / "manifest.lock.json", json.dumps({}))
    _write(run_dir / "train_history.csv", "epoch,val_macro_f1\n1,0.8\n2,0.9\n")
    summary = {
        "run_id": "run",
        "seed": 123,
        "split_manifest_sha256": finalize.sha256_file(splits / "manifest.lock.json"),
        "architecture": {},
        "hyperparameters": {},
        "preprocessing": {},
        "class_to_idx": {},
        "test_used": False,
        "evaluation_mode": "validation_only",
        "metrics": {},
        "best_epoch": 2,
        "best_val_macro_f1": 0.9,
    }
    monkeypatch.setattr(
        finalize, "validate_run_contract", lambda *_args, **_kw: SimpleNamespace(summary=summary)
    )
    reference = {
        key: summary[key]
        for key in ("architecture", "hyperparameters", "preprocessing", "class_to_idx")
    }
    assert finalize.verify_validation_run(run_dir, splits, "baseline", 123, reference) == summary
    summary["test_used"] = True
    with pytest.raises(ValueError, match="validation-only"):
        finalize.verify_validation_run(run_dir, splits, "baseline", 123, reference)


def test_best_checkpoint_must_match_validation_history(tmp_path, monkeypatch) -> None:
    run_dir, splits = tmp_path / "run", tmp_path / "splits"
    run_dir.mkdir()
    splits.mkdir()
    _write(splits / "manifest.lock.json", "{}")
    pd.DataFrame({"epoch": [1, 2], "val_macro_f1": [0.7, 0.9]}).to_csv(
        run_dir / "train_history.csv", index=False
    )
    summary = {
        "seed": 2026,
        "split_manifest_sha256": finalize.sha256_file(splits / "manifest.lock.json"),
        "architecture": {},
        "hyperparameters": {},
        "preprocessing": {},
        "class_to_idx": {},
        "test_used": False,
        "evaluation_mode": "validation_only",
        "metrics": {},
        "best_epoch": 1,
        "best_val_macro_f1": 0.7,
    }
    monkeypatch.setattr(
        finalize, "validate_run_contract", lambda *_args, **_kw: SimpleNamespace(summary=summary)
    )
    reference = {
        key: summary[key]
        for key in ("architecture", "hyperparameters", "preprocessing", "class_to_idx")
    }
    with pytest.raises(ValueError, match="Best checkpoint"):
        finalize.verify_validation_run(run_dir, splits, "baseline", 2026, reference)
