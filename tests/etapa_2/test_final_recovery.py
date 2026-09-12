import json

import pytest

from scripts.etapa_2 import stage2_experiments as stage
from src.provenance import atomic_json


def test_final_recovery_freezes_selection_and_does_not_repeat_completed_work(tmp_path, monkeypatch):
    for path in (
        "master_manifest.csv",
        "holdout.csv",
        "holdout.lock.json",
        "tuning/best_params.json",
        "features/tiny.json",
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("frozen fixture")
    atomic_json(tmp_path / "ensemble/selection.json", {"models": ["tiny"]})
    monkeypatch.setattr(stage, "validate_holdout_lock", lambda *_: None)
    attempts = []

    def execute(root):
        attempts.append(1)
        final = root / "final"
        final.mkdir(exist_ok=True)
        if len(attempts) == 1:
            raise OSError("simulated interruption")
        atomic_json(final / "metrics.json", {"fixture": True})

    monkeypatch.setattr(stage, "_final_evaluation_locked", execute)
    with pytest.raises(OSError, match="interruption"):
        stage.final_evaluation(tmp_path)
    assert json.loads((tmp_path / "final_attempt.json").read_text())["status"] == "interrupted"
    stage.final_evaluation(tmp_path)
    stage.final_evaluation(tmp_path)
    assert len(attempts) == 2
    atomic_json(
        tmp_path / "ensemble/selection.json", {"models": ["tiny"], "weights": {"tiny": 0.5}}
    )
    with pytest.raises(ValueError, match="changed after holdout"):
        stage.final_evaluation(tmp_path)
    assert len(attempts) == 2
