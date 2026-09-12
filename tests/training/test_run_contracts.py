import json

import pytest
import torch

from src.data.transforms import CornTransformFactory
from src.provenance import sha256_file
from src.training.hyperparameters import load_best_params
from src.training.runs import load_run, resolve_checkpoint, validate_ensemble_runs


def make_run(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.runs.build_model", lambda *a, **kw: torch.nn.Linear(2, 2))
    run = tmp_path / "main" / "tiny" / "r1"
    run.mkdir(parents=True)
    torch.save(torch.nn.Linear(2, 2).state_dict(), run / "best.pth")
    summary = {
        "checkpoint_sha256": sha256_file(run / "best.pth"),
        "model": "tiny",
        "class_to_idx": {"a": 0, "b": 1},
        "image_size": [16, 16],
        "preprocessing": CornTransformFactory(target_size=(16, 16)).to_contract(),
    }
    (run / "summary.json").write_text(json.dumps(summary))
    return run, summary


def test_missing_explicit_run_does_not_select_other_run(tmp_path, monkeypatch):
    run, summary = make_run(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("tiny", tmp_path, run_id="does-not-exist")
    assert resolve_checkpoint("tiny", tmp_path, run_id="r1") == run / "best.pth"
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("tiny", tmp_path, explicit_checkpoint=tmp_path / "missing.pth")


def test_architecture_and_weights_must_match(tmp_path, monkeypatch):
    run, summary = make_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="architecture"):
        load_run(run / "best.pth", "other")
    torch.save(torch.nn.Linear(3, 2).state_dict(), run / "best.pth")
    summary["checkpoint_sha256"] = sha256_file(run / "best.pth")
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_run(run / "best.pth", "tiny")


def test_reordered_classes_cannot_be_averaged(tmp_path, monkeypatch):
    path, summary = make_run(tmp_path, monkeypatch)
    first = load_run(path / "best.pth", "tiny")
    summary["class_to_idx"] = {"b": 0, "a": 1}
    (path / "summary.json").write_text(json.dumps(summary))
    second = load_run(path / "best.pth", "tiny")
    with pytest.raises(ValueError, match="class order"):
        validate_ensemble_runs([first, second])


@pytest.mark.parametrize(
    "params", [{"alpha": 0.01}, {"clahe": "false"}, {"learning_rate": -1}, {"batch_size": 1.5}]
)
def test_hpo_rejects_other_schema_and_invalid_values(tmp_path, params):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))
    with pytest.raises(ValueError):
        load_best_params(path)
