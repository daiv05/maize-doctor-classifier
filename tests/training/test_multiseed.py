"""Guardas multi-seed y smoke CPU sintético; cero entrenamientos formales."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.pipeline import train
from scripts.pipeline.multiseed import aggregate, report, statistics_for
from src.data.identity import ensure_sample_ids
from src.data.preparation import atomic_write_json, sha256_file
from src.training import multiseed as ms
from src.training.runs import (
    CheckpointIntegrityError,
    build_run_contract,
    validate_run_contract,
    write_run_contract,
)


@pytest.mark.parametrize("skip_test", [True, False])
def test_main_cpu_smoke(tmp_path, tmp_splits_dir, fake_image_root, monkeypatch, skip_test):
    """Loop/datos/augmentations reales; red mínima sin descargar pesos."""
    import src.config as config

    monkeypatch.setattr(config, "DATASET_ROOT", fake_image_root)
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        train,
        "build_model",
        lambda name, num_classes, pretrained: torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, num_classes)
        ),
    )
    original_dataset = train.CornDataset
    dataset_calls = []

    def guarded_dataset(**kwargs):
        name = Path(kwargs["csv_path"]).name
        dataset_calls.append(name)
        if skip_test:
            assert name != "test.csv", "Test dataset must never be constructed"
        return original_dataset(**kwargs)

    monkeypatch.setattr(train, "CornDataset", guarded_dataset)
    read_csv = pd.read_csv

    def guard_csv(path, *args, **kwargs):
        if skip_test:
            assert Path(path).name != "test.csv", "Test CSV must not be parsed"
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guard_csv)
    atomic_write_json(tmp_splits_dir / "manifest.lock.json", {"synthetic": True})
    args = [
        "train",
        "--models",
        "efficientnet_lite0",
        "--epochs",
        "1",
        "--warmup-epochs",
        "0",
        "--batch-size",
        "20",
        "--num-workers",
        "0",
        "--no-pretrained",
        "--splits-dir",
        str(tmp_splits_dir),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    if skip_test:
        args += ["--skip-test"]
    monkeypatch.setattr(sys, "argv", args)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        train.main()
    finally:
        torch.set_num_threads(previous)
    run = next((tmp_path / "out").glob("*/*/summary.json")).parent
    summary = validate_run_contract(run, splits_dir=tmp_splits_dir).summary
    if skip_test:
        assert dataset_calls == ["train.csv", "val.csv"]
        assert summary["test_used"] is False
        assert "test" not in summary["metrics"]
        assert not list(run.glob("test*"))
        predictions = pd.read_csv(run / "validation_predictions.csv")
        assert predictions.sample_id.is_unique
        expected = ensure_sample_ids(read_csv(tmp_splits_dir / "val.csv"))
        assert set(predictions.sample_id) == set(expected.sample_id)
        diag = ms.read_json(run / "validation_diagnostics.json")
        assert 0 <= diag["calibration"]["ece"] <= 1
        assert "potassium_deficiency" in diag["per_class"]
    else:
        assert dataset_calls == ["train.csv", "val.csv", "test.csv"]
        assert "test" in summary["metrics"]
        assert (run / "test_calibration.json").exists()


def test_export_with_skip_test_fails_before_data(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--skip-test", "--export", "onnx"])
    monkeypatch.setattr(train, "CornDataset", lambda **kw: pytest.fail("Loaded data"))
    with pytest.raises(SystemExit, match="no permite"):
        train.main()


def test_canonical_configs_and_seed_only_changes(tmp_path):
    protocol = ms.canonical_protocol()
    assert protocol["configurations"]["baseline"]["hyperparameters"]["batch_size"] == 32
    assert protocol["configurations"]["hpo_trial_0"]["hyperparameters"]["batch_size"] == 16
    for name in ms.CONFIGURATIONS:
        commands = [
            ms.training_command(protocol, name, seed, tmp_path, tmp_path) for seed in ms.SEEDS
        ]
        for command in commands:
            assert "--skip-test" in command
            assert "--export" not in command
            command[command.index("--seed") + 1] = "SEED"
        assert all(command == commands[0] for command in commands)


@pytest.fixture
def synthetic_splits(tmp_path):
    frame = ensure_sample_ids(
        pd.DataFrame(
            [
                {
                    "image_path": f"clean/healthy/real/{i}.png",
                    "label": "healthy",
                    "sha256": f"content_{i}",
                    "effective_group_id": f"group_{i}",
                }
                for i in range(6)
            ]
        )
    )
    frame.to_csv(tmp_path / "master_manifest.csv", index=False)
    for name, part in zip(
        ("train", "val", "test"), (frame.iloc[:2], frame.iloc[2:4], frame.iloc[4:])
    ):
        part.to_csv(tmp_path / f"{name}.csv", index=False)
    atomic_write_json(tmp_path / "manifest.lock.json", {"synthetic": True})
    hashes = {p.name: sha256_file(p) for p in tmp_path.iterdir()}
    return tmp_path, hashes


def test_audit_no_test_parse_and_hash_guard(synthetic_splits, monkeypatch):
    path, hashes = synthetic_splits
    original = pd.read_csv

    def guard(path, **kwargs):
        assert Path(path).name != "test.csv"
        return original(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guard)
    audit = ms.audit_splits(path, hashes)
    assert audit["integrity"] == {
        "sample_overlap_count": 0,
        "sha256_overlap_count": 0,
        "group_overlap_count": 0,
    }
    (path / "train.csv").write_text("modified")
    with pytest.raises(ValueError, match="hashes"):
        ms.audit_splits(path, hashes)


@pytest.mark.parametrize("leak", ["sample_id", "sha256", "effective_group_id"])
def test_audit_detects_leakage(synthetic_splits, leak):
    path, hashes = synthetic_splits
    if leak == "sample_id":
        pd.read_csv(path / "train.csv").to_csv(path / "val.csv", index=False)
    else:
        master = pd.read_csv(path / "master_manifest.csv")
        master.loc[2, leak] = master.loc[0, leak]
        master.to_csv(path / "master_manifest.csv", index=False)
    hashes = {name: sha256_file(path / name) for name in hashes}
    with pytest.raises(ValueError, match="comparten"):
        ms.audit_splits(path, hashes)


def test_statistics_paired_sample_sd_and_pending():
    rows = []
    for seed, base, delta in zip(
        ms.SEEDS, [0.8, 0.81, 0.82, 0.83, 0.84], [0.01, -0.01, 0, 0.02, 0.03]
    ):
        for name, value in (("baseline", base), ("hpo_trial_0", base + delta)):
            rows.append(
                {
                    "configuration": name,
                    "seed": seed,
                    "status": "COMPLETE",
                    **{m: value for m in ms.METRICS},
                }
            )
    result = aggregate(rows)
    assert result["complete"]
    assert result["paired_delta"]["mean"] == pytest.approx(0.01)
    assert result["paired_delta"]["std"] == pytest.approx(0.0158113883)
    assert (result["wins_hpo"], result["wins_baseline"], result["ties"]) == (3, 1, 1)
    assert statistics_for([])["mean"] is None
    assert statistics_for([0.9])["std"] is None
    assert not aggregate(rows[:-1])["complete"]


@pytest.fixture
def mocked_execution(tmp_path, monkeypatch):
    protocol = ms.canonical_protocol()
    calls = []
    monkeypatch.setattr(ms, "verify_sources", lambda *a: None)
    monkeypatch.setattr(ms, "audit_splits", lambda *a: {})
    monkeypatch.setattr(ms, "environment", lambda: {"gpu": "SYNTHETIC_MOCK"})

    def fake_run(command, **kwargs):
        calls.append(command)
        slot = Path(command[command.index("--output-dir") + 1])
        seed = int(command[command.index("--seed") + 1])
        name = slot.parent.name
        run = slot / ms.MODEL / "20260924_120000"
        run.mkdir(parents=True)
        (run / "best.pth").write_bytes(b"SYNTHETIC; NEVER LOAD")
        split_dir = tmp_path / "splits"
        split_dir.mkdir(exist_ok=True)
        (split_dir / "manifest.lock.json").write_text("{}")
        monkeypatch.setattr(
            "src.training.runs.split_manifest_fingerprint",
            lambda *a: (protocol["split_hashes"]["manifest.lock.json"], "manifest.lock.json"),
        )
        summary = build_run_contract(
            run_dir=run,
            model_name=ms.MODEL,
            seed=seed,
            hyperparameters=protocol["configurations"][name]["hyperparameters"],
            preprocessing=protocol["configurations"][name]["preprocessing"],
            class_to_idx=protocol["class_to_idx"],
            splits_dir=split_dir,
            best_epoch=1,
            metrics={"best_validation": {"epoch": 1, "macro_f1": 0.9, "accuracy": 0.95}},
            historical_fields={"test_used": False, "best_val_macro_f1": 0.9},
        )
        write_run_contract(run, summary)
        atomic_write_json(
            run / "validation_diagnostics.json",
            {
                "calibration": {"ece": 0.05},
                "per_class": {
                    f"{n}_deficiency": {"f1-score": 0.8}
                    for n in ("nitrogen", "phosphorus", "potassium")
                },
            },
        )
        (run / "train_history.csv").write_text("epoch,val_macro_f1\n1,0.9\n")
        (run / "validation_predictions.csv").write_text("sample_id\nsynthetic\n")

    monkeypatch.setattr(ms.subprocess, "run", fake_run)
    return tmp_path / "experiment", protocol, calls


def test_ten_slots_resume_without_duplicates_and_report(mocked_execution):
    directory, protocol, calls = mocked_execution
    ms.execute(directory, directory, protocol, max_new_runs=2)
    ms.execute(directory, directory, protocol, max_new_runs=10)
    ms.execute(directory, directory, protocol)
    assert len(calls) == 10
    summary = report(directory)
    assert summary["completed_runs"] == 10
    assert len(summary["figures"]) == 5
    assert len(pd.read_csv(directory / "MULTISEED_RESULTS.csv")) == 10
    receipt = next(directory.glob("*/seed_*/COMPLETE.json"))
    data = json.loads(receipt.read_text())
    checkpoint = receipt.parent / data["run_relative"] / "best.pth"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(CheckpointIntegrityError):
        report(directory)


def test_interrupted_slot_and_protocol_changes_block(mocked_execution):
    directory, protocol, calls = mocked_execution
    slot = directory / "baseline/seed_42"
    slot.mkdir(parents=True)
    atomic_write_json(slot / "STARTED.json", {})
    with pytest.raises(RuntimeError, match="incompleto"):
        ms.execute(directory, directory, protocol)
    assert not calls
    with pytest.raises(ValueError, match="1 y 10"):
        ms.execute(directory, directory, protocol, max_new_runs=11)
    changed = dict(protocol, num_workers=0)
    with pytest.raises(ValueError, match="protocolo"):
        ms.freeze_protocol(directory, changed)


def test_pending_report_does_not_fabricate_metrics(tmp_path):
    ms.freeze_protocol(tmp_path, ms.canonical_protocol())
    result = report(tmp_path)
    assert result["completed_runs"] == 0
    assert not result["pairs"]
    assert all(r["macro_f1"] is None for r in result["results"])
    assert result["test_used"] is False


def test_exclusive_writer(tmp_path):
    with ms.exclusive_experiment(tmp_path):
        with pytest.raises(RuntimeError, match="escritor"):
            with ms.exclusive_experiment(tmp_path):
                pass
