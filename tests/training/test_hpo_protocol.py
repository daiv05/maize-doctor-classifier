"""Casos de no leakage, recuperación, presupuesto total y checkpoint congelado."""

import json
from pathlib import Path
from unittest.mock import Mock

import optuna
import pandas as pd
import pytest
import torch

from scripts.pipeline.tune import _build_pruner, _parse_args
from src.config import PROJECT_ROOT
from src.data.identity import IdentifiedValues
from src.data.preparation import atomic_write_json, sha256_file
from src.training.hyperparameters import load_best_params
from src.training.tuning import (
    HyperparameterSpace,
    SplitIntegrityError,
    TuningObjective,
    export_tuning_artifacts,
    split_hash_snapshot,
    write_selection_lock,
)
from src.training.tuning_study import (
    evaluate_locked_winner,
    exclusive_study,
    open_study,
    persist_sampler,
    run_to_budget,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr_min": 0},
        {"lr_max": float("inf")},
        {"weight_decay_min": 0},
        {"weight_decay_max": -1},
        {"batch_sizes": [True]},
        {"label_smoothing_values": [float("nan")]},
        {"label_smoothing_values": []},
        {"warmup_epochs_min": 6},
        {"class_weights_options": ["unknown"]},
    ],
)
def test_invalid_space_rejected(kwargs):
    with pytest.raises(ValueError):
        HyperparameterSpace(**kwargs).validate()


@pytest.mark.parametrize(
    "extra",
    [
        ["--epochs", "30"],
        ["--n-trials", "61"],
        ["--models", "efficientnet_b0"],
        ["--search-clahe"],
        ["--no-pretrained"],
        ["--pruner", "none"],
        ["--patience", "6"],
    ],
)
def test_formal_profile_rejects_protocol_changes(extra):
    with pytest.raises(SystemExit):
        _parse_args(["--formal-hpo", *extra])


def sampler_objective(study):
    objective = TuningObjective.__new__(TuningObjective)
    objective.space = HyperparameterSpace()

    def objective_fn(trial):
        params = objective.sample_parameters(trial)
        persist_sampler(study, trial)
        return params["learning_rate"]

    return objective_fn


def test_resume_preserves_tpe_sequence_and_total_budget(tmp_path):
    """Incluye la fase TPE después de startup, no solo muestreo aleatorio inicial."""

    def start(path):
        return open_study(path, "resume", {"seed": 42}, _build_pruner("none"))

    whole = start(tmp_path / "whole")
    run_to_budget(whole, sampler_objective(whole), 14, tmp_path / "whole")
    split = start(tmp_path / "split")
    run_to_budget(split, sampler_objective(split), 14, tmp_path / "split", max_new_trials=11)
    split._storage.remove_session()
    resumed = start(tmp_path / "split")
    run_to_budget(resumed, sampler_objective(resumed), 14, tmp_path / "split")
    assert [t.params for t in whole.trials] == [t.params for t in resumed.trials]
    run_to_budget(resumed, Mock(side_effect=AssertionError("extra trial")), 14, tmp_path)
    assert len(resumed.trials) == 14
    with pytest.raises(RuntimeError, match="protocolo"):
        open_study(tmp_path / "split", "resume", {"seed": 43}, _build_pruner("none"))


def test_interruption_and_failures_are_counted_and_ids_not_reused(tmp_path):
    study = open_study(tmp_path, "fail", {}, _build_pruner("none"))
    interrupted = study.ask()
    persist_sampler(study, interrupted)
    resumed = open_study(tmp_path, "fail", {}, _build_pruner("none"))

    def fn(trial):
        if trial.number == 1:
            raise RuntimeError("CUDA OOM")
        if trial.number == 2:
            trial.report(0.4, 1)
            raise optuna.TrialPruned()
        return 0.7

    run_to_budget(resumed, fn, 4, tmp_path)
    assert [t.number for t in resumed.trials] == [0, 1, 2, 3]
    assert [t.state.name for t in resumed.trials] == ["FAIL", "FAIL", "PRUNED", "COMPLETE"]
    assert "CUDA OOM" in resumed.trials[1].user_attrs["error"]
    assert (tmp_path / "trials/trial_000/interruption.json").exists()


def test_integrity_failure_aborts_entire_study(tmp_path):
    study = open_study(tmp_path, "integrity", {}, _build_pruner("none"))
    with pytest.raises(SplitIntegrityError):
        run_to_budget(study, Mock(side_effect=SplitIntegrityError("modified split")), 60, tmp_path)
    assert len(study.trials) == 1


def test_concurrent_writer_rejected(tmp_path):
    with exclusive_study(tmp_path):
        with pytest.raises(RuntimeError, match="escritor"):
            with exclusive_study(tmp_path):
                pass


@pytest.fixture
def trial_setup(tmp_path, tmp_splits_dir, fake_image_root, monkeypatch):
    import src.config as config
    import src.training.tuning as tuning

    monkeypatch.setattr(config, "DATASET_ROOT", fake_image_root)
    # Materialización sintética contractual; los tests de splits cubren su generación.
    frame = pd.read_csv(tmp_splits_dir / "train.csv")
    frame.to_csv(tmp_splits_dir / "master_manifest.csv", index=False)
    atomic_write_json(
        tmp_splits_dir / "manifest.lock.json",
        {
            "seed": 42,
            **{
                f"{name}_sha256": sha256_file(tmp_splits_dir / f"{name}.csv")
                for name in ("master_manifest", "train", "val", "test")
            },
        },
    )
    model = torch.nn.Linear(3, 9)
    monkeypatch.setattr(tuning, "build_model", lambda *a, **kw: model)
    calls = []

    def epoch(model, loader, criterion, device, optimizer=None, **kwargs):
        calls.append(
            (
                Path(loader.dataset.csv_path).name
                if hasattr(loader.dataset, "csv_path")
                else "train"
                if optimizer is not None
                else "val",
                optimizer is not None,
            )
        )
        if optimizer is not None:
            model.weight.data.add_(1)
            return {"loss": 1.0, "accuracy": 0.8, "macro_f1": 0.8}, [], [], []
        count = sum(not is_train for _, is_train in calls)
        values = [0.6, 0.9, 0.7]
        ds = loader.dataset
        labels = IdentifiedValues([ds.class_to_idx[v] for v in ds.data_frame["label"]])
        labels.sample_ids = ds.data_frame["sample_id"].tolist()
        predictions = IdentifiedValues(list(labels))
        predictions.sample_ids = labels.sample_ids
        return (
            {"loss": 0.4, "accuracy": 0.9, "macro_f1": values[(count - 1) % 3]},
            labels,
            predictions,
            [0.8] * len(labels),
        )

    monkeypatch.setattr(tuning, "run_epoch", epoch)
    seed = Mock()
    monkeypatch.setattr(tuning, "set_global_seed", seed)
    directory = tmp_path / "study"
    study = open_study(directory, "efficientnet_lite0_seed42_hpo_v1", {}, _build_pruner("none"))
    objective = TuningObjective(
        "efficientnet_lite0",
        tmp_splits_dir,
        PROJECT_ROOT / "config/dataset.yaml",
        epochs=3,
        device=torch.device("cpu"),
        study_dir=directory,
        num_workers=0,
        expected_split_hashes=split_hash_snapshot(tmp_splits_dir)["sha256"],
    )
    return study, objective, directory, tmp_splits_dir, calls, seed, epoch


def test_objective_best_validation_not_last_and_test_never_constructed(trial_setup, monkeypatch):
    import src.training.tuning as tuning

    study, objective, directory, _, calls, seed, _ = trial_setup
    real_dataset = tuning.CornDataset
    dataset_calls = []

    def guard(**kwargs):
        dataset_calls.append(Path(kwargs["csv_path"]).name)
        assert not kwargs["csv_path"].endswith("test.csv")
        return real_dataset(**kwargs)

    monkeypatch.setattr(tuning, "CornDataset", guard)
    study.optimize(objective, n_trials=2)
    assert dataset_calls == ["train.csv", "val.csv"] * 2
    assert [t.value for t in study.trials] == [0.9, 0.9]
    assert seed.call_args_list == [((42,),), ((42,),)]
    for number in range(2):
        contract = json.loads((directory / f"trials/trial_{number:03d}/summary.json").read_text())
        assert contract["best_epoch"] == 2
        assert "test" not in contract["metrics"]
        assert contract["metrics"]["best_validation"]["macro_f1"] == 0.9
        assert study.trials[number].user_attrs["executed_epochs"] == 3
    assert len(calls) == 12


@pytest.mark.parametrize("budget", [25, 60])
def test_lock_requires_budget_and_final_test_only_once(trial_setup, monkeypatch, budget):
    import src.training.runs as runs
    import src.training.tuning_study as manager

    study, objective, directory, splits, _, _, epoch = trial_setup
    study.optimize(objective, n_trials=1)
    snapshot = split_hash_snapshot(splits)
    study.set_user_attr("protocol", {"n_trials": budget})
    with pytest.raises(RuntimeError, match=str(budget)):
        write_selection_lock(
            study=study, study_dir=directory, split_snapshot=snapshot, search_space=objective.space,
            requested_trials=budget,
        )
    first = study.trials[0]
    for _ in range(budget - 1):
        study.add_trial(
            optuna.trial.create_trial(
                value=0.1, params=first.params, distributions=first.distributions
            )
        )
    export_tuning_artifacts(
        study, directory, "efficientnet_lite0", 0.8, epochs=3, requested_trials=budget
    )
    lock = write_selection_lock(
        study=study, study_dir=directory, split_snapshot=snapshot, search_space=objective.space,
        requested_trials=budget,
    )
    assert load_best_params(directory / "best_hyperparameters.json")["epochs"] == 3
    assert lock["best_trial"] == 0
    assert (
        write_selection_lock(
            study=study, study_dir=directory, split_snapshot=snapshot, search_space=objective.space,
            requested_trials=budget,
        )
        == lock
    )
    monkeypatch.setattr(runs, "build_model", lambda *a, **kw: torch.nn.Linear(3, 9))
    evaluate = Mock(side_effect=epoch)
    monkeypatch.setattr(manager, "run_epoch", evaluate)
    args = (
        directory,
        splits,
        PROJECT_ROOT / "config/dataset.yaml",
        snapshot,
        torch.device("cpu"),
        0,
    )
    result = evaluate_locked_winner(*args)
    assert evaluate.call_count == 1
    assert evaluate_locked_winner(*args) == result
    assert evaluate.call_count == 1
    assert (directory / "final_test/test_by_source.csv").exists()
    assert (directory / "final_test/test_calibration.json").exists()
    run_to_budget(study, Mock(side_effect=AssertionError("post-test optimize")), budget, directory)
    with pytest.raises(RuntimeError, match="optimizar"):
        run_to_budget(study, Mock(), 61, directory)
    atomic_write_json(directory / "best_hyperparameters.json", {"learning_rate": 0.5})
    with pytest.raises(RuntimeError, match="alterados"):
        evaluate_locked_winner(*args)


def test_test_loader_requires_lock(tmp_path, monkeypatch):
    import src.training.tuning_study as manager

    dataset = Mock(side_effect=AssertionError("No test before lock"))
    monkeypatch.setattr(manager, "CornDataset", dataset)
    with pytest.raises(FileNotFoundError):
        evaluate_locked_winner(tmp_path, tmp_path, tmp_path, {}, torch.device("cpu"), 0)
    dataset.assert_not_called()


def test_export_empty_study(tmp_path):
    study = optuna.create_study()
    summary = export_tuning_artifacts(study, tmp_path, "efficientnet_lite0", 0.8)
    assert summary["n_recorded"] == 0
    assert len(pd.read_csv(tmp_path / "trials.csv")) == 0
