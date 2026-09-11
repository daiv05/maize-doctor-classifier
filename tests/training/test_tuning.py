import json
from pathlib import Path
import optuna
import pytest

from src.training.tuning import (
    HyperparameterSpace,
    TuningObjective,
    export_tuning_artifacts,
    save_optimization_plots,
)


def test_hyperparameter_space_defaults():
    space = HyperparameterSpace()
    assert space.lr_min == 1e-5
    assert space.lr_max == 1e-3
    assert 32 in space.batch_sizes
    assert "sqrt_inverse" in space.class_weights_options
    assert space.allow_clahe is True


def test_sample_parameters():
    space = HyperparameterSpace()
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    # Mocking objective sampling without loading real datasets
    class DummyObjective:
        def __init__(self, space):
            self.space = space

        def sample_parameters(self, trial):
            return TuningObjective.sample_parameters(self, trial)

    dummy = DummyObjective(space)
    params = dummy.sample_parameters(trial)

    assert "learning_rate" in params
    assert 1e-5 <= params["learning_rate"] <= 1e-3
    assert "weight_decay" in params
    assert params["batch_size"] in [16, 32, 64]
    assert params["class_weights"] in ["sqrt_inverse", "inverse", "none"]
    assert "label_smoothing" in params
    assert "warmup_epochs" in params
    assert "clahe" in params


def test_export_tuning_artifacts(tmp_path: Path):
    study = optuna.create_study(study_name="test_study", direction="maximize")

    def dummy_objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [16, 32])
        return 0.92 if bs == 32 else 0.88

    study.optimize(dummy_objective, n_trials=3)

    output_dir = tmp_path / "tuning_artifacts"
    summary = export_tuning_artifacts(
        study=study,
        output_dir=output_dir,
        model_name="efficientnet_b0",
        baseline_macro_f1=0.9146,
    )

    assert (output_dir / "best_params.json").exists()
    assert (output_dir / "trials.csv").exists()
    assert (output_dir / "comparison_vs_baseline.csv").exists()
    assert (output_dir / "optimization_history.png").exists()

    with open(output_dir / "best_params.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model_name"] == "efficientnet_b0"
    assert data["total_trials"] == 3
    assert data["best_val_macro_f1"] == study.best_value
    assert "best_params" in data
