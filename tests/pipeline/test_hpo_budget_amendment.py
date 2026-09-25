"""La revisión 60 → 25 conserva trials/RNG y rechaza cambios de entrenamiento."""

import json
import sqlite3
import zipfile

import optuna
import pytest

from scripts.pipeline.amend_hpo_budget import amend, trial_fingerprint
from scripts.pipeline.tune import _parse_args
from src.data.preparation import atomic_write_json, sha256_file
from src.training.tuning_study import FORMAL_STUDY, open_study, source_snapshot


@pytest.fixture
def amendment_input(tmp_path):
    root = tmp_path / "project"
    files = {
        "src/training/tuning.py": "class TuningObjective:\n    pass\n",
        "scripts/pipeline/tune.py": "BUDGET = 60\n",
        "scripts/modal/train.py": "BUDGET = 60\n",
        "scripts/modal/_common.py": "# unchanged\n",
        "pyproject.toml": "# unchanged\n",
        "config/dataset.yaml": "seed: 42\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source = source_snapshot(root)
    protocol = {
        "study_name": FORMAL_STUDY, "n_trials": 60,
        "config_sha256": sha256_file(root / "config/dataset.yaml"),
        "source_sha256": source["sha256"],
    }
    directory = tmp_path / "original"
    study = open_study(directory, FORMAL_STUDY, protocol, optuna.pruners.NopPruner())
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
    study.ask()  # Intento interrumpido, conservado hasta recuperación normal.
    atomic_write_json(directory / "preflight.json", {"protocol": protocol, "source": source})
    with zipfile.ZipFile(directory / "source_code.zip", "w") as bundle:
        for name in files:
            bundle.write(root / name, name)
    atomic_write_json(directory / "source_archive.json", {
        "archive_sha256": sha256_file(directory / "source_code.zip"),
    })
    atomic_write_json(directory / "study_summary.json", {"n_trials_requested": 60})
    (directory / "trials.csv").write_text("trial_number,state\n0,COMPLETE\n1,COMPLETE\n")
    (root / "scripts/pipeline/tune.py").write_text("BUDGET = 25\n")
    return directory, tmp_path / "amended", root


def test_amendment_preserves_database_trials_sampler_and_original(amendment_input):
    original, destination, root = amendment_input
    original_hash = sha256_file(original / "study.db")
    record = amend(original, destination, root, "Usuario autoriza reducir costo: 25 intentos.")
    assert record["new_budget"] == 25
    assert record["trials_at_amendment"] == [(0, "COMPLETE"), (1, "COMPLETE"), (2, "RUNNING")]
    assert sha256_file(original / "study.db") == original_hash
    assert sha256_file(destination / "budget_revisions/60-to-25/study.db") == original_hash
    with sqlite3.connect(destination / "study.db") as connection:
        assert trial_fingerprint(connection) == record["trial_tables_sha256"]
        protocol = json.loads(connection.execute(
            "SELECT value_json FROM study_user_attributes WHERE key='protocol'"
        ).fetchone()[0])
    assert protocol["n_trials"] == 25
    reopened = open_study(destination, FORMAL_STUDY, protocol, optuna.pruners.NopPruner())
    assert len(reopened.trials) == 3
    with pytest.raises(ValueError, match="destino"):
        amend(original, destination, root, "Duplicate")


@pytest.mark.parametrize("change", ["objective", "config", "other_file", "test"])
def test_amendment_rejects_unrelated_changes(amendment_input, change):
    original, destination, root = amendment_input
    if change == "objective":
        (root / "src/training/tuning.py").write_text("class TuningObjective:\n    changed = True\n")
    elif change == "config":
        (root / "config/dataset.yaml").write_text("seed: 43\n")
    elif change == "other_file":
        (root / "scripts/modal/_common.py").write_text("# new environment\n")
    else:
        atomic_write_json(original / "FINAL_TEST_STARTED.json", {})
    with pytest.raises(ValueError):
        amend(original, destination, root, "budget only")
    assert not destination.exists()


def test_formal_budget_defaults_to_25_without_changing_epochs():
    args = _parse_args(["--formal-hpo"])
    assert args.n_trials == 25
    assert args.epochs == 60
    with pytest.raises(SystemExit):
        _parse_args(["--formal-hpo", "--n-trials", "60"])
