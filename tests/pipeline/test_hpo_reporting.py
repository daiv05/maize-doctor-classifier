"""El cierre documental exige evidencia íntegra y no vuelve a inferir sobre test."""

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import Mock

import optuna
import pytest
import torch

from scripts.pipeline import report_hpo, watch_hpo
from src.config import PROJECT_ROOT
from src.data.preparation import atomic_write_json, sha256_file
from src.training.tuning import export_tuning_artifacts, split_hash_snapshot, write_selection_lock
from src.training.tuning_study import evaluate_locked_winner
from tests.training.test_hpo_protocol import trial_setup  # noqa: F401


@pytest.fixture(params=[25, 60])
def completed_hpo(request, monkeypatch):
    import src.training.runs as runs
    import src.training.tuning_study as manager

    study, objective, directory, splits, _, _, epoch = request.getfixturevalue("trial_setup")
    budget = request.param
    study.optimize(objective, n_trials=1)
    first = study.trials[0]
    for _ in range(budget - 1):
        study.add_trial(
            optuna.trial.create_trial(
                value=0.1, params=first.params, distributions=first.distributions
            )
        )
    snapshot = split_hash_snapshot(splits)
    export_tuning_artifacts(
        study, directory, "efficientnet_lite0", 0.8, epochs=3, requested_trials=budget
    )
    write_selection_lock(
        study=study, study_dir=directory, split_snapshot=snapshot, search_space=objective.space,
        requested_trials=budget,
    )
    monkeypatch.setattr(runs, "build_model", lambda *a, **kw: torch.nn.Linear(3, 9))
    monkeypatch.setattr(manager, "run_epoch", epoch)
    evaluate_locked_winner(
        directory, splits, PROJECT_ROOT / "config/dataset.yaml", snapshot, torch.device("cpu"), 0
    )
    source = PROJECT_ROOT / "src/training/tuning.py"
    atomic_write_json(
        directory / "preflight.json",
        {
            "protocol": {"split_snapshot": snapshot, "n_trials": budget},
            "source": {"files": {"src/training/tuning.py": sha256_file(source)}},
            "environment": {"python": "synthetic"},
        },
    )
    atomic_write_json(directory / "split_hashes_after.json", snapshot)
    with zipfile.ZipFile(directory / "source_code.zip", "w") as bundle:
        bundle.write(source, "src/training/tuning.py")
    atomic_write_json(
        directory / "source_archive.json",
        {
            "archive_sha256": sha256_file(directory / "source_code.zip"),
        },
    )
    if budget == 25:
        previous = directory / "budget_revisions/60-to-25"
        previous.mkdir(parents=True)
        for name in ("study.db", "preflight.json", "source_code.zip", "source_archive.json",
                     "study_summary.json", "trials.csv"):
            shutil.copy2(directory / name, previous / name)
        atomic_write_json(directory / "BUDGET_AMENDMENT.json", {
            "old_budget": 60, "new_budget": 25, "reason": "synthetic authorized amendment",
            "new_protocol": {"split_snapshot": snapshot, "n_trials": budget},
            "test_used": False, "timestamp": "2000-01-01T00:00:00+00:00",
            "prior_evidence_sha256": {p.name: sha256_file(p) for p in previous.iterdir()},
        })
    return directory


def test_report_updates_docs_only_after_verified_completion(completed_hpo, tmp_path):
    report = report_hpo.build_report(completed_hpo)
    budget = json.loads((completed_hpo / "study_summary.json").read_text())["n_trials_requested"]
    assert f"trials_requested: {budget}" in report
    assert "test_used_during_hpo: NO" in report
    root = tmp_path / "project"
    docs = root / "docs/es"
    paths = [
        "experimentos/hpo.md",
        "tesis/PROJECT_EVOLUTION.md",
        "tesis/DECISION_LOG.md",
        "tesis/EVIDENCE_REGISTRY.md",
        "tesis/MILESTONES.md",
        "reproducibilidad/figuras.md",
    ]
    for name in paths:
        path = docs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Histórico\n\nPreservar esta tabla y su evidencia.\n")
    hpo_page = docs / "experimentos/hpo.md"
    hpo_page.write_text(
        hpo_page.read_text()
        + "\n**Estado al 2026-09-23: EN EJECUCIÓN; "
        "los 60 trials formales aún no se declaran completados.**\n"
    )
    report_hpo.update_docs(completed_hpo, root, report)
    first = {name: (docs / name).read_text() for name in paths}
    report_hpo.update_docs(completed_hpo, root, report)
    assert first == {name: (docs / name).read_text() for name in paths}
    assert all("Preservar esta tabla" in content for content in first.values())
    assert "COMPLETADO" in first["tesis/MILESTONES.md"]
    assert "EN EJECUCIÓN" not in first["experimentos/hpo.md"]
    evidence = docs / "reproducibilidad/evidencia/hpo_lite0_seed42"
    assert (evidence / "FINAL_TEST_STARTED.json").exists()
    lock = json.loads((completed_hpo / "HPO_SELECTION_LOCK.json").read_text())
    assert (evidence / lock["winner_checkpoint"]).with_name("summary.json").exists()
    assert (docs / "tesis/HPO_BASELINE_COMPARISON.md").exists()


def test_report_rejects_incomplete_budget(completed_hpo):
    path = completed_hpo / "study_summary.json"
    payload = json.loads(path.read_text())
    payload["n_recorded"] = payload["n_trials_requested"] - 1
    atomic_write_json(path, payload)
    with pytest.raises(ValueError, match=str(payload["n_trials_requested"])):
        report_hpo.build_report(completed_hpo)


def test_report_rejects_changed_checkpoint(completed_hpo):
    lock = json.loads((completed_hpo / "HPO_SELECTION_LOCK.json").read_text())
    with (completed_hpo / lock["winner_checkpoint"]).open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(ValueError, match="Hash"):
        report_hpo.build_report(completed_hpo)


def test_watcher_only_collects_after_final_marker(tmp_path, monkeypatch):
    fetch = Mock(side_effect=[None, {"n_recorded": 3}, {"evaluation_count": 1}])
    monkeypatch.setattr(watch_hpo, "fetch_json", fetch)
    monkeypatch.setattr(
        watch_hpo,
        "modal_call",
        Mock(
            return_value=subprocess.CompletedProcess(
                [], 0, json.dumps([{"app_id": "app", "stopped_at": None}]), ""
            )
        ),
    )
    collect = Mock(return_value=tmp_path / "receipt")
    monkeypatch.setattr(watch_hpo, "collect", collect)
    monkeypatch.setattr(watch_hpo.time, "sleep", Mock())
    watch_hpo.monitor("modal", "app", tmp_path, 300)
    collect.assert_called_once()
    status = json.loads(
        (tmp_path / "outputs/hpo-monitor" / watch_hpo.STUDY / "status.json").read_text()
    )
    assert status["state"] == "complete"


def test_watcher_reports_stopped_app_without_relaunch(tmp_path, monkeypatch):
    monkeypatch.setattr(watch_hpo, "fetch_json", Mock(return_value=None))
    monkeypatch.setattr(
        watch_hpo, "modal_call", Mock(return_value=subprocess.CompletedProcess([], 0, "[]", ""))
    )
    collect = Mock()
    monkeypatch.setattr(watch_hpo, "collect", collect)
    monkeypatch.setattr(watch_hpo.time, "sleep", Mock())
    watch_hpo.monitor("modal", "app", tmp_path, 300)
    collect.assert_not_called()
    status = json.loads(
        (tmp_path / "outputs/hpo-monitor" / watch_hpo.STUDY / "status.json").read_text()
    )
    assert status["state"] == "needs_audit"


def test_collect_uses_actual_modal_directory_layout(tmp_path, monkeypatch):
    def download(executable, *args, **kwargs):
        directory = Path(args[-1]) / watch_hpo.STUDY
        directory.mkdir()
        (directory / "study.db").touch()
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(watch_hpo, "modal_call", download)
    report = Mock(return_value=subprocess.CompletedProcess([], 0, "report", ""))
    monkeypatch.setattr(watch_hpo.subprocess, "run", report)
    receipt = watch_hpo.collect("modal", tmp_path, tmp_path / "receipts")
    assert receipt.name == watch_hpo.STUDY
    assert report.call_args.args[0][3] == str(receipt)
