"""Persistencia y cierre del Optuna existente; un solo escritor por estudio.

El objetivo de entrenamiento permanece en tuning.TuningObjective. Este módulo
orquesta el presupuesto, la recuperación y la evaluación posterior a selección.
"""

from __future__ import annotations

import base64
import fcntl
import json
import pickle
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CornDataset
from src.data.preparation import atomic_write_json, sha256_file, sha256_json
from src.training.artifacts import write_extended_metrics, write_predictions_csv, write_test_outputs
from src.training.loop import run_epoch
from src.training.losses import build_criterion
from src.training.runs import load_validated_run
from src.training.tuning import (
    SplitIntegrityError,
    assert_split_snapshot_unchanged,
    fail_stale_running_trials,
    terminal_trial_count,
)

FORMAL_STUDY = "efficientnet_lite0_seed42_hpo_v1"
BASELINE_F1 = 0.9560862657056215
PRUNER_CONFIG = {"n_startup_trials": 5, "n_warmup_steps": 8, "interval_steps": 1}
EXPECTED_SPLIT_HASHES = {
    "master_manifest.csv": "64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163",
    "train.csv": "231048178f3450bf84925f8d19eb5a672669ee9f2658a23fe87d88d6e9949434",
    "val.csv": "6f37710ebd797e470bc918fec6bf9502f0635e8220542336e88fe5c13b5ae6a5",
    "test.csv": "08c81aeec5a57e04416edf2c94f997c422bfcada357c6a3434e73aa9f771f724",
    "manifest.lock.json": "0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8",
}


@contextmanager
def exclusive_study(directory: Path):
    """Bloqueo de proceso Linux; Modal además limita esta función a un contenedor."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Ya hay un escritor activo para este study.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def persist_sampler(study: optuna.Study, trial=None) -> None:
    """Guarda TPE en el mismo SQLite. Solo cargar estudios propios, nunca DB ajenos.

    Optuna no persiste el RNG del sampler en RDB; conservarlo evita reiniciar su
    secuencia al reabrir un study. Se guarda antes de entrenar y después de tell.
    """
    study.set_user_attr(
        "sampler_state",
        {
            "optuna_version": optuna.__version__,
            "last_trial": trial.number if trial is not None else -1,
            "pickle_base64": base64.b64encode(pickle.dumps(study.sampler)).decode("ascii"),
        },
    )


def open_study(directory: Path, name: str, protocol: dict, pruner) -> optuna.Study:
    """Reabre sin cambiar protocolo ni RNG. El llamador debe poseer exclusive_study."""
    directory.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{directory / 'study.db'}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=pruner,
    )
    previous = study.user_attrs.get("protocol")
    if previous is not None and previous != protocol:
        raise RuntimeError("El protocolo difiere del study persistido. No se permite alterarlo.")
    if previous is None:
        if study.trials:
            raise RuntimeError(
                "Study histórico sin protocolo; no se puede adoptar como HPO formal."
            )
        study.set_user_attr("protocol", protocol)
    saved = study.user_attrs.get("sampler_state")
    if saved:
        if saved["optuna_version"] != optuna.__version__:
            raise RuntimeError("La versión de Optuna cambió; no es seguro restaurar el sampler.")
        study.sampler = pickle.loads(base64.b64decode(saved["pickle_base64"]))
    elif study.trials:
        raise RuntimeError("Falta el estado del sampler; se detiene para evitar reiniciar TPE.")
    else:
        persist_sampler(study)
    return study


def run_to_budget(
    study,
    objective,
    target: int,
    directory: Path,
    *,
    callback=None,
    max_new_trials: int | None = None,
    timeout: float | None = None,
):
    """Presupuesto TOTAL, incluidos FAIL; nunca genera el intento target+1.

    Los cortes duros se recuperan como FAIL, sin reutilizar identidad. Si un trial
    falla antes de muestrear, conserva el RNG previo. Los fallos de integridad
    abortan toda la ejecución. El timeout se comprueba ENTRE trials.
    """
    if target < 1 or (max_new_trials is not None and max_new_trials < 1):
        raise ValueError("Los presupuestos deben ser positivos.")
    if (directory / "HPO_SELECTION_LOCK.json").exists():
        if terminal_trial_count(study) != target:
            raise RuntimeError("No se permite optimizar después de bloquear selección.")
        return
    if len(study.trials) > target:
        raise RuntimeError("El study ya excede el presupuesto solicitado.")
    recovered = fail_stale_running_trials(study)
    for number in recovered:
        trial_dir = directory / "trials" / f"trial_{number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            trial_dir / "interruption.json",
            {
                "trial_number": number,
                "state": "FAIL",
                "error": "Proceso interrumpido",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
    if any(t.state == optuna.trial.TrialState.WAITING for t in study.trials):
        raise RuntimeError("No se admiten trials WAITING externos al protocolo.")
    started, new_trials = monotonic(), 0
    while terminal_trial_count(study) < target:
        if max_new_trials is not None and new_trials >= max_new_trials:
            break
        if timeout is not None and monotonic() - started >= timeout:
            break
        trial = study.ask()
        persist_sampler(study, trial)
        fatal = None
        try:
            value = float(objective(trial))
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError("Objective no finito")
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as error:
            trial.set_user_attr("error", "".join(traceback.format_exception(error))[-12000:])
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            # Errores de datos/inicialización requieren diagnóstico; no gastar
            # automáticamente los 60 intentos por la misma avería estructural.
            if isinstance(error, (SplitIntegrityError, FileNotFoundError)):
                fatal = error
        else:
            study.tell(trial, value)
        finally:
            persist_sampler(study, trial)
        new_trials += 1
        if callback:
            callback(study, study.trials[trial.number])
        if fatal is not None:
            raise fatal
        recent = study.trials[-3:]
        if len(recent) == 3 and all(t.state == optuna.trial.TrialState.FAIL for t in recent):
            raise RuntimeError("Tres fallos consecutivos; auditar los errores antes de reanudar.")


def evaluate_locked_winner(
    study_dir: Path, splits_dir: Path, config_path: Path, snapshot: dict, device, num_workers: int
) -> dict:
    """Única puerta al test: exige lock, hashes y ausencia de evaluación anterior."""
    lock_path = study_dir / "HPO_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())  # Sin lock nunca llega a CornDataset.
    assert_split_snapshot_unchanged(splits_dir, snapshot)
    if lock["split_hashes"] != snapshot["sha256"]:
        raise SplitIntegrityError("Selection lock usa otros splits.")
    checkpoint = study_dir / lock["winner_checkpoint"]
    if sha256_file(checkpoint) != lock["winner_checkpoint_sha256"]:
        raise RuntimeError("Checkpoint ganador alterado después de selección.")
    if sha256_file(checkpoint.parent / "summary.json") != lock["winner_summary_sha256"]:
        raise RuntimeError("Contrato ganador alterado después de selección.")
    if sha256_file(study_dir / "best_hyperparameters.json") != lock["best_hyperparameters_sha256"]:
        raise RuntimeError("Hiperparámetros alterados después de selección.")
    completed = study_dir / "FINAL_TEST_COMPLETE.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if result["selection_lock_sha256"] != sha256_file(lock_path):
            raise RuntimeError("El test existente corresponde a otra selección.")
        for name, digest in result["artifact_hashes"].items():
            if sha256_file(study_dir / "final_test" / name) != digest:
                raise RuntimeError(f"Artefacto de test alterado: {name}")
        return result
    started_path = study_dir / "FINAL_TEST_STARTED.json"
    if started_path.exists():
        raise RuntimeError(
            "Test iniciado pero no cerrado; auditar antes de permitir otra evaluación."
        )
    loaded = load_validated_run(
        checkpoint,
        expected_model="efficientnet_lite0",
        device=device,
        splits_dir=splits_dir,
        config_path=str(config_path),
    )
    params = loaded.summary["hyperparameters"]
    # La pérdida se construye exclusivamente con frecuencias TRAIN.
    train_labels = pd.read_csv(splits_dir / "train.csv", usecols=["label"])["label"].tolist()
    criterion = build_criterion(
        labels=train_labels,
        class_to_idx=loaded.class_to_idx,
        strategy=params["class_weights"],
        label_smoothing=params["label_smoothing"],
        device=device,
    )
    atomic_write_json(
        started_path,
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "selection_lock_sha256": sha256_file(lock_path),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    test = CornDataset(
        csv_path=str(splits_dir / "test.csv"),
        config_path=str(config_path),
        transform=loaded.factory.get_pipeline("test"),
        class_to_idx=loaded.class_to_idx,
    )
    loader = DataLoader(
        test,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics, labels, predictions, probs = run_epoch(
        loaded.model, loader, criterion, device, desc="HPO winner final test (once)"
    )
    out = study_dir / "final_test"
    out.mkdir(exist_ok=True)
    idx_to_class = {index: name for name, index in loaded.class_to_idx.items()}
    write_test_outputs(out, idx_to_class, labels, predictions)
    frame = write_predictions_csv(out, test, idx_to_class, labels, predictions, probs)
    write_extended_metrics(out, frame, loaded.class_to_idx)
    assert_split_snapshot_unchanged(splits_dir, snapshot)
    result = {
        "schema_version": 1,
        "study_name": lock["study_name"],
        "best_trial": lock["best_trial"],
        "metrics": metrics,
        "finished_at": datetime.now(UTC).isoformat(),
        "selection_lock_sha256": sha256_file(lock_path),
        "checkpoint_sha256": lock["winner_checkpoint_sha256"],
        "test_used_during_hpo": False,
        "evaluation_count": 1,
        "artifact_hashes": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()},
    }
    atomic_write_json(completed, result)
    return result


def source_snapshot(project_root: Path) -> dict:
    """Código efectivo, incluso cambios sin commit; no incluye resultados ni docs."""
    paths = list((project_root / "src").rglob("*.py")) + [
        project_root / "scripts/pipeline/tune.py",
        project_root / "scripts/modal/train.py",
        project_root / "scripts/modal/_common.py",
        project_root / "pyproject.toml",
    ]
    hashes = {p.relative_to(project_root).as_posix(): sha256_file(p) for p in sorted(paths)}
    return {"files": hashes, "sha256": sha256_json(hashes)}
