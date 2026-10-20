"""Comparación pareada congelada. Orquesta train.py; nunca optimiza ni evalúa test."""

from __future__ import annotations

import fcntl
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.config import PROJECT_ROOT
from src.data.identity import validate_sample_ids
from src.data.preparation import (
    atomic_write_json,
    sha256_file,
    sha256_json,
    validate_split_integrity,
)
from src.data.transforms import CornTransformFactory
from src.training.hyperparameters import load_best_params, validate_effective_hyperparameters
from src.training.runs import validate_run_contract

EXPERIMENT = "efficientnet_lite0_baseline_vs_hpo"
MODEL = "efficientnet_lite0"
SEEDS = (42, 123, 2026, 3407, 7777)
CONFIGURATIONS = ("baseline", "hpo_trial_0")
EVIDENCE = Path("docs/es/reproducibilidad/evidencia")
CANONICAL_FILES = {
    "baseline": EVIDENCE / "hpo_baseline_summary.json",
    "parameters": EVIDENCE / "hpo_lite0_seed42/best_hyperparameters.json",
    "hpo": EVIDENCE / "hpo_lite0_seed42/trials/trial_000/summary.json",
    "splits": EVIDENCE / "hpo_lite0_seed42/split_hashes_after.json",
    "preflight": EVIDENCE / "hpo_lite0_seed42/preflight.json",
}
METRICS = ("macro_f1", "accuracy", "nitrogen_f1", "phosphorus_f1", "potassium_f1", "ece")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_files(root=PROJECT_ROOT):
    """Incluye código efectivo no committed; excluye datos, pesos y credenciales."""
    paths = [*root.glob("src/**/*.py"), *root.glob("scripts/**/*.py"), root / "pyproject.toml"]
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths)}


def canonical_protocol(root=PROJECT_ROOT):
    """Congela fuentes canónicas, no defaults recordados ni métricas de test."""
    sources = {key: root / path for key, path in CANONICAL_FILES.items()}
    baseline, hpo = read_json(sources["baseline"]), read_json(sources["hpo"])
    params = load_best_params(sources["parameters"])
    expected = {
        "batch_size": 16,
        "clahe": False,
        "class_weights": "none",
        "epochs": 60,
        "label_smoothing": 0.075,
        "learning_rate": 8.468008575248323e-05,
        "warmup_epochs": 1,
        "weight_decay": 0.005669849511478858,
    }
    if params != expected or any(hpo["hyperparameters"].get(k) != v for k, v in params.items()):
        raise ValueError("El artefacto HPO difiere de la configuración congelada del trial 0")
    if baseline["run_id"] != "20260921_204608" or hpo["trial_number"] != 0:
        raise ValueError("Identidad canónica incorrecta")
    if sha256_file(sources["baseline"]) != (
        "20bb0945574913ffa8c736fecbcf25d932e6b87ce3e7a7c59d98439558432346"
    ):
        raise ValueError("El artefacto baseline canónico cambió")
    hashes = read_json(sources["splits"])["sha256"]
    configs = {}
    for name, summary in (("baseline", baseline), ("hpo_trial_0", hpo)):
        hp = validate_effective_hyperparameters(summary["hyperparameters"])
        hp.pop("clahe", None)  # El contrato main registra CLAHE en preprocessing.
        if hp["optimizer"] != "AdamW" or hp["sampler"] is not None or hp["dropout"] is not None:
            raise ValueError("Configuración incompatible con el pipeline main")
        if (
            summary["model"] != MODEL
            or summary["split_manifest_sha256"] != hashes["manifest.lock.json"]
        ):
            raise ValueError("Modelo/split canónico incompatible")
        configs[name] = {"hyperparameters": hp, "preprocessing": summary["preprocessing"]}
    if baseline["class_to_idx"] != hpo["class_to_idx"]:
        raise ValueError("class_to_idx diferente")
    if baseline["preprocessing"] != hpo["preprocessing"]:
        raise ValueError("Preprocessing diferente")
    cfg_path = root / "config/dataset.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    if sha256_file(cfg_path) != read_json(sources["preflight"])["protocol"]["config_sha256"]:
        raise ValueError("dataset.yaml cambió respecto al HPO")
    mapping = {c: i for i, c in enumerate(cfg["dataset"]["classes"])}
    if mapping != baseline["class_to_idx"]:
        raise ValueError("Las clases actuales difieren del baseline")
    factory = CornTransformFactory(config_path=str(cfg_path))
    if factory.to_contract() != baseline["preprocessing"]:
        raise ValueError("Transforms actuales incompatibles")
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "model": MODEL,
        "seeds": list(SEEDS),
        "configurations": configs,
        "class_to_idx": mapping,
        "training_preprocessing": factory.training_contract(),
        "num_workers": 32,
        "max_per_class": 0,
        "test_used": False,
        "dataset_config": cfg,
        "dataset_config_sha256": sha256_file(cfg_path),
        "split_hashes": hashes,
        "dataset_fingerprint": hashes["master_manifest.csv"],
        "source_files": source_files(root),
        "canonical_files": {str(p.relative_to(root)): sha256_file(p) for p in sources.values()},
        "git_commit": git,
        "git_dirty": bool(dirty),
        "git_status": dirty.splitlines(),
        "previous_runs_reused": [],
        "reuse_decision": "No equivalencia verificable de entorno/workers y diagnósticos "
        "validation para ambas runs históricas; nueva cohorte homogénea de 10 runs.",
    }


def audit_splits(directory, expected):
    """Solo metadatos. test.csv se hashea en bytes; nunca se parsea ni carga imágenes."""
    directory = Path(directory)
    actual = {name: sha256_file(directory / name) for name in expected}
    if actual != expected:
        raise ValueError("Los hashes de splits/manifiestos cambiaron")
    master = pd.read_csv(directory / "master_manifest.csv")
    train = pd.read_csv(directory / "train.csv")
    val = pd.read_csv(directory / "val.csv")
    for frame in (master, train, val):
        validate_sample_ids(frame)
    indexed = master.set_index("sample_id")
    for frame in (train, val):
        source = indexed.loc[frame.sample_id]
        for column in ("image_path", "label"):
            if source[column].tolist() != frame[column].tolist():
                raise ValueError(f"Metadatos de {column} incompatibles con master")
    remaining = master[~master.sample_id.isin(set(train.sample_id) | set(val.sample_id))]
    # Cobertura train/val/resto contractual; no requiere abrir el CSV de test.
    integrity = validate_split_integrity(
        master, {"train": train, "val": val, "held_out": remaining}
    )
    return {
        "hashes": actual,
        "integrity": integrity,
        "counts": {"train": len(train), "val": len(val), "held_out": len(remaining)},
        "test_csv_parsed": False,
        "test_images_read": False,
        "near_duplicate_check": "not_executed",
    }


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "timm",
                "numpy",
                "pandas",
                "scikit-learn",
                "Pillow",
            )
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def training_command(protocol, name, seed, splits_dir, slot, root=PROJECT_ROOT):
    hp = protocol["configurations"][name]["hyperparameters"]
    command = [
        sys.executable,
        "-m",
        "scripts.pipeline.train",
        "--models",
        MODEL,
        "--skip-test",
        "--seed",
        str(seed),
        "--splits-dir",
        str(splits_dir),
        "--output-dir",
        str(slot),
        "--config",
        str(root / "config/dataset.yaml"),
        "--num-workers",
        str(protocol["num_workers"]),
        "--max-per-class",
        "0",
    ]
    for key in (
        "batch_size",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "epochs",
        "patience",
        "warmup_epochs",
        "min_lr",
        "class_weights",
        "label_smoothing",
        "clip_grad_norm",
    ):
        command += ["--" + key.replace("_", "-"), str(hp[key])]
    command += [
        "--clahe"
        if protocol["configurations"][name]["preprocessing"]["clahe"]["enabled"]
        else "--no-clahe"
    ]
    if not hp["pretrained"]:
        command += ["--no-pretrained"]
    return command


@contextmanager
def exclusive_experiment(directory):
    """Bloqueo local; Modal restringe adicionalmente a un contenedor/entrada."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Ya existe un escritor multi-seed") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def freeze_protocol(directory, protocol):
    path = directory / "PROTOCOL.json"
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError("El protocolo existente difiere; no mezclar cohortes")
    else:
        atomic_write_json(path, protocol)


def verify_sources(protocol, root=PROJECT_ROOT):
    if source_files(root) != protocol["source_files"]:
        raise ValueError("Cambió el código efectivo del protocolo")
    if sha256_file(root / "config/dataset.yaml") != protocol["dataset_config_sha256"]:
        raise ValueError("Cambió dataset.yaml")


def completed_result(directory, protocol, name, seed):
    """Reanudación idempotente solo con recibo íntegro; un intento incompleto no se repite."""
    slot = directory / name / f"seed_{seed}"
    receipt_path = slot / "COMPLETE.json"
    if not receipt_path.exists():
        if slot.exists() and any(slot.iterdir()):
            raise RuntimeError(f"Intento incompleto en {slot}; no se relanza automáticamente")
        return None
    receipt = read_json(receipt_path)
    if (
        receipt["protocol_sha256"] != sha256_json(protocol)
        or receipt["configuration"] != name
        or receipt["seed"] != seed
    ):
        raise ValueError("Recibo incompatible")
    run = slot / receipt["run_relative"]
    checked = validate_run_contract(
        run,
        expected_model=MODEL,
        expected_class_to_idx=protocol["class_to_idx"],
        expected_preprocessing=protocol["configurations"][name]["preprocessing"],
        expected_split_manifest_sha256=protocol["split_hashes"]["manifest.lock.json"],
    ).summary
    required = {
        "summary.json",
        "best.pth",
        "train_history.csv",
        "validation_predictions.csv",
        "validation_diagnostics.json",
        "multiseed_metadata.json",
    }
    if not required.issubset(receipt["artifact_hashes"]):
        raise ValueError("Recibo sin artefactos requeridos")
    for filename, digest in receipt["artifact_hashes"].items():
        if sha256_file(run / filename) != digest:
            raise ValueError(f"Artefacto alterado: {run / filename}")
    if (
        checked["hyperparameters"] != protocol["configurations"][name]["hyperparameters"]
        or checked["seed"] != seed
        or checked.get("test_used") is not False
        or "test" in checked["metrics"]
    ):
        raise ValueError("Contrato previo incompatible con el protocolo")
    diag = read_json(run / "validation_diagnostics.json")
    expected = {
        "macro_f1": checked["metrics"]["best_validation"]["macro_f1"],
        "accuracy": checked["metrics"]["best_validation"]["accuracy"],
        "ece": diag["calibration"]["ece"],
        **{
            f"{n}_f1": diag["per_class"][f"{n}_deficiency"]["f1-score"]
            for n in ("nitrogen", "phosphorus", "potassium")
        },
    }
    if any(receipt[key] != value or not math.isfinite(value) for key, value in expected.items()):
        raise ValueError("Métricas del recibo inconsistentes/no finitas")
    return receipt


def execute(directory, splits_dir, protocol, *, max_new_runs=10, after_run=None):
    """Máximo diez slots, sin retries. No hay restauración de optimizador en fit actual."""
    directory, splits_dir = Path(directory), Path(splits_dir)
    if not 1 <= max_new_runs <= 10:
        raise ValueError("max_new_runs debe estar entre 1 y 10")
    if protocol["seeds"] != list(SEEDS) or set(protocol["configurations"]) != set(CONFIGURATIONS):
        raise ValueError("Diseño experimental no autorizado")
    with exclusive_experiment(directory):
        verify_sources(protocol)
        audit = audit_splits(splits_dir, protocol["split_hashes"])
        freeze_protocol(directory, protocol)
        atomic_write_json(directory / "PREFLIGHT.json", audit)
        env = environment()
        if env["gpu"] is None:
            raise RuntimeError("La cohorte formal requiere GPU; use pruebas sintéticas para smoke")
        env_path = directory / "ENVIRONMENT.json"
        if env_path.exists() and read_json(env_path) != env:
            raise ValueError("Entorno diferente al de la cohorte; no continuar")
        atomic_write_json(env_path, env)
        pending = []
        for seed in SEEDS:
            for name in CONFIGURATIONS:
                if completed_result(directory, protocol, name, seed) is None:
                    pending.append((name, seed))
        for name, seed in pending[:max_new_runs]:
            verify_sources(protocol)
            audit_splits(splits_dir, protocol["split_hashes"])
            slot = directory / name / f"seed_{seed}"
            slot.mkdir(parents=True, exist_ok=False)
            command = training_command(protocol, name, seed, splits_dir, slot)
            metadata = {
                "experiment_id": EXPERIMENT,
                "configuration": name,
                "seed": seed,
                "timestamp": datetime.now(UTC).isoformat(),
                "environment": env,
                "protocol_sha256": sha256_json(protocol),
                "protocol": protocol,
                "command": command,
                "test_used": False,
            }
            atomic_write_json(slot / "STARTED.json", metadata)
            try:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                runs = list((slot / MODEL).glob("*/summary.json"))
                if len(runs) != 1:
                    raise ValueError("Se esperaba exactamente un contrato de run")
                run = runs[0].parent
                checked = validate_run_contract(
                    run,
                    expected_model=MODEL,
                    expected_class_to_idx=protocol["class_to_idx"],
                    expected_preprocessing=protocol["configurations"][name]["preprocessing"],
                    splits_dir=splits_dir,
                ).summary
                if (
                    checked["hyperparameters"]
                    != protocol["configurations"][name]["hyperparameters"]
                    or checked["seed"] != seed
                    or checked.get("test_used") is not False
                    or "test" in checked["metrics"]
                ):
                    raise ValueError("El contrato ejecutado no coincide con el protocolo")
                audit_splits(splits_dir, protocol["split_hashes"])
                atomic_write_json(run / "multiseed_metadata.json", metadata)
                diag = read_json(run / "validation_diagnostics.json")
                metrics = checked["metrics"]["best_validation"]
                if abs(metrics["macro_f1"] - checked["best_val_macro_f1"]) > 1e-12:
                    raise ValueError("La reevaluación validation no coincide con best.pth")
                receipt = {
                    "experiment_id": EXPERIMENT,
                    "configuration": name,
                    "seed": seed,
                    "protocol_sha256": sha256_json(protocol),
                    "status": "COMPLETE",
                    "run_relative": run.relative_to(slot).as_posix(),
                    "best_epoch": checked["best_epoch"],
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "ece": diag["calibration"]["ece"],
                    **{
                        f"{nutrient}_f1": diag["per_class"][f"{nutrient}_deficiency"]["f1-score"]
                        for nutrient in ("nitrogen", "phosphorus", "potassium")
                    },
                    "checkpoint_sha256": checked["checkpoint_sha256"],
                    "finished_at": datetime.now(UTC).isoformat(),
                    "test_used": False,
                    "artifact_hashes": {
                        p.name: sha256_file(p) for p in sorted(run.iterdir()) if p.is_file()
                    },
                }
                if any(not math.isfinite(receipt[key]) for key in METRICS):
                    raise ValueError("Métricas no finitas; run no válida")
                atomic_write_json(slot / "COMPLETE.json", receipt)
            except Exception as error:
                atomic_write_json(
                    slot / "FAILED.json",
                    {
                        "error": repr(error),
                        "seed": seed,
                        "configuration": name,
                        "automatic_retry": False,
                    },
                )
                raise
            finally:
                if after_run:
                    after_run()
        return pending[:max_new_runs]
