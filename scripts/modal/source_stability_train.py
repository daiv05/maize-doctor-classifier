"""Six pre-registered validation-only scenario/seed slots on frozen splits.

The final test and source holdouts are evaluated separately, after selection locks.
Seed 42 is never launched; a preempted slot may make one bounded retry.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

from scripts.modal._common import (
    REPO_ANCHOR,
    dataset_vol,
    image,
    outputs_vol,
    run_with_periodic_commit,
)

app = modal.App("doctor-maiz-source-stability", image=image)
MODEL = "efficientnet_lite0"
SEEDS = (123, 2026)
SCENARIOS = ("baseline", "maize-diseases", "multicrop-disease-maiz")
LOCK_SHA256 = {
    "baseline": "0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8",
    "maize-diseases": "6486473932ae2937ff5a97881953d6ffbfac3725aa81170ef2b20a6bb49cbc07",
    "multicrop-disease-maiz": "d8aac805a3d768166a63be664f1a658478926ab7d44a48e617c950f795e1537b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scenario_paths(scenario: str, seed: int) -> tuple[Path, Path]:
    if scenario not in SCENARIOS or seed not in SEEDS:
        raise ValueError("Escenario o seed fuera del protocolo pre-registrado")
    if scenario == "baseline":
        return (
            Path("/outputs/splits/seed_42"),
            Path(f"/outputs/multiseed/source_stability/baseline/seed_{seed}"),
        )
    base = Path("/outputs/loso/efficientnet_lite0_baseline") / scenario
    return base / "seed_42/splits", base / f"seed_{seed}"


def verify_splits(scenario: str, splits: Path) -> dict:
    """Check byte-identical manifests; the LOSO holdout is hashed, never opened."""
    lock_path = splits / "manifest.lock.json"
    if sha256_file(lock_path) != LOCK_SHA256[scenario]:
        raise ValueError(f"Manifest lock distinto: {scenario}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if scenario == "baseline":
        expected = {
            "master_manifest.csv": lock["master_manifest_sha256"],
            "train.csv": lock["train_sha256"],
            "val.csv": lock["val_sha256"],
            "test.csv": lock["test_sha256"],
        }
    else:
        if lock["protocol"] != "loso_baseline_seed42_v1" or lock["held_out_source"] != scenario:
            raise ValueError(f"Protocolo LOSO distinto: {scenario}")
        expected = lock["derived_sha256"]
        frozen = Path("/outputs/splits/seed_42")
        verify_splits("baseline", frozen)
        for name, digest in lock["frozen_sha256"].items():
            if sha256_file(frozen / name) != digest:
                raise ValueError(f"Input congelado distinto: {name}")
    for name, digest in expected.items():
        if sha256_file(splits / name) != digest:
            raise ValueError(f"CSV distinto: {scenario}/{name}")
    return lock


def command(scenario: str, seed: int, splits: Path, root: Path) -> list[str]:
    """Reuse the exact main baseline training CLI; only --seed differs."""
    return [
        sys.executable,
        "-m",
        "scripts.pipeline.train",
        "--models",
        MODEL,
        "--splits-dir",
        str(splits),
        "--output-dir",
        str(root / "runs"),
        "--skip-test",
        "--seed",
        str(seed),
        "--epochs",
        "60",
        "--batch-size",
        "32",
        "--learning-rate",
        "0.0001",
        "--weight-decay",
        "0.0001",
        "--scheduler",
        "cosine",
        "--warmup-epochs",
        "3",
        "--min-lr",
        "0.000001",
        "--patience",
        "8",
        "--class-weights",
        "sqrt_inverse",
        "--label-smoothing",
        "0.1",
        "--clip-grad-norm",
        "1.0",
        "--num-workers",
        "32",
        "--max-per-class",
        "0",
        "--no-clahe",
    ]


def verify_run(run_dir: Path, scenario: str, seed: int, reference: dict) -> dict:
    """Reject incomplete runs and configuration drift before accepting a slot."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for field in ("architecture", "hyperparameters", "preprocessing", "class_to_idx"):
        if summary[field] != reference[field]:
            raise ValueError(f"Configuración distinta en {scenario}/seed_{seed}: {field}")
    if (
        summary["seed"] != seed
        or summary["model"] != MODEL
        or summary["split_manifest_sha256"] != LOCK_SHA256[scenario]
        or summary.get("test_used") is not False
        or summary.get("evaluation_mode") != "validation_only"
        or "test" in summary.get("metrics", {})
    ):
        raise ValueError(f"Run no es validation-only bajo el split congelado: {run_dir}")
    if sha256_file(run_dir / "best.pth") != summary["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint distinto: {run_dir}")
    for name in ("train_history.csv", "validation_predictions.csv", "validation_diagnostics.json"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(run_dir / name)
    return summary


def inspect_slot(root: Path, scenario: str, seed: int, reference: dict) -> tuple[bool, int]:
    """Verify completed slots and preserve one interrupted attempt for retry."""
    metadata_path = root / "experiment_metadata.json"
    runs = root / "runs" / MODEL
    run_dirs = [path for path in runs.iterdir() if path.is_dir()] if runs.exists() else []
    if not metadata_path.exists():
        if runs.exists() and any(runs.iterdir()):
            raise FileExistsError(f"Run sin metadatos; revisión manual: {runs}")
        return False, 1

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    splits, _ = scenario_paths(scenario, seed)
    if (
        metadata.get("scenario") != scenario
        or metadata.get("seed") != seed
        or metadata.get("split_lock_sha256") != LOCK_SHA256[scenario]
        or metadata.get("command") != command(scenario, seed, splits, root)
    ):
        raise ValueError(f"Metadatos incompatibles; revisión manual: {metadata_path}")
    if len(run_dirs) > 1:
        raise ValueError(f"Más de un run en el mismo slot: {runs}")

    if run_dirs and (run_dirs[0] / "summary.json").exists():
        summary = verify_run(run_dirs[0], scenario, seed, reference)
        if metadata.get("status") == "validation_complete":
            if (
                metadata.get("run_id") != summary["run_id"]
                or metadata.get("checkpoint_sha256") != summary["checkpoint_sha256"]
            ):
                raise ValueError(f"Metadatos y checkpoint no coinciden: {root}")
        else:
            # Modal may preempt after training but before the metadata commit.
            metadata.update(
                {
                    "status": "validation_complete",
                    "run_id": summary["run_id"],
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_json(metadata_path, metadata)
        return True, int(metadata.get("attempt_count", 1))

    if metadata.get("status") == "validation_complete":
        raise ValueError(f"Slot marcado completo sin run verificable: {root}")
    if runs.exists() and any(not path.is_dir() for path in runs.iterdir()):
        raise ValueError(f"Artefactos inesperados en runs: {runs}")

    attempt = int(metadata.get("attempt_count", 1))
    if run_dirs:
        if attempt >= 2:
            raise RuntimeError(f"Dos intentos incompletos; detener para revisión: {root}")
        archive = root / "interrupted_attempts" / f"attempt_{attempt}"
        if archive.exists():
            raise FileExistsError(f"Archivo de intento ya existente: {archive}")
        archive.mkdir(parents=True)
        runs.rename(archive / MODEL)
        write_json(archive / "experiment_metadata.json", metadata)
        return False, attempt + 1

    # A restart can happen after archiving but before starting the retry.
    archive = root / "interrupted_attempts" / f"attempt_{attempt}"
    return False, attempt + 1 if archive.exists() else attempt


@app.function(
    gpu="A10G",
    cpu=32,
    memory=32768,
    timeout=24 * 3600,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    max_containers=1,
    retries=0,
)
@modal.concurrent(max_inputs=1)
def run_batch(reference: dict, max_new_runs: int = 6) -> None:
    if not 1 <= max_new_runs <= 6:
        raise ValueError("Presupuesto de entrenamiento fuera de 1..6")
    dataset_vol.reload()
    outputs_vol.reload()
    if not (Path("/data") / "clean").is_dir():
        raise FileNotFoundError("Falta /data/clean")
    from src.training.multiseed import environment

    planned = [(scenario, seed) for scenario in SCENARIOS for seed in SEEDS]
    for scenario in SCENARIOS:
        splits, _ = scenario_paths(scenario, 123)
        verify_splits(scenario, splits)
    for scenario, seed in planned[:max_new_runs]:
        splits, root = scenario_paths(scenario, seed)
        verify_splits(scenario, splits)
        complete, attempt = inspect_slot(root, scenario, seed, reference)
        if complete:
            print(f"Run verificado; se omite {scenario}/seed_{seed}", flush=True)
            outputs_vol.commit()
            continue
        runs = root / "runs" / MODEL
        root.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        metadata = {
            "attempt_count": attempt,
            "scenario": scenario,
            "seed": seed,
            "status": "training",
            "started_at_utc": started.isoformat(),
            "environment": environment(),
            "git_commit": reference["git_commit"],
            "git_dirty": reference["git_dirty"],
            "split_lock_sha256": LOCK_SHA256[scenario],
            "command": command(scenario, seed, splits, root),
        }
        write_json(root / "experiment_metadata.json", metadata)
        outputs_vol.commit()
        try:
            run_with_periodic_commit(metadata["command"], REPO_ANCHOR, outputs_vol)
            run_dirs = [path for path in runs.iterdir() if path.is_dir()]
            if len(run_dirs) != 1:
                raise ValueError(f"Se esperaba exactamente un run: {runs}")
            summary = verify_run(run_dirs[0], scenario, seed, reference)
            metadata.update(
                {
                    "status": "validation_complete",
                    "run_id": summary["run_id"],
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                }
            )
            write_json(root / "experiment_metadata.json", metadata)
            outputs_vol.commit()
        except Exception:
            metadata["status"] = "failed_or_incomplete"
            metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(root / "experiment_metadata.json", metadata)
            outputs_vol.commit()
            raise


@app.local_entrypoint()
def main(execute: bool = False, max_new_runs: int = 6) -> None:
    if not 1 <= max_new_runs <= 6:
        raise ValueError("Presupuesto de entrenamiento fuera de 1..6")
    project_root = Path(__file__).resolve().parents[2]

    path = (
        project_root
        / "outputs/archives/DoctorMaiz_efficientnet_lite0_20260921_204608"
        / "efficientnet_lite0/20260921_204608/summary.json"
    )
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference["config_sha256"] != (
        "53cc091e505057f6751a7b6151f403e03f830fac3c0966df9ba51e2df2814e8b"
    ):
        raise ValueError("Baseline histórico cambió")
    reference = {
        key: reference[key]
        for key in ("architecture", "hyperparameters", "preprocessing", "class_to_idx")
    }
    reference["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    reference["git_dirty"] = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project_root, text=True
        ).strip()
    )
    print(f"Plan: {[(s, n) for s in SCENARIOS for n in SEEDS][:max_new_runs]}")
    if execute:
        run_batch.remote(reference, max_new_runs)
    else:
        print("Preflight local: sin --execute no se lanza entrenamiento")
