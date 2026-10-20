"""Evaluate six frozen checkpoints only after all six trainings are complete.

The remote function refuses to start if any planned seed is incomplete. This keeps
the six pre-registered training outcomes independent of final test/holdout results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

from scripts.modal._common import dataset_vol, image, outputs_vol

app = modal.App("doctor-maiz-source-stability-finalize", image=image)
REFERENCE_SUMMARY = Path("/outputs/main/efficientnet_lite0/20260921_204608/summary.json")
REFERENCE_SHA256 = "20bb0945574913ffa8c736fecbcf25d932e6b87ce3e7a7c59d98439558432346"
SCENARIOS = ("baseline", "maize-diseases", "multicrop-disease-maiz")
SEEDS = (123, 2026)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_existing_lock(
    run_dir: Path, splits: Path, scenario: str, seed: int, metadata: dict
) -> dict:
    """Accept only a lock bound to this immutable training outcome."""
    from scripts.experiments.source_stability_finalize import _read

    locked = _read(run_dir / "selection.lock.json")
    summary = _read(run_dir / "summary.json")
    expected_source = None if scenario == "baseline" else scenario
    if (
        locked.get("scenario") != scenario
        or locked.get("seed") != seed
        or locked.get("held_out_source") != expected_source
        or locked.get("run_id") != metadata["run_id"]
        or locked.get("checkpoint_sha256") != metadata["checkpoint_sha256"]
        or locked.get("config_sha256") != summary["config_sha256"]
        or locked.get("git_commit") != metadata["git_commit"]
        or locked.get("git_dirty") != metadata["git_dirty"]
        or locked.get("final_evaluation_count") != 0
        or locked.get("test_or_holdout_used_before_lock") is not False
    ):
        raise ValueError(f"Selection lock incompatible: {run_dir}")
    for path, expected in (
        (run_dir / "best.pth", locked["checkpoint_sha256"]),
        (run_dir / "summary.json", locked["summary_sha256"]),
        (splits / "manifest.lock.json", locked["split_lock_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Artefacto modificado tras selection lock: {path}")
    return locked


def final_status(run_dir: Path, scenario: str, seed: int) -> tuple[str, dict | None]:
    """Skip a finished evaluation; never repeat a started but incomplete pass."""
    from scripts.experiments.source_stability_finalize import _read

    prefix = "test" if scenario == "baseline" else "holdout"
    guard = run_dir / "final_evaluation.started.json"
    metrics_path = run_dir / f"{prefix}_metrics.json"
    outputs = [
        run_dir / f"{prefix}_{suffix}"
        for suffix in (
            "predictions.csv",
            "class_metrics.csv",
            "confusion_matrix.csv",
            "confusion_normalized.csv",
        )
    ]
    if (
        not guard.exists()
        and not metrics_path.exists()
        and not any(path.exists() for path in outputs)
    ):
        return "pending", None
    if (
        not guard.exists()
        or not metrics_path.exists()
        or not all(path.exists() for path in outputs)
    ):
        raise RuntimeError(f"Evaluación final iniciada pero incompleta; no repetir: {run_dir}")
    marker, metrics = _read(guard), _read(metrics_path)
    lock_sha = sha256(run_dir / "selection.lock.json")
    if (
        marker.get("scenario") != scenario
        or marker.get("seed") != seed
        or marker.get("evaluation_count") != 1
        or marker.get("selection_lock_sha256") != lock_sha
        or metrics.get("scenario") != scenario
        or metrics.get("seed") != seed
        or metrics.get("evaluation_count") != 1
        or metrics.get("selection_lock_sha256") != lock_sha
    ):
        raise ValueError(f"Evaluación final incompatible: {run_dir}")
    return "complete", metrics


@app.function(
    gpu="A10G",
    cpu=16,
    memory=32768,
    timeout=6 * 3600,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    max_containers=1,
    retries=0,
)
@modal.concurrent(max_inputs=1)
def finalize_batch() -> None:
    from scripts.experiments.source_stability_finalize import (
        evaluate_final,
        lock_selection,
        verify_validation_run,
    )
    from scripts.modal.source_stability_train import scenario_paths, verify_splits

    dataset_vol.reload()
    outputs_vol.reload()
    if sha256(REFERENCE_SUMMARY) != REFERENCE_SHA256:
        raise ValueError("Baseline histórico remoto diferente")
    reference = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))
    slots = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            splits, root = scenario_paths(scenario, seed)
            verify_splits(scenario, splits)
            metadata = json.loads((root / "experiment_metadata.json").read_text())
            if (
                metadata.get("status") != "validation_complete"
                or metadata.get("scenario") != scenario
                or metadata.get("seed") != seed
            ):
                raise ValueError(f"Entrenamiento incompleto: {scenario}/seed_{seed}")
            run_dir = root / "runs/efficientnet_lite0" / metadata["run_id"]
            summary = verify_validation_run(run_dir, splits, scenario, seed, reference)
            if metadata.get("checkpoint_sha256") != summary["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint no coincide con metadatos: {run_dir}")
            slots.append((scenario, seed, run_dir, splits, metadata))
    if len(slots) != 6:
        raise ValueError("Se requieren las seis runs nuevas completas antes del test")

    # Freeze all six selections before opening any test or holdout.
    for scenario, seed, run_dir, splits, metadata in slots:
        if (run_dir / "selection.lock.json").exists():
            verify_existing_lock(run_dir, splits, scenario, seed, metadata)
        else:
            if (run_dir / "final_evaluation.started.json").exists():
                raise ValueError(f"Evaluación sin selection lock: {run_dir}")
            lock_selection(run_dir, splits, scenario, seed, reference, metadata)
            outputs_vol.commit()
        print(f"Selección congelada: {scenario}/seed_{seed}", flush=True)

    for scenario, seed, run_dir, splits, _metadata in slots:
        status, recorded = final_status(run_dir, scenario, seed)
        if status == "complete":
            print(
                json.dumps(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "already_complete": True,
                        "macro_f1": recorded["macro_f1"],
                    }
                ),
                flush=True,
            )
            continue
        result = evaluate_final(run_dir, splits, scenario, seed, commit_guard=outputs_vol.commit)
        outputs_vol.commit()
        print(
            json.dumps(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "run_id": result["run_id"],
                    "macro_f1": result["macro_f1"],
                    "accuracy": result["accuracy"],
                }
            ),
            flush=True,
        )


@app.local_entrypoint()
def main(execute: bool = False) -> None:
    print("Finalizar 6 runs: lock de selección -> evaluación única de test/holdout")
    if execute:
        finalize_batch.remote()
    else:
        print("Sin --execute no se evalúa ningún split final")
