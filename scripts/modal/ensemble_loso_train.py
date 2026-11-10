"""Train only missing historical ensemble members on the frozen multicrop LOSO split."""

from __future__ import annotations

import json
import subprocess
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
from scripts.modal.source_stability_train import verify_splits
from src.data.preparation import sha256_file
from src.training.ensemble_loso_protocol import (
    HISTORICAL_SUMMARY_SHA256,
    LOCK_SHA256,
    MODELS,
    ROOT,
    SEEDS,
    SOURCE,
    SPLITS,
    command,
    inspect_slot,
    verify_run,
    write_json,
)

app = modal.App("doctor-maiz-ensemble-loso-train", image=image)

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
def train_batch(historical_summaries: dict, git_commit: str, git_dirty: bool) -> None:
    from src.training.multiseed import environment

    dataset_vol.reload()
    outputs_vol.reload()
    if not (Path("/data") / "clean").is_dir():
        raise FileNotFoundError("Falta dataset limpio en corn-clean")
    lock = verify_splits(SOURCE, SPLITS)
    if sha256_file(SPLITS / "manifest.lock.json") != LOCK_SHA256:
        raise ValueError("Lock LOSO distinto")
    import pandas as pd

    for name in ("train.csv", "val.csv"):
        if SOURCE in set(pd.read_csv(SPLITS / name, usecols=["source_id"])["source_id"]):
            raise ValueError(f"Fuente holdout encontrada en {name}")
    if lock["preflight"]["counts"] != {"train": 19328, "val": 4150, "holdout": 5816}:
        raise ValueError("Conteos LOSO distintos")
    for seed in SEEDS:
        for model in MODELS:
            root = ROOT / f"seed_{seed}" / model
            complete, attempt = inspect_slot(root, model, seed, historical_summaries[model])
            if complete:
                print(f"Reutilizado {model}/seed_{seed}", flush=True)
                outputs_vol.commit()
                continue
            root.mkdir(parents=True, exist_ok=True)
            started = datetime.now(timezone.utc)
            metadata = {
                "attempt_count": attempt,
                "model": model,
                "seed": seed,
                "status": "training",
                "started_at_utc": started.isoformat(),
                "environment": environment(),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "split_lock_sha256": LOCK_SHA256,
                "historical_summary_sha256": HISTORICAL_SUMMARY_SHA256[model],
                "command": command(model, seed, root),
            }
            write_json(root / "experiment_metadata.json", metadata)
            outputs_vol.commit()
            try:
                run_with_periodic_commit(metadata["command"], REPO_ANCHOR, outputs_vol)
                runs = root / "runs" / model
                run_dirs = [path for path in runs.iterdir() if path.is_dir()]
                if len(run_dirs) != 1:
                    raise ValueError(f"Se esperaba un único run: {runs}")
                summary = verify_run(run_dirs[0], model, seed, historical_summaries[model])
                finished = datetime.now(timezone.utc)
                metadata.update(
                    status="validation_complete",
                    run_id=summary["run_id"],
                    checkpoint_sha256=summary["checkpoint_sha256"],
                    finished_at_utc=finished.isoformat(),
                    duration_seconds=(finished - started).total_seconds(),
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
def main(execute: bool = False) -> None:
    project = Path(__file__).resolve().parents[2]
    summaries = {}
    for model, digest in HISTORICAL_SUMMARY_SHA256.items():
        path = project / "outputs/outputs-11092026/main" / model
        run_id = "20260910_170120" if model == "efficientnet_b0" else "20260910_184521"
        path = path / run_id / "summary.json"
        if sha256_file(path) != digest:
            raise ValueError(f"Summary histórico alterado: {path}")
        summaries[model] = json.loads(path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=project, text=True).strip()
    )
    print("Plan congelado: B0 + ShuffleNet × seeds 42/123/2026; seis slots A10G")
    if execute:
        train_batch.remote(summaries, commit, dirty)
    else:
        print("Preflight local únicamente; no se consume GPU")
