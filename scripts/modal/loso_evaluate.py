"""Run LOSO selection-lock and holdout inference after training has completed."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import modal

from scripts.modal._common import REPO_ANCHOR, dataset_vol, image, outputs_vol

app = modal.App("doctor-maiz-loso-evaluate", image=image)
BASE = Path("/outputs/loso/efficientnet_lite0_baseline")
SOURCES = ("maize-diseases", "multicrop-disease-maiz")
RUN_ID = re.compile(r"^[0-9]{8}_[0-9]{6}$")


@app.function(
    gpu="A10G",
    cpu=32,
    memory=32768,
    timeout=2 * 3600,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    max_containers=1,
    retries=0,
)
def evaluate_one(source: str, run_id: str) -> None:
    if source not in SOURCES or not RUN_ID.fullmatch(run_id):
        raise ValueError("Fuente o run_id no permitidos")
    dataset_vol.reload()
    outputs_vol.reload()
    root = BASE / source / "seed_42"
    run_dir = root / "runs" / "efficientnet_lite0" / run_id
    if not (run_dir / "summary.json").exists():
        raise FileNotFoundError("Entrenamiento aún no completado")
    command = [
        sys.executable, "-m", "scripts.experiments.loso_finalize",
        "--source", source, "--run-dir", str(run_dir),
        "--splits-dir", str(root / "splits"), "--workers", "16",
    ]
    try:
        subprocess.run(command, check=True, cwd=REPO_ANCHOR)
    finally:
        outputs_vol.commit()


@app.local_entrypoint()
def main(source: str, run_id: str, execute: bool = False) -> None:
    if source not in SOURCES or not RUN_ID.fullmatch(run_id):
        raise ValueError("Fuente o run_id no permitidos")
    if execute:
        evaluate_one.remote(source, run_id)
    else:
        print(f"Preflight: {source}/{run_id}; sin --execute no abre holdout")
