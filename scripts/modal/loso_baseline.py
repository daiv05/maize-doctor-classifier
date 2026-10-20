"""Run one pre-registered LOSO baseline on Modal; never evaluates the holdout.

Upload the derived splits first. One invocation starts at most one training run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import modal

from scripts.modal._common import (
    REPO_ANCHOR,
    dataset_vol,
    image,
    outputs_vol,
    run_with_periodic_commit,
)

app = modal.App("doctor-maiz-loso-baseline", image=image)
BASE = Path("/outputs/loso/efficientnet_lite0_baseline")
SOURCES = ("maize-diseases", "multicrop-disease-maiz")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(
    gpu="A10G",
    cpu=32,
    memory=32768,
    timeout=12 * 3600,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    max_containers=1,
    retries=0,
)
@modal.concurrent(max_inputs=1)
def train_one(source: str, lock_sha256: str) -> None:
    """Verify the manifest lock and reuse the unchanged main training CLI."""
    if source not in SOURCES:
        raise ValueError(f"Fuente no predefinida: {source}")
    dataset_vol.reload()
    outputs_vol.reload()
    root = BASE / source / "seed_42"
    splits = root / "splits"
    lock_path = splits / "manifest.lock.json"
    if digest(lock_path) != lock_sha256:
        raise ValueError("El lock remoto no coincide con el lock preflight local")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["held_out_source"] != source or lock["protocol"] != "loso_baseline_seed42_v1":
        raise ValueError("Protocolo LOSO remoto diferente")
    for name in ("train.csv", "val.csv", "holdout.csv"):
        if digest(splits / name) != lock["derived_sha256"][name]:
            raise ValueError(f"Hash LOSO remoto diferente: {name}")
    frozen = Path("/outputs/splits/seed_42")
    for name in ("master_manifest.csv", "train.csv", "val.csv", "test.csv",
                 "manifest.lock.json"):
        if digest(frozen / name) != lock["frozen_sha256"][name]:
            raise ValueError(f"Hash seed_42 remoto diferente: {name}")
    if not (Path("/data") / "clean").exists():
        raise FileNotFoundError("No existe /data/clean")
    runs = root / "runs" / "efficientnet_lite0"
    if runs.exists() and any(runs.iterdir()):
        raise FileExistsError(f"Ya existe al menos un intento; no duplicar: {runs}")
    command = [
        sys.executable, "scripts/pipeline/train.py",
        "--models", "efficientnet_lite0", "--splits-dir", str(splits),
        "--output-dir", str(root / "runs"), "--skip-test", "--seed", "42",
        "--epochs", "60", "--batch-size", "32", "--learning-rate", "0.0001",
        "--weight-decay", "0.0001", "--scheduler", "cosine",
        "--warmup-epochs", "3", "--min-lr", "0.000001", "--patience", "8",
        "--class-weights", "sqrt_inverse", "--label-smoothing", "0.1",
        "--clip-grad-norm", "1.0", "--num-workers", "32",
    ]
    try:
        run_with_periodic_commit(command, cwd=REPO_ANCHOR, volume=outputs_vol)
    finally:
        outputs_vol.commit()


@app.local_entrypoint()
def main(source: str, lock_sha256: str, execute: bool = False) -> None:
    """Require an explicit source, lock SHA, and --execute to spend GPU time."""
    if source not in SOURCES or len(lock_sha256) != 64:
        raise ValueError("Fuente o SHA-256 de lock inválido")
    if execute:
        train_one.remote(source, lock_sha256)
    else:
        print(f"Preflight: {source}, lock={lock_sha256}; sin --execute no entrena")
