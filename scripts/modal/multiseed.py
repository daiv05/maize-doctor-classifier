"""Orquestación explícita, secuencial y acotada; no inicia HPO ni evaluación test."""

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

app = modal.App("doctor-maiz-multiseed", image=image)


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
def run_batch(protocol: dict, max_new_runs: int = 10):
    """El protocolo viene del plan local; cada slot es de un único intento."""
    from src.data.preparation import atomic_write_json
    from src.training.multiseed import EXPERIMENT, freeze_protocol, verify_sources

    dataset_vol.reload()
    outputs_vol.reload()
    verify_sources(protocol)
    directory = Path("/outputs/multiseed") / EXPERIMENT
    directory.mkdir(parents=True, exist_ok=True)
    freeze_protocol(directory, protocol)
    atomic_write_json(directory / "LAUNCH.json", {"max_new_runs": max_new_runs})
    command = [
        sys.executable,
        "-m",
        "scripts.pipeline.multiseed",
        "run",
        "--output-dir",
        str(directory),
        "--splits-dir",
        "/outputs/splits/seed_42",
        "--max-new-runs",
        str(max_new_runs),
    ]
    try:
        run_with_periodic_commit(command, cwd=REPO_ANCHOR, volume=outputs_vol)
    finally:
        outputs_vol.commit()


@app.local_entrypoint()
def main(protocol: str, execute: bool = False, max_new_runs: int = 10):
    """Sin --execute solo muestra el plan; no invoca ninguna función remota."""
    if not 1 <= max_new_runs <= 10:
        raise ValueError("Presupuesto permitido: 1..10")
    payload = json.loads(Path(protocol).read_text())
    print(
        f"Plan congelado: {payload['seeds']}, 2 configuraciones; máximo {max_new_runs} nuevas runs"
    )
    if execute:
        run_batch.remote(payload, max_new_runs)
    else:
        print("Sin --execute: no se inicia entrenamiento. Verificar créditos antes de lanzar.")
