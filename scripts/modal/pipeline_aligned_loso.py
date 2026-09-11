"""Validación por fuente alineada al pipeline principal, en GPU de Modal.

Orquesta por subprocess el mismo CLI que corre en local
(scripts/experiments/pipeline_aligned_loso.py), heredando el entorno de la imagen
(DATASET_ROOT=/data, OUTPUT_ROOT=/outputs).

Uso:
    modal run --detach scripts/modal/pipeline_aligned_loso.py --arm baseline
    modal run --detach scripts/modal/pipeline_aligned_loso.py --arm colour_strong
"""

import subprocess
import sys
from pathlib import Path

import modal

from scripts.modal._common import REPO_ANCHOR, dataset_vol, image, outputs_vol

app = modal.App("corn-pipeline-aligned-loso", image=image)

RESULTS_TEMPLATE = "/outputs/experiments/aligned_{mode}_{arm}{seed}.json"


@app.function(
    gpu="A10",
    cpu=8.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    # Once pliegues sin tope de entrenamiento y hasta 60 epocas: es el perfil del pipeline
    # principal multiplicado por los pliegues, asi que el techo va holgado.
    timeout=20 * 3600,
)
def run_aligned_loso(
    arm: str = "baseline",
    split_mode: str = "source",
    model: str = "efficientnet_lite0",
    epochs: int = 60,
    train_cap: int = 1500,
    val_cap: int = 0,
    batch_size: int = 32,
    seed: int = 42,
    folds: str = "",
) -> None:
    """Ejecuta la validación por fuente con los componentes del pipeline principal.

    @param {str} arm Brazo de augmentation: baseline, colour_strong, crop_all o minority_for_all.
    @param {int} train_cap Tope por clase; 0 usa el split completo como el pipeline real.
    """
    dataset_vol.reload()
    outputs_vol.reload()

    splits_dir = Path("/outputs/splits/seed_42")
    if not (splits_dir / "test.csv").exists():
        raise FileNotFoundError(f"No existe {splits_dir}/test.csv en el Volume corn-outputs")

    command = [
        sys.executable,
        "scripts/experiments/pipeline_aligned_loso.py",
        "--splits-dir", str(splits_dir),
        "--arm", arm,
        "--split-mode", split_mode,
        "--model", model,
        "--epochs", str(epochs),
        "--train-cap", str(train_cap),
        "--val-cap", str(val_cap),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--num-workers", "8",
        "--output", RESULTS_TEMPLATE.format(
            mode=split_mode, arm=arm,
            seed="" if seed == 42 else f"_seed{seed}"),
    ]
    if folds:
        command += ["--folds", folds]

    subprocess.run(command, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


@app.local_entrypoint()
def main(
    arm: str = "baseline",
    split_mode: str = "source",
    model: str = "efficientnet_lite0",
    epochs: int = 60,
    train_cap: int = 1500,
    val_cap: int = 0,
    batch_size: int = 32,
    seed: int = 42,
    folds: str = "",
) -> None:
    """Entrypoint de `modal run`: dispara la validación alineada en la GPU remota."""
    run_aligned_loso.remote(
        arm=arm, split_mode=split_mode, model=model, epochs=epochs,
        train_cap=train_cap, val_cap=val_cap,
        batch_size=batch_size, seed=seed, folds=folds,
    )
