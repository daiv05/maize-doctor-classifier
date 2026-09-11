"""Endurecimiento por augmentation en GPU de Modal.

Orquesta por subprocess el mismo CLI que corre en local
(scripts/experiments/augmentation_hardening.py), heredando el entorno de la imagen
(DATASET_ROOT=/data, OUTPUT_ROOT=/outputs) para que los Volumes montados resuelvan igual
que en el pipeline principal.

Uso:
    modal run --detach scripts/modal/augmentation_hardening.py --augment none
    modal run --detach scripts/modal/augmentation_hardening.py --augment hardened
"""

import subprocess
import sys
from pathlib import Path

import modal

from scripts.modal._common import REPO_ANCHOR, dataset_vol, image, outputs_vol

app = modal.App("corn-augmentation-hardening", image=image)

RESULTS_TEMPLATE = "/outputs/experiments/augmentation_{augment}{seed}.json"


@app.function(
    gpu="A10",
    # La augmentation endurecida recomprime y remuestrea en CPU dentro del DataLoader, asi que
    # pesa mas que en los experimentos previos: 8 cores siguen alimentando a la A10.
    cpu=8.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    timeout=8 * 3600,
)
def run_augmentation_hardening(
    augment: str = "none",
    model: str = "efficientnet_lite0",
    epochs: int = 25,
    train_cap: int = 1000,
    val_cap: int = 2000,
    batch_size: int = 64,
    seed: int = 0,
    min_crop_scale: float = 0.30,
) -> None:
    """Ejecuta los once pliegues en la GPU remota y persiste el JSON en el Volume.

    @param {str} augment Uno de none o hardened.
    @param {float} min_crop_scale Fracción mínima del área conservada por el recorte.
    """
    dataset_vol.reload()
    outputs_vol.reload()

    splits_dir = Path("/outputs/splits/seed_42")
    if not (splits_dir / "test.csv").exists():
        raise FileNotFoundError(f"No existe {splits_dir}/test.csv en el Volume corn-outputs")

    command = [
        sys.executable,
        "scripts/experiments/augmentation_hardening.py",
        "--splits-dir", str(splits_dir),
        "--augment", augment,
        "--model", model,
        "--epochs", str(epochs),
        "--train-cap", str(train_cap),
        "--val-cap", str(val_cap),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--min-crop-scale", str(min_crop_scale),
        "--num-workers", "8",
        "--output", RESULTS_TEMPLATE.format(
            augment=augment, seed="" if seed == 0 else f"_seed{seed}"),
    ]
    subprocess.run(command, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


@app.local_entrypoint()
def main(
    augment: str = "none",
    model: str = "efficientnet_lite0",
    epochs: int = 25,
    train_cap: int = 1000,
    val_cap: int = 2000,
    batch_size: int = 64,
    seed: int = 0,
    min_crop_scale: float = 0.30,
) -> None:
    """Entrypoint de `modal run`: dispara el endurecimiento en la GPU remota."""
    run_augmentation_hardening.remote(
        augment=augment,
        model=model,
        epochs=epochs,
        train_cap=train_cap,
        val_cap=val_cap,
        batch_size=batch_size,
        seed=seed,
        min_crop_scale=min_crop_scale,
    )
