"""Entrenamiento de baselines en GPU de Modal (https://modal.com/docs/guide).

No importa funciones internas del pipeline: orquesta por
subprocess el mismo script CLI que corre `make train-baselines` (train_baselines.py, que a
su vez genera splits/seed_42_baseline de forma lazy si faltan), heredando el entorno de la
imagen (DATASET_ROOT=/data, OUTPUT_ROOT=/outputs) para que get_dataset_root()/
get_output_root() resuelvan a los Volumes montados.

Uso:
    modal run scripts/modal/train.py::seed_dataset            # 1 vez: dataset -> Volume
    modal run scripts/modal/train.py --models "efficientnet_b0" --epochs 30
    modal run scripts/modal/train.py::clean_outputs            # vacía el Volume corn-outputs
Requiere: `pip install -e ".[cloud]"`, `modal setup`, y el secret:
    modal secret create hf HF_TOKEN=hf_xxx
"""

import shutil
import subprocess
import sys
from pathlib import Path

import modal

from scripts.modal._common import DEFAULT_MODELS, REPO_ANCHOR, dataset_vol, image, outputs_vol

app = modal.App("corn-leaf-baselines", image=image)


@app.function(
    cpu=2.0,  # descarga I/O-bound; 2 cores bastan para descompresión/hash del dataset
    volumes={"/data": dataset_vol},
    secrets=[modal.Secret.from_name("hf")],
    timeout=3600,
)
def seed_dataset() -> None:
    """Descarga el dataset limpio al Volume corn-clean. Idempotente: download_dataset.py
    salta si /data/clean ya tiene contenido."""
    subprocess.run(
        [sys.executable, "scripts/dataset/download_dataset.py"],
        check=True,
        cwd=REPO_ANCHOR,
    )
    dataset_vol.commit()


@app.function(
    # Indexado y hashing SHA-256 de ~31k imagenes: I/O-bound y paralelizable, sin GPU.
    cpu=8.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    timeout=2 * 3600,
)
def make_splits(baseline: bool = False, no_cap: bool = False, max_per_class: int = 0) -> None:
    """
    Genera los splits CSV en el Volume corn-outputs.

    El pipeline principal (train_main) requiere outputs/splits/seed_42 y falla si no
    existe: a diferencia de train_baselines, no los genera de forma lazy. Regenerar es
    idempotente porque create_splits.py fija la semilla (42) y escanea con sorted().

    @param {bool} baseline Genera seed_42_baseline (perfil capado) en vez de seed_42.
    @param {bool} no_cap Sin tope por clase; solo aplica junto con baseline.
    @param {int} max_per_class Tope por clase; 0 usa el default del YAML. Solo con baseline.
    """
    dataset_vol.reload()
    command = [sys.executable, "scripts/pipeline/create_splits.py"]
    if baseline:
        command.append("--baseline")
        if no_cap:
            command.append("--no-cap")
        elif max_per_class:
            command += ["--max-per-class", str(max_per_class)]

    subprocess.run(command, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


@app.function(
    gpu="A10",
    # CPU explícita (el default de Modal es 0.125 cores): garantiza cores reales para el
    # indexado paralelo de splits y el DataLoader, sin depender del burst. Alineado con
    # SPLITS_INDEX_WORKERS=24. Facturación: se cobra max(request, uso real).
    cpu=4.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    # Techo dimensionado para el peor caso `--models all` (7 baselines) x 30 epochs. Con --no-cap
    # el train baseline es ~11.5k imgs (data/clean: healthy 8744 + common_rust 2256 +
    # fall_armyworm 4857 + nitrogen 523, split 70%), ~8x el perfil capado (~12 min/modelo).
    # Estimado ~1.5 h/modelo -> ~11 h los 7 secuenciales; 14 h dan margen para no morir por
    # timeout. El default son 3 modelos, así que sobra holgura.
    timeout=14 * 3600,
)
def train_baselines(
    models: str = DEFAULT_MODELS,
    epochs: int = 30,
    max_per_class: int = 0,
    no_cap: bool = False,
    regenerate_splits: bool = False,
    batch_size: int = 0,
    image_size: int = 0,
    learning_rate: float = 0.0,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    no_pretrained: bool = False,
    lime: bool = False,
) -> None:
    """Entrena los baselines indicados, persistiendo resultados en el Volume corn-outputs.
    Espeja `make train-baselines`/`train_baselines.py` - misma CLI, mismo comportamiento
    (incluida la generación lazy de splits/seed_42_baseline si aún no existen).
    """
    dataset_vol.reload()  # ve el dataset seedeado por seed_dataset

    train_args = [
        sys.executable,
        "scripts/pipeline/train_baselines.py",
        "--models",
        *models.split(),
        "--baseline",
        "--epochs",
        str(epochs),
    ]
    if no_cap:
        train_args.append("--no-cap")
    elif max_per_class:
        train_args += ["--max-per-class", str(max_per_class)]
    if regenerate_splits:
        train_args.append("--regenerate-splits")
    if batch_size:
        train_args += ["--batch-size", str(batch_size)]
    if image_size:
        train_args += ["--image-size", str(image_size)]
    if learning_rate:
        train_args += ["--learning-rate", str(learning_rate)]
    if weight_decay:
        train_args += ["--weight-decay", str(weight_decay)]
    if num_workers:
        train_args += ["--num-workers", str(num_workers)]
    if no_pretrained:
        train_args.append("--no-pretrained")
    if lime:
        train_args.append("--lime")
    subprocess.run(train_args, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


@app.function(
    gpu="A10",
    cpu=4.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    # El pipeline principal corre sobre las 31 623 imagenes (3.2x el perfil capado) con
    # hasta 60 epocas: ~2-4 h por corrida. El techo de 8 h deja margen para la variante
    # con CLAHE sin arriesgar una muerte por timeout a mitad de entrenamiento.
    timeout=8 * 3600,
)
def train_main(
    models: str = "shufflenet_v2_x1_0",
    epochs: int = 60,
    batch_size: int = 0,
    learning_rate: float = 0.0,
    class_weights: str = "",
    label_smoothing: float = -1.0,
    patience: int = 0,
    clahe: bool = False,
    no_pretrained: bool = False,
    num_workers: int = 0,
) -> None:
    """
    Entrena el pipeline principal en GPU, persistiendo en el Volume corn-outputs.

    @param {str} models Modelos separados por espacio.
    @param {int} epochs Techo de epocas; el early stopping puede cortar antes.
    @param {int} batch_size 0 usa el default del script.
    @param {float} learning_rate 0.0 usa el default del script.
    @param {str} class_weights Estrategia de pesos; "" usa el default (sqrt_inverse).
    @param {float} label_smoothing Negativo usa el default del script.
    @param {int} patience 0 usa el default del script.
    @param {bool} clahe Activa CLAHE como preprocesamiento.
    @param {bool} no_pretrained Entrena desde cero.
    @param {int} num_workers 0 usa el default del script.
    """
    dataset_vol.reload()
    command = [
        sys.executable,
        "scripts/pipeline/train.py",
        "--models",
        *models.split(),
        "--epochs",
        str(epochs),
    ]
    if batch_size:
        command += ["--batch-size", str(batch_size)]
    if learning_rate:
        command += ["--learning-rate", str(learning_rate)]
    if class_weights:
        command += ["--class-weights", class_weights]
    if label_smoothing >= 0:
        command += ["--label-smoothing", str(label_smoothing)]
    if patience:
        command += ["--patience", str(patience)]
    if num_workers:
        command += ["--num-workers", str(num_workers)]
    if clahe:
        command.append("--clahe")
    if no_pretrained:
        command.append("--no-pretrained")

    subprocess.run(command, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


@app.function(volumes={"/outputs": outputs_vol}, timeout=600)
def clean_outputs() -> None:
    """Vacía el contenido del Volume corn-outputs (splits/runs/reportes). No borra el Volume."""
    outputs_root = Path("/outputs")
    for entry in outputs_root.iterdir():
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    outputs_vol.commit()


@app.local_entrypoint()
def main(
    models: str = DEFAULT_MODELS,
    epochs: int = 30,
    max_per_class: int = 0,
    no_cap: bool = False,
    regenerate_splits: bool = False,
    batch_size: int = 0,
    image_size: int = 0,
    learning_rate: float = 0.0,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    no_pretrained: bool = False,
    lime: bool = False,
) -> None:
    """Entrypoint de `modal run`: dispara train_baselines en la GPU remota."""
    train_baselines.remote(
        models=models,
        epochs=epochs,
        max_per_class=max_per_class,
        no_cap=no_cap,
        regenerate_splits=regenerate_splits,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        no_pretrained=no_pretrained,
        lime=lime,
    )
