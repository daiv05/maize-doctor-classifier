"""Ejecución de pre-segmentación del dataset en GPU de Modal.

Montajes de Volumes:
  /data           -> corn-clean (lectura del dataset original /data/clean)
  /segmenter      -> doctor-maiz-leaf-segmentation-outputs (pesos del segmentador)
  /data_segmented -> corn-clean-segmented (escritura de imágenes con hoja aislada)
  /outputs        -> corn-outputs (previews visuales y métricas de auditoría)

Uso detached (desacoplado de la máquina local):
  modal run --detach scripts/modal/segment_dataset.py
  modal run --detach scripts/modal/segment_dataset.py --profile crop_mask_letterbox
  modal run --detach scripts/modal/segment_dataset.py --max-images 1000  # prueba rápida
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "corn-leaf-dataset-segmentation"
REPO_ANCHOR = "/root"

dataset_vol = modal.Volume.from_name("corn-clean")
segmenter_vol = modal.Volume.from_name("doctor-maiz-leaf-segmentation-outputs")
segmented_dataset_vol = modal.Volume.from_name("corn-clean-segmented", create_if_missing=True)
outputs_vol = modal.Volume.from_name("corn-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.12.1",
        "torchvision==0.27.1",
        "ultralytics==8.4.104",
        "Pillow==12.3.0",
        "tqdm>=4.66,<4.69",
        "pyyaml>=6.0,<7.0",
        "numpy==2.5.1",
        "opencv-python-headless>=4.8,<5.0",
        "python-dotenv>=1.0,<1.3",
        "pandas>=2.0,<3.0",
    )
    .env(
        {
            "DATASET_ROOT": "/data",
            "SEGMENTED_DATASET_ROOT": "/data_segmented",
            "OUTPUT_ROOT": "/outputs",
        }
    )
    .add_local_dir("config", f"{REPO_ANCHOR}/config", copy=True)
    .add_local_python_source("src", "scripts")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="A10G",
    cpu=4.0,
    volumes={
        "/data": dataset_vol,
        "/segmenter": segmenter_vol,
        "/data_segmented": segmented_dataset_vol,
        "/outputs": outputs_vol,
    },
    timeout=24 * 3600,
)
def run_segmentation_job(
    profile: str = "crop_mask_letterbox",
    max_images: int = 0,
    max_previews: int = 50,
    splits_dir: str = "/outputs/splits/seed_42",
    checkpoint: str = "/segmenter/leaf_detection/models/doctor_maiz_leaf_segmenter_best.pt",
) -> None:
    print("[*] Recargando volúmenes de entrada...", flush=True)
    dataset_vol.reload()
    segmenter_vol.reload()

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint explícito inexistente: {checkpoint_path}")

    print(f"[*] Checkpoint resuelto en Modal: {checkpoint_path}", flush=True)

    cmd = [
        sys.executable,
        "scripts/pipeline/segment_dataset.py",
        "--dataset-dir",
        "/data",
        "--output-dir",
        "/data_segmented/clean",
        "--preview-dir",
        "/outputs/segmentation_previews",
        "--checkpoint",
        str(checkpoint_path),
        "--splits-dir",
        splits_dir,
        "--profile",
        profile,
        "--max-previews",
        str(max_previews),
        "--device",
        "cuda:0",
    ]

    if max_images > 0:
        cmd.extend(["--max-images", str(max_images)])

    print(f"[*] Ejecutando comando: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ANCHOR)

    print("[*] Confirmando cambios en volúmenes persistentes...", flush=True)
    segmented_dataset_vol.commit()
    outputs_vol.commit()
    print("[*] Proceso completado exitosamente en Modal.", flush=True)


@app.local_entrypoint()
def main(
    profile: str = "crop_mask_letterbox",
    max_images: int = 0,
    max_previews: int = 50,
    splits_dir: str = "/outputs/splits/seed_42",
    checkpoint: str = "/segmenter/leaf_detection/models/doctor_maiz_leaf_segmenter_best.pt",
) -> None:
    print(
        f"Lanzando pre-segmentación en Modal (profile={profile}, "
        f"max_images={max_images}, max_previews={max_previews})..."
    )
    run_segmentation_job.remote(
        profile=profile,
        max_images=max_images,
        max_previews=max_previews,
        splits_dir=splits_dir,
        checkpoint=checkpoint,
    )
