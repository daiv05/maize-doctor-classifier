"""Exportacion de checkpoints del pipeline principal a ONNX/TFLite, en Modal.

Espeja scripts/pipeline/export.py: orquesta por subprocess el mismo CLI, leyendo
checkpoints/splits ya persistidos en el Volume corn-outputs por train_main.

Uso:
    modal run scripts/modal/export.py::export_main --models "efficientnet_b0"
    modal run scripts/modal/export.py::export_main --models "shufflenet_v2_x1_0" \
        --formats "onnx,tflite"
    modal run scripts/modal/export.py::export_main --formats "tflite" --quantize "int8"
    modal run scripts/modal/export.py::evaluate_export_main --formats "onnx,tflite"
Requiere: `pip install -e ".[cloud]"`, `modal setup`, y el secret:
    modal secret create hf HF_TOKEN=hf_xxx
"""

import os
import subprocess
import sys

import modal

from scripts.modal._common import (
    DEFAULT_MODELS,
    OUTPUTS_MOUNT,
    REPO_ANCHOR,
    SEGMENTED_DATASET_MOUNT,
    dataset_vol,
    export_image,
    image,
    outputs_vol,
    segmented_dataset_vol,
)

app = modal.App("corn-leaf-export", image=image)

_VOLUMES = {
    "/data": dataset_vol,
    SEGMENTED_DATASET_MOUNT: segmented_dataset_vol,
    "/outputs": outputs_vol,
}


def _segmented_env(segmented: bool) -> dict:
    """Entorno del subprocess: DATASET_ROOT apunta a /data_segmented si `segmented`."""
    env = dict(os.environ)
    if segmented:
        env["DATASET_ROOT"] = SEGMENTED_DATASET_MOUNT
    return env


@app.function(
    image=export_image,
    volumes=_VOLUMES,
    secrets=[modal.Secret.from_name("hf")],
    timeout=3600,
)
def export_main(
    models: str = DEFAULT_MODELS,
    run: str = "",
    checkpoint: str = "",
    formats: str = "onnx",
    quantize: str = "",
    splits_dir: str = "",
    tolerance: float = 0.0,
    min_agreement_rate: float = 0.0,
    parity_sample_size: int = 0,
    no_parity: bool = False,
    segmented: bool = False,
) -> None:
    """
    Exporta checkpoints del pipeline principal a ONNX/TFLite. Espeja `make export-main`.

    @param {str} models Modelos separados por espacio.
    @param {str} run run_id especifico; vacio usa latest.json.
    @param {str} checkpoint Ruta explicita a un checkpoint .pth (ignora run si se pasa).
    @param {str} formats Formatos a exportar, CSV (ej: "onnx,tflite").
    @param {str} quantize Cuantizacion: "int8" o vacio para FP32.
    @param {str} splits_dir Directorio con test.csv; vacio usa el de summary.json.
    @param {float} tolerance Tolerancia de paridad; 0.0 usa el default del script.
    @param {float} min_agreement_rate Acuerdo minimo; 0.0 usa el default del script.
    @param {int} parity_sample_size Muestras para paridad; 0 usa el default del script.
    @param {bool} no_parity Omite la validacion de paridad numerica.
    @param {bool} segmented El run fue entrenado sobre corn-clean-segmented: usa DATASET_ROOT=
        /data_segmented para que la muestra de paridad lea las mismas imagenes que el checkpoint.
    """
    dataset_vol.reload()
    segmented_dataset_vol.reload()
    args = [
        sys.executable,
        "scripts/pipeline/export.py",
        "--models",
        *models.split(),
        "--output-dir",
        f"{OUTPUTS_MOUNT}/main",
        "--formats",
        formats,
    ]
    if quantize:
        args += ["--quantize", quantize]
    if run:
        args += ["--run", run]
    if checkpoint:
        args += ["--checkpoint", checkpoint]
    if splits_dir:
        args += ["--splits-dir", splits_dir]
    if tolerance:
        args += ["--tolerance", str(tolerance)]
    if min_agreement_rate:
        args += ["--min-agreement-rate", str(min_agreement_rate)]
    if parity_sample_size:
        args += ["--parity-sample-size", str(parity_sample_size)]
    if no_parity:
        args.append("--no-parity")
    try:
        subprocess.run(args, check=True, cwd=REPO_ANCHOR, env=_segmented_env(segmented))
    finally:
        # Se commitea aunque un modelo falle: los que si exportaron deben persistir.
        outputs_vol.commit()


@app.function(
    image=export_image,
    volumes=_VOLUMES,
    secrets=[modal.Secret.from_name("hf")],
    timeout=7200,
)
def evaluate_export_main(
    models: str = DEFAULT_MODELS,
    run: str = "",
    formats: str = "onnx",
    quantize: str = "",
    splits_dir: str = "",
    batch_size: int = 0,
    no_torch_baseline: bool = False,
    max_macro_f1_drop: float = 0.0,
    segmented: bool = False,
) -> None:
    """
    Evalua modelos exportados sobre el split de test completo. Espeja `make eval-export-main`.

    @param {str} models Modelos separados por espacio.
    @param {str} run run_id especifico; vacio usa latest.json.
    @param {str} formats Formatos a evaluar, CSV (ej: "onnx,tflite").
    @param {str} quantize Variante a evaluar: "int8" o vacio para FP32.
    @param {str} splits_dir Directorio con test.csv; vacio usa el de summary.json.
    @param {int} batch_size Tamano de batch; 0 usa el default del script.
    @param {bool} no_torch_baseline Omite la referencia PyTorch (sin deltas).
    @param {float} max_macro_f1_drop Caida maxima de macro-F1; 0.0 usa el default.
    @param {bool} segmented El run fue entrenado sobre corn-clean-segmented: usa DATASET_ROOT=
        /data_segmented para evaluar sobre las mismas imagenes que vio el checkpoint.
    """
    dataset_vol.reload()
    segmented_dataset_vol.reload()
    outputs_vol.reload()
    args = [
        sys.executable,
        "-u",
        "scripts/pipeline/evaluate_export.py",
        "--models",
        *models.split(),
        "--output-dir",
        f"{OUTPUTS_MOUNT}/main",
        "--formats",
        formats,
    ]
    if quantize:
        args += ["--quantize", quantize]
    if run:
        args += ["--run", run]
    if splits_dir:
        args += ["--splits-dir", splits_dir]
    if batch_size:
        args += ["--batch-size", str(batch_size)]
    if no_torch_baseline:
        args.append("--no-torch-baseline")
    if max_macro_f1_drop:
        args += ["--max-macro-f1-drop", str(max_macro_f1_drop)]
    try:
        subprocess.run(args, check=True, cwd=REPO_ANCHOR, env=_segmented_env(segmented))
    finally:
        outputs_vol.commit()


@app.function(
    gpu="A10",
    volumes=_VOLUMES,
    secrets=[modal.Secret.from_name("hf")],
    timeout=3600,
)
def compute_ood_stats(
    models: str = DEFAULT_MODELS,
    run: str = "",
    checkpoint: str = "",
    splits_dir: str = "",
    batch_size: int = 0,
    percentile: float = 0.0,
    segmented: bool = False,
) -> None:
    """
    Calcula ood_stats.json (centroides + covarianza + umbral Mahalanobis) por modelo,
    persistiendo en <run_dir>/export/. Espeja `make compute-ood-stats`.

    @param {str} models Modelos separados por espacio.
    @param {str} run run_id especifico; vacio usa latest.json.
    @param {str} checkpoint Ruta explicita a un checkpoint .pth (solo con un unico modelo).
    @param {str} splits_dir Directorio con train.csv/val.csv; vacio usa el de summary.json.
    @param {int} batch_size 0 usa el default del script.
    @param {float} percentile 0.0 usa el default del script (95).
    @param {bool} segmented El run fue entrenado sobre corn-clean-segmented: usa DATASET_ROOT=
        /data_segmented para extraer features sobre las mismas imagenes que vio el checkpoint.
    """
    dataset_vol.reload()
    segmented_dataset_vol.reload()
    args = [
        sys.executable,
        "scripts/pipeline/compute_ood_stats.py",
        "--models",
        *models.split(),
        "--output-dir",
        f"{OUTPUTS_MOUNT}/main",
    ]
    if run:
        args += ["--run", run]
    if checkpoint:
        args += ["--checkpoint", checkpoint]
    if splits_dir:
        args += ["--splits-dir", splits_dir]
    if batch_size:
        args += ["--batch-size", str(batch_size)]
    if percentile:
        args += ["--percentile", str(percentile)]
    try:
        subprocess.run(args, check=True, cwd=REPO_ANCHOR, env=_segmented_env(segmented))
    finally:
        outputs_vol.commit()
