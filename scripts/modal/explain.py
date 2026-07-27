"""Explicabilidad (LIME + Grad-CAM) sobre checkpoints ya entrenados, en GPU de Modal.

Espeja scripts/pipeline/explain_lime.py / explain_report.py: orquesta por subprocess los
mismos scripts CLI que corre el Makefile (explain-lime, explain-report, explain-errors),
leyendo checkpoints/splits ya persistidos en el Volume corn-outputs por train.py.

Sirve a los dos pipelines vía --pipeline: "baselines" (default, runs de train.py) o "main"
(runs de train_main). Se elige por nombre y no por ruta absoluta a propósito: un
--output-dir /outputs/main sería reescrito por la conversión de rutas de MSYS al invocarlo
desde Git Bash en Windows.

Uso:
    modal run scripts/modal/explain.py::explain_lime --models "efficientnet_b0"
    modal run scripts/modal/explain.py::explain_report --models "efficientnet_b0" --sample-size 50
    modal run scripts/modal/explain.py::explain_errors --models "efficientnet_b0"
    modal run scripts/modal/explain.py::explain_report --models "shufflenet_v2_x1_0" \
        --pipeline main
Requiere: `pip install -e ".[cloud]"`, `modal setup`, y el secret:
    modal secret create hf HF_TOKEN=hf_xxx
"""

import subprocess
import sys

import modal

from scripts.modal._common import (
    DATASET_MOUNT,
    DEFAULT_MODELS,
    OUTPUTS_MOUNT,
    REPO_ANCHOR,
    dataset_vol,
    image,
    outputs_vol,
)

app = modal.App("corn-leaf-explain", image=image)

# A10 (misma que training): el costo de LIME escala linealmente con num_samples (300 hoy,
# con planes de subir a ~1000-2000) - el mayor throughput sobre T4 amortiza esa subida
# futura sin tener que revisar el tier de cómputo otra vez.
_GPU = "A10"
_VOLUMES = {DATASET_MOUNT: dataset_vol, OUTPUTS_MOUNT: outputs_vol}

_PIPELINE_DIRS = {
    "baselines": f"{OUTPUTS_MOUNT}/baselines",
    "main": f"{OUTPUTS_MOUNT}/main",
}


def _output_dir_args(pipeline: str) -> list[str]:
    """
    Traduce el nombre de pipeline al flag --output-dir del script CLI correspondiente.

    @param {str} pipeline "baselines" o "main".
    @returns {list[str]} Par ["--output-dir", <ruta en el Volume corn-outputs>].
    @throws {ValueError} Si el pipeline no es uno de los conocidos.
    """
    if pipeline not in _PIPELINE_DIRS:
        raise ValueError(f"pipeline debe ser uno de {sorted(_PIPELINE_DIRS)}, no {pipeline!r}")
    return ["--output-dir", _PIPELINE_DIRS[pipeline]]


@app.function(gpu=_GPU, volumes=_VOLUMES, secrets=[modal.Secret.from_name("hf")], timeout=3600)
def explain_lime(
    models: str = DEFAULT_MODELS,
    run: str = "",
    baseline: bool = False,
    image: str = "",
    output: str = "",
    pipeline: str = "baselines",
) -> None:
    """Reporte visual LIME+Grad-CAM por imagen. Espeja `make explain-lime`.

    image/output son rutas dentro del contenedor (relativas a /data o /outputs, los
    Volumes montados) - no rutas del filesystem local del caller. pipeline elige de qué
    directorio de runs leer: "baselines" (default) o "main".
    """
    args = [sys.executable, "scripts/pipeline/explain_lime.py", "--models", *models.split()]
    args += _output_dir_args(pipeline)
    if run:
        args += ["--run", run]
    if baseline:
        args += ["--baseline"]
    if image:
        args += ["--image", image]
    if output:
        args += ["--output", output]
    subprocess.run(args, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


# Techo para el peor caso: num_samples=1000 x report_sample_size=30/clase x 4 clases x 7 modelos
# (--models all) roza la hora; 3 h dan holgura para el barrido completo sin cortes. El default
# son 3 modelos.
@app.function(gpu=_GPU, volumes=_VOLUMES, secrets=[modal.Secret.from_name("hf")], timeout=3 * 3600)
def explain_report(
    models: str = DEFAULT_MODELS,
    run: str = "",
    baseline: bool = False,
    sample_size: int = 0,
    num_samples: int = 0,
    pipeline: str = "baselines",
) -> None:
    """Fidelidad agregada sobre una muestra amplia. Espeja `make explain-report`.

    pipeline elige de qué directorio de runs leer: "baselines" (default) o "main".
    """
    args = [sys.executable, "scripts/pipeline/explain_report.py", "--models", *models.split()]
    args += _output_dir_args(pipeline)
    if run:
        args += ["--run", run]
    if baseline:
        args += ["--baseline"]
    if sample_size:
        args += ["--sample-size", str(sample_size)]
    if num_samples:
        args += ["--num-samples", str(num_samples)]
    subprocess.run(args, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


@app.function(gpu=_GPU, volumes=_VOLUMES, secrets=[modal.Secret.from_name("hf")], timeout=3600)
def explain_errors(
    models: str = DEFAULT_MODELS,
    run: str = "",
    baseline: bool = False,
    num_samples: int = 0,
    pipeline: str = "baselines",
) -> None:
    """LIME dirigido a falsos positivos/negativos. Espeja `make explain-errors`.

    pipeline elige de qué directorio de runs leer: "baselines" (default) o "main".
    """
    args = [
        sys.executable,
        "scripts/pipeline/explain_report.py",
        "--models",
        *models.split(),
        "--errors-only",
    ]
    args += _output_dir_args(pipeline)
    if run:
        args += ["--run", run]
    if baseline:
        args += ["--baseline"]
    if num_samples:
        args += ["--num-samples", str(num_samples)]
    subprocess.run(args, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()
