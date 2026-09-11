"""Imagen y Volumes de Modal compartidos entre train.py y explain.py.

Factorizado para que ambos módulos usen exactamente la misma imagen (versión de torch,
extras instalados) y los mismos Volumes - divergir entre ellos rompería la reutilización
de checkpoints/splits generados por uno y consumidos por el otro.
"""

import modal

REPO_ANCHOR = "/root"
HF_DATASET_REPO = "daiv05/corn-leaf-diseases-pests-and-deficiencies"

DATASET_MOUNT = "/data"
SEGMENTED_DATASET_MOUNT = "/data_segmented"
OUTPUTS_MOUNT = "/outputs"

DEFAULT_MODELS = "efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0"

dataset_vol = modal.Volume.from_name("corn-clean", create_if_missing=True)
segmented_dataset_vol = modal.Volume.from_name("corn-clean-segmented", create_if_missing=True)
outputs_vol = modal.Volume.from_name("corn-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
        "torchvision==0.27.1",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["cloud", "xai", "export", "tuning"])
    # Explicito ademas del extra 'export': alli van con marcador sys_platform == 'linux'
    # (litert-torch no existe para Windows/macOS) y no queremos depender de como Modal
    # resuelva ese marcador al construir la imagen. El contenedor siempre es Linux.
    .pip_install("litert-torch>=0.9,<0.10", "ai-edge-litert>=2.1,<3")
    .env(
        {
            "DATASET_ROOT": DATASET_MOUNT,
            "OUTPUT_ROOT": OUTPUTS_MOUNT,
            "HF_DATASET_REPO": HF_DATASET_REPO,
            "SPLITS_INDEX_WORKERS": "24",
        }
    )
    .add_local_dir("config", f"{REPO_ANCHOR}/config", copy=True)
    .add_local_python_source("src", "scripts")
)
