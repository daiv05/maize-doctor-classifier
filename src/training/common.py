"""Utilidades compartidas por los scripts de entrenamiento (train.py y train_baselines.py)."""

import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.models.registry import ModelRegistry
from src.training.runs import (
    read_run_contract,
    write_latest_pointer,
)
from src.training.runs import (
    resolve_run_dir as _resolve_run_dir,
)

logger = logging.getLogger(__name__)


def resolve_model_names(requested: list[str], registry: ModelRegistry) -> list[str]:
    available = registry.list_names()
    if requested == ["all"]:
        return available
    unknown = [n for n in requested if n not in registry]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {available}")
    return requested


def worker_init_fn(worker_id: int) -> None:
    """Propaga la semilla per-worker de PyTorch (variable por época) a `random` y `numpy`."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device() -> torch.device:
    """Selecciona cuda si está disponible y deja registro del hardware detectado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Dispositivo: GPU - {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")
    else:
        logger.warning(
            "Dispositivo: CPU (no se detectó GPU - el entrenamiento será "
            "significativamente más lento)"
        )
    return device


def generate_run_id() -> str:
    """Identificador de run basado en timestamp (YYYYMMDD_HHMMSS)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_dir(output_dir: Path, model_name: str, run_id: str) -> Path:
    """Crea y devuelve <output_dir>/<model_name>/<run_id>/."""
    run_dir = output_dir / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def update_latest_pointer(output_dir: Path, model_name: str, run_id: str) -> None:
    """Escribe <output_dir>/<model_name>/latest.json. Llamar solo tras un run exitoso
    (summary.json ya escrito), para no apuntar a runs a medias."""
    write_latest_pointer(output_dir, model_name, run_id)


def resolve_run_dir(output_dir: Path, model_name: str, run_id: str | None = None) -> Path:
    """Resuelve el directorio de un run: sin run_id, lee latest.json; falla con
    SystemExit claro si no hay ningún run registrado o el run_id pedido no existe."""
    return _resolve_run_dir(output_dir, model_name, run_id)


def load_run_metadata(
    run_dir: Path,
    fallback_splits_dir: Path,
    fallback_classes: list[str],
    fallback_target_size: tuple[int, int],
) -> tuple[Path, dict[str, int], dict[int, str], tuple[int, int]]:
    """Resuelve (splits_dir, class_to_idx, idx_to_class, target_size) para un run baseline.

    Fuente de verdad: el `summary.json` del run, que persiste el mapeo clase->índice y el
    tamaño de entrada con los que se entrenó ese checkpoint. Solo si falta summary.json se
    cae al fallback derivado del YAML + train.csv.

    Compartida por los cinco subcomandos de `scripts/pipeline/explain.py`: garantiza que
    todos traduzcan el argmax del modelo con EXACTAMENTE el mismo mapeo que el head
    entrenado. Reconstruir el mapeo desde `baseline.classes` (cuyo orden puede diferir del
    canónico `dataset.classes` que usa CornDataset) produce etiquetas permutadas - ese fue
    el bug de rótulos de los reportes LIME.
    """
    # Los argumentos fallback se conservan en la firma por compatibilidad de API, pero un
    # run legacy ya no puede cargarse silenciosamente: debe migrarse explícitamente.
    del fallback_classes, fallback_target_size
    summary = read_run_contract(run_dir)
    recorded_splits = Path(summary["splits_dir"])
    splits_dir = recorded_splits if recorded_splits.exists() else fallback_splits_dir
    class_to_idx = dict(summary["class_to_idx"])
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    image_size = summary["architecture"]["input_size"]
    return splits_dir, class_to_idx, idx_to_class, (int(image_size[0]), int(image_size[1]))
