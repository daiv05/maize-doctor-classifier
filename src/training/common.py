"""Utilidades compartidas por los scripts de entrenamiento (train.py y train_baselines.py)."""

import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.models.registry import ModelRegistry

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
    latest_path = output_dir / model_name / "latest.json"
    latest_path.write_text(json.dumps({"run_id": run_id}, indent=2))


def resolve_run_dir(output_dir: Path, model_name: str, run_id: str | None = None) -> Path:
    """Resuelve el directorio de un run: sin run_id, lee latest.json; falla con
    SystemExit claro si no hay ningún run registrado o el run_id pedido no existe."""
    model_dir = output_dir / model_name
    if run_id is None:
        latest_path = model_dir / "latest.json"
        if not latest_path.exists():
            raise SystemExit(
                f"No hay runs registrados para '{model_name}' en {model_dir}. "
                "Entrena primero con: make train-baselines"
            )
        run_id = json.loads(latest_path.read_text())["run_id"]
    run_dir = model_dir / run_id
    if not run_dir.exists():
        raise SystemExit(f"No se encontró el run '{run_id}' para '{model_name}' en {model_dir}")
    return run_dir


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
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        splits_dir = Path(summary.get("splits_dir", fallback_splits_dir))
        class_to_idx = {
            str(class_name): int(class_idx)
            for class_name, class_idx in summary["class_to_idx"].items()
        }
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        image_size = summary.get("image_size", list(fallback_target_size))
        target_size = (int(image_size[0]), int(image_size[1]))
        return splits_dir, class_to_idx, idx_to_class, target_size

    # Import diferido: evita arrastrar pandas/yaml (cadena de src.data.dataset) salvo que
    # realmente falte summary.json.
    from src.data.dataset import resolve_class_mapping

    class_to_idx, idx_to_class = resolve_class_mapping(
        fallback_splits_dir / "train.csv", fallback_classes
    )
    return fallback_splits_dir, class_to_idx, idx_to_class, fallback_target_size
