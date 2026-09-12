"""Construcción del loader de test compartido por exportación y evaluación.

Ambos entrypoints necesitan exactamente el mismo split, el mismo `class_to_idx` y el mismo
pipeline de transforms ('test', determinista y sin augmentation). Factorizarlo evita que
la paridad y la evaluación terminen mirando datos distintos.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.config import get_output_root
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory
from src.provenance import sha256_file

logger = logging.getLogger(__name__)


def resolve_split_csv(run_dir: Path, splits_dir: str | None, split_name: str) -> Path:
    """
    Resuelve la ruta de `<split_name>.csv`, tolerando runs movidos entre Modal y local.

    `summary.json` guarda `splits_dir` como ruta **absoluta** del entorno donde se entrenó.
    Un run entrenado en Modal lo deja como `/outputs/splits/seed_42`, que no existe al
    bajarlo con `make modal-pull`. Por eso, si la ruta registrada no existe, se reintenta
    el mismo nombre de split bajo el `OUTPUT_ROOT` actual antes de rendirse.

    Orden de resolución: `--splits-dir` > `splits_dir` de summary.json > remapeo al
    OUTPUT_ROOT local.

    @param {Path} run_dir Directorio del run.
    @param {str|None} splits_dir Override explícito del directorio de splits.
    @param {str} split_name Nombre del split ("train", "val" o "test").
    @returns {Path} Ruta a `<split_name>.csv`.
    @throws {SystemExit} Si no se encuentra en ninguna de las rutas candidatas.
    """
    summary = json.loads((run_dir / "summary.json").read_text())
    if splits_dir:
        candidates = [Path(splits_dir)]
    else:
        summary = json.loads((run_dir / "summary.json").read_text())
        default_dir = get_output_root() / "splits" / "seed_42"
        recorded = Path(summary.get("splits_dir", default_dir))
        # Mismo nombre de split (seed_42, seed_42_baseline, ...) bajo el OUTPUT_ROOT local.
        remapped = get_output_root() / "splits" / recorded.name
        candidates = [recorded] if recorded == remapped else [recorded, remapped]

    for candidate in candidates:
        split_csv = candidate / f"{split_name}.csv"
        if split_csv.exists():
            expected = summary.get("split_sha256", {}).get(split_name)
            if expected and sha256_file(split_csv) != expected:
                raise ValueError(f"Split does not match the training contract: {split_csv}")
            if candidate != candidates[0]:
                logger.warning(
                    "El splits_dir del run (%s) no existe en esta maquina; usando %s.",
                    candidates[0],
                    candidate,
                )
            return split_csv

    intentadas = "\n  ".join(str(c / f"{split_name}.csv") for c in candidates)
    raise SystemExit(
        f"No se encontro {split_name}.csv. Rutas intentadas:\n  {intentadas}\n"
        "Pasa --splits-dir con el directorio correcto, o genera los splits con: make splits"
    )


def resolve_test_csv(run_dir: Path, splits_dir: str | None) -> Path:
    """Resuelve la ruta de test.csv. Ver `resolve_split_csv`."""
    return resolve_split_csv(run_dir, splits_dir, "test")


def build_test_loader(
    test_csv: Path,
    config_path: Path,
    class_to_idx: dict[str, int],
    image_size: tuple[int, int],
    batch_size: int = 32,
    num_workers: int = 0,
    preprocessing: dict | None = None,
) -> tuple[DataLoader, pd.Series]:
    """
    Construye el DataLoader del split de test y la serie de entornos alineada.

    `shuffle=False` no es opcional: la serie de entornos se alinea por posición con el
    orden en que el loader entrega las imágenes.

    @param {Path} test_csv Ruta al CSV del split de test.
    @param {Path} config_path Ruta al YAML de configuración.
    @param {dict[str,int]} class_to_idx Mapeo clase->índice del run.
    @param {tuple[int,int]} image_size Alto y ancho de entrada (h, w).
    @param {int} batch_size Tamaño de batch.
    @param {int} num_workers Workers del DataLoader.
    @returns {tuple[DataLoader, pd.Series]} Loader y entornos por imagen.
    """
    factory = (
        CornTransformFactory.from_contract(preprocessing, config_path=str(config_path))
        if preprocessing
        else CornTransformFactory(config_path=str(config_path), target_size=image_size)
    )
    dataset = CornDataset(
        csv_path=str(test_csv),
        config_path=str(config_path),
        transform=factory.get_pipeline("test"),
        class_to_idx=class_to_idx,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Se lee del DataFrame del propio dataset, no del CSV: si algún día CornDataset
    # filtra filas, la serie sigue alineada con lo que entrega el loader.
    frame = dataset.data_frame
    environments = (
        frame["environment"].reset_index(drop=True)
        if "environment" in frame.columns
        else pd.Series(dtype=str)
    )
    return loader, environments
