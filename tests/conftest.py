"""Fixtures compartidas por la suite de tests."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from PIL import Image

_CLASS_DISTRIBUTION = {
    "healthy": 12,
    "common_rust": 6,
    "potassium_deficiency": 2,
}


@pytest.fixture
def fake_image_root(tmp_path: Path) -> Path:
    """
    Crea un árbol clean/<clase>/<entorno>/ con imágenes RGB de 32x32.

    @param {Path} tmp_path Directorio temporal provisto por pytest.
    @returns {Path} Raíz del dataset sintético.
    """
    rng = np.random.default_rng(42)
    root = tmp_path / "dataset"
    for class_name, count in _CLASS_DISTRIBUTION.items():
        for index in range(count):
            environment = "lab" if index % 2 == 0 else "real"
            directory = root / "clean" / class_name / environment
            directory.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / f"{class_name}_{index}.png")
    return root


@pytest.fixture
def tmp_splits_dir(tmp_path: Path, fake_image_root: Path) -> Path:
    """
    Genera train/val/test.csv apuntando a las imágenes de `fake_image_root`.

    @param {Path} tmp_path Directorio temporal provisto por pytest.
    @param {Path} fake_image_root Raíz del dataset sintético.
    @returns {Path} Directorio con los tres manifiestos.
    """
    rows = []
    for image_path in sorted((fake_image_root / "clean").rglob("*.png")):
        relative = image_path.relative_to(fake_image_root)
        rows.append(
            {
                "image_path": relative.as_posix(),
                "label": image_path.parent.parent.name,
                "environment": image_path.parent.name,
            }
        )
    data_frame = pd.DataFrame(rows)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for split_name in ("train", "val", "test"):
        data_frame.to_csv(splits_dir / f"{split_name}.csv", index=False)
    return splits_dir
