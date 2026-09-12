import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.config import PROJECT_ROOT
from src.data.dataset import CornDataset
from src.data.image_cache import ImageCache, build_cache

_CONFIG = str(PROJECT_ROOT / "config" / "dataset.yaml")


@pytest.fixture
def corpus(tmp_path):
    """Corpus mínimo en disco con una imagen por clase y su manifiesto."""
    root = tmp_path / "dataset"
    rows = []
    for index, label in enumerate(("healthy", "common_rust", "gray_leaf_spot")):
        relative = f"clean/{label}/real/{label}_src_real_{index}.jpg"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        colour = (index * 80, 255 - index * 60, 120)
        Image.new("RGB", (640, 480), colour).save(path, quality=95)
        rows.append({"image_path": relative, "label": label, "environment": "real"})

    manifest = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return root, manifest, [row["image_path"] for row in rows]


def test_la_cache_conserva_tamano_y_contenido(corpus, tmp_path):
    """Cada entrada vuelve como imagen RGB del lado declarado y con su color original."""
    root, _, paths = corpus
    destination = tmp_path / "cache" / "corpus"
    build_cache(paths, root, destination, side=64, workers=2)

    cache = ImageCache(destination)
    assert cache.side == 64
    assert all(path in cache for path in paths)

    for index, path in enumerate(paths):
        image = cache.get(path)
        assert image.size == (64, 64)
        esperado = np.array([index * 80, 255 - index * 60, 120])
        assert np.abs(np.asarray(image).mean(axis=(0, 1)) - esperado).max() < 8


def test_una_ruta_ausente_no_esta_en_la_cache(corpus, tmp_path):
    """La pertenencia se decide por el índice, no por el sistema de archivos."""
    root, _, paths = corpus
    destination = tmp_path / "cache" / "parcial"
    build_cache(paths[:1], root, destination, side=32, workers=1)

    cache = ImageCache(destination)
    assert paths[0] in cache
    assert paths[1] not in cache


def test_el_dataset_lee_de_la_cache_sin_tocar_el_disco(corpus, tmp_path, monkeypatch):
    """Con caché el dataset no vuelve a decodificar el JPEG original."""
    root, manifest, paths = corpus
    destination = tmp_path / "cache" / "corpus"
    build_cache(paths, root, destination, side=64, workers=2)

    monkeypatch.setenv("DATASET_ROOT", str(root))

    def explotar(path):
        raise AssertionError(f"se leyó del disco: {path}")

    monkeypatch.setattr("src.data.dataset.load_and_normalize_image", explotar)

    dataset = CornDataset(
        csv_path=str(manifest),
        config_path=_CONFIG,
        image_cache=ImageCache(destination),
    )
    imagen, etiqueta = dataset[0]

    assert imagen.size == (64, 64)
    assert etiqueta in dataset.idx_to_class
