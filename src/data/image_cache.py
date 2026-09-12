"""Caché de imágenes predecodificadas para acelerar el entrenamiento.

Medido sobre una A10 con ocho núcleos: 225 imágenes por segundo y la GPU al 2 %. El cuello
no es el cómputo sino decodificar JPEG de varios megapíxeles para quedarse con 224 píxeles
de lado. La caché hace esa decodificación una sola vez.

La imagen se guarda ya corregida por EXIF y reescalada a un lado fijo, así que el pipeline
de transformaciones recibe lo mismo que recibiría del disco salvo por un detalle: el
reescalado ocurre en dos pasos (original a ``side``, luego ``side`` a 224) en lugar de uno.
Con ``side`` por encima del tamaño de entrada la diferencia es pequeña, pero existe, así que
la caché es opcional y las corridas que la usan deben declararlo.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.loader import load_and_normalize_image

DEFAULT_SIDE = 256


def build_cache(
    image_paths: list[str],
    dataset_root: Path,
    destination: Path,
    side: int = DEFAULT_SIDE,
    workers: int = 8,
) -> Path:
    """Decodifica y reescala las imágenes una sola vez sobre un memmap.

    @param {list[str]} image_paths Rutas relativas a la raíz del dataset.
    @param {Path} dataset_root Raíz del dataset.
    @param {Path} destination Ruta base; se escriben un .npy y un .json de índice.
    @param {int} side Lado de la imagen cacheada.
    @param {int} workers Hilos de decodificación.
    @returns {Path} Ruta del memmap generado.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    array_path = destination.with_suffix(".npy")
    memmap = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.uint8, shape=(len(image_paths), side, side, 3)
    )

    def store(pair: tuple[int, str]) -> str | None:
        index, relative = pair
        try:
            image = load_and_normalize_image(dataset_root / relative)
            memmap[index] = np.asarray(
                image.resize((side, side), Image.Resampling.BILINEAR), dtype=np.uint8
            )
            return None
        except Exception as error:  # noqa: BLE001 - se reporta, no se oculta
            return f"{relative}: {error}"

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        failures = [message for message in pool.map(store, enumerate(image_paths)) if message]
    memmap.flush()

    destination.with_suffix(".json").write_text(
        json.dumps({"side": side, "index": {path: position
                                            for position, path in enumerate(image_paths)},
                    "failures": failures}, ensure_ascii=False),
        encoding="utf-8",
    )
    if failures:
        print(f"[!] {len(failures)} imágenes no se pudieron cachear", flush=True)
    return array_path


class ImageCache:
    """Lectura de una caché construida por :func:`build_cache`."""

    def __init__(self, destination: Path):
        """
        @param {Path} destination Ruta base usada al construir la caché.
        """
        meta = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
        self.side = int(meta["side"])
        self.index: dict[str, int] = meta["index"]
        self.array = np.load(destination.with_suffix(".npy"), mmap_mode="r")

    def __contains__(self, image_path: str) -> bool:
        return image_path in self.index

    def get(self, image_path: str) -> Image.Image:
        """Devuelve la imagen cacheada como PIL, lista para el pipeline de transformaciones.

        @param {str} image_path Ruta relativa a la raíz del dataset.
        @returns {Image.Image} Imagen RGB del lado cacheado.
        """
        return Image.fromarray(np.array(self.array[self.index[image_path]]))
