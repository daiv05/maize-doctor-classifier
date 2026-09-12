"""Detección de casi-duplicados por hash perceptual.

La deduplicación por contenido que ya existe en el corpus sólo ve archivos idénticos byte a
byte. No detecta la misma foto reescalada, recomprimida o volteada, que es lo que ocurre
entre versiones de un mismo dataset y entre datasets que comparten origen. Si esas copias
caen a ambos lados de una partición, el conjunto de prueba deja de medir generalización.

Medido sobre el corpus limpio: con hash idéntico quedan 52 grupos repartidos entre
particiones, y con distancia de Hamming 4 hasta 242, aunque a esa distancia la tasa de falsos
positivos es alta en primeros planos de nervadura, donde un hash de 64 bits degenera.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def perceptual_hash(path: str | Path, side: int = 9) -> np.ndarray:
    """Calcula un hash perceptual de diferencia, empaquetado en bytes.

    @param {str|Path} path Ruta de la imagen.
    @param {int} side Ancho de la rejilla; produce (side - 1) * (side - 1) bits.
    @returns {np.ndarray} Hash empaquetado en uint8.
    """
    with Image.open(path) as raw:
        if raw.format == "JPEG":
            raw.draft("L", (128, 128))
        image = raw.convert("L").resize((side, side - 1), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.int16)
    return np.packbits((array[:, 1:] > array[:, :-1]).flatten())


def compute_hashes(paths, workers: int = 8) -> np.ndarray:
    """Hashea una colección de rutas en paralelo.

    @param {Iterable} paths Rutas absolutas de las imágenes.
    @param {int} workers Hilos de decodificación.
    @returns {np.ndarray} Vector de hashes de 64 bits sin signo.
    """
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        packed = list(pool.map(perceptual_hash, paths))
    return np.stack(packed).view(">u8").ravel()


def group_near_duplicates(hashes: np.ndarray, max_distance: int = 0) -> dict[int, list[int]]:
    """Agrupa índices cuyo hash difiere en como mucho ``max_distance`` bits.

    Con distancia cero basta agrupar por igualdad; para distancias mayores se generan
    candidatos por bandas de 16 bits y sólo esos pares se comparan.

    @param {np.ndarray} hashes Vector de hashes de 64 bits.
    @param {int} max_distance Distancia de Hamming máxima admitida.
    @returns {dict[int, list[int]]} Grupos con más de un miembro.
    """
    parent = list(range(len(hashes)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    band_count, band_width = (1, 64) if max_distance == 0 else (4, 16)
    for band in range(band_count):
        key = (hashes >> np.uint64(band * band_width)) & np.uint64((1 << band_width) - 1)
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, value in enumerate(key.tolist()):
            buckets[value].append(index)
        for members in buckets.values():
            if not 1 < len(members) <= 400:
                continue
            indices = np.asarray(members)
            values = hashes[indices]
            for offset in range(len(indices)):
                xor = np.bitwise_xor(values[offset], values[offset + 1:])
                if xor.size == 0:
                    continue
                distance = POPCOUNT[
                    np.frombuffer(xor.astype(">u8").tobytes(), dtype=np.uint8).reshape(-1, 8)
                ].sum(axis=1)
                for position in np.where(distance <= max_distance)[0]:
                    union(int(indices[offset]), int(indices[offset + 1 + position]))

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(hashes)):
        groups[find(index)].append(index)
    return {root: members for root, members in groups.items() if len(members) > 1}


def drop_near_duplicates(
    manifest: pd.DataFrame,
    dataset_root: Path,
    max_distance: int = 0,
    workers: int = 8,
    path_column: str = "image_path",
) -> tuple[pd.DataFrame, int]:
    """Conserva un representante por grupo de casi-duplicados.

    @param {pd.DataFrame} manifest Manifiesto con la columna de rutas relativas.
    @param {Path} dataset_root Raíz del dataset para resolver esas rutas.
    @param {int} max_distance Distancia de Hamming máxima; 0 exige hash idéntico.
    @param {int} workers Hilos de decodificación.
    @returns {tuple[pd.DataFrame, int]} Manifiesto filtrado y número de filas descartadas.
    """
    if manifest.empty:
        return manifest, 0
    paths = [str(dataset_root / relative) for relative in manifest[path_column]]
    hashes = compute_hashes(paths, workers)
    groups = group_near_duplicates(hashes, max_distance)

    discarded: set[int] = set()
    for members in groups.values():
        discarded.update(sorted(members)[1:])
    keep = [index for index in range(len(manifest)) if index not in discarded]
    return manifest.iloc[keep].reset_index(drop=True), len(discarded)
