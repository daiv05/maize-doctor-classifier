"""Identidad estable de muestras basada únicamente en su ruta lógica relativa.

La representación canónica reemplaza ``\\`` por ``/``, elimina separadores
redundantes y componentes ``.`` mediante :class:`pathlib.PurePosixPath`, y se
serializa con ``as_posix()``. Las rutas absolutas, con unidad de Windows o con
componentes ``..`` se rechazan antes de normalizar. El ``sample_id`` es el
SHA-256 hexadecimal de esa representación codificada en UTF-8.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import PurePosixPath, PureWindowsPath
from typing import Generic, Iterable, TypeVar

import pandas as pd
from torch.utils.data import Subset

T = TypeVar("T")


def normalize_sample_path(path: str | os.PathLike[str]) -> str:
    """Devuelve la representación POSIX canónica de una ruta lógica relativa."""
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError("image_path debe ser texto o PathLike[str]")
    if not raw.strip():
        raise ValueError("image_path no puede estar vacío")

    # PurePosixPath no reconoce ``C:\\...`` como absoluto en Linux. Validar con
    # ambas semánticas antes de unificar separadores evita identidades ligadas a
    # la ubicación local de Windows o POSIX.
    windows_path = PureWindowsPath(raw)
    posix_text = raw.replace("\\", "/")
    posix_path = PurePosixPath(posix_text)
    if windows_path.drive or windows_path.is_absolute() or posix_path.is_absolute():
        raise ValueError(f"image_path debe ser relativo: {raw!r}")

    raw_parts = [part for part in posix_text.split("/") if part not in ("", ".")]
    if ".." in raw_parts:
        raise ValueError(f"image_path no puede contener '..': {raw!r}")

    normalized = posix_path.as_posix()
    if normalized in ("", "."):
        raise ValueError("image_path debe identificar un archivo, no '.'")
    return normalized


def sample_id_for_path(path: str | os.PathLike[str]) -> str:
    """Calcula ``SHA256(normalize_sample_path(path).encode('utf-8'))``."""
    normalized = normalize_sample_path(path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_sample_ids(frame: pd.DataFrame) -> None:
    """Valida presencia, correspondencia y unicidad de ``sample_id`` en un manifiesto."""
    if "image_path" not in frame.columns:
        raise ValueError("El manifiesto no contiene la columna 'image_path'")
    if "sample_id" not in frame.columns:
        raise ValueError("El manifiesto no contiene la columna 'sample_id'")
    if frame["image_path"].isna().any():
        raise ValueError("El manifiesto contiene image_path vacío")
    if frame["sample_id"].isna().any():
        raise ValueError("El manifiesto contiene sample_id vacío")

    normalized_paths = frame["image_path"].map(normalize_sample_path)
    sample_ids = frame["sample_id"].astype(str)
    if sample_ids.str.strip().eq("").any():
        raise ValueError("El manifiesto contiene sample_id vacío")

    expected_ids = normalized_paths.map(sample_id_for_path)
    mismatches = sample_ids.ne(expected_ids)
    if mismatches.any():
        row = frame.index[mismatches][0]
        raise ValueError(
            f"sample_id no corresponde al SHA-256 de image_path normalizado en la fila {row}"
        )

    identity_pairs = pd.DataFrame(
        {"sample_id": sample_ids, "normalized_path": normalized_paths}, index=frame.index
    )
    paths_per_id = identity_pairs.groupby("sample_id")["normalized_path"].nunique()
    if paths_per_id.gt(1).any():
        raise ValueError("Dos rutas diferentes producen el mismo sample_id")
    if sample_ids.duplicated().any():
        raise ValueError("El manifiesto contiene sample_id duplicado")


def ensure_sample_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Copia el manifiesto, genera IDs ausentes y valida siempre los existentes."""
    result = frame.copy()
    if "image_path" not in result.columns:
        raise ValueError("El manifiesto no contiene la columna 'image_path'")
    if result["image_path"].isna().any():
        raise ValueError("El manifiesto contiene image_path vacío")
    expected = result["image_path"].map(sample_id_for_path)
    if "sample_id" not in result.columns:
        result.insert(0, "sample_id", expected)
    validate_sample_ids(result)
    return result


class IdentifiedValues(list[T], Generic[T]):
    """Lista compatible con consumidores actuales que conserva IDs de inferencia."""

    def __init__(self, values: Iterable[T] = (), *, sample_ids: list[str] | None = None) -> None:
        super().__init__(values)
        self.sample_ids = None if sample_ids is None else list(sample_ids)


def unpack_batch(batch):
    """Acepta batches ``(images, labels)`` o ``(images, labels, sample_ids)``."""
    if not isinstance(batch, (list, tuple)) or len(batch) not in (2, 3):
        raise ValueError("Un batch debe contener images, labels y opcionalmente sample_ids")
    images, labels = batch[0], batch[1]
    if len(batch) == 2:
        return images, labels, None

    raw_ids = batch[2]
    sample_ids = [raw_ids] if isinstance(raw_ids, str) else [str(value) for value in raw_ids]
    if len(sample_ids) != len(labels):
        raise ValueError("El batch no contiene un sample_id por etiqueta")
    if any(not value.strip() for value in sample_ids):
        raise ValueError("El batch contiene sample_id vacío")
    return images, labels, sample_ids


def sample_ids_from(values, *, field_name: str = "valores") -> list[str]:
    """Obtiene los IDs transportados por una :class:`IdentifiedValues`."""
    sample_ids = getattr(values, "sample_ids", None)
    if sample_ids is None:
        raise ValueError(f"{field_name} no conserva sample_id del DataLoader")
    if len(sample_ids) != len(values):
        raise ValueError(f"{field_name} tiene una cantidad inconsistente de sample_id")
    return list(sample_ids)


def dataset_manifest(dataset) -> pd.DataFrame | None:
    """Recupera el manifiesto de un dataset, respetando el orden de ``Subset``."""
    if isinstance(dataset, Subset):
        base = dataset_manifest(dataset.dataset)
        if base is None:
            return None
        return base.iloc[list(dataset.indices)].reset_index(drop=True)
    frame = getattr(dataset, "data_frame", None)
    return None if frame is None else ensure_sample_ids(frame)


def align_manifest_to_sample_ids(frame: pd.DataFrame, sample_ids: Iterable[str]) -> pd.DataFrame:
    """Ordena/repite filas del manifiesto según los IDs observados por inferencia."""
    manifest = ensure_sample_ids(frame)
    ordered_ids = [str(value) for value in sample_ids]
    if any(not value.strip() for value in ordered_ids):
        raise ValueError("La inferencia contiene sample_id vacío")
    indexed = manifest.set_index("sample_id", drop=False)
    unknown = set(ordered_ids) - set(indexed.index)
    if unknown:
        preview = sorted(unknown)[:3]
        raise ValueError(f"La inferencia referencia sample_id desconocidos: {preview}")
    return indexed.loc[ordered_ids].reset_index(drop=True)
