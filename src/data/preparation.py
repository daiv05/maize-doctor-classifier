"""Utilidades reproducibles para artefactos de preparación del dataset.

Los hashes contractuales se calculan sobre bytes reales. Los CSV se serializan
en UTF-8, con ``\n`` y orden estable; los JSON usan claves ordenadas y se
reemplazan atómicamente después de ``flush`` + ``fsync``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.data.identity import normalize_sample_path

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Exclusion:
    """Exclusión lógica verificada por ruta relativa y contenido."""

    image_path: str
    sha256: str
    reason: str


def sha256_bytes(data: bytes) -> str:
    """Devuelve el SHA-256 hexadecimal de ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Calcula SHA-256 por bloques, sin cargar un artefacto completo en memoria."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serializa JSON canónico compacto, UTF-8, claves ordenadas y newline final."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def sha256_json(payload: Any) -> str:
    """Calcula el SHA-256 de la representación JSON canónica de ``payload``."""
    return sha256_bytes(canonical_json_bytes(payload))


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Escribe texto UTF-8 mediante temporal hermano y reemplazo atómico."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Escribe JSON estable de forma atómica, con formato legible y newline final."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return atomic_write_text(path, text)


def canonicalize_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sort_by: Sequence[str] = ("sample_id",),
) -> pd.DataFrame:
    """Selecciona columnas contractuales y ordena filas de manera estable."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas contractuales en el artefacto: {missing}")
    return (
        frame.loc[:, list(columns)]
        .sort_values(list(sort_by), kind="mergesort")
        .reset_index(drop=True)
    )


def write_canonical_csv(
    path: str | os.PathLike[str],
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sort_by: Sequence[str] = ("sample_id",),
) -> pd.DataFrame:
    """Escribe un CSV contractual en UTF-8, con columnas y filas estables."""
    canonical = canonicalize_frame(frame, columns, sort_by=sort_by)
    content = canonical.to_csv(index=False, lineterminator="\n")
    atomic_write_text(path, content)
    return canonical


def load_exclusions(
    path: str | os.PathLike[str] | None,
    discovered_paths: Iterable[str],
) -> dict[str, Exclusion]:
    """Carga y valida exclusiones contra las rutas descubiertas por el pipeline."""
    if path is None:
        return {}

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as error:
        raise ValueError("El CSV de exclusiones no contiene encabezados") from error

    required = ["image_path", "sha256", "reason"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"El CSV de exclusiones requiere columnas {required}; faltan {missing}")

    available = {normalize_sample_path(value) for value in discovered_paths}
    exclusions: dict[str, Exclusion] = {}
    for row_number, row in enumerate(frame[required].itertuples(index=False), start=2):
        try:
            image_path = normalize_sample_path(row.image_path)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Exclusión inválida en fila {row_number}: {error}") from error
        digest = row.sha256.strip().lower()
        reason = row.reason.strip()
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(
                f"Exclusión inválida en fila {row_number}: sha256 debe tener 64 hexadecimales"
            )
        if not reason:
            raise ValueError(
                f"Exclusión inválida en fila {row_number}: reason no puede estar vacío"
            )
        if image_path in exclusions:
            raise ValueError(f"El CSV de exclusiones repite image_path: {image_path}")
        if image_path not in available:
            raise ValueError(f"La exclusión no corresponde a una muestra descubierta: {image_path}")
        exclusions[image_path] = Exclusion(image_path, digest, reason)
    return exclusions


def validate_split_integrity(
    master_manifest: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
) -> None:
    """Exige splits disjuntos cuya unión sea exactamente el manifiesto maestro."""
    master_ids = set(master_manifest["sample_id"].astype(str))
    split_ids: dict[str, set[str]] = {}
    for name, frame in splits.items():
        ids = frame["sample_id"].astype(str)
        if ids.duplicated().any():
            raise ValueError(f"El split {name} contiene sample_id duplicados")
        split_ids[name] = set(ids)

    names = list(split_ids)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = split_ids[left_name] & split_ids[right_name]
            if overlap:
                preview = sorted(overlap)[:3]
                raise ValueError(
                    f"Los splits {left_name} y {right_name} comparten sample_id: {preview}"
                )

    union = set().union(*split_ids.values()) if split_ids else set()
    if union != master_ids:
        missing = sorted(master_ids - union)[:3]
        unknown = sorted(union - master_ids)[:3]
        raise ValueError(
            "La unión de splits no coincide con master_manifest "
            f"(faltantes={missing}, desconocidos={unknown})"
        )


def build_split_audit_report(
    master_manifest: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Construye distribución larga por split, clase, entorno y procedencia disponible."""
    dimensions = ["label", "environment"]
    source_lookup = None
    if "source_id" in master_manifest.columns:
        dimensions.append("source_id")
        source_lookup = master_manifest.set_index("sample_id")["source_id"]

    parts: list[pd.DataFrame] = []
    for split_name, split_frame in splits.items():
        audit_frame = split_frame.copy()
        if source_lookup is not None and "source_id" not in audit_frame.columns:
            audit_frame["source_id"] = audit_frame["sample_id"].map(source_lookup)
        counts = audit_frame.groupby(dimensions, dropna=False).size().rename("count").reset_index()
        counts.insert(0, "split", split_name)
        parts.append(counts)

    report = pd.concat(parts, ignore_index=True)
    report["count"] = report["count"].astype(int)
    return report.sort_values(["split", *dimensions], kind="mergesort").reset_index(drop=True)
