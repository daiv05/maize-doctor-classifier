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

from src.data.identity import normalize_sample_path, sample_id_for_path

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


def load_group_manifest(
    path: str | os.PathLike[str] | None,
    eligible_manifest: pd.DataFrame,
) -> dict[str, str]:
    """Carga grupos explícitos y los resuelve contra las muestras todavía elegibles.

    Cada fila puede identificar la muestra por ``sample_id`` o por ``image_path``. Si
    proporciona ambos, deben describir la misma identidad. Las filas que apuntan a una
    muestra excluida, deduplicada o desconocida se rechazan para no aceptar manifiestos
    obsoletos silenciosamente.
    """
    if path is None:
        return {}

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as error:
        raise ValueError("El CSV de grupos no contiene encabezados") from error

    if "group_id" not in frame.columns:
        raise ValueError("El CSV de grupos requiere la columna 'group_id'")
    identity_columns = [column for column in ("sample_id", "image_path") if column in frame.columns]
    if not identity_columns:
        raise ValueError("El CSV de grupos requiere 'sample_id' o 'image_path'")

    eligible_ids = set(eligible_manifest["sample_id"].astype(str))
    eligible_paths = {
        normalize_sample_path(image_path): str(sample_id)
        for sample_id, image_path in eligible_manifest[["sample_id", "image_path"]].itertuples(
            index=False, name=None
        )
    }
    groups: dict[str, str] = {}

    for row_number, row in enumerate(frame.itertuples(index=False), start=2):
        values = row._asdict()
        group_id = values["group_id"].strip()
        if not group_id:
            raise ValueError(
                f"Manifiesto de grupos inválido en fila {row_number}: group_id no puede estar vacío"
            )

        raw_sample_id = values.get("sample_id", "").strip()
        raw_image_path = values.get("image_path", "").strip()
        if not raw_sample_id and not raw_image_path:
            raise ValueError(
                f"Manifiesto de grupos inválido en fila {row_number}: "
                "se requiere sample_id o image_path"
            )

        if raw_sample_id and not _SHA256_PATTERN.fullmatch(raw_sample_id):
            raise ValueError(
                f"Manifiesto de grupos inválido en fila {row_number}: "
                "sample_id debe tener 64 hexadecimales minúsculos"
            )

        normalized_path = None
        path_sample_id = None
        if raw_image_path:
            try:
                normalized_path = normalize_sample_path(raw_image_path)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Manifiesto de grupos inválido en fila {row_number}: {error}"
                ) from error
            path_sample_id = sample_id_for_path(normalized_path)

        if raw_sample_id and path_sample_id and raw_sample_id != path_sample_id:
            raise ValueError(
                f"Manifiesto de grupos ambiguo en fila {row_number}: sample_id e image_path "
                "identifican muestras diferentes"
            )

        sample_id = raw_sample_id or path_sample_id
        assert sample_id is not None
        if sample_id not in eligible_ids:
            raise ValueError(
                f"El manifiesto de grupos referencia una muestra no elegible: {sample_id}"
            )
        if normalized_path is not None and eligible_paths.get(normalized_path) != sample_id:
            raise ValueError(
                f"El manifiesto de grupos referencia un image_path no elegible: {normalized_path}"
            )

        previous = groups.get(sample_id)
        if previous is not None and previous != group_id:
            raise ValueError(
                f"La muestra {sample_id} tiene dos group_id diferentes: {previous!r} y {group_id!r}"
            )
        groups[sample_id] = group_id

    return groups


def apply_effective_groups(
    manifest: pd.DataFrame,
    explicit_groups: Mapping[str, str],
) -> pd.DataFrame:
    """Aplica prioridad grupo explícito > procedencia > identidad individual.

    ``group_origin`` conserva de dónde salió la decisión y ``effective_group_id`` mantiene
    literalmente el ID resuelto. Así, dos filas con el mismo ID lógico siguen perteneciendo
    al mismo grupo aunque una lo declare explícitamente y otra lo obtenga por procedencia.
    """
    result = manifest.copy()
    if "source_id" not in result.columns:
        result["source_id"] = None

    result["explicit_group_id"] = result["sample_id"].map(explicit_groups)
    has_explicit = result["explicit_group_id"].notna()
    has_source = result["source_id"].notna() & result["source_id"].astype(str).str.strip().ne("")

    result["group_origin"] = "individual"
    result.loc[has_source, "group_origin"] = "inferred"
    result.loc[has_explicit, "group_origin"] = "explicit"

    result["group_id"] = result["sample_id"].astype(str)
    result.loc[has_source, "group_id"] = result.loc[has_source, "source_id"].astype(str)
    result.loc[has_explicit, "group_id"] = result.loc[has_explicit, "explicit_group_id"].astype(str)
    result["effective_group_id"] = result["group_id"].astype(str)
    return result


def validate_split_integrity(
    master_manifest: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
) -> dict[str, int]:
    """Exige cobertura exacta y cero fuga por identidad, contenido o grupo efectivo."""
    if master_manifest["sample_id"].astype(str).duplicated().any():
        raise ValueError("master_manifest contiene sample_id duplicados")

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

    indexed_master = master_manifest.set_index("sample_id")
    overlap_metrics = {"sample_overlap_count": 0}
    for column, metric_name in (
        ("sha256", "sha256_overlap_count"),
        ("effective_group_id", "group_overlap_count"),
    ):
        if column not in master_manifest.columns:
            continue
        empty_values = master_manifest[column].astype(str).str.strip().eq("").any()
        if master_manifest[column].isna().any() or empty_values:
            raise ValueError(f"master_manifest contiene {column} vacío")

        values_by_split = {
            name: set(indexed_master.loc[sorted(ids), column].astype(str))
            for name, ids in split_ids.items()
        }
        leaked: set[str] = set()
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                leaked.update(values_by_split[left_name] & values_by_split[right_name])
        overlap_metrics[metric_name] = len(leaked)
        if leaked:
            preview = sorted(leaked)[:3]
            raise ValueError(
                f"Los splits comparten valores de {column}: {preview} (total={len(leaked)})"
            )
    return overlap_metrics


def build_split_audit_report(
    master_manifest: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Construye distribución por split, clase, entorno, procedencia y grupo."""
    dimensions = ["label", "environment"]
    group_dimensions = [
        column
        for column in (
            "source_id",
            "explicit_group_id",
            "group_id",
            "effective_group_id",
            "group_origin",
        )
        if column in master_manifest.columns
    ]
    dimensions.extend(group_dimensions)
    lookups = master_manifest.set_index("sample_id")[group_dimensions]

    parts: list[pd.DataFrame] = []
    for split_name, split_frame in splits.items():
        audit_frame = split_frame.copy()
        for column in group_dimensions:
            if column not in audit_frame.columns:
                audit_frame[column] = audit_frame["sample_id"].map(lookups[column])
        counts = audit_frame.groupby(dimensions, dropna=False).size().rename("count").reset_index()
        counts.insert(0, "split", split_name)
        parts.append(counts)

    report = pd.concat(parts, ignore_index=True)
    report["count"] = report["count"].astype(int)
    return report.sort_values(["split", *dimensions], kind="mergesort").reset_index(drop=True)


def build_group_split_summary(
    master_manifest: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
    target_ratios: Mapping[str, float],
    overlap_metrics: Mapping[str, int],
) -> dict[str, Any]:
    """Resume cobertura de grupos, tamaños objetivo y grupos que explican desvíos."""
    total = len(master_manifest)
    summary: dict[str, Any] = {
        "total_groups": int(master_manifest["effective_group_id"].nunique()),
        "explicit_groups": int(
            master_manifest.loc[
                master_manifest["group_origin"].eq("explicit"), "effective_group_id"
            ].nunique()
        ),
        "inferred_groups": int(
            master_manifest.loc[
                master_manifest["group_origin"].eq("inferred"), "effective_group_id"
            ].nunique()
        ),
        "ungrouped_samples": int(master_manifest["group_origin"].eq("individual").sum()),
        **{key: int(value) for key, value in overlap_metrics.items()},
    }

    master_by_id = master_manifest.set_index("sample_id")
    group_rows: list[dict[str, Any]] = []
    for split_name, frame in splits.items():
        split_master = master_by_id.loc[frame["sample_id"].astype(str)]
        actual = len(frame)
        target = float(target_ratios[split_name]) * total
        summary[f"{split_name}_groups"] = int(split_master["effective_group_id"].nunique())
        summary[f"target_{split_name}_size"] = target
        summary[f"actual_{split_name}_size"] = actual
        summary[f"{split_name}_size_deviation"] = actual - target

        if actual > target:
            counts = split_master["effective_group_id"].value_counts()
            for effective_group_id, count in counts.head(5).items():
                group = split_master[split_master["effective_group_id"].eq(effective_group_id)]
                first = group.iloc[0]
                group_rows.append(
                    {
                        "split": split_name,
                        "group_id": str(first["group_id"]),
                        "effective_group_id": str(effective_group_id),
                        "group_origin": str(first["group_origin"]),
                        "samples": int(count),
                    }
                )
    summary["groups_responsible_for_size_deviation"] = group_rows
    return summary
