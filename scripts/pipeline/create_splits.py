import argparse
import io
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from src.config import get_dataset_root, get_output_root
from src.data.deduplicate import drop_near_duplicates
from src.data.identity import ensure_sample_ids, sample_id_for_path
from src.data.preparation import (
    Exclusion,
    apply_effective_groups,
    atomic_write_json,
    build_group_split_summary,
    build_split_audit_report,
    load_exclusions,
    load_group_manifest,
    sha256_bytes,
    sha256_file,
    sha256_json,
    validate_split_integrity,
    write_canonical_csv,
)
from src.data.provenance import provenance_from_path
from src.data.splitter import HierarchicalStratifiedSplitter, SourceGroupedSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# sklearn exige >=2 muestras por estrato en cada corte del doble split; con 70/15/15
# eso se garantiza a partir de ~7 imágenes por estrato label+environment.
_MIN_STRATUM_IMAGES = 7
_SPLIT_NAMES = ("train", "val", "test")
_MASTER_COLUMNS = ("sample_id", "image_path", "label", "environment", "sha256")
_SPLIT_COLUMNS = ("sample_id", "image_path", "label", "environment")
_GROUP_COLUMNS = (
    "source_id",
    "explicit_group_id",
    "group_id",
    "effective_group_id",
    "group_origin",
)
_TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class _ImageInspection:
    """Resultado de una única lectura para SHA-256 y validación PIL."""

    sha256: str
    valid: bool
    error: str | None = None


def _resolve_index_workers() -> int:
    """Nº de hilos para el indexado. Override por `SPLITS_INDEX_WORKERS` para alinearlo con la
    CPU asignada en entornos con cuota (p.ej. Modal, donde `os.cpu_count()` reporta los cores del
    HOST, no la asignación del contenedor - sin override lanzaría demasiados hilos a ciegas).
    Fallback: escala con los cores locales, acotado a 32."""
    raw = os.getenv("SPLITS_INDEX_WORKERS", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n >= 1:
            return n
        logger.warning(f"SPLITS_INDEX_WORKERS inválido ({raw!r}); usando el default por cores.")
    return min(32, (os.cpu_count() or 4) * 4)


def _verify_and_hash(abs_path: Path) -> _ImageInspection:
    """Lee el archivo una sola vez; valida integridad PIL y calcula el SHA-256.

    Siempre conserva el digest de los bytes si pudieron leerse, incluso cuando PIL
    rechaza la imagen. Así una exclusión de un archivo inválido también puede quedar
    fijada por contenido sin una segunda lectura.
    """
    try:
        data = abs_path.read_bytes()
    except OSError as error:
        return _ImageInspection(sha256="", valid=False, error=str(error))

    digest = sha256_bytes(data)
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return _ImageInspection(sha256=digest, valid=True)
    except Exception as error:  # noqa: BLE001 - cualquier fallo PIL = imagen inválida
        return _ImageInspection(sha256=digest, valid=False, error=str(error))


def _cap_manifest_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Limita cada clase a lo sumo `max_per_class` imágenes (muestreo aleatorio reproducible).

    Clases con menos imágenes que el límite quedan intactas (p.ej. nitrogen_deficiency).
    El muestreo es proporcional por `environment` (resto mayor) para conservar el balance
    lab/real de cada clase.

    Nota: se itera el groupby en vez de usar `.apply(lambda g: ...)` porque, al agrupar por
    el nombre literal de una columna, pandas >= 2.2 puede excluir esa columna del grupo
    pasado a la función (comportamiento por defecto desde pandas 3.0), lo que rompía el
    split posterior al perder la columna "label".
    """
    parts = []
    for _, group in df.groupby("label"):
        if len(group) <= max_per_class:
            parts.append(group)
            continue

        # Cuotas proporcionales por entorno con método del resto mayor (suman exacto).
        env_sizes = group["environment"].value_counts()
        quotas = env_sizes * max_per_class / len(group)
        base = quotas.astype(int)
        remainders = (quotas - base).sort_values(ascending=False)
        for env in remainders.index[: max_per_class - int(base.sum())]:
            base[env] += 1

        for env, n in base.items():
            if n > 0:
                env_group = group[group["environment"] == env]
                parts.append(env_group.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def _split_output_dir(base: Path, suffix: str | None = None) -> Path:
    if suffix:
        return base.parent / (base.name + f"_{suffix}")
    return base


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Cuenta una columna con claves estables y una categoría explícita para nulos."""
    if column not in frame.columns:
        return {}
    values = frame[column].fillna("unknown").astype(str)
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def _conflict_error(conflicts: dict[str, list[dict[str, str]]]) -> ValueError:
    """Construye un error explícito con identidad, ruta y etiqueta de cada conflicto."""
    lines = ["Contenido idéntico con etiquetas conflictivas:"]
    for digest, samples in sorted(conflicts.items()):
        lines.append(f"sha256={digest}")
        for sample in samples:
            lines.append(
                "  sample_id={sample_id} image_path={image_path} label={label}".format(**sample)
            )
    return ValueError("\n".join(lines))


def _audit_payload(
    *,
    seed: int,
    samples_discovered: int,
    samples_valid: int,
    exclusions: list[dict[str, str]],
    exact_duplicates: int,
    conflicts: dict[str, list[dict[str, str]]],
    invalid_images: list[dict[str, str]],
    class_cap_dropped: int = 0,
    perceptual_duplicates: int = 0,
    master_manifest: pd.DataFrame | None = None,
    group_split_summary: dict | None = None,
    perceptual_validation_enabled: bool = False,
    status: str = "complete",
) -> dict:
    """Crea el contenido estable de ``preparation_audit.json``."""
    conflict_samples = sum(max(len(samples) - 1, 0) for samples in conflicts.values())
    payload: dict = {
        "schema_version": 1,
        "status": status,
        "seed": seed,
        "samples_discovered": samples_discovered,
        "samples_valid": samples_valid,
        "samples_excluded": len(exclusions),
        "exact_duplicates": exact_duplicates,
        "label_conflicts": len(conflicts),
        "label_conflict_samples": conflict_samples,
        "invalid_images": len(invalid_images),
        "class_cap_dropped": class_cap_dropped,
        "perceptual_duplicates": perceptual_duplicates,
        "exclusions_by_reason": dict(
            sorted(Counter(item["reason"] for item in exclusions).items())
        ),
        "excluded_samples": exclusions,
        "invalid_samples": invalid_images,
        "label_conflict_details": [
            {"sha256": digest, "samples": samples} for digest, samples in sorted(conflicts.items())
        ],
        "artifacts": {"preparation_audit": "preparation_audit.json"},
    }
    if master_manifest is not None:
        payload["artifacts"].update(
            {
                "master_manifest": "master_manifest.csv",
                "train": "train.csv",
                "val": "val.csv",
                "test": "test.csv",
                "split_audit_report": "split_audit_report.csv",
                "manifest_lock": "manifest.lock.json",
            }
        )
        payload.update(
            {
                "samples_eligible": int(len(master_manifest)),
                "distribution_by_label": _distribution(master_manifest, "label"),
                "distribution_by_environment": _distribution(master_manifest, "environment"),
                "distribution_by_source": _distribution(master_manifest, "source_id"),
                "grouping": group_split_summary or {},
                "perceptual_overlap_count": 0 if perceptual_validation_enabled else None,
                "perceptual_leakage_validation": (
                    "global_pre_split_deduplication"
                    if perceptual_validation_enabled
                    else "not_requested"
                ),
            }
        )
        if group_split_summary:
            payload.update(group_split_summary)
    return payload


def _split_parameters(
    *,
    seed: int,
    allowed_classes: list[str],
    max_per_class: int | None,
    group_by_source: bool,
    deduplicate: bool,
    dedup_distance: int,
    allow_incomplete: bool,
    group_manifest_sha256: str | None,
    split_strategy: str,
) -> dict:
    """Parámetros semánticos que determinan membresía de los splits."""
    return {
        "seed": seed,
        "classes": list(allowed_classes),
        "max_per_class": max_per_class,
        "group_by_source": group_by_source,
        "deduplicate_perceptual": deduplicate,
        "dedup_distance": dedup_distance,
        "allow_incomplete_splits": allow_incomplete,
        "group_manifest_sha256": group_manifest_sha256,
        "split_strategy": split_strategy,
        "grouping_column": (
            "effective_group_id" if split_strategy != "stratified_label_environment" else None
        ),
        "stratify_columns": (
            ["label", "environment"] if split_strategy == "stratified_label_environment" else []
        ),
        "ratios": dict(_TARGET_RATIOS),
    }


def _validate_exclusion(
    exclusion: Exclusion,
    inspection: _ImageInspection,
) -> None:
    """Verifica que la exclusión siga apuntando exactamente al contenido declarado."""
    if inspection.sha256 != exclusion.sha256:
        actual = inspection.sha256 or "unavailable"
        raise ValueError(
            "El contenido de la exclusión cambió: "
            f"image_path={exclusion.image_path} "
            f"sha256_declarado={exclusion.sha256} sha256_actual={actual}"
        )


def run_data_preparation_pipeline(
    config_path: str,
    baseline: bool = False,
    classes: list[str] | None = None,
    max_per_class: int | None = None,
    no_cap: bool = False,
    group_by_source: bool = False,
    deduplicate: bool = False,
    dedup_distance: int = 0,
    allow_incomplete: bool = False,
    exclusions: str | Path | None = None,
    group_manifest: str | Path | None = None,
) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if exclusions is None:
        configured_exclusions = config.get("paths", {}).get("exclusions_file")
        if configured_exclusions:
            exclusions = Path(configured_exclusions)
            if not exclusions.is_absolute():
                exclusions = Path(config_path).resolve().parent / exclusions

    dataset_root = get_dataset_root()
    if not dataset_root.exists():
        raise SystemExit(
            f"DATASET_ROOT no encontrado: {dataset_root}. Verifica DATASET_ROOT en .env"
        )

    baseline_cfg = config.get("baseline", {}) if baseline else {}
    allowed_classes = classes or baseline_cfg.get("classes") or config["dataset"]["classes"]
    if no_cap:
        max_per_class = None
    else:
        max_per_class = (
            max_per_class if max_per_class is not None else baseline_cfg.get("max_images_per_class")
        )

    clean_dir = dataset_root / config["paths"]["raw_dir"]
    base_output_dir = get_output_root() / config["paths"]["split_output_dir"]
    suffix_parts = []
    if baseline:
        suffix_parts.append("baseline")
    if group_by_source:
        suffix_parts.append("source_grouped")
    output_dir = _split_output_dir(base_output_dir, suffix="_".join(suffix_parts) or None)
    seed = config["dataset"]["seed"]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Escaneando directorios para calcular la carga de trabajo...")
    raw_image_paths: list[tuple[str, str, Path, str]] = []

    # sorted(): la dedup conserva la primera copia vista; sin orden estable el manifiesto
    # variaría entre máquinas pese al seed.
    for class_name in sorted(os.listdir(clean_dir)):
        if class_name not in allowed_classes:
            continue
        class_path = clean_dir / class_name
        if not class_path.is_dir():
            continue

        for environment in sorted(os.listdir(class_path)):
            env_path = class_path / environment
            if environment not in ("real", "lab") or not env_path.is_dir():
                continue

            for img_name in sorted(os.listdir(env_path)):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    abs_path = env_path / img_name
                    rel_path = abs_path.relative_to(dataset_root).as_posix()
                    raw_image_paths.append((class_name, environment, abs_path, rel_path))

    if not raw_image_paths:
        raise ValueError(f"El pipeline no pudo indexar ninguna imagen válida en '{clean_dir}'.")

    exclusion_by_path = load_exclusions(exclusions, (record[3] for record in raw_image_paths))
    all_records: list[dict] = []
    seen_hashes: dict[str, dict[str, str]] = {}
    excluded_samples: list[dict[str, str]] = []
    invalid_samples: list[dict[str, str]] = []
    label_conflicts: dict[str, list[dict[str, str]]] = {}
    duplicates_found = 0
    corrupt_found = 0
    valid_found = 0

    logger.info(
        f"Indexando {len(raw_image_paths)} imágenes con verificación SHA-256 y validación PIL..."
    )

    # Fase 1 (paralela, I/O-bound): validar + hashear cada imagen. El resultado por archivo
    # es independiente del orden, así que se calcula concurrentemente para ocultar la latencia
    # de disco - crítico en volúmenes remotos (p.ej. el de Modal), donde cada lectura es lenta.
    # Se usan hilos (no procesos): el trabajo es I/O + C de hashlib/PIL, que liberan el GIL,
    # y así se evita el coste de serializar rutas/bytes entre procesos.
    max_workers = _resolve_index_workers()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(
            tqdm(
                pool.map(_verify_and_hash, (rec[2] for rec in raw_image_paths)),
                total=len(raw_image_paths),
                desc="Indexando",
                unit="img",
            )
        )

    # Fase 2 (secuencial, en el mismo orden sorted() del escaneo): dedup determinista. Con los
    # digests ya calculados, conservar la primera copia vista sigue siendo reproducible entre
    # máquinas, idéntico al comportamiento previo - solo que ahora sin el cuello de botella serial.
    for (class_name, environment, _abs_path, rel_path), inspection in zip(raw_image_paths, results):
        sample = {
            "sample_id": sample_id_for_path(rel_path),
            "image_path": rel_path,
            "label": class_name,
        }
        exclusion = exclusion_by_path.get(rel_path)
        if exclusion is not None:
            _validate_exclusion(exclusion, inspection)
            excluded_samples.append(
                {
                    **sample,
                    "sha256": inspection.sha256,
                    "reason": exclusion.reason,
                }
            )
            if not inspection.valid:
                corrupt_found += 1
                invalid_samples.append(
                    {
                        **sample,
                        "sha256": inspection.sha256,
                        "error": inspection.error or "error de lectura desconocido",
                    }
                )
            else:
                valid_found += 1
            continue
        if not inspection.valid:
            error = inspection.error or "error de lectura desconocido"
            tqdm.write(f"Imagen corrupta o ilegible, omitida: {rel_path} - {error}")
            corrupt_found += 1
            invalid_samples.append(
                {
                    **sample,
                    "sha256": inspection.sha256,
                    "error": error,
                }
            )
            continue

        valid_found += 1
        record = {
            **sample,
            "environment": environment,
            "sha256": inspection.sha256,
        }
        previous = seen_hashes.get(inspection.sha256)
        if previous is not None:
            if previous["label"] != class_name:
                conflict_samples = label_conflicts.setdefault(inspection.sha256, [previous])
                conflict_samples.append(sample)
                continue
            logger.warning(f"Duplicado exacto detectado y omitido: {rel_path}")
            duplicates_found += 1
            continue
        seen_hashes[inspection.sha256] = sample
        all_records.append(record)

    if label_conflicts:
        atomic_write_json(
            output_dir / "preparation_audit.json",
            _audit_payload(
                seed=seed,
                samples_discovered=len(raw_image_paths),
                samples_valid=valid_found,
                exclusions=excluded_samples,
                exact_duplicates=duplicates_found,
                conflicts=label_conflicts,
                invalid_images=invalid_samples,
                status="failed_label_conflicts",
            ),
        )
        raise _conflict_error(label_conflicts)

    if not all_records:
        raise ValueError("No quedaron imágenes válidas después de exclusiones y validación")

    df_manifest = ensure_sample_ids(pd.DataFrame(all_records))
    logger.info(
        f"Manifiesto construido: {len(df_manifest)} imágenes válidas "
        f"(duplicados exactos omitidos: {duplicates_found} | corruptas omitidas: {corrupt_found})"
    )

    missing_classes = sorted(set(allowed_classes) - set(df_manifest["label"].unique()))
    if missing_classes:
        raise SystemExit(
            f"Clases configuradas sin ninguna imagen válida indexada en '{clean_dir}': "
            f"{missing_classes}. El dataset parece incompleto o desactualizado en "
            "DATASET_ROOT (revisa `make download-dataset`); si es intencional, quítalas de "
            "dataset.classes / baseline.classes en config/dataset.yaml o usa --classes."
        )

    class_cap_dropped = 0
    if max_per_class is not None:
        before = len(df_manifest)
        df_manifest = _cap_manifest_per_class(df_manifest, max_per_class, seed)
        class_cap_dropped = before - len(df_manifest)
        logger.info(
            f"Límite de {max_per_class} imágenes por clase aplicado: "
            f"{before} -> {len(df_manifest)} imágenes"
        )

    strata = df_manifest.groupby(["label", "environment"]).size()
    too_small = strata[strata < _MIN_STRATUM_IMAGES]
    if not too_small.empty:
        detail = ", ".join(f"{label}/{env}={n}" for (label, env), n in too_small.items())
        raise SystemExit(
            f"Estratos con menos de {_MIN_STRATUM_IMAGES} imágenes, insuficientes para el "
            f"split estratificado 70/15/15: {detail}. Agrega imágenes a esos estratos o "
            "excluye esas clases (--classes)."
        )

    perceptual_duplicates = 0
    if deduplicate:
        logger.info("Eliminando casi-duplicados antes de particionar...")
        df_manifest, perceptual_duplicates = drop_near_duplicates(
            df_manifest, get_dataset_root(), max_distance=dedup_distance
        )
        logger.info("Casi-duplicados descartados: %d", perceptual_duplicates)

    df_manifest = df_manifest.copy()
    df_manifest["source_id"] = df_manifest["image_path"].map(provenance_from_path)
    explicit_groups = load_group_manifest(group_manifest, df_manifest)
    grouped_split = group_by_source or group_manifest is not None
    df_manifest = apply_effective_groups(
        df_manifest,
        explicit_groups,
        fallback_to_source=grouped_split,
    )
    fallback_samples = int(df_manifest["group_origin"].eq("individual").sum())
    if grouped_split and fallback_samples:
        logger.info(
            "%d muestras sin grupo explícito ni procedencia usarán fallback individual.",
            fallback_samples,
        )
    if grouped_split:
        split_strategy = "source_grouped" if group_by_source else "explicit_grouped"
        logger.info(
            "Repartiendo grupos efectivos completos (70%% Train, 15%% Val, 15%% Test): "
            "%d explícitos, %d inferidos, %d individuales.",
            df_manifest.loc[
                df_manifest["group_origin"].eq("explicit"), "effective_group_id"
            ].nunique(),
            df_manifest.loc[
                df_manifest["group_origin"].eq("inferred"), "effective_group_id"
            ].nunique(),
            fallback_samples,
        )
        splitter = SourceGroupedSplitter(
            seed=seed,
            group_column="effective_group_id",
            allow_incomplete=allow_incomplete,
        )
    else:
        split_strategy = "stratified_label_environment"
        if allow_incomplete:
            logger.warning("--allow-incomplete-splits no tiene efecto en el reparto estratificado.")
        logger.info(
            "Repartiendo por imagen con estratificación label+environment "
            "(70%% Train, 15%% Val, 15%% Test)."
        )
        splitter = HierarchicalStratifiedSplitter(seed=seed)

    master_columns = [*_MASTER_COLUMNS, *_GROUP_COLUMNS]
    master_manifest = write_canonical_csv(
        output_dir / "master_manifest.csv",
        df_manifest,
        master_columns,
    )

    train_df, val_df, test_df = splitter.split(
        df_manifest, **{f"{name}_size": ratio for name, ratio in _TARGET_RATIOS.items()}
    )

    split_columns = [*_SPLIT_COLUMNS, *_GROUP_COLUMNS]
    splits = {
        name: write_canonical_csv(output_dir / f"{name}.csv", frame, split_columns)
        for name, frame in zip(_SPLIT_NAMES, (train_df, val_df, test_df))
    }
    overlap_metrics = validate_split_integrity(master_manifest, splits)
    group_split_summary = build_group_split_summary(
        master_manifest,
        splits,
        _TARGET_RATIOS,
        overlap_metrics,
    )
    group_split_summary["split_strategy"] = split_strategy

    logger.info(f"Pipeline finalizado. Splits guardados en {output_dir}")
    logger.info(
        "Distribución -> Train: %d | Val: %d | Test: %d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    logger.info("Generando reporte de auditoría del split...")

    report_df = build_split_audit_report(master_manifest, splits)
    report_sort = ["split", "label", "environment", *_GROUP_COLUMNS]
    write_canonical_csv(
        output_dir / "split_audit_report.csv",
        report_df,
        [*report_sort, "count"],
        sort_by=report_sort,
    )

    group_manifest_sha256 = sha256_file(group_manifest) if group_manifest is not None else None
    split_parameters = _split_parameters(
        seed=seed,
        allowed_classes=list(allowed_classes),
        max_per_class=max_per_class,
        group_by_source=group_by_source,
        deduplicate=deduplicate,
        dedup_distance=dedup_distance,
        allow_incomplete=allow_incomplete,
        group_manifest_sha256=group_manifest_sha256,
        split_strategy=split_strategy,
    )
    lock = {
        "schema_version": 1,
        "seed": seed,
        "config_sha256": sha256_json(split_parameters),
        "split_parameters": split_parameters,
        "master_manifest_sha256": sha256_file(output_dir / "master_manifest.csv"),
        "train_sha256": sha256_file(output_dir / "train.csv"),
        "val_sha256": sha256_file(output_dir / "val.csv"),
        "test_sha256": sha256_file(output_dir / "test.csv"),
        "exclusions_sha256": sha256_file(exclusions) if exclusions is not None else None,
        "group_manifest_sha256": group_manifest_sha256,
    }
    atomic_write_json(output_dir / "manifest.lock.json", lock)
    atomic_write_json(
        output_dir / "preparation_audit.json",
        _audit_payload(
            seed=seed,
            samples_discovered=len(raw_image_paths),
            samples_valid=valid_found,
            exclusions=excluded_samples,
            exact_duplicates=duplicates_found,
            conflicts=label_conflicts,
            invalid_images=invalid_samples,
            class_cap_dropped=class_cap_dropped,
            perceptual_duplicates=perceptual_duplicates,
            master_manifest=master_manifest,
            group_split_summary=group_split_summary,
            perceptual_validation_enabled=deduplicate,
        ),
    )

    logger.info(f"Reporte de auditoría guardado en: {output_dir / 'split_audit_report.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera splits CSV estratificados.")
    parser.add_argument(
        "--config",
        default="config/dataset.yaml",
        help="Ruta al archivo de configuración (default: config/dataset.yaml)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Usa el perfil 'baseline' de config/dataset.yaml (subset de clases + límite de "
        "imágenes por clase) como defaults. --classes/--max-per-class lo sobrescriben.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Lista explícita de clases a incluir (sobrescribe dataset.classes / baseline.classes)",
    )
    cap_group = parser.add_mutually_exclusive_group()
    cap_group.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        dest="max_per_class",
        help="Límite de imágenes por clase, aplicado antes del split (sobrescribe "
        "baseline.max_images_per_class).",
    )
    cap_group.add_argument(
        "--no-cap",
        action="store_true",
        dest="no_cap",
        help="Ignora baseline.max_images_per_class: usa el 100%% de las imágenes disponibles "
        "por clase. Solo tiene efecto junto con --baseline (sin --baseline nunca hay cap).",
    )
    parser.add_argument(
        "--group-by-source",
        action="store_true",
        dest="group_by_source",
        help="Genera un benchmark estricto manteniendo cada fuente completa en un solo split. "
        "Se guarda con sufijo _source_grouped; el default estratifica por label+environment.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Elimina casi-duplicados por hash perceptual antes de particionar, conservando "
        "un representante. Agrupar por fuente sin esto deja la misma foto a ambos lados.",
    )
    parser.add_argument(
        "--dedup-distance",
        type=int,
        default=0,
        dest="dedup_distance",
        help="Distancia de Hamming máxima entre hashes para considerarlos duplicados "
        "(default: 0, hash idéntico). Valores altos elevan los falsos positivos.",
    )
    parser.add_argument(
        "--allow-incomplete-splits",
        action="store_true",
        dest="allow_incomplete",
        help="Continúa aunque alguna clase quede fuera de val o test al conservar grupos.",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=None,
        help="CSV opcional image_path,sha256,reason de exclusiones lógicas verificadas.",
    )
    parser.add_argument(
        "--group-manifest",
        type=Path,
        default=None,
        help="CSV opcional sample_id,group_id o image_path,group_id. El grupo explícito "
        "prevalece sobre la procedencia inferida.",
    )
    args = parser.parse_args()
    run_data_preparation_pipeline(
        config_path=args.config,
        baseline=args.baseline,
        classes=args.classes,
        max_per_class=args.max_per_class,
        no_cap=args.no_cap,
        group_by_source=args.group_by_source,
        deduplicate=args.deduplicate,
        dedup_distance=args.dedup_distance,
        allow_incomplete=args.allow_incomplete,
        exclusions=args.exclusions,
        group_manifest=args.group_manifest,
    )
