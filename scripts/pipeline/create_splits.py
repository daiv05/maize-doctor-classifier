import argparse
import hashlib
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from src.config import get_dataset_root, get_output_root
from src.data.identity import ensure_sample_ids, sample_id_for_path
from src.data.splitter import HierarchicalStratifiedSplitter
from src.provenance import atomic_json, sha256_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# sklearn exige >=2 muestras por estrato en cada corte del doble split; con 70/15/15
# eso se garantiza a partir de ~7 imágenes por estrato label+environment.
_MIN_STRATUM_IMAGES = 7


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


def _verify_and_hash(abs_path: Path) -> tuple[bool, str]:
    """Lee el archivo una sola vez; valida integridad PIL y calcula el SHA-256.

    Devuelve `(True, digest)` si la imagen es válida, o `(False, mensaje_error)` si es
    corrupta/ilegible. Es una función pura del contenido del archivo (el resultado no
    depende del orden ni de otras imágenes), así que es segura para ejecutarse en paralelo.
    Lee los bytes una vez y los reutiliza para PIL (vía BytesIO) y para el hash, evitando
    la doble lectura de disco del enfoque anterior.
    """
    try:
        data = abs_path.read_bytes()
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return True, hashlib.sha256(data).hexdigest()
    except Exception as e:  # noqa: BLE001 - cualquier fallo = imagen inutilizable, se omite
        return False, str(e)


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


def run_data_preparation_pipeline(
    config_path: str,
    baseline: bool = False,
    classes: list[str] | None = None,
    max_per_class: int | None = None,
    no_cap: bool = False,
    output_dir: Path | None = None,
    group_manifest: Path | None = None,
    exclusions: Path | None = None,
) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

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
    output_dir = (
        Path(output_dir)
        if output_dir
        else _split_output_dir(base_output_dir, suffix="baseline" if baseline else None)
    )
    seed = config["dataset"]["seed"]

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Splits inmutables: use --output-dir con una versión nueva.")
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
    excluded_paths = {}
    if exclusions:
        excluded = pd.read_csv(exclusions, dtype=str)
        required = ["image_path", "sha256", "reason"]
        if not set(required).issubset(excluded.columns):
            raise ValueError("Exclusiones requieren image_path, sha256 y reason")
        if (
            excluded[required].isna().any().any()
            or excluded[required].apply(lambda column: column.str.strip().eq("")).any().any()
            or excluded.image_path.duplicated().any()
        ):
            raise ValueError("Exclusiones incompletas o duplicadas")
        excluded_paths = excluded.set_index("image_path").to_dict("index")
        if not set(excluded_paths).issubset({record[3] for record in raw_image_paths}):
            raise ValueError("Exclusiones contienen rutas ajenas a esta versión de datos")

    all_records: list[dict] = []
    seen_hashes: dict[str, dict] = {}
    audit_records = []
    conflicts = []
    duplicates_found = 0
    corrupt_found = 0

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
    for (class_name, environment, abs_path, rel_path), (ok, value) in zip(raw_image_paths, results):
        if rel_path in excluded_paths:
            exclusion = excluded_paths[rel_path]
            if not ok or value != exclusion["sha256"]:
                raise ValueError(f"Contenido de exclusión cambió: {rel_path}")
            audit_records.append(
                {
                    "image_path": rel_path,
                    "status": "explicit_exclusion",
                    "sha256": value,
                    "reason": exclusion["reason"],
                }
            )
            continue
        if not ok:
            tqdm.write(f"Imagen corrupta o ilegible, omitida: {rel_path} - {value}")
            corrupt_found += 1
            audit_records.append({"image_path": rel_path, "status": "corrupt", "reason": value})
            continue
        if value in seen_hashes:
            previous = seen_hashes[value]
            if previous["label"] != class_name:
                conflicts.append(
                    {
                        "image_path": rel_path,
                        "label": class_name,
                        "sha256": value,
                        "other": previous,
                    }
                )
            audit_records.append(
                {
                    "image_path": rel_path,
                    "status": "duplicate",
                    "sha256": value,
                    "kept": previous["image_path"],
                }
            )
            logger.warning(f"Duplicado exacto detectado y omitido: {rel_path}")
            duplicates_found += 1
            continue
        record = {
            "image_path": rel_path,
            "label": class_name,
            "environment": environment,
            "sha256": value,
        }
        seen_hashes[value] = record
        all_records.append(record)

    atomic_json(
        output_dir / "preparation_audit.json",
        {
            "excluded": audit_records,
            "label_conflicts": conflicts,
            "exclusions_manifest_sha256": sha256_file(exclusions) if exclusions else None,
        },
    )
    if conflicts:
        raise ValueError(
            "Contenido idéntico con etiquetas conflictivas; ver preparation_audit.json"
        )
    if not all_records:
        raise ValueError("No quedaron imágenes válidas después de verificar integridad.")
    df_manifest = ensure_sample_ids(pd.DataFrame(all_records))
    if group_manifest:
        groups = pd.read_csv(group_manifest, dtype=str)
        if "sample_id" not in groups and "image_path" in groups:
            groups["sample_id"] = groups.image_path.map(sample_id_for_path)
        required = ["sample_id", "group_id"]
        if not set(required).issubset(groups.columns):
            raise ValueError("Grupos requieren sample_id o image_path, y group_id")
        if (
            groups[required].isna().any().any()
            or groups[required].apply(lambda column: column.str.strip().eq("")).any().any()
            or groups.sample_id.duplicated().any()
        ):
            raise ValueError("Manifiesto de grupos incompleto o con IDs duplicados")
        df_manifest = df_manifest.merge(
            groups[["sample_id", "group_id"]], on="sample_id", how="left", validate="one_to_one"
        )
        if df_manifest["group_id"].isna().any():
            raise ValueError("El manifiesto de grupos no cubre todas las muestras.")
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

    if max_per_class is not None:
        before = len(df_manifest)
        df_manifest = _cap_manifest_per_class(df_manifest, max_per_class, seed)
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

    logger.info("Ejecutando división jerárquica estratificada (70% Train, 15% Val, 15% Test)...")
    splitter = HierarchicalStratifiedSplitter(seed=seed)
    train_df, val_df, test_df = splitter.split(
        df_manifest, train_size=0.70, val_size=0.15, test_size=0.15
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    df_manifest.to_csv(output_dir / "master_manifest.csv", index=False)
    atomic_json(
        output_dir / "manifest.lock.json",
        {
            "schema_version": 1,
            "seed": seed,
            "config_sha256": sha256_file(Path(config_path)),
            "master_sha256": sha256_file(output_dir / "master_manifest.csv"),
            "split_sha256": {
                s: sha256_file(output_dir / f"{s}.csv") for s in ("train", "val", "test")
            },
            "grouping": "group_id" if group_manifest else "exact_content_dedup_only",
            "group_manifest_sha256": sha256_file(group_manifest) if group_manifest else None,
            "exclusions_manifest_sha256": sha256_file(exclusions) if exclusions else None,
            "group_leakage_limitation": (
                "Only supplied groups are protected; completeness of source/plant/session "
                "relationships requires independent review"
                if group_manifest
                else "No source/plant/session metadata supplied"
            ),
        },
    )

    logger.info(f"Pipeline finalizado. Splits guardados en {output_dir}")
    logger.info(
        f"Distribución -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )

    logger.info("Generando reporte de auditoría del split...")

    train_counts = train_df.groupby(["label", "environment"]).size().rename("train_count")
    val_counts = val_df.groupby(["label", "environment"]).size().rename("val_count")
    test_counts = test_df.groupby(["label", "environment"]).size().rename("test_count")

    report_df = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int)
    report_df["total_count"] = report_df.sum(axis=1)
    report_df = report_df.reset_index()
    # Junto a los CSV del split, para que los perfiles completo y baseline no se pisen.
    report_df.to_csv(output_dir / "split_audit_report.csv", index=False)

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
    parser.add_argument("--output-dir", type=Path, help="Nueva versión inmutable de splits.")
    parser.add_argument(
        "--exclusions",
        type=Path,
        help="CSV image_path,sha256,reason de exclusiones explícitas y trazables",
    )
    parser.add_argument(
        "--group-manifest",
        type=Path,
        help="CSV sample_id,group_id de planta/fuente/sesión; no se infieren grupos.",
    )
    args = parser.parse_args()
    run_data_preparation_pipeline(
        config_path=args.config,
        baseline=args.baseline,
        classes=args.classes,
        max_per_class=args.max_per_class,
        no_cap=args.no_cap,
        output_dir=args.output_dir,
        group_manifest=args.group_manifest,
        exclusions=args.exclusions,
    )
