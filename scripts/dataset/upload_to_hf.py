"""Empaqueta y sube `clean/` a Hugging Face Datasets Hub.

El repo lleva dos representaciones: shards `clean-<NNNNN>.tar` con el árbol
`<clase>/<entorno>/<archivo>` que `download_dataset.py` extrae, y metadata ligera
(`metadata.csv`, `dataset_infos.json`) para que HF lo reconozca como dataset de
clasificación de imágenes. Se empaqueta en shards porque 33k blobs sueltos implican 33k
requests por descarga, prohibitivo sobre el Volume remoto de Modal. No se suben Parquet con
imágenes embebidas: duplicarían el tamaño del repo a cambio del Dataset Viewer.

Uso:
    python scripts/dataset/upload_to_hf.py --stage-dir /ruta/con/espacio --dry-run
    python scripts/dataset/upload_to_hf.py --stage-dir /ruta/con/espacio
"""

import argparse
import csv
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path

from src.config import get_dataset_root
from src.data.provenance import provenance_from_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ENVIRONMENTS = ("lab", "real")

# Techo por shard: chico para poder reintentar una subida fallida sin repetir gigabytes.
SHARD_MAX_BYTES = 800 * 1024 * 1024

SHARD_PREFIX = "clean"

SOURCE_LICENSES = {
    "corn-leaf-diseases-classification-roboflow": "CC-BY-4.0",
    "corn-leaf-roboflow": "CC-BY-4.0",
    "cropdg-unified-multidomain": "CC-BY-NC-SA-4.0",
    "maize-2-roboflow": "CC-BY-4.0",
    "maize-beans-tomatoes-africa": "Apache-2.0 / CC-BY-4.0 / CC0-1.0",
    "maize-deficiency-scanner-roboflow": "CC-BY-4.0",
    "maize-diseases": "CC-BY-NC-SA-4.0",
    "maize-in-field-dataset": "CC-BY-NC-SA-4.0",
    "maize-leaf-roboflow": "CC-BY-4.0",
    "maize-nutrient-deficiency": "CC-BY-4.0",
    "multicrop-disease-maiz": "CC-BY-4.0",
}


def collect_images(clean_dir: Path, classes: list[str]) -> list[tuple[Path, str, str]]:
    """Indexa `clean/`. El `sorted()` hace el reparto en shards reproducible entre máquinas.

    @param {Path} clean_dir Raíz de `clean/`.
    @param {list[str]} classes Clases permitidas (orden canónico de `config/dataset.yaml`).
    @returns {list[tuple[Path, str, str]]} Tuplas (ruta_absoluta, clase, entorno).
    """
    images: list[tuple[Path, str, str]] = []

    for class_name in sorted(classes):
        class_dir = clean_dir / class_name
        if not class_dir.is_dir():
            logger.warning(f"Clase declarada sin directorio, omitida: {class_name}")
            continue

        for environment in ENVIRONMENTS:
            env_dir = class_dir / environment
            if not env_dir.is_dir():
                continue
            for image_path in sorted(env_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                    images.append((image_path, class_name, environment))

    return images


def plan_shards(
    images: list[tuple[Path, str, str]], max_bytes: int
) -> list[list[tuple[Path, str, str]]]:
    """Reparte las imágenes en shards por tamaño acumulado.

    Corta por bytes y no por conteo: con resoluciones de 54 px a 5184 px, repartir por número
    de archivos daría shards de tamaño dispar.

    @returns {list[list]} Lista de shards, cada uno con sus tuplas de imagen.
    """
    shards: list[list[tuple[Path, str, str]]] = [[]]
    current_bytes = 0

    for item in images:
        size = item[0].stat().st_size
        # El `and shards[-1]` deja sola en su shard a una imagen que exceda el techo.
        if current_bytes + size > max_bytes and shards[-1]:
            shards.append([])
            current_bytes = 0
        shards[-1].append(item)
        current_bytes += size

    return shards


def write_metadata_csv(images: list[tuple[Path, str, str]], destination: Path) -> None:
    """Escribe `metadata.csv` sin bytes de imagen, con la fuente y la licencia de cada una.

    `file_name` es la ruta relativa dentro del árbol ya extraído.
    """
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "label", "environment", "source_id", "license"])
        for image_path, class_name, environment in images:
            file_name = f"{class_name}/{environment}/{image_path.name}"
            source_id = provenance_from_path(f"{SHARD_PREFIX}/{file_name}")
            writer.writerow(
                [file_name, class_name, environment, source_id, SOURCE_LICENSES.get(source_id)]
            )


def write_dataset_infos(
    images: list[tuple[Path, str, str]], classes: list[str], destination: Path
) -> None:
    """Escribe `dataset_infos.json` declarando el esquema y las clases."""
    counts_by_class: dict[str, int] = {}
    counts_by_environment: dict[str, int] = {}
    for _, class_name, environment in images:
        counts_by_class[class_name] = counts_by_class.get(class_name, 0) + 1
        counts_by_environment[environment] = counts_by_environment.get(environment, 0) + 1

    payload = {
        "default": {
            "description": "Imágenes de hojas de maíz para clasificación de enfermedades, "
            "plagas y deficiencias nutricionales.",
            "features": {
                "file_name": {"dtype": "string", "_type": "Value"},
                "label": {"names": sorted(classes), "_type": "ClassLabel"},
                "environment": {"names": list(ENVIRONMENTS), "_type": "ClassLabel"},
                "source_id": {"dtype": "string", "_type": "Value"},
                "license": {"dtype": "string", "_type": "Value"},
            },
            "splits": {
                "train": {"name": "train", "num_examples": len(images)},
            },
            "num_images_per_class": dict(sorted(counts_by_class.items())),
            "num_images_per_environment": dict(sorted(counts_by_environment.items())),
        }
    }
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_stage(
    images: list[tuple[Path, str, str]],
    classes: list[str],
    stage_dir: Path,
    max_bytes: int,
) -> list[Path]:
    """Materializa shards .tar + metadata en `stage_dir`.

    @returns {list[Path]} Rutas de los artefactos generados.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)

    shards = plan_shards(images, max_bytes)
    logger.info(f"Empaquetando {len(images)} imágenes en {len(shards)} shard(s)...")

    written: list[Path] = []
    for index, shard in enumerate(shards):
        shard_path = stage_dir / f"{SHARD_PREFIX}-{index:05d}.tar"
        # Sin compresión: son JPEG/PNG, gzip solo gastaría CPU en ambos extremos.
        with tarfile.open(shard_path, "w") as tar:
            for image_path, class_name, environment in shard:
                tar.add(image_path, arcname=f"{class_name}/{environment}/{image_path.name}")
        size_mb = shard_path.stat().st_size / 1e6
        logger.info(f"  {shard_path.name}: {len(shard)} imágenes, {size_mb:.1f} MB")
        written.append(shard_path)

    metadata_path = stage_dir / "metadata.csv"
    write_metadata_csv(images, metadata_path)
    written.append(metadata_path)

    infos_path = stage_dir / "dataset_infos.json"
    write_dataset_infos(images, classes, infos_path)
    written.append(infos_path)

    logger.info(f"Metadata escrita: {metadata_path.name}, {infos_path.name}")
    return written


def _load_classes() -> list[str]:
    import yaml

    from src.config import PROJECT_ROOT

    with open(PROJECT_ROOT / "config" / "dataset.yaml") as handle:
        return yaml.safe_load(handle)["dataset"]["classes"]


def pending_uploads(written: list[Path], remote_files: set[str]) -> list[Path]:
    """Selecciona los artefactos que hay que subir.

    Un shard ya presente se omite porque el empaquetado es determinista; la metadata se sube
    siempre porque conserva el nombre aunque cambie su contenido.

    @param {list[Path]} written Artefactos generados en el staging.
    @param {set[str]} remote_files Nombres presentes en el repo.
    @returns {list[Path]} Artefactos pendientes, en su orden original.
    """
    return [p for p in written if p.suffix != ".tar" or p.name not in remote_files]


def _delete_stale_shards(api, repo_id: str, keep: set[str], token: str | None) -> None:
    """Borra los shards de una subida anterior que ya no se regeneraron.

    Si el número de shards baja, los sobrantes quedarían huérfanos y `download_dataset.py`
    extraería imágenes obsoletas junto a las nuevas.
    """
    from huggingface_hub import CommitOperationDelete

    remote_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    stale = [f for f in remote_files if f.endswith(".tar") and f not in keep]
    if not stale:
        return

    logger.info(f"Eliminando {len(stale)} shard(s) obsoleto(s) del repo: {stale[:5]}...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
        commit_message="Elimina shards obsoletos de la subida anterior",
        token=token,
    )


def _upload_files_sequentially(api, repo_id: str, paths: list[Path], retries: int = 3) -> None:
    """Sube archivo por archivo, reintentando cada uno.

    No se usa `upload_folder`: paraleliza varios shards en memoria y muere por OOM con poca
    RAM (con 3 GB el proceso se cae sin traza). Así el pico es el de un solo shard.
    """
    import time

    for index, path in enumerate(paths, start=1):
        for attempt in range(1, retries + 1):
            try:
                started = time.time()
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path.name,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Añade {path.name}",
                )
                logger.info(
                    f"  [{index}/{len(paths)}] {path.name} subido ({time.time() - started:.0f}s)"
                )
                break
            except Exception as error:  # noqa: BLE001 - se reintenta y se relanza al agotar
                logger.warning(
                    f"  [{index}/{len(paths)}] {path.name} intento {attempt}/{retries} "
                    f"falló: {type(error).__name__}: {str(error)[:150]}"
                )
                if attempt == retries:
                    raise
                time.sleep(5)


def upload_clean_dataset(
    repo_id: str,
    stage_dir: Path,
    token: str | None = None,
    private: bool = False,
    keep_stage: bool = False,
    max_bytes: int = SHARD_MAX_BYTES,
    dry_run: bool = False,
) -> None:
    """Empaqueta `clean/` en shards + metadata y los sube al repo de dataset."""
    from huggingface_hub import HfApi

    clean_dir = get_dataset_root() / "clean"
    if not clean_dir.is_dir():
        raise SystemExit(f"No se encontró {clean_dir}. Verifica DATASET_ROOT en .env")

    classes = _load_classes()
    images = collect_images(clean_dir, classes)
    if not images:
        raise SystemExit(f"No se indexó ninguna imagen en {clean_dir}.")

    total_bytes = sum(p.stat().st_size for p, _, _ in images)
    logger.info(f"Indexadas {len(images)} imágenes ({total_bytes / 1e9:.1f} GB) en {clean_dir}")

    # El staging necesita espacio equivalente al dataset entero. En dry-run solo se avisa:
    # planificar no escribe nada y sirve para dimensionar el volumen.
    stage_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(stage_dir).free
    if free_bytes < total_bytes:
        message = (
            f"Espacio insuficiente en {stage_dir}: {free_bytes / 1e9:.1f} GB libres, "
            f"se necesitan ~{total_bytes / 1e9:.1f} GB. Usa --stage-dir en otro volumen."
        )
        if not dry_run:
            raise SystemExit(message)
        logger.warning(f"[dry-run] {message}")

    if dry_run:
        shards = plan_shards(images, max_bytes)
        logger.info(
            f"[dry-run] {len(images)} imágenes -> {len(shards)} shard(s) + metadata.csv "
            f"+ dataset_infos.json -> hf://datasets/{repo_id}"
        )
        for index, shard in enumerate(shards):
            shard_bytes = sum(p.stat().st_size for p, _, _ in shard)
            logger.info(
                f"[dry-run]   {SHARD_PREFIX}-{index:05d}.tar: "
                f"{len(shard)} imágenes, {shard_bytes / 1e6:.1f} MB"
            )
        return

    written = build_stage(images, classes, stage_dir, max_bytes)

    api = HfApi(token=token)
    logger.info(f"Creando/verificando repo de dataset: {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    pending = pending_uploads(written, remote_files)
    if len(pending) < len(written):
        logger.info(f"{len(written) - len(pending)} archivo(s) ya estaban en el repo; se omiten.")

    logger.info(f"Subiendo {len(pending)} archivo(s) -> hf://datasets/{repo_id}")
    _upload_files_sequentially(api, repo_id, pending)

    keep = {p.name for p in written if p.suffix == ".tar"}
    _delete_stale_shards(api, repo_id, keep, token)

    logger.info("Subida completada.")

    if not keep_stage:
        shutil.rmtree(stage_dir, ignore_errors=True)
        logger.info(f"Staging eliminado: {stage_dir}")
    else:
        logger.info(f"Staging conservado en {stage_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sube el dataset limpio (clean/) a Hugging Face Datasets Hub como shards .tar."
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_DATASET_REPO"),
        help="Repo destino, formato 'usuario/nombre-dataset'. Default: env HF_DATASET_REPO.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Token de HF. Si se omite, usa HF_TOKEN o la sesión de `huggingface-cli login`.",
    )
    parser.add_argument("--private", action="store_true", help="Crea el repo como privado.")
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Directorio donde empaquetar los shards. Necesita espacio ~= al dataset (~18 GB).",
    )
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=SHARD_MAX_BYTES // (1024 * 1024),
        help="Tamaño máximo por shard en MB (default: 800).",
    )
    parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="No borra el staging al terminar (útil para inspeccionar los tars).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo reporta el plan de shards, sin escribir ni subir nada.",
    )
    args = parser.parse_args()

    if not args.repo_id:
        raise SystemExit(
            "Falta --repo-id (o define HF_DATASET_REPO en .env), formato 'usuario/nombre-dataset'."
        )
    if not args.stage_dir:
        raise SystemExit(
            "Falta --stage-dir: el empaquetado necesita ~18 GB libres. "
            "Pasa un directorio en un volumen con espacio."
        )

    upload_clean_dataset(
        repo_id=args.repo_id,
        stage_dir=args.stage_dir,
        token=args.token,
        private=args.private,
        keep_stage=args.keep_stage,
        max_bytes=args.shard_size_mb * 1024 * 1024,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
