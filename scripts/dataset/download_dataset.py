import argparse
import json
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

import yaml

from src.config import PROJECT_ROOT, get_dataset_root
from src.data.loader import load_and_normalize_image
from src.provenance import atomic_json, sha256_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _clean_dir_has_content(clean_dir: Path) -> bool:
    """Verify complete class coverage and the exact hashed image inventory, not directories."""
    if not clean_dir.is_dir() or not (clean_dir / "download_manifest.json").is_file():
        return False
    if any(clean_dir.rglob("*.tar")):
        return False
    with open(PROJECT_ROOT / "config" / "dataset.yaml") as f:
        classes = yaml.safe_load(f)["dataset"]["classes"]
    try:
        manifest = json.loads((clean_dir / "download_manifest.json").read_text())
        rows = manifest["files"]
        actual = {
            p.relative_to(clean_dir).as_posix()
            for p in clean_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        }
        return (
            manifest.get("complete") is True
            and actual == {r["path"] for r in rows}
            and all(
                (clean_dir / r["path"]).resolve().is_relative_to(clean_dir.resolve()) for r in rows
            )
            and set(classes) <= {r["label"] for r in rows}
            and all(
                (clean_dir / r["path"]).is_file()
                and sha256_file(clean_dir / r["path"]) == r["sha256"]
                for r in rows
            )
        )
    except (ValueError, KeyError, OSError):
        return False


def _extract_and_remove_tars(clean_dir: Path) -> None:
    """Descomprime cada .tar en su lugar y lo borra al terminar"""
    tar_paths = sorted(clean_dir.rglob("*.tar"))
    for tar_path in tar_paths:
        logger.info(f"Extrayendo {tar_path.name}...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(clean_dir, filter="data")
        tar_path.unlink()
        logger.info(f"{tar_path.name} extraído y eliminado.")


def _download_from_hf(
    repo_id: str, clean_dir: Path, token: str | None, revision: str | None = None
) -> str:
    from huggingface_hub import HfApi, snapshot_download

    resolved_revision = HfApi(token=token).dataset_info(repo_id, revision=revision).sha

    logger.info(f"Descargando desde Hugging Face Datasets Hub: {repo_id}")
    clean_dir.mkdir(parents=True, exist_ok=True)
    identity = {"source": "hf", "repo": repo_id, "revision": resolved_revision}
    marker = clean_dir / ".download_source.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Download staging belongs to another source revision")
    atomic_json(marker, identity)
    # La metadata solo existe para que HF reconozca el repo como dataset de imágenes; el
    # pipeline saca label/environment del árbol extraído, así que no se descarga.
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=str(clean_dir),
        token=token,
        max_workers=int(os.getenv("HF_DOWNLOAD_WORKERS", "4")),
        ignore_patterns=[
            ".gitattributes",
            "README.md",
            "metadata.csv",
            "dataset_infos.json",
        ],
    )
    _extract_and_remove_tars(clean_dir)
    # Elimina los metadatos de descarga que snapshot_download deja en clean/.cache/
    shutil.rmtree(clean_dir / ".cache", ignore_errors=True)
    logger.info(f"Dataset descargado en {clean_dir}")
    return resolved_revision


def _download_from_gdrive(gdrive_id: str, clean_dir: Path) -> None:
    import gdown

    logger.info(f"Descargando desde Google Drive (fallback): {gdrive_id}")
    clean_dir.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(id=gdrive_id, output=str(clean_dir), quiet=False, use_cookies=False)
    logger.info(f"Dataset descargado en {clean_dir}")


def download_clean_dataset(
    source: str = "auto",
    force: bool = False,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    gdrive_id: str | None = None,
    dry_run: bool = False,
    revision: str | None = None,
    resume_staging: Path | None = None,
) -> None:
    clean_dir = get_dataset_root() / "clean"
    hf_repo = hf_repo or os.getenv("HF_DATASET_REPO")
    hf_token = hf_token or os.getenv("HF_TOKEN")
    gdrive_id = gdrive_id or os.getenv("GDRIVE_DATASET_ID")
    if resume_staging is not None:
        resume_staging = Path(resume_staging).resolve()
        if (
            source != "hf"
            or not revision
            or resume_staging.parent != clean_dir.parent.resolve()
            or not resume_staging.name.startswith(".clean-download-")
            or not (resume_staging / ".download_source.json").is_file()
        ):
            raise ValueError("Resume requires a pinned HF staging directory with a source marker")

    if not force and _clean_dir_has_content(clean_dir) and not dry_run:
        logger.info(
            f"{clean_dir} ya tiene contenido; se omite la descarga (usa --force para reintentar)."
        )
        return

    if source == "hf" and not hf_repo:
        raise SystemExit("--source hf requiere HF_DATASET_REPO (env) o --hf-repo.")
    if source == "gdrive" and not gdrive_id:
        raise SystemExit("--source gdrive requiere GDRIVE_DATASET_ID (env) o --gdrive-id.")
    if source == "auto" and not hf_repo and not gdrive_id:
        raise SystemExit(
            "No hay fuente configurada: define HF_DATASET_REPO o GDRIVE_DATASET_ID en .env "
            "(o pasa --hf-repo/--gdrive-id)."
        )

    if dry_run:
        plan = f"source={source} hf_repo={hf_repo!r} gdrive_id={gdrive_id!r} -> {clean_dir}"
        logger.info(f"[dry-run] Resolución de fuente válida: {plan}")
        return

    sources = ["hf", "gdrive"] if source == "auto" else [source]
    failures = []
    for candidate in sources:
        if (candidate == "hf" and not hf_repo) or (candidate == "gdrive" and not gdrive_id):
            continue
        staging = resume_staging or Path(
            tempfile.mkdtemp(prefix=".clean-download-", dir=clean_dir.parent)
        )
        try:
            resolved = None
            if candidate == "hf":
                resolved = _download_from_hf(hf_repo, staging, token=hf_token, revision=revision)
            else:
                _download_from_gdrive(gdrive_id, staging)
            with open(PROJECT_ROOT / "config/dataset.yaml") as handle:
                classes = yaml.safe_load(handle)["dataset"]["classes"]
            files = []
            for label in classes:
                paths = sorted(
                    p
                    for p in (staging / label).rglob("*")
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                )
                if not paths:
                    raise ValueError(f"Descarga incompleta: clase vacía {label}")
                for path in paths:
                    load_and_normalize_image(str(path))
                    files.append(
                        {
                            "path": path.relative_to(staging).as_posix(),
                            "label": label,
                            "sha256": sha256_file(path),
                        }
                    )
                    if len(files) % 1000 == 0:
                        logger.info("Validadas %s imágenes", len(files))
            atomic_json(
                staging / "download_manifest.json",
                {
                    "schema_version": 1,
                    "complete": True,
                    "source": candidate,
                    "repo": hf_repo if candidate == "hf" else gdrive_id,
                    "revision": resolved,
                    "files": files,
                    "source_versioning": "pinned_commit" if resolved else "content_snapshot_only",
                },
            )
            backup = clean_dir.with_name(f"clean.backup-{uuid4().hex}")
            if clean_dir.exists():
                os.replace(clean_dir, backup)
            try:
                os.replace(staging, clean_dir)
            except BaseException:
                if backup.exists():
                    os.replace(backup, clean_dir)
                raise
            if backup.exists():
                logger.info("Dataset previo conservado en %s", backup)
            return
        except Exception as e:
            failures.append(f"{candidate}: {e}")
            logger.warning(
                "Fuente %s falló; staging conservado para auditoría: %s", candidate, staging
            )
    raise RuntimeError("Ninguna descarga fue validada: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga el dataset limpio (clean/) hacia $DATASET_ROOT/clean/."
    )
    parser.add_argument(
        "--source",
        choices=["hf", "gdrive", "auto"],
        default="auto",
        help="Fuente a usar. 'auto' intenta Hugging Face y cae a Google Drive (default).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Vuelve a descargar aunque clean/ ya tenga contenido."
    )
    parser.add_argument(
        "--hf-repo", dest="hf_repo", default=None, help="Override de HF_DATASET_REPO."
    )
    parser.add_argument("--hf-token", dest="hf_token", default=None, help="Override de HF_TOKEN.")
    parser.add_argument(
        "--gdrive-id", dest="gdrive_id", default=None, help="Override de GDRIVE_DATASET_ID."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo valida qué fuente se usaría, sin descargar nada.",
    )
    parser.add_argument(
        "--revision", default=None, help="Commit/tag HF; se registra el SHA resuelto."
    )
    parser.add_argument(
        "--resume-staging",
        type=Path,
        default=None,
        help="Reanuda un staging HF compatible; requiere --source hf y --revision",
    )
    args = parser.parse_args()

    download_clean_dataset(
        source=args.source,
        force=args.force,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        gdrive_id=args.gdrive_id,
        dry_run=args.dry_run,
        revision=args.revision,
        resume_staging=args.resume_staging,
    )


if __name__ == "__main__":
    main()
